#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

// Distributed hybrid Cholesky decomposition (MPI + OpenMP + CUDA).
//
// Decomposition:
//   - MPI: rows are distributed cyclically over ranks (global row i lives on rank i % P,
//     local index i / P). For each block of columns, ranks exchange the finished rows of
//     the block and the partial sums needed to factor the diagonal block (MPI_Allgatherv).
//   - CUDA: each rank keeps its local rows resident on its GPU. The O(n^3) work (the
//     panel GEMM against previously finished columns and the triangular solve below the
//     diagonal block) runs in CUDA kernels.
//   - OpenMP: host-side matrix generation, validation, and the final host copy are
//     parallelized with OpenMP.
//
// Numerics: every element L[i][j] is computed as
//     (A[i][j] - sum_{k<j} L[i][k]*L[j][k]) / L[j][j]
// with the sum accumulated in increasing k, as in the sequential algorithm. Blocking only
// splits that accumulation into chunks. Device code may use fused multiply-add, so results
// can differ from the CPU reference in the last bit (relative differences around 1e-16).

namespace {

constexpr size_t kBlock = 128;   // columns per panel (block of the blocked algorithm)
constexpr size_t kJTile = 64;    // column tile of the host-side O(n^3) loops (generation, validation)

// Panel GEMM tiling: each 16x16 thread block computes a 32x32 output tile, 2x2 per thread,
// streaming k in chunks of kTileK through shared memory.
constexpr int kTile = 32;        // output tile edge (rows and columns)
constexpr int kTileK = 32;       // k-depth of a GEMM tile
constexpr int kThreadsGemm = 256;
constexpr int kLoads = kTile * kTileK / kThreadsGemm;  // elements loaded per thread per tile

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err_),   \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Number of local rows held by `rank` for an n x n matrix (cyclic row distribution).
size_t localRowCount(size_t n, int rank, int nranks) {
    const size_t r = static_cast<size_t>(rank);
    const size_t P = static_cast<size_t>(nranks);
    return n > r ? (n - r - 1) / P + 1 : 0;
}

// Local index of the first row owned by `rank` whose global index is >= bound.
size_t firstLocalRow(size_t bound, int rank, size_t P) {
    const size_t r = static_cast<size_t>(rank);
    return bound <= r ? 0 : (bound - r + P - 1) / P;
}

// S[row][jj] = sum_{k<K} A[row][k] * P[jj][k], accumulated in increasing k.
// Output row r corresponds to local row rowBegin + r; A is row-major with leading dim lda.
__global__ void panelGemmKernel(const double* __restrict__ A, int lda, int rowBegin, int rows,
                                const double* __restrict__ Pnl, int K, int N,
                                double* __restrict__ S, int ldS) {
    __shared__ double As[kTileK][kTile + 1];
    __shared__ double Bs[kTileK][kTile + 1];
    static_assert(kTile * kTileK % kThreadsGemm == 0, "tile must divide evenly");

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int r0 = blockIdx.x * kTile;
    const int c0 = blockIdx.y * kTile;

    double acc[2][2] = {};
    for (int k0 = 0; k0 < K; k0 += kTileK) {
        const int kl = min(kTileK, K - k0);
        const int kk = tid % kTileK;
        const int rb = tid / kTileK;
        #pragma unroll
        for (int s = 0; s < kLoads; ++s) {
            const int r = rb + (kThreadsGemm / kTileK) * s;
            As[kk][r] = (kk < kl && r0 + r < rows)
                            ? A[static_cast<size_t>(rowBegin + r0 + r) * lda + k0 + kk]
                            : 0.0;
            Bs[kk][r] = (kk < kl && c0 + r < N)
                            ? Pnl[static_cast<size_t>(c0 + r) * K + k0 + kk]
                            : 0.0;
        }
        __syncthreads();
        // Increasing k within and across tiles: each output is the same sequential sum
        // as the reference dot product.
        for (int q = 0; q < kl; ++q) {
            double a[2], b[2];
            for (int i = 0; i < 2; ++i) {
                a[i] = As[q][ty + 16 * i];
                b[i] = Bs[q][tx + 16 * i];
            }
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            const int r = r0 + ty + 16 * i;
            const int c = c0 + tx + 16 * j;
            if (r < rows && c < N)
                S[static_cast<size_t>(rowBegin + r) * ldS + c] = acc[i][j];
        }
    }
}

// For each local row li in [rowBegin, rowEnd) (global row >= c + w), solve the panel
// columns c..c+w-1 against the factored diagonal block Ld (w x w, row-major, lower).
// S holds the partial sums over k < c computed by panelGemmKernel.
__global__ void panelSolveKernel(double* __restrict__ A, int lda, int c, int w,
                                 const double* __restrict__ S, int ldS,
                                 const double* __restrict__ Ld, int rowBegin, int rowEnd) {
    const int li = rowBegin + blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= rowEnd) return;

    double* row = A + static_cast<size_t>(li) * lda + c;
    const double* s = S + static_cast<size_t>(li) * ldS;
    for (int jj = 0; jj < w; ++jj) {
        const double* ldj = Ld + static_cast<size_t>(jj) * w;
        double sum = s[jj];
        // Unrolling only overlaps loads; the additions keep their sequential order.
        #pragma unroll 8
        for (int kk = 0; kk < jj; ++kk)
            sum += row[kk] * ldj[kk];
        row[jj] = (row[jj] - sum) / ldj[jj];
    }
}

// Sequential factorization of the w x w diagonal block starting at global column c.
// S holds the partial sums over k < c (w x w), Ablk the original block of A, Ld the output.
// Returns -1 on success or the global index of the first non-positive pivot.
long factorDiagonalBlock(size_t c, size_t w, const double* S, const double* Ablk, double* Ld) {
    for (size_t jj = 0; jj < w; ++jj) {
        double sum = S[jj * w + jj];
        for (size_t kk = 0; kk < jj; ++kk)
            sum += Ld[jj * w + kk] * Ld[jj * w + kk];
        const double val = Ablk[jj * w + jj] - sum;
        if (val <= 0.0)
            return static_cast<long>(c + jj);
        Ld[jj * w + jj] = sqrt(val);

        for (size_t ii = jj + 1; ii < w; ++ii) {
            double s = S[ii * w + jj];
            for (size_t kk = 0; kk < jj; ++kk)
                s += Ld[ii * w + kk] * Ld[jj * w + kk];
            Ld[ii * w + jj] = (Ablk[ii * w + jj] - s) / Ld[jj * w + jj];
        }
    }
    return -1;
}

// Distributed blocked Cholesky. dA holds the local rows of the input matrix (row-major,
// n columns each) on this rank's GPU and is overwritten with L (lower part; the upper part
// is garbage). On success, L_local receives the local rows with the upper part zeroed.
// Returns false if the matrix is not positive definite (identically on all ranks).
bool choleskyDecomposition(double* dA, const size_t n, const size_t nloc, const int rank,
                           const int nranks, std::vector<double>& L_local) {
    const size_t P = static_cast<size_t>(nranks);
    const size_t dn = n;

    double* dS = nullptr;      // partial sums S[local row][jj] for the current panel
    double* dPanel = nullptr;  // finished rows of the current block, prefix k < c
    double* dLd = nullptr;     // factored diagonal block
    CUDA_CHECK(cudaMalloc(&dS, std::max<size_t>(1, nloc * kBlock) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dPanel, std::max<size_t>(1, kBlock * n) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dLd, kBlock * kBlock * sizeof(double)));

    // Pinned staging buffers for the MPI exchange and the PCIe transfers.
    double* hSend = nullptr;   // this rank's block rows (prefix + original A block)
    double* hRecv = nullptr;   // all block rows, ordered by owning rank
    double* hSendS = nullptr;  // this rank's partial sums for block rows
    double* hRecvS = nullptr;  // all block partial sums
    CUDA_CHECK(cudaMallocHost(&hSend, std::max<size_t>(1, kBlock * n) * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hRecv, std::max<size_t>(1, kBlock * n) * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hSendS, kBlock * kBlock * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hRecvS, kBlock * kBlock * sizeof(double)));

    std::vector<double> hAblk, hSblk, hLd, hWB;
    std::vector<int> cnt(nranks), disp(nranks);
    bool ok = true;

    for (size_t c = 0; c < n; c += kBlock) {
        const size_t w = std::min(kBlock, n - c);
        const size_t rowLen = c + w;
        const size_t lBegin = firstLocalRow(c, rank, P);   // owned rows of this block
        const size_t lEnd = firstLocalRow(c + w, rank, P);
        const size_t nOwn = lEnd - lBegin;

        // 1. Gather the block rows: prefix [0, c) (finished L) and original A[c..c+w).
        if (nOwn)
            CUDA_CHECK(cudaMemcpy2D(hSend, rowLen * sizeof(double), dA + lBegin * dn,
                                    dn * sizeof(double), rowLen * sizeof(double), nOwn,
                                    cudaMemcpyDeviceToHost));
        int total = 0;
        for (int q = 0; q < nranks; ++q) {
            const size_t rows = firstLocalRow(c + w, q, P) - firstLocalRow(c, q, P);
            cnt[q] = static_cast<int>(rows * rowLen);
            disp[q] = total;
            total += cnt[q];
        }
        MPI_Allgatherv(hSend, static_cast<int>(nOwn * rowLen), MPI_DOUBLE, hRecv, cnt.data(),
                       disp.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Unpack: the panel prefix goes straight to the GPU (rows of rank q are strided by P
        // in the panel), the original block A values go to a small host array.
        hAblk.resize(w * w);
        for (int q = 0; q < nranks; ++q) {
            const size_t qb = firstLocalRow(c, q, P);
            const size_t qe = firstLocalRow(c + w, q, P);
            if (qe == qb) continue;
            if (c)
                CUDA_CHECK(cudaMemcpy2D(dPanel + (qb * P + q - c) * c, P * c * sizeof(double),
                                        hRecv + disp[q], rowLen * sizeof(double),
                                        c * sizeof(double), qe - qb, cudaMemcpyHostToDevice));
            for (size_t li = qb; li < qe; ++li) {
                const size_t jj = (li * P + q) - c;
                const double* src = hRecv + disp[q] + (li - qb) * rowLen;
                std::copy(src + c, src + rowLen, hAblk.data() + jj * w);
            }
        }

        // 2. Partial sums over k < c for all local rows at or below the block (GPU GEMM).
        const size_t rowsGemm = nloc - lBegin;
        if (rowsGemm) {
            const dim3 grid(static_cast<unsigned>((rowsGemm + kTile - 1) / kTile),
                            static_cast<unsigned>((w + kTile - 1) / kTile));
            panelGemmKernel<<<grid, dim3(16, 16)>>>(dA, static_cast<int>(n),
                                                    static_cast<int>(lBegin),
                                                    static_cast<int>(rowsGemm), dPanel,
                                                    static_cast<int>(c), static_cast<int>(w),
                                                    dS, static_cast<int>(w));
            CUDA_CHECK(cudaGetLastError());
        }

        // 3. Gather the partial sums of the block rows so every rank can factor the block.
        if (nOwn)
            CUDA_CHECK(cudaMemcpy(hSendS, dS + lBegin * w, nOwn * w * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        total = 0;
        for (int q = 0; q < nranks; ++q) {
            const size_t rows = firstLocalRow(c + w, q, P) - firstLocalRow(c, q, P);
            cnt[q] = static_cast<int>(rows * w);
            disp[q] = total;
            total += cnt[q];
        }
        MPI_Allgatherv(hSendS, static_cast<int>(nOwn * w), MPI_DOUBLE, hRecvS, cnt.data(),
                       disp.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        hSblk.resize(w * w);
        for (int q = 0; q < nranks; ++q) {
            const size_t qb = firstLocalRow(c, q, P);
            const size_t qe = firstLocalRow(c + w, q, P);
            for (size_t li = qb; li < qe; ++li) {
                const size_t jj = (li * P + q) - c;
                const double* src = hRecvS + disp[q] + (li - qb) * w;
                std::copy(src, src + w, hSblk.data() + jj * w);
            }
        }

        // 4. Factor the diagonal block (computed redundantly on every rank, so all agree).
        hLd.assign(w * w, 0.0);
        const long failIdx = factorDiagonalBlock(c, w, hSblk.data(), hAblk.data(), hLd.data());
        if (failIdx >= 0) {
            if (rank == 0)
                printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                       static_cast<size_t>(failIdx));
            ok = false;
            break;
        }

        // 5. Store the finished block rows owned by this rank and solve the rows below.
        CUDA_CHECK(cudaMemcpy(dLd, hLd.data(), w * w * sizeof(double), cudaMemcpyHostToDevice));
        if (nOwn) {
            hWB.resize(nOwn * w);
            for (size_t m = 0; m < nOwn; ++m) {
                const size_t jj = ((lBegin + m) * P + rank) - c;
                std::copy(hLd.data() + jj * w, hLd.data() + (jj + 1) * w, hWB.data() + m * w);
            }
            CUDA_CHECK(cudaMemcpy2D(dA + lBegin * dn + c, dn * sizeof(double), hWB.data(),
                                    w * sizeof(double), w * sizeof(double), nOwn,
                                    cudaMemcpyHostToDevice));
        }
        if (lEnd < nloc) {
            const int threads = 128;
            const int blocks = static_cast<int>((nloc - lEnd + threads - 1) / threads);
            panelSolveKernel<<<blocks, threads>>>(dA, static_cast<int>(n), static_cast<int>(c),
                                                  static_cast<int>(w), dS, static_cast<int>(w),
                                                  dLd, static_cast<int>(lEnd),
                                                  static_cast<int>(nloc));
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (ok) {
        L_local.resize(nloc * n);
        if (nloc)
            CUDA_CHECK(cudaMemcpy(L_local.data(), dA, nloc * n * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        #pragma omp parallel for schedule(static)
        for (long long li = 0; li < static_cast<long long>(nloc); ++li) {
            const size_t i = static_cast<size_t>(li) * P + rank;
            for (size_t j = i + 1; j < n; ++j)
                L_local[static_cast<size_t>(li) * n + j] = 0.0;
        }
    }

    CUDA_CHECK(cudaFreeHost(hSend));
    CUDA_CHECK(cudaFreeHost(hRecv));
    CUDA_CHECK(cudaFreeHost(hSendS));
    CUDA_CHECK(cudaFreeHost(hRecvS));
    CUDA_CHECK(cudaFree(dS));
    CUDA_CHECK(cudaFree(dPanel));
    CUDA_CHECK(cudaFree(dLd));
    return ok;
}

// Generate the local rows of a symmetric positive definite matrix. Every rank builds the
// same random matrix B (deterministic seed) and computes only its own rows of A = B * B^T.
void generatePositiveDefiniteMatrix(std::vector<double>& A_local, const size_t n, const int rank,
                                    const int nranks) {
    const size_t P = static_cast<size_t>(nranks);
    const size_t nloc = localRowCount(n, rank, nranks);

    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Columns are processed in tiles so that a tile of rows of B stays in cache while all
    // local rows are swept over it. The k-sum of every element is unchanged.
    for (size_t j0 = 0; j0 < n; j0 += kJTile) {
        const size_t j1 = std::min(n, j0 + kJTile);
        #pragma omp parallel for schedule(dynamic)
        for (long long li = 0; li < static_cast<long long>(nloc); ++li) {
            const size_t i = static_cast<size_t>(li) * P + rank;
            for (size_t j = j0; j < j1; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += B[i * n + k] * B[j * n + k];
                }
                A_local[static_cast<size_t>(li) * n + j] = sum;
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (long long li = 0; li < static_cast<long long>(nloc); ++li) {
        const size_t i = static_cast<size_t>(li) * P + rank;
        A_local[static_cast<size_t>(li) * n + i] += n;
    }
}

// Gather all rows of a row-distributed matrix into a full row-major matrix on every rank.
std::vector<double> gatherFullMatrix(const std::vector<double>& L_local, const size_t n,
                                     const int rank, const int nranks) {
    const size_t P = static_cast<size_t>(nranks);
    const size_t nloc = localRowCount(n, rank, nranks);

    std::vector<int> cnt(nranks), disp(nranks);
    int total = 0;
    for (int q = 0; q < nranks; ++q) {
        cnt[q] = static_cast<int>(localRowCount(n, q, nranks) * n);
        disp[q] = total;
        total += cnt[q];
    }
    std::vector<double> recv(n * n);
    MPI_Allgatherv(L_local.data(), static_cast<int>(nloc * n), MPI_DOUBLE, recv.data(),
                   cnt.data(), disp.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    std::vector<double> full(n * n);
    for (int q = 0; q < nranks; ++q) {
        const size_t rows = localRowCount(n, q, nranks);
        for (size_t li = 0; li < rows; ++li) {
            const size_t i = li * P + q;
            std::copy(recv.data() + disp[q] + li * n, recv.data() + disp[q] + (li + 1) * n,
                      full.data() + i * n);
        }
    }
    return full;
}

// Validate by computing L * L^T and comparing with the original matrix. Each rank checks
// its own rows; the maxima are combined with MPI reductions. Returns the same result on
// every rank.
bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_local,
                      const size_t n, const int rank, const int nranks) {
    const size_t P = static_cast<size_t>(nranks);
    const size_t nloc = localRowCount(n, rank, nranks);

    std::vector<double> rowMax(nloc, 0.0), rowRel(nloc, 0.0);
    for (size_t j0 = 0; j0 < n; j0 += kJTile) {
        const size_t j1 = std::min(n, j0 + kJTile);
        #pragma omp parallel for schedule(dynamic)
        for (long long li = 0; li < static_cast<long long>(nloc); ++li) {
            const size_t i = static_cast<size_t>(li) * P + rank;
            double maxError = rowMax[li];
            double relError = rowRel[li];
            for (size_t j = j0; j < j1; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += L[i * n + k] * L[j * n + k];
                }
                const double a = A_local[static_cast<size_t>(li) * n + j];
                const double error = fabs(sum - a);
                maxError = std::max(maxError, error);
                const double rel = error / (fabs(a) + 1e-10);
                relError = std::max(relError, rel);
            }
            rowMax[li] = maxError;
            rowRel[li] = relError;
        }
    }

    double maxError = 0.0;
    double relError = 0.0;
    for (size_t li = 0; li < nloc; ++li) {
        maxError = std::max(maxError, rowMax[li]);
        relError = std::max(relError, rowRel[li]);
    }
    MPI_Allreduce(MPI_IN_PLACE, &maxError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &relError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    if (relError > 1e-6) {
        if (rank == 0) printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    int exitCode = -1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            exitCode = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            break;
        }
    }
    if (exitCode >= 0) {
        MPI_Finalize();
        return exitCode;
    }

    // One GPU per rank (ranks on a node are spread over the node's GPUs).
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    int numDevices = 0;
    if (cudaGetDeviceCount(&numDevices) != cudaSuccess || numDevices == 0) {
        fprintf(stderr, "rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    const size_t nloc = localRowCount(n, rank, nranks);

    // Generate the matrix (local rows) and upload it to the GPU
    std::vector<double> A_local(nloc * n);
    generatePositiveDefiniteMatrix(A_local, n, rank, nranks);

    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, std::max<size_t>(1, nloc * n) * sizeof(double)));
    if (nloc)
        CUDA_CHECK(cudaMemcpy(dA, A_local.data(), nloc * n * sizeof(double),
                              cudaMemcpyHostToDevice));

    std::vector<double> A_orig;  // local rows of the original matrix, for validation
    if (validate) {
        A_orig = std::move(A_local);
    } else {
        std::vector<double>().swap(A_local);
    }

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    std::vector<double> L_local;
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const bool success = choleskyDecomposition(dA, n, nloc, rank, nranks, L_local);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsed_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    auto duration = std::chrono::milliseconds(elapsed_ms);

    CUDA_CHECK(cudaFree(dA));

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> fullL;
    if (printResults || validate) {
        fullL = gatherFullMatrix(L_local, n, rank, nranks);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(fullL, "CholeskyL");
    }

    // Validation
    int status = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateCholesky(fullL, A_orig, n, rank, nranks);
        if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        status = valid ? 0 : 1;
    }

    MPI_Finalize();
    return status;
}
