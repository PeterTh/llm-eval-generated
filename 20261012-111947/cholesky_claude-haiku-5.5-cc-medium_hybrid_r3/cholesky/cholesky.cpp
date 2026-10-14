#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition (left-looking, blocked)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// - MPI:    row blocks of kBlock rows are dealt block-cyclically over the ranks, so every rank
//           holds an equal share of the shrinking trailing matrix.
// - CUDA:   each rank keeps its rows resident on one GPU; matrix generation, the column-block updates,
//           the in-block triangular solves and validation run on the device.
// - OpenMP: host-side work: diagonal shift, triangle zeroing, error reduction.
//
// Every element is still accumulated sequentially over k (sum over k = 0..j-1, then subtracted from A),
// so the summation order matches the sequential algorithm and results agree with it to rounding.

constexpr size_t kBlock = 128;  // rows/columns per block
constexpr size_t kTileM = 64;   // GEMM rows per thread block
constexpr size_t kTileN = 32;   // GEMM columns per thread block
constexpr size_t kTileK = 32;   // GEMM k-step staged in shared memory
constexpr int kThreads = 128;   // GEMM: 16 x 8 threads, each owning a 4 x 4 register tile

#define CUDA_CHECK(call)                                                                         \
    do {                                                                                         \
        cudaError_t err_ = (call);                                                               \
        if (err_ != cudaSuccess) {                                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                        \
        }                                                                                        \
    } while (0)

// C[r*ldc + c] = sum_{k<K} X[r*ldx + k] * Y[c*ldy + k] for r < m, c < W. Every element is accumulated
// in ascending k with separately rounded multiply and add, so the summation order matches the sequential code.
__global__ void __launch_bounds__(kThreads)
gemmKernel(const double* X, size_t ldx, size_t m, const double* Y, size_t ldy, size_t W, size_t K, double* C, size_t ldc) {
    __shared__ double As[kTileK][kTileM + 1];
    __shared__ double Bs[kTileK][kTileN + 1];

    const size_t row0 = blockIdx.x * kTileM;
    const size_t col0 = blockIdx.y * kTileN;
    const size_t wt = (W - col0 < kTileN) ? W - col0 : kTileN;
    const size_t tx = threadIdx.x % 8;
    const size_t ty = threadIdx.x / 8;

    double acc[4][4] = {};
    for (size_t k0 = 0; k0 < K; k0 += kTileK) {
        const size_t kmax = (K - k0 < kTileK) ? K - k0 : kTileK;
        for (size_t q = threadIdx.x; q < kTileM * kTileK; q += kThreads) {
            const size_t r = q / kTileK, kk = q % kTileK;
            As[kk][r] = (row0 + r < m && kk < kmax) ? X[(row0 + r) * ldx + k0 + kk] : 0.0;
        }
        for (size_t q = threadIdx.x; q < kTileN * kTileK; q += kThreads) {
            const size_t c = q / kTileK, kk = q % kTileK;
            Bs[kk][c] = (c < wt && kk < kmax) ? Y[(col0 + c) * ldy + k0 + kk] : 0.0;
        }
        __syncthreads();

        for (size_t kk = 0; kk < kmax; ++kk) {
            double a[4], b[4];
            for (int i = 0; i < 4; ++i) a[i] = As[kk][ty * 4 + i];
            for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx * 4 + j];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) acc[i][j] = __dadd_rn(acc[i][j], __dmul_rn(a[i], b[j]));
        }
        __syncthreads();
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const size_t r = row0 + ty * 4 + i, c = tx * 4 + j;
            if (r < m && c < wt) C[r * ldc + col0 + c] = acc[i][j];
        }
    }
}

// Finishes the columns of one block for each row, in ascending column order:
// L[r][jj] = (A[r][jj] - (acc[r][jj] + sum_{kk<jj} L[r][kk] * LJJ[jj][kk])) / LJJ[jj][jj]
// L points at the first column of the block; LJJ is the factored diagonal block.
__global__ void blockSolveKernel(double* L, size_t ldl, size_t m, const double* acc, const double* LJJ, size_t ldj, size_t w) {
    const size_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;

    // Work on a thread-local copy of the row instead of rereading global memory for every term
    double* row = L + r * ldl;
    double x[kBlock];
    for (size_t jj = 0; jj < w; ++jj) x[jj] = row[jj];
    for (size_t jj = 0; jj < w; ++jj) {
        double sum = acc[r * w + jj];
        for (size_t kk = 0; kk < jj; ++kk) {
            sum = __dadd_rn(sum, __dmul_rn(x[kk], LJJ[jj * ldj + kk]));
        }
        x[jj] = __ddiv_rn(__dsub_rn(x[jj], sum), LJJ[jj * ldj + jj]);
    }
    for (size_t jj = 0; jj < w; ++jj) row[jj] = x[jj];
}

void launchGemm(const double* X, size_t ldx, size_t m, const double* Y, size_t ldy, size_t W, size_t K, double* C, size_t ldc) {
    if (m == 0 || W == 0) return;
    const dim3 grid(static_cast<unsigned int>((m + kTileM - 1) / kTileM), static_cast<unsigned int>((W + kTileN - 1) / kTileN));
    gemmKernel<<<grid, kThreads>>>(X, ldx, m, Y, ldy, W, K, C, ldc);
    CUDA_CHECK(cudaGetLastError());
}

// Factors the diagonal block in place. acc holds the sums over k < J0 for the lower triangle, ad the
// original block values, and L points at the block's first column in the panel (row stride ldl).
bool factorDiagonal(const double* acc, const double* ad, double* L, size_t ldl, size_t J0, size_t w, size_t& failedAt) {
    for (size_t jj = 0; jj < w; ++jj) {
        for (size_t ll = 0; ll <= jj; ++ll) {
            double sum = acc[jj * w + ll];
            for (size_t kk = 0; kk < ll; ++kk) {
                sum += L[jj * ldl + kk] * L[ll * ldl + kk];
            }

            if (ll == jj) {
                // Diagonal element
                const double val = ad[jj * w + jj] - sum;
                if (val <= 0.0) {
                    // Matrix is not positive definite
                    failedAt = J0 + jj;
                    return false;
                }
                L[jj * ldl + jj] = std::sqrt(val);
            } else {
                // Off-diagonal element
                L[jj * ldl + ll] = (ad[jj * w + ll] - sum) / L[ll * ldl + ll];
            }
        }
    }
    return true;
}

// Block-cyclic distribution of row blocks over MPI ranks.
struct Layout {
    size_t n;
    size_t nb;  // number of row blocks
    size_t p;   // number of ranks

    Layout(size_t n_, size_t p_) : n(n_), nb((n_ + kBlock - 1) / kBlock), p(p_) {}

    size_t ownerOf(size_t t) const { return t % p; }
    size_t blockRows(size_t t) const { return std::min(kBlock, n - t * kBlock); }
    // Local row offset of the first row of block t on its owner
    size_t localBase(size_t t) const { return (t / p) * kBlock; }

    size_t rowsOf(size_t rank) const {
        size_t rows = 0;
        for (size_t t = rank; t < nb; t += p) rows += blockRows(t);
        return rows;
    }

    // Global row of local row l on the given rank
    size_t rowOf(size_t rank, size_t l) const { return ((l / kBlock) * p + rank) * kBlock + l % kBlock; }

    // First local row on the given rank that belongs to a block >= t
    size_t firstLocalFrom(size_t rank, size_t t) const {
        const size_t first = t + (rank + p - t % p) % p;
        return first >= nb ? rowsOf(rank) : localBase(first);
    }
};

// Generates B exactly as the original generator and returns this rank's rows of A = B * B^T + n * I.
// The product runs on the GPU with the same ascending-k summation as the sequential code.
std::vector<double> generateLocalRows(const Layout& lay, size_t rank) {
    const size_t n = lay.n;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    const size_t nloc = lay.rowsOf(rank);
    std::vector<double> A(nloc * n);
    double* dB = nullptr;
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, std::max<size_t>(n * n, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dA, std::max<size_t>(nloc * n, 1) * sizeof(double)));
    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    }

    // A = B * B^T, one row block at a time (local rows of a block are contiguous)
    for (size_t t = rank; t < lay.nb; t += lay.p) {
        launchGemm(dB + t * kBlock * n, n, lay.blockRows(t), dB, n, n, n, dA + lay.localBase(t) * n, n);
    }
    if (nloc > 0) {
        CUDA_CHECK(cudaMemcpy(A.data(), dA, nloc * n * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));

    // Add diagonal dominance to ensure positive definiteness
    #pragma omp parallel for
    for (long l = 0; l < static_cast<long>(nloc); ++l) {
        A[l * n + lay.rowOf(rank, static_cast<size_t>(l))] += n;
    }
    return A;
}

// Left-looking blocked Cholesky on the GPUs. On entry A holds this rank's rows of the matrix; on exit
// it holds this rank's rows of L with the strictly upper triangle zeroed.
bool choleskyDistributed(const Layout& lay, size_t rank, std::vector<double>& A, size_t& failedAt) {
    const size_t n = lay.n;
    const size_t nloc = lay.rowsOf(rank);

    double* dA = nullptr;      // this rank's rows
    double* dPanel = nullptr;  // rows of the current column block, columns < J1 (L) and the factored block
    double* dAcc = nullptr;    // sums over k < J0 for the rows being updated
    double* dDiag = nullptr;   // sums over k < J0 for the diagonal block (owner only)
    CUDA_CHECK(cudaMalloc(&dA, std::max<size_t>(nloc * n, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dPanel, kBlock * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dAcc, std::max<size_t>(nloc * kBlock, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dDiag, kBlock * kBlock * sizeof(double)));
    if (nloc > 0) {
        CUDA_CHECK(cudaMemcpy(dA, A.data(), nloc * n * sizeof(double), cudaMemcpyHostToDevice));
    }

    std::vector<double> hPanel(kBlock * n);
    std::vector<double> hDiagAcc(kBlock * kBlock);
    std::vector<double> hDiagA(kBlock * kBlock);

    bool ok = true;
    for (size_t t = 0; t < lay.nb; ++t) {
        const size_t J0 = t * kBlock;
        const size_t w = lay.blockRows(t);
        const size_t J1 = J0 + w;
        const size_t root = lay.ownerOf(t);

        // The owner factors the diagonal block on the host; -1 signals success.
        int status = -1;
        if (rank == root) {
            double* dBlk = dA + lay.localBase(t) * n;
            launchGemm(dBlk, n, w, dBlk, n, w, J0, dDiag, w);
            CUDA_CHECK(cudaMemcpy(hDiagAcc.data(), dDiag, w * w * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy2D(hDiagA.data(), w * sizeof(double), dBlk + J0, n * sizeof(double),
                                    w * sizeof(double), w, cudaMemcpyDeviceToHost));
            if (J0 > 0) {
                CUDA_CHECK(cudaMemcpy2D(hPanel.data(), J1 * sizeof(double), dBlk, n * sizeof(double),
                                        J0 * sizeof(double), w, cudaMemcpyDeviceToHost));
            }
            if (factorDiagonal(hDiagAcc.data(), hDiagA.data(), hPanel.data() + J0, J1, J0, w, failedAt)) {
                CUDA_CHECK(cudaMemcpy2D(dBlk + J0, n * sizeof(double), hPanel.data() + J0, J1 * sizeof(double),
                                        w * sizeof(double), w, cudaMemcpyHostToDevice));
            } else {
                status = static_cast<int>(failedAt);
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, static_cast<int>(root), MPI_COMM_WORLD);
        if (status >= 0) {
            failedAt = static_cast<size_t>(status);
            ok = false;
            break;
        }

        // Broadcast the L values of columns < J1 for block rows t, and the factored diagonal block
        MPI_Bcast(hPanel.data(), w * J1, MPI_DOUBLE, static_cast<int>(root), MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dPanel, hPanel.data(), w * J1 * sizeof(double), cudaMemcpyHostToDevice));

        // Update all rows below block t: sums over k < J0 on the GPU, then the in-block part
        const size_t ls = lay.firstLocalFrom(rank, t + 1);
        const size_t m = nloc - ls;
        if (m > 0) {
            launchGemm(dA + ls * n, n, m, dPanel, J1, w, J0, dAcc, w);
            const unsigned int blocks = static_cast<unsigned int>((m + kThreads - 1) / kThreads);
            blockSolveKernel<<<blocks, kThreads>>>(dA + ls * n + J0, n, m, dAcc, dPanel + J0, J1, w);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (ok && nloc > 0) {
        CUDA_CHECK(cudaMemcpy(A.data(), dA, nloc * n * sizeof(double), cudaMemcpyDeviceToHost));
        #pragma omp parallel for
        for (long l = 0; l < static_cast<long>(nloc); ++l) {
            const size_t i = lay.rowOf(rank, static_cast<size_t>(l));
            double* row = A.data() + l * n;
            std::fill(row + i + 1, row + n, 0.0);
        }
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dPanel));
    CUDA_CHECK(cudaFree(dAcc));
    CUDA_CHECK(cudaFree(dDiag));
    return ok;
}

// Assembles the full row-major L (zero upper triangle) on every rank from the local row blocks.
std::vector<double> gatherFull(const Layout& lay, size_t rank, const std::vector<double>& local) {
    const size_t n = lay.n;
    std::vector<int> counts(lay.p), displs(lay.p);
    int offset = 0;
    for (size_t r = 0; r < lay.p; ++r) {
        counts[r] = static_cast<int>(lay.rowsOf(r) * n);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<double> packed(n * n);
    MPI_Allgatherv(local.data(), counts[rank], MPI_DOUBLE, packed.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    std::vector<double> full(n * n);
    for (size_t r = 0; r < lay.p; ++r) {
        for (size_t l = 0; l < lay.rowsOf(r); ++l) {
            const size_t i = lay.rowOf(r, l);
            std::copy(packed.begin() + displs[r] + l * n, packed.begin() + displs[r] + (l + 1) * n, full.begin() + i * n);
        }
    }
    return full;
}

// Checks L * L^T against the original matrix on this rank's rows. L is the full factor.
void validateCholesky(const Layout& lay, size_t rank, const std::vector<double>& L, const std::vector<double>& A_orig,
                      double& maxError, double& relError) {
    const size_t n = lay.n;
    const size_t nloc = lay.rowsOf(rank);

    // Reconstruct this rank's rows of L * L^T on the GPU
    double* dL = nullptr;
    double* dRec = nullptr;
    CUDA_CHECK(cudaMalloc(&dL, std::max<size_t>(n * n, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dRec, std::max<size_t>(nloc * n, 1) * sizeof(double)));
    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(dL, L.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    }
    for (size_t t = rank; t < lay.nb; t += lay.p) {
        launchGemm(dL + t * kBlock * n, n, lay.blockRows(t), dL, n, n, n, dRec + lay.localBase(t) * n, n);
    }
    std::vector<double> rec(nloc * n);
    if (nloc > 0) {
        CUDA_CHECK(cudaMemcpy(rec.data(), dRec, nloc * n * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(dL));
    CUDA_CHECK(cudaFree(dRec));

    maxError = 0.0;
    relError = 0.0;
    #pragma omp parallel for reduction(max : maxError, relError)
    for (long l = 0; l < static_cast<long>(nloc); ++l) {
        for (size_t j = 0; j < n; ++j) {
            const double error = std::fabs(rec[l * n + j] - A_orig[l * n + j]);
            maxError = std::max(maxError, error);

            const double rel = error / (std::fabs(A_orig[l * n + j]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    // One GPU per rank; the OpenMP threads of ranks sharing a node split its cores
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    // Generate positive definite matrix (local rows)
    const Layout lay(n, static_cast<size_t>(size));
    std::vector<double> A = generateLocalRows(lay, static_cast<size_t>(rank));
    std::vector<double> A_orig;
    if (validate) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    size_t failedAt = 0;
    const bool success = choleskyDistributed(lay, static_cast<size_t>(rank), A, failedAt);

    auto end = std::chrono::high_resolution_clock::now();

    if (!success) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", failedAt);
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }

    // The slowest rank defines the computation time
    double seconds = std::chrono::duration<double>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(seconds));

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Full L is needed for result output and validation
    std::vector<double> L;
    if (printResults || validate) {
        L = gatherFull(lay, static_cast<size_t>(rank), A);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(L, "CholeskyL");
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");

        double errors[2];
        validateCholesky(lay, static_cast<size_t>(rank), L, A_orig, errors[0], errors[1]);
        MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        const bool valid = !(errors[1] > 1e-6);
        if (rank == 0) {
            printf("Max absolute error: %.10e\n", errors[0]);
            printf("Max relative error: %.10e\n", errors[1]);
            if (!valid) {
                printf("Validation failed: relative error too large\n");
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = runBenchmark(argc, argv);
    MPI_Finalize();
    return rc;
}
