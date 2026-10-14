// C bindings only; the MPI C++ bindings are not used
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Hybrid Cholesky decomposition: MPI + OpenMP + CUDA.
//
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Algorithm: left-looking blocked Cholesky over block columns of width BS.
// Row blocks of height BS are distributed block-cyclically over all GPUs of
// all MPI ranks (one worker per GPU). Every element L[i][j] is computed as
//   (A[i][j] - sum_{k<j} L[i][k] * L[j][k]) / L[j][j]
// with the sum accumulated in ascending k order into a single accumulator, the
// same order as the sequential algorithm. Results therefore do not depend on the
// number of ranks or GPUs. They can differ from the sequential build in the last
// bits, because FMA contraction is chosen per code shape by each compiler.
//
// Per block column t:
//   1. Owner GPU factors the diagonal block (CUDA kernel, sequential over columns).
//   2. The panel L[t*BS:(t+1)*BS, 0:j1] is broadcast with MPI to all ranks.
//   3. Every GPU (one OpenMP thread per local GPU) updates its row blocks below
//      the diagonal block with a tiled GEMM (prefix sum over k < j0) and a
//      short in-block fix-up kernel.

namespace {

constexpr size_t BS = 64;  // row block height and panel width
constexpr int TM = 16;     // output rows per CTA of the prefix GEMM
constexpr int TN = 16;     // output columns per CTA (BS / TN column tiles)
constexpr int TK = 32;     // k-tile depth of the prefix GEMM

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Sum over k < j0 of L[i][k] * P[c][k] for a range of rows and the panel columns
// c < w. Each output is accumulated by one thread in ascending k order, so the
// result does not depend on the tiling. Grid: (row tiles, BS / TN column tiles).
// L rows are addressed as L[(lrStart + r) * n + k]; S receives BS entries per row.
__global__ void gemmPrefixKernel(const double* __restrict__ L, const double* __restrict__ P,
                                 double* __restrict__ S, size_t lrStart, size_t nRows,
                                 int w, size_t j0, size_t n) {
    __shared__ double sA[TM][TK + 1];
    __shared__ double sB[TN][TK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TN + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.x) * TM;
    const int colBase = blockIdx.y * TN;

    double acc = 0.0;
    for (size_t kt = 0; kt < j0; kt += TK) {
        for (int idx = tid; idx < TM * TK; idx += TM * TN) {
            const int r = idx / TK;
            const int kk = idx % TK;
            const size_t rr = rowBase + r;
            const size_t k = kt + kk;
            sA[r][kk] = (rr < nRows && k < j0) ? L[(lrStart + rr) * n + k] : 0.0;
        }
        for (int idx = tid; idx < TN * TK; idx += TM * TN) {
            const int c = idx / TK;
            const int kk = idx % TK;
            const size_t k = kt + kk;
            sB[c][kk] = (colBase + c < w && k < j0) ? P[(colBase + c) * n + k] : 0.0;
        }
        __syncthreads();

        const int kmax = (j0 - kt < static_cast<size_t>(TK)) ? static_cast<int>(j0 - kt) : TK;
        for (int kk = 0; kk < kmax; ++kk) {
            acc += sA[ty][kk] * sB[tx][kk];
        }
        __syncthreads();
    }

    const size_t rr = rowBase + ty;
    const int c = colBase + tx;
    if (rr < nRows && c < w) {
        S[(lrStart + rr) * BS + c] = acc;
    }
}

// Factor the BS x w diagonal block of block column t (rows lrT.., columns j0..j0+w).
// S holds the prefix sums over k < j0. Columns are processed sequentially, rows in parallel.
__global__ void diagBlockKernel(double* __restrict__ L, const double* __restrict__ S,
                                size_t lrT, size_t j0, int w, size_t n, int* fail) {
    __shared__ double sL[BS][BS];
    __shared__ double sD[BS];

    const int r = threadIdx.x;
    const size_t lr = lrT + r;

    for (int c = 0; c < w; ++c) {
        if (r == c) {
            double sum = S[lr * BS + c];
            for (int k = 0; k < c; ++k) {
                sum += sL[c][k] * sL[c][k];
            }
            const double val = L[lr * n + j0 + c] - sum;
            if (val <= 0.0) {
                atomicMin(fail, static_cast<int>(j0 + c));
            }
            const double d = sqrt(val);
            sD[c] = d;
            sL[c][c] = d;
            L[lr * n + j0 + c] = d;
        }
        __syncthreads();

        if (r > c && r < w) {
            double sum = S[lr * BS + c];
            for (int k = 0; k < c; ++k) {
                sum += sL[r][k] * sL[c][k];
            }
            const double v = (L[lr * n + j0 + c] - sum) / sD[c];
            sL[r][c] = v;
            L[lr * n + j0 + c] = v;
        }
        __syncthreads();
    }
}

// Finish L[i][j0..j0+w) for rows whose prefix sums are in S: the in-block part of
// each dot product (k in [j0, j)) is accumulated after the prefix, in ascending k.
// P holds the finished panel rows (stride n), with the diagonal L[j][j] on its diagonal.
__global__ void updateRowsKernel(double* __restrict__ L, const double* __restrict__ S,
                                 const double* __restrict__ P, size_t lrStart, size_t nRows,
                                 int w, size_t j0, size_t n) {
    __shared__ double sP[BS][BS];

    for (int idx = threadIdx.x; idx < w * w; idx += blockDim.x) {
        const int c = idx / w;
        const int cc = idx % w;
        sP[c][cc] = P[c * n + j0 + cc];
    }
    __syncthreads();

    const size_t r = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r < nRows) {
        double* row = L + (lrStart + r) * n + j0;
        const double* srow = S + (lrStart + r) * BS;
        for (int c = 0; c < w; ++c) {
            double sum = srow[c];
            for (int cc = 0; cc < c; ++cc) {
                sum += row[cc] * sP[c][cc];
            }
            row[c] = (row[c] - sum) / sP[c][c];
        }
    }
}

// Number of row blocks owned by a worker (GPU): blocks p with p % W == worker.
size_t blocksOwnedBy(size_t worker, size_t nb, size_t W) {
    return worker < nb ? (nb - 1 - worker) / W + 1 : 0;
}

// First local block index whose global block index is >= p0.
size_t firstLocalBlock(size_t p0, size_t worker, size_t W) {
    return p0 <= worker ? 0 : (p0 - worker + W - 1) / W;
}

size_t rowsInBlock(size_t p, size_t n) {
    return std::min(BS, n - p * BS);
}

struct DeviceSlot {
    int id = 0;             // CUDA device ordinal
    size_t worker = 0;      // global worker index
    size_t blocks = 0;      // number of row blocks owned
    double* L = nullptr;    // blocks*BS rows x n, row-major
    double* P = nullptr;    // BS x n panel of the current block column
    double* S = nullptr;    // blocks*BS rows x BS prefix sums
    int* fail = nullptr;    // smallest non-positive diagonal index (INT_MAX if none)
};

struct Layout {
    int rank = 0;
    int size = 1;
    size_t n = 0;
    size_t nb = 0;
    size_t W = 0;                    // total workers (GPUs) over all ranks
    size_t firstWorker = 0;          // first worker of this rank
    std::vector<int> workerRank;     // MPI rank that owns each worker
    std::vector<DeviceSlot> devs;    // GPUs used by this rank
    double* panel = nullptr;         // pinned host buffer for the broadcast panel
};

// Each rank drives the GPUs of its node that are not used by the other ranks on the node.
// If there are more ranks than GPUs, the ranks share GPUs round robin.
Layout setupLayout(size_t n) {
    Layout ly;
    MPI_Comm_rank(MPI_COMM_WORLD, &ly.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ly.size);

    int ndev = 0;
    CHECK_CUDA(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", ly.rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, ly.rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    int nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_free(&nodeComm);

    std::vector<int> myDevices;
    if (nodeSize <= ndev) {
        for (int d = 0; d < ndev; ++d) {
            if (d % nodeSize == nodeRank) myDevices.push_back(d);
        }
    } else {
        myDevices.push_back(nodeRank % ndev);
    }

    const int myCount = static_cast<int>(myDevices.size());
    std::vector<int> counts(ly.size);
    MPI_Allgather(&myCount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    ly.W = 0;
    for (int r = 0; r < ly.size; ++r) {
        if (r < ly.rank) ly.firstWorker += counts[r];
        for (int g = 0; g < counts[r]; ++g) {
            ly.workerRank.push_back(r);
        }
        ly.W += counts[r];
    }

    ly.n = n;
    ly.nb = (n + BS - 1) / BS;
    for (int g = 0; g < myCount; ++g) {
        DeviceSlot slot;
        slot.id = myDevices[g];
        slot.worker = ly.firstWorker + g;
        slot.blocks = blocksOwnedBy(slot.worker, ly.nb, ly.W);
        ly.devs.push_back(slot);
    }
    return ly;
}

// Allocate device storage on every GPU of this rank (zero-initialised).
void allocate(Layout& ly) {
    const size_t n = ly.n;
    const int numLocal = static_cast<int>(ly.devs.size());

    CHECK_CUDA(cudaMallocHost(&ly.panel, std::max<size_t>(1, BS * n) * sizeof(double)));

#pragma omp parallel for num_threads(numLocal) schedule(static, 1)
    for (int g = 0; g < numLocal; ++g) {
        DeviceSlot& d = ly.devs[g];
        CHECK_CUDA(cudaSetDevice(d.id));

        const size_t rows = std::max<size_t>(1, d.blocks * BS);
        CHECK_CUDA(cudaMalloc(&d.L, rows * n * sizeof(double)));
        CHECK_CUDA(cudaMemset(d.L, 0, rows * n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d.P, std::max<size_t>(1, BS * n) * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d.S, rows * BS * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d.fail, sizeof(int)));
        const int none = INT_MAX;
        CHECK_CUDA(cudaMemcpy(d.fail, &none, sizeof(int), cudaMemcpyHostToDevice));
    }
}

// Copy the row blocks of A to the GPUs that own them.
void upload(Layout& ly, const double* A) {
    const size_t n = ly.n;
    const int numLocal = static_cast<int>(ly.devs.size());

#pragma omp parallel for num_threads(numLocal) schedule(static, 1)
    for (int g = 0; g < numLocal; ++g) {
        DeviceSlot& d = ly.devs[g];
        CHECK_CUDA(cudaSetDevice(d.id));
        for (size_t lp = 0; lp < d.blocks; ++lp) {
            const size_t p = d.worker + lp * ly.W;
            CHECK_CUDA(cudaMemcpy2D(d.L + lp * BS * n, n * sizeof(double),
                                    A + p * BS * n, n * sizeof(double),
                                    n * sizeof(double), rowsInBlock(p, n),
                                    cudaMemcpyHostToDevice));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

// Left-looking blocked factorization. The result stays distributed on the GPUs.
// Returns false (with the failing diagonal index) if A is not positive definite.
bool factorize(Layout& ly, size_t& failIdx) {
    const size_t n = ly.n;
    const size_t W = ly.W;
    const int numLocal = static_cast<int>(ly.devs.size());

    for (size_t t = 0; t < ly.nb; ++t) {
        const size_t j0 = t * BS;
        const size_t j1 = std::min(n, j0 + BS);
        const int w = static_cast<int>(j1 - j0);
        const size_t ownerWorker = t % W;
        const int ownerRank = ly.workerRank[ownerWorker];

        int fail = INT_MAX;
        if (ownerRank == ly.rank) {
            DeviceSlot& d = ly.devs[ownerWorker - ly.firstWorker];
            CHECK_CUDA(cudaSetDevice(d.id));
            const size_t lrT = ((t - ownerWorker) / W) * BS;

            // Prefix sums for the diagonal block rows; the panel prefix is in L itself.
            gemmPrefixKernel<<<dim3(BS / TM, BS / TN), dim3(TN, TM)>>>(d.L, d.L + lrT * n, d.S, lrT,
                                                                        BS, w, j0, n);
            diagBlockKernel<<<1, BS>>>(d.L, d.S, lrT, j0, w, n, d.fail);
            CHECK_CUDA(cudaGetLastError());

            CHECK_CUDA(cudaMemcpy2D(ly.panel, j1 * sizeof(double), d.L + lrT * n, n * sizeof(double),
                                    j1 * sizeof(double), w, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(&fail, d.fail, sizeof(int), cudaMemcpyDeviceToHost));
        }

        MPI_Bcast(ly.panel, static_cast<int>(w * j1), MPI_DOUBLE, ownerRank, MPI_COMM_WORLD);
        MPI_Bcast(&fail, 1, MPI_INT, ownerRank, MPI_COMM_WORLD);
        if (fail != INT_MAX) {
            failIdx = static_cast<size_t>(fail);
            return false;
        }

#pragma omp parallel for num_threads(numLocal) schedule(static, 1)
        for (int g = 0; g < numLocal; ++g) {
            DeviceSlot& d = ly.devs[g];
            CHECK_CUDA(cudaSetDevice(d.id));
            CHECK_CUDA(cudaMemcpy2D(d.P, n * sizeof(double), ly.panel, j1 * sizeof(double),
                                    j1 * sizeof(double), w, cudaMemcpyHostToDevice));

            const size_t lpFrom = firstLocalBlock(t + 1, d.worker, W);
            if (lpFrom < d.blocks) {
                const size_t lrStart = lpFrom * BS;
                const size_t nRows = (d.blocks - lpFrom) * BS;
                gemmPrefixKernel<<<dim3(static_cast<unsigned>((nRows + TM - 1) / TM), BS / TN),
                                   dim3(TN, TM)>>>(d.L, d.P, d.S, lrStart, nRows, w, j0, n);
                updateRowsKernel<<<static_cast<unsigned>((nRows + 127) / 128), 128>>>(
                    d.L, d.S, d.P, lrStart, nRows, w, j0, n);
                CHECK_CUDA(cudaGetLastError());
            }
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }
    return true;
}

// Copy the distributed L into A on rank 0 (upper triangle zeroed), like the in-place original.
void gatherToRoot(Layout& ly, double* A) {
    const size_t n = ly.n;
    std::vector<double> stage(std::max<size_t>(1, BS * n));

    for (size_t p = 0; p < ly.nb; ++p) {
        const size_t worker = p % ly.W;
        const int ownerRank = ly.workerRank[worker];
        const size_t h = rowsInBlock(p, n);
        double* dst = A + p * BS * n;

        if (ownerRank == ly.rank) {
            DeviceSlot& d = ly.devs[worker - ly.firstWorker];
            CHECK_CUDA(cudaSetDevice(d.id));
            const size_t lp = (p - worker) / ly.W;
            double* host = (ly.rank == 0) ? dst : stage.data();
            CHECK_CUDA(cudaMemcpy2D(host, n * sizeof(double), d.L + lp * BS * n, n * sizeof(double),
                                    n * sizeof(double), h, cudaMemcpyDeviceToHost));
            if (ly.rank != 0) {
                MPI_Send(stage.data(), static_cast<int>(h * n), MPI_DOUBLE, 0, static_cast<int>(p),
                         MPI_COMM_WORLD);
            }
        } else if (ly.rank == 0) {
            MPI_Recv(dst, static_cast<int>(h * n), MPI_DOUBLE, ownerRank, static_cast<int>(p),
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (ly.rank == 0) {
            for (size_t r = 0; r < h; ++r) {
                const size_t i = p * BS + r;
                for (size_t j = i + 1; j < n; ++j) {
                    dst[r * n + j] = 0.0;
                }
            }
        }
    }
}

void releaseLayout(Layout& ly) {
    for (DeviceSlot& d : ly.devs) {
        CHECK_CUDA(cudaSetDevice(d.id));
        cudaFree(d.L);
        cudaFree(d.P);
        cudaFree(d.S);
        cudaFree(d.fail);
    }
    cudaFreeHost(ly.panel);
}

}  // namespace

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T (each element is independent, OpenMP over rows)
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);

        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const bool isRoot = (rank == 0);

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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (isRoot) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    Layout ly = setupLayout(n);

    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (isRoot) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);

    if (validate && isRoot) {
        A_orig = A; // Save original for validation
    }

    // Device setup (context creation, allocation) is done before the timer, like
    // allocating the matrix; the timed region covers upload, factorization and sync.
    allocate(ly);

    // Perform Cholesky decomposition
    if (isRoot) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    upload(ly, A.data());
    size_t failIdx = 0;
    bool success = factorize(ly, failIdx);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int rc = 0;
    if (!success) {
        if (isRoot) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", failIdx);
            printf("Cholesky decomposition failed\n");
        }
        rc = 1;
    } else {
        gatherToRoot(ly, A.data());

        if (isRoot) {
            printf("Computation time: %ld ms\n", duration.count());

            // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
            double ops = (double)n * n * n / 3.0;
            double gflops = ops / (duration.count() / 1000.0) / 1e9;
            printf("Performance: %.3f GFLOPS\n", gflops);

            // Print results for external validation
            if (printResults) {
                print_results(A, "CholeskyL");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateCholesky(A, A_orig, n);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    rc = 1;
                }
            }
        }
    }

    releaseLayout(ly);
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
