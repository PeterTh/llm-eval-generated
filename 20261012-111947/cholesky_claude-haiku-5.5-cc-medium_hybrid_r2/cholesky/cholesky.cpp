#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition (left-looking, blocked).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The matrix is split into row blocks of kBlock rows. Row block I is owned by one
// GPU ("worker"): global worker index I % nWorkers, where nWorkers is the number of
// ranks times the GPUs per rank. Each worker keeps its rows in one device buffer.
//
//   MPI    one rank per node. Ranks exchange the finished panel of L and the
//          factored diagonal block with MPI_Bcast, and gather results to rank 0.
//   OpenMP one host thread per local GPU drives that GPU in every phase.
//   CUDA   tiled GEMM for the panel partial sums, a row-parallel triangular solve,
//          and generation of the test matrix on the GPUs.
//
// Every element is computed with the same sequence of operations as the sequential
// unblocked algorithm: a dot product accumulates k in increasing order, and the
// accumulation continues across blocks. Each term is one explicit fused multiply-add
// on both the host and the GPU, so results do not depend on compiler contraction or
// vectorization choices.

namespace {

constexpr size_t kBlock = 64;  // rows per block, also the panel width
constexpr int kTileM = 32;     // GEMM tile: rows, columns and k depth
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreads = 128;  // (kTileM / 4) * (kTileN / 4)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t err_ = (call);                                       \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// T[m * W + j] = sum_k A[m * lda + k] * R[j * K + k] for m < M and j < W.
// Each output accumulates k in increasing order (tiles are visited in order), so it
// is the same sequential sum as the reference. Tiling only changes data reuse.
__global__ void __launch_bounds__(kThreads)
partialSumKernel(const double* __restrict__ A, size_t lda, const double* __restrict__ R,
                 size_t K, double* __restrict__ T, int M, int W) {
    __shared__ double As[kTileK][kTileM];
    __shared__ double Bs[kTileK][kTileN];

    const int tid = threadIdx.x;
    const int tx = tid % (kTileN / 4);
    const int ty = tid / (kTileN / 4);
    const int m0 = blockIdx.x * kTileM;
    const int j0 = blockIdx.y * kTileN;

    double acc[4][4] = {};
    for (size_t kt = 0; kt < K; kt += kTileK) {
        const size_t rem = K - kt;
        const int kc = rem < (size_t)kTileK ? (int)rem : kTileK;
        for (int idx = tid; idx < kTileM * kTileK; idx += kThreads) {
            const int kk = idx % kTileK;
            const int mm = idx / kTileK;
            const int m = m0 + mm;
            As[kk][mm] = (kk < kc && m < M) ? A[(size_t)m * lda + kt + kk] : 0.0;
        }
        for (int idx = tid; idx < kTileN * kTileK; idx += kThreads) {
            const int kk = idx % kTileK;
            const int jj = idx / kTileK;
            const int j = j0 + jj;
            Bs[kk][jj] = (kk < kc && j < W) ? R[(size_t)j * K + kt + kk] : 0.0;
        }
        __syncthreads();
        for (int kk = 0; kk < kc; ++kk) {
            double a[4], b[4];
            for (int i = 0; i < 4; ++i) a[i] = As[kk][ty * 4 + i];
            for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx * 4 + j];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < 4; ++i) {
        const int m = m0 + ty * 4 + i;
        if (m >= M) continue;
        for (int j = 0; j < 4; ++j) {
            const int jj = j0 + tx * 4 + j;
            if (jj < W) T[(size_t)m * W + jj] = acc[i][j];
        }
    }
}

// Finishes the rows below the current diagonal block. Row r holds A(r, c0 .. c0+w-1)
// on entry and L on exit. T[r * w + j] holds the partial sum over k < c0. D is the
// factored diagonal block (lower triangle, row-major w x w). Each row is independent.
__global__ void solveBelowKernel(double* __restrict__ A, size_t lda, size_t c0,
                                 const double* __restrict__ T, const double* __restrict__ D,
                                 int w, int rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    double* row = A + (size_t)r * lda + c0;
    const double* t = T + (size_t)r * w;
    for (int j = 0; j < w; ++j) {
        const double* dj = D + (size_t)j * w;
        double sum = t[j];
        for (int k = 0; k < j; ++k) sum = fma(row[k], dj[k], sum);
        row[j] = (row[j] - sum) / dj[j];
    }
}

// A(i, i) += value for each stored row. Local row r is offset r % nb in local block
// r / nb, which is global block (r / nb) * nWorkers + worker.
__global__ void addDiagonalKernel(double* __restrict__ A, size_t rows, size_t n, size_t nb,
                                  size_t worker, size_t nWorkers, double value) {
    const size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const size_t globalBlock = (r / nb) * nWorkers + worker;
    const size_t gi = globalBlock * nb + r % nb;
    if (gi < n) A[r * n + gi] += value;
}

// Per-GPU state. A is the working matrix that becomes L in place.
struct Device {
    int ordinal = 0;      // CUDA device index on this rank
    size_t worker = 0;    // global worker index: rank * devCount + ordinal
    size_t rows = 0;      // matrix rows stored on this GPU
    double* A = nullptr;  // rows x n
    double* B = nullptr;  // rows x n: owned rows of the generator matrix (setup only)
    double* T = nullptr;  // rows x kBlock: partial sums of the current panel
    double* P = nullptr;  // kBlock x n: finished panel of L, or generator chunk
    double* D = nullptr;  // kBlock x kBlock: factored diagonal block
};

struct Cluster {
    int rank = 0;
    size_t n = 0;
    size_t numBlocks = 0;
    size_t devCount = 0;     // GPUs per rank (every local GPU is used)
    size_t nWorkers = 0;     // total GPUs across all ranks
    std::vector<Device> dev;
    std::vector<double> panel;    // host: kBlock x n staging for the broadcast panel
    std::vector<double> diag;     // host: kBlock x kBlock diagonal block
    std::vector<double> diagSum;  // host: kBlock x kBlock partial sums of the diagonal block
};

// Block I has blockRows(I) rows. Block I belongs to worker I % nWorkers and sits at
// local row (I / nWorkers) * kBlock on that worker.
size_t blockRows(size_t I, size_t n) { return std::min(kBlock, n - I * kBlock); }

// Number of matrix rows stored by a worker.
size_t ownedRows(size_t worker, size_t n, size_t nWorkers) {
    size_t rows = 0;
    for (size_t I = worker; I * kBlock < n; I += nWorkers) rows += blockRows(I, n);
    return rows;
}

// First local row of a worker's blocks with index >= J (may be past the last row).
size_t firstLocalRow(size_t worker, size_t J, size_t nWorkers) {
    const size_t q = J <= worker ? 0 : (J - worker + nWorkers - 1) / nWorkers;
    return q * kBlock;
}

double* deviceAlloc(size_t count) {
    double* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, std::max<size_t>(count, 1) * sizeof(double)));
    return ptr;
}

void allocateDevices(Cluster& c) {
    c.dev.resize(c.devCount);
    for (size_t g = 0; g < c.devCount; ++g) {
        Device& d = c.dev[g];
        d.ordinal = (int)g;
        d.worker = (size_t)c.rank * c.devCount + g;
        d.rows = ownedRows(d.worker, c.n, c.nWorkers);
        CUDA_CHECK(cudaSetDevice(d.ordinal));
        d.A = deviceAlloc(d.rows * c.n);
        d.B = deviceAlloc(d.rows * c.n);
        d.T = deviceAlloc(d.rows * kBlock);
        d.P = deviceAlloc(kBlock * c.n);
        d.D = deviceAlloc(kBlock * kBlock);
    }
}

void launchPartialSums(const double* A, size_t lda, const double* R, size_t K, double* T,
                       size_t M, size_t W) {
    const dim3 grid((unsigned)((M + kTileM - 1) / kTileM), (unsigned)((W + kTileN - 1) / kTileN));
    partialSumKernel<<<grid, kThreads>>>(A, lda, R, K, T, (int)M, (int)W);
    CUDA_CHECK(cudaGetLastError());
}

// Fills one row of the generator matrix B from the rand_r stream of the reference.
void fillRowOfB(unsigned int& seed, double* row, size_t n) {
    for (size_t k = 0; k < n; ++k) row[k] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
}

// Builds A = B * B^T + n * I, with B from the same rand_r stream as the reference.
// A(i, j) is one full dot product over k in order, as in the reference. Each GPU
// stores its rows of A. Setup is not part of the timed region.
void generateMatrix(Cluster& c) {
    const size_t n = c.n;
    const int G = (int)c.devCount;

    // Pass 1: each GPU keeps its own rows of B.
    unsigned int seed = 42;
    std::vector<double> row(n);
    for (size_t i = 0; i < n; ++i) {
        fillRowOfB(seed, row.data(), n);
        const size_t I = i / kBlock;
        const size_t worker = I % c.nWorkers;
        if (worker / c.devCount != (size_t)c.rank) continue;
        Device& d = c.dev[worker % c.devCount];
        const size_t lr = (I / c.nWorkers) * kBlock + i % kBlock;
        CUDA_CHECK(cudaSetDevice(d.ordinal));
        CUDA_CHECK(cudaMemcpy(d.B + lr * n, row.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Pass 2: stream B in chunks of kBlock rows. Each GPU multiplies its rows of B by
    // the chunk to fill the matching columns of its rows of A.
    seed = 42;
    for (size_t c0 = 0; c0 < n; c0 += kBlock) {
        const size_t w = std::min(kBlock, n - c0);
        for (size_t jj = 0; jj < w; ++jj) fillRowOfB(seed, c.panel.data() + jj * n, n);

        #pragma omp parallel for num_threads(G) schedule(static, 1)
        for (int g = 0; g < G; ++g) {
            Device& d = c.dev[(size_t)g];
            CUDA_CHECK(cudaSetDevice(d.ordinal));
            CUDA_CHECK(cudaMemcpy(d.P, c.panel.data(), w * n * sizeof(double), cudaMemcpyHostToDevice));
            if (d.rows > 0) {
                launchPartialSums(d.B, n, d.P, n, d.T, d.rows, w);
                CUDA_CHECK(cudaMemcpy2D(d.A + c0, n * sizeof(double), d.T, w * sizeof(double),
                                        w * sizeof(double), d.rows, cudaMemcpyDeviceToDevice));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Diagonal dominance, added after the full dot product as in the reference.
    #pragma omp parallel for num_threads(G) schedule(static, 1)
    for (int g = 0; g < G; ++g) {
        Device& d = c.dev[(size_t)g];
        CUDA_CHECK(cudaSetDevice(d.ordinal));
        if (d.rows > 0) {
            const unsigned blocks = (unsigned)((d.rows + 127) / 128);
            addDiagonalKernel<<<blocks, 128>>>(d.A, d.rows, n, kBlock, d.worker, c.nWorkers, (double)n);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d.B));
        d.B = nullptr;
    }
}

// Factors the w x w diagonal block of the panel that starts at global column c0.
// On entry D holds the original A values and T the partial sums over k < c0 (both
// row-major, ld = w). On exit the lower triangle of D holds L. Returns -1 on success,
// or the global index of the first non-positive pivot, as the reference reports it.
long long factorDiagonalBlock(double* D, const double* T, size_t w, size_t c0) {
    for (size_t j = 0; j < w; ++j) {
        double sum = T[j * w + j];
        for (size_t k = 0; k < j; ++k) sum = std::fma(D[j * w + k], D[j * w + k], sum);
        const double val = D[j * w + j] - sum;
        if (val <= 0.0) return (long long)(c0 + j);
        const double ljj = sqrt(val);
        D[j * w + j] = ljj;
        for (size_t i = j + 1; i < w; ++i) {
            double s = T[i * w + j];
            for (size_t k = 0; k < j; ++k) s = std::fma(D[i * w + k], D[j * w + k], s);
            D[i * w + j] = (D[i * w + j] - s) / ljj;
        }
    }
    return -1;
}

// Left-looking blocked factorization in place over all GPUs. For block column J:
//   1. the owner sends the finished columns of block row J (L(J, 0:c0)) to every rank;
//   2. every GPU computes partial sums over k < c0 for its rows at or below block J;
//   3. the owner factors the diagonal block on the host;
//   4. every GPU solves its rows below the diagonal block against that factor.
// Returns false on a non-positive pivot, with the same error report as the reference.
bool factorize(Cluster& c) {
    const size_t n = c.n;
    const size_t nW = c.nWorkers;
    const int G = (int)c.devCount;

    for (size_t J = 0; J < c.numBlocks; ++J) {
        const size_t c0 = J * kBlock;
        const size_t w = blockRows(J, n);
        const size_t owner = J % nW;
        const int ownerRank = (int)(owner / c.devCount);
        const bool ownerHere = ownerRank == c.rank;
        const size_t ownerLb = (J / nW) * kBlock;  // local row of block J on its owner

        // 1. Finished columns of block row J, from the owner, to every rank.
        if (ownerHere && c0 > 0) {
            Device& d = c.dev[owner % c.devCount];
            CUDA_CHECK(cudaSetDevice(d.ordinal));
            CUDA_CHECK(cudaMemcpy2D(c.panel.data(), c0 * sizeof(double), d.A + ownerLb * n,
                                    n * sizeof(double), c0 * sizeof(double), w, cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(c.panel.data(), (int)(w * c0), MPI_DOUBLE, ownerRank, MPI_COMM_WORLD);

        // 2. Partial sums over k < c0 for every local row at or below block J.
        #pragma omp parallel for num_threads(G) schedule(static, 1)
        for (int g = 0; g < G; ++g) {
            Device& d = c.dev[(size_t)g];
            CUDA_CHECK(cudaSetDevice(d.ordinal));
            const size_t lb = firstLocalRow(d.worker, J, nW);
            if (lb < d.rows) {
                if (c0 > 0) CUDA_CHECK(cudaMemcpy(d.P, c.panel.data(), w * c0 * sizeof(double),
                                                  cudaMemcpyHostToDevice));
                launchPartialSums(d.A + lb * n, n, d.P, c0, d.T, d.rows - lb, w);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // 3. The owner factors the diagonal block on the host, reporting any failure.
        long long status = -1;
        if (ownerHere) {
            Device& d = c.dev[owner % c.devCount];
            CUDA_CHECK(cudaSetDevice(d.ordinal));
            // The owner's block J rows are the first w rows of its T, ld = w.
            CUDA_CHECK(cudaMemcpy(c.diagSum.data(), d.T, w * w * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy2D(c.diag.data(), w * sizeof(double), d.A + ownerLb * n + c0,
                                    n * sizeof(double), w * sizeof(double), w, cudaMemcpyDeviceToHost));
            status = factorDiagonalBlock(c.diag.data(), c.diagSum.data(), w, c0);
            if (status < 0) {
                CUDA_CHECK(cudaMemcpy2D(d.A + ownerLb * n + c0, n * sizeof(double), c.diag.data(),
                                        w * sizeof(double), w * sizeof(double), w, cudaMemcpyHostToDevice));
            }
        }
        MPI_Bcast(&status, 1, MPI_LONG_LONG, ownerRank, MPI_COMM_WORLD);
        if (status >= 0) {
            if (c.rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)status);
            }
            return false;
        }
        MPI_Bcast(c.diag.data(), (int)(w * w), MPI_DOUBLE, ownerRank, MPI_COMM_WORLD);

        // 4. Rows below the diagonal block, solved on every GPU. The owner's block J
        //    rows are skipped because step 3 already finished them.
        #pragma omp parallel for num_threads(G) schedule(static, 1)
        for (int g = 0; g < G; ++g) {
            Device& d = c.dev[(size_t)g];
            CUDA_CHECK(cudaSetDevice(d.ordinal));
            const size_t lb = firstLocalRow(d.worker, J, nW);
            const size_t start = lb + (d.worker == owner ? w : 0);
            if (start < d.rows) {
                const size_t cnt = d.rows - start;
                CUDA_CHECK(cudaMemcpy(d.D, c.diag.data(), w * w * sizeof(double), cudaMemcpyHostToDevice));
                const unsigned blocks = (unsigned)((cnt + 127) / 128);
                solveBelowKernel<<<blocks, 128>>>(d.A + start * n, n, c0, d.T + (start - lb) * w,
                                                  d.D, (int)w, (int)cnt);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    return true;
}

// Collects the distributed n x n matrix into `full` on rank 0. Blocks are sent in
// block order, so the receives on rank 0 match the sends without deadlock.
void gatherMatrix(Cluster& c, std::vector<double>& full) {
    const size_t n = c.n;
    if (c.rank == 0) full.assign(n * n, 0.0);
    for (size_t I = 0; I < c.numBlocks; ++I) {
        const size_t owner = I % c.nWorkers;
        const int ownerRank = (int)(owner / c.devCount);
        const size_t w = blockRows(I, n);
        if (ownerRank == c.rank) {
            Device& d = c.dev[owner % c.devCount];
            CUDA_CHECK(cudaSetDevice(d.ordinal));
            const double* src = d.A + (I / c.nWorkers) * kBlock * n;
            if (c.rank == 0) {
                CUDA_CHECK(cudaMemcpy(full.data() + I * kBlock * n, src, w * n * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            } else {
                CUDA_CHECK(cudaMemcpy(c.panel.data(), src, w * n * sizeof(double), cudaMemcpyDeviceToHost));
                MPI_Send(c.panel.data(), (int)(w * n), MPI_DOUBLE, 0, (int)I, MPI_COMM_WORLD);
            }
        } else if (c.rank == 0) {
            MPI_Recv(full.data() + I * kBlock * n, (int)(w * n), MPI_DOUBLE, ownerRank, (int)I,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

void releaseDevices(Cluster& c) {
    for (Device& d : c.dev) {
        CUDA_CHECK(cudaSetDevice(d.ordinal));
        CUDA_CHECK(cudaFree(d.A));
        CUDA_CHECK(cudaFree(d.T));
        CUDA_CHECK(cudaFree(d.P));
        CUDA_CHECK(cudaFree(d.D));
    }
}

} // namespace

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T. Rows are independent, so they run in parallel.
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

int runBenchmark(int argc, char** argv, int rank, int size) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    const bool root = rank == 0;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount < 1) {
        fprintf(stderr, "rank %d: no CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Cluster c;
    c.rank = rank;
    c.n = n;
    c.numBlocks = (n + kBlock - 1) / kBlock;
    c.devCount = (size_t)devCount;
    c.nWorkers = (size_t)size * c.devCount;
    c.panel.resize(kBlock * n);
    c.diag.resize(kBlock * kBlock);
    c.diagSum.resize(kBlock * kBlock);

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    allocateDevices(c);

    // Generate positive definite matrix
    if (root) printf("Generating positive definite matrix...\n");
    generateMatrix(c);

    std::vector<double> A_orig;
    if (validate) {
        gatherMatrix(c, A_orig);  // Save original for validation
    }

    // Perform Cholesky decomposition
    if (root) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const bool success = factorize(c);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDurationMs = duration.count();
    long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (root) printf("Cholesky decomposition failed\n");
        releaseDevices(c);
        return 1;
    }

    if (root) {
        printf("Computation time: %ld ms\n", maxDurationMs);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (maxDurationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> L;
    if (printResults || validate) {
        gatherMatrix(c, L);
        if (root) {
            // Upper triangle is not part of L
            for (size_t i = 0; i < n; ++i)
                for (size_t j = i + 1; j < n; ++j) L[i * n + j] = 0.0;
        }
    }

    // Print results for external validation
    if (root && printResults) {
        print_results(L, "CholeskyL");
    }

    int ret = 0;
    if (validate) {
        if (root) {
            printf("Validating result...\n");
            bool valid = validateCholesky(L, A_orig, n);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
        MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    releaseDevices(c);
    return ret;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int ret = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return ret;
}
