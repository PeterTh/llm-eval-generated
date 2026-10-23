#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// -----------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA Cholesky benchmark.
//
// The matrix is distributed over MPI ranks in a (block-cyclic) column layout.
// At every step a panel is factored on the GPU of the owning rank, broadcast to
// the other ranks, and the trailing matrix update is performed with a batched
// DGEMM on every GPU.  OpenMP is used for the host-side data-parallel work
// (matrix generation, validation and result assembly).
// -----------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), \
                    __FILE__, __LINE__);                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _s = (call);                                             \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s, __FILE__,    \
                    __LINE__);                                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

namespace {

struct BlockDistribution {
    size_t n = 0;
    int ranks = 1;
    int rank = 0;
    int block = 64;
    size_t local_cols = 0;

    BlockDistribution() = default;

    BlockDistribution(size_t n_, int ranks_, int rank_, int block_)
        : n(n_), ranks(ranks_), rank(rank_), block(block_) {
        local_cols = 0;
        for (size_t t = (size_t)rank; t * (size_t)block < n;
             t += (size_t)ranks) {
            const size_t start = t * (size_t)block;
            local_cols += std::min<size_t>((size_t)block, n - start);
        }
    }

    inline size_t globalCol(size_t local_col) const {
        const size_t q = local_col / (size_t)block;
        const size_t within = local_col % (size_t)block;
        const size_t t = q * (size_t)ranks + (size_t)rank;
        return t * (size_t)block + within;
    }

    // First local column that belongs to a block strictly after the panel
    // whose first global column is kb.
    inline size_t trailingStart(size_t kb) const {
        const size_t t_panel = kb / (size_t)block;
        if (t_panel < (size_t)rank) {
            return 0;
        }
        const size_t qmax = (t_panel - (size_t)rank) / (size_t)ranks;
        const size_t start = (qmax + 1) * (size_t)block;
        return std::min(start, local_cols);
    }

    inline size_t localOffsetOfBlock(size_t t_block) const {
        return (t_block / (size_t)ranks) * (size_t)block;
    }
};

size_t localColumnCount(size_t n, int rank, int ranks, int block) {
    size_t count = 0;
    for (size_t t = (size_t)rank; t * (size_t)block < n;
         t += (size_t)ranks) {
        const size_t start = t * (size_t)block;
        count += std::min<size_t>((size_t)block, n - start);
    }
    return count;
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Factor the (small) diagonal block of a panel.  A is column-major with
// leading dimension n.  Only elements on and below the diagonal are touched.
__global__ void panelDiagonalKernel(double* __restrict__ A, int n, int kb,
                                    int b, int loff, int* __restrict__ err,
                                    int* __restrict__ err_idx) {
    const int tid = threadIdx.x;
    for (int jj = 0; jj < b; ++jj) {
        const int c = kb + jj;
        const int col = loff + jj;

        if (tid == 0) {
            // Clear the upper-triangular part of this panel column.
            for (int r = kb; r < c; ++r) {
                A[(size_t)col * n + r] = 0.0;
            }

            double s = 0.0;
            for (int k = kb; k < c; ++k) {
                const int prev_col = loff + (k - kb);
                const double v = A[(size_t)prev_col * n + c];
                s += v * v;
            }
            const double val = A[(size_t)col * n + c] - s;
            if (val <= 0.0) {
                err[0] = 1;
                err_idx[0] = c;
                A[(size_t)col * n + c] = 0.0;
            } else {
                A[(size_t)col * n + c] = sqrt(val);
            }
        }
        __syncthreads();

        if (err[0]) {
            break;
        }

        const double diag = A[(size_t)col * n + c];
        for (int r = c + 1 + tid; r < kb + b; r += blockDim.x) {
            double s = 0.0;
            for (int k = kb; k < c; ++k) {
                const int prev_col = loff + (k - kb);
                s += A[(size_t)prev_col * n + r] *
                     A[(size_t)prev_col * n + c];
            }
            A[(size_t)col * n + r] = (A[(size_t)col * n + r] - s) / diag;
        }
        __syncthreads();
    }
}

// Gather the panel rows corresponding to this rank's trailing local columns.
// Panel is column-major (n x b), P2 is column-major (M x b).
__global__ void gatherPanelKernel(const double* __restrict__ panel,
                                  double* __restrict__ P2, int n, int b,
                                  int rank, int ranks, int block, int suffix,
                                  int M) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = b * M;
    if (idx >= total) {
        return;
    }
    const int t = idx / M;
    const int i = idx - t * M;
    const int loc = suffix + i;
    const int q = loc / block;
    const int within = loc - q * block;
    const int gcol = (q * ranks + rank) * block + within;
    P2[idx] = panel[(size_t)t * n + gcol];
}

// Zero the global upper triangle of the distributed matrix.
__global__ void zeroUpperKernel(double* __restrict__ A, int n, int rank,
                                int ranks, int block, long long total) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int i = (int)(idx / n);
    const int r = (int)(idx % n);
    const int q = i / block;
    const int within = i - q * block;
    const int gcol = (q * ranks + rank) * block + within;
    if (r < gcol) {
        A[(size_t)i * n + r] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

// Generate this rank's columns of a symmetric positive definite matrix
// (A = B * B^T + n*I).  B uses the same random stream as the original code and
// the distributed product is evaluated on the GPU.
void generateLocalColumns(std::vector<double>& localA,
                          const BlockDistribution& dist, int threadCount,
                          cublasHandle_t handle) {
    const size_t n = dist.n;
    const size_t localCols = dist.local_cols;

    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    if (localCols > 0) {
        std::vector<double> Blocal(localCols * n);
#pragma omp parallel for schedule(static) num_threads(threadCount)
        for (long long ii = 0; ii < (long long)localCols; ++ii) {
            const size_t i = (size_t)ii;
            const size_t gcol = dist.globalCol(i);
            memcpy(&Blocal[i * n], &B[gcol * n], n * sizeof(double));
        }

        double* d_B = nullptr;
        double* d_Blocal = nullptr;
        double* d_C = nullptr;
        CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Blocal, localCols * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_C, localCols * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * n * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Blocal, Blocal.data(),
                              localCols * n * sizeof(double),
                              cudaMemcpyHostToDevice));

        const double alpha = 1.0;
        const double beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)n,
                                 (int)localCols, (int)n, &alpha, d_B, (int)n,
                                 d_Blocal, (int)n, &beta, d_C, (int)n));
        CUDA_CHECK(cudaMemcpy(localA.data(), d_C,
                              localCols * n * sizeof(double),
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_Blocal));
        CUDA_CHECK(cudaFree(d_C));
    }

#pragma omp parallel for schedule(static) num_threads(threadCount)
    for (long long ii = 0; ii < (long long)localCols; ++ii) {
        const size_t i = (size_t)ii;
        const size_t gcol = dist.globalCol(i);
        localA[i * n + gcol] += (double)n;
    }
}

void gatherToFull(const std::vector<double>& localA,
                  const BlockDistribution& dist, int root,
                  std::vector<double>& full, MPI_Comm comm, int threadCount) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    const int ranks = dist.ranks;
    const size_t n = dist.n;

    std::vector<int> counts(ranks, 0);
    std::vector<int> displs(ranks, 0);
    int total = 0;
    for (int r = 0; r < ranks; ++r) {
        const size_t lc = localColumnCount(n, r, ranks, dist.block);
        counts[r] = (int)(lc * n);
        displs[r] = total;
        total += counts[r];
    }

    std::vector<double> recv;
    if (rank == root) {
        recv.resize((size_t)total);
    }

    MPI_Gatherv(localA.data(), (int)localA.size(), MPI_DOUBLE,
                (rank == root) ? recv.data() : nullptr, counts.data(),
                displs.data(), MPI_DOUBLE, root, comm);

    if (rank != root) {
        return;
    }

    full.assign(n * n, 0.0);
    for (int r = 0; r < ranks; ++r) {
        const size_t lc = (size_t)counts[r] / n;
        const double* src = recv.data() + displs[r];
#pragma omp parallel for schedule(static) num_threads(threadCount)
        for (long long rr = 0; rr < (long long)n; ++rr) {
            const size_t row = (size_t)rr;
            for (size_t i = 0; i < lc; ++i) {
                const size_t q = i / (size_t)dist.block;
                const size_t within = i % (size_t)dist.block;
                const size_t t = q * (size_t)ranks + (size_t)r;
                const size_t gcol = t * (size_t)dist.block + within;
                full[row * n + gcol] = src[i * n + row];
            }
        }
    }
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig, const size_t n,
                      int threadCount) {
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for collapse(2) schedule(static) num_threads(threadCount) \
    reduction(max : maxError) reduction(max : relError)
    for (long long ii = 0; ii < (long long)n; ++ii) {
        for (long long jj = 0; jj < (long long)n; ++jj) {
            const size_t i = (size_t)ii;
            const size_t j = (size_t)jj;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const size_t idx = i * n + j;
            const double error = fabs(sum - A_orig[idx]);
            maxError = std::max(maxError, error);
            const double rel = error / (fabs(A_orig[idx]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }

    return true;
}

bool choleskyDecomposition(double* d_A, const BlockDistribution& dist,
                           int ranks, int rank, int block,
                           cublasHandle_t handle, MPI_Comm comm) {
    const size_t n = dist.n;
    const size_t localCols = dist.local_cols;
    const int n_int = (int)n;

    std::vector<double> hPanel(n * (size_t)block, 0.0);

    double* d_panel = nullptr;
    double* d_P2 = nullptr;
    int* d_err = nullptr;
    CUDA_CHECK(cudaMalloc(&d_panel, n * (size_t)block * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_P2,
                          std::max<size_t>(localCols, 1) * (size_t)block *
                              sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_err, 2 * sizeof(int)));

    bool ok = true;

    for (size_t kb = 0; kb < n; kb += (size_t)block) {
        const int b = (int)std::min<size_t>((size_t)block, n - kb);
        const size_t panelEnd = kb + (size_t)b;
        const size_t tPanel = kb / (size_t)block;
        const int owner = (int)(tPanel % (size_t)ranks);

        int localErr = 0;
        int localErrIdx = -1;

        if (rank == owner) {
            CUDA_CHECK(cudaMemsetAsync(d_err, 0, 2 * sizeof(int)));
            const size_t loff = dist.localOffsetOfBlock(tPanel);
            int threads = b < 128 ? b : 128;
            if (threads < 32) {
                threads = 32;
            }
            panelDiagonalKernel<<<1, threads>>>(d_A, n_int, (int)kb, b,
                                                (int)loff, d_err, d_err + 1);
            CUDA_CHECK(cudaGetLastError());

            int hErr[2] = {0, 0};
            CUDA_CHECK(cudaMemcpy(hErr, d_err, 2 * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            localErr = hErr[0];
            localErrIdx = hErr[1];

            if (!localErr) {
                if (panelEnd < n) {
                    const double alpha = 1.0;
                    CUBLAS_CHECK(cublasDtrsm(
                        handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                        (int)(n - panelEnd), b, &alpha,
                        d_A + loff * n + kb, n_int,
                        d_A + loff * n + panelEnd, n_int));
                }
                CUDA_CHECK(cudaMemcpy(hPanel.data(), d_A + loff * n,
                                      n * (size_t)b * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            }
        }

        int globalErr = 0;
        int globalErrIdx = -1;
        MPI_Allreduce(&localErr, &globalErr, 1, MPI_INT, MPI_MAX, comm);
        MPI_Allreduce(&localErrIdx, &globalErrIdx, 1, MPI_INT, MPI_MAX, comm);

        if (globalErr) {
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal "
                       "element %d\n",
                       globalErrIdx);
            }
            ok = false;
            break;
        }

        MPI_Bcast(hPanel.data(), (int)(n * (size_t)b), MPI_DOUBLE, owner,
                  comm);
        CUDA_CHECK(cudaMemcpy(d_panel, hPanel.data(),
                              n * (size_t)b * sizeof(double),
                              cudaMemcpyHostToDevice));

        const size_t suffix = dist.trailingStart(kb);
        const size_t M = localCols - suffix;
        const size_t nrows = n - panelEnd;

        if (M > 0 && nrows > 0 && b > 0) {
            const int total = (int)((size_t)b * M);
            const int gatherThreads = 256;
            const int gatherBlocks = (total + gatherThreads - 1) / gatherThreads;
            gatherPanelKernel<<<gatherBlocks, gatherThreads>>>(
                d_panel, d_P2, n_int, b, rank, ranks, block, (int)suffix,
                (int)M);
            CUDA_CHECK(cudaGetLastError());

            const double alpha = -1.0;
            const double beta = 1.0;
            CUBLAS_CHECK(cublasDgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_T, (int)nrows, (int)M, b,
                &alpha, d_panel + panelEnd, n_int, d_P2, (int)M, &beta,
                d_A + suffix * n + panelEnd, n_int));
        }
    }

    if (ok && localCols > 0) {
        const long long total = (long long)(localCols * n);
        const int threads = 256;
        const long long blocks = (total + threads - 1) / threads;
        zeroUpperKernel<<<(unsigned int)blocks, threads>>>(
            d_A, n_int, rank, ranks, block, total);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(d_panel));
    CUDA_CHECK(cudaFree(d_P2));
    CUDA_CHECK(cudaFree(d_err));

    return ok;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int chooseBlockSize(size_t n, int ranks) {
    int block = 64;
    if (n >= 2048 && n < 8192) {
        block = 128;
    } else if (n >= 8192) {
        block = 256;
    }
    while (block > 32 && (n / (size_t)block) < (size_t)(2 * ranks)) {
        block /= 2;
    }
    return block;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (n == 0) {
        if (rank == 0) {
            printf("Cholesky Decomposition Benchmark\n");
            printf("Matrix size: 0 x 0\n");
        }
        MPI_Finalize();
        return 0;
    }

    const int block = chooseBlockSize(n, ranks);
    const BlockDistribution dist(n, ranks, rank, block);

    // Assign one (or a fraction of) GPU(s) to each MPI rank on a node.
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    int threadCount = omp_get_max_threads();

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> localA(n * dist.local_cols);
    std::vector<double> localOrig;
    if (validate) {
        localOrig.resize(localA.size());
    }

    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generateLocalColumns(localA, dist, threadCount, handle);
    if (validate) {
        localOrig = localA;
    }

    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,
                          std::max<size_t>(localA.size(), 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, localA.data(),
                          localA.size() * sizeof(double),
                          cudaMemcpyHostToDevice));

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const bool success = choleskyDecomposition(d_A, dist, ranks, rank, block,
                                               handle, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();

    const double seconds =
        std::chrono::duration<double>(end - start).count();
    const long long milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    if (rank == 0) {
        printf("Computation time: %lld ms\n", milliseconds);
        const double ops = (double)n * n * n / 3.0;
        const double gflops = ops / seconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> fullA;
    std::vector<double> fullOrig;

    if (success && (printResults || validate)) {
        if (!localA.empty()) {
            CUDA_CHECK(cudaMemcpy(localA.data(), d_A,
                                  localA.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        gatherToFull(localA, dist, 0, fullA, MPI_COMM_WORLD, threadCount);
        if (validate) {
            gatherToFull(localOrig, dist, 0, fullOrig, MPI_COMM_WORLD,
                         threadCount);
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    CUBLAS_CHECK(cublasDestroy(handle));
    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_free(&nodeComm);
    }

    int rc = success ? 0 : 1;

    if (rank == 0) {
        if (!success) {
            printf("Cholesky decomposition failed\n");
        } else {
            if (printResults) {
                print_results(fullA, "CholeskyL");
            }
            if (validate) {
                // The generated matrix is mathematically symmetric; use the
                // lower triangle (the part consumed by Cholesky) to remove any
                // last-bit asymmetry from the distributed GEMM.
#pragma omp parallel for schedule(static) num_threads(threadCount)
                for (long long ii = 0; ii < (long long)n; ++ii) {
                    for (long long jj = 0; jj < ii; ++jj) {
                        const size_t i = (size_t)ii;
                        const size_t j = (size_t)jj;
                        fullOrig[j * n + i] = fullOrig[i * n + j];
                    }
                }
                printf("Validating result...\n");
                const bool valid =
                    validateCholesky(fullA, fullOrig, n, threadCount);
                if (valid) {
                    printf("Validation: PASSED\n");
                    rc = 0;
                } else {
                    printf("Validation: FAILED\n");
                    rc = 1;
                }
            }
        }
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
