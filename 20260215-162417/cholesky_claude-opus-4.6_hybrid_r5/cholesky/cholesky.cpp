#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Blocked Cholesky with MPI (inter-node) + OpenMP (intra-node) + CUDA (GPU compute)
// 1D block-cyclic column distribution across MPI ranks
// CUDA kernels for POTRF, TRSM, GEMM
// OpenMP parallelism across independent GEMM updates with multiple CUDA streams

static constexpr int BLK  = 64;   // block size for tiling
static constexpr int TILE = 16;   // thread-block tile for GEMM kernel

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// ===================================================================
// CUDA Kernels
// ===================================================================

// In-place Cholesky factorization of a bs×bs block.
// Block located at pointer A with leading dimension ld (row-major).
// Single thread-block; threads cooperate on off-diagonal rows.
__global__ void potrf_kernel(double* A, int bs, int ld) {
    int tid = threadIdx.x;
    int nt  = blockDim.x;
    for (int j = 0; j < bs; j++) {
        if (tid == 0) {
            double s = A[j * ld + j];
            for (int k = 0; k < j; k++) {
                double v = A[j * ld + k];
                s -= v * v;
            }
            A[j * ld + j] = sqrt(s);
        }
        __syncthreads();
        double djj = A[j * ld + j];
        for (int i = j + 1 + tid; i < bs; i += nt) {
            double s = A[i * ld + j];
            for (int k = 0; k < j; k++)
                s -= A[i * ld + k] * A[j * ld + k];
            A[i * ld + j] = s / djj;
        }
        __syncthreads();
    }
}

// Right triangular solve  X · L^T = B   (X overwrites B in-place).
// X/B is rows×bs with leading dimension ld_x.
// L   is bs×bs lower-triangular with leading dimension ld_l.
// Each CUDA thread handles one row independently.
__global__ void trsm_kernel(double* X,
                            const double* __restrict__ L,
                            int rows, int bs, int ld_x, int ld_l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    for (int j = 0; j < bs; j++) {
        double s = X[i * ld_x + j];
        for (int k = 0; k < j; k++)
            s -= X[i * ld_x + k] * L[j * ld_l + k];
        X[i * ld_x + j] = s / L[j * ld_l + j];
    }
}

// GEMM: C -= A · B^T   (A is m×p, B is n×p, C is m×n, row-major).
// Shared-memory tiled kernel with bank-conflict padding.
__global__ void gemm_nt_kernel(double* C,
                               const double* __restrict__ A,
                               const double* __restrict__ B,
                               int m, int n, int p,
                               int ld_c, int ld_a, int ld_b) {
    __shared__ double As[TILE][TILE + 1];
    __shared__ double Bs[TILE][TILE + 1];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    double acc = 0.0;
    int nT = (p + TILE - 1) / TILE;
    for (int t = 0; t < nT; t++) {
        int ak = t * TILE + threadIdx.x;
        int bk = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < m && ak < p) ? A[row * ld_a + ak] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (col < n && bk < p) ? B[col * ld_b + bk] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE; kk++)
            acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n)
        C[row * ld_c + col] -= acc;
}

// Zero out the strict upper triangle of an n×n row-major matrix.
__global__ void zero_upper_kernel(double* A, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    while (idx < total) {
        int r = idx / n;
        int c = idx % n;
        if (c > r) A[idx] = 0.0;
        idx += blockDim.x * gridDim.x;
    }
}

// ===================================================================
// Blocked Cholesky  (MPI + OpenMP + CUDA)
// ===================================================================

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int N  = static_cast<int>(n);
    const int nb = (N + BLK - 1) / BLK;

    // Remember which GPU this rank uses
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    // Device matrix
    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)N * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), (size_t)N * N * sizeof(double),
                           cudaMemcpyHostToDevice));

    // Host buffer for MPI broadcasts (max size: full block-column)
    std::vector<double> bcast_buf((size_t)N * BLK);

    // One CUDA stream per OpenMP thread for concurrent GEMM launches
    int nstreams = std::max(1, omp_get_max_threads());
    std::vector<cudaStream_t> streams(nstreams);
    for (int i = 0; i < nstreams; i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // Ensure all OpenMP threads are bound to the correct GPU
    #pragma omp parallel
    { CUDA_CHECK(cudaSetDevice(device)); }

    for (int k = 0; k < nb; k++) {
        int owner  = k % nprocs;
        int ks     = k * BLK;
        int bs_k   = std::min(BLK, N - ks);
        int pstart = ks + bs_k;          // first row of panel below diagonal block
        int prows  = N - pstart;          // rows in that panel

        // ---- 1. POTRF: factorize diagonal block on owner ----
        if (rank == owner) {
            potrf_kernel<<<1, std::min(bs_k, 256)>>>(
                d_A + (size_t)ks * N + ks, bs_k, N);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ---- 2. Broadcast diagonal block ----
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy2D(
                bcast_buf.data(), bs_k * sizeof(double),
                d_A + (size_t)ks * N + ks, N * sizeof(double),
                bs_k * sizeof(double), bs_k,
                cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(bcast_buf.data(), bs_k * bs_k, MPI_DOUBLE,
                  owner, MPI_COMM_WORLD);
        if (rank != owner) {
            CUDA_CHECK(cudaMemcpy2D(
                d_A + (size_t)ks * N + ks, N * sizeof(double),
                bcast_buf.data(), bs_k * sizeof(double),
                bs_k * sizeof(double), bs_k,
                cudaMemcpyHostToDevice));
        }

        if (prows <= 0) continue;

        // ---- 3. TRSM: solve panel below diagonal on owner ----
        if (rank == owner) {
            int trsm_threads = 256;
            int trsm_blocks  = (prows + trsm_threads - 1) / trsm_threads;
            trsm_kernel<<<trsm_blocks, trsm_threads>>>(
                d_A + (size_t)pstart * N + ks,   // panel X
                d_A + (size_t)ks * N + ks,       // diagonal L
                prows, bs_k, N, N);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ---- 4. Broadcast panel column k ----
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy2D(
                bcast_buf.data(), bs_k * sizeof(double),
                d_A + (size_t)pstart * N + ks, N * sizeof(double),
                bs_k * sizeof(double), prows,
                cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(bcast_buf.data(), prows * bs_k, MPI_DOUBLE,
                  owner, MPI_COMM_WORLD);
        if (rank != owner) {
            CUDA_CHECK(cudaMemcpy2D(
                d_A + (size_t)pstart * N + ks, N * sizeof(double),
                bcast_buf.data(), bs_k * sizeof(double),
                bs_k * sizeof(double), prows,
                cudaMemcpyHostToDevice));
        }

        // ---- 5. GEMM: update trailing sub-matrix (owned columns) ----
        //   A[bi][bj] -= L[bi][k] · L[bj][k]^T   for bj > k owned by this rank, bi >= bj
        #pragma omp parallel for schedule(dynamic)
        for (int bj = k + 1; bj < nb; bj++) {
            if (bj % nprocs != rank) continue;
            int js   = bj * BLK;
            int bs_j = std::min(BLK, N - js);
            int sid  = omp_get_thread_num() % nstreams;

            for (int bi = bj; bi < nb; bi++) {
                int is_  = bi * BLK;
                int bs_i = std::min(BLK, N - is_);
                dim3 grid((bs_j + TILE - 1) / TILE,
                          (bs_i + TILE - 1) / TILE);
                dim3 block(TILE, TILE);
                gemm_nt_kernel<<<grid, block, 0, streams[sid]>>>(
                    d_A + (size_t)is_ * N + js,   // C block(bi,bj)
                    d_A + (size_t)is_ * N + ks,   // A block(bi,k)
                    d_A + (size_t)js  * N + ks,   // B block(bj,k)
                    bs_i, bs_j, bs_k, N, N, N);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Zero strict upper triangle
    {
        int t = 256;
        int g = std::min(((int)(N * N) + t - 1) / t, 65535);
        zero_upper_kernel<<<g, t>>>(d_A, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy result back
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, (size_t)N * N * sizeof(double),
                           cudaMemcpyDeviceToHost));

    for (int i = 0; i < nstreams; i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaFree(d_A));
    return true;
}

// ===================================================================
// Matrix generation (deterministic, identical on every rank)
// ===================================================================

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
    }

    for (size_t i = 0; i < n; ++i)
        A[i * n + i] += n;
}

// ===================================================================
// Validation (rank 0 only)
// ===================================================================

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += L[i * n + k] * L[j * n + k];
            reconstructed[i * n + j] = sum;
        }
    }

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

    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

// ===================================================================
// Main
// ===================================================================

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Assign GPU: one GPU per rank, cycling through available devices
    int ndevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    CUDA_CHECK(cudaSetDevice(rank % ndevices));

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
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

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs: %d\n",
               nprocs, omp_get_max_threads(), ndevices);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);

    if (validate) A_orig = A;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");

    auto start = std::chrono::high_resolution_clock::now();
    bool success = choleskyDecomposition(A, n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }

    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
