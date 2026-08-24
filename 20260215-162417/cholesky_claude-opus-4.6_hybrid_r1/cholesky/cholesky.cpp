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

// Tile size for blocked Cholesky
#define BLOCK_SIZE 128
// Shared-memory tile dimension for GEMM kernel
#define TILE_DIM 16
// Number of CUDA streams for overlapping GPU work
#define NUM_STREAMS 4

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// ===================== CUDA Kernels =====================

// C[M×N] -= A[M×K] * B[N×K]^T   (row-major, strided)
__global__ void dgemm_nt_kernel(int M, int N, int K,
                                double* __restrict__ C, int ldc,
                                const double* __restrict__ A, int lda,
                                const double* __restrict__ B, int ldb) {
    __shared__ double As[TILE_DIM][TILE_DIM + 1];
    __shared__ double Bs[TILE_DIM][TILE_DIM + 1];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    double sum = 0.0;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        int acol = t * TILE_DIM + threadIdx.x;
        int bcol = t * TILE_DIM + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && acol < K) ? A[row * lda + acol] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (col < N && bcol < K) ? B[col * ldb + bcol] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE_DIM; kk++)
            sum += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * ldc + col] -= sum;
}

// Solve B := B * inv(L^T), L lower-triangular N×N, B is M×N (one thread per row)
__global__ void dtrsm_rlt_kernel(int M, int N,
                                 double* __restrict__ B, int ldb,
                                 const double* __restrict__ L, int ldl) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    for (int j = 0; j < N; j++) {
        double v = B[row * ldb + j];
        for (int k = 0; k < j; k++)
            v -= L[j * ldl + k] * B[row * ldb + k];
        B[row * ldb + j] = v / L[j * ldl + j];
    }
}

// ===================== Host helpers =====================

// In-place Cholesky of small tile (row-major, stride lda)
static bool potrf_cpu(double* A, int n, int lda) {
    for (int j = 0; j < n; j++) {
        double s = 0.0;
        for (int k = 0; k < j; k++)
            s += A[j * lda + k] * A[j * lda + k];
        double v = A[j * lda + j] - s;
        if (v <= 0.0) {
            fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %d\n", j);
            return false;
        }
        A[j * lda + j] = sqrt(v);
        for (int i = j + 1; i < n; i++) {
            s = 0.0;
            for (int k = 0; k < j; k++)
                s += A[i * lda + k] * A[j * lda + k];
            A[i * lda + j] = (A[i * lda + j] - s) / A[j * lda + j];
        }
    }
    return true;
}

// Generate symmetric positive definite matrix (deterministic across ranks)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    // Compute A = B * B^T  (parallelized with OpenMP)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
    for (size_t i = 0; i < n; ++i)
        A[i * n + i] += n;
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> recon(n * n);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k)
                s += L[i * n + k] * L[j * n + k];
            recon[i * n + j] = s;
        }

    double maxError = 0.0, relError = 0.0;
    #pragma omp parallel for reduction(max:maxError,relError)
    for (size_t i = 0; i < n * n; ++i) {
        double e = fabs(recon[i] - A_orig[i]);
        maxError = std::max(maxError, e);
        relError = std::max(relError, e / (fabs(A_orig[i]) + 1e-10));
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

// =========== Blocked Cholesky with MPI + OpenMP + CUDA ===========

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    if (ndev > 0) cudaSetDevice(rank % ndev);

    const int nb = std::min((int)n, BLOCK_SIZE);
    const int nt = ((int)n + nb - 1) / nb;

    // Device copy of the full matrix
    double* d_A = nullptr;
    size_t mat_bytes = n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_A, mat_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), mat_bytes, cudaMemcpyHostToDevice));

    // Host contiguous tile buffer for MPI exchange
    std::vector<double> h_tile((size_t)nb * nb);

    // Multiple CUDA streams for GPU concurrency
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    bool success = true;

    for (int bk = 0; bk < nt && success; bk++) {
        int kb = std::min(nb, (int)n - bk * nb);
        int owner_k = bk % nprocs;
        size_t kk_off = (size_t)bk * nb * n + (size_t)bk * nb;

        // --- POTRF: factor diagonal tile on CPU ---
        int potrf_ok = 1;
        if (rank == owner_k) {
            CUDA_CHECK(cudaMemcpy2D(h_tile.data(), kb * sizeof(double),
                                    d_A + kk_off, n * sizeof(double),
                                    kb * sizeof(double), kb,
                                    cudaMemcpyDeviceToHost));
            potrf_ok = potrf_cpu(h_tile.data(), kb, kb) ? 1 : 0;
        }
        MPI_Bcast(&potrf_ok, 1, MPI_INT, owner_k, MPI_COMM_WORLD);
        if (!potrf_ok) { success = false; break; }
        MPI_Bcast(h_tile.data(), kb * kb, MPI_DOUBLE, owner_k, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy2D(d_A + kk_off, n * sizeof(double),
                                h_tile.data(), kb * sizeof(double),
                                kb * sizeof(double), kb,
                                cudaMemcpyHostToDevice));

        if (bk == nt - 1) break;  // no trailing submatrix

        // --- TRSM: solve tiles in column bk below diagonal ---
        {
            int sid = 0;
            for (int bi = bk + 1; bi < nt; bi++) {
                if (bi % nprocs == rank) {
                    int ib = std::min(nb, (int)n - bi * nb);
                    double* d_B = d_A + (size_t)bi * nb * n + (size_t)bk * nb;
                    double* d_L = d_A + kk_off;
                    int nblk = (ib + 255) / 256;
                    dtrsm_rlt_kernel<<<nblk, 256, 0, streams[sid % NUM_STREAMS]>>>(
                        ib, kb, d_B, (int)n, d_L, (int)n);
                    sid++;
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Exchange column bk tiles via MPI ---
        for (int bi = bk + 1; bi < nt; bi++) {
            int ib = std::min(nb, (int)n - bi * nb);
            int owner_i = bi % nprocs;
            size_t ik_off = (size_t)bi * nb * n + (size_t)bk * nb;
            if (rank == owner_i)
                CUDA_CHECK(cudaMemcpy2D(h_tile.data(), kb * sizeof(double),
                                        d_A + ik_off, n * sizeof(double),
                                        kb * sizeof(double), ib,
                                        cudaMemcpyDeviceToHost));
            MPI_Bcast(h_tile.data(), ib * kb, MPI_DOUBLE, owner_i, MPI_COMM_WORLD);
            if (rank != owner_i)
                CUDA_CHECK(cudaMemcpy2D(d_A + ik_off, n * sizeof(double),
                                        h_tile.data(), kb * sizeof(double),
                                        kb * sizeof(double), ib,
                                        cudaMemcpyHostToDevice));
        }

        // --- GEMM: update trailing submatrix ---
        {
            int sid = 0;
            for (int bi = bk + 1; bi < nt; bi++) {
                if (bi % nprocs == rank) {
                    int ib = std::min(nb, (int)n - bi * nb);
                    double* d_Aik = d_A + (size_t)bi * nb * n + (size_t)bk * nb;
                    for (int bj = bk + 1; bj <= bi; bj++) {
                        int jb = std::min(nb, (int)n - bj * nb);
                        double* d_C   = d_A + (size_t)bi * nb * n + (size_t)bj * nb;
                        double* d_Ajk = d_A + (size_t)bj * nb * n + (size_t)bk * nb;
                        dim3 block(TILE_DIM, TILE_DIM);
                        dim3 grid((jb + TILE_DIM - 1) / TILE_DIM,
                                  (ib + TILE_DIM - 1) / TILE_DIM);
                        dgemm_nt_kernel<<<grid, block, 0, streams[sid % NUM_STREAMS]>>>(
                            ib, jb, kb, d_C, (int)n, d_Aik, (int)n, d_Ajk, (int)n);
                        sid++;
                    }
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // --- Gather factored rows to all ranks ---
    for (int bi = 0; bi < nt; bi++) {
        int ib = std::min(nb, (int)n - bi * nb);
        int owner_i = bi % nprocs;
        size_t row_off = (size_t)bi * nb * n;
        if (rank == owner_i)
            CUDA_CHECK(cudaMemcpy(A.data() + row_off, d_A + row_off,
                                  (size_t)ib * n * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        MPI_Bcast(A.data() + row_off, ib * (int)n, MPI_DOUBLE,
                  owner_i, MPI_COMM_WORLD);
    }

    // Zero upper triangle (OpenMP parallelized)
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        for (size_t j = i + 1; j < n; j++)
            A[i * n + j] = 0.0;

    for (int s = 0; s < NUM_STREAMS; s++)
        cudaStreamDestroy(streams[s]);
    cudaFree(d_A);
    return success;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d, OpenMP threads: %d\n", nprocs, omp_get_max_threads());
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

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    int ret = 0;
    if (rank == 0) {
        printf("Computation time: %lld ms\n", global_duration_ms);
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (global_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) print_results(A, "CholeskyL");

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }

    MPI_Finalize();
    return ret;
}
