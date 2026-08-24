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
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)st); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t st = (call); \
    if (st != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, (int)st); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Blocked Cholesky decomposition using MPI + OpenMP + CUDA
// Uses column-major layout for GPU compatibility
// 1D block-cyclic distribution of block-columns across MPI ranks

static cublasHandle_t cublasH;
static cusolverDnHandle_t cusolverH;

// Convert row-major to column-major and vice versa
static void rowToCol(const double* row, double* col, size_t n) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            col[j * n + i] = row[i * n + j];
}

static void colToRow(const double* col, double* row, size_t n) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            row[i * n + j] = col[j * n + i];
}

// GPU POTRF: Cholesky factor a small diagonal block on GPU
static bool gpuPotrf(double* d_block, int nb, int lda) {
    int worksize = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, nb, d_block, lda, &worksize));
    double* d_work;
    int* d_info;
    CUDA_CHECK(cudaMalloc(&d_work, worksize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, nb, d_block, lda, d_work, worksize, d_info));
    int h_info;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    return (h_info == 0);
}

// GPU TRSM: Solve L * X^T = B for panel blocks
static void gpuTrsm(double* d_L, double* d_B, int nb, int mb, int lda) {
    const double alpha = 1.0;
    // B = B * L^{-T} => solve X * L^T = B => TRSM right, lower, transpose
    CUBLAS_CHECK(cublasDtrsm(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                             CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                             mb, nb, &alpha, d_L, lda, d_B, lda));
}

// GPU SYRK: Symmetric rank-k update
static void gpuSyrk(double* d_C, const double* d_A, int nb, int kb, int lda) {
    const double alpha = -1.0, beta = 1.0;
    CUBLAS_CHECK(cublasDsyrk(cublasH, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             nb, kb, &alpha, d_A, lda, &beta, d_C, lda));
}

// GPU GEMM: C -= A * B^T
static void gpuGemm(double* d_C, const double* d_A, const double* d_B, int m, int n, int k, int lda) {
    const double alpha = -1.0, beta = 1.0;
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                             m, n, k, &alpha, d_A, lda, d_B, lda, &beta, d_C, lda));
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Block size for tiled algorithm
    const int NB = 256;
    const int nblocks = ((int)n + NB - 1) / NB;

    // Convert to column-major for GPU BLAS
    std::vector<double> Acol(n * n);
    rowToCol(A.data(), Acol.data(), n);

    // Allocate full matrix on GPU (each rank has full copy for simplicity)
    double* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, Acol.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    // Temporary GPU buffer for broadcast blocks
    double* d_panel;
    CUDA_CHECK(cudaMalloc(&d_panel, n * NB * sizeof(double)));

    // Host buffer for MPI communication
    std::vector<double> h_buf(n * NB);

    for (int kb = 0; kb < nblocks; ++kb) {
        const int k_start = kb * NB;
        const int k_nb = std::min(NB, (int)n - k_start);
        const int owner = kb % nprocs;

        // Step 1: POTRF on diagonal block (owner rank)
        if (rank == owner) {
            // d_A + k_start * n + k_start points to diagonal block in col-major
            if (!gpuPotrf(d_A + (size_t)k_start * n + k_start, k_nb, (int)n)) {
                fprintf(stderr, "Error: Matrix is not positive definite at block %d\n", kb);
                CUDA_CHECK(cudaFree(d_A));
                CUDA_CHECK(cudaFree(d_panel));
                return false;
            }
        }

        // Broadcast diagonal block to all ranks
        if (rank == owner) {
            // Copy diagonal block to contiguous host buffer (column-major, k_nb x k_nb)
            CUDA_CHECK(cudaMemcpy2D(h_buf.data(), k_nb * sizeof(double),
                                     d_A + (size_t)k_start * n + k_start, n * sizeof(double),
                                     k_nb * sizeof(double), k_nb, cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(h_buf.data(), k_nb * k_nb, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (rank != owner) {
            CUDA_CHECK(cudaMemcpy2D(d_A + (size_t)k_start * n + k_start, n * sizeof(double),
                                     h_buf.data(), k_nb * sizeof(double),
                                     k_nb * sizeof(double), k_nb, cudaMemcpyHostToDevice));
        }

        // Step 2: TRSM for sub-diagonal blocks in column kb
        // Distribute block-rows among ranks
        const int remaining = (int)n - k_start - k_nb;
        if (remaining > 0) {
            // Each rank processes a subset of the panel rows
            // For simplicity, owner does TRSM then broadcasts
            if (rank == owner) {
                gpuTrsm(d_A + (size_t)k_start * n + k_start,
                        d_A + (size_t)k_start * n + k_start + k_nb,
                        k_nb, remaining, (int)n);
            }

            // Broadcast the panel column to all ranks
            if (rank == owner) {
                CUDA_CHECK(cudaMemcpy2D(h_buf.data(), remaining * sizeof(double),
                                         d_A + (size_t)k_start * n + k_start + k_nb, n * sizeof(double),
                                         remaining * sizeof(double), k_nb, cudaMemcpyDeviceToHost));
            }
            MPI_Bcast(h_buf.data(), remaining * k_nb, MPI_DOUBLE, owner, MPI_COMM_WORLD);
            if (rank != owner) {
                CUDA_CHECK(cudaMemcpy2D(d_A + (size_t)k_start * n + k_start + k_nb, n * sizeof(double),
                                         h_buf.data(), remaining * sizeof(double),
                                         remaining * sizeof(double), k_nb, cudaMemcpyHostToDevice));
            }
        }

        // Step 3 & 4: Update trailing submatrix
        // Distribute block-columns of trailing matrix among ranks
        for (int jb = kb + 1; jb < nblocks; ++jb) {
            if (jb % nprocs != rank) continue; // 1D block-cyclic distribution

            const int j_start = jb * NB;
            const int j_nb = std::min(NB, (int)n - j_start);

            // SYRK for diagonal block: A[jb,jb] -= A[jb,kb] * A[jb,kb]^T
            if (jb == jb) { // diagonal update
                gpuSyrk(d_A + (size_t)j_start * n + j_start,
                        d_A + (size_t)k_start * n + j_start,
                        j_nb, k_nb, (int)n);
            }

            // GEMM for sub-diagonal blocks: A[ib,jb] -= A[ib,kb] * A[jb,kb]^T
            for (int ib = jb + 1; ib < nblocks; ++ib) {
                const int i_start = ib * NB;
                const int i_nb = std::min(NB, (int)n - i_start);

                gpuGemm(d_A + (size_t)j_start * n + i_start,
                        d_A + (size_t)k_start * n + i_start,
                        d_A + (size_t)k_start * n + j_start,
                        i_nb, j_nb, k_nb, (int)n);
            }
        }

        // Synchronize GPU after all updates for this step
        CUDA_CHECK(cudaDeviceSynchronize());

        // Gather updated columns from all ranks to all ranks for next iteration
        // Each rank broadcasts the block-columns it owns in the trailing matrix
        for (int jb = kb + 1; jb < nblocks; ++jb) {
            const int j_owner = jb % nprocs;
            const int j_start = jb * NB;
            const int j_nb = std::min(NB, (int)n - j_start);
            const int col_rows = (int)n - j_start; // rows from j_start to end

            if (col_rows <= 0) continue;

            if (rank == j_owner) {
                CUDA_CHECK(cudaMemcpy2D(h_buf.data(), col_rows * sizeof(double),
                                         d_A + (size_t)j_start * n + j_start, n * sizeof(double),
                                         col_rows * sizeof(double), j_nb, cudaMemcpyDeviceToHost));
            }
            MPI_Bcast(h_buf.data(), col_rows * j_nb, MPI_DOUBLE, j_owner, MPI_COMM_WORLD);
            if (rank != j_owner) {
                CUDA_CHECK(cudaMemcpy2D(d_A + (size_t)j_start * n + j_start, n * sizeof(double),
                                         h_buf.data(), col_rows * sizeof(double),
                                         col_rows * sizeof(double), j_nb, cudaMemcpyHostToDevice));
            }
        }
    }

    // Copy result back
    CUDA_CHECK(cudaMemcpy(Acol.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_panel));

    // Convert back to row-major
    colToRow(Acol.data(), A.data(), n);

    // Zero out upper triangular part
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T using OpenMP
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

    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    // Compute L * L^T using OpenMP
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

    double maxError = 0.0;
    double relError = 0.0;

    #pragma omp parallel for reduction(max:maxError, relError) schedule(static)
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

    // Assign GPU to rank (round-robin over available devices)
    int ndevices;
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    CUDA_CHECK(cudaSetDevice(rank % ndevices));

    // Initialize cuBLAS and cuSOLVER
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

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
        printf("Cholesky Decomposition Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d, OpenMP threads: %d, GPUs: %d\n", nprocs, omp_get_max_threads(), ndevices);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix (all ranks generate same matrix)
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);

    if (validate) {
        A_orig = A;
    }

    // Perform Cholesky decomposition
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = static_cast<long>(duration.count());
    long global_duration_ms;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        cublasDestroy(cublasH);
        cusolverDnDestroy(cusolverH);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration_ms);

        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (global_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }

    // Validation (rank 0 only)
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            cublasDestroy(cublasH);
            cusolverDnDestroy(cusolverH);
            MPI_Finalize();
            return 1;
        }
    }

    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);
    MPI_Finalize();
    return 0;
}
