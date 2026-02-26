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
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, stat); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t stat = call; \
    if (stat != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, stat); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Hybrid MPI+OpenMP+CUDA Cholesky decomposition using blocked algorithm
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

struct CholeskyContext {
    int rank, nprocs;
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;
    double *d_work;
    int *d_info;
    int lwork;
};

void initCholeskyContext(CholeskyContext& ctx, int rank, int nprocs, size_t n) {
    ctx.rank = rank;
    ctx.nprocs = nprocs;
    
    CUBLAS_CHECK(cublasCreate(&ctx.cublasHandle));
    CUSOLVER_CHECK(cusolverDnCreate(&ctx.cusolverHandle));
    
    // Allocate workspace for cuSOLVER
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(ctx.cusolverHandle, 
        CUBLAS_FILL_MODE_LOWER, n, nullptr, n, &ctx.lwork));
    
    CUDA_CHECK(cudaMalloc(&ctx.d_work, sizeof(double) * ctx.lwork));
    CUDA_CHECK(cudaMalloc(&ctx.d_info, sizeof(int)));
}

void destroyCholeskyContext(CholeskyContext& ctx) {
    CUDA_CHECK(cudaFree(ctx.d_work));
    CUDA_CHECK(cudaFree(ctx.d_info));
    CUBLAS_CHECK(cublasDestroy(ctx.cublasHandle));
    CUSOLVER_CHECK(cusolverDnDestroy(ctx.cusolverHandle));
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n, 
                          CholeskyContext& ctx) {
    
    // Convert row-major to column-major on rank 0
    std::vector<double> A_colmajor;
    if (ctx.rank == 0) {
        A_colmajor.resize(n * n);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A_colmajor[j * n + i] = A[i * n + j];
            }
        }
    }
    
    // Broadcast column-major matrix to all ranks for distributed processing
    if (ctx.rank != 0) {
        A_colmajor.resize(n * n);
    }
    MPI_Bcast(&A_colmajor[0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Each rank gets a GPU device and processes data
    // Rank 0 performs the main Cholesky factorization on GPU
    if (ctx.rank == 0) {
        double *d_A = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * n * n));
        CUDA_CHECK(cudaMemcpy(d_A, &A_colmajor[0], sizeof(double) * n * n, cudaMemcpyHostToDevice));
        
        int info_h = 0;
        CUSOLVER_CHECK(cusolverDnDpotrf(ctx.cusolverHandle,
            CUBLAS_FILL_MODE_LOWER,
            n,
            d_A,
            n,
            ctx.d_work,
            ctx.lwork,
            ctx.d_info));
        
        CUDA_CHECK(cudaMemcpy(&info_h, ctx.d_info, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (info_h != 0) {
            printf("Error: Matrix is not positive definite (info = %d)\n", info_h);
            CUDA_CHECK(cudaFree(d_A));
            return false;
        }
        
        CUDA_CHECK(cudaMemcpy(&A_colmajor[0], d_A, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_A));
    }
    
    // Broadcast result to all ranks
    MPI_Bcast(&A_colmajor[0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Parallel conversion back to row-major across all ranks
    // Each rank processes a portion of rows
    size_t rows_per_rank = (n + ctx.nprocs - 1) / ctx.nprocs;
    size_t start_row = ctx.rank * rows_per_rank;
    size_t end_row = std::min(start_row + rows_per_rank, n);
    
    #pragma omp parallel for collapse(2)
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i * n + j] = A_colmajor[j * n + i];
        }
    }
    
    // Gather results at rank 0
    if (ctx.rank == 0) {
        for (int src = 1; src < ctx.nprocs; ++src) {
            size_t src_start = src * rows_per_rank;
            size_t src_end = std::min(src_start + rows_per_rank, n);
            if (src_start < n) {
                size_t src_rows = src_end - src_start;
                MPI_Recv(&A[src_start * n], src_rows * n, MPI_DOUBLE,
                        src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else if (start_row < n) {
        size_t local_rows = end_row - start_row;
        MPI_Send(&A[start_row * n], local_rows * n, MPI_DOUBLE,
                0, 0, MPI_COMM_WORLD);
    }
    
    // Broadcast final result to all ranks
    MPI_Bcast(&A[0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Zero out upper triangular part in parallel with OpenMP across all ranks
    #pragma omp parallel for collapse(2)
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    
    // Final synchronization to ensure all ranks have zeroed out upper triangular
    MPI_Barrier(MPI_COMM_WORLD);
    if (ctx.rank == 0) {
        for (int src = 1; src < ctx.nprocs; ++src) {
            size_t src_start = src * rows_per_rank;
            size_t src_end = std::min(src_start + rows_per_rank, n);
            if (src_start < n) {
                size_t src_rows = src_end - src_start;
                MPI_Recv(&A[src_start * n], src_rows * n, MPI_DOUBLE,
                        src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else if (start_row < n) {
        size_t local_rows = end_row - start_row;
        MPI_Send(&A[start_row * n], local_rows * n, MPI_DOUBLE,
                0, 1, MPI_COMM_WORLD);
    }
    
    // Final broadcast
    MPI_Bcast(&A[0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    return true;
}

// Generate a symmetric positive definite matrix (parallelized with OpenMP)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    
    // Generate random matrix B in parallel
    #pragma omp parallel
    {
        unsigned int seed = 42 + omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < n * n; ++i) {
            B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
    }
    
    // Compute A = B * B^T in parallel
    #pragma omp parallel for collapse(2)
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
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix (parallelized with OpenMP)
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T in parallel
    #pragma omp parallel for collapse(2)
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
    
    #pragma omp parallel for reduction(max:maxError,relError)
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    // Set GPU device based on rank
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    
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
    
    if (rank == 0) {
        printf("Hybrid MPI+OpenMP+CUDA Cholesky Decomposition Benchmark\n");
        printf("MPI processes: %d\n", nprocs);
        printf("OpenMP threads per process: %d\n", omp_get_max_threads());
        printf("CUDA devices: %d\n", deviceCount);
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix on rank 0
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        
        if (validate) {
            A_orig = A; // Save original for validation
        }
    }
    
    // Broadcast matrix to all ranks
    MPI_Bcast(&A[0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (validate && rank != 0) {
        A_orig = A;
    }
    
    // Initialize Cholesky context
    CholeskyContext ctx;
    initCholeskyContext(ctx, rank, nprocs, n);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, ctx);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        destroyCholeskyContext(ctx);
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
    
    // Print results for external validation (only rank 0)
    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, n);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                destroyCholeskyContext(ctx);
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    destroyCholeskyContext(ctx);
    MPI_Finalize();
    
    return 0;
}
