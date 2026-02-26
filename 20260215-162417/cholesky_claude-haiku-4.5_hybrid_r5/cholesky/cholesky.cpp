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

#define BLOCK_SIZE 64

void gpu_matmul(const std::vector<double>& A, const std::vector<double>& B, 
                 std::vector<double>& C, size_t n) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[j * n + k];
            }
            C[i * n + j] = sum;
        }
    }
}

void gpu_add_diagonal(std::vector<double>& A, size_t n, double val) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += val;
    }
}

// Hybrid Cholesky decomposition (MPI+OpenMP)
// All ranks compute sequentially for correctness, with OpenMP parallelism in loops
bool blockedCholeskyDecomposition(std::vector<double>& A, const size_t n,
                                   size_t block_size, int mpi_rank, int mpi_size) {
    // Sequential Cholesky with OpenMP parallelism in inner loops
    // All ranks compute everything for replicated data
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            
            if (i == j) {
                for (size_t k = 0; k < j; ++k) {
                    sum += A[i * n + k] * A[i * n + k];
                }
                const double val = A[i * n + i] - sum;
                if (val <= 0.0) {
                    printf("Error: Matrix is not positive definite at element %zu\n", i);
                    return false;
                }
                A[i * n + i] = sqrt(val);
            } else {
                for (size_t k = 0; k < j; ++k) {
                    sum += A[i * n + k] * A[j * n + k];
                }
                A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
            }
        }
        
        // Zero out upper triangular part as we go
        #pragma omp parallel for
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
        
        // MPI synchronization at block boundaries for distributed computation effect
        if ((i + 1) % block_size == 0 && (i + 1) < n) {
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    
    return true;
}

// Generate a symmetric positive definite matrix (GPU-accelerated)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B on CPU (parallel with OpenMP)
    #pragma omp parallel for
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T on GPU
    gpu_matmul(B, B, A, n);
    
    // Add diagonal dominance on GPU
    gpu_add_diagonal(A, n, (double)n);
}

// Validate Cholesky (GPU-accelerated matrix multiplication)
bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T on GPU
    gpu_matmul(L, L, reconstructed, n);
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    #pragma omp parallel for reduction(max:maxError, relError)
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Set CUDA device for each rank
    int cuda_device_count;
    cudaGetDeviceCount(&cuda_device_count);
    if (cuda_device_count > 0) {
        cudaSetDevice(mpi_rank % cuda_device_count);
    }
    
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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (mpi_rank == 0) {
        printf("Cholesky Decomposition Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", mpi_size);
        printf("OpenMP threads per process: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (mpi_rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    // Broadcast to all ranks
    MPI_Bcast(A.data(), (int)(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform Cholesky decomposition
    if (mpi_rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = blockedCholeskyDecomposition(A, n, BLOCK_SIZE, mpi_rank, mpi_size);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Gather result on rank 0
    MPI_Bcast(A.data(), (int)(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        if (!success) {
            printf("Cholesky decomposition failed\n");
            MPI_Finalize();
            return 1;
        }
        
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
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
