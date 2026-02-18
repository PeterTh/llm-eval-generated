#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <mpi.h>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <cusolverDn.h>
#endif

#include "../common/results_output.hpp"

#ifdef ENABLE_CUDA
// CUDA error checking
#define CUDA_CHECK(err) \
    if ((err) != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        return false; \
    }

#define CUSOLVER_CHECK(err) \
    if ((err) != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER Error: %d\n", err); \
        return false; \
    }
#endif

// Hybrid Cholesky decomposition using CUDA, OpenMP, and MPI
class HybridCholesky {
private:
    int rank, world_size;
    int cuda_device;
    bool use_gpu;
    
public:
    HybridCholesky() : rank(0), world_size(1), cuda_device(0), use_gpu(false) {
        // Initialize MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        
        // Check for GPU availability
#ifdef ENABLE_CUDA
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            cuda_device = rank % device_count;
            cudaSetDevice(cuda_device);
            use_gpu = true;
            if (rank == 0) {
                printf("CUDA GPU detected and enabled\n");
            }
        }
#endif
    }
    
    ~HybridCholesky() {
#ifdef ENABLE_CUDA
        if (use_gpu) {
            cudaDeviceSynchronize();
        }
#endif
    }
    
    bool decompose(std::vector<double>& A, const size_t n) {
        if (use_gpu) {
            return decomposeGPU(A, n);
        } else {
            return decomposeCPU(A, n);
        }
    }
    
private:
#ifdef ENABLE_CUDA
    bool decomposeGPU(std::vector<double>& A, const size_t n) {
        // GPU-accelerated Cholesky using cuSOLVER
        cusolverDnHandle_t cusolver_handle = nullptr;
        cudaStream_t cuda_stream = nullptr;
        
        cusolverDnCreate(&cusolver_handle);
        cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
        cusolverDnSetStream(cusolver_handle, cuda_stream);
        
        // Allocate GPU memory
        double *d_A = nullptr;
        int *d_info = nullptr;
        int *info = nullptr;
        
        CUDA_CHECK(cudaMalloc((void**)&d_A, sizeof(double) * n * n));
        CUDA_CHECK(cudaMalloc((void**)&d_info, sizeof(int)));
        
        info = (int *)malloc(sizeof(int));
        
        // Copy matrix to GPU
        CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * n * n, 
                                    cudaMemcpyHostToDevice, cuda_stream));
        
        // Query workspace size
        int workspace_size = 0;
        cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, 
                                    n, d_A, n, &workspace_size);
        
        double *d_work = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_work, sizeof(double) * workspace_size));
        
        // Perform Cholesky decomposition
        cusolverStatus_t status = cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                                                    n, d_A, n, d_work, workspace_size, d_info);
        CUSOLVER_CHECK(status);
        
        // Copy info back
        CUDA_CHECK(cudaMemcpyAsync(info, d_info, sizeof(int), 
                                    cudaMemcpyDeviceToHost, cuda_stream));
        
        // Copy result back to CPU
        CUDA_CHECK(cudaMemcpyAsync(A.data(), d_A, sizeof(double) * n * n, 
                                    cudaMemcpyDeviceToHost, cuda_stream));
        
        // Synchronize
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
        
        // Check for errors
        if (*info != 0) {
            fprintf(stderr, "Error: Matrix is not positive definite at position %d\n", *info);
            cudaFree(d_A);
            cudaFree(d_work);
            cudaFree(d_info);
            free(info);
            cusolverDnDestroy(cusolver_handle);
            cudaStreamDestroy(cuda_stream);
            return false;
        }
        
        // Zero out upper triangular part using OpenMP
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                A[i * n + j] = 0.0;
            }
        }
        
        // Cleanup
        cudaFree(d_A);
        cudaFree(d_work);
        cudaFree(d_info);
        free(info);
        cusolverDnDestroy(cusolver_handle);
        cudaStreamDestroy(cuda_stream);
        
        return true;
    }
#else
    bool decomposeGPU(std::vector<double>& A, const size_t n) {
        // CUDA not available, fall back to CPU
        return decomposeCPU(A, n);
    }
#endif
    
    bool decomposeCPU(std::vector<double>& A, const size_t n) {
        // OpenMP-parallelized Cholesky on CPU
        
        for (size_t i = 0; i < n; ++i) {
            // Diagonal element computation
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum) schedule(static)
            for (size_t k = 0; k < i; ++k) {
                sum += A[i * n + k] * A[i * n + k];
            }
            
            const double val = A[i * n + i] - sum;
            if (val <= 0.0) {
                fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %zu\n", i);
                return false;
            }
            A[i * n + i] = sqrt(val);
            
            // Off-diagonal elements
            #pragma omp parallel for schedule(dynamic, 32)
            for (size_t j = i + 1; j < n; ++j) {
                double row_sum = 0.0;
                for (size_t k = 0; k < i; ++k) {
                    row_sum += A[j * n + k] * A[i * n + k];
                }
                A[j * n + i] = (A[j * n + i] - row_sum) / A[i * n + i];
            }
        }
        
        // Zero out upper triangular part
        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t i = 0; i < n * n; ++i) {
            size_t row = i / n;
            size_t col = i % n;
            if (col > row) {
                A[i] = 0.0;
            }
        }
        
        return true;
    }
};

// Generate a symmetric positive definite matrix with OpenMP parallelization
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    
    // Generate random matrix B with OpenMP
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n * n; ++i) {
        unsigned int seed = 42 + i;
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T with OpenMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
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
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T with OpenMP parallelization
    #pragma omp parallel for collapse(2) schedule(dynamic)
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
    
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
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
    
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Parse command line arguments (only on rank 0)
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
        
        printf("Cholesky Decomposition Benchmark (Hybrid MPI/OpenMP/CUDA)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Number of MPI processes: %d\n", 1); // Will show actual count after broadcast
    }
    
    // Broadcast parameters
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    // Allocate matrix (all ranks allocate, but only rank 0 computes)
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    if (rank == 0) {
        // Generate positive definite matrix
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        
        if (validate) {
            A_orig = A; // Save original for validation
        }
        
        // Perform Cholesky decomposition with hybrid approach
        printf("Computing Cholesky decomposition...\n");
    }
    
    // Broadcast matrix to all ranks
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Only rank 0 performs computation (can be extended for distributed approach)
    bool success = true;
    if (rank == 0) {
        HybridCholesky hybrid;
        success = hybrid.decompose(A, n);
    }
    
    // Broadcast result back
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&success, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
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
