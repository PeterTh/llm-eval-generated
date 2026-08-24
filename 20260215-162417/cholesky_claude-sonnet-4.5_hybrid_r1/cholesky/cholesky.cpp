#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "../common/results_output.hpp"

#ifdef HAVE_CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)
#endif

#ifdef HAVE_CUDA
// CUDA kernel for dot product computation
__global__ void dotProductKernel(const double* __restrict__ A, const double* __restrict__ B,
                                  double* __restrict__ result, size_t n, size_t start_idx) {
    extern __shared__ double sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double sum = 0.0;
    if (i < n) {
        sum = A[i] * B[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// CUDA kernel for row update
__global__ void rowUpdateKernel(double* __restrict__ A, const double* __restrict__ rowJ,
                                 double diagJ, size_t n, size_t j, size_t startRow, size_t numRows) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numRows) {
        size_t row = startRow + i;
        if (row > j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A[i * n + k] * rowJ[k];
            }
            A[i * n + j] = (A[i * n + j] - sum) / diagJ;
        }
    }
}
#endif

// Hybrid MPI+OpenMP+CUDA Cholesky decomposition
bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int size) {
#ifdef HAVE_CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count > 0) {
        CUDA_CHECK(cudaSetDevice(rank % device_count));
    }
    
    // Allocate GPU memory for temporary computations
    double *d_temp1 = nullptr, *d_temp2 = nullptr, *d_result = nullptr;
    size_t max_size = n * sizeof(double);
    
    if (device_count > 0) {
        CUDA_CHECK(cudaMalloc(&d_temp1, max_size));
        CUDA_CHECK(cudaMalloc(&d_temp2, max_size));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    }
#endif
    
    std::vector<double> rowBuffer(n);
    
    for (size_t j = 0; j < n; ++j) {
        // Determine which rank owns row j
        int owner = j % size;
        
        if (rank == owner) {
            // This rank computes the diagonal element
            double sum = 0.0;
            
#ifdef HAVE_CUDA
            // Use CUDA for diagonal computation if available
            if (device_count > 0 && j > 32) {
                CUDA_CHECK(cudaMemset(d_result, 0, sizeof(double)));
                CUDA_CHECK(cudaMemcpy(d_temp1, &A[j * n], j * sizeof(double), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_temp2, &A[j * n], j * sizeof(double), cudaMemcpyHostToDevice));
                
                int blockSize = 256;
                int gridSize = (j + blockSize - 1) / blockSize;
                dotProductKernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(
                    d_temp1, d_temp2, d_result, j, 0);
                CUDA_CHECK(cudaMemcpy(&sum, d_result, sizeof(double), cudaMemcpyDeviceToHost));
            } else
#endif
            {
                #pragma omp parallel for reduction(+:sum)
                for (size_t k = 0; k < j; ++k) {
                    sum += A[j * n + k] * A[j * n + k];
                }
            }
            
            const double val = A[j * n + j] - sum;
            if (val <= 0.0) {
                fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %zu\n", j);
#ifdef HAVE_CUDA
                if (device_count > 0) {
                    cudaFree(d_temp1);
                    cudaFree(d_temp2);
                    cudaFree(d_result);
                }
#endif
                MPI_Abort(MPI_COMM_WORLD, 1);
                return false;
            }
            A[j * n + j] = sqrt(val);
            
            // Copy row j to buffer for broadcasting
            std::copy(&A[j * n], &A[j * n + n], rowBuffer.begin());
        }
        
        // Broadcast the computed row to all ranks
        MPI_Bcast(rowBuffer.data(), n, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        
        // Update row j in all ranks
        if (rank == owner) {
            std::copy(rowBuffer.begin(), rowBuffer.end(), &A[j * n]);
        }
        
        // All ranks update their owned rows
        #pragma omp parallel for schedule(dynamic, 8)
        for (size_t i = j + 1; i < n; ++i) {
            // Check if this rank owns row i
            if ((i % size) == (size_t)rank) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < j; ++k) {
                    sum += A[i * n + k] * rowBuffer[k];
                }
                A[i * n + j] = (A[i * n + j] - sum) / rowBuffer[j];
            }
        }
        
        // Zero out upper triangular part for row j
        if (rank == owner) {
            #pragma omp parallel for
            for (size_t jj = j + 1; jj < n; ++jj) {
                A[j * n + jj] = 0.0;
            }
        }
    }
    
    // Gather all rows back to all ranks (for validation)
    for (size_t i = 0; i < n; ++i) {
        int owner = i % size;
        MPI_Bcast(&A[i * n], n, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    }
    
#ifdef HAVE_CUDA
    if (device_count > 0) {
        cudaFree(d_temp1);
        cudaFree(d_temp2);
        cudaFree(d_result);
    }
#endif
    
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, int rank) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Only rank 0 generates the matrix
    if (rank == 0) {
        // Generate random matrix B
        for (size_t i = 0; i < n * n; ++i) {
            B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
        
        // Compute A = B * B^T with OpenMP
        #pragma omp parallel for schedule(dynamic, 16)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
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
    
    // Broadcast the matrix to all ranks
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n, int rank) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T with OpenMP
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
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
    
    if (rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }
    
    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (rank == 0) {
            printf("Validation failed: relative error too large\n");
        }
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
    
    int rank, size;
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
        printf("Cholesky Decomposition Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        
#ifdef HAVE_CUDA
        int device_count;
        cudaGetDeviceCount(&device_count);
        printf("CUDA devices: %d\n", device_count);
#else
        printf("CUDA devices: 0 (not compiled with CUDA support)\n");
#endif
        
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n, rank);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long local_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long global_duration = 0;
    MPI_Reduce(&local_duration, &global_duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration);
        
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (global_duration / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(A, "CholeskyL");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(A, A_orig, n, rank);
        
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
