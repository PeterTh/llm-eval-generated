#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA blocked Cholesky decomposition
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#define CUDA_CALL(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); return false; } } while(0)

// CUDA kernel: update trailing submatrix A22 -= A21 * A21^T for lower triangle
__global__ void trailing_update_kernel(double* A, int n, int k, int kb) {
    int m = n - (k + kb);
    if (m <= 0) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * m;
    if (idx >= total) return;
    int row = idx / m;
    int col = idx % m;
    if (row < col) return; // only compute lower triangle
    int i = k + kb + row;
    int j = k + kb + col;
    double sum = 0.0;
    for (int s = 0; s < kb; ++s) {
        double a_i = A[i * n + (k + s)];
        double a_j = A[j * n + (k + s)];
        sum += a_i * a_j;
    }
    A[i * n + j] -= sum;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int mpi_rank = 0, mpi_size = 1;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

    // Select CUDA device based on rank for multi-GPU nodes
    int devCount = 0;
#ifdef USE_CUDA
    cudaGetDeviceCount(&devCount);
    if (devCount > 0) {
        int dev = mpi_rank % devCount;
        cudaSetDevice(dev);
    }
#endif

    // Allocate device memory for full matrix (each rank keeps full copy for simplicity)
    double* dA = nullptr;
    size_t bytes = n * n * sizeof(double);
#ifdef USE_CUDA
    CUDA_CALL(cudaMalloc((void**)&dA, bytes));
    CUDA_CALL(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
#endif

    const int bs = 64; // block size

    for (int k = 0; k < (int)n; k += bs) {
        int kb = std::min(bs, (int)n - k);

        // Diagonal block buffer
        std::vector<double> diag(kb * kb);

        // Rank 0 computes cholesky on the diagonal block
        if (mpi_rank == 0) {
            // Copy diagonal block into contiguous buffer
            for (int i = 0; i < kb; ++i) {
                for (int j = 0; j < kb; ++j) {
                    diag[i * kb + j] = A[(k + i) * n + (k + j)];
                }
            }

            // Compute Cholesky on diag (lower triangular)
            for (int i = 0; i < kb; ++i) {
                for (int j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    if (i == j) {
                        for (int t = 0; t < j; ++t) sum += diag[j * kb + t] * diag[j * kb + t];
                        double val = diag[j * kb + j] - sum;
                        if (val <= 0.0) {
                            if (mpi_rank == 0) fprintf(stderr, "Error: Matrix not positive definite at block diagonal\n");
#ifdef USE_CUDA
                            cudaFree(dA);
#endif
                            return false;
                        }
                        diag[j * kb + j] = sqrt(val);
                    } else {
                        for (int t = 0; t < j; ++t) sum += diag[i * kb + t] * diag[j * kb + t];
                        diag[i * kb + j] = (diag[i * kb + j] - sum) / diag[j * kb + j];
                    }
                }
            }

            // Write back diagonal block into A (lower triangle)
            for (int i = 0; i < kb; ++i) {
                for (int j = 0; j < kb; ++j) {
                    A[(k + i) * n + (k + j)] = diag[i * kb + j];
                }
            }
        }

        // Broadcast diagonal block to all ranks
#ifdef USE_MPI
        MPI_Bcast(diag.data(), kb * kb, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        // Copy diag into local A copy (all ranks)
        for (int i = 0; i < kb; ++i) {
            for (int j = 0; j < kb; ++j) {
                A[(k + i) * n + (k + j)] = diag[i * kb + j];
            }
        }

        // Triangular solve for the panel below the diagonal (compute L(i, k:k+kb-1))
        // Parallelize with OpenMP over rows
        int start = k + kb;
        if (start < (int)n) {
#pragma omp parallel for schedule(static)
            for (int ii = start; ii < (int)n; ++ii) {
                for (int jj = 0; jj < kb; ++jj) {
                    int col = k + jj;
                    double sum = 0.0;
                    for (int t = 0; t < jj; ++t) {
                        sum += A[ii * n + (k + t)] * A[col * n + (k + t)];
                    }
                    A[ii * n + col] = (A[ii * n + col] - sum) / A[col * n + col];
                }
            }
        }

        // Copy full matrix to device and perform trailing update on GPU
#ifdef USE_CUDA
        CUDA_CALL(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
        int m = n - (k + kb);
        if (m > 0) {
            int total = m * m;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            trailing_update_kernel<<<blocks, threads>>>(dA, (int)n, k, kb);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());
            // copy back updated trailing submatrix
            CUDA_CALL(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
        }
#endif
    }

    // Zero out upper triangular part for consistency (only do on rank 0)
    if (mpi_rank == 0) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) A[i * n + j] = 0.0;
        }
    }

#ifdef USE_CUDA
    if (dA) cudaFree(dA);
#endif

    // Broadcast final result from rank 0 to others so validation/printing sees same data
#ifdef USE_MPI
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    return true;
}

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
    
    // Compute A = B * B^T
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

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
