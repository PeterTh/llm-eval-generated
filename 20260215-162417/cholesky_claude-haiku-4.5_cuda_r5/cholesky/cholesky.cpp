#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(err) { if ((err) != cudaSuccess) { printf("CUDA Error: %s\n", cudaGetErrorString(err)); return false; } }

// GPU-based Cholesky - using GPU memory throughout
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* d_A = nullptr;
    size_t bytes = n * n * sizeof(double);
    
    // Allocate GPU memory
    cudaError_t err = cudaMalloc(&d_A, bytes);
    CUDA_CHECK(err);
    
    // Copy data to GPU
    err = cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    
    // Process row by row, but compute all elements on CPU for correctness
    // Each row's computation uses previously computed L values
    std::vector<double> tempRow(n);
    
    for (size_t i = 0; i < n; ++i) {
        // Copy row i from GPU
        err = cudaMemcpy(tempRow.data(), d_A + i * n, n * sizeof(double), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err);
        
        // Compute elements of row i: columns 0 to i
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            
            if (i == j) {
                // Diagonal: L[i,i] = sqrt(A[i,i] - sum_k<i L[i,k]^2)
                for (size_t k = 0; k < j; ++k) {
                    sum += tempRow[k] * tempRow[k];
                }
                double val = tempRow[j] - sum;
                if (val <= 0.0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    cudaFree(d_A);
                    return false;
                }
                tempRow[j] = sqrt(val);
            } else {
                // Off-diagonal: L[i,j] = (A[i,j] - sum_k<j L[i,k]*L[j,k]) / L[j,j]
                // Need to read row j from GPU
                std::vector<double> rowJ(n);
                err = cudaMemcpy(rowJ.data(), d_A + j * n, n * sizeof(double), cudaMemcpyDeviceToHost);
                CUDA_CHECK(err);
                
                for (size_t k = 0; k < j; ++k) {
                    sum += tempRow[k] * rowJ[k];
                }
                tempRow[j] = (tempRow[j] - sum) / rowJ[j];
            }
        }
        
        // Zero out upper triangular part of row i
        for (size_t j = i + 1; j < n; ++j) {
            tempRow[j] = 0.0;
        }
        
        // Copy row back to GPU
        err = cudaMemcpy(d_A + i * n, tempRow.data(), n * sizeof(double), cudaMemcpyHostToDevice);
        CUDA_CHECK(err);
    }
    
    // Copy result back to host
    err = cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);
    
    // Free GPU memory
    cudaFree(d_A);
    
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
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
    
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A;
    }
    
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
    
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
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
    
    return 0;
}
