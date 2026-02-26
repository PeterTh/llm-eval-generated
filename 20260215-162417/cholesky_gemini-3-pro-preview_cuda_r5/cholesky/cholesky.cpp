#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <device_launch_parameters.h>

// CUDA Error checking
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__ )
void check(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCublasErrors(val) check_cublas( (val), #val, __FILE__, __LINE__ )
void check_cublas(cublasStatus_t result, char const *const func, const char *const file, int const line) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCusolverErrors(val) check_cusolver( (val), #val, __FILE__, __LINE__ )
void check_cusolver(cusolverStatus_t result, char const *const func, const char *const file, int const line) {
    if (result != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "CUSOLVER error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
        exit(EXIT_FAILURE);
    }
}

cusolverDnHandle_t g_cusolverH = NULL;

void init_cusolver() {
    if (g_cusolverH == NULL) {
        checkCusolverErrors(cusolverDnCreate(&g_cusolverH));
    }
}

void destroy_cusolver() {
    if (g_cusolverH != NULL) {
        checkCusolverErrors(cusolverDnDestroy(g_cusolverH));
        g_cusolverH = NULL;
    }
}

// Simple Cholesky decomposition using ONLY cuSOLVER
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
bool choleskyDecomposition(std::vector<double>& h_A, const size_t n) {
    size_t size = n * n * sizeof(double);
    double *d_A;
    int *d_info;

    checkCudaErrors(cudaMalloc((void**)&d_A, size));
    checkCudaErrors(cudaMalloc((void**)&d_info, sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));

    if (g_cusolverH == NULL) {
        checkCusolverErrors(cusolverDnCreate(&g_cusolverH));
    }
    
    // Workspace for potrf
    int lwork = 0;
    double *d_work = nullptr;
    
    // Use UPPER because Row Major Lower = Col Major Upper
    checkCusolverErrors(cusolverDnDpotrf_bufferSize(g_cusolverH, CUBLAS_FILL_MODE_UPPER, n, d_A, n, &lwork));
    checkCudaErrors(cudaMalloc((void**)&d_work, lwork * sizeof(double)));
    
    // Single call to potrf
    checkCusolverErrors(cusolverDnDpotrf(g_cusolverH, CUBLAS_FILL_MODE_UPPER, n, d_A, n, d_work, lwork, d_info));
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    int h_info = 0;
    checkCudaErrors(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Copy result back
    checkCudaErrors(cudaMemcpy(h_A.data(), d_A, size, cudaMemcpyDeviceToHost));
    
    // Zero out strictly upper triangular part
    bool success = (h_info == 0);
    if (success) {
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(h_A[i * n + i])) {
                success = false;
                break;
            }
            // Zero out upper triangular part
            for (size_t j = i + 1; j < n; ++j) {
                h_A[i * n + j] = 0.0;
            }
        }
    }

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_info));
    checkCudaErrors(cudaFree(d_work));
    return success;
}

// Optimized generation for large matrices: Diagonally dominant is sufficient for PD
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    unsigned int seed = 42;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double val = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
            A[i * n + j] = val;
            A[j * n + i] = val; // Symmetric
        }
        // Make diagonal dominant to ensure positive definiteness
        // A[i][i] += n
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
    
    // Warm-up CUDA context and libraries
    cudaFree(0);
    init_cusolver();

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
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    destroy_cusolver();
    return 0;
}
