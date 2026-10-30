#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* d_A = nullptr;
    size_t bytes = n * n * sizeof(double);
    if (cudaMalloc(&d_A, bytes) != cudaSuccess) return false;
    if (cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(d_A); return false; }

    cusolverDnHandle_t handle;
    if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS) { cudaFree(d_A); return false; }
    int lwork = 0;
    cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, (int)n, d_A, (int)n, &lwork);
    double* d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(double));
    int* d_info = nullptr;
    cudaMalloc(&d_info, sizeof(int));

    cusolverStatus_t st = cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_UPPER, (int)n, d_A, (int)n, d_work, lwork, d_info);

    int info = 0;
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    if (st != CUSOLVER_STATUS_SUCCESS || info != 0) { cudaFree(d_work); cudaFree(d_info); cusolverDnDestroy(handle); cudaFree(d_A); return false; }

    if (cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { cudaFree(d_work); cudaFree(d_info); cusolverDnDestroy(handle); cudaFree(d_A); return false; }

    cudaFree(d_work);
    cudaFree(d_info);
    cusolverDnDestroy(handle);
    cudaFree(d_A);

    // Zero upper triangular part (row-major)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    return true;
}

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
