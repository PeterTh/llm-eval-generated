#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

double matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                      std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    cublasHandle_t handle = nullptr;

    auto checkCuda = [](cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
            std::exit(EXIT_FAILURE);
        }
    };
    auto checkCublas = [](cublasStatus_t status, const char* operation) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "%s failed (cuBLAS status %d)\n", operation, static_cast<int>(status));
            std::exit(EXIT_FAILURE);
        }
    };

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA), bytes), "cudaMalloc(A)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), bytes), "cudaMalloc(B)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC), bytes), "cudaMalloc(C)");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copy A to GPU");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copy B to GPU");
    checkCublas(cublasCreate(&handle), "cublasCreate");

    const double alpha = 1.0;
    const double beta = 0.0;
    const int n = static_cast<int>(N);
    // Row-major C = A * B is column-major C^T = B^T * A^T.
    // Trigger cuBLAS's first-use setup before measuring the multiplication.
    checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            1, 1, 1, &alpha, deviceB, 1, deviceA, 1,
                            &beta, deviceC, 1), "cuBLAS warmup");
    checkCuda(cudaDeviceSynchronize(), "CUDA warmup synchronization");
    const auto start = std::chrono::high_resolution_clock::now();
    checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            n, n, n, &alpha, deviceB, n, deviceA, n,
                            &beta, deviceC, n), "cublasDgemm");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    const auto end = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "copy C from GPU");

    checkCublas(cublasDestroy(handle), "cublasDestroy");
    checkCuda(cudaFree(deviceC), "cudaFree(C)");
    checkCuda(cudaFree(deviceB), "cudaFree(B)");
    checkCuda(cudaFree(deviceA), "cudaFree(A)");
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }
            
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            
            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
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

    if (N == 0 || N > static_cast<size_t>(INT_MAX) ||
        N > static_cast<size_t>(std::sqrt(static_cast<double>(SIZE_MAX / sizeof(double))))) {
        fprintf(stderr, "Matrix size must be between 1 and %d and fit in memory.\n", INT_MAX);
        return 1;
    }
    
    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    const double durationMs = matrixMultiply(A, B, C, N);

    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (durationMs / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(A, B, C, N);
        
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
