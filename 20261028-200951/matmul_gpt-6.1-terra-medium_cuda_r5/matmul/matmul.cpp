#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <cstdint>

#include <cuda_runtime.h>
#include <cublas_v2.h>

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

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

static void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuBLAS error %d\n", operation, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

class GpuMultiply {
public:
    GpuMultiply(const std::vector<double>& A, const std::vector<double>& B, size_t N)
        : n_(static_cast<int>(N)), bytes_(A.size() * sizeof(double)) {
        if (N == 0 || N > INT_MAX) {
            fprintf(stderr, "Matrix dimension must be between 1 and %d\n", INT_MAX);
            std::exit(EXIT_FAILURE);
        }
        checkCuda(cudaMalloc(&dA_, bytes_), "Allocating A on GPU");
        checkCuda(cudaMalloc(&dB_, bytes_), "Allocating B on GPU");
        checkCuda(cudaMalloc(&dC_, bytes_), "Allocating C on GPU");
        checkCuda(cudaMemcpy(dA_, A.data(), bytes_, cudaMemcpyHostToDevice), "Copying A to GPU");
        checkCuda(cudaMemcpy(dB_, B.data(), bytes_, cudaMemcpyHostToDevice), "Copying B to GPU");
        checkCublas(cublasCreate(&handle_), "Creating cuBLAS handle");
    }
    void multiply() {
        // Row-major C = A * B is column-major C^T = B^T * A^T.
        const double alpha = 1.0, beta = 0.0;
        checkCublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n_, n_, n_,
                                &alpha, dB_, n_, dA_, n_, &beta, dC_, n_), "Multiplying matrices");
        checkCuda(cudaDeviceSynchronize(), "Synchronizing matrix multiplication");
    }
    void copyResult(std::vector<double>& C) {
        checkCuda(cudaMemcpy(C.data(), dC_, bytes_, cudaMemcpyDeviceToHost), "Copying C from GPU");
    }
    ~GpuMultiply() {
        cublasDestroy(handle_);
        cudaFree(dA_);
        cudaFree(dB_);
        cudaFree(dC_);
    }
private:
    int n_;
    size_t bytes_;
    double *dA_ = nullptr, *dB_ = nullptr, *dC_ = nullptr;
    cublasHandle_t handle_{};
};

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
    
    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    if (N == 0 || N > INT_MAX || N > SIZE_MAX / N / sizeof(double)) {
        fprintf(stderr, "Invalid matrix size\n");
        return 1;
    }

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
    GpuMultiply gpu(A, B, N);
    gpu.multiply(); // Initialize the cuBLAS kernel before timing.
    auto start = std::chrono::high_resolution_clock::now();
    
    gpu.multiply();
    
    auto end = std::chrono::high_resolution_clock::now();
    gpu.copyResult(C);
    const double duration = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", duration);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration / 1000.0) / 1e9;
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
