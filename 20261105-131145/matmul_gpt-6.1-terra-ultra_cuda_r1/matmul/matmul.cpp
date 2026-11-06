#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuBLAS error %d\n", operation, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

class GpuMatrixMultiply {
public:
    GpuMatrixMultiply(const std::vector<double>& A, const std::vector<double>& B, size_t N)
        : n_(static_cast<int>(N)), bytes_(A.size() * sizeof(double)) {
        checkCublas(cublasCreate(&handle_), "Creating cuBLAS handle");
        if (N == 0) return;
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&a_), bytes_), "Allocating GPU matrix A");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&b_), bytes_), "Allocating GPU matrix B");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&c_), bytes_), "Allocating GPU matrix C");
        checkCuda(cudaMemcpy(a_, A.data(), bytes_, cudaMemcpyHostToDevice), "Copying matrix A to GPU");
        checkCuda(cudaMemcpy(b_, B.data(), bytes_, cudaMemcpyHostToDevice), "Copying matrix B to GPU");
        // Initialize cuBLAS's lazy setup before timing the actual multiplication.
        const double alpha = 1.0, beta = 0.0;
        checkCublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1,
                                &alpha, b_, 1, a_, 1, &beta, c_, 1),
                    "Initializing GPU multiplication");
        checkCuda(cudaDeviceSynchronize(), "Synchronizing GPU initialization");
    }

    ~GpuMatrixMultiply() {
        cudaFree(a_);
        cudaFree(b_);
        cudaFree(c_);
        cublasDestroy(handle_);
    }

    void multiply() {
        if (n_ == 0) return;
        const double alpha = 1.0, beta = 0.0;
        // cuBLAS uses column-major storage: C^T = B^T * A^T.
        checkCublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n_, n_, n_,
                                &alpha, b_, n_, a_, n_, &beta, c_, n_),
                    "Multiplying matrices on GPU");
        checkCuda(cudaDeviceSynchronize(), "Synchronizing GPU multiplication");
    }

    void copyResult(std::vector<double>& C) const {
        if (n_ != 0) {
            checkCuda(cudaMemcpy(C.data(), c_, bytes_, cudaMemcpyDeviceToHost),
                      "Copying result from GPU");
        }
    }

    GpuMatrixMultiply(const GpuMatrixMultiply&) = delete;
    GpuMatrixMultiply& operator=(const GpuMatrixMultiply&) = delete;

private:
    int n_;
    size_t bytes_;
    cublasHandle_t handle_{};
    double *a_ = nullptr, *b_ = nullptr, *c_ = nullptr;
};

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    if (N == 0) return true;
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
    
    if (N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (N != 0 && N > std::numeric_limits<size_t>::max() / N / sizeof(double))) {
        fprintf(stderr, "Matrix size is too large for CUDA matrix multiplication\n");
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

    GpuMatrixMultiply gpu(A, B, N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    gpu.multiply();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    gpu.copyResult(C);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    const double seconds = std::chrono::duration<double>(end - start).count();
    double gflops = (2.0 * N * N * N) / seconds / 1e9;
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
