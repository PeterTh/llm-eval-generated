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

class GpuMatmul {
public:
    explicit GpuMatmul(size_t n) : n_(static_cast<int>(n)), bytes_(n * n * sizeof(double)) {
        checkCublas(cublasCreate(&handle_), "cublasCreate");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&a_), bytes_), "cudaMalloc A");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&b_), bytes_), "cudaMalloc B");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&c_), bytes_), "cudaMalloc C");
    }

    ~GpuMatmul() {
        cudaFree(c_);
        cudaFree(b_);
        cudaFree(a_);
        cublasDestroy(handle_);
    }

    GpuMatmul(const GpuMatmul&) = delete;
    GpuMatmul& operator=(const GpuMatmul&) = delete;

    void multiply(const std::vector<double>& A, const std::vector<double>& B,
                  std::vector<double>& C) {
        checkCuda(cudaMemcpy(a_, A.data(), bytes_, cudaMemcpyHostToDevice), "copy A to GPU");
        checkCuda(cudaMemcpy(b_, B.data(), bytes_, cudaMemcpyHostToDevice), "copy B to GPU");
        const double alpha = 1.0;
        const double beta = 0.0;
        // cuBLAS uses column-major storage: C^T = B^T * A^T.
        checkCublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n_, n_, n_,
                                &alpha, b_, n_, a_, n_, &beta, c_, n_), "cublasDgemm");
        // This copy also waits for the multiplication to finish.
        checkCuda(cudaMemcpy(C.data(), c_, bytes_, cudaMemcpyDeviceToHost), "copy C from GPU");
    }

private:
    int n_;
    size_t bytes_;
    cublasHandle_t handle_{};
    double *a_{}, *b_{}, *c_{};
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
            char* end = nullptr;
            const char* value = argv[++i];
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (value[0] == '-' || end == value || *end != '\0' || parsed == 0 ||
                parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
                parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed) {
                fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            N = static_cast<size_t>(parsed);
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
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    // As with the host matrices, allocate GPU resources outside the timed region.
    GpuMatmul gpu(N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    gpu.multiply(A, B, C);
    
    auto end = std::chrono::high_resolution_clock::now();
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
