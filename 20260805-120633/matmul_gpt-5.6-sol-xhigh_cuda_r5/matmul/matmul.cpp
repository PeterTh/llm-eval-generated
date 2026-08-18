#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

[[noreturn]] void cudaFailure(const cudaError_t status, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while evaluating %s: %s\n",
                 file, line, expression, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

[[noreturn]] void cublasFailure(const cublasStatus_t status, const char* expression,
                                const char* file, const int line) {
    std::fprintf(stderr, "cuBLAS error at %s:%d while evaluating %s: status %d\n",
                 file, line, expression, static_cast<int>(status));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t status_ = (expression);                                \
        if (status_ != cudaSuccess) {                                             \
            cudaFailure(status_, #expression, __FILE__, __LINE__);               \
        }                                                                        \
    } while (false)

#define CUBLAS_CHECK(expression)                                                 \
    do {                                                                         \
        const cublasStatus_t status_ = (expression);                              \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                   \
            cublasFailure(status_, #expression, __FILE__, __LINE__);             \
        }                                                                        \
    } while (false)

}  // namespace

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
    if (N == 0 || N > static_cast<size_t>(INT_MAX) ||
        N > std::numeric_limits<size_t>::max() / N ||
        N * N > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Matrix size is outside the supported CUDA range.\n");
        std::exit(EXIT_FAILURE);
    }

    const int dimension = static_cast<int>(N);
    const size_t bytes = N * N * sizeof(double);
    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cublasHandle_t handle = nullptr;

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceA), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceB), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceC), bytes));

    CUDA_CHECK(cudaMemcpyAsync(deviceA, A.data(), bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(deviceB, B.data(), bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    // Force the one-time cuBLAS module initialization to finish before the
    // benchmark event. This tiny operation avoids doing the full product twice.
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             1, 1, 1, &alpha, deviceB, 1, deviceA, 1,
                             &beta, deviceC, 1));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));

    // A row-major product C = A * B has the same memory layout as the
    // column-major product C^T = B^T * A^T. Swapping the operands therefore
    // lets cuBLAS operate directly on the original contiguous allocations.
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             dimension, dimension, dimension,
                             &alpha, deviceB, dimension, deviceA, dimension,
                             &beta, deviceC, dimension));

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaMemcpyAsync(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, stop));

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(deviceC));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceA));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return static_cast<double>(elapsedMilliseconds);
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
    const double durationMilliseconds = matrixMultiply(A, B, C, N);

    printf("Computation time: %.3f ms\n", durationMilliseconds);
    
    // Calculate GFLOPS
    const double dimension = static_cast<double>(N);
    const double gflops = (2.0 * dimension * dimension * dimension) /
                          (durationMilliseconds * 1.0e6);
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
