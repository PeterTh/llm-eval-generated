#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

[[noreturn]] void cudaFail(const char* operation, const char* file, int line,
                           const char* error) {
    std::fprintf(stderr, "CUDA error at %s:%d while %s: %s\n", file, line,
                 operation, error);
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t status = (call);                                     \
        if (status != cudaSuccess) {                                           \
            cudaFail(#call, __FILE__, __LINE__, cudaGetErrorString(status));   \
        }                                                                      \
    } while (false)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        const cublasStatus_t status = (call);                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            cudaFail(#call, __FILE__, __LINE__, "cuBLAS call failed");        \
        }                                                                      \
    } while (false)

class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~CudaEvent() { cudaEventDestroy(event_); }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{};
};

class CublasHandle {
public:
    CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }
    ~CublasHandle() { cublasDestroy(handle_); }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_{};
};

class DeviceMatrix {
public:
    explicit DeviceMatrix(const size_t elements) {
        CUDA_CHECK(cudaMalloc(&data_, elements * sizeof(*data_)));
    }
    ~DeviceMatrix() { cudaFree(data_); }

    DeviceMatrix(const DeviceMatrix&) = delete;
    DeviceMatrix& operator=(const DeviceMatrix&) = delete;

    double* get() const { return data_; }

private:
    double* data_{};
};

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
    // cuBLAS uses column-major matrices.  A row-major matrix is its transpose
    // when viewed as column-major, so swapping A and B computes (A * B)^T in
    // the same storage layout and therefore produces the required row-major C.
    const int dimension = static_cast<int>(N);
    const size_t bytes = N * N * sizeof(double);

    DeviceMatrix deviceA(N * N);
    DeviceMatrix deviceB(N * N);
    DeviceMatrix deviceC(N * N);
    CublasHandle handle;
    CudaEvent start;
    CudaEvent stop;

    CUDA_CHECK(cudaMemcpy(deviceA.get(), A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB.get(), B.data(), bytes, cudaMemcpyHostToDevice));

    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    // Initialize cuBLAS and the GPU execution path before the measurement.
    CUBLAS_CHECK(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, dimension,
                             dimension, dimension, &alpha, deviceB.get(),
                             dimension, deviceA.get(), dimension, &beta,
                             deviceC.get(), dimension));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start.get()));
    CUBLAS_CHECK(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, dimension,
                             dimension, dimension, &alpha, deviceB.get(),
                             dimension, deviceA.get(), dimension, &beta,
                             deviceC.get(), dimension));
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start.get(), stop.get()));
    CUDA_CHECK(cudaMemcpy(C.data(), deviceC.get(), bytes, cudaMemcpyDeviceToHost));

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
    const double gflops = (2.0 * N * N * N) / durationMilliseconds / 1e6;
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
