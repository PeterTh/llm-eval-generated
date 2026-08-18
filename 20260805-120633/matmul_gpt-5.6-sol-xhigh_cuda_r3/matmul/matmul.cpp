#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

namespace {

[[noreturn]] void throwCudaError(const cudaError_t error, const char* operation) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

void checkCuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throwCudaError(error, operation);
    }
}

const char* cublasStatusString(const cublasStatus_t status) noexcept {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "success";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED: return "allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "architecture mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR: return "mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "internal error";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "not supported";
        case CUBLAS_STATUS_LICENSE_ERROR: return "license error";
        default: return "unknown error";
    }
}

void checkCublas(const cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + cublasStatusString(status));
    }
}

class DeviceBuffer {
  public:
    explicit DeviceBuffer(const size_t bytes) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), bytes), "cudaMalloc");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    double* get() const noexcept { return data_; }

  private:
    double* data_ = nullptr;
};

class CublasHandle {
  public:
    CublasHandle() {
        checkCublas(cublasCreate(&handle_), "cublasCreate");
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    cublasHandle_t get() const noexcept { return handle_; }

  private:
    cublasHandle_t handle_ = nullptr;
};

class CudaEvent {
  public:
    CudaEvent() {
        checkCuda(cudaEventCreate(&event_), "cudaEventCreate");
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    cudaEvent_t get() const noexcept { return event_; }

  private:
    cudaEvent_t event_ = nullptr;
};

} // namespace

// Compute C = A * B in double precision on the GPU. cuBLAS stores matrices in
// column-major order, so reversing A and B produces the row-major result:
// (A * B)^T = B^T * A^T.
double matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                      std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return 0.0;
    }
    if (N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("matrix dimension exceeds the cuBLAS limit");
    }

    const size_t bytes = A.size() * sizeof(double);
    DeviceBuffer deviceA(bytes);
    DeviceBuffer deviceB(bytes);
    DeviceBuffer deviceC(bytes);
    CublasHandle handle;
    CudaEvent start;
    CudaEvent stop;

    checkCuda(cudaMemcpy(deviceA.get(), A.data(), bytes, cudaMemcpyHostToDevice),
              "copying matrix A to the GPU");
    checkCuda(cudaMemcpy(deviceB.get(), B.data(), bytes, cudaMemcpyHostToDevice),
              "copying matrix B to the GPU");

    const int n = static_cast<int>(N);
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    // cuBLAS loads and selects its GEMM kernels lazily. Warm up the exact shape
    // once so one-time module loading and clock ramp-up are not counted as
    // steady-state multiplication time.
    checkCublas(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                            n, n, n, &alpha,
                            deviceB.get(), n, deviceA.get(), n,
                            &beta, deviceC.get(), n),
                "warming up cublasDgemm");
    checkCuda(cudaDeviceSynchronize(), "waiting for cuBLAS warm-up");

    // Creating the handle and moving the inputs before the event ensures the
    // benchmark measures the same operation as the original CPU timer: GEMM.
    checkCuda(cudaEventRecord(start.get()), "recording the start event");
    checkCublas(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                            n, n, n, &alpha,
                            deviceB.get(), n, deviceA.get(), n,
                            &beta, deviceC.get(), n),
                "cublasDgemm");
    checkCuda(cudaEventRecord(stop.get()), "recording the stop event");
    checkCuda(cudaEventSynchronize(stop.get()), "waiting for matrix multiplication");

    float elapsedMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, start.get(), stop.get()),
              "measuring matrix multiplication");
    checkCuda(cudaMemcpy(C.data(), deviceC.get(), bytes, cudaMemcpyDeviceToHost),
              "copying matrix C from the GPU");

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
    double elapsedMilliseconds = 0.0;
    try {
        elapsedMilliseconds = matrixMultiply(A, B, C, N);
    } catch (const std::exception& error) {
        fprintf(stderr, "CUDA matrix multiplication failed: %s\n", error.what());
        return 1;
    }

    printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    
    // Calculate GFLOPS
    const double gflops = elapsedMilliseconds > 0.0
        ? (2.0 * static_cast<double>(N) * static_cast<double>(N) *
           static_cast<double>(N)) / (elapsedMilliseconds * 1.0e6)
        : 0.0;
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
