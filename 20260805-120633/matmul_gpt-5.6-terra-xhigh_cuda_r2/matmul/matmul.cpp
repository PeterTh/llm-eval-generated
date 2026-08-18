#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "../common/results_output.hpp"

namespace {

void cudaCheck(const cudaError_t status, const char* call,
               const char* file, const int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
                     file, line, call, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

void cublasCheck(const cublasStatus_t status, const char* call,
                 const char* file, const int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error at %s:%d while executing %s: status %d\n",
                     file, line, call, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)
#define CUBLAS_CHECK(call) cublasCheck((call), #call, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(const size_t elementCount) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), elementCount * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }

  private:
    T* data_ = nullptr;
};

class CublasHandle {
  public:
    CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }

    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const { return handle_; }

  private:
    cublasHandle_t handle_ = nullptr;
};

class CudaEvent {
  public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const { return event_; }

  private:
    cudaEvent_t event_ = nullptr;
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

float matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                     std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return 0.0F;
    }
    if (N > static_cast<size_t>(INT_MAX)) {
        std::fprintf(stderr, "Matrix dimension exceeds cuBLAS's supported range: %zu\n", N);
        std::exit(EXIT_FAILURE);
    }

    const size_t elementCount = N * N;
    const size_t byteCount = elementCount * sizeof(double);
    const int n = static_cast<int>(N);

    // Allocation and input transfers are deliberately outside the timed region:
    // the reported rate is the GPU matrix multiplication rate, as in the original
    // code whose timer enclosed only matrixMultiply.
    DeviceBuffer<double> deviceA(elementCount);
    DeviceBuffer<double> deviceB(elementCount);
    DeviceBuffer<double> deviceC(elementCount);
    CUDA_CHECK(cudaMemcpy(deviceA.get(), A.data(), byteCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB.get(), B.data(), byteCount, cudaMemcpyHostToDevice));

    CublasHandle handle;
    CudaEvent start;
    CudaEvent stop;
    const double alpha = 1.0;
    const double beta = 0.0;

    // cuBLAS lazily loads its selected kernel on the first invocation.  Warm it
    // up before starting the benchmark clock so startup work is not reported as
    // matrix-multiplication throughput.
    CUBLAS_CHECK(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n, &alpha,
                             deviceB.get(), n, deviceA.get(), n,
                             &beta, deviceC.get(), n));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start.get()));
    // The host matrices are row-major while cuBLAS is column-major.  Viewing the
    // same storage as transposed column-major matrices gives C = A * B by
    // multiplying B then A in cuBLAS order, without any transpose or extra copy.
    CUBLAS_CHECK(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n, &alpha,
                             deviceB.get(), n, deviceA.get(), n,
                             &beta, deviceC.get(), n));
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));

    float elapsedMs = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start.get(), stop.get()));
    CUDA_CHECK(cudaMemcpy(C.data(), deviceC.get(), byteCount, cudaMemcpyDeviceToHost));
    return elapsedMs;
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
    const float durationMs = matrixMultiply(A, B, C, N);
    
    printf("Computation time: %.3f ms\n", static_cast<double>(durationMs));
    
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
