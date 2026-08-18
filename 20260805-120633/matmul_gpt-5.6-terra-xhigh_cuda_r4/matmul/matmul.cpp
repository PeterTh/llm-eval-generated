#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

[[noreturn]] void cudaFailure(const char* operation, const cudaError_t status) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                 cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

[[noreturn]] void cublasFailure(const char* operation, const cublasStatus_t status) {
    std::fprintf(stderr, "cuBLAS error during %s (status %d)\n", operation,
                 static_cast<int>(status));
    std::exit(EXIT_FAILURE);
}

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        cudaFailure(operation, status);
    }
}

void checkCublas(const cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasFailure(operation, status);
    }
}

// The host matrices are row-major.  cuBLAS treats their storage as the
// corresponding column-major transposes, so B * A produces the row-major
// result A * B in C without a transpose or extra device buffer.
class DeviceMatrices {
public:
    DeviceMatrices(const std::vector<double>& A, const std::vector<double>& B,
                   const size_t N)
        : N_(N), elementCount_(N * N), byteCount_(elementCount_ * sizeof(double)) {
        if (N_ > static_cast<size_t>(INT_MAX)) {
            std::fprintf(stderr, "Matrix size is too large for cuBLAS: %zu\n", N_);
            std::exit(EXIT_FAILURE);
        }

        int deviceCount = 0;
        checkCuda(cudaGetDeviceCount(&deviceCount), "querying CUDA devices");
        if (deviceCount == 0) {
            std::fprintf(stderr, "No CUDA-capable device is available.\n");
            std::exit(EXIT_FAILURE);
        }

        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                  "creating CUDA stream");
        checkCublas(cublasCreate(&handle_), "creating cuBLAS handle");
        checkCublas(cublasSetStream(handle_, stream_), "setting cuBLAS stream");
        checkCuda(cudaEventCreate(&start_), "creating start event");
        checkCuda(cudaEventCreate(&stop_), "creating stop event");

        checkCuda(cudaMalloc(reinterpret_cast<void**>(&A_), byteCount_), "allocating matrix A");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&B_), byteCount_), "allocating matrix B");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&C_), byteCount_), "allocating matrix C");

        checkCuda(cudaMemcpyAsync(A_, A.data(), byteCount_, cudaMemcpyHostToDevice, stream_),
                  "copying matrix A to the GPU");
        checkCuda(cudaMemcpyAsync(B_, B.data(), byteCount_, cudaMemcpyHostToDevice, stream_),
                  "copying matrix B to the GPU");
        checkCuda(cudaStreamSynchronize(stream_), "initializing GPU matrices");
    }

    DeviceMatrices(const DeviceMatrices&) = delete;
    DeviceMatrices& operator=(const DeviceMatrices&) = delete;

    ~DeviceMatrices() {
        if (C_ != nullptr) {
            cudaFree(C_);
        }
        if (B_ != nullptr) {
            cudaFree(B_);
        }
        if (A_ != nullptr) {
            cudaFree(A_);
        }
        if (stop_ != nullptr) {
            cudaEventDestroy(stop_);
        }
        if (start_ != nullptr) {
            cudaEventDestroy(start_);
        }
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    // Warm up cuBLAS dispatch and GPU clocks.  Its output is deliberately
    // overwritten by the following, timed multiplication.
    void warmUp() {
        constexpr double alpha = 1.0;
        constexpr double beta = 0.0;
        const int n = static_cast<int>(N_);

        checkCublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                n, n, n, &alpha, B_, n, A_, n, &beta, C_, n),
                    "warming up matrix multiplication");
        checkCuda(cudaStreamSynchronize(stream_), "warming up matrix multiplication");
    }

    // Returns GPU execution time only.  Host-to-device setup and the result
    // download are intentionally outside the timed multiplication, just as
    // allocation and initialization are outside the original CPU timing.
    float multiply() {
        constexpr double alpha = 1.0;
        constexpr double beta = 0.0;
        const int n = static_cast<int>(N_);

        checkCuda(cudaEventRecord(start_, stream_), "recording start event");
        checkCublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                n, n, n, &alpha, B_, n, A_, n, &beta, C_, n),
                    "performing matrix multiplication");
        checkCuda(cudaEventRecord(stop_, stream_), "recording stop event");
        checkCuda(cudaEventSynchronize(stop_), "waiting for matrix multiplication");

        float elapsedMilliseconds = 0.0F;
        checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, start_, stop_),
                  "measuring matrix multiplication");
        return elapsedMilliseconds;
    }

    void copyResult(std::vector<double>& C) const {
        checkCuda(cudaMemcpyAsync(C.data(), C_, byteCount_, cudaMemcpyDeviceToHost, stream_),
                  "copying matrix C from the GPU");
        checkCuda(cudaStreamSynchronize(stream_), "copying matrix C from the GPU");
    }

private:
    size_t N_ = 0;
    size_t elementCount_ = 0;
    size_t byteCount_ = 0;
    double* A_ = nullptr;
    double* B_ = nullptr;
    double* C_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t handle_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

} // namespace

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
    
    // Transfer the initialized row-major inputs once, before the timed GPU work.
    DeviceMatrices deviceMatrices(A, B, N);
    deviceMatrices.warmUp();

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    const float elapsedMilliseconds = deviceMatrices.multiply();
    const long reportedMilliseconds = static_cast<long>(elapsedMilliseconds);
    
    printf("Computation time: %ld ms\n", reportedMilliseconds);
    
    // Calculate GFLOPS
    const double timedMilliseconds = std::max(static_cast<double>(elapsedMilliseconds), 1.0e-6);
    double gflops = (2.0 * N * N * N) / timedMilliseconds / 1.0e6;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Keep C on the host for external result printing and CPU validation.
    deviceMatrices.copyResult(C);
    
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
