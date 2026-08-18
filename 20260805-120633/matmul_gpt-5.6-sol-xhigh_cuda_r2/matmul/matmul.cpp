#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

const char* cublasStatusString(cublasStatus_t status) noexcept {
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

[[noreturn]] void cudaFailure(cudaError_t status, const char* expression,
                              const char* file, int line) {
    std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line,
                 expression, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

[[noreturn]] void cublasFailure(cublasStatus_t status, const char* expression,
                                const char* file, int line) {
    std::fprintf(stderr, "cuBLAS error at %s:%d for %s: %s\n", file, line,
                 expression, cublasStatusString(status));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                \
    do {                                                                      \
        const cudaError_t status_ = (expression);                             \
        if (status_ != cudaSuccess) {                                         \
            cudaFailure(status_, #expression, __FILE__, __LINE__);            \
        }                                                                     \
    } while (false)

#define CUBLAS_CHECK(expression)                                              \
    do {                                                                      \
        const cublasStatus_t status_ = (expression);                          \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                               \
            cublasFailure(status_, #expression, __FILE__, __LINE__);          \
        }                                                                     \
    } while (false)

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t elements) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_),
                              elements * sizeof(double)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    double* get() noexcept { return data_; }

private:
    double* data_ = nullptr;
};

class CublasHandle {
public:
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }

    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const noexcept { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

}  // namespace

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Compute C = A * B on the GPU and return kernel execution time in ms.
float matrixMultiply(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::vector<double>& C, const size_t N) {
    const size_t elements = N * N;
    const size_t bytes = elements * sizeof(double);
    const int dimension = static_cast<int>(N);

    DeviceBuffer deviceA(elements);
    DeviceBuffer deviceB(elements);
    DeviceBuffer deviceC(elements);

    CUDA_CHECK(cudaMemcpy(deviceA.get(), A.data(), bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB.get(), B.data(), bytes,
                          cudaMemcpyHostToDevice));

    CublasHandle handle;
    const double alpha = 1.0;
    const double beta = 0.0;

    // cuBLAS matrices are column-major. Row-major C=A*B has the same memory
    // layout as column-major C^T=B^T*A^T, hence the intentionally swapped
    // deviceB/deviceA arguments below.
    auto launchGemm = [&]() {
        CUBLAS_CHECK(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                                 dimension, dimension, dimension, &alpha,
                                 deviceB.get(), dimension, deviceA.get(),
                                 dimension, &beta, deviceC.get(), dimension));
    };

    // Initialize lazy CUDA/cuBLAS state before measuring the multiplication.
    launchGemm();
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaEvent start;
    CudaEvent stop;
    CUDA_CHECK(cudaEventRecord(start.get()));
    launchGemm();
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));

    float elapsedMs = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start.get(), stop.get()));
    CUDA_CHECK(cudaMemcpy(C.data(), deviceC.get(), bytes,
                          cudaMemcpyDeviceToHost));
    return elapsedMs;
}

// Simple validation: compute selected elements on the CPU and compare.
bool validateResult(const std::vector<double>& A,
                    const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
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
            const double relError =
                std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, "
                            "got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseDimension(const char* text, size_t& value) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        return false;
    }

    const size_t dimension = static_cast<size_t>(parsed);
    if (dimension > std::numeric_limits<size_t>::max() / dimension ||
        dimension * dimension >
            std::numeric_limits<size_t>::max() / sizeof(double)) {
        return false;
    }

    value = dimension;
    return true;
}

int main(int argc, char** argv) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseDimension(argv[++i], N)) {
                std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    std::printf("Matrix Multiplication Benchmark\n");
    std::printf("Matrix size: %zu x %zu\n", N, N);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t elements = N * N;
    std::vector<double> A(elements);
    std::vector<double> B(elements);
    std::vector<double> C(elements);

    std::printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    std::printf("Computing matrix multiplication...\n");
    const float durationMs = matrixMultiply(A, B, C, N);
    std::printf("Computation time: %.3f ms\n", durationMs);

    const double operations = 2.0 * static_cast<double>(N) *
                              static_cast<double>(N) * static_cast<double>(N);
    const double gflops = operations / (static_cast<double>(durationMs) * 1e6);
    std::printf("Performance: %.3f GFLOPS\n", gflops);

    if (printResults) {
        print_results(C, "MatrixC");
    }

    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(A, B, C, N)) {
            std::printf("Validation: PASSED\n");
            return 0;
        }

        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
