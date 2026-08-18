#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/results_output.hpp"

namespace {

[[noreturn]] void throwCudaError(cudaError_t status, const char* operation) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throwCudaError(status, operation);
    }
}

void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " failed (cuBLAS status " +
                                 std::to_string(static_cast<int>(status)) + ")");
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) {
        if (bytes != 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), bytes), "cudaMalloc");
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    double* get() const noexcept { return data_; }

private:
    double* data_ = nullptr;
};

class CublasHandle {
public:
    CublasHandle() { checkCublas(cublasCreate(&handle_), "cublasCreate"); }
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

// cuBLAS stores matrices column-major. Row-major A and B are therefore seen as
// A^T and B^T, so computing B^T * A^T produces the row-major bytes for A * B.
double matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                      std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return 0.0;
    }

    const size_t bytes = A.size() * sizeof(double);
    DeviceBuffer deviceA(bytes);
    DeviceBuffer deviceB(bytes);
    DeviceBuffer deviceC(bytes);
    CublasHandle handle;

    checkCuda(cudaMemcpy(deviceA.get(), A.data(), bytes, cudaMemcpyHostToDevice),
              "copying A to the GPU");
    checkCuda(cudaMemcpy(deviceB.get(), B.data(), bytes, cudaMemcpyHostToDevice),
              "copying B to the GPU");

    const double alpha = 1.0;
    const double beta = 0.0;
    // Force cuBLAS's lazy kernel/module setup before timing. The 1x1 operation
    // has negligible cost and the real GEMM below overwrites its output.
    checkCublas(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1,
                            &alpha, deviceB.get(), 1, deviceA.get(), 1, &beta,
                            deviceC.get(), 1),
                "warming up cublasDgemm");
    checkCuda(cudaDeviceSynchronize(), "warming up the GPU");

    // Host transfers and one-time setup are intentionally outside the measured
    // multiplication time, matching the original benchmark's compute-only timer.
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    try {
        checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
        checkCuda(cudaEventRecord(start), "cudaEventRecord(start)");

        const int dimension = static_cast<int>(N);
        checkCublas(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                                dimension, dimension, dimension, &alpha,
                                deviceB.get(), dimension, deviceA.get(), dimension,
                                &beta, deviceC.get(), dimension),
                    "cublasDgemm");

        checkCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
        checkCuda(cudaEventSynchronize(stop), "waiting for matrix multiplication");

        float elapsedMilliseconds = 0.0F;
        checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, start, stop),
                  "cudaEventElapsedTime");
        checkCuda(cudaMemcpy(C.data(), deviceC.get(), bytes, cudaMemcpyDeviceToHost),
                  "copying C from the GPU");

        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return static_cast<double>(elapsedMilliseconds);
    } catch (...) {
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        cudaEventDestroy(start);
        throw;
    }
}

// Simple validation: compute selected elements and compare.
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return true;
    }
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
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f "
                            "(error: %.10e)\n",
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

}  // namespace

int main(int argc, char** argv) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' || parsed > INT_MAX) {
                std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            N = static_cast<size_t>(parsed);
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

    if (N != 0 && N > std::numeric_limits<size_t>::max() / N) {
        std::fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }

    std::printf("Matrix Multiplication Benchmark\n");
    std::printf("Matrix size: %zu x %zu\n", N, N);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    try {
        std::vector<double> A(N * N);
        std::vector<double> B(N * N);
        std::vector<double> C(N * N);

        std::printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);

        std::printf("Computing matrix multiplication...\n");
        const double elapsedMilliseconds = matrixMultiply(A, B, C, N);

        std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);
        const double seconds = elapsedMilliseconds / 1000.0;
        const double gflops = seconds > 0.0
                                  ? (2.0 * static_cast<double>(N) * N * N) /
                                        seconds / 1.0e9
                                  : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(A, B, C, N)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                return 1;
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }

    return 0;
}
