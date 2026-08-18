#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

[[noreturn]] void cudaFailure(const char* expression, const cudaError_t status,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while evaluating %s: %s\n",
                 file, line, expression, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

void checkCuda(const cudaError_t status, const char* expression,
               const char* file, const int line) {
    if (status != cudaSuccess) {
        cudaFailure(expression, status, file, line);
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

[[noreturn]] void cublasFailure(const char* expression, const cublasStatus_t status,
                                const char* file, const int line) {
    std::fprintf(stderr, "cuBLAS error at %s:%d while evaluating %s: %s (%d)\n",
                 file, line, expression, cublasStatusString(status),
                 static_cast<int>(status));
    std::exit(EXIT_FAILURE);
}

void checkCublas(const cublasStatus_t status, const char* expression,
                 const char* file, const int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasFailure(expression, status, file, line);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)
#define CUBLAS_CHECK(expression) checkCublas((expression), #expression, __FILE__, __LINE__)

struct DeviceWork {
    int device = 0;
    size_t firstRow = 0;
    size_t rows = 0;
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
};

// Owns one row slab per GPU.  A row-major MxN matrix is the same memory image
// as its NxM transpose in column-major order, so each slab is evaluated as
// C^T = B^T A^T, which lets cuBLAS operate without costly transposes.
class CudaMatrixMultiplier {
public:
    CudaMatrixMultiplier(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C, const size_t N)
        : hostA_(A.data()), hostB_(B.data()), hostC_(C.data()), N_(N),
          bytes_(N * N * sizeof(double)) {
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            std::fprintf(stderr, "CUDA error: no CUDA-capable GPU is visible\n");
            std::exit(EXIT_FAILURE);
        }

        const size_t workerCount = std::min(N_, static_cast<size_t>(deviceCount));
        workers_.resize(workerCount);

        // Pin the existing vector storage so transfers from all GPUs can overlap.
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaHostRegister(const_cast<double*>(hostA_), bytes_,
                                    cudaHostRegisterPortable));
        aRegistered_ = true;
        CUDA_CHECK(cudaHostRegister(const_cast<double*>(hostB_), bytes_,
                                    cudaHostRegisterPortable));
        bRegistered_ = true;
        CUDA_CHECK(cudaHostRegister(hostC_, bytes_, cudaHostRegisterPortable));
        cRegistered_ = true;

        const size_t rowsPerWorker = N_ / workerCount;
        const size_t extraRows = N_ % workerCount;
        size_t firstRow = 0;

        for (size_t index = 0; index < workerCount; ++index) {
            DeviceWork& work = workers_[index];
            work.device = static_cast<int>(index);
            work.firstRow = firstRow;
            work.rows = rowsPerWorker + (index < extraRows ? 1 : 0);
            firstRow += work.rows;

            CUDA_CHECK(cudaSetDevice(work.device));
            CUDA_CHECK(cudaStreamCreateWithFlags(&work.stream, cudaStreamNonBlocking));
            CUBLAS_CHECK(cublasCreate(&work.handle));
            CUBLAS_CHECK(cublasSetStream(work.handle, work.stream));
            CUBLAS_CHECK(cublasSetAtomicsMode(work.handle, CUBLAS_ATOMICS_ALLOWED));

            const size_t slabBytes = work.rows * N_ * sizeof(double);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&work.dA), slabBytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&work.dB), bytes_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&work.dC), slabBytes));

            CUDA_CHECK(cudaMemcpyAsync(work.dA, hostA_ + work.firstRow * N_,
                                       slabBytes, cudaMemcpyHostToDevice, work.stream));
            CUDA_CHECK(cudaMemcpyAsync(work.dB, hostB_, bytes_,
                                       cudaMemcpyHostToDevice, work.stream));
        }

        synchronizeAll();

        // Force context, module, and GEMM-kernel initialization out of the
        // measured region.  Repeating beta=0 GEMM does not alter the result.
        launchAll();
        synchronizeAll();
    }

    CudaMatrixMultiplier(const CudaMatrixMultiplier&) = delete;
    CudaMatrixMultiplier& operator=(const CudaMatrixMultiplier&) = delete;

    ~CudaMatrixMultiplier() {
        for (DeviceWork& work : workers_) {
            cudaSetDevice(work.device);
            if (work.handle != nullptr) cublasDestroy(work.handle);
            if (work.dA != nullptr) cudaFree(work.dA);
            if (work.dB != nullptr) cudaFree(work.dB);
            if (work.dC != nullptr) cudaFree(work.dC);
            if (work.stream != nullptr) cudaStreamDestroy(work.stream);
        }

        if (!workers_.empty()) cudaSetDevice(workers_.front().device);
        if (cRegistered_) cudaHostUnregister(hostC_);
        if (bRegistered_) cudaHostUnregister(const_cast<double*>(hostB_));
        if (aRegistered_) cudaHostUnregister(const_cast<double*>(hostA_));
    }

    size_t deviceCount() const noexcept { return workers_.size(); }

    double multiply() {
        const auto start = std::chrono::steady_clock::now();
        launchAll();
        synchronizeAll();
        const auto end = std::chrono::steady_clock::now();

        // Keep result copies outside the GEMM timer, just as initialization and
        // allocation were outside the CPU benchmark's multiplication timer.
        for (DeviceWork& work : workers_) {
            CUDA_CHECK(cudaSetDevice(work.device));
            const size_t slabBytes = work.rows * N_ * sizeof(double);
            CUDA_CHECK(cudaMemcpyAsync(hostC_ + work.firstRow * N_, work.dC,
                                       slabBytes, cudaMemcpyDeviceToHost, work.stream));
        }
        synchronizeAll();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

private:
    void launchAll() {
        constexpr double alpha = 1.0;
        constexpr double beta = 0.0;
        const int n = static_cast<int>(N_);

        for (DeviceWork& work : workers_) {
            CUDA_CHECK(cudaSetDevice(work.device));
            CUBLAS_CHECK(cublasDgemm(work.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, static_cast<int>(work.rows), n,
                                     &alpha, work.dB, n, work.dA, n,
                                     &beta, work.dC, n));
        }
    }

    void synchronizeAll() {
        for (DeviceWork& work : workers_) {
            CUDA_CHECK(cudaSetDevice(work.device));
            CUDA_CHECK(cudaStreamSynchronize(work.stream));
        }
    }

    const double* hostA_;
    const double* hostB_;
    double* hostC_;
    size_t N_;
    size_t bytes_;
    std::vector<DeviceWork> workers_;
    bool aRegistered_ = false;
    bool bRegistered_ = false;
    bool cRegistered_ = false;
};

bool parseMatrixSize(const char* text, size_t& value) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') return false;

    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

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

// Simple validation: compute selected elements on the CPU and compare.
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    // Include both ends and the middle so multi-GPU row slabs are all sampled.
    const size_t checkPoints[] = {
        0,
        std::min<size_t>(1, N - 1),
        N / 2,
        N > 1 ? N - 2 : 0,
        N - 1,
    };

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi];
            const size_t j = checkPoints[pj];

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

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

int main(int argc, char** argv) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseMatrixSize(argv[++i], N)) {
                std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return EXIT_FAILURE;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > std::numeric_limits<size_t>::max() / N ||
        N * N > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Matrix size is too large for CUDA DGEMM\n");
        return EXIT_FAILURE;
    }

    std::printf("Matrix Multiplication Benchmark\n");
    std::printf("Matrix size: %zu x %zu\n", N, N);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    std::printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    std::printf("Preparing CUDA matrices...\n");
    CudaMatrixMultiplier multiplier(A, B, C, N);
    std::printf("CUDA devices: %zu\n", multiplier.deviceCount());

    std::printf("Computing matrix multiplication...\n");
    const double durationMs = multiplier.multiply();

    std::printf("Computation time: %.3f ms\n", durationMs);
    const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                           static_cast<double>(N)) / (durationMs / 1000.0) / 1e9;
    std::printf("Performance: %.3f GFLOPS\n", gflops);

    if (printResults) {
        print_results(C, "MatrixC");
    }

    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(A, B, C, N)) {
            std::printf("Validation: PASSED\n");
            return EXIT_SUCCESS;
        }

        std::printf("Validation: FAILED\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
