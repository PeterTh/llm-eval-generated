#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + cublasStatusString(status));
    }
}

// Each GPU owns a disjoint group of output rows. B is replicated, while A and
// C are partitioned, so the GEMMs can execute independently and concurrently.
class CudaMatrixMultiplier {
public:
    CudaMatrixMultiplier(const std::vector<double>& A,
                         const std::vector<double>& B,
                         size_t N)
        : N_(N), dimension_(static_cast<int>(N)) {
        int availableDevices = 0;
        checkCuda(cudaGetDeviceCount(&availableDevices), "cudaGetDeviceCount");
        if (availableDevices == 0) {
            throw std::runtime_error("no CUDA-capable GPU is available");
        }

        // A slice of at least 256 rows gives cuBLAS enough work to amortize a
        // launch on current GPUs while still scaling larger matrices broadly.
        constexpr size_t targetRowsPerDevice = 256;
        const size_t usefulDevices = std::max<size_t>(1, N_ / targetRowsPerDevice);
        const size_t deviceCount = std::min<size_t>(availableDevices,
                                                    std::min(N_, usefulDevices));
        devices_.reserve(deviceCount);

        const size_t baseRows = N_ / deviceCount;
        const size_t remainder = N_ % deviceCount;
        size_t firstRow = 0;

        try {
            for (size_t index = 0; index < deviceCount; ++index) {
                DeviceContext context;
                context.device = static_cast<int>(index);
                context.firstRow = firstRow;
                context.rows = baseRows + (index < remainder ? 1 : 0);
                devices_.push_back(context);

                DeviceContext& device = devices_.back();
                checkCuda(cudaSetDevice(device.device), "cudaSetDevice");
                checkCuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking),
                          "cudaStreamCreateWithFlags");
                checkCublas(cublasCreate(&device.handle), "cublasCreate");
                checkCublas(cublasSetStream(device.handle, device.stream), "cublasSetStream");

                const size_t sliceElements = device.rows * N_;
                checkCuda(cudaMalloc(reinterpret_cast<void**>(&device.A),
                                     sliceElements * sizeof(double)),
                          "cudaMalloc(A slice)");
                checkCuda(cudaMalloc(reinterpret_cast<void**>(&device.B),
                                     N_ * N_ * sizeof(double)),
                          "cudaMalloc(B)");
                checkCuda(cudaMalloc(reinterpret_cast<void**>(&device.C),
                                     sliceElements * sizeof(double)),
                          "cudaMalloc(C slice)");

                checkCuda(cudaMemcpyAsync(device.A, A.data() + firstRow * N_,
                                          sliceElements * sizeof(double),
                                          cudaMemcpyHostToDevice, device.stream),
                          "cudaMemcpyAsync(A slice)");
                checkCuda(cudaMemcpyAsync(device.B, B.data(), N_ * N_ * sizeof(double),
                                          cudaMemcpyHostToDevice, device.stream),
                          "cudaMemcpyAsync(B)");

                firstRow += device.rows;
            }
            synchronize();

            // Warm up CUDA context/module loading so it is not charged to the
            // benchmark. The measured GEMM completely overwrites this result.
            launchGemm();
            synchronize();
        } catch (...) {
            release();
            throw;
        }
    }

    CudaMatrixMultiplier(const CudaMatrixMultiplier&) = delete;
    CudaMatrixMultiplier& operator=(const CudaMatrixMultiplier&) = delete;

    ~CudaMatrixMultiplier() {
        release();
    }

    double multiply() {
        const auto start = std::chrono::steady_clock::now();
        launchGemm();
        synchronize();
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    void copyResult(std::vector<double>& C) {
        for (DeviceContext& device : devices_) {
            checkCuda(cudaSetDevice(device.device), "cudaSetDevice");
            checkCuda(cudaMemcpyAsync(C.data() + device.firstRow * N_, device.C,
                                      device.rows * N_ * sizeof(double),
                                      cudaMemcpyDeviceToHost, device.stream),
                      "cudaMemcpyAsync(C slice)");
        }
        synchronize();
    }

    size_t deviceCount() const noexcept {
        return devices_.size();
    }

private:
    struct DeviceContext {
        int device = 0;
        size_t firstRow = 0;
        size_t rows = 0;
        cudaStream_t stream = nullptr;
        cublasHandle_t handle = nullptr;
        double* A = nullptr;
        double* B = nullptr;
        double* C = nullptr;
    };

    void launchGemm() {
        constexpr double alpha = 1.0;
        constexpr double beta = 0.0;
        for (DeviceContext& device : devices_) {
            checkCuda(cudaSetDevice(device.device), "cudaSetDevice");

            // cuBLAS uses column-major storage. A row-major C = A * B is the
            // same memory as column-major C^T = B^T * A^T, hence the swapped
            // operand order below. A partition becomes N-by-rows in this view.
            checkCublas(cublasDgemm(device.handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    dimension_, static_cast<int>(device.rows), dimension_,
                                    &alpha,
                                    device.B, dimension_,
                                    device.A, dimension_,
                                    &beta,
                                    device.C, dimension_),
                        "cublasDgemm");
        }
    }

    void synchronize() {
        for (DeviceContext& device : devices_) {
            checkCuda(cudaSetDevice(device.device), "cudaSetDevice");
            checkCuda(cudaStreamSynchronize(device.stream), "cudaStreamSynchronize");
        }
    }

    void release() noexcept {
        for (DeviceContext& device : devices_) {
            cudaSetDevice(device.device);
            if (device.handle != nullptr) {
                cublasDestroy(device.handle);
            }
            if (device.A != nullptr) {
                cudaFree(device.A);
            }
            if (device.B != nullptr) {
                cudaFree(device.B);
            }
            if (device.C != nullptr) {
                cudaFree(device.C);
            }
            if (device.stream != nullptr) {
                cudaStreamDestroy(device.stream);
            }
        }
        devices_.clear();
    }

    size_t N_;
    int dimension_;
    std::vector<DeviceContext> devices_;
};

size_t parseMatrixSize(const char* text) {
    if (text[0] == '\0' || text[0] == '-') {
        throw std::invalid_argument("matrix size must be a positive integer");
    }

    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (*end != '\0' || value == 0 || value > std::numeric_limits<size_t>::max()) {
        throw std::invalid_argument("matrix size must be a positive integer");
    }
    if (value > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("matrix size exceeds the cuBLAS dimension limit");
    }

    const size_t N = static_cast<size_t>(value);
    if (N > std::numeric_limits<size_t>::max() / N ||
        N * N > std::numeric_limits<size_t>::max() / sizeof(double)) {
        throw std::invalid_argument("matrix size is too large");
    }
    return N;
}

// Recompute representative elements on the CPU and compare.
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Include the beginning, partition interiors, and the last row/column so
    // multi-GPU slice boundaries are covered as well as the first slice.
    const size_t checkPoints[] = {0, 1 % N, N / 4, N / 2, N - 1};
    
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
            try {
                N = parseMatrixSize(argv[++i]);
            } catch (const std::exception& error) {
                fprintf(stderr, "Invalid matrix size: %s\n", error.what());
                return 1;
            }
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
    
    try {
        // Allocate matrices
        std::vector<double> A(N * N);
        std::vector<double> B(N * N);
        std::vector<double> C(N * N);

        // Initialize matrices
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);

        CudaMatrixMultiplier multiplier(A, B, N);
        printf("CUDA devices used: %zu\n", multiplier.deviceCount());

        // Perform matrix multiplication
        printf("Computing matrix multiplication...\n");
        const double durationMs = multiplier.multiply();
        multiplier.copyResult(C);

        printf("Computation time: %.3f ms\n", durationMs);

        // Calculate GFLOPS without overflowing an integer intermediate.
        const double dimension = static_cast<double>(N);
        const double gflops = (2.0 * dimension * dimension * dimension) /
                              (durationMs * 1.0e6);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(A, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
                return 0;
            }
            printf("Validation: FAILED\n");
            return 1;
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "CUDA matrix multiplication failed: %s\n", error.what());
        return 1;
    }

    return 0;
}
