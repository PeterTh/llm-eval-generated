#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {

constexpr unsigned int kThreadsPerBlock = 256;

bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

bool checkCublas(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "cuBLAS error during %s (status %d)\n", operation,
                 static_cast<int>(status));
    return false;
}

bool checkCusolver(cusolverStatus_t status, const char* operation) {
    if (status == CUSOLVER_STATUS_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "cuSOLVER error during %s (status %d)\n", operation,
                 static_cast<int>(status));
    return false;
}

// The matrix is physically row-major. cuSOLVER sees the same memory as a
// column-major matrix, so its upper factor is the transpose of the row-major
// lower factor required by the benchmark. This kernel clears the unused half.
__global__ void zeroRowMajorUpperTriangle(double* matrix, int n) {
    const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row < n && column < n && column > row) {
        matrix[static_cast<size_t>(row) * n + column] = 0.0;
    }
}

__global__ void addDiagonal(double* matrix, int n, double value) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < n) {
        matrix[static_cast<size_t>(index) * n + index] += value;
    }
}

__device__ void atomicMaxNonnegativeDouble(double* address, double value) {
    atomicMax(reinterpret_cast<unsigned long long*>(address),
              __double_as_longlong(value));
}

__global__ void errorMaxima(const double* reconstructed, const double* original,
                            size_t count, double* maxima) {
    __shared__ double absoluteErrors[kThreadsPerBlock];
    __shared__ double relativeErrors[kThreadsPerBlock];

    double maxAbsolute = 0.0;
    double maxRelative = 0.0;
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t index = first; index < count; index += stride) {
        const double error = fabs(reconstructed[index] - original[index]);
        maxAbsolute = fmax(maxAbsolute, error);
        maxRelative = fmax(maxRelative, error / (fabs(original[index]) + 1e-10));
    }

    absoluteErrors[threadIdx.x] = maxAbsolute;
    relativeErrors[threadIdx.x] = maxRelative;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            absoluteErrors[threadIdx.x] =
                fmax(absoluteErrors[threadIdx.x], absoluteErrors[threadIdx.x + offset]);
            relativeErrors[threadIdx.x] =
                fmax(relativeErrors[threadIdx.x], relativeErrors[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMaxNonnegativeDouble(maxima, absoluteErrors[0]);
        atomicMaxNonnegativeDouble(maxima + 1, relativeErrors[0]);
    }
}

class CudaCholesky {
public:
    CudaCholesky() = default;
    CudaCholesky(const CudaCholesky&) = delete;
    CudaCholesky& operator=(const CudaCholesky&) = delete;

    ~CudaCholesky() {
        if (workspace_ != nullptr) {
            cudaFree(workspace_);
        }
        if (info_ != nullptr) {
            cudaFree(info_);
        }
        if (original_ != nullptr) {
            cudaFree(original_);
        }
        if (matrix_ != nullptr) {
            cudaFree(matrix_);
        }
        if (cusolver_ != nullptr) {
            cusolverDnDestroy(cusolver_);
        }
        if (cublas_ != nullptr) {
            cublasDestroy(cublas_);
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    bool initialize(size_t n, bool preserveOriginal) {
        if (n > static_cast<size_t>(INT_MAX)) {
            std::fprintf(stderr, "Matrix size exceeds the CUDA library limit\n");
            return false;
        }
        if (n != 0 && n > std::numeric_limits<size_t>::max() / n / sizeof(double)) {
            std::fprintf(stderr, "Matrix allocation size overflow\n");
            return false;
        }

        n_ = static_cast<int>(n);
        elementCount_ = n * n;
        byteCount_ = elementCount_ * sizeof(double);

        if (!checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                       "stream creation") ||
            !checkCublas(cublasCreate(&cublas_), "handle creation") ||
            !checkCusolver(cusolverDnCreate(&cusolver_), "handle creation") ||
            !checkCublas(cublasSetStream(cublas_, stream_), "stream setup") ||
            !checkCusolver(cusolverDnSetStream(cusolver_, stream_), "stream setup")) {
            return false;
        }

        // CUDA permits no useful factorization for an empty matrix, but the
        // original n=0 behavior is a successful no-op.
        if (n_ == 0) {
            return true;
        }

        if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&matrix_), byteCount_),
                       "matrix allocation") ||
            (preserveOriginal &&
             !checkCuda(cudaMalloc(reinterpret_cast<void**>(&original_), byteCount_),
                        "validation matrix allocation")) ||
            !checkCuda(cudaMalloc(reinterpret_cast<void**>(&info_), sizeof(int)),
                       "factorization status allocation")) {
            return false;
        }

        int workspaceElements = 0;
        if (!checkCusolver(
                cusolverDnDpotrf_bufferSize(cusolver_, CUBLAS_FILL_MODE_UPPER, n_,
                                            matrix_, n_, &workspaceElements),
                "workspace query")) {
            return false;
        }
        workspaceElements_ = workspaceElements;
        if (workspaceElements_ > 0 &&
            !checkCuda(cudaMalloc(reinterpret_cast<void**>(&workspace_),
                                  static_cast<size_t>(workspaceElements_) * sizeof(double)),
                       "factorization workspace allocation")) {
            return false;
        }
        return true;
    }

    // Generates A = B * B^T + nI on the GPU. B is initialized on the host with
    // the same deterministic rand_r sequence as the sequential benchmark.
    bool generatePositiveDefiniteMatrix(bool preserveOriginal) {
        if (n_ == 0) {
            return true;
        }

        std::vector<double> randomMatrix(elementCount_);
        unsigned int seed = 42;
        for (double& value : randomMatrix) {
            value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        }

        double* deviceRandom = nullptr;
        if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceRandom), byteCount_),
                       "random matrix allocation") ||
            !checkCuda(cudaMemcpyAsync(deviceRandom, randomMatrix.data(), byteCount_,
                                       cudaMemcpyHostToDevice, stream_),
                       "random matrix upload")) {
            if (deviceRandom != nullptr) {
                cudaFree(deviceRandom);
            }
            return false;
        }

        // Row-major B is column-major B^T in device memory. Therefore B_col^T
        // * B_col produces exactly the desired symmetric row-major B * B^T.
        const double one = 1.0;
        const double zero = 0.0;
        bool ok = checkCublas(
            cublasDgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, n_, n_, n_, &one,
                        deviceRandom, n_, deviceRandom, n_, &zero, matrix_, n_),
            "positive-definite matrix multiplication");

        if (ok) {
            const unsigned int blocks =
                (static_cast<unsigned int>(n_) + kThreadsPerBlock - 1) /
                kThreadsPerBlock;
            addDiagonal<<<blocks, kThreadsPerBlock, 0, stream_>>>(matrix_, n_,
                                                                  static_cast<double>(n_));
            ok = checkCuda(cudaGetLastError(), "diagonal kernel launch");
        }
        if (ok && preserveOriginal) {
            ok = checkCuda(cudaMemcpyAsync(original_, matrix_, byteCount_,
                                           cudaMemcpyDeviceToDevice, stream_),
                           "validation matrix copy");
        }
        if (ok) {
            ok = checkCuda(cudaStreamSynchronize(stream_), "matrix generation");
        }

        const bool freeOk = checkCuda(cudaFree(deviceRandom), "random matrix release");
        return ok && freeOk;
    }

    bool factorize() {
        if (n_ == 0) {
            return true;
        }

        if (!checkCusolver(
                cusolverDnDpotrf(cusolver_, CUBLAS_FILL_MODE_UPPER, n_, matrix_, n_,
                                 workspace_, workspaceElements_, info_),
                "Cholesky factorization")) {
            return false;
        }

        const dim3 block(32, 8);
        const dim3 grid((static_cast<unsigned int>(n_) + block.x - 1) / block.x,
                        (static_cast<unsigned int>(n_) + block.y - 1) / block.y);
        zeroRowMajorUpperTriangle<<<grid, block, 0, stream_>>>(matrix_, n_);
        if (!checkCuda(cudaGetLastError(), "upper-triangle kernel launch")) {
            return false;
        }

        int factorizationInfo = 0;
        if (!checkCuda(cudaMemcpyAsync(&factorizationInfo, info_, sizeof(int),
                                       cudaMemcpyDeviceToHost, stream_),
                       "factorization status copy") ||
            !checkCuda(cudaStreamSynchronize(stream_), "factorization completion")) {
            return false;
        }
        if (factorizationInfo < 0) {
            std::fprintf(stderr, "cuSOLVER rejected argument %d\n", -factorizationInfo);
            return false;
        }
        if (factorizationInfo > 0) {
            std::printf("Error: Matrix is not positive definite at diagonal element %d\n",
                        factorizationInfo - 1);
            return false;
        }
        return true;
    }

    bool downloadFactor(std::vector<double>& factor) {
        factor.resize(elementCount_);
        if (n_ == 0) {
            return true;
        }
        return checkCuda(cudaMemcpy(factor.data(), matrix_, byteCount_,
                                    cudaMemcpyDeviceToHost),
                         "factor download");
    }

    bool validate(double& maxAbsoluteError, double& maxRelativeError) {
        maxAbsoluteError = 0.0;
        maxRelativeError = 0.0;
        if (n_ == 0) {
            return true;
        }
        if (original_ == nullptr) {
            std::fprintf(stderr, "Internal error: validation matrix was not retained\n");
            return false;
        }

        double* reconstructed = nullptr;
        double* maxima = nullptr;
        if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&reconstructed), byteCount_),
                       "reconstruction allocation") ||
            !checkCuda(cudaMalloc(reinterpret_cast<void**>(&maxima), 2 * sizeof(double)),
                       "validation reduction allocation")) {
            if (reconstructed != nullptr) {
                cudaFree(reconstructed);
            }
            if (maxima != nullptr) {
                cudaFree(maxima);
            }
            return false;
        }

        // matrix_ is column-major U (and row-major L), so U^T * U reconstructs A.
        const double one = 1.0;
        const double zero = 0.0;
        bool ok = checkCublas(
            cublasDgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, n_, n_, n_, &one,
                        matrix_, n_, matrix_, n_, &zero, reconstructed, n_),
            "validation matrix multiplication");
        if (ok) {
            ok = checkCuda(cudaMemsetAsync(maxima, 0, 2 * sizeof(double), stream_),
                           "validation reduction initialization");
        }
        if (ok) {
            const size_t requestedBlocks =
                (elementCount_ + kThreadsPerBlock - 1) / kThreadsPerBlock;
            const unsigned int blocks = static_cast<unsigned int>(
                std::min<size_t>(requestedBlocks, 65535));
            errorMaxima<<<blocks, kThreadsPerBlock, 0, stream_>>>(
                reconstructed, original_, elementCount_, maxima);
            ok = checkCuda(cudaGetLastError(), "validation reduction kernel launch");
        }

        double hostMaxima[2] = {0.0, 0.0};
        if (ok) {
            ok = checkCuda(cudaMemcpyAsync(hostMaxima, maxima, sizeof(hostMaxima),
                                           cudaMemcpyDeviceToHost, stream_),
                           "validation result copy");
        }
        if (ok) {
            ok = checkCuda(cudaStreamSynchronize(stream_), "validation completion");
        }
        maxAbsoluteError = hostMaxima[0];
        maxRelativeError = hostMaxima[1];

        const bool reconstructedFreeOk =
            checkCuda(cudaFree(reconstructed), "reconstruction release");
        const bool maximaFreeOk = checkCuda(cudaFree(maxima), "validation reduction release");
        return ok && reconstructedFreeOk && maximaFreeOk;
    }

private:
    int n_ = 0;
    size_t elementCount_ = 0;
    size_t byteCount_ = 0;
    int workspaceElements_ = 0;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    cusolverDnHandle_t cusolver_ = nullptr;
    double* matrix_ = nullptr;
    double* original_ = nullptr;
    double* workspace_ = nullptr;
    int* info_ = nullptr;
};

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(std::atoll(argv[++i]));
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

    std::printf("Cholesky Decomposition Benchmark\n");
    std::printf("Matrix size: %zu x %zu\n", n, n);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    CudaCholesky gpu;
    if (!gpu.initialize(n, validate)) {
        return 1;
    }

    std::printf("Generating positive definite matrix...\n");
    if (!gpu.generatePositiveDefiniteMatrix(validate)) {
        std::printf("Matrix generation failed\n");
        return 1;
    }

    std::printf("Computing Cholesky decomposition...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    const bool success = gpu.factorize();
    const auto end = std::chrono::high_resolution_clock::now();

    if (!success) {
        std::printf("Cholesky decomposition failed\n");
        return 1;
    }

    const std::chrono::duration<double, std::milli> duration = end - start;
    std::printf("Computation time: %.3f ms\n", duration.count());

    const double operations = static_cast<double>(n) * static_cast<double>(n) *
                              static_cast<double>(n) / 3.0;
    const double seconds = duration.count() / 1000.0;
    const double gflops = seconds > 0.0 ? operations / seconds / 1e9 : 0.0;
    std::printf("Performance: %.3f GFLOPS\n", gflops);

    if (printResults) {
        std::vector<double> factor;
        if (!gpu.downloadFactor(factor)) {
            return 1;
        }
        print_results(factor, "CholeskyL");
    }

    if (validate) {
        std::printf("Validating result...\n");
        double maxAbsoluteError = 0.0;
        double maxRelativeError = 0.0;
        const bool validationCompleted =
            gpu.validate(maxAbsoluteError, maxRelativeError);
        if (!validationCompleted) {
            std::printf("Validation: FAILED\n");
            return 1;
        }

        std::printf("Max absolute error: %.10e\n", maxAbsoluteError);
        std::printf("Max relative error: %.10e\n", maxRelativeError);
        if (maxRelativeError <= 1e-6) {
            std::printf("Validation: PASSED\n");
            return 0;
        }

        std::printf("Validation failed: relative error too large\n");
        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
