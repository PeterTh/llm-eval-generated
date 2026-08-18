#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {

bool cudaCheck(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

bool solverCheck(cusolverStatus_t status, const char* operation) {
    if (status == CUSOLVER_STATUS_SUCCESS) return true;
    std::fprintf(stderr, "cuSOLVER error during %s (status %d)\n", operation,
                 static_cast<int>(status));
    return false;
}

bool blasCheck(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    std::fprintf(stderr, "cuBLAS error during %s (status %d)\n", operation,
                 static_cast<int>(status));
    return false;
}

// cuSOLVER uses column-major matrices. Factoring the upper triangle in its view
// produces the lower triangle in the program's row-major view.
__global__ void zeroRowMajorUpper(double* matrix, size_t n) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row) matrix[row * n + col] = 0.0;
}

__global__ void addDiagonal(double* matrix, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) matrix[i * n + i] += static_cast<double>(n);
}

class CudaCholesky {
  public:
    CudaCholesky(const std::vector<double>& matrix, size_t n) : n_(n) {
        if (n == 0) {
            ready_ = true;
            return;
        }
        if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            n > std::numeric_limits<size_t>::max() / n ||
            n * n > std::numeric_limits<size_t>::max() / sizeof(double)) {
            std::fprintf(stderr, "Matrix size is too large for CUDA Cholesky\n");
            return;
        }

        const size_t bytes = n * n * sizeof(double);
        if (!solverCheck(cusolverDnCreate(&handle_), "handle creation") ||
            !cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceMatrix_), bytes),
                       "matrix allocation") ||
            !cudaCheck(cudaMemcpy(deviceMatrix_, matrix.data(), bytes,
                                  cudaMemcpyHostToDevice),
                       "matrix upload"))
            return;

        int workspaceElements = 0;
        if (!solverCheck(cusolverDnDpotrf_bufferSize(
                             handle_, CUBLAS_FILL_MODE_UPPER,
                             static_cast<int>(n), deviceMatrix_, static_cast<int>(n),
                             &workspaceElements),
                         "workspace sizing") ||
            !cudaCheck(cudaMalloc(reinterpret_cast<void**>(&workspace_),
                                  static_cast<size_t>(workspaceElements) * sizeof(double)),
                       "workspace allocation") ||
            !cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceInfo_), sizeof(int)),
                       "status allocation"))
            return;
        workspaceElements_ = workspaceElements;
        ready_ = true;
    }

    CudaCholesky(const CudaCholesky&) = delete;
    CudaCholesky& operator=(const CudaCholesky&) = delete;

    ~CudaCholesky() {
        if (deviceInfo_) cudaFree(deviceInfo_);
        if (workspace_) cudaFree(workspace_);
        if (deviceMatrix_) cudaFree(deviceMatrix_);
        if (handle_) cusolverDnDestroy(handle_);
    }

    bool ready() const { return ready_; }

    bool factorize() {
        if (!ready_) return false;
        if (n_ == 0) return true;

        if (!solverCheck(cusolverDnDpotrf(
                             handle_, CUBLAS_FILL_MODE_UPPER,
                             static_cast<int>(n_), deviceMatrix_, static_cast<int>(n_),
                             workspace_, workspaceElements_, deviceInfo_),
                         "factorization"))
            return false;

        constexpr unsigned tileX = 32;
        constexpr unsigned tileY = 8;
        const dim3 threads(tileX, tileY);
        const dim3 blocks(static_cast<unsigned>((n_ + tileX - 1) / tileX),
                          static_cast<unsigned>((n_ + tileY - 1) / tileY));
        zeroRowMajorUpper<<<blocks, threads>>>(deviceMatrix_, n_);
        if (!cudaCheck(cudaGetLastError(), "upper-triangle cleanup launch") ||
            !cudaCheck(cudaDeviceSynchronize(), "factorization synchronization"))
            return false;

        int info = 0;
        if (!cudaCheck(cudaMemcpy(&info, deviceInfo_, sizeof(info),
                                  cudaMemcpyDeviceToHost),
                       "factorization status download"))
            return false;
        if (info < 0) {
            std::fprintf(stderr, "cuSOLVER received an invalid argument at position %d\n", -info);
            return false;
        }
        if (info > 0) {
            std::printf("Error: Matrix is not positive definite at diagonal element %d\n",
                        info - 1);
            return false;
        }
        return true;
    }

    bool download(std::vector<double>& matrix) const {
        if (!ready_) return false;
        if (n_ == 0) return true;
        return cudaCheck(cudaMemcpy(matrix.data(), deviceMatrix_,
                                    n_ * n_ * sizeof(double), cudaMemcpyDeviceToHost),
                         "result download");
    }

  private:
    size_t n_ = 0;
    cusolverDnHandle_t handle_ = nullptr;
    double* deviceMatrix_ = nullptr;
    double* workspace_ = nullptr;
    int* deviceInfo_ = nullptr;
    int workspaceElements_ = 0;
    bool ready_ = false;
};

}  // namespace

// Generate a symmetric positive definite matrix from the same deterministic
// seeded input and the same mathematical B * B^T construction as the original.
bool generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;

    cublasHandle_t handle = nullptr;
    double* deviceB = nullptr;
    double* deviceA = nullptr;
    const size_t bytes = n * n * sizeof(double);
    bool ok = blasCheck(cublasCreate(&handle), "generator handle creation") &&
              cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceB), bytes),
                        "generator input allocation") &&
              cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceA), bytes),
                        "generator output allocation") &&
              cudaCheck(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice),
                        "generator input upload");

    if (ok) {
        // Row-major B is column-major B^T in the same storage. Therefore this
        // column-major product is exactly the desired row-major B * B^T.
        const double alpha = 1.0;
        const double beta = 0.0;
        ok = blasCheck(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  static_cast<int>(n), static_cast<int>(n),
                                  static_cast<int>(n), &alpha, deviceB,
                                  static_cast<int>(n), deviceB, static_cast<int>(n),
                                  &beta, deviceA, static_cast<int>(n)),
                       "positive-definite matrix multiplication");
    }
    if (ok) {
        constexpr unsigned threads = 256;
        addDiagonal<<<static_cast<unsigned>((n + threads - 1) / threads), threads>>>(
            deviceA, n);
        ok = cudaCheck(cudaGetLastError(), "diagonal kernel launch") &&
             cudaCheck(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost),
                       "generated matrix download");
    }

    if (deviceA) cudaFree(deviceA);
    if (deviceB) cudaFree(deviceB);
    if (handle) cublasDestroy(handle);
    return ok;
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t terms = std::min(i, j) + 1;
            for (size_t k = 0; k < terms; ++k)
                sum += L[i * n + k] * L[j * n + k];
            const double expected = A_orig[i * n + j];
            const double error = std::fabs(sum - expected);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(expected) + 1e-10));
        }
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* text = argv[++i];
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(text, &end, 10);
            if (text[0] == '-' || !end || *end != '\0' || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max()) {
                std::fprintf(stderr, "Invalid matrix size: %s\n", text);
                return 1;
            }
            n = static_cast<size_t>(parsed);
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

    if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > std::numeric_limits<size_t>::max() / n ||
        n * n > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Matrix size exceeds the supported address range\n");
        return 1;
    }
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    std::printf("Generating positive definite matrix...\n");
    if (!generatePositiveDefiniteMatrix(A, n)) return 1;
    if (validate) A_orig = A;

    // Creation uploads the input and allocates reusable workspace. The timed
    // region consequently measures the GPU factorization itself.
    CudaCholesky gpu(A, n);
    if (!gpu.ready()) return 1;
    std::printf("Computing Cholesky decomposition...\n");
    const auto start = std::chrono::steady_clock::now();
    const bool success = gpu.factorize();
    const auto end = std::chrono::steady_clock::now();
    if (!success || !gpu.download(A)) {
        std::printf("Cholesky decomposition failed\n");
        return 1;
    }

    const double milliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
    std::printf("Computation time: %.3f ms\n", milliseconds);
    const double ops = static_cast<double>(n) * static_cast<double>(n) *
                       static_cast<double>(n) / 3.0;
    std::printf("Performance: %.3f GFLOPS\n", ops / milliseconds / 1e6);

    if (printResults) print_results(A, "CholeskyL");
    if (validate) {
        std::printf("Validating result...\n");
        const bool valid = validateCholesky(A, A_orig, n);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
