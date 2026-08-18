#include <algorithm>
#include <chrono>
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

// The matrix is exposed to the rest of the program in row-major order.  CUDA
// dense libraries are column-major, so the same allocation is viewed as the
// transpose.  For a symmetric input, an upper-triangular POTRF in column-major
// layout therefore produces the requested lower-triangular factor in row-major
// layout without a transpose or an additional matrix copy.

namespace {

constexpr int kThreadsPerBlock = 256;

const char* cublasErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

[[noreturn]] void cudaFailure(const char* operation, cudaError_t status) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

[[noreturn]] void cublasFailure(const char* operation, cublasStatus_t status) {
    std::fprintf(stderr, "cuBLAS error during %s: %s\n", operation, cublasErrorString(status));
    std::exit(EXIT_FAILURE);
}

[[noreturn]] void cusolverFailure(const char* operation, cusolverStatus_t status) {
    std::fprintf(stderr, "cuSOLVER error during %s: status %d\n", operation, static_cast<int>(status));
    std::exit(EXIT_FAILURE);
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        cudaFailure(operation, status);
    }
}

void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasFailure(operation, status);
    }
}

void checkCusolver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        cusolverFailure(operation, status);
    }
}

__global__ void addDiagonalKernel(double* matrix, int n, double value) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        matrix[static_cast<size_t>(index) * n + index] += value;
    }
}

__global__ void zeroUpperTriangleKernel(double* matrix, int n) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements = static_cast<size_t>(n) * n;
    if (index < elements) {
        const int row = static_cast<int>(index / n);
        const int column = static_cast<int>(index - static_cast<size_t>(row) * n);
        if (row < column) {
            matrix[index] = 0.0;
        }
    }
}

int blocksFor(size_t count) {
    return static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

class GpuCholesky {
public:
    explicit GpuCholesky(size_t n)
        : n_(static_cast<int>(n)), elements_(n * n), bytes_(elements_ * sizeof(double)) {
        checkCublas(cublasCreate(&blas_), "cublasCreate");
        checkCusolver(cusolverDnCreate(&solver_), "cusolverDnCreate");
        checkCuda(cudaMalloc(&matrix_, bytes_), "allocating the matrix");
        checkCuda(cudaMalloc(&info_, sizeof(*info_)), "allocating POTRF status");
        checkCuda(cudaEventCreate(&factorStart_), "creating factorization start event");
        checkCuda(cudaEventCreate(&factorStop_), "creating factorization stop event");
    }

    GpuCholesky(const GpuCholesky&) = delete;
    GpuCholesky& operator=(const GpuCholesky&) = delete;

    ~GpuCholesky() {
        if (factorStop_ != nullptr) cudaEventDestroy(factorStop_);
        if (factorStart_ != nullptr) cudaEventDestroy(factorStart_);
        if (workspace_ != nullptr) cudaFree(workspace_);
        if (info_ != nullptr) cudaFree(info_);
        if (matrix_ != nullptr) cudaFree(matrix_);
        if (randomMatrix_ != nullptr) cudaFree(randomMatrix_);
        if (solver_ != nullptr) cusolverDnDestroy(solver_);
        if (blas_ != nullptr) cublasDestroy(blas_);
    }

    // Forms A = B * B^T + nI with a symmetric rank-k update on the GPU.
    // B itself is generated on the CPU to preserve the benchmark's seeded input.
    void generatePositiveDefiniteMatrix(const std::vector<double>& randomMatrix) {
        checkCuda(cudaMalloc(&randomMatrix_, bytes_), "allocating the random matrix");
        checkCuda(cudaMemcpyAsync(randomMatrix_, randomMatrix.data(), bytes_, cudaMemcpyHostToDevice),
                  "copying the random matrix to the GPU");
        checkCuda(cudaMemsetAsync(matrix_, 0, bytes_), "initializing the matrix");

        const double one = 1.0;
        const double zero = 0.0;
        // randomMatrix_ is B^T when read by a column-major library.  OP_T
        // computes B * B^T and writes the triangle that maps to row-major L.
        checkCublas(cublasDsyrk(blas_, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                n_, n_, &one, randomMatrix_, n_, &zero, matrix_, n_),
                    "forming B * B^T");
        addDiagonalKernel<<<blocksFor(n_), kThreadsPerBlock>>>(matrix_, n_, static_cast<double>(n_));
        checkCuda(cudaGetLastError(), "adding diagonal dominance");

        // The random matrix is not needed after SYRK.  cudaFree synchronizes
        // this stream, making the generated input ready for POTRF as well.
        checkCuda(cudaFree(randomMatrix_), "releasing the random matrix");
        randomMatrix_ = nullptr;
    }

    // The generated matrix occupies the lower row-major triangle.  Mirror it
    // on the host only when validation needs the original full matrix.
    void copyOriginalToHost(std::vector<double>& original) const {
        checkCuda(cudaMemcpy(original.data(), matrix_, bytes_, cudaMemcpyDeviceToHost),
                  "copying the original matrix from the GPU");
        for (int row = 0; row < n_; ++row) {
            for (int column = row + 1; column < n_; ++column) {
                original[static_cast<size_t>(row) * n_ + column] =
                    original[static_cast<size_t>(column) * n_ + row];
            }
        }
    }

    bool factorize(float& elapsedMilliseconds) {
        int workspaceElements = 0;
        checkCusolver(cusolverDnDpotrf_bufferSize(solver_, CUBLAS_FILL_MODE_UPPER, n_, matrix_, n_,
                                                  &workspaceElements),
                       "querying POTRF workspace");
        checkCuda(cudaMalloc(&workspace_, static_cast<size_t>(workspaceElements) * sizeof(double)),
                  "allocating POTRF workspace");

        checkCuda(cudaEventRecord(factorStart_), "recording factorization start");
        checkCusolver(cusolverDnDpotrf(solver_, CUBLAS_FILL_MODE_UPPER, n_, matrix_, n_, workspace_,
                                       workspaceElements, info_),
                       "running GPU Cholesky factorization");
        checkCuda(cudaEventRecord(factorStop_), "recording factorization stop");
        checkCuda(cudaEventSynchronize(factorStop_), "waiting for GPU Cholesky factorization");
        checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, factorStart_, factorStop_),
                  "measuring GPU Cholesky factorization");

        int info = 0;
        checkCuda(cudaMemcpy(&info, info_, sizeof(info), cudaMemcpyDeviceToHost),
                  "reading POTRF status");
        if (info < 0) {
            std::fprintf(stderr, "cuSOLVER POTRF received an invalid parameter: %d\n", -info);
            return false;
        }
        if (info > 0) {
            std::printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
            return false;
        }

        // POTRF's column-major upper factor is the required row-major lower
        // factor.  Match the original routine by explicitly clearing its upper
        // row-major triangle before exposing the result.
        zeroUpperTriangleKernel<<<blocksFor(elements_), kThreadsPerBlock>>>(matrix_, n_);
        checkCuda(cudaGetLastError(), "clearing the upper triangular matrix");
        checkCuda(cudaDeviceSynchronize(), "finishing the Cholesky result");
        return true;
    }

    void copyFactorToHost(std::vector<double>& factor) const {
        checkCuda(cudaMemcpy(factor.data(), matrix_, bytes_, cudaMemcpyDeviceToHost),
                  "copying the Cholesky factor from the GPU");
    }

    bool validate(const std::vector<double>& original) const {
        double* reconstructed = nullptr;
        checkCuda(cudaMalloc(&reconstructed, bytes_), "allocating validation matrix");

        const double one = 1.0;
        const double zero = 0.0;
        // matrix_ is U in column-major form, where U = L^T.  This computes
        // U^T * U = L * L^T directly and produces the full reconstruction.
        checkCublas(cublasDgemm(blas_, CUBLAS_OP_T, CUBLAS_OP_N, n_, n_, n_, &one,
                                matrix_, n_, matrix_, n_, &zero, reconstructed, n_),
                    "reconstructing L * L^T for validation");

        std::vector<double> reconstructedHost(elements_);
        checkCuda(cudaMemcpy(reconstructedHost.data(), reconstructed, bytes_, cudaMemcpyDeviceToHost),
                  "copying the validation matrix from the GPU");
        checkCuda(cudaFree(reconstructed), "releasing the validation matrix");

        double maxError = 0.0;
        double relativeError = 0.0;
        for (size_t i = 0; i < elements_; ++i) {
            const double error = std::fabs(reconstructedHost[i] - original[i]);
            maxError = std::max(maxError, error);
            relativeError = std::max(relativeError, error / (std::fabs(original[i]) + 1e-10));
        }

        std::printf("Max absolute error: %.10e\n", maxError);
        std::printf("Max relative error: %.10e\n", relativeError);
        if (relativeError > 1e-6) {
            std::printf("Validation failed: relative error too large\n");
            return false;
        }
        return true;
    }

private:
    int n_;
    size_t elements_;
    size_t bytes_;
    cublasHandle_t blas_ = nullptr;
    cusolverDnHandle_t solver_ = nullptr;
    double* matrix_ = nullptr;
    double* randomMatrix_ = nullptr;
    double* workspace_ = nullptr;
    int* info_ = nullptr;
    cudaEvent_t factorStart_ = nullptr;
    cudaEvent_t factorStop_ = nullptr;
};

// Generate the same deterministic random B as the original benchmark.  Its
// O(n^3) product is then formed by cuBLAS instead of the former CPU triple loop.
void generateRandomMatrix(std::vector<double>& matrix, size_t n) {
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        matrix[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
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
            n = std::atoi(argv[++i]);
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

    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > std::numeric_limits<size_t>::max() / n) {
        std::fprintf(stderr, "Matrix size must be between 1 and %d\n", std::numeric_limits<int>::max());
        return 1;
    }

    std::printf("Cholesky Decomposition Benchmark\n");
    std::printf("Matrix size: %zu x %zu\n", n, n);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t elements = n * n;
    std::vector<double> randomMatrix(elements);
    std::vector<double> factor(elements);
    std::vector<double> original;

    std::printf("Generating positive definite matrix...\n");
    generateRandomMatrix(randomMatrix, n);

    GpuCholesky cholesky(n);
    cholesky.generatePositiveDefiniteMatrix(randomMatrix);
    randomMatrix.clear();
    randomMatrix.shrink_to_fit();

    if (validate) {
        original.resize(elements);
        cholesky.copyOriginalToHost(original);
    }

    std::printf("Computing Cholesky decomposition...\n");
    float elapsedMilliseconds = 0.0F;
    const bool success = cholesky.factorize(elapsedMilliseconds);
    if (!success) {
        std::printf("Cholesky decomposition failed\n");
        return 1;
    }

    // Preserve the original millisecond-oriented output while avoiding a zero
    // denominator for very small GPU workloads.
    const long displayedMilliseconds = std::max(1L, static_cast<long>(std::ceil(elapsedMilliseconds)));
    std::printf("Computation time: %ld ms\n", displayedMilliseconds);

    const double ops = static_cast<double>(n) * n * n / 3.0;
    const double gflops = ops / (static_cast<double>(elapsedMilliseconds) / 1000.0) / 1e9;
    std::printf("Performance: %.3f GFLOPS\n", gflops);

    cholesky.copyFactorToHost(factor);
    if (printResults) {
        print_results(factor, "CholeskyL");
    }

    if (validate) {
        std::printf("Validating result...\n");
        if (cholesky.validate(original)) {
            std::printf("Validation: PASSED\n");
            return 0;
        }
        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
