#include <algorithm>
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

[[noreturn]] void cudaFailure(const char* operation, const char* detail) {
    std::fprintf(stderr, "CUDA error in %s: %s\n", operation, detail);
    std::exit(EXIT_FAILURE);
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) cudaFailure(operation, cudaGetErrorString(status));
}

void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[64];
        std::snprintf(message, sizeof(message), "cuBLAS status %d", static_cast<int>(status));
        cudaFailure(operation, message);
    }
}

void checkCusolver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        char message[64];
        std::snprintf(message, sizeof(message), "cuSOLVER status %d", static_cast<int>(status));
        cudaFailure(operation, message);
    }
}

__global__ void addDiagonal(double* matrix, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) matrix[static_cast<size_t>(i) * n + i] += static_cast<double>(n);
}

__global__ void clearUpperTriangle(double* matrix, int n) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && column < n && column > row) {
        matrix[static_cast<size_t>(row) * n + column] = 0.0;
    }
}

// Construct A = B * B^T on the GPU. cuBLAS is column-major, while the
// benchmark arrays are row-major; viewing B as B^T and computing B^T * B
// produces the same (symmetric) byte layout.
void generatePositiveDefiniteMatrix(double* deviceA, int n, cublasHandle_t cublas) {
    const size_t elements = static_cast<size_t>(n) * n;
    std::vector<double> B(elements);
    unsigned int seed = 42;
    for (size_t i = 0; i < elements; ++i) {
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    double* deviceB = nullptr;
    checkCuda(cudaMalloc(&deviceB, elements * sizeof(double)), "allocating B");
    checkCuda(cudaMemcpy(deviceB, B.data(), elements * sizeof(double),
                         cudaMemcpyHostToDevice), "copying B to the GPU");

    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;
    checkCublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                            &alpha, deviceB, n, deviceB, n, &beta, deviceA, n),
                "forming B * B^T");
    addDiagonal<<<(n + 255) / 256, 256>>>(deviceA, n);
    checkCuda(cudaGetLastError(), "adding diagonal dominance");
    checkCuda(cudaFree(deviceB), "freeing B");
}

// cuSOLVER's upper-triangular column-major result has exactly the desired
// lower-triangular row-major layout in memory.
bool choleskyDecomposition(double* deviceA, int n, cusolverDnHandle_t solver,
                           float& elapsedMilliseconds) {
    int workspaceElements = 0;
    checkCusolver(cusolverDnDpotrf_bufferSize(
                      solver, CUBLAS_FILL_MODE_UPPER, n, deviceA, n,
                      &workspaceElements),
                  "querying Cholesky workspace");

    double* workspace = nullptr;
    int* deviceInfo = nullptr;
    checkCuda(cudaMalloc(&workspace, static_cast<size_t>(workspaceElements) * sizeof(double)),
              "allocating Cholesky workspace");
    checkCuda(cudaMalloc(&deviceInfo, sizeof(int)), "allocating Cholesky status");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&stop), "creating stop event");
    checkCuda(cudaEventRecord(start), "recording start event");

    checkCusolver(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, n, deviceA, n,
                                   workspace, workspaceElements, deviceInfo),
                  "computing Cholesky decomposition");
    const dim3 block(32, 8);
    const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    clearUpperTriangle<<<grid, block>>>(deviceA, n);
    checkCuda(cudaGetLastError(), "clearing the upper triangle");
    checkCuda(cudaEventRecord(stop), "recording stop event");
    checkCuda(cudaEventSynchronize(stop), "waiting for Cholesky decomposition");
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, start, stop),
              "measuring Cholesky decomposition");

    int info = 0;
    checkCuda(cudaMemcpy(&info, deviceInfo, sizeof(int), cudaMemcpyDeviceToHost),
              "reading Cholesky status");
    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(stop), "destroying stop event");
    checkCuda(cudaFree(deviceInfo), "freeing Cholesky status");
    checkCuda(cudaFree(workspace), "freeing Cholesky workspace");

    if (info < 0) {
        std::fprintf(stderr, "Error: cuSOLVER received an invalid argument at position %d\n", -info);
        return false;
    }
    if (info > 0) {
        std::printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        return false;
    }
    return true;
}

bool validateCholesky(const double* deviceL, const std::vector<double>& original,
                      int n, cublasHandle_t cublas) {
    const size_t elements = static_cast<size_t>(n) * n;
    double* reconstructedDevice = nullptr;
    checkCuda(cudaMalloc(&reconstructedDevice, elements * sizeof(double)),
              "allocating validation matrix");

    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;
    // The column-major view of row-major L is L^T, so this forms L * L^T.
    checkCublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                            &alpha, deviceL, n, deviceL, n, &beta,
                            reconstructedDevice, n),
                "reconstructing L * L^T");

    std::vector<double> reconstructed(elements);
    checkCuda(cudaMemcpy(reconstructed.data(), reconstructedDevice,
                         elements * sizeof(double), cudaMemcpyDeviceToHost),
              "copying validation matrix");
    checkCuda(cudaFree(reconstructedDevice), "freeing validation matrix");

    double maxError = 0.0;
    double relError = 0.0;
    for (size_t i = 0; i < elements; ++i) {
        const double error = std::fabs(reconstructed[i] - original[i]);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (std::fabs(original[i]) + 1e-10));
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    size_t matrixSize = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (!end || *end != '\0' || parsed == 0 || parsed > INT_MAX) {
                std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            matrixSize = static_cast<size_t>(parsed);
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

    if (matrixSize > std::numeric_limits<size_t>::max() / matrixSize ||
        matrixSize * matrixSize > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }
    const int n = static_cast<int>(matrixSize);
    const size_t elements = matrixSize * matrixSize;

    std::printf("Cholesky Decomposition Benchmark\n");
    std::printf("Matrix size: %zu x %zu\n", matrixSize, matrixSize);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    checkCuda(cudaSetDevice(0), "selecting CUDA device");
    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    checkCublas(cublasCreate(&cublas), "creating cuBLAS handle");
    checkCusolver(cusolverDnCreate(&solver), "creating cuSOLVER handle");

    double* deviceA = nullptr;
    checkCuda(cudaMalloc(&deviceA, elements * sizeof(double)), "allocating matrix A");
    std::printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(deviceA, n, cublas);

    std::vector<double> original;
    if (validate) {
        original.resize(elements);
        checkCuda(cudaMemcpy(original.data(), deviceA, elements * sizeof(double),
                             cudaMemcpyDeviceToHost), "saving original matrix");
    }

    std::printf("Computing Cholesky decomposition...\n");
    float elapsedMilliseconds = 0.0f;
    const bool success = choleskyDecomposition(deviceA, n, solver, elapsedMilliseconds);
    if (!success) {
        std::printf("Cholesky decomposition failed\n");
        checkCuda(cudaFree(deviceA), "freeing matrix A");
        checkCusolver(cusolverDnDestroy(solver), "destroying cuSOLVER handle");
        checkCublas(cublasDestroy(cublas), "destroying cuBLAS handle");
        return 1;
    }

    std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    const double ops = static_cast<double>(n) * n * n / 3.0;
    const double gflops = ops / (static_cast<double>(elapsedMilliseconds) * 1.0e6);
    std::printf("Performance: %.3f GFLOPS\n", gflops);

    std::vector<double> result;
    if (printResults) {
        result.resize(elements);
        checkCuda(cudaMemcpy(result.data(), deviceA, elements * sizeof(double),
                             cudaMemcpyDeviceToHost), "copying Cholesky result");
        print_results(result, "CholeskyL");
    }

    bool valid = true;
    if (validate) {
        std::printf("Validating result...\n");
        valid = validateCholesky(deviceA, original, n, cublas);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    checkCuda(cudaFree(deviceA), "freeing matrix A");
    checkCusolver(cusolverDnDestroy(solver), "destroying cuSOLVER handle");
    checkCublas(cublasDestroy(cublas), "destroying cuBLAS handle");
    return valid ? 0 : 1;
}
