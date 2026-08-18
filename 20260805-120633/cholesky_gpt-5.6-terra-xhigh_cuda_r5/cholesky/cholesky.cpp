#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// A is row-major, while cuSOLVER is column-major.  Since the input is symmetric,
// an upper Cholesky factorization in column-major storage is exactly the desired
// lower factor in this row-major view.

namespace {

bool checkCuda(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }

    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool checkCusolver(const cusolverStatus_t status, const char* operation) {
    if (status == CUSOLVER_STATUS_SUCCESS) {
        return true;
    }

    fprintf(stderr, "cuSOLVER error during %s: status %d\n", operation, static_cast<int>(status));
    return false;
}

struct CudaResources {
    cudaStream_t stream = nullptr;
    cusolverDnHandle_t solver = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    double* matrix = nullptr;
    double* workspace = nullptr;
    int* deviceInfo = nullptr;
    int* hostInfo = nullptr;

    ~CudaResources() {
        if (hostInfo != nullptr) cudaFreeHost(hostInfo);
        if (deviceInfo != nullptr) cudaFree(deviceInfo);
        if (workspace != nullptr) cudaFree(workspace);
        if (matrix != nullptr) cudaFree(matrix);
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
        if (solver != nullptr) cusolverDnDestroy(solver);
        if (stream != nullptr) cudaStreamDestroy(stream);
    }
};

__global__ void zeroUpperTriangle(double* const matrix, const size_t n) {
    const size_t column = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (row < n && column < n && column > row) {
        matrix[row * n + column] = 0.0;
    }
}

}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n, float& elapsedMilliseconds) {
    elapsedMilliseconds = 0.0F;

    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "CUDA Cholesky supports matrix dimensions up to %d\n",
                std::numeric_limits<int>::max());
        return false;
    }

    const size_t matrixBytes = n * n * sizeof(double);
    const int dimension = static_cast<int>(n);
    CudaResources resources;

    if (!checkCuda(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking),
                   "stream creation") ||
        !checkCusolver(cusolverDnCreate(&resources.solver), "handle creation") ||
        !checkCusolver(cusolverDnSetStream(resources.solver, resources.stream), "stream setup") ||
        !checkCuda(cudaMalloc(reinterpret_cast<void**>(&resources.matrix), matrixBytes),
                   "matrix allocation") ||
        !checkCuda(cudaMalloc(reinterpret_cast<void**>(&resources.deviceInfo), sizeof(int)),
                   "status allocation") ||
        !checkCuda(cudaMallocHost(reinterpret_cast<void**>(&resources.hostInfo), sizeof(int)),
                   "pinned status allocation") ||
        !checkCuda(cudaEventCreate(&resources.start), "start-event creation") ||
        !checkCuda(cudaEventCreate(&resources.stop), "stop-event creation")) {
        return false;
    }

    int workspaceElements = 0;
    if (!checkCusolver(cusolverDnDpotrf_bufferSize(resources.solver, CUBLAS_FILL_MODE_UPPER,
                                                    dimension, resources.matrix, dimension,
                                                    &workspaceElements),
                       "workspace query") ||
        !checkCuda(cudaMalloc(reinterpret_cast<void**>(&resources.workspace),
                              static_cast<size_t>(workspaceElements) * sizeof(double)),
                   "workspace allocation") ||
        !checkCuda(cudaMemcpyAsync(resources.matrix, A.data(), matrixBytes, cudaMemcpyHostToDevice,
                                   resources.stream),
                   "matrix upload")) {
        return false;
    }

    // cuSOLVER uses its blocked GPU implementation, which maps the panel work to
    // highly parallel GEMM/TRSM kernels and scales with the available GPU resources.
    if (!checkCuda(cudaEventRecord(resources.start, resources.stream), "start-event record") ||
        !checkCusolver(cusolverDnDpotrf(resources.solver, CUBLAS_FILL_MODE_UPPER, dimension,
                                         resources.matrix, dimension, resources.workspace,
                                         workspaceElements, resources.deviceInfo),
                       "Cholesky factorization")) {
        return false;
    }

    constexpr unsigned int blockColumns = 32;
    constexpr unsigned int blockRows = 8;
    const dim3 block(blockColumns, blockRows);
    const dim3 grid(static_cast<unsigned int>((n + blockColumns - 1) / blockColumns),
                    static_cast<unsigned int>((n + blockRows - 1) / blockRows));
    zeroUpperTriangle<<<grid, block, 0, resources.stream>>>(resources.matrix, n);

    if (!checkCuda(cudaGetLastError(), "upper-triangle cleanup launch") ||
        !checkCuda(cudaMemcpyAsync(resources.hostInfo, resources.deviceInfo, sizeof(int),
                                   cudaMemcpyDeviceToHost, resources.stream),
                   "factorization status download") ||
        !checkCuda(cudaEventRecord(resources.stop, resources.stream), "stop-event record") ||
        !checkCuda(cudaEventSynchronize(resources.stop), "factorization completion") ||
        !checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, resources.start, resources.stop),
                   "elapsed-time query")) {
        return false;
    }

    if (*resources.hostInfo != 0) {
        if (*resources.hostInfo > 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n",
                   *resources.hostInfo - 1);
        } else {
            fprintf(stderr, "cuSOLVER received an invalid argument at position %d\n",
                    -*resources.hostInfo);
        }
        return false;
    }

    if (!checkCuda(cudaMemcpyAsync(A.data(), resources.matrix, matrixBytes, cudaMemcpyDeviceToHost,
                                   resources.stream),
                   "factor download") ||
        !checkCuda(cudaStreamSynchronize(resources.stream), "factor download completion")) {
        return false;
    }

    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    float elapsedMilliseconds = 0.0F;
    bool success = choleskyDecomposition(A, n, elapsedMilliseconds);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    const double elapsedSeconds = std::max(static_cast<double>(elapsedMilliseconds) / 1000.0,
                                           std::numeric_limits<double>::min());
    double gflops = ops / elapsedSeconds / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
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
