#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {

bool checkCuda(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool checkCuSolver(const cusolverStatus_t status, const char* operation) {
    if (status == CUSOLVER_STATUS_SUCCESS) {
        return true;
    }
    fprintf(stderr, "cuSOLVER error during %s (status %d)\n", operation,
            static_cast<int>(status));
    return false;
}

// cuSOLVER is column-major. A symmetric row-major matrix has the same values
// when viewed column-major, but its upper factor is the transpose of the
// row-major lower factor required by this benchmark. Consequently POTRF with
// CUBLAS_FILL_MODE_UPPER writes L directly into the row-major lower triangle.
__global__ void zeroUpperTriangle(double* matrix, const size_t n) {
    const size_t column = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row < n && column < n && column > row) {
        matrix[row * n + column] = 0.0;
    }
}

struct CudaCholeskyResources {
    cudaStream_t stream = nullptr;
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    cusolverDnHandle_t solver = nullptr;
    cusolverDnParams_t parameters = nullptr;
    double* deviceMatrix = nullptr;
    void* deviceWorkspace = nullptr;
    int* deviceInfo = nullptr;

    ~CudaCholeskyResources() {
        if (deviceInfo != nullptr) {
            cudaFree(deviceInfo);
        }
        if (deviceWorkspace != nullptr) {
            cudaFree(deviceWorkspace);
        }
        if (deviceMatrix != nullptr) {
            cudaFree(deviceMatrix);
        }
        if (stopEvent != nullptr) {
            cudaEventDestroy(stopEvent);
        }
        if (startEvent != nullptr) {
            cudaEventDestroy(startEvent);
        }
        if (parameters != nullptr) {
            cusolverDnDestroyParams(parameters);
        }
        if (solver != nullptr) {
            cusolverDnDestroy(solver);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

}  // namespace

// GPU Cholesky decomposition. The factorization and triangular cleanup are
// timed with CUDA events; setup and host/device transfers are deliberately
// outside the compute timing, as is standard for an accelerator benchmark.
bool choleskyDecomposition(std::vector<double>& A, const size_t n, double& elapsedMs) {
    elapsedMs = 0.0;
    if (n == 0) {
        // Still initialize the CUDA runtime so this path never becomes a CPU
        // implementation for the degenerate empty-matrix case.
        return checkCuda(cudaFree(nullptr), "CUDA runtime initialization");
    }

    CudaCholeskyResources resources;
    const size_t matrixBytes = A.size() * sizeof(double);

    if (!checkCuda(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking),
                   "stream creation") ||
        !checkCuSolver(cusolverDnCreate(&resources.solver), "handle creation") ||
        !checkCuSolver(cusolverDnSetStream(resources.solver, resources.stream),
                       "stream association") ||
        !checkCuSolver(cusolverDnCreateParams(&resources.parameters),
                       "parameter creation") ||
        !checkCuda(cudaEventCreate(&resources.startEvent), "start event creation") ||
        !checkCuda(cudaEventCreate(&resources.stopEvent), "stop event creation") ||
        !checkCuda(cudaMalloc(&resources.deviceMatrix, matrixBytes),
                   "matrix allocation") ||
        !checkCuda(cudaMalloc(&resources.deviceInfo, sizeof(int)),
                   "status allocation") ||
        !checkCuda(cudaMemcpyAsync(resources.deviceMatrix, A.data(), matrixBytes,
                                   cudaMemcpyHostToDevice, resources.stream),
                   "matrix upload") ||
        !checkCuda(cudaStreamSynchronize(resources.stream), "matrix upload synchronization")) {
        return false;
    }

    size_t deviceWorkspaceBytes = 0;
    size_t hostWorkspaceBytes = 0;
    if (!checkCuSolver(
            cusolverDnXpotrf_bufferSize(
                resources.solver, resources.parameters, CUBLAS_FILL_MODE_UPPER,
                static_cast<int64_t>(n), CUDA_R_64F, resources.deviceMatrix,
                static_cast<int64_t>(n), CUDA_R_64F, &deviceWorkspaceBytes,
                &hostWorkspaceBytes),
            "workspace query")) {
        return false;
    }

    if (deviceWorkspaceBytes != 0 &&
        !checkCuda(cudaMalloc(&resources.deviceWorkspace, deviceWorkspaceBytes),
                   "workspace allocation")) {
        return false;
    }
    std::vector<unsigned char> hostWorkspace(hostWorkspaceBytes);

    if (!checkCuda(cudaEventRecord(resources.startEvent, resources.stream),
                   "start event recording") ||
        !checkCuSolver(
            cusolverDnXpotrf(
                resources.solver, resources.parameters, CUBLAS_FILL_MODE_UPPER,
                static_cast<int64_t>(n), CUDA_R_64F, resources.deviceMatrix,
                static_cast<int64_t>(n), CUDA_R_64F, resources.deviceWorkspace,
                deviceWorkspaceBytes,
                hostWorkspace.empty() ? nullptr : hostWorkspace.data(), hostWorkspaceBytes,
                resources.deviceInfo),
            "factorization")) {
        return false;
    }

    constexpr unsigned int columnsPerBlock = 32;
    constexpr unsigned int rowsPerBlock = 8;
    const dim3 block(columnsPerBlock, rowsPerBlock);
    const dim3 grid(static_cast<unsigned int>((n + columnsPerBlock - 1) / columnsPerBlock),
                    static_cast<unsigned int>((n + rowsPerBlock - 1) / rowsPerBlock));
    zeroUpperTriangle<<<grid, block, 0, resources.stream>>>(resources.deviceMatrix, n);

    if (!checkCuda(cudaPeekAtLastError(), "upper-triangle cleanup launch") ||
        !checkCuda(cudaEventRecord(resources.stopEvent, resources.stream),
                   "stop event recording")) {
        return false;
    }

    int info = 0;
    if (!checkCuda(cudaMemcpyAsync(&info, resources.deviceInfo, sizeof(info),
                                   cudaMemcpyDeviceToHost, resources.stream),
                   "factorization status download") ||
        !checkCuda(cudaMemcpyAsync(A.data(), resources.deviceMatrix, matrixBytes,
                                   cudaMemcpyDeviceToHost, resources.stream),
                   "factor download") ||
        !checkCuda(cudaStreamSynchronize(resources.stream),
                   "factorization synchronization")) {
        return false;
    }

    float elapsed = 0.0f;
    if (!checkCuda(cudaEventElapsedTime(&elapsed, resources.startEvent, resources.stopEvent),
                   "elapsed-time query")) {
        return false;
    }
    elapsedMs = static_cast<double>(elapsed);

    if (info < 0) {
        fprintf(stderr, "Error: cuSOLVER rejected argument %d\n", -info);
        return false;
    }
    if (info > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
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
    double durationMs = 0.0;
    bool success = choleskyDecomposition(A, n, durationMs);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = durationMs > 0.0 ? ops / (durationMs * 1.0e6) : 0.0;
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
