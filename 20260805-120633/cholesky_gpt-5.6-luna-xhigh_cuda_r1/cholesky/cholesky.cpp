#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>

#include "../common/results_output.hpp"

namespace {

bool checkCuda(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    printf("CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool checkCuSolver(const cusolverStatus_t status, const char* operation) {
    if (status == CUSOLVER_STATUS_SUCCESS) {
        return true;
    }
    printf("cuSOLVER error during %s (status %d)\n", operation,
           static_cast<int>(status));
    return false;
}

// A is row-major, while cuSOLVER's dense API is column-major.  The input is
// symmetric, so asking cuSOLVER for the upper factor places the transposed
// factor in the row-major lower triangle without a matrix transpose/copy.
__global__ void zeroUpperTriangle(double* const matrix, const int n) {
    const int row = static_cast<int>(blockIdx.x);
    const int col = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
    if (row < n && col < n && col > row) {
        matrix[static_cast<size_t>(row) * static_cast<size_t>(n) +
               static_cast<size_t>(col)] = 0.0;
    }
}

}  // namespace

// GPU-parallel Cholesky decomposition.  The cuSOLVER dense factorization is
// blocked internally and launches parallel CUDA kernels for the factorization
// and trailing updates.  A is retained as a row-major host-side vector.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(INT_MAX)) {
        printf("Error: Matrix size is too large for the CUDA dense solver\n");
        return false;
    }

    const int dimension = static_cast<int>(n);
    const size_t bytes = n * n * sizeof(double);
    double* deviceMatrix = nullptr;
    double* deviceWorkspace = nullptr;
    int* deviceInfo = nullptr;
    int workspaceElements = 0;
    cusolverDnHandle_t solver = nullptr;
    bool success = false;

    do {
        if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceMatrix), bytes),
                       "allocating the device matrix")) {
            break;
        }
        if (!checkCuda(cudaMemcpy(deviceMatrix, A.data(), bytes,
                                  cudaMemcpyHostToDevice),
                       "copying the matrix to the device")) {
            break;
        }
        if (!checkCuSolver(cusolverDnCreate(&solver), "creating the solver handle")) {
            break;
        }
        if (!checkCuSolver(cusolverDnDpotrf_bufferSize(
                solver, CUBLAS_FILL_MODE_UPPER, dimension, deviceMatrix,
                dimension, &workspaceElements),
                           "querying the factorization workspace")) {
            break;
        }
        if (workspaceElements > 0 &&
            !checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceWorkspace),
                                  static_cast<size_t>(workspaceElements) *
                                      sizeof(double)),
                       "allocating the factorization workspace")) {
            break;
        }
        if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceInfo), sizeof(int)),
                       "allocating the solver status")) {
            break;
        }
        if (!checkCuSolver(cusolverDnDpotrf(
                solver, CUBLAS_FILL_MODE_UPPER, dimension, deviceMatrix,
                dimension, deviceWorkspace, workspaceElements, deviceInfo),
                           "computing the Cholesky factorization")) {
            break;
        }

        // cuSOLVER writes the upper factor in column-major addressing, which
        // is the desired lower factor in this row-major host representation.
        const dim3 block(256, 1, 1);
        const dim3 grid(static_cast<unsigned int>(dimension),
                        (static_cast<unsigned int>(dimension) + block.x - 1) /
                            block.x,
                        1);
        zeroUpperTriangle<<<grid, block>>>(deviceMatrix, dimension);
        if (!checkCuda(cudaGetLastError(), "launching the triangular cleanup")) {
            break;
        }
        if (!checkCuda(cudaDeviceSynchronize(), "finishing the GPU computation")) {
            break;
        }

        int info = 0;
        if (!checkCuda(cudaMemcpy(&info, deviceInfo, sizeof(info),
                                  cudaMemcpyDeviceToHost),
                       "reading the solver status")) {
            break;
        }
        if (info != 0) {
            if (info > 0) {
                printf("Error: Matrix is not positive definite at diagonal element %d\n",
                       info - 1);
            } else {
                printf("Error: cuSOLVER rejected argument %d\n", -info);
            }
            break;
        }
        if (!checkCuda(cudaMemcpy(A.data(), deviceMatrix, bytes,
                                  cudaMemcpyDeviceToHost),
                       "copying the factor back to the host")) {
            break;
        }
        success = true;
    } while (false);

    if (solver != nullptr) {
        cusolverDnDestroy(solver);
    }
    if (deviceInfo != nullptr) {
        cudaFree(deviceInfo);
    }
    if (deviceWorkspace != nullptr) {
        cudaFree(deviceWorkspace);
    }
    if (deviceMatrix != nullptr) {
        cudaFree(deviceMatrix);
    }
    return success;
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
    // Initialize the CUDA context before timing so one-time driver startup
    // does not dominate the factorization benchmark.
    if (!checkCuda(cudaFree(nullptr), "initializing the CUDA context")) {
        return 1;
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double durationMs =
        std::chrono::duration<double, std::milli>(end - start).count();
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = durationMs > 0.0 ? ops / (durationMs / 1000.0) / 1e9 : 0.0;
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
