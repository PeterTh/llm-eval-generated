#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// A is row-major and symmetric. Interpreting it as column-major transposes it,
// so cuSOLVER's upper-triangular factor occupies A's lower triangle.
__global__ void clearUpperTriangle(double* matrix, size_t n) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row) {
        matrix[row * n + col] = 0.0;
    }
}

struct SolverResources {
    cusolverDnHandle_t handle = nullptr;
    double* matrix = nullptr;
    double* workspace = nullptr;
    int* info = nullptr;

    ~SolverResources() {
        if (info) cudaFree(info);
        if (workspace) cudaFree(workspace);
        if (matrix) cudaFree(matrix);
        if (handle) cusolverDnDestroy(handle);
    }
};

bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    printf("CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool checkSolver(cusolverStatus_t status, const char* operation) {
    if (status == CUSOLVER_STATUS_SUCCESS) return true;
    printf("cuSOLVER %s failed with status %d\n", operation, static_cast<int>(status));
    return false;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        printf("Matrix size exceeds cuSOLVER's supported dimension\n");
        return false;
    }

    SolverResources resources;
    const size_t bytes = A.size() * sizeof(double);
    if (!checkSolver(cusolverDnCreate(&resources.handle), "handle creation") ||
        !checkCuda(cudaMalloc(reinterpret_cast<void**>(&resources.matrix), bytes), "matrix allocation") ||
        !checkCuda(cudaMalloc(reinterpret_cast<void**>(&resources.info), sizeof(int)), "status allocation") ||
        !checkCuda(cudaMemcpy(resources.matrix, A.data(), bytes, cudaMemcpyHostToDevice), "matrix upload")) {
        return false;
    }

    const int dimension = static_cast<int>(n);
    int workspaceSize = 0;
    if (!checkSolver(cusolverDnDpotrf_bufferSize(resources.handle, CUBLAS_FILL_MODE_UPPER,
                                                dimension, resources.matrix, dimension,
                                                &workspaceSize), "workspace query") ||
        !checkCuda(cudaMalloc(reinterpret_cast<void**>(&resources.workspace),
                              static_cast<size_t>(workspaceSize) * sizeof(double)),
                   "workspace allocation") ||
        !checkSolver(cusolverDnDpotrf(resources.handle, CUBLAS_FILL_MODE_UPPER,
                                      dimension, resources.matrix, dimension,
                                      resources.workspace, workspaceSize, resources.info),
                     "factorization")) {
        return false;
    }

    int info = 0;
    if (!checkCuda(cudaMemcpy(&info, resources.info, sizeof(int), cudaMemcpyDeviceToHost),
                   "factorization status download")) return false;
    if (info != 0) {
        if (info > 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        } else {
            printf("cuSOLVER factorization rejected argument %d\n", -info);
        }
        return false;
    }

    const dim3 threads(16, 16);
    const dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
    clearUpperTriangle<<<blocks, threads>>>(resources.matrix, n);
    if (!checkCuda(cudaGetLastError(), "upper triangle kernel") ||
        !checkCuda(cudaMemcpy(A.data(), resources.matrix, bytes, cudaMemcpyDeviceToHost),
                   "factor download")) return false;

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

    // CUDA context startup is one-time process setup, not factorization work.
    if (!checkCuda(cudaFree(nullptr), "initialization")) return 1;
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
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
