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

static bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

static bool checkSolver(cusolverStatus_t status, const char* operation) {
    if (status == CUSOLVER_STATUS_SUCCESS) return true;
    fprintf(stderr, "%s: cuSOLVER error %d\n", operation, static_cast<int>(status));
    return false;
}

static bool checkBlas(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    fprintf(stderr, "%s: cuBLAS error %d\n", operation, static_cast<int>(status));
    return false;
}

__global__ void clearUpperTriangle(double* A, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row) A[static_cast<size_t>(row) * n + col] = 0.0;
}

__global__ void completeSymmetricMatrix(double* A, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;
    if (col > row) A[static_cast<size_t>(row) * n + col] = A[static_cast<size_t>(col) * n + row];
    if (col == row) A[static_cast<size_t>(row) * n + col] += n;
}

struct CholeskyGpu {
    double* deviceA = nullptr;
    double* workspace = nullptr;
    int* deviceInfo = nullptr;
    cusolverDnHandle_t solver = nullptr;
    int workspaceSize = 0;
    int order = 0;

    ~CholeskyGpu() {
        if (workspace) cudaFree(workspace);
        if (solver) cusolverDnDestroy(solver);
        if (deviceInfo) cudaFree(deviceInfo);
        if (deviceA) cudaFree(deviceA);
    }

    bool prepare(size_t n) {
        if (n == 0) return true;
        if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
            fprintf(stderr, "Matrix size exceeds cuSOLVER's integer limit\n");
            return false;
        }
        order = static_cast<int>(n);
        return checkCuda(cudaMalloc(&deviceA, n * n * sizeof(double)), "Allocate GPU matrix") &&
               checkCuda(cudaMalloc(&deviceInfo, sizeof(int)), "Allocate GPU status") &&
               checkSolver(cusolverDnCreate(&solver), "Create cuSOLVER handle") &&
               checkSolver(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER,
                                                        order, deviceA, order, &workspaceSize),
                           "Query Cholesky workspace") &&
               checkCuda(cudaMalloc(&workspace, static_cast<size_t>(workspaceSize) * sizeof(double)),
                         "Allocate Cholesky workspace");
    }
};

// cuSOLVER uses column-major storage. Its upper-triangular result is exactly
// the lower-triangular result in our row-major storage.
bool choleskyDecomposition(std::vector<double>& A, const size_t n, CholeskyGpu& gpu) {
    if (n == 0) return true;
    int info = 0;
    if (!checkSolver(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_UPPER, gpu.order,
                                      gpu.deviceA, gpu.order, gpu.workspace,
                                      gpu.workspaceSize, gpu.deviceInfo), "Factor matrix") ||
        !checkCuda(cudaMemcpy(&info, gpu.deviceInfo, sizeof(int), cudaMemcpyDeviceToHost),
                   "Read Cholesky status")) return false;

    if (info > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        return false;
    }
    if (info < 0) {
        fprintf(stderr, "cuSOLVER rejected argument %d\n", -info);
        return false;
    }

    {
        const dim3 threads(32, 8);
        const dim3 blocks((gpu.order + 31) / 32, (gpu.order + 7) / 8);
        clearUpperTriangle<<<blocks, threads>>>(gpu.deviceA, gpu.order);
        if (!checkCuda(cudaGetLastError(), "Clear upper triangle") ||
            !checkCuda(cudaMemcpy(A.data(), gpu.deviceA, A.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost), "Copy result from GPU")) return false;
    }
    return true;
}

// Generate a symmetric positive definite matrix
bool generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n,
                                    CholeskyGpu& gpu, bool copyToHost) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    if (n == 0) return true;
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "Matrix size exceeds cuBLAS's integer limit\n");
        return false;
    }

    double* deviceB = nullptr;
    cublasHandle_t blas = nullptr;
    const int order = static_cast<int>(n);
    const double alpha = 1.0;
    const double beta = 0.0;
    bool ok = false;
    if (!checkCuda(cudaMalloc(&deviceB, B.size() * sizeof(double)), "Allocate GPU input") ||
        !checkBlas(cublasCreate(&blas), "Create cuBLAS handle") ||
        !checkCuda(cudaMemcpy(deviceB, B.data(), B.size() * sizeof(double),
                              cudaMemcpyHostToDevice), "Copy input to GPU") ||
        !checkBlas(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                               order, order, &alpha, deviceB, order,
                               &beta, gpu.deviceA, order), "Generate positive definite matrix")) goto generationCleanup;

    {
        const dim3 threads(32, 8);
        const dim3 blocks((order + 31) / 32, (order + 7) / 8);
        completeSymmetricMatrix<<<blocks, threads>>>(gpu.deviceA, order);
        if (!checkCuda(cudaGetLastError(), "Complete symmetric matrix")) goto generationCleanup;
        if (copyToHost && !checkCuda(cudaMemcpy(A.data(), gpu.deviceA, A.size() * sizeof(double),
                                               cudaMemcpyDeviceToHost),
                                     "Copy generated matrix from GPU")) goto generationCleanup;
    }
    ok = true;

generationCleanup:
    if (blas) cublasDestroy(blas);
    if (deviceB) cudaFree(deviceB);
    return ok;
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
    CholeskyGpu gpu;
    if (!gpu.prepare(n)) return 1;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    if (!generatePositiveDefiniteMatrix(A, n, gpu, validate)) return 1;
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, gpu);
    
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
