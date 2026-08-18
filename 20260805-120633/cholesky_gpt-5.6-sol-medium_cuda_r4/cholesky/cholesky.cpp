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

bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                 cudaGetErrorString(status));
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

bool checkCublas(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "cuBLAS error during %s (status %d)\n", operation,
                 static_cast<int>(status));
    return false;
}

struct SolverContext {
    cusolverDnHandle_t handle = nullptr;
    cublasHandle_t blas = nullptr;

    ~SolverContext() {
        if (blas) cublasDestroy(blas);
        if (handle) cusolverDnDestroy(handle);
    }
};

SolverContext& solverContext() {
    static SolverContext context;
    return context;
}

bool initializeGpuSolver() {
    if (!checkCuda(cudaFree(nullptr), "CUDA initialization")) return false;
    SolverContext& context = solverContext();
    if (!context.handle &&
        !checkCusolver(cusolverDnCreate(&context.handle), "solver creation")) {
        return false;
    }
    if (!context.blas &&
        !checkCublas(cublasCreate(&context.blas), "BLAS handle creation")) {
        return false;
    }
    return true;
}

// cuSOLVER is column-major.  The input is symmetric, so it has the same byte
// representation in either layout.  Factoring its column-major upper triangle
// places U = L^T in memory exactly where row-major L belongs.
__global__ void zeroRowMajorUpper(double* matrix, int n) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && column < n && column > row) {
        matrix[static_cast<size_t>(row) * n + column] = 0.0;
    }
}

__global__ void addDiagonal(double* matrix, int n, double value) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        matrix[static_cast<size_t>(index) * n + index] += value;
    }
}

}  // namespace

// GPU Cholesky decomposition backed by NVIDIA's highly tuned blocked solver.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "Matrix size must be in the range [1, %d]\n",
                     std::numeric_limits<int>::max());
        return false;
    }

    const int order = static_cast<int>(n);
    const size_t bytes = n * n * sizeof(double);
    cusolverDnHandle_t solver = nullptr;
    double* deviceMatrix = nullptr;
    double* workspace = nullptr;
    int* deviceInfo = nullptr;
    int workspaceElements = 0;
    int info = 0;
    bool ok = true;

    ok = initializeGpuSolver();
    if (ok) solver = solverContext().handle;
    if (ok) ok = checkCuda(cudaMalloc(&deviceMatrix, bytes), "matrix allocation");
    if (ok) ok = checkCuda(cudaMalloc(&deviceInfo, sizeof(int)), "status allocation");
    if (ok) ok = checkCuda(cudaMemcpy(deviceMatrix, A.data(), bytes,
                                      cudaMemcpyHostToDevice), "matrix upload");
    if (ok) {
        ok = checkCusolver(cusolverDnDpotrf_bufferSize(
                               solver, CUBLAS_FILL_MODE_UPPER, order,
                               deviceMatrix, order, &workspaceElements),
                           "workspace query");
    }
    if (ok && workspaceElements > 0) {
        ok = checkCuda(cudaMalloc(&workspace,
                                  static_cast<size_t>(workspaceElements) * sizeof(double)),
                       "workspace allocation");
    }
    if (ok) {
        ok = checkCusolver(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER,
                                           order, deviceMatrix, order, workspace,
                                           workspaceElements, deviceInfo),
                           "Cholesky factorization");
    }

    if (ok) {
        const dim3 threads(32, 8);
        const dim3 blocks((order + threads.x - 1) / threads.x,
                          (order + threads.y - 1) / threads.y);
        zeroRowMajorUpper<<<blocks, threads>>>(deviceMatrix, order);
        ok = checkCuda(cudaGetLastError(), "upper-triangle kernel launch");
    }
    if (ok) {
        ok = checkCuda(cudaMemcpy(&info, deviceInfo, sizeof(int),
                                  cudaMemcpyDeviceToHost), "solver status download");
    }
    if (ok && info == 0) {
        ok = checkCuda(cudaMemcpy(A.data(), deviceMatrix, bytes,
                                  cudaMemcpyDeviceToHost), "factor download");
    }

    if (workspace) cudaFree(workspace);
    if (deviceInfo) cudaFree(deviceInfo);
    if (deviceMatrix) cudaFree(deviceMatrix);

    if (!ok) return false;
    if (info < 0) {
        std::fprintf(stderr, "cuSOLVER rejected argument %d\n", -info);
        return false;
    }
    if (info > 0) {
        std::printf("Error: Matrix is not positive definite at diagonal element %d\n",
                    info - 1);
        return false;
    }
    return true;
}

// Generate a symmetric positive definite matrix
bool generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    if (!initializeGpuSolver()) return false;

    const int order = static_cast<int>(n);
    const size_t bytes = n * n * sizeof(double);
    double* deviceB = nullptr;
    double* deviceA = nullptr;
    bool ok = checkCuda(cudaMalloc(&deviceB, bytes), "generator input allocation");
    if (ok) ok = checkCuda(cudaMalloc(&deviceA, bytes), "generator output allocation");
    if (ok) {
        ok = checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice),
                       "generator input upload");
    }
    if (ok) {
        const double alpha = 1.0;
        const double beta = 0.0;
        // Row-major B is column-major B^T in the same storage.  Computing
        // B_col^T * B_col therefore produces row-major B * B^T directly.
        ok = checkCublas(cublasDgemm(solverContext().blas,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    order, order, order, &alpha,
                                    deviceB, order, deviceB, order,
                                    &beta, deviceA, order),
                         "positive-definite matrix multiplication");
    }
    if (ok) {
        constexpr int threads = 256;
        addDiagonal<<<(order + threads - 1) / threads, threads>>>(
            deviceA, order, static_cast<double>(n));
        ok = checkCuda(cudaGetLastError(), "diagonal kernel launch");
    }
    if (ok) {
        ok = checkCuda(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost),
                       "generated matrix download");
    }
    if (deviceA) cudaFree(deviceA);
    if (deviceB) cudaFree(deviceB);
    return ok;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    const int order = static_cast<int>(n);
    const size_t bytes = n * n * sizeof(double);
    double* deviceL = nullptr;
    double* deviceReconstructed = nullptr;
    bool ok = initializeGpuSolver();
    if (ok) ok = checkCuda(cudaMalloc(&deviceL, bytes), "validation input allocation");
    if (ok) ok = checkCuda(cudaMalloc(&deviceReconstructed, bytes),
                           "validation output allocation");
    if (ok) {
        ok = checkCuda(cudaMemcpy(deviceL, L.data(), bytes, cudaMemcpyHostToDevice),
                       "validation input upload");
    }
    if (ok) {
        const double alpha = 1.0;
        const double beta = 0.0;
        ok = checkCublas(cublasDgemm(solverContext().blas,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    order, order, order, &alpha,
                                    deviceL, order, deviceL, order,
                                    &beta, deviceReconstructed, order),
                         "validation matrix multiplication");
    }
    if (ok) {
        ok = checkCuda(cudaMemcpy(reconstructed.data(), deviceReconstructed, bytes,
                                  cudaMemcpyDeviceToHost),
                       "validation result download");
    }
    if (deviceReconstructed) cudaFree(deviceReconstructed);
    if (deviceL) cudaFree(deviceL);
    if (!ok) return false;
    
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

    // Pay the one-time CUDA driver/context startup cost before the benchmark.
    // The timed region still includes allocation, transfers, factorization, and
    // synchronization, but not process-wide GPU initialization.
    if (!initializeGpuSolver()) {
        return 1;
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    if (!generatePositiveDefiniteMatrix(A, n)) {
        return 1;
    }
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
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
