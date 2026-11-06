#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// CUDA libraries interpret row-major matrices as their column-major transpose.
// In particular, the row-major lower triangle is the column-major upper triangle.
static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

static void blasCheck(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuBLAS error %d\n", operation, static_cast<int>(status));
        exit(EXIT_FAILURE);
    }
}

static void solverCheck(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuSOLVER error %d\n", operation, static_cast<int>(status));
        exit(EXIT_FAILURE);
    }
}

class CudaContext {
public:
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    CudaContext() {
        blasCheck(cublasCreate(&blas), "cublasCreate");
        solverCheck(cusolverDnCreate(&solver), "cusolverDnCreate");
    }
    ~CudaContext() {
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
    }
};

static CudaContext& context() {
    static CudaContext ctx;
    return ctx;
}

template <typename T> class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

static int matrixSize(size_t n) {
    if (n > static_cast<size_t>(INT_MAX) || (n && n > SIZE_MAX / sizeof(double) / n)) {
        fprintf(stderr, "Matrix size is too large for CUDA\n");
        exit(EXIT_FAILURE);
    }
    return static_cast<int>(n);
}

__global__ void addDiagonal(double* A, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        A[i * n + i] += static_cast<double>(n);
    }
}

__global__ void clearUpper(double* A, size_t n) {
    for (size_t pos = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pos < n * n; pos += static_cast<size_t>(blockDim.x) * gridDim.x) {
        if (pos % n > pos / n) A[pos] = 0.0;
    }
}

static unsigned int blocksFor(size_t count) {
    return static_cast<unsigned int>(std::min<size_t>((count + 255) / 256, 65535));
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const int dim = matrixSize(n);
    if (!n) return true;
    CudaContext& ctx = context();
    DeviceBuffer<double> dA(n * n);
    DeviceBuffer<int> info(1);
    cudaCheck(cudaMemcpy(dA.data, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice), "copy matrix to GPU");
    int workspaceSize = 0;
    solverCheck(cusolverDnDpotrf_bufferSize(ctx.solver, CUBLAS_FILL_MODE_UPPER, dim,
                                            dA.data, dim, &workspaceSize), "potrf workspace query");
    DeviceBuffer<double> workspace(static_cast<size_t>(workspaceSize));
    solverCheck(cusolverDnDpotrf(ctx.solver, CUBLAS_FILL_MODE_UPPER, dim, dA.data, dim,
                                 workspace.data, workspaceSize, info.data), "potrf");
    int result = 0;
    cudaCheck(cudaMemcpy(&result, info.data, sizeof(result), cudaMemcpyDeviceToHost), "copy potrf status");
    if (result > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", result - 1);
        return false;
    }
    if (result < 0) {
        fprintf(stderr, "potrf: invalid argument %d\n", -result);
        return false;
    }
    clearUpper<<<blocksFor(n * n), 256>>>(dA.data, n);
    cudaCheck(cudaGetLastError(), "clear upper triangle");
    cudaCheck(cudaMemcpy(A.data(), dA.data, n * n * sizeof(double), cudaMemcpyDeviceToHost), "copy factor to host");
    return true;
}

// Generate the same seeded B as the original, then compute B * B^T on the GPU.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    const int dim = matrixSize(n);
    if (!n) return;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    CudaContext& ctx = context();
    DeviceBuffer<double> dB(n * n), dA(n * n);
    cudaCheck(cudaMemcpy(dB.data, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice), "copy random matrix to GPU");
    const double alpha = 1.0, beta = 0.0;
    blasCheck(cublasDgemm(ctx.blas, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, dim,
                          &alpha, dB.data, dim, dB.data, dim, &beta, dA.data, dim), "generate matrix");
    addDiagonal<<<blocksFor(n), 256>>>(dA.data, n);
    cudaCheck(cudaGetLastError(), "add diagonal");
    cudaCheck(cudaMemcpy(A.data(), dA.data, n * n * sizeof(double), cudaMemcpyDeviceToHost), "copy generated matrix to host");
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    const int dim = matrixSize(n);
    std::vector<double> reconstructed(n * n);
    if (n) {
        CudaContext& ctx = context();
        DeviceBuffer<double> dL(n * n), dResult(n * n);
        cudaCheck(cudaMemcpy(dL.data, L.data(), n * n * sizeof(double), cudaMemcpyHostToDevice), "copy factor to GPU");
        const double alpha = 1.0, beta = 0.0;
        blasCheck(cublasDgemm(ctx.blas, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, dim,
                              &alpha, dL.data, dim, dL.data, dim, &beta, dResult.data, dim), "reconstruct matrix");
        cudaCheck(cudaMemcpy(reconstructed.data(), dResult.data, n * n * sizeof(double), cudaMemcpyDeviceToHost), "copy reconstructed matrix to host");
    }
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
    
    // Check the dimensions before multiplying them for allocation.
    matrixSize(n);
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
