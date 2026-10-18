#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}

void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error " + std::to_string(static_cast<int>(status)));
}

void check(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cuSOLVER error " + std::to_string(static_cast<int>(status)));
}

template <typename T> class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// One stream orders all library operations and kernels without per-step host waits.
struct CudaContext {
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    CudaContext() {
        try {
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            check(cublasCreate(&blas));
            check(cusolverDnCreate(&solver));
            check(cublasSetStream(blas, stream));
            check(cusolverDnSetStream(solver, stream));
        } catch (...) {
            if (solver) cusolverDnDestroy(solver);
            if (blas) cublasDestroy(blas);
            if (stream) cudaStreamDestroy(stream);
            throw;
        }
    }
    ~CudaContext() {
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
    }
};

CudaContext& context() {
    static CudaContext ctx;
    return ctx;
}

__global__ void addDiagonal(double* A, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += size_t(blockDim.x) * gridDim.x)
        A[i * n + i] += static_cast<double>(n);
}

__global__ void clearUpper(double* A, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n * n; i += size_t(blockDim.x) * gridDim.x)
        if (i % n > i / n) A[i] = 0.0;
}

unsigned int blocks(size_t count) {
    return static_cast<unsigned int>(std::min(size_t(65535), (count + 255) / 256));
}
} // namespace

// Row-major lower L is column-major upper U, with A = U^T U.
// cuSOLVER performs a blocked FP64 factorization with parallel BLAS-3 updates.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    auto& ctx = context();
    if (n == 0) return true;
    const int dim = static_cast<int>(n);
    const size_t bytes = A.size() * sizeof(double);
    DeviceBuffer<double> matrix(A.size());
    DeviceBuffer<int> info(1);
    check(cudaMemcpyAsync(matrix.data, A.data(), bytes, cudaMemcpyHostToDevice, ctx.stream));
    int workspaceSize = 0;
    check(cusolverDnDpotrf_bufferSize(ctx.solver, CUBLAS_FILL_MODE_UPPER,
                                    dim, matrix.data, dim, &workspaceSize));
    DeviceBuffer<double> workspace(workspaceSize);
    check(cusolverDnDpotrf(ctx.solver, CUBLAS_FILL_MODE_UPPER, dim,
                          matrix.data, dim, workspace.data, workspaceSize, info.data));
    int result = 0;
    check(cudaMemcpyAsync(&result, info.data, sizeof(result), cudaMemcpyDeviceToHost, ctx.stream));
    check(cudaStreamSynchronize(ctx.stream));
    if (result > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", result - 1);
        return false;
    }
    if (result < 0) throw std::runtime_error("Invalid cuSOLVER factorization argument");
    clearUpper<<<blocks(A.size()), 256, 0, ctx.stream>>>(matrix.data, n);
    check(cudaGetLastError());
    check(cudaMemcpyAsync(A.data(), matrix.data, bytes, cudaMemcpyDeviceToHost, ctx.stream));
    check(cudaStreamSynchronize(ctx.stream));
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
    
    auto& ctx = context();
    if (n == 0) return;
    const int dim = static_cast<int>(n);
    const size_t bytes = A.size() * sizeof(double);
    DeviceBuffer<double> input(B.size()), output(A.size());
    check(cudaMemcpyAsync(input.data, B.data(), bytes, cudaMemcpyHostToDevice, ctx.stream));
    const double one = 1.0, zero = 0.0;
    // Interpreting row-major B as column-major B^T avoids transposition.
    check(cublasDgemm(ctx.blas, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, dim,
                      &one, input.data, dim, input.data, dim, &zero, output.data, dim));
    addDiagonal<<<blocks(n), 256, 0, ctx.stream>>>(output.data, n);
    check(cudaGetLastError());
    check(cudaMemcpyAsync(A.data(), output.data, bytes, cudaMemcpyDeviceToHost, ctx.stream));
    check(cudaStreamSynchronize(ctx.stream));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    if (n != 0) {
        auto& ctx = context();
        const int dim = static_cast<int>(n);
        const size_t bytes = L.size() * sizeof(double);
        DeviceBuffer<double> input(L.size()), output(L.size());
        check(cudaMemcpyAsync(input.data, L.data(), bytes, cudaMemcpyHostToDevice, ctx.stream));
        const double one = 1.0, zero = 0.0;
        check(cublasDgemm(ctx.blas, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, dim,
                          &one, input.data, dim, input.data, dim, &zero, output.data, dim));
        check(cudaMemcpyAsync(reconstructed.data(), output.data, bytes,
                              cudaMemcpyDeviceToHost, ctx.stream));
        check(cudaStreamSynchronize(ctx.stream));
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        if (!std::isfinite(error)) {
            printf("Validation failed: non-finite result\n");
            return false;
        }
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

int main(int argc, char** argv) try {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value == end || *end || *value == '-' || parsed > INT_MAX ||
                (parsed != 0 && parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed)) {
                fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            n = static_cast<size_t>(parsed);
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
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
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
} catch (const std::exception& error) {
    fprintf(stderr, "Error: %s\n", error.what());
    return 1;
}
