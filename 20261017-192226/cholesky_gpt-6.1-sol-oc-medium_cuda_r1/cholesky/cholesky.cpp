#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        throw std::runtime_error("cuBLAS error " + std::to_string(status));
}

void check(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cuSOLVER error " + std::to_string(status));
}

// Reuse handles and a single stream throughout the benchmark. All GPU work is
// ordered on this stream; host results are exposed only after synchronization.
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
            release();
            throw;
        }
    }
    void release() noexcept {
        if (solver) cusolverDnDestroy(solver);
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }
    ~CudaContext() { release(); }
    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;
};

CudaContext& context() {
    static CudaContext ctx;
    return ctx;
}

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// cuBLAS/cuSOLVER see the transpose of our row-major matrices. Consequently,
// their upper triangle is our lower triangle, with no transpose/copy required.
__global__ void finishMatrix(double* a, size_t n, bool factor) {
    const size_t count = n * n;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += size_t(blockDim.x) * gridDim.x) {
        const size_t row = index / n;
        const size_t col = index % n;
        if (row < col) a[index] = factor ? 0.0 : a[col * n + row];
        else if (!factor && row == col) a[index] += double(n);
    }
}

unsigned int gridSize(size_t count) {
    return static_cast<unsigned int>(std::min(size_t(65535), (count + 255) / 256));
}

} // namespace

// Decomposes A into L * L^T, using the toolkit's blocked, GPU-parallel FP64
// factorization. There is deliberately no sequential fallback.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    auto& ctx = context();
    if (n == 0) return true;
    const int dim = static_cast<int>(n);
    DeviceBuffer<double> matrix(n * n);
    DeviceBuffer<int> info(1);
    check(cudaMemcpyAsync(matrix.data, A.data(), A.size() * sizeof(double),
                          cudaMemcpyHostToDevice, ctx.stream));
    int workspaceSize = 0;
    check(cusolverDnDpotrf_bufferSize(ctx.solver, CUBLAS_FILL_MODE_UPPER,
                                    dim, matrix.data, dim, &workspaceSize));
    DeviceBuffer<double> workspace(workspaceSize);
    check(cusolverDnDpotrf(ctx.solver, CUBLAS_FILL_MODE_UPPER, dim,
                         matrix.data, dim, workspace.data, workspaceSize, info.data));
    finishMatrix<<<gridSize(n * n), 256, 0, ctx.stream>>>(matrix.data, n, true);
    check(cudaGetLastError());
    int hostInfo = 0;
    check(cudaMemcpyAsync(&hostInfo, info.data, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
    check(cudaMemcpyAsync(A.data(), matrix.data, A.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, ctx.stream));
    check(cudaStreamSynchronize(ctx.stream));
    if (hostInfo > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", hostInfo - 1);
        return false;
    }
    if (hostInfo < 0) throw std::runtime_error("Invalid cuSOLVER factorization argument");
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
    DeviceBuffer<double> input(n * n), matrix(n * n);
    check(cudaMemcpyAsync(input.data, B.data(), B.size() * sizeof(double),
                          cudaMemcpyHostToDevice, ctx.stream));
    const double one = 1.0, zero = 0.0;
    // Only compute one triangle, then mirror it and add n * I on the GPU.
    check(cublasDsyrk(ctx.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                     dim, dim, &one, input.data, dim, &zero, matrix.data, dim));
    finishMatrix<<<gridSize(n * n), 256, 0, ctx.stream>>>(matrix.data, n, false);
    check(cudaGetLastError());
    check(cudaMemcpyAsync(A.data(), matrix.data, A.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, ctx.stream));
    check(cudaStreamSynchronize(ctx.stream));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    if (n != 0) {
        auto& ctx = context();
        const int dim = static_cast<int>(n);
        DeviceBuffer<double> factor(n * n), product(n * n);
        check(cudaMemcpyAsync(factor.data, L.data(), L.size() * sizeof(double),
                              cudaMemcpyHostToDevice, ctx.stream));
        const double one = 1.0, zero = 0.0;
        check(cublasDgemm(ctx.blas, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, dim,
                         &one, factor.data, dim, factor.data, dim,
                         &zero, product.data, dim));
        check(cudaMemcpyAsync(reconstructed.data(), product.data,
                              reconstructed.size() * sizeof(double),
                              cudaMemcpyDeviceToHost, ctx.stream));
        check(cudaStreamSynchronize(ctx.stream));
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    for (size_t i = 0; i < n * n; ++i) {
        if (!std::isfinite(reconstructed[i]) || !std::isfinite(A_orig[i])) {
            printf("Validation failed: non-finite matrix element\n");
            return false;
        }
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

int main(int argc, char** argv) try {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* text = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = strtoull(text, &end, 10);
            if (text[0] == '-' || end == text || *end != '\0' || errno != 0 ||
                value > static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
                (value != 0 && value > std::numeric_limits<size_t>::max() / sizeof(double) / value)) {
                fprintf(stderr, "Invalid matrix size: %s\n", text);
                return 1;
            }
            n = static_cast<size_t>(value);
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
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    const double seconds = std::chrono::duration<double>(end - start).count();
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
