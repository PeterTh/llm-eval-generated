#include <algorithm>
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
template <typename Status>
void check(Status status, const char* operation) {
    if (status != 0)
        throw std::runtime_error(std::string(operation) + " failed (status " +
                                 std::to_string(static_cast<int>(status)) + ")");
}

#define CUDA_CHECK(operation) check((operation), #operation)

template <typename T>
class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct CudaContext {
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    CudaContext() {
        CUDA_CHECK(cublasCreate(&blas));
        const auto status = cusolverDnCreate(&solver);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            cublasDestroy(blas);
            CUDA_CHECK(status);
        }
        // Preserve full FP64 arithmetic; do not enable reduced-precision math.
    }
    ~CudaContext() {
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
    }
};

CudaContext& context() {
    static CudaContext instance;
    return instance;
}

__global__ void addDiagonal(double* a, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
        a[i * n + i] += static_cast<double>(n);
}

__global__ void clearUpper(double* a, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n * n;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
        if (i % n > i / n) a[i] = 0.0;
}

// In column-major notation the row-major input is B^T. Thus B^T * B
// below produces precisely the row-major B * B^T required by the benchmark.
void gramMatrix(const std::vector<double>& input, std::vector<double>& output,
                size_t n, bool addIdentity) {
    auto& ctx = context();
    if (n == 0) return;
    const size_t bytes = n * n * sizeof(double);
    DeviceBuffer<double> b(n * n), a(n * n);
    CUDA_CHECK(cudaMemcpy(b.data, input.data(), bytes, cudaMemcpyHostToDevice));
    const double one = 1.0, zero = 0.0;
    CUDA_CHECK(cublasDgemm(ctx.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                          static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
                          &one, b.data, static_cast<int>(n), b.data, static_cast<int>(n),
                          &zero, a.data, static_cast<int>(n)));
    if (addIdentity) {
        addDiagonal<<<static_cast<unsigned>((n + 255) / 256), 256>>>(a.data, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy(output.data(), a.data, bytes, cudaMemcpyDeviceToHost));
}
} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    auto& ctx = context();
    if (n == 0) return true;
    const int dim = static_cast<int>(n);
    const size_t bytes = n * n * sizeof(double);
    DeviceBuffer<double> a(n * n);
    DeviceBuffer<int> info(1);
    CUDA_CHECK(cudaMemcpy(a.data, A.data(), bytes, cudaMemcpyHostToDevice));
    int workspaceSize = 0;
    // Column-major upper U is row-major lower L, avoiding any transpose.
    CUDA_CHECK(cusolverDnDpotrf_bufferSize(ctx.solver, CUBLAS_FILL_MODE_UPPER,
                                          dim, a.data, dim, &workspaceSize));
    DeviceBuffer<double> workspace(static_cast<size_t>(workspaceSize));
    CUDA_CHECK(cusolverDnDpotrf(ctx.solver, CUBLAS_FILL_MODE_UPPER, dim,
                               a.data, dim, workspace.data, workspaceSize, info.data));
    clearUpper<<<static_cast<unsigned>(std::min<size_t>((n * n + 255) / 256, 65535)), 256>>>(a.data, n);
    CUDA_CHECK(cudaGetLastError());
    int result = 0;
    CUDA_CHECK(cudaMemcpy(&result, info.data, sizeof(result), cudaMemcpyDeviceToHost));
    if (result > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", result - 1);
        return false;
    }
    if (result < 0) throw std::runtime_error("Invalid cuSOLVER Cholesky argument");
    CUDA_CHECK(cudaMemcpy(A.data(), a.data, bytes, cudaMemcpyDeviceToHost));
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
    
    gramMatrix(B, A, n, true);
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    gramMatrix(L, reconstructed, n, false);
    
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
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (*value == '-' || end == value || *end != '\0' ||
                parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
                (parsed && parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed))
                throw std::runtime_error("Invalid matrix size");
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
