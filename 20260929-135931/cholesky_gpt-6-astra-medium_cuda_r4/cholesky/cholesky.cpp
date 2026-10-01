#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// CUDA toolkit libraries provide blocked, GPU-parallel double-precision kernels.
// There is deliberately no CPU fallback.
void checkCuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
void checkBlas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error " + std::to_string(status));
}
void checkSolver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cuSOLVER error " + std::to_string(status));
}

template<class T> class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

class GpuContext {
public:
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    GpuContext() {
        checkBlas(cublasCreate(&blas));
        const auto status = cusolverDnCreate(&solver);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            cublasDestroy(blas);
            checkSolver(status);
        }
    }
    ~GpuContext() {
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
    }
    GpuContext(const GpuContext&) = delete;
    GpuContext& operator=(const GpuContext&) = delete;
};
GpuContext& gpu() {
    static GpuContext context;
    return context;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    auto& context = gpu();
    DeviceBuffer<double> matrix(n * n);
    DeviceBuffer<int> info(1);
    const size_t bytes = n * n * sizeof(double);
    checkCuda(cudaMemcpy(matrix.data, A.data(), bytes, cudaMemcpyHostToDevice));
    int workspaceSize = 0;
    // Column-major upper is row-major lower. This also preserves the original
    // convention of reading only the lower triangle of the input matrix.
    checkSolver(cusolverDnDpotrf_bufferSize(context.solver, CUBLAS_FILL_MODE_UPPER,
                static_cast<int>(n), matrix.data, static_cast<int>(n), &workspaceSize));
    DeviceBuffer<double> workspace(std::max(workspaceSize, 1));
    checkSolver(cusolverDnDpotrf(context.solver, CUBLAS_FILL_MODE_UPPER,
                static_cast<int>(n), matrix.data, static_cast<int>(n),
                workspace.data, workspaceSize, info.data));
    int result = 0;
    // The blocking copy waits for the factorization and reports numerical failure.
    checkCuda(cudaMemcpy(&result, info.data, sizeof(result), cudaMemcpyDeviceToHost));
    if (result > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", result - 1);
        return false;
    }
    if (result < 0) throw std::runtime_error("Invalid cuSOLVER factorization argument");
    checkCuda(cudaMemcpy(A.data(), matrix.data, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i)
        std::fill(A.begin() + i * n + i + 1, A.begin() + (i + 1) * n, 0.0);
    return true;
}

// Compute a row-major X * X^T with a GPU matrix multiplication. In column-major
// notation the same buffer holds X^T, hence the transpose on the first operand.
void gramMatrix(const std::vector<double>& X, std::vector<double>& product, size_t n) {
    if (n == 0) return;
    auto& context = gpu();
    DeviceBuffer<double> input(n * n), output(n * n);
    const size_t bytes = n * n * sizeof(double);
    checkCuda(cudaMemcpy(input.data, X.data(), bytes, cudaMemcpyHostToDevice));
    const double one = 1.0, zero = 0.0;
    checkBlas(cublasDgemm(context.blas, CUBLAS_OP_T, CUBLAS_OP_N,
              static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
              &one, input.data, static_cast<int>(n), input.data, static_cast<int>(n),
              &zero, output.data, static_cast<int>(n)));
    checkCuda(cudaMemcpy(product.data(), output.data, bytes, cudaMemcpyDeviceToHost));
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    // Keep the original random sequence exactly, independent of GPU scheduling.
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    gramMatrix(B, A, n);
    for (size_t i = 0; i < n; ++i) A[i * n + i] += n;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    gramMatrix(L, reconstructed, n);
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        if (!std::isfinite(error)) {
            printf("Validation failed: non-finite result\n");
            return false;
        }
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (fabs(A_orig[i]) + 1e-10));
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

int main(int argc, char** argv) try {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            const unsigned long long value = strtoull(argument, &end, 10);
            if (*argument == '-' || end == argument || *end != '\0' ||
                value > INT_MAX || (value != 0 &&
                value > std::numeric_limits<size_t>::max() / sizeof(double) / value)) {
                fprintf(stderr, "Invalid matrix size: %s\n", argument);
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
    fprintf(stderr, "Cholesky benchmark failed: %s\n", error.what());
    return 1;
}
