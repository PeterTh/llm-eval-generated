#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <cerrno>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// CUDA toolkit libraries provide blocked, GPU-parallel double precision kernels.
// A row-major lower triangle is the column-major upper triangle of A^T.
static void checkCuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
static void checkBlas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error " + std::to_string(status));
}
static void checkSolver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cuSOLVER error " + std::to_string(status));
}

template <typename T> struct DeviceBuffer {
    T* data = nullptr;
    void allocate(size_t count) {
        if (count) checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__global__ void addDiagonal(double* a, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x)
        a[i * n + i] += double(n);
}

__global__ void clearUpper(double* a, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n * n;
         i += size_t(blockDim.x) * gridDim.x)
        if (i % n > i / n) a[i] = 0.0;
}

class CudaCholesky {
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    DeviceBuffer<double> matrix, scratch, workspace;
    DeviceBuffer<int> info;
    int n, workspaceSize = 0;
    size_t elements;

    void releaseHandles() {
        if (solver) cusolverDnDestroy(solver);
        if (blas) cublasDestroy(blas);
    }
public:
    explicit CudaCholesky(size_t size) : n(int(size)), elements(size * size) {
        try {
            checkBlas(cublasCreate(&blas));
            checkSolver(cusolverDnCreate(&solver));
            matrix.allocate(elements);
            scratch.allocate(elements);
            info.allocate(1);
            if (n) {
                checkSolver(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER,
                    n, matrix.data, n, &workspaceSize));
                workspace.allocate(workspaceSize);
            }
        } catch (...) {
            releaseHandles();
            throw;
        }
    }
    ~CudaCholesky() { releaseHandles(); }
    CudaCholesky(const CudaCholesky&) = delete;
    CudaCholesky& operator=(const CudaCholesky&) = delete;

    void generatePositiveDefiniteMatrix(std::vector<double>& a) {
        if (!n) return;
        // Retain rand_r and its exact seed/order to preserve benchmark inputs.
        unsigned int seed = 42;
        for (double& value : a) value = rand_r(&seed) / double(RAND_MAX) - 0.5;
        checkCuda(cudaMemcpy(scratch.data, a.data(), elements * sizeof(double),
                             cudaMemcpyHostToDevice));
        const double one = 1.0, zero = 0.0;
        // Row-major B is column-major B^T, so compute (B^T)^T B^T.
        checkBlas(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
            &one, scratch.data, n, scratch.data, n, &zero, matrix.data, n));
        addDiagonal<<<(n + 255u) / 256, 256>>>(matrix.data, n);
        checkCuda(cudaGetLastError());
        checkCuda(cudaMemcpy(a.data(), matrix.data, elements * sizeof(double),
                             cudaMemcpyDeviceToHost));
    }

    bool choleskyDecomposition(std::vector<double>& a) {
        if (!n) return true;
        checkSolver(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, n,
            matrix.data, n, workspace.data, workspaceSize, info.data));
        int result = 0;
        checkCuda(cudaMemcpy(&result, info.data, sizeof(result), cudaMemcpyDeviceToHost));
        if (result > 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", result - 1);
            return false;
        }
        if (result < 0) throw std::runtime_error("Invalid cuSOLVER factorization argument");
        clearUpper<<<unsigned(std::min<size_t>((elements + 255) / 256, 65535)), 256>>>(matrix.data, n);
        checkCuda(cudaGetLastError());
        // This synchronous copy also ensures the benchmark includes all GPU work.
        checkCuda(cudaMemcpy(a.data(), matrix.data, elements * sizeof(double),
                             cudaMemcpyDeviceToHost));
        return true;
    }

    void reconstruct(std::vector<double>& a) {
        if (!n) return;
        const double one = 1.0, zero = 0.0;
        checkBlas(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
            &one, matrix.data, n, matrix.data, n, &zero, scratch.data, n));
        checkCuda(cudaMemcpy(a.data(), scratch.data, elements * sizeof(double),
                             cudaMemcpyDeviceToHost));
    }
};

bool validateCholesky(CudaCholesky& gpu, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    gpu.reconstruct(reconstructed);

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
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (*value == '-' || end == value || *end || errno || parsed > INT_MAX ||
                (parsed && parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed)) {
                fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            n = size_t(parsed);
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
    CudaCholesky gpu(n);
    gpu.generatePositiveDefiniteMatrix(A);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::steady_clock::now();
    
    bool success = gpu.choleskyDecomposition(A);
    
    auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / seconds / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(gpu, A_orig, n);
        
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
