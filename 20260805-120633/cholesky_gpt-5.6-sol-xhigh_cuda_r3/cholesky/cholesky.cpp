#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {

bool checkCuda(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool checkCusolver(const cusolverStatus_t status, const char* operation) {
    if (status == CUSOLVER_STATUS_SUCCESS) {
        return true;
    }
    fprintf(stderr, "cuSOLVER error during %s (status %d)\n", operation,
            static_cast<int>(status));
    return false;
}

bool checkCublas(const cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }
    fprintf(stderr, "cuBLAS error during %s (status %d)\n", operation,
            static_cast<int>(status));
    return false;
}

class CudaCholeskyContext {
public:
    CudaCholeskyContext() = default;
    CudaCholeskyContext(const CudaCholeskyContext&) = delete;
    CudaCholeskyContext& operator=(const CudaCholeskyContext&) = delete;

    ~CudaCholeskyContext() {
        if (solver_ != nullptr) {
            checkCusolver(cusolverDnDestroy(solver_), "handle destruction");
        }
        if (blas_ != nullptr) {
            checkCublas(cublasDestroy(blas_), "handle destruction");
        }
        if (stream_ != nullptr) {
            checkCuda(cudaStreamDestroy(stream_), "stream destruction");
        }
    }

    bool initialize() {
        if (!checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                       "stream creation")) {
            return false;
        }
        if (!checkCublas(cublasCreate(&blas_), "handle creation")) {
            return false;
        }
        if (!checkCublas(cublasSetStream(blas_, stream_), "stream setup")) {
            return false;
        }
        if (!checkCusolver(cusolverDnCreate(&solver_), "handle creation")) {
            return false;
        }
        return checkCusolver(cusolverDnSetStream(solver_, stream_), "stream setup");
    }

    cusolverDnHandle_t solver() const { return solver_; }
    cublasHandle_t blas() const { return blas_; }
    cudaStream_t stream() const { return stream_; }

private:
    cusolverDnHandle_t solver_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

__global__ void addDiagonal(double* matrix, const size_t n, const double value) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = first; i < n; i += stride) {
        matrix[i * n + i] += value;
    }
}

// cuSOLVER uses column-major matrices. A symmetric row-major matrix has the
// same byte representation, and its column-major upper factor is the desired
// row-major lower factor. The other triangle is not touched by potrf, so clear
// it in parallel to retain the original program's output representation.
__global__ void zeroRowMajorUpper(double* matrix, const size_t n) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t elements = n * n;

    for (size_t index = first; index < elements; index += stride) {
        const size_t row = index / n;
        const size_t column = index - row * n;
        if (column > row) {
            matrix[index] = 0.0;
        }
    }
}

}  // namespace

// GPU-blocked Cholesky decomposition. cuSOLVER dispatches the panel updates
// and matrix multiplications to highly parallel CUDA kernels.
bool choleskyDecomposition(std::vector<double>& A, const size_t n,
                           const CudaCholeskyContext& context) {
    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Matrix dimension is too large for cuSOLVER\n");
        return false;
    }

    const int dimension = static_cast<int>(n);
    const size_t matrixBytes = A.size() * sizeof(double);
    double* deviceMatrix = nullptr;
    double* workspace = nullptr;
    int* deviceInfo = nullptr;
    int workspaceElements = 0;
    int info = 0;
    bool ok = true;

    ok = checkCuda(cudaMalloc(&deviceMatrix, matrixBytes), "matrix allocation");
    if (ok) {
        ok = checkCuda(cudaMalloc(&deviceInfo, sizeof(int)), "status allocation");
    }
    if (ok) {
        ok = checkCuda(cudaMemcpyAsync(deviceMatrix, A.data(), matrixBytes,
                                       cudaMemcpyHostToDevice, context.stream()),
                       "matrix upload");
    }
    if (ok) {
        ok = checkCusolver(
            cusolverDnDpotrf_bufferSize(context.solver(), CUBLAS_FILL_MODE_UPPER, dimension,
                                        deviceMatrix, dimension, &workspaceElements),
            "workspace query");
    }
    if (ok && workspaceElements > 0) {
        ok = checkCuda(cudaMalloc(&workspace,
                                  static_cast<size_t>(workspaceElements) * sizeof(double)),
                       "workspace allocation");
    }
    if (ok) {
        ok = checkCusolver(
            cusolverDnDpotrf(context.solver(), CUBLAS_FILL_MODE_UPPER, dimension, deviceMatrix,
                             dimension, workspace, workspaceElements, deviceInfo),
            "Cholesky factorization");
    }
    if (ok) {
        constexpr unsigned int threads = 256;
        const size_t blocksNeeded = (A.size() + threads - 1) / threads;
        const unsigned int blocks = static_cast<unsigned int>(
            std::min<size_t>(blocksNeeded, 65535));
        zeroRowMajorUpper<<<blocks, threads, 0, context.stream()>>>(deviceMatrix, n);
        ok = checkCuda(cudaGetLastError(), "upper-triangle kernel launch");
    }
    if (ok) {
        ok = checkCuda(cudaMemcpyAsync(A.data(), deviceMatrix, matrixBytes,
                                       cudaMemcpyDeviceToHost, context.stream()),
                       "factor download");
    }
    if (ok) {
        ok = checkCuda(cudaMemcpyAsync(&info, deviceInfo, sizeof(int),
                                       cudaMemcpyDeviceToHost, context.stream()),
                       "status download");
    }
    if (ok) {
        ok = checkCuda(cudaStreamSynchronize(context.stream()),
                       "factorization synchronization");
    }

    if (workspace != nullptr) {
        checkCuda(cudaFree(workspace), "workspace release");
    }
    if (deviceInfo != nullptr) {
        checkCuda(cudaFree(deviceInfo), "status release");
    }
    if (deviceMatrix != nullptr) {
        checkCuda(cudaFree(deviceMatrix), "matrix release");
    }
    if (!ok) {
        return false;
    }
    if (info < 0) {
        fprintf(stderr, "cuSOLVER reported an invalid argument at position %d\n", -info);
        return false;
    }
    if (info > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        return false;
    }
    return true;
}

// Generate a symmetric positive definite matrix. Random-number generation is
// kept identical to the original, while the cubic B * B^T operation runs on
// the GPU.
bool generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n,
                                    const CudaCholeskyContext& context) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Matrix dimension is too large for cuBLAS\n");
        return false;
    }

    const int dimension = static_cast<int>(n);
    const size_t matrixBytes = A.size() * sizeof(double);
    double* deviceB = nullptr;
    double* deviceA = nullptr;
    bool ok = checkCuda(cudaMalloc(&deviceB, matrixBytes), "generator input allocation");
    if (ok) {
        ok = checkCuda(cudaMalloc(&deviceA, matrixBytes), "generator output allocation");
    }
    if (ok) {
        ok = checkCuda(cudaMemcpyAsync(deviceB, B.data(), matrixBytes,
                                       cudaMemcpyHostToDevice, context.stream()),
                       "generator input upload");
    }
    if (ok) {
        const double alpha = 1.0;
        const double beta = 0.0;
        // Row-major B is column-major B^T in the same storage. Thus B^T * B
        // in the column-major view is B * B^T in the row-major view.
        ok = checkCublas(
            cublasDgemm(context.blas(), CUBLAS_OP_T, CUBLAS_OP_N, dimension,
                        dimension, dimension, &alpha, deviceB, dimension, deviceB,
                        dimension, &beta, deviceA, dimension),
            "positive-definite matrix multiplication");
    }
    if (ok) {
        constexpr unsigned int threads = 256;
        const unsigned int blocks = static_cast<unsigned int>(
            std::min<size_t>((n + threads - 1) / threads, 65535));
        addDiagonal<<<blocks, threads, 0, context.stream()>>>(
            deviceA, n, static_cast<double>(n));
        ok = checkCuda(cudaGetLastError(), "diagonal kernel launch");
    }
    if (ok) {
        ok = checkCuda(cudaMemcpyAsync(A.data(), deviceA, matrixBytes,
                                       cudaMemcpyDeviceToHost, context.stream()),
                       "generated matrix download");
    }
    if (ok) {
        ok = checkCuda(cudaStreamSynchronize(context.stream()),
                       "matrix generation synchronization");
    }

    if (deviceA != nullptr) {
        checkCuda(cudaFree(deviceA), "generator output release");
    }
    if (deviceB != nullptr) {
        checkCuda(cudaFree(deviceB), "generator input release");
    }
    return ok;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig,
                      const size_t n, const CudaCholeskyContext& context) {
    // Validate by computing L * L^T on the GPU and comparing with the original.
    std::vector<double> reconstructed(n * n);

    if (n > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Matrix dimension is too large for cuBLAS validation\n");
        return false;
    }
    if (n != 0) {
        const int dimension = static_cast<int>(n);
        const size_t matrixBytes = L.size() * sizeof(double);
        double* deviceL = nullptr;
        double* deviceReconstructed = nullptr;
        bool ok = checkCuda(cudaMalloc(&deviceL, matrixBytes),
                            "validation input allocation");
        if (ok) {
            ok = checkCuda(cudaMalloc(&deviceReconstructed, matrixBytes),
                           "validation output allocation");
        }
        if (ok) {
            ok = checkCuda(cudaMemcpyAsync(deviceL, L.data(), matrixBytes,
                                           cudaMemcpyHostToDevice, context.stream()),
                           "validation input upload");
        }
        if (ok) {
            const double alpha = 1.0;
            const double beta = 0.0;
            ok = checkCublas(
                cublasDgemm(context.blas(), CUBLAS_OP_T, CUBLAS_OP_N, dimension,
                            dimension, dimension, &alpha, deviceL, dimension, deviceL,
                            dimension, &beta, deviceReconstructed, dimension),
                "validation matrix multiplication");
        }
        if (ok) {
            ok = checkCuda(cudaMemcpyAsync(reconstructed.data(), deviceReconstructed,
                                           matrixBytes, cudaMemcpyDeviceToHost,
                                           context.stream()),
                           "validation output download");
        }
        if (ok) {
            ok = checkCuda(cudaStreamSynchronize(context.stream()),
                           "validation synchronization");
        }
        if (deviceReconstructed != nullptr) {
            checkCuda(cudaFree(deviceReconstructed), "validation output release");
        }
        if (deviceL != nullptr) {
            checkCuda(cudaFree(deviceL), "validation input release");
        }
        if (!ok) {
            return false;
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

    CudaCholeskyContext cudaContext;
    if (!cudaContext.initialize()) {
        printf("CUDA initialization failed\n");
        return 1;
    }

    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    if (!generatePositiveDefiniteMatrix(A, n, cudaContext)) {
        printf("Matrix generation failed\n");
        return 1;
    }
    
    if (validate) {
        A_orig = A; // Save original for validation
    }

    // CUDA context and library initialization are one-time process setup, not
    // part of the factorization being benchmarked. Transfers to and from the
    // GPU remain inside the timed decomposition.
    constexpr size_t warmupSize = 32;
    std::vector<double> warmup(warmupSize * warmupSize, 0.0);
    for (size_t i = 0; i < warmupSize; ++i) {
        warmup[i * warmupSize + i] = 1.0;
    }
    if (!choleskyDecomposition(warmup, warmupSize, cudaContext)) {
        printf("CUDA warm-up failed\n");
        return 1;
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, cudaContext);
    
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
    double gflops = ops / (durationMs / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n, cudaContext);
        
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
