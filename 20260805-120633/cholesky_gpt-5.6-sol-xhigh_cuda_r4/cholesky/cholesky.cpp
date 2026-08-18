#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {

const char* cusolverErrorString(const cusolverStatus_t status) {
    switch (status) {
        case CUSOLVER_STATUS_SUCCESS: return "success";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "not initialized";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "allocation failed";
        case CUSOLVER_STATUS_INVALID_VALUE: return "invalid value";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "architecture mismatch";
        case CUSOLVER_STATUS_MAPPING_ERROR: return "mapping error";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "execution failed";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "internal error";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "matrix type not supported";
        default: return "unknown error";
    }
}

const char* cublasErrorString(const cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "success";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED: return "allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "architecture mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR: return "mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "internal error";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "operation not supported";
        default: return "unknown error";
    }
}

bool checkCuda(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    printf("CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool checkCusolver(const cusolverStatus_t status, const char* operation) {
    if (status == CUSOLVER_STATUS_SUCCESS) {
        return true;
    }
    printf("cuSOLVER error during %s: %s\n", operation, cusolverErrorString(status));
    return false;
}

bool checkCublas(const cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }
    printf("cuBLAS error during %s: %s\n", operation, cublasErrorString(status));
    return false;
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    bool allocate(const size_t count, const char* description) {
        if (count == 0) {
            return true;
        }
        return checkCuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), description);
    }

    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() = default;
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    bool create() { return checkCuda(cudaEventCreate(&event_), "event creation"); }
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

// cuSOLVER uses column-major matrices. A symmetric row-major matrix has the same
// byte representation when viewed column-major. Requesting an upper factor U
// therefore leaves U^T -- the desired lower factor L -- in row-major storage.
__global__ void clearRowMajorUpperTriangle(double* const matrix, const int n) {
    const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row < n && column < n && column > row) {
        matrix[static_cast<size_t>(row) * n + column] = 0.0;
    }
}

__global__ void addDiagonalShift(double* const matrix, const int n,
                                 const double shift) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < n) {
        matrix[static_cast<size_t>(index) * n + index] += shift;
    }
}

bool initializeCuda() {
    int device = 0;
    if (!checkCuda(cudaGetDevice(&device), "device selection")) {
        return false;
    }

    cudaDeviceProp properties{};
    if (!checkCuda(cudaGetDeviceProperties(&properties, device), "device query")) {
        return false;
    }

    // Create the CUDA context before the timed region so one-time driver setup is
    // not mistaken for factorization work.
    if (!checkCuda(cudaFree(nullptr), "CUDA context initialization")) {
        return false;
    }
    printf("CUDA device: %s (compute capability %d.%d)\n",
           properties.name, properties.major, properties.minor);
    return true;
}

// Compute output = input * input^T. As with the factorization, row-major input
// is viewed as its column-major transpose, making M^T*M the required row Gram
// matrix. This replaces the cubic serial setup and validation loops.
bool computeRowGramMatrix(const std::vector<double>& input,
                          std::vector<double>& output,
                          const size_t n,
                          const double diagonalShift) {
    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(INT_MAX) ||
        n > std::numeric_limits<size_t>::max() / n) {
        printf("Error: Matrix dimension exceeds the CUDA BLAS limit\n");
        return false;
    }

    const int dimension = static_cast<int>(n);
    const size_t elementCount = n * n;
    const size_t byteCount = elementCount * sizeof(double);
    cublasHandle_t blas = nullptr;
    cudaStream_t stream = nullptr;
    DeviceBuffer<double> deviceInput;
    DeviceBuffer<double> deviceOutput;

    bool ok = checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                        "Gram-matrix stream creation");
    ok = ok && checkCublas(cublasCreate(&blas), "BLAS handle creation");
    ok = ok && checkCublas(cublasSetStream(blas, stream), "BLAS stream assignment");
    ok = ok && deviceInput.allocate(elementCount, "Gram-matrix input allocation");
    ok = ok && deviceOutput.allocate(elementCount, "Gram-matrix output allocation");
    if (ok) {
        ok = checkCuda(cudaMemcpyAsync(deviceInput.get(), input.data(), byteCount,
                                       cudaMemcpyHostToDevice, stream),
                       "Gram-matrix input upload");
    }

    const double alpha = 1.0;
    const double beta = 0.0;
    if (ok) {
        ok = checkCublas(
            cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        dimension, dimension, dimension,
                        &alpha, deviceInput.get(), dimension,
                        deviceInput.get(), dimension,
                        &beta, deviceOutput.get(), dimension),
            "Gram-matrix multiplication");
    }
    if (ok && diagonalShift != 0.0) {
        constexpr unsigned int threads = 256;
        const unsigned int blocks =
            (static_cast<unsigned int>(dimension) + threads - 1) / threads;
        addDiagonalShift<<<blocks, threads, 0, stream>>>(
            deviceOutput.get(), dimension, diagonalShift);
        ok = checkCuda(cudaGetLastError(), "diagonal-shift kernel launch");
    }
    if (ok) {
        ok = checkCuda(cudaMemcpyAsync(output.data(), deviceOutput.get(), byteCount,
                                       cudaMemcpyDeviceToHost, stream),
                       "Gram-matrix output download");
    }
    if (ok) {
        ok = checkCuda(cudaStreamSynchronize(stream),
                       "Gram-matrix computation synchronization");
    }

    if (blas != nullptr) {
        const cublasStatus_t status = cublasDestroy(blas);
        if (ok) {
            ok = checkCublas(status, "BLAS handle destruction");
        }
    }
    if (stream != nullptr) {
        const cudaError_t status = cudaStreamDestroy(stream);
        if (ok) {
            ok = checkCuda(status, "Gram-matrix stream destruction");
        }
    }
    return ok;
}

} // namespace

// GPU Cholesky decomposition using cuSOLVER's blocked, parallel DPOTRF. The
// result is copied back into A with the same row-major lower-triangular layout as
// the original implementation. Timings cover all GPU factorization work and the
// parallel clearing of the unused triangle, but not one-time setup or transfers.
bool choleskyDecomposition(std::vector<double>& A, const size_t n, float& elapsedMs) {
    elapsedMs = 0.0F;
    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(INT_MAX)) {
        printf("Error: Matrix dimension exceeds the CUDA solver limit\n");
        return false;
    }
    if (n > std::numeric_limits<size_t>::max() / n ||
        n * n > std::numeric_limits<size_t>::max() / sizeof(double)) {
        printf("Error: Matrix size overflows addressable memory\n");
        return false;
    }

    const int dimension = static_cast<int>(n);
    const size_t elementCount = n * n;
    const size_t byteCount = elementCount * sizeof(double);

    cusolverDnHandle_t solver = nullptr;
    cudaStream_t stream = nullptr;
    DeviceBuffer<double> deviceMatrix;
    DeviceBuffer<double> workspace;
    DeviceBuffer<int> deviceInfo;
    CudaEvent start;
    CudaEvent stop;

    bool ok = checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                        "stream creation");
    ok = ok && checkCusolver(cusolverDnCreate(&solver), "solver creation");
    ok = ok && checkCusolver(cusolverDnSetStream(solver, stream), "stream assignment");
    ok = ok && deviceMatrix.allocate(elementCount, "device matrix allocation");
    ok = ok && deviceInfo.allocate(1, "solver status allocation");
    ok = ok && start.create() && stop.create();

    int workspaceElements = 0;
    if (ok) {
        ok = checkCuda(cudaMemcpyAsync(deviceMatrix.get(), A.data(), byteCount,
                                       cudaMemcpyHostToDevice, stream),
                       "matrix upload");
    }
    if (ok) {
        ok = checkCusolver(
            cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, dimension,
                                        deviceMatrix.get(), dimension, &workspaceElements),
            "workspace sizing");
    }
    if (ok) {
        ok = workspace.allocate(static_cast<size_t>(workspaceElements),
                                "solver workspace allocation");
    }

    if (ok) {
        ok = checkCuda(cudaEventRecord(start.get(), stream), "start event recording");
    }
    if (ok) {
        ok = checkCusolver(
            cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, dimension,
                             deviceMatrix.get(), dimension, workspace.get(),
                             workspaceElements, deviceInfo.get()),
            "Cholesky factorization");
    }
    if (ok) {
        constexpr unsigned int blockX = 32;
        constexpr unsigned int blockY = 8;
        const dim3 block(blockX, blockY);
        const dim3 grid((dimension + blockX - 1) / blockX,
                        (dimension + blockY - 1) / blockY);
        clearRowMajorUpperTriangle<<<grid, block, 0, stream>>>(deviceMatrix.get(), dimension);
        ok = checkCuda(cudaGetLastError(), "upper-triangle kernel launch");
    }
    if (ok) {
        ok = checkCuda(cudaEventRecord(stop.get(), stream), "stop event recording");
    }
    if (ok) {
        ok = checkCuda(cudaEventSynchronize(stop.get()), "factorization synchronization");
    }
    if (ok) {
        ok = checkCuda(cudaEventElapsedTime(&elapsedMs, start.get(), stop.get()),
                       "elapsed-time query");
    }

    int factorizationInfo = 0;
    if (ok) {
        ok = checkCuda(cudaMemcpy(&factorizationInfo, deviceInfo.get(), sizeof(int),
                                  cudaMemcpyDeviceToHost),
                       "solver status download");
    }
    if (ok && factorizationInfo < 0) {
        printf("Error: cuSOLVER rejected argument %d\n", -factorizationInfo);
        ok = false;
    } else if (ok && factorizationInfo > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n",
               factorizationInfo - 1);
        ok = false;
    }

    if (ok) {
        ok = checkCuda(cudaMemcpyAsync(A.data(), deviceMatrix.get(), byteCount,
                                       cudaMemcpyDeviceToHost, stream),
                       "factor download");
    }
    if (ok) {
        ok = checkCuda(cudaStreamSynchronize(stream), "factor download synchronization");
    }

    if (solver != nullptr) {
        const cusolverStatus_t status = cusolverDnDestroy(solver);
        if (ok) {
            ok = checkCusolver(status, "solver destruction");
        }
    }
    if (stream != nullptr) {
        const cudaError_t status = cudaStreamDestroy(stream);
        if (ok) {
            ok = checkCuda(status, "stream destruction");
        }
    }
    return ok;
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
    
    // Compute A = B * B^T and add diagonal dominance on the GPU.
    return computeRowGramMatrix(B, A, n, static_cast<double>(n));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T on the GPU.
    if (!computeRowGramMatrix(L, reconstructed, n, 0.0)) {
        printf("Validation failed: CUDA reconstruction failed\n");
        return false;
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

    if (!initializeCuda()) {
        printf("CUDA initialization failed\n");
        return 1;
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    if (!generatePositiveDefiniteMatrix(A, n)) {
        printf("Matrix generation failed\n");
        return 1;
    }
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    float durationMs = 0.0F;
    const bool success = choleskyDecomposition(A, n, durationMs);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", static_cast<double>(durationMs));
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    const double seconds = static_cast<double>(durationMs) / 1000.0;
    const double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
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
