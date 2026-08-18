#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int kPanelSize = 16;
constexpr int kCleanupBlockSize = 16;

bool checkCuda(const cudaError_t status, const char* const operation) {
    if (status == cudaSuccess) {
        return true;
    }
    printf("CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool checkCublas(const cublasStatus_t status, const char* const operation) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }
    printf("cuBLAS error during %s (status %d)\n", operation, static_cast<int>(status));
    return false;
}

// The matrix is row-major.  Its row-major storage is a column-major view of
// the transpose, so CUDA/cuBLAS use their upper triangle.  This is exactly
// the lower triangle of the caller's matrix, where the Cholesky factor lives.
__global__ void zeroUpperTriangle(double* const matrix, const int n) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && column < n && row < column) {
        matrix[static_cast<size_t>(row) * n + column] = 0.0;
    }
}

// A diagonal panel is small, sequentially dependent work.  One CUDA block
// keeps the dependencies on-device while all rows below each diagonal entry
// are computed in parallel.  Avoiding a host round trip per column is
// important for the small and medium matrix sizes common to this benchmark.
__global__ void factorDiagonalPanel(double* const matrix, const int leadingDimension,
                                    const int offset, const int panelSize,
                                    int* const panelInfo) {
    __shared__ int factorIsPositive;
    const int thread = threadIdx.x;

    for (int column = 0; column < panelSize; ++column) {
        if (thread == 0) {
            const size_t diagonalIndex =
                static_cast<size_t>(offset + column) * leadingDimension + offset + column;
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                const double value = matrix[static_cast<size_t>(offset + column) *
                                                leadingDimension + offset + k];
                sum += value * value;
            }
            const double remainder = matrix[diagonalIndex] - sum;
            factorIsPositive = remainder > 0.0;
            if (factorIsPositive) {
                matrix[diagonalIndex] = sqrt(remainder);
            } else {
                *panelInfo = column + 1;
            }
        }
        __syncthreads();

        if (!factorIsPositive) {
            return;
        }

        const double diagonal = matrix[static_cast<size_t>(offset + column) *
                                       leadingDimension + offset + column];
        for (int row = column + 1 + thread; row < panelSize; row += blockDim.x) {
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                sum += matrix[static_cast<size_t>(offset + row) * leadingDimension +
                              offset + k] *
                       matrix[static_cast<size_t>(offset + column) * leadingDimension +
                              offset + k];
            }
            matrix[static_cast<size_t>(offset + row) * leadingDimension + offset + column] =
                (matrix[static_cast<size_t>(offset + row) * leadingDimension + offset +
                        column] -
                 sum) /
                diagonal;
        }
        __syncthreads();
    }
}

struct CublasHandle {
    cublasHandle_t value = nullptr;

    ~CublasHandle() {
        if (value != nullptr) {
            cublasDestroy(value);
        }
    }
};

struct CudaEvent {
    cudaEvent_t value = nullptr;

    ~CudaEvent() {
        if (value != nullptr) {
            cudaEventDestroy(value);
        }
    }
};

template <typename T>
struct DeviceAllocation {
    T* value = nullptr;

    ~DeviceAllocation() {
        if (value != nullptr) {
            cudaFree(value);
        }
    }
};

}  // namespace

// Blocked CUDA Cholesky decomposition.  The panel factorization, triangular
// solve, and trailing matrix update are all GPU operations.  The only
// host-side loop is over dependent panels, which is required by Cholesky.
bool choleskyDecomposition(std::vector<double>& A, const size_t n,
                           float& elapsedMilliseconds) {
    elapsedMilliseconds = 0.0F;
    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        printf("Error: Matrix dimension is too large for CUDA linear algebra routines\n");
        return false;
    }

    const int dimension = static_cast<int>(n);
    const int panelCount = (dimension + kPanelSize - 1) / kPanelSize;
    const size_t matrixElements = n * n;

    DeviceAllocation<double> deviceMatrix;
    DeviceAllocation<double> warmupStorage;
    DeviceAllocation<int> deviceInfo;
    CublasHandle cublas;
    CudaEvent startEvent;
    CudaEvent stopEvent;

    if (!checkCuda(cudaMalloc(&deviceMatrix.value, matrixElements * sizeof(double)),
                   "matrix allocation") ||
        !checkCuda(cudaMalloc(&warmupStorage.value,
                              2ULL * kPanelSize * kPanelSize * sizeof(double)),
                   "warmup allocation") ||
        !checkCuda(cudaMalloc(&deviceInfo.value, panelCount * sizeof(int)),
                   "solver-info allocation") ||
        !checkCublas(cublasCreate(&cublas.value), "handle creation")) {
        return false;
    }

    // Load the cuBLAS kernels before timing.  CUDA's first library call may
    // initialize module state lazily; charging that one-time setup to a small
    // Cholesky factorization would obscure the actual parallel computation.
    std::vector<double> warmupValues(2ULL * kPanelSize * kPanelSize, 0.0);
    for (int i = 0; i < kPanelSize; ++i) {
        warmupValues[static_cast<size_t>(i) * kPanelSize + i] = 1.0;
    }
    std::fill(warmupValues.begin() + static_cast<size_t>(kPanelSize) * kPanelSize,
              warmupValues.end(), 1.0);
    const double minusOne = -1.0;
    const double one = 1.0;
    if (!checkCuda(cudaMemcpy(warmupStorage.value, warmupValues.data(),
                              warmupValues.size() * sizeof(double),
                              cudaMemcpyHostToDevice),
                   "warmup upload") ||
        !checkCublas(cublasDtrsm(cublas.value, CUBLAS_SIDE_LEFT,
                                 CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                 CUBLAS_DIAG_NON_UNIT, kPanelSize, kPanelSize, &one,
                                 warmupStorage.value, kPanelSize,
                                 warmupStorage.value +
                                     static_cast<size_t>(kPanelSize) * kPanelSize,
                                 kPanelSize),
                     "triangular-solve warmup") ||
        !checkCublas(cublasDgemm(cublas.value, CUBLAS_OP_T, CUBLAS_OP_N,
                                 kPanelSize, kPanelSize, kPanelSize, &minusOne,
                                 warmupStorage.value +
                                     static_cast<size_t>(kPanelSize) * kPanelSize,
                                 kPanelSize,
                                 warmupStorage.value +
                                     static_cast<size_t>(kPanelSize) * kPanelSize,
                                 kPanelSize, &one, warmupStorage.value, kPanelSize),
                     "matrix-multiply warmup") ||
        !checkCuda(cudaDeviceSynchronize(), "library warmup synchronization") ||
        !checkCuda(cudaMemcpy(deviceMatrix.value, A.data(),
                              matrixElements * sizeof(double),
                              cudaMemcpyHostToDevice),
                   "matrix upload") ||
        !checkCuda(cudaMemset(deviceInfo.value, 0, panelCount * sizeof(int)),
                   "solver-info initialization") ||
        !checkCuda(cudaEventCreate(&startEvent.value), "start-event creation") ||
        !checkCuda(cudaEventCreate(&stopEvent.value), "stop-event creation") ||
        !checkCuda(cudaEventRecord(startEvent.value), "start-event recording")) {
        return false;
    }

    for (int panel = 0; panel < panelCount; ++panel) {
        const int offset = panel * kPanelSize;
        const int panelSize = std::min(kPanelSize, dimension - offset);
        double* const diagonal = deviceMatrix.value +
                                 static_cast<size_t>(offset) * dimension + offset;

        factorDiagonalPanel<<<1, 256>>>(deviceMatrix.value, dimension, offset,
                                         panelSize, deviceInfo.value + panel);
        if (!checkCuda(cudaGetLastError(), "diagonal-panel factorization launch")) {
            return false;
        }

        const int trailingSize = dimension - offset - panelSize;
        if (trailingSize == 0) {
            continue;
        }

        // In the column-major transposed view this is
        // L_kk * A_ik^T = A_ik^T, equivalent to A_ik * L_kk^-T.
        double* const panelBelow = deviceMatrix.value +
                                   static_cast<size_t>(offset + panelSize) * dimension + offset;
        if (!checkCublas(cublasDtrsm(
                             cublas.value, CUBLAS_SIDE_LEFT,
                             CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                             CUBLAS_DIAG_NON_UNIT, panelSize, trailingSize,
                             &one, diagonal, dimension, panelBelow, dimension),
                         "panel triangular solve") ||
            // A full GEMM updates both triangles.  Besides keeping the
            // row-major matrix symmetric for the next panel, GEMM maps to
            // the GPU's most highly tuned matrix-multiply path.
            !checkCublas(cublasDgemm(
                             cublas.value, CUBLAS_OP_T, CUBLAS_OP_N,
                             trailingSize, trailingSize, panelSize, &minusOne,
                             panelBelow, dimension, panelBelow, dimension, &one,
                             deviceMatrix.value +
                                 static_cast<size_t>(offset + panelSize) * dimension +
                                 offset + panelSize,
                             dimension),
                         "trailing rank-k update")) {
            return false;
        }
    }

    const dim3 block(kCleanupBlockSize, kCleanupBlockSize);
    const dim3 grid((dimension + kCleanupBlockSize - 1) / kCleanupBlockSize,
                    (dimension + kCleanupBlockSize - 1) / kCleanupBlockSize);
    zeroUpperTriangle<<<grid, block>>>(deviceMatrix.value, dimension);
    if (!checkCuda(cudaGetLastError(), "upper-triangle cleanup launch") ||
        !checkCuda(cudaEventRecord(stopEvent.value), "stop-event recording") ||
        !checkCuda(cudaEventSynchronize(stopEvent.value), "computation synchronization") ||
        !checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, startEvent.value,
                                        stopEvent.value),
                   "elapsed-time measurement")) {
        return false;
    }

    std::vector<int> hostInfo(panelCount);
    if (!checkCuda(cudaMemcpy(hostInfo.data(), deviceInfo.value,
                              panelCount * sizeof(int), cudaMemcpyDeviceToHost),
                   "solver-info download")) {
        return false;
    }
    for (int panel = 0; panel < panelCount; ++panel) {
        if (hostInfo[panel] > 0) {
            const size_t failedDiagonal =
                static_cast<size_t>(panel) * kPanelSize + hostInfo[panel] - 1;
            printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                   failedDiagonal);
            return false;
        }
        if (hostInfo[panel] < 0) {
            printf("CUDA received an invalid argument during diagonal-panel factorization\n");
            return false;
        }
    }

    return checkCuda(cudaMemcpy(A.data(), deviceMatrix.value,
                                matrixElements * sizeof(double),
                                cudaMemcpyDeviceToHost),
                     "factor download");
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
    
    // Compute A = B * B^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
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
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    float elapsedMilliseconds = 0.0F;
    const bool success = choleskyDecomposition(A, n, elapsedMilliseconds);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (elapsedMilliseconds / 1000.0) / 1e9;
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
