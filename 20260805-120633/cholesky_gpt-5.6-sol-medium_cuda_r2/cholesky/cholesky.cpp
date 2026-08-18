#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// A right-looking blocked Cholesky factorization.  The small diagonal block is
// dependency limited, while the panel solves and the O(n^3) trailing updates
// run in parallel on the GPU.  A remains row-major and is overwritten by L.
namespace {
constexpr int kBlockSize = 32;
constexpr int kUpdateTile = 16;

__global__ void factorDiagonal(double* __restrict__ A, size_t n, size_t first,
                               int width, int* failure) {
    if (threadIdx.x != 0) return;

    for (int j = 0; j < width; ++j) {
        const size_t row = first + static_cast<size_t>(j);
        double sum = 0.0;
        for (int p = 0; p < j; ++p) {
            const double x = A[row * n + first + static_cast<size_t>(p)];
            sum += x * x;
        }
        const double value = A[row * n + row] - sum;
        if (!(value > 0.0)) {
            atomicMin(failure, static_cast<int>(row));
            return;
        }
        A[row * n + row] = sqrt(value);

        for (int i = j + 1; i < width; ++i) {
            const size_t target = first + static_cast<size_t>(i);
            double dot = 0.0;
            for (int p = 0; p < j; ++p) {
                dot += A[target * n + first + static_cast<size_t>(p)] *
                       A[row * n + first + static_cast<size_t>(p)];
            }
            A[target * n + row] =
                (A[target * n + row] - dot) / A[row * n + row];
        }
    }
}

__global__ void solvePanel(double* __restrict__ A, size_t n, size_t first,
                           int width) {
    const size_t row = first + static_cast<size_t>(width) +
                       blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    // Keeping the current row in registers avoids repeatedly fetching panel
    // values while forward-substituting against the diagonal block.
    double values[kBlockSize];
    for (int j = 0; j < width; ++j) {
        double dot = 0.0;
        for (int p = 0; p < j; ++p) {
            dot += values[p] *
                   A[(first + static_cast<size_t>(j)) * n + first + p];
        }
        values[j] = (A[row * n + first + static_cast<size_t>(j)] - dot) /
                    A[(first + static_cast<size_t>(j)) * n + first + j];
    }
    for (int j = 0; j < width; ++j)
        A[row * n + first + static_cast<size_t>(j)] = values[j];
}

__global__ void updateTrailing(double* __restrict__ A, size_t n, size_t first,
                               int width, size_t trailingFirst) {
    // Blocks above the diagonal do not belong to the stored lower triangle.
    if (blockIdx.y < blockIdx.x) return;

    // Padding removes the shared-memory bank conflicts from column-wise reads.
    __shared__ double rowPanel[kUpdateTile][kBlockSize + 1];
    __shared__ double colPanel[kUpdateTile][kBlockSize + 1];
    const int tid = threadIdx.y * kUpdateTile + threadIdx.x;
    for (int index = tid; index < kUpdateTile * kBlockSize;
         index += kUpdateTile * kUpdateTile) {
        const int localRow = index / kBlockSize;
        const int p = index % kBlockSize;
        const size_t globalRow = trailingFirst +
                                 blockIdx.y * kUpdateTile + localRow;
        const size_t globalCol = trailingFirst +
                                 blockIdx.x * kUpdateTile + localRow;
        rowPanel[localRow][p] =
            (globalRow < n && p < width) ? A[globalRow * n + first + p] : 0.0;
        colPanel[localRow][p] =
            (globalCol < n && p < width) ? A[globalCol * n + first + p] : 0.0;
    }
    __syncthreads();

    const size_t row = trailingFirst + blockIdx.y * kUpdateTile + threadIdx.y;
    const size_t col = trailingFirst + blockIdx.x * kUpdateTile + threadIdx.x;
    if (row < n && col < n && row >= col) {
        double dot = 0.0;
#pragma unroll
        for (int p = 0; p < kBlockSize; ++p)
            dot += rowPanel[threadIdx.y][p] * colPanel[threadIdx.x][p];
        A[row * n + col] -= dot;
    }
}

__global__ void zeroUpper(double* A, size_t n) {
    const size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) +
                         threadIdx.x;
    if (index < n * n) {
        const size_t row = index / n;
        const size_t col = index - row * n;
        if (col > row) A[index] = 0.0;
    }
}

bool cudaSucceeded(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) return true;
    printf("CUDA error during %s: %s\n", operation, cudaGetErrorString(result));
    return false;
}
}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    if (n > static_cast<size_t>(INT_MAX)) {
        printf("Error: matrix dimension is too large\n");
        return false;
    }

    double* deviceA = nullptr;
    int* deviceFailure = nullptr;
    const size_t bytes = n * n * sizeof(double);
    int failure = INT_MAX;

    // One allocation holds both the matrix and the four-byte failure flag,
    // reducing allocator and synchronization overhead for smaller matrices.
    if (!cudaSucceeded(cudaMalloc(&deviceA, bytes + sizeof(int)),
                       "device allocation")) {
        cudaFree(deviceA);
        return false;
    }
    deviceFailure = reinterpret_cast<int*>(
        reinterpret_cast<unsigned char*>(deviceA) + bytes);
    if (!cudaSucceeded(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice),
                       "matrix upload") ||
        !cudaSucceeded(cudaMemset(deviceFailure, 0x7f, sizeof(int)),
                       "status initialization")) {
        cudaFree(deviceA);
        return false;
    }

    for (size_t first = 0; first < n; first += kBlockSize) {
        const int width = static_cast<int>(std::min<size_t>(kBlockSize, n - first));
        factorDiagonal<<<1, 32>>>(deviceA, n, first, width, deviceFailure);

        const size_t trailingFirst = first + static_cast<size_t>(width);
        if (trailingFirst < n) {
            const size_t rows = n - trailingFirst;
            solvePanel<<<static_cast<unsigned>((rows + 127) / 128), 128>>>(
                deviceA, n, first, width);
            const unsigned tiles =
                static_cast<unsigned>((rows + kUpdateTile - 1) / kUpdateTile);
            updateTrailing<<<dim3(tiles, tiles), dim3(kUpdateTile, kUpdateTile)>>>(
                deviceA, n, first, width, trailingFirst);
        }
    }

    const size_t elements = n * n;
    zeroUpper<<<static_cast<unsigned>((elements + 255) / 256), 256>>>(deviceA, n);
    bool ok = cudaSucceeded(cudaGetLastError(), "kernel launch") &&
              cudaSucceeded(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost),
                            "result download") &&
              cudaSucceeded(cudaMemcpy(&failure, deviceFailure, sizeof(int),
                                       cudaMemcpyDeviceToHost), "status download");
    cudaFree(deviceA);

    if (ok && failure != 0x7f7f7f7f) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", failure);
        return false;
    }
    return ok;
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
    // Establish the CUDA context outside the measured region. Context startup
    // is a one-time process cost, not part of the factorization.
    if (!cudaSucceeded(cudaFree(nullptr), "CUDA initialization")) return 1;
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / elapsedSeconds / 1e9;
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
