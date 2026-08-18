#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

// A 64-column algorithmic block amortizes global-memory traffic in the
// trailing update.  Each update thread computes a 2x2 output micro-tile.
constexpr int kFactorBlock = 64;
constexpr int kUpdateTile = 32;
constexpr int kKernelThreads = 256;
constexpr int kGramTile = 16;

bool checkCuda(const cudaError_t result, const char* operation) {
    if (result == cudaSuccess) {
        return true;
    }
    printf("CUDA error during %s: %s\n", operation, cudaGetErrorString(result));
    return false;
}

bool initializeCuda() {
    int deviceCount = 0;
    if (!checkCuda(cudaGetDeviceCount(&deviceCount), "device discovery")) {
        return false;
    }
    if (deviceCount == 0) {
        printf("CUDA error during device discovery: no CUDA-capable GPU found\n");
        return false;
    }
    // Materialize the runtime context before the measured factorization.
    return checkCuda(cudaSetDevice(0), "device selection") &&
           checkCuda(cudaFree(nullptr), "device initialization");
}

__global__ void factorDiagonalKernel(double* __restrict__ matrix, const int n,
                                     const int offset, const int width,
                                     int* __restrict__ info) {
    __shared__ double tile[kFactorBlock][kFactorBlock + 1];

    for (int index = threadIdx.x; index < kFactorBlock * kFactorBlock;
         index += blockDim.x) {
        const int row = index / kFactorBlock;
        const int column = index - row * kFactorBlock;
        tile[row][column] = (row < width && column < width)
                                ? matrix[(offset + row) * n + offset + column]
                                : 0.0;
    }
    __syncthreads();

    for (int column = 0; column < width; ++column) {
        if (threadIdx.x == 0) {
            const double diagonal = tile[column][column];
            if (!(diagonal > 0.0)) {
                atomicCAS(info, 0, offset + column + 1);
                // Keep all subsequent kernels numerically defined.  The host
                // reports the saved failure after the queued work completes.
                tile[column][column] = 1.0;
            } else {
                tile[column][column] = sqrt(diagonal);
            }
        }
        __syncthreads();

        for (int row = column + 1 + threadIdx.x; row < width;
             row += blockDim.x) {
            tile[row][column] /= tile[column][column];
        }
        __syncthreads();

        for (int index = threadIdx.x; index < width * width;
             index += blockDim.x) {
            const int row = index / width;
            const int targetColumn = index - row * width;
            if (row >= targetColumn && targetColumn > column) {
                tile[row][targetColumn] -=
                    tile[row][column] * tile[targetColumn][column];
            }
        }
        __syncthreads();
    }

    for (int index = threadIdx.x; index < width * width;
         index += blockDim.x) {
        const int row = index / width;
        const int column = index - row * width;
        if (row >= column) {
            matrix[(offset + row) * n + offset + column] = tile[row][column];
        }
    }
}

// One warp solves one panel row.  Lanes own columns, which makes both the
// input and output accesses coalesced.  A lane retains each solved value in a
// register for all later dot products.
__global__ void solvePanelKernel(double* __restrict__ matrix, const int n,
                                 const int offset, const int width) {
    __shared__ double diagonal[kFactorBlock][kFactorBlock + 1];

    for (int index = threadIdx.x; index < kFactorBlock * kFactorBlock;
         index += blockDim.x) {
        const int row = index / kFactorBlock;
        const int column = index - row * kFactorBlock;
        diagonal[row][column] = (row < width && column <= row)
                                    ? matrix[(offset + row) * n + offset + column]
                                    : 0.0;
    }
    __syncthreads();

    const int lane = threadIdx.x & (warpSize - 1);
    const int warpInBlock = threadIdx.x / warpSize;
    const int warpsPerBlock = blockDim.x / warpSize;
    const int row = offset + width + blockIdx.x * warpsPerBlock + warpInBlock;
    const unsigned int mask = 0xffffffffu;

    double value0 = 0.0;
    double value1 = 0.0;
    if (row < n) {
        if (lane < width) {
            value0 = matrix[row * n + offset + lane];
        }
        if (lane + warpSize < width) {
            value1 = matrix[row * n + offset + lane + warpSize];
        }
    }

    for (int column = 0; column < width; ++column) {
        double product = 0.0;
        if (row < n) {
            if (column < warpSize) {
                if (lane < column) {
                    product = value0 * diagonal[column][lane];
                }
            } else {
                product = value0 * diagonal[column][lane];
                if (lane < column - warpSize) {
                    product += value1 * diagonal[column][lane + warpSize];
                }
            }
        }

#pragma unroll
        for (int delta = warpSize / 2; delta > 0; delta >>= 1) {
            product += __shfl_down_sync(mask, product, delta);
        }
        const double sum = __shfl_sync(mask, product, 0);

        if (row < n) {
            if (column < warpSize && lane == column) {
                value0 = (value0 - sum) / diagonal[column][column];
            } else if (column >= warpSize && lane == column - warpSize) {
                value1 = (value1 - sum) / diagonal[column][column];
            }
        }
    }

    if (row < n) {
        if (lane < width) {
            matrix[row * n + offset + lane] = value0;
        }
        if (lane + warpSize < width) {
            matrix[row * n + offset + lane + warpSize] = value1;
        }
    }
}

__global__ void updateTrailingKernel(double* __restrict__ matrix, const int n,
                                     const int panelOffset,
                                     const int panelWidth) {
    __shared__ double rowPanel[kUpdateTile][kFactorBlock + 1];
    __shared__ double columnPanel[kUpdateTile][kFactorBlock + 1];

    // Whole tiles above the diagonal contain no required output.
    if (blockIdx.y < blockIdx.x) {
        return;
    }

    const int thread = threadIdx.y * blockDim.x + threadIdx.x;
    const int trailingOffset = panelOffset + panelWidth;
    const int rowBase = trailingOffset + blockIdx.y * kUpdateTile;
    const int columnBase = trailingOffset + blockIdx.x * kUpdateTile;

    for (int index = thread; index < kUpdateTile * kFactorBlock;
         index += blockDim.x * blockDim.y) {
        const int localRow = index / kFactorBlock;
        const int k = index - localRow * kFactorBlock;
        const int globalRow = rowBase + localRow;
        const int globalColumn = columnBase + localRow;
        rowPanel[localRow][k] = (globalRow < n && k < panelWidth)
                                    ? matrix[globalRow * n + panelOffset + k]
                                    : 0.0;
        columnPanel[localRow][k] = (globalColumn < n && k < panelWidth)
                                       ? matrix[globalColumn * n + panelOffset + k]
                                       : 0.0;
    }
    __syncthreads();

    const int row0 = rowBase + threadIdx.y;
    const int row1 = row0 + blockDim.y;
    const int column0 = columnBase + threadIdx.x;
    const int column1 = column0 + blockDim.x;
    double sum00 = 0.0;
    double sum01 = 0.0;
    double sum10 = 0.0;
    double sum11 = 0.0;

#pragma unroll
    for (int k = 0; k < kFactorBlock; ++k) {
        const double a0 = rowPanel[threadIdx.y][k];
        const double a1 = rowPanel[threadIdx.y + blockDim.y][k];
        const double b0 = columnPanel[threadIdx.x][k];
        const double b1 = columnPanel[threadIdx.x + blockDim.x][k];
        sum00 += a0 * b0;
        sum01 += a0 * b1;
        sum10 += a1 * b0;
        sum11 += a1 * b1;
    }

    if (row0 < n && column0 < n && row0 >= column0) {
        matrix[row0 * n + column0] -= sum00;
    }
    if (row0 < n && column1 < n && row0 >= column1) {
        matrix[row0 * n + column1] -= sum01;
    }
    if (row1 < n && column0 < n && row1 >= column0) {
        matrix[row1 * n + column0] -= sum10;
    }
    if (row1 < n && column1 < n && row1 >= column1) {
        matrix[row1 * n + column1] -= sum11;
    }
}

__global__ void zeroUpperKernel(double* matrix, const int n) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && column < n && column > row) {
        matrix[row * n + column] = 0.0;
    }
}

// Computes output = input * input^T.  Both source row tiles are staged in
// shared memory, so the O(n^3) matrix construction and validation products do
// not become serial bottlenecks around the factorization benchmark.
__global__ void gramMatrixKernel(const double* __restrict__ input,
                                 double* __restrict__ output, const int n,
                                 const double diagonalBoost) {
    __shared__ double rowTile[kGramTile][kGramTile + 1];
    __shared__ double columnTile[kGramTile][kGramTile + 1];

    const int row = blockIdx.y * kGramTile + threadIdx.y;
    const int column = blockIdx.x * kGramTile + threadIdx.x;
    double sum = 0.0;

    for (int kBase = 0; kBase < n; kBase += kGramTile) {
        const int rowK = kBase + threadIdx.x;
        const int columnK = kBase + threadIdx.y;
        rowTile[threadIdx.y][threadIdx.x] =
            (row < n && rowK < n) ? input[row * n + rowK] : 0.0;
        columnTile[threadIdx.x][threadIdx.y] =
            (column < n && columnK < n) ? input[column * n + columnK] : 0.0;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kGramTile; ++k) {
            sum += rowTile[threadIdx.y][k] * columnTile[threadIdx.x][k];
        }
        __syncthreads();
    }

    if (row < n && column < n) {
        output[row * n + column] =
            sum + ((row == column) ? diagonalBoost : 0.0);
    }
}

bool computeGramMatrix(const std::vector<double>& input,
                       std::vector<double>& output, const size_t n,
                       const double diagonalBoost) {
    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(INT_MAX) ||
        n > static_cast<size_t>(-1) / n / sizeof(double)) {
        printf("Error: Matrix is too large for the CUDA implementation\n");
        return false;
    }

    const int dimension = static_cast<int>(n);
    const size_t matrixBytes = n * n * sizeof(double);
    double* deviceInput = nullptr;
    double* deviceOutput = nullptr;
    if (!checkCuda(cudaMalloc(&deviceInput, matrixBytes), "Gram input allocation") ||
        !checkCuda(cudaMalloc(&deviceOutput, matrixBytes), "Gram output allocation")) {
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return false;
    }

    bool cudaOk = checkCuda(cudaMemcpy(deviceInput, input.data(), matrixBytes,
                                       cudaMemcpyHostToDevice),
                            "Gram input upload");
    if (cudaOk) {
        const dim3 threads(kGramTile, kGramTile);
        const dim3 blocks((dimension + kGramTile - 1) / kGramTile,
                          (dimension + kGramTile - 1) / kGramTile);
        gramMatrixKernel<<<blocks, threads>>>(deviceInput, deviceOutput,
                                              dimension, diagonalBoost);
        cudaOk = checkCuda(cudaGetLastError(), "Gram-matrix kernel launch");
    }
    if (cudaOk) {
        cudaOk = checkCuda(cudaMemcpy(output.data(), deviceOutput, matrixBytes,
                                      cudaMemcpyDeviceToHost),
                           "Gram output download");
    }

    const bool freeInputOk =
        checkCuda(cudaFree(deviceInput), "Gram input release");
    const bool freeOutputOk =
        checkCuda(cudaFree(deviceOutput), "Gram output release");
    return cudaOk && freeInputOk && freeOutputOk;
}

}  // namespace

// Blocked right-looking Cholesky decomposition.  The matrix remains row-major
// and is returned as the same lower-triangular in-place representation as the
// original implementation.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(INT_MAX) ||
        n > static_cast<size_t>(-1) / n / sizeof(double)) {
        printf("Error: Matrix is too large for the CUDA implementation\n");
        return false;
    }

    const int dimension = static_cast<int>(n);
    const size_t matrixBytes = n * n * sizeof(double);
    double* deviceMatrix = nullptr;
    int* deviceInfo = nullptr;

    if (!checkCuda(cudaMalloc(&deviceMatrix, matrixBytes), "matrix allocation") ||
        !checkCuda(cudaMalloc(&deviceInfo, sizeof(int)), "status allocation")) {
        cudaFree(deviceMatrix);
        cudaFree(deviceInfo);
        return false;
    }

    bool cudaOk = checkCuda(cudaMemcpy(deviceMatrix, A.data(), matrixBytes,
                                       cudaMemcpyHostToDevice),
                            "matrix upload") &&
                  checkCuda(cudaMemset(deviceInfo, 0, sizeof(int)),
                            "status initialization");

    for (int offset = 0; cudaOk && offset < dimension; offset += kFactorBlock) {
        const int width = min(kFactorBlock, dimension - offset);
        factorDiagonalKernel<<<1, kKernelThreads>>>(
            deviceMatrix, dimension, offset, width, deviceInfo);
        cudaOk = checkCuda(cudaGetLastError(), "diagonal kernel launch");

        const int remaining = dimension - offset - width;
        if (cudaOk && remaining > 0) {
            constexpr int warpsPerBlock = kKernelThreads / 32;
            const int panelBlocks = (remaining + warpsPerBlock - 1) / warpsPerBlock;
            solvePanelKernel<<<panelBlocks, kKernelThreads>>>(
                deviceMatrix, dimension, offset, width);
            cudaOk = checkCuda(cudaGetLastError(), "panel kernel launch");

            const int updateBlocks = (remaining + kUpdateTile - 1) / kUpdateTile;
            const dim3 updateThreads(kUpdateTile / 2, kUpdateTile / 2);
            const dim3 updateGrid(updateBlocks, updateBlocks);
            updateTrailingKernel<<<updateGrid, updateThreads>>>(
                deviceMatrix, dimension, offset, width);
            cudaOk = checkCuda(cudaGetLastError(), "trailing-update kernel launch");
        }
    }

    if (cudaOk) {
        const dim3 threads(32, 8);
        const dim3 blocks((dimension + threads.x - 1) / threads.x,
                          (dimension + threads.y - 1) / threads.y);
        zeroUpperKernel<<<blocks, threads>>>(deviceMatrix, dimension);
        cudaOk = checkCuda(cudaGetLastError(), "upper-triangle kernel launch");
    }

    int info = 0;
    if (cudaOk) {
        cudaOk = checkCuda(cudaMemcpy(&info, deviceInfo, sizeof(int),
                                      cudaMemcpyDeviceToHost),
                           "factorization completion");
    }
    if (cudaOk && info == 0) {
        cudaOk = checkCuda(cudaMemcpy(A.data(), deviceMatrix, matrixBytes,
                                      cudaMemcpyDeviceToHost),
                           "matrix download");
    }

    const bool freeMatrixOk = checkCuda(cudaFree(deviceMatrix), "matrix release");
    const bool freeInfoOk = checkCuda(cudaFree(deviceInfo), "status release");
    if (!cudaOk || !freeMatrixOk || !freeInfoOk) {
        return false;
    }
    if (info != 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n",
               info - 1);
        return false;
    }
    return true;
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
    return computeGramMatrix(B, A, n, static_cast<double>(n));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);

    // Compute L * L^T on the GPU.
    if (!computeGramMatrix(L, reconstructed, n, 0.0)) {
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
        return 1;
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    if (!generatePositiveDefiniteMatrix(A, n)) {
        return 1;
    }
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> duration = end - start;
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / duration.count() / 1e6;
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
