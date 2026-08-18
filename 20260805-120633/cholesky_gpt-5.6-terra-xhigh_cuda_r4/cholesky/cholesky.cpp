#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// The panel size is deliberately a multiple of the update tile size.  It keeps
// the panel factorization small while making the rank-k updates large enough to
// use the GPU efficiently.
constexpr int kPanelWidth = 64;
constexpr int kUpdateTile = 32;
constexpr int kWarpWidth = 32;

static bool checkCuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(error));
        return false;
    }
    return true;
}

__device__ __forceinline__ double warpReduceSum(double value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

// Panel columns are dependent, but every entry below one diagonal value is
// independent.  This block uses a parallel diagonal reduction and lets one
// thread solve each remaining row of the current panel column.
__global__ void factorDiagonalPanel(double* A, const size_t n,
                                    const size_t panelStart, const int width,
                                    int* success,
                                    unsigned long long* failureIndex) {
    __shared__ double panel[kPanelWidth][kPanelWidth + 1];
    __shared__ double warpSums[2];

    const int thread = threadIdx.x;
    const int elements = width * width;
    for (int index = thread; index < elements; index += blockDim.x) {
        const int row = index / width;
        const int col = index - row * width;
        panel[row][col] = A[(panelStart + static_cast<size_t>(row)) * n +
                            panelStart + static_cast<size_t>(col)];
    }
    __syncthreads();

    const int lane = thread & (warpSize - 1);
    const int warp = thread / warpSize;
    for (int col = 0; col < width; ++col) {
        double diagonalSum = thread < col ? panel[col][thread] * panel[col][thread] : 0.0;
        diagonalSum = warpReduceSum(diagonalSum);
        if (lane == 0) {
            warpSums[warp] = diagonalSum;
        }
        __syncthreads();

        if (thread == 0) {
            const double value = panel[col][col] - warpSums[0] - warpSums[1];
            if (value <= 0.0) {
                *success = 0;
                *failureIndex = static_cast<unsigned long long>(panelStart + col);
                panel[col][col] = 0.0;
            } else {
                panel[col][col] = sqrt(value);
            }
        }
        __syncthreads();

        const int row = col + 1 + thread;
        if (row < width) {
            double sum = 0.0;
            for (int inner = 0; inner < col; ++inner) {
                sum += panel[row][inner] * panel[col][inner];
            }
            panel[row][col] = (panel[row][col] - sum) / panel[col][col];
        }
        __syncthreads();
    }

    for (int index = thread; index < elements; index += blockDim.x) {
        const int row = index / width;
        const int col = index - row * width;
        if (col > row) {
            panel[row][col] = 0.0;
        }
    }
    __syncthreads();

    for (int index = thread; index < elements; index += blockDim.x) {
        const int row = index / width;
        const int col = index - row * width;
        A[(panelStart + static_cast<size_t>(row)) * n +
          panelStart + static_cast<size_t>(col)] = panel[row][col];
    }
}

// Compute L_ik = A_ik * inv(L_kk^T).  One warp owns a row: lanes form the dot
// products cooperatively, which creates enough independent blocks to occupy
// the GPU even for the relatively narrow panel solve.
__global__ void solvePanelRows(double* A, const size_t n,
                               const size_t panelStart, const size_t firstRow,
                               const int width) {
    const int lane = threadIdx.x;
    const size_t row = firstRow + blockIdx.x;
    if (row >= n) {
        return;
    }

    __shared__ double rowValues[kPanelWidth];
    __shared__ double partialSums[kWarpWidth];
    double* const currentRow = A + row * n + panelStart;
    for (int index = lane; index < width; index += kWarpWidth) {
        rowValues[index] = currentRow[index];
    }
    __syncthreads();

    for (int col = 0; col < width; ++col) {
        double partial = 0.0;
        for (int inner = lane; inner < col; inner += kWarpWidth) {
            partial += rowValues[inner] *
                A[(panelStart + static_cast<size_t>(col)) * n +
                  panelStart + static_cast<size_t>(inner)];
        }
        partialSums[lane] = partial;
        __syncthreads();
        for (int offset = kWarpWidth / 2; offset > 0; offset /= 2) {
            if (lane < offset) {
                partialSums[lane] += partialSums[lane + offset];
            }
            __syncthreads();
        }
        if (lane == 0) {
            const double sum = partialSums[0];
            rowValues[col] = (rowValues[col] - sum) /
                A[(panelStart + static_cast<size_t>(col)) * n +
                  panelStart + static_cast<size_t>(col)];
        }
        __syncthreads();
    }
    for (int index = lane; index < width; index += kWarpWidth) {
        currentRow[index] = rowValues[index];
    }
}

// Tiled lower-triangular rank-k update:
// A_ij <- A_ij - L_ik * L_jk^T.  A 16x16 thread block computes a 32x32 tile
// (four values per thread), which gives each panel value substantial reuse.
__global__ void updateTrailingMatrix(double* A, const size_t n,
                                     const size_t panelStart,
                                     const size_t trailingStart,
                                     const int panelWidth) {
    const size_t tileRow = blockIdx.y;
    const size_t tileCol = blockIdx.x;
    if (tileRow < tileCol) {
        return;
    }

    const size_t rowBase = trailingStart + tileRow * kUpdateTile;
    const size_t colBase = trailingStart + tileCol * kUpdateTile;
    const int localRow = threadIdx.y;
    const int localCol = threadIdx.x;

    __shared__ double left[kUpdateTile][kPanelWidth + 1];
    // Stored transposed with respect to the mathematical right operand so its
    // global loads are coalesced, while the dot-product reads stay conflict-free.
    __shared__ double right[kUpdateTile][kPanelWidth + 1];

    for (int inner = localCol; inner < kPanelWidth; inner += blockDim.x) {
        if (inner < panelWidth) {
            const size_t panelColumn = panelStart + static_cast<size_t>(inner);
            const size_t row0 = rowBase + static_cast<size_t>(localRow);
            const size_t row1 = row0 + blockDim.y;
            const size_t col0 = colBase + static_cast<size_t>(localRow);
            const size_t col1 = col0 + blockDim.y;

            left[localRow][inner] = row0 < n ? A[row0 * n + panelColumn] : 0.0;
            left[localRow + blockDim.y][inner] =
                row1 < n ? A[row1 * n + panelColumn] : 0.0;
            right[localRow][inner] = col0 < n ? A[col0 * n + panelColumn] : 0.0;
            right[localRow + blockDim.y][inner] =
                col1 < n ? A[col1 * n + panelColumn] : 0.0;
        }
    }
    __syncthreads();

    const size_t row0 = rowBase + static_cast<size_t>(localRow);
    const size_t row1 = row0 + blockDim.y;
    const size_t col0 = colBase + static_cast<size_t>(localCol);
    const size_t col1 = col0 + blockDim.x;

    double value00 = 0.0;
    double value01 = 0.0;
    double value10 = 0.0;
    double value11 = 0.0;

    if (row0 < n && row0 >= col0) {
        value00 = A[row0 * n + col0];
    }
    if (row0 < n && row0 >= col1 && col1 < n) {
        value01 = A[row0 * n + col1];
    }
    if (row1 < n && row1 >= col0) {
        value10 = A[row1 * n + col0];
    }
    if (row1 < n && row1 >= col1 && col1 < n) {
        value11 = A[row1 * n + col1];
    }

#pragma unroll
    for (int inner = 0; inner < kPanelWidth; ++inner) {
        if (inner < panelWidth) {
            const double left0 = left[localRow][inner];
            const double left1 = left[localRow + blockDim.y][inner];
            const double right0 = right[localCol][inner];
            const double right1 = right[localCol + blockDim.x][inner];
            value00 -= left0 * right0;
            value01 -= left0 * right1;
            value10 -= left1 * right0;
            value11 -= left1 * right1;
        }
    }

    if (row0 < n && row0 >= col0) {
        A[row0 * n + col0] = value00;
    }
    if (row0 < n && row0 >= col1 && col1 < n) {
        A[row0 * n + col1] = value01;
    }
    if (row1 < n && row1 >= col0) {
        A[row1 * n + col0] = value10;
    }
    if (row1 < n && row1 >= col1 && col1 < n) {
        A[row1 * n + col1] = value11;
    }
}

// CUDA blocked, right-looking Cholesky decomposition.  Transfers are outside
// the event interval so the reported benchmark time measures the decomposition
// itself, just as the CPU implementation measured only its factorization loop.
bool choleskyDecomposition(std::vector<double>& A, const size_t n, double& elapsedMs) {
    elapsedMs = 0.0;
    if (n == 0) {
        return true;
    }
    if (n > std::numeric_limits<size_t>::max() / n) {
        std::fprintf(stderr, "Matrix size is too large\n");
        return false;
    }

    const size_t elementCount = n * n;
    if (elementCount > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Matrix allocation is too large\n");
        return false;
    }

    double* deviceMatrix = nullptr;
    int* deviceSuccess = nullptr;
    unsigned long long* deviceFailure = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const auto releaseResources = [&] {
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
        if (deviceMatrix != nullptr) cudaFree(deviceMatrix);
        if (deviceSuccess != nullptr) cudaFree(deviceSuccess);
        if (deviceFailure != nullptr) cudaFree(deviceFailure);
    };

    const size_t bytes = elementCount * sizeof(double);
    if (!checkCuda(cudaMalloc(&deviceMatrix, bytes), "matrix allocation") ||
        !checkCuda(cudaMalloc(&deviceSuccess, sizeof(*deviceSuccess)), "status allocation") ||
        !checkCuda(cudaMalloc(&deviceFailure, sizeof(*deviceFailure)), "failure-index allocation")) {
        releaseResources();
        return false;
    }

    const int initialSuccess = 1;
    const unsigned long long initialFailure = static_cast<unsigned long long>(n);
    if (!checkCuda(cudaMemcpy(deviceMatrix, A.data(), bytes, cudaMemcpyHostToDevice),
                   "matrix upload") ||
        !checkCuda(cudaMemcpy(deviceSuccess, &initialSuccess, sizeof(initialSuccess),
                   cudaMemcpyHostToDevice), "status upload") ||
        !checkCuda(cudaMemcpy(deviceFailure, &initialFailure, sizeof(initialFailure),
                   cudaMemcpyHostToDevice), "failure-index upload") ||
        !checkCuda(cudaEventCreate(&start), "start-event creation") ||
        !checkCuda(cudaEventCreate(&stop), "stop-event creation") ||
        !checkCuda(cudaEventRecord(start), "start-event recording")) {
        releaseResources();
        return false;
    }

    for (size_t panelStart = 0; panelStart < n; panelStart += kPanelWidth) {
        const int panelWidth = static_cast<int>(std::min<size_t>(kPanelWidth, n - panelStart));
        factorDiagonalPanel<<<1, kPanelWidth>>>(deviceMatrix, n, panelStart, panelWidth,
                                                   deviceSuccess, deviceFailure);
        if (!checkCuda(cudaPeekAtLastError(), "diagonal-panel factorization launch")) {
            releaseResources();
            return false;
        }

        const size_t trailingStart = panelStart + static_cast<size_t>(panelWidth);
        if (trailingStart >= n) {
            continue;
        }

        const size_t trailingRows = n - trailingStart;
        const unsigned int solveBlocks = static_cast<unsigned int>(trailingRows);
        solvePanelRows<<<solveBlocks, kWarpWidth>>>(deviceMatrix, n, panelStart,
                                                        trailingStart, panelWidth);
        if (!checkCuda(cudaPeekAtLastError(), "panel solve launch")) {
            releaseResources();
            return false;
        }

        const size_t tileCount = (trailingRows + kUpdateTile - 1) / kUpdateTile;
        const dim3 updateBlock(16, 16);
        const dim3 updateGrid(static_cast<unsigned int>(tileCount),
                              static_cast<unsigned int>(tileCount));
        updateTrailingMatrix<<<updateGrid, updateBlock>>>(deviceMatrix, n, panelStart,
                                                          trailingStart, panelWidth);
        if (!checkCuda(cudaPeekAtLastError(), "trailing-update launch")) {
            releaseResources();
            return false;
        }
    }

    float elapsedMsFloat = 0.0F;
    if (!checkCuda(cudaEventRecord(stop), "stop-event recording") ||
        !checkCuda(cudaEventSynchronize(stop), "decomposition synchronization") ||
        !checkCuda(cudaEventElapsedTime(&elapsedMsFloat, start, stop), "elapsed-time calculation")) {
        releaseResources();
        return false;
    }
    elapsedMs = elapsedMsFloat;

    int success = 0;
    unsigned long long failureIndex = initialFailure;
    if (!checkCuda(cudaMemcpy(&success, deviceSuccess, sizeof(success), cudaMemcpyDeviceToHost),
                   "status download") ||
        !checkCuda(cudaMemcpy(&failureIndex, deviceFailure, sizeof(failureIndex),
                   cudaMemcpyDeviceToHost), "failure-index download")) {
        releaseResources();
        return false;
    }

    if (success == 0) {
        std::printf("Error: Matrix is not positive definite at diagonal element %llu\n",
                    failureIndex);
        releaseResources();
        return false;
    }

    if (!checkCuda(cudaMemcpy(A.data(), deviceMatrix, bytes, cudaMemcpyDeviceToHost),
                   "result download")) {
        releaseResources();
        return false;
    }
    releaseResources();

    // Trailing updates intentionally operate only on the lower triangle.  Match
    // the original API's convention before exposing the result to callers.
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = row + 1; col < n; ++col) {
            A[row * n + col] = 0.0;
        }
    }
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random.
    // Then add identity scaled by n to make it strictly positive definite.
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    double maxError = 0.0;
    double relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = std::fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        const double rel = error / (std::fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    std::printf("Cholesky Decomposition Benchmark\n");
    std::printf("Matrix size: %zu x %zu\n", n, n);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    std::printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);

    if (validate) {
        A_orig = A;
    }

    std::printf("Computing Cholesky decomposition...\n");
    double elapsedMs = 0.0;
    const bool success = choleskyDecomposition(A, n, elapsedMs);
    if (!success) {
        std::printf("Cholesky decomposition failed\n");
        return 1;
    }

    const long reportedMs = static_cast<long>(elapsedMs);
    std::printf("Computation time: %ld ms\n", reportedMs);

    const double ops = static_cast<double>(n) * n * n / 3.0;
    const double gflops = elapsedMs > 0.0 ? ops / (elapsedMs * 1.0e6) : 0.0;
    std::printf("Performance: %.3f GFLOPS\n", gflops);

    if (printResults) {
        print_results(A, "CholeskyL");
    }

    if (validate) {
        std::printf("Validating result...\n");
        if (validateCholesky(A, A_orig, n)) {
            std::printf("Validation: PASSED\n");
            return 0;
        }
        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
