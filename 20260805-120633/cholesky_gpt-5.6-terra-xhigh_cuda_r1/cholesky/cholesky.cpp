#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// The factorization uses a right-looking, blocked Cholesky algorithm.  Each
// iteration factors one diagonal tile, solves all of the independent rows in
// the panel, then updates every lower-triangular tile in the trailing matrix.
// A 64-wide factor tile gives the GEMM update enough work per launch while
// leaving abundant parallelism in the trailing matrix.
namespace {

constexpr int kFactorTile = 64;
constexpr int kDiagonalThreads = 128;
constexpr int kPanelThreads = 128;
constexpr int kPanelRowsPerBlock = 128;
constexpr int kUpdateTile = 32;
constexpr int kUpdateK = 16;
constexpr int kUpdateThreadsX = 16;
constexpr int kUpdateThreadsY = 16;

bool checkCuda(const cudaError_t error, const char* const operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(error));
        return false;
    }
    return true;
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

    bool allocate(const size_t count, const char* const name) {
        return checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), name);
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

    bool create(const char* const name) { return checkCuda(cudaEventCreate(&event_), name); }
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

// Factor the current diagonal tile.  The column dependency is sequential, but
// the rows below each pivot are independent and are processed by the block in
// parallel.  Diagonal work is only O(tile^3), while the bulk of the algorithm
// is handled by the highly parallel trailing-update kernel below.
__global__ void factorDiagonalTile(double* const __restrict__ matrix, const int n,
                                   const int offset, const int width,
                                   int* const __restrict__ factorStatus) {
    __shared__ int valid;

    if (threadIdx.x == 0) {
        // A preceding tile may already have found a failed pivot.  Keep that
        // first failure, matching the early-exit behavior of the CPU code.
        valid = (*factorStatus < 0) ? 1 : 0;
    }
    __syncthreads();

    if (valid == 0) {
        return;
    }

    for (int column = 0; column < width; ++column) {
        if (threadIdx.x == 0) {
            double sum = 0.0;
            const int diagonalRow = offset + column;
            const size_t diagonalBase = static_cast<size_t>(diagonalRow) * n + offset;
            for (int p = 0; p < column; ++p) {
                const double value = matrix[diagonalBase + p];
                sum = fma(value, value, sum);
            }

            const size_t diagonalIndex = diagonalBase + column;
            const double pivot = matrix[diagonalIndex] - sum;
            if (pivot <= 0.0) {
                valid = 0;
                *factorStatus = diagonalRow;
            } else {
                matrix[diagonalIndex] = sqrt(pivot);
            }
        }
        __syncthreads();

        if (valid == 0) {
            return;
        }

        const int pivotRow = offset + column;
        const double pivot = matrix[static_cast<size_t>(pivotRow) * n + pivotRow];
        for (int row = column + 1 + threadIdx.x; row < width; row += blockDim.x) {
            const int globalRow = offset + row;
            const size_t rowBase = static_cast<size_t>(globalRow) * n + offset;
            const size_t pivotBase = static_cast<size_t>(pivotRow) * n + offset;
            double sum = 0.0;
            for (int p = 0; p < column; ++p) {
                sum = fma(matrix[rowBase + p], matrix[pivotBase + p], sum);
            }
            matrix[rowBase + column] = (matrix[rowBase + column] - sum) / pivot;
        }
        __syncthreads();
    }

    // Match the original representation: the returned factor has a zeroed
    // upper triangle.  The rest is cleared in one coalesced pass at the end.
    for (int linear = threadIdx.x; linear < width * width; linear += blockDim.x) {
        const int row = linear / width;
        const int column = linear - row * width;
        if (column > row) {
            matrix[static_cast<size_t>(offset + row) * n + offset + column] = 0.0;
        }
    }
}

// Solve L_ik * L_kk^T = A_ik for all rows below the diagonal tile.  A shared
// copy of L_kk avoids repeatedly fetching the same small factor tile.
__global__ void solvePanel(double* const __restrict__ matrix, const int n,
                           const int offset, const int width) {
    __shared__ double diagonal[kFactorTile * kFactorTile];

    for (int linear = threadIdx.x; linear < width * width; linear += blockDim.x) {
        const int row = linear / width;
        const int column = linear - row * width;
        diagonal[linear] = matrix[static_cast<size_t>(offset + row) * n + offset + column];
    }
    __syncthreads();

    const int row = offset + width + blockIdx.x * kPanelRowsPerBlock + threadIdx.x;
    if (row >= n) {
        return;
    }

    const size_t rowBase = static_cast<size_t>(row) * n + offset;
    for (int column = 0; column < width; ++column) {
        // Two accumulators expose independent FMAs for better latency hiding
        // within each thread's short dot product.
        double sumEven = 0.0;
        double sumOdd = 0.0;
        int p = 0;
        for (; p + 1 < column; p += 2) {
            sumEven = fma(matrix[rowBase + p], diagonal[column * width + p], sumEven);
            sumOdd = fma(matrix[rowBase + p + 1], diagonal[column * width + p + 1], sumOdd);
        }
        if (p < column) {
            sumEven = fma(matrix[rowBase + p], diagonal[column * width + p], sumEven);
        }
        matrix[rowBase + column] =
            (matrix[rowBase + column] - (sumEven + sumOdd)) / diagonal[column * width + column];
    }
}

// Update a 32x32 output subtile: A_ij -= L_ik * L_jk^T.  Each thread computes
// a 2x2 patch so a 16x16 block produces 32x32 double-precision output values.
__global__ __launch_bounds__(kUpdateThreadsX * kUpdateThreadsY)
void updateTrailingMatrix(double* const __restrict__ matrix, const int n,
                          const int offset, const int width, const int tileCount,
                          const int subtilesPerTile) {
    const unsigned long long outputBlocksPerTile =
        static_cast<unsigned long long>(subtilesPerTile) * subtilesPerTile;
    const unsigned long long packedTile = blockIdx.x / outputBlocksPerTile;
    const int subtile = static_cast<int>(blockIdx.x % outputBlocksPerTile);

    // packedTile enumerates the lower triangle in row-major packed form:
    // (0,0), (1,0), (1,1), (2,0), ... .  This avoids launching empty upper
    // triangular tiles and keeps all large trailing updates available to the GPU.
    const unsigned long long tileRow =
        (static_cast<unsigned long long>(sqrt(static_cast<double>(8ULL * packedTile + 1ULL))) - 1ULL) /
        2ULL;
    const unsigned long long tileColumn = packedTile - tileRow * (tileRow + 1ULL) / 2ULL;
    if (tileRow >= static_cast<unsigned long long>(tileCount)) {
        return;
    }

    const int subtileRow = subtile / subtilesPerTile;
    const int subtileColumn = subtile - subtileRow * subtilesPerTile;
    if (tileRow == tileColumn && subtileColumn > subtileRow) {
        return;
    }

    const int trailingStart = offset + width;
    const int rowBase = trailingStart + static_cast<int>(tileRow) * width + subtileRow * kUpdateTile;
    const int columnBase = trailingStart + static_cast<int>(tileColumn) * width + subtileColumn * kUpdateTile;
    const int localRow = threadIdx.y;
    const int localColumn = threadIdx.x;
    const int row0 = rowBase + localRow;
    const int row1 = row0 + kUpdateThreadsY;
    const int column0 = columnBase + localColumn;
    const int column1 = column0 + kUpdateThreadsX;

    __shared__ double left[kUpdateTile][kUpdateK];
    // Store the right operand transposed.  The inner loop reads a fixed k and
    // consecutive output columns; this layout makes those reads conflict-free
    // in shared memory instead of striding by 16 doubles through one bank.
    __shared__ double right[kUpdateK][kUpdateTile];

    double sum00 = 0.0;
    double sum01 = 0.0;
    double sum10 = 0.0;
    double sum11 = 0.0;

    for (int kBase = 0; kBase < width; kBase += kUpdateK) {
        for (int sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;
             sharedIndex < kUpdateTile * kUpdateK;
             sharedIndex += blockDim.x * blockDim.y) {
            const int sharedRow = sharedIndex / kUpdateK;
            const int sharedColumn = sharedIndex - sharedRow * kUpdateK;
            const int factorColumn = offset + kBase + sharedColumn;

            left[sharedRow][sharedColumn] =
                (rowBase + sharedRow < n && factorColumn < offset + width)
                    ? matrix[static_cast<size_t>(rowBase + sharedRow) * n + factorColumn]
                    : 0.0;
            right[sharedColumn][sharedRow] =
                (columnBase + sharedRow < n && factorColumn < offset + width)
                    ? matrix[static_cast<size_t>(columnBase + sharedRow) * n + factorColumn]
                    : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int p = 0; p < kUpdateK; ++p) {
            const double left0 = left[localRow][p];
            const double left1 = left[localRow + kUpdateThreadsY][p];
            const double right0 = right[p][localColumn];
            const double right1 = right[p][localColumn + kUpdateThreadsX];
            sum00 = fma(left0, right0, sum00);
            sum01 = fma(left0, right1, sum01);
            sum10 = fma(left1, right0, sum10);
            sum11 = fma(left1, right1, sum11);
        }
        __syncthreads();
    }

    if (row0 < n && column0 < n && row0 >= column0) {
        matrix[static_cast<size_t>(row0) * n + column0] -= sum00;
    }
    if (row0 < n && column1 < n && row0 >= column1) {
        matrix[static_cast<size_t>(row0) * n + column1] -= sum01;
    }
    if (row1 < n && column0 < n && row1 >= column0) {
        matrix[static_cast<size_t>(row1) * n + column0] -= sum10;
    }
    if (row1 < n && column1 < n && row1 >= column1) {
        matrix[static_cast<size_t>(row1) * n + column1] -= sum11;
    }
}

__global__ void clearUpperTriangle(double* const __restrict__ matrix, const size_t elements,
                                   const int n) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements && index % n > index / n) {
        matrix[index] = 0.0;
    }
}

}  // namespace

// GPU blocked Cholesky decomposition.  Matrix transfers are deliberately kept
// outside the timed CUDA event interval, so the reported performance measures
// the decomposition itself, just as the original timer measured only its CPU
// compute loop.
bool choleskyDecomposition(std::vector<double>& A, const size_t n, float& elapsedMilliseconds) {
    elapsedMilliseconds = 0.0F;
    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > std::numeric_limits<size_t>::max() / n) {
        std::fprintf(stderr, "Error: Matrix size is too large for the CUDA implementation\n");
        return false;
    }

    const size_t elements = n * n;
    const int dimension = static_cast<int>(n);
    DeviceBuffer<double> deviceMatrix;
    DeviceBuffer<int> deviceStatus;
    CudaEvent start;
    CudaEvent stop;

    if (!deviceMatrix.allocate(elements, "matrix allocation") ||
        !deviceStatus.allocate(1, "status allocation") ||
        !start.create("start-event creation") || !stop.create("stop-event creation")) {
        return false;
    }
    // -1 represents success; a non-negative value is the first diagonal
    // position found not to be positive definite.
    const int initialStatus = -1;
    if (!checkCuda(cudaMemcpy(deviceMatrix.get(), A.data(), elements * sizeof(double),
                              cudaMemcpyHostToDevice),
                   "matrix upload") ||
        !checkCuda(cudaMemcpy(deviceStatus.get(), &initialStatus, sizeof(initialStatus),
                              cudaMemcpyHostToDevice),
                   "status initialization") ||
        !checkCuda(cudaEventRecord(start.get()), "start-event recording")) {
        return false;
    }

    for (int offset = 0; offset < dimension;) {
        const int width = std::min(kFactorTile, dimension - offset);
        factorDiagonalTile<<<1, kDiagonalThreads>>>(deviceMatrix.get(), dimension, offset, width,
                                                      deviceStatus.get());

        const int remaining = dimension - offset - width;
        if (remaining > 0) {
            const int panelBlocks = (remaining + kPanelRowsPerBlock - 1) / kPanelRowsPerBlock;
            solvePanel<<<panelBlocks, kPanelThreads>>>(deviceMatrix.get(), dimension, offset, width);

            const int tileCount = (remaining + width - 1) / width;
            const int subtilesPerTile = (width + kUpdateTile - 1) / kUpdateTile;
            const unsigned long long packedTiles =
                static_cast<unsigned long long>(tileCount) * (tileCount + 1ULL) / 2ULL;
            const unsigned long long updateBlocks =
                packedTiles * subtilesPerTile * subtilesPerTile;
            updateTrailingMatrix<<<static_cast<unsigned int>(updateBlocks),
                                   dim3(kUpdateThreadsX, kUpdateThreadsY)>>>(
                deviceMatrix.get(), dimension, offset, width, tileCount, subtilesPerTile);
        }
        offset += width;
    }

    const size_t clearBlocks = (elements + 255) / 256;
    clearUpperTriangle<<<static_cast<unsigned int>(clearBlocks), 256>>>(deviceMatrix.get(), elements,
                                                                         dimension);

    if (!checkCuda(cudaGetLastError(), "kernel launch") ||
        !checkCuda(cudaEventRecord(stop.get()), "stop-event recording") ||
        !checkCuda(cudaEventSynchronize(stop.get()), "GPU factorization")) {
        return false;
    }
    if (!checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, start.get(), stop.get()),
                   "elapsed-time measurement")) {
        return false;
    }

    int factorStatus = 0;
    if (!checkCuda(cudaMemcpy(&factorStatus, deviceStatus.get(), sizeof(factorStatus),
                              cudaMemcpyDeviceToHost),
                   "status download")) {
        return false;
    }
    if (factorStatus >= 0) {
        std::printf("Error: Matrix is not positive definite at diagonal element %d\n",
                    factorStatus);
        return false;
    }
    return checkCuda(cudaMemcpy(A.data(), deviceMatrix.get(), elements * sizeof(double),
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
        A_orig = A;  // Save original for validation
    }

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    float elapsedMilliseconds = 0.0F;
    bool success = choleskyDecomposition(A, n, elapsedMilliseconds);
    const long durationMilliseconds = static_cast<long>(elapsedMilliseconds);

    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    printf("Computation time: %ld ms\n", durationMilliseconds);

    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (durationMilliseconds / 1000.0) / 1e9;
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
