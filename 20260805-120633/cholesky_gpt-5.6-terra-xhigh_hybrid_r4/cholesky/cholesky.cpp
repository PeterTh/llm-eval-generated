#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// The matrix is distributed by cyclic block rows.  A rank therefore owns every
// nranks-th row tile, which keeps the trailing-update work balanced throughout
// the factorization.  Each owned tile is contiguous on its GPU.
namespace {

constexpr int kPreferredBlockSize = 128;
constexpr int kUpdateTile = 16;

[[noreturn]] void abortWithMessage(const char* action, const char* detail, int rank) {
    std::fprintf(stderr, "Rank %d: %s: %s\n", rank, action, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* action, int rank) {
    if (status != cudaSuccess) {
        abortWithMessage(action, cudaGetErrorString(status), rank);
    }
}

void checkMpi(int status, const char* action, int rank) {
    if (status != MPI_SUCCESS) {
        char error[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, error, &length);
        error[length] = '\0';
        abortWithMessage(action, error, rank);
    }
}

int blockCount(int n, int blockSize) {
    return (n + blockSize - 1) / blockSize;
}

int localBlockCount(int blocks, int rank, int ranks) {
    return rank < blocks ? (blocks - 1 - rank) / ranks + 1 : 0;
}

__host__ __device__ inline int firstBlockAfter(int owner, int panelBlock, int ranks) {
    if (owner > panelBlock) {
        return owner;
    }
    return owner + ((panelBlock - owner) / ranks + 1) * ranks;
}

int trailingBlockCount(int owner, int panelBlock, int blocks, int ranks) {
    const int first = firstBlockAfter(owner, panelBlock, ranks);
    return first >= blocks ? 0 : (blocks - 1 - first) / ranks + 1;
}

// glibc's rand_r advances its 32-bit LCG state three times per returned value.
// Jumping that affine LCG lets independently distributed rows retain the exact
// deterministic random stream used by the original sequential generator.
std::uint32_t advanceRandRState(std::uint32_t state, std::uint64_t steps) {
    std::uint32_t accumulatedMultiplier = 1;
    std::uint32_t accumulatedIncrement = 0;
    std::uint32_t multiplier = 1103515245U;
    std::uint32_t increment = 12345U;
    while (steps != 0) {
        if ((steps & 1U) != 0) {
            accumulatedMultiplier *= multiplier;
            accumulatedIncrement = accumulatedIncrement * multiplier + increment;
        }
        increment = (multiplier + 1U) * increment;
        multiplier *= multiplier;
        steps >>= 1U;
    }
    return accumulatedMultiplier * state + accumulatedIncrement;
}

void fillRandomRow(double* destination, int row, int n) {
    const std::uint64_t valuesBeforeRow = static_cast<std::uint64_t>(row) * n;
    unsigned int seed = advanceRandRState(42U, 3U * valuesBeforeRow);
    for (int column = 0; column < n; ++column) {
        destination[column] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
    }
}

__global__ void potrfDiagonalKernel(double* matrix, int n, int blockSize,
                                    int localBlock, int diagonalBlock,
                                    int diagonalRows, int* failure) {
    const int thread = threadIdx.x;
    double* const tile = matrix + static_cast<size_t>(localBlock) * blockSize * n +
                         diagonalBlock * blockSize;

    for (int column = 0; column < diagonalRows; ++column) {
        if (thread == 0) {
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                const double value = tile[static_cast<size_t>(column) * n + k];
                sum += value * value;
            }
            const double value = tile[static_cast<size_t>(column) * n + column] - sum;
            if (value <= 0.0 || !isfinite(value)) {
                *failure = 1;
                tile[static_cast<size_t>(column) * n + column] = 1.0;
            } else {
                tile[static_cast<size_t>(column) * n + column] = sqrt(value);
            }
        }
        __syncthreads();

        const int row = column + 1 + thread;
        if (row < diagonalRows) {
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                sum += tile[static_cast<size_t>(row) * n + k] *
                       tile[static_cast<size_t>(column) * n + k];
            }
            tile[static_cast<size_t>(row) * n + column] =
                (tile[static_cast<size_t>(row) * n + column] - sum) /
                tile[static_cast<size_t>(column) * n + column];
        }
        __syncthreads();
    }
}

__global__ void trsmPanelKernel(double* matrix, const double* diagonal,
                                int n, int blockSize, int panelBlock,
                                int rank, int ranks, int localBlocks) {
    const int localBlock = blockIdx.x;
    const int row = threadIdx.x;
    const int globalBlock = rank + localBlock * ranks;
    if (localBlock >= localBlocks || globalBlock <= panelBlock ||
        globalBlock * blockSize + row >= n) {
        return;
    }
    double* const tile = matrix + static_cast<size_t>(localBlock) * blockSize * n +
                         panelBlock * blockSize;
    const int columns = min(blockSize, n - panelBlock * blockSize);
    for (int column = 0; column < columns; ++column) {
        double value = tile[static_cast<size_t>(row) * n + column];
        for (int k = 0; k < column; ++k) {
            value -= tile[static_cast<size_t>(row) * n + k] *
                     diagonal[static_cast<size_t>(column) * blockSize + k];
        }
        tile[static_cast<size_t>(row) * n + column] =
            value / diagonal[static_cast<size_t>(column) * blockSize + column];
    }
}

// Pack the local pieces of the current column panel before MPI_Allgatherv.
__global__ void packPanelKernel(const double* matrix, double* packed,
                                int n, int blockSize, int panelBlock,
                                int rank, int ranks, int localBlocks) {
    const int localBlock = blockIdx.x;
    const int globalBlock = rank + localBlock * ranks;
    if (localBlock >= localBlocks || globalBlock <= panelBlock) {
        return;
    }
    const int first = firstBlockAfter(rank, panelBlock, ranks);
    const int outputBlock = (globalBlock - first) / ranks;
    const double* const source = matrix + static_cast<size_t>(localBlock) * blockSize * n +
                                 panelBlock * blockSize;
    double* const destination = packed + static_cast<size_t>(outputBlock) * blockSize * blockSize;

    for (int index = threadIdx.x; index < blockSize * blockSize; index += blockDim.x) {
        const int row = index / blockSize;
        const int column = index % blockSize;
        const int globalRow = globalBlock * blockSize + row;
        destination[index] = globalRow < n ? source[static_cast<size_t>(row) * n + column] : 0.0;
    }
}

// Convert rank-contiguous MPI receive data back into globally addressed panel
// tiles on the device.  rankDisplacements is measured in doubles.
__global__ void unpackPanelKernel(const double* packed, double* panel,
                                  const int* rankDisplacements,
                                  int n, int blockSize, int panelBlock,
                                  int blocks, int ranks) {
    const int globalBlock = panelBlock + 1 + blockIdx.x;
    if (globalBlock >= blocks) {
        return;
    }
    const int owner = globalBlock % ranks;
    const int first = firstBlockAfter(owner, panelBlock, ranks);
    const int blockOffset = (globalBlock - first) / ranks;
    const double* const source = packed + rankDisplacements[owner] +
                                 static_cast<size_t>(blockOffset) * blockSize * blockSize;
    double* const destination = panel + static_cast<size_t>(globalBlock) * blockSize * blockSize;

    for (int index = threadIdx.x; index < blockSize * blockSize; index += blockDim.x) {
        const int row = index / blockSize;
        const int globalRow = globalBlock * blockSize + row;
        if (globalRow < n) {
            destination[index] = source[index];
        }
    }
}

// A 16x16 CUDA thread tile updates one output tile fragment.  The left and
// gathered right panel fragments are reused from shared memory across 256
// output elements, avoiding per-tile BLAS-launch latency for this fine-grain
// distributed schedule.
__global__ void trailingUpdateKernel(double* matrix, const double* panel,
                                     int n, int blockSize, int panelBlock,
                                     int rank, int ranks, int localBlocks) {
    const int localBlock = blockIdx.x;
    const int targetBlock = panelBlock + 1 + blockIdx.y;
    const int globalBlock = rank + localBlock * ranks;
    if (localBlock >= localBlocks || globalBlock <= panelBlock || targetBlock > globalBlock) {
        return;
    }

    const int tilesPerDimension = (blockSize + kUpdateTile - 1) / kUpdateTile;
    const int tileRow = blockIdx.z / tilesPerDimension;
    const int tileColumn = blockIdx.z % tilesPerDimension;
    const int row = tileRow * kUpdateTile + threadIdx.y;
    const int column = tileColumn * kUpdateTile + threadIdx.x;
    const int globalRow = globalBlock * blockSize + row;
    const int globalColumn = targetBlock * blockSize + column;
    double* const localTile = matrix + static_cast<size_t>(localBlock) * blockSize * n;

    __shared__ double left[kUpdateTile][kUpdateTile];
    __shared__ double right[kUpdateTile][kUpdateTile];
    double sum = 0.0;
    const int panelColumns = min(blockSize, n - panelBlock * blockSize);
    for (int phase = 0; phase < panelColumns; phase += kUpdateTile) {
        const int leftColumn = phase + threadIdx.x;
        const int rightRow = phase + threadIdx.y;
        left[threadIdx.y][threadIdx.x] = globalRow < n && leftColumn < panelColumns
            ? localTile[static_cast<size_t>(row) * n + panelBlock * blockSize + leftColumn]
            : 0.0;
        right[threadIdx.x][threadIdx.y] = globalColumn < n && rightRow < panelColumns
            ? panel[static_cast<size_t>(targetBlock * blockSize + column) * blockSize + rightRow]
            : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < kUpdateTile; ++k) {
            sum += left[threadIdx.y][k] * right[threadIdx.x][k];
        }
        __syncthreads();
    }
    if (globalRow < n && globalColumn < n) {
        localTile[static_cast<size_t>(row) * n + globalColumn] -= sum;
    }
}

__global__ void zeroUpperKernel(double* matrix, int n, int blockSize,
                                int rank, int ranks, int localBlocks) {
    const int localBlock = blockIdx.y;
    const int globalBlock = rank + localBlock * ranks;
    const int globalRow = globalBlock * blockSize + blockIdx.z * blockDim.y + threadIdx.y;
    const int column = threadIdx.x + blockIdx.x * blockDim.x;
    if (localBlock < localBlocks && globalRow < n && column < n && column > globalRow) {
        matrix[static_cast<size_t>(localBlock) * blockSize * n +
               static_cast<size_t>(globalRow - globalBlock * blockSize) * n + column] = 0.0;
    }
}

void generateLocalPositiveDefiniteMatrix(std::vector<double>& matrix,
                                         std::vector<double>* original,
                                         int n, int blockSize, int blocks,
                                         int rank, int ranks) {
    const int localBlocks = localBlockCount(blocks, rank, ranks);
    const size_t tileElements = static_cast<size_t>(blockSize) * n;
    std::vector<double> localB(static_cast<size_t>(localBlocks) * tileElements, 0.0);
    std::vector<double> sourceB(tileElements, 0.0);

    #pragma omp parallel for schedule(static)
    for (int localBlock = 0; localBlock < localBlocks; ++localBlock) {
        const int globalBlock = rank + localBlock * ranks;
        for (int row = 0; row < blockSize; ++row) {
            const int globalRow = globalBlock * blockSize + row;
            if (globalRow >= n) {
                break;
            }
            double* const destination = localB.data() + static_cast<size_t>(localBlock) * tileElements +
                                        static_cast<size_t>(row) * n;
            fillRandomRow(destination, globalRow, n);
        }
    }

    // Traversing source tiles serially keeps B_j in a compact cache-resident
    // buffer, while the expensive independent output rows use all OpenMP cores.
    for (int sourceBlock = 0; sourceBlock < blocks; ++sourceBlock) {
        std::fill(sourceB.begin(), sourceB.end(), 0.0);
        const int sourceStart = sourceBlock * blockSize;
        const int sourceRows = std::min(blockSize, n - sourceStart);
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < sourceRows; ++row) {
            double* const destination = sourceB.data() + static_cast<size_t>(row) * n;
            fillRandomRow(destination, sourceStart + row, n);
        }

        #pragma omp parallel for schedule(dynamic)
        for (int localBlock = 0; localBlock < localBlocks; ++localBlock) {
            const int globalBlock = rank + localBlock * ranks;
            if (globalBlock < sourceBlock) {
                continue;
            }
            const int rowStart = globalBlock * blockSize;
            const int rows = std::min(blockSize, n - rowStart);
            double* const output = matrix.data() + static_cast<size_t>(localBlock) * tileElements;
            const double* const leftRows = localB.data() + static_cast<size_t>(localBlock) * tileElements;

            for (int row = 0; row < rows; ++row) {
                const int columns = globalBlock == sourceBlock
                                        ? std::min(blockSize, row + 1)
                                        : sourceRows;
                const double* const left = leftRows + static_cast<size_t>(row) * n;
                for (int column = 0; column < columns; ++column) {
                    const double* const right = sourceB.data() + static_cast<size_t>(column) * n;
                    double sum = 0.0;
                    for (int k = 0; k < n; ++k) {
                        sum += left[k] * right[k];
                    }
                    const int globalColumn = sourceStart + column;
                    output[static_cast<size_t>(row) * n + globalColumn] =
                        sum + (rowStart + row == globalColumn ? static_cast<double>(n) : 0.0);
                }
            }
        }
    }

    if (original != nullptr) {
        *original = matrix;
    }
}

void gatherMatrix(const std::vector<double>& localMatrix, std::vector<double>& globalMatrix,
                  int n, int blockSize, int blocks, int rank, int ranks) {
    const int localBlocks = localBlockCount(blocks, rank, ranks);
    const int tileElements = blockSize * n;
    const int sendCount = localBlocks * tileElements;
    std::vector<int> counts(ranks);
    std::vector<int> displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = localBlockCount(blocks, r, ranks) * tileElements;
        if (r > 0) {
            displacements[r] = displacements[r - 1] + counts[r - 1];
        }
    }

    std::vector<double> packed;
    if (rank == 0) {
        packed.resize(static_cast<size_t>(displacements.back()) + counts.back());
    }
    checkMpi(MPI_Gatherv(localMatrix.data(), sendCount, MPI_DOUBLE,
                         rank == 0 ? packed.data() : nullptr, counts.data(), displacements.data(),
                         MPI_DOUBLE, 0, MPI_COMM_WORLD), "gathering distributed matrix", rank);
    if (rank != 0) {
        return;
    }

    globalMatrix.assign(static_cast<size_t>(n) * n, 0.0);
    for (int sourceRank = 0; sourceRank < ranks; ++sourceRank) {
        const int sourceBlocks = localBlockCount(blocks, sourceRank, ranks);
        const double* const source = packed.data() + displacements[sourceRank];
        for (int localBlock = 0; localBlock < sourceBlocks; ++localBlock) {
            const int globalBlock = sourceRank + localBlock * ranks;
            const int rowStart = globalBlock * blockSize;
            const int rows = std::min(blockSize, n - rowStart);
            const double* const tile = source + static_cast<size_t>(localBlock) * tileElements;
            for (int row = 0; row < rows; ++row) {
                std::copy_n(tile + static_cast<size_t>(row) * n, n,
                            globalMatrix.data() + static_cast<size_t>(rowStart + row) * n);
            }
        }
    }
}

bool validateCholesky(const std::vector<double>& lower, const std::vector<double>& original, int n) {
    double maxError = 0.0;
    double relError = 0.0;

    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for (int row = 0; row < n; ++row) {
        for (int column = 0; column <= row; ++column) {
            double sum = 0.0;
            for (int k = 0; k <= column; ++k) {
                sum += lower[static_cast<size_t>(row) * n + k] *
                       lower[static_cast<size_t>(column) * n + k];
            }
            const double error = std::fabs(sum - original[static_cast<size_t>(row) * n + column]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error /
                                (std::fabs(original[static_cast<size_t>(row) * n + column]) + 1e-10));
        }
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        abortWithMessage("MPI initialization", "MPI_THREAD_FUNNELED support is required", rank);
    }

    int n = 512;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;
    bool showHelp = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const long parsed = std::strtol(argv[++i], nullptr, 10);
            if (parsed <= 0 || parsed > INT_MAX) {
                parseError = true;
            } else {
                n = static_cast<int>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseError = true;
        }
    }
    if (parseError || showHelp) {
        if (rank == 0) {
            if (parseError) {
                std::printf("Invalid command line\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    int localRank = 0;
    MPI_Comm localComm;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &localComm),
             "creating node-local MPI communicator", rank);
    MPI_Comm_rank(localComm, &localRank);
    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices), "querying CUDA devices", rank);
    if (devices == 0) {
        abortWithMessage("CUDA initialization", "no CUDA device is available", rank);
    }
    const int device = localRank % devices;
    checkCuda(cudaSetDevice(device), "selecting CUDA device", rank);

    const int blockSize = std::min(kPreferredBlockSize, n);
    const int blocks = blockCount(n, blockSize);
    const int localBlocks = localBlockCount(blocks, rank, ranks);
    const size_t localElements = static_cast<size_t>(localBlocks) * blockSize * n;
    if (localElements > static_cast<size_t>(INT_MAX)) {
        abortWithMessage("matrix allocation", "local MPI message exceeds the supported size", rank);
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %d x %d\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI ranks, OpenMP host workers, CUDA block size %d\n",
                    ranks, blockSize);
    }

    std::vector<double> hostMatrix(localElements, 0.0);
    std::vector<double> original;
    if (rank == 0) {
        std::printf("Generating distributed positive definite matrix...\n");
    }
    generateLocalPositiveDefiniteMatrix(hostMatrix, validate ? &original : nullptr,
                                        n, blockSize, blocks, rank, ranks);

    double* deviceMatrix = nullptr;
    double* deviceDiagonal = nullptr;
    double* devicePackedPanel = nullptr;
    double* devicePanel = nullptr;
    int* deviceDisplacements = nullptr;
    int* deviceFailure = nullptr;
    double* hostDiagonal = nullptr;
    double* localPanel = nullptr;
    double* receivedPanel = nullptr;
    const size_t matrixAllocation = std::max<size_t>(1, localElements);
    checkCuda(cudaMalloc(&deviceMatrix, matrixAllocation * sizeof(double)), "allocating device matrix", rank);
    checkCuda(cudaMalloc(&deviceDiagonal, static_cast<size_t>(blockSize) * blockSize * sizeof(double)),
              "allocating device diagonal tile", rank);
    checkCuda(cudaMalloc(&devicePackedPanel, static_cast<size_t>(blocks) * blockSize * blockSize * sizeof(double)),
              "allocating device packed panel", rank);
    checkCuda(cudaMalloc(&devicePanel, static_cast<size_t>(blocks) * blockSize * blockSize * sizeof(double)),
              "allocating device global panel", rank);
    checkCuda(cudaMalloc(&deviceDisplacements, static_cast<size_t>(ranks) * sizeof(int)),
              "allocating device panel displacements", rank);
    checkCuda(cudaMalloc(&deviceFailure, sizeof(int)), "allocating device failure flag", rank);
    const size_t panelCapacity = static_cast<size_t>(blocks) * blockSize * blockSize;
    checkCuda(cudaMallocHost(&hostDiagonal, static_cast<size_t>(blockSize) * blockSize * sizeof(double)),
              "allocating pinned diagonal tile", rank);
    checkCuda(cudaMallocHost(&localPanel,
                             std::max<size_t>(1, static_cast<size_t>(localBlocks) * blockSize * blockSize) *
                                 sizeof(double)),
              "allocating pinned local panel", rank);
    checkCuda(cudaMallocHost(&receivedPanel, panelCapacity * sizeof(double)),
              "allocating pinned gathered panel", rank);
    if (localElements != 0) {
        checkCuda(cudaMemcpy(deviceMatrix, hostMatrix.data(), localElements * sizeof(double), cudaMemcpyHostToDevice),
                  "copying matrix to device", rank);
    }

    std::vector<int> receiveCounts(ranks);
    std::vector<int> receiveDisplacements(ranks);

    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "synchronizing before factorization", rank);
    const double start = MPI_Wtime();
    bool success = true;

    for (int panelBlock = 0; panelBlock < blocks; ++panelBlock) {
        const int owner = panelBlock % ranks;
        const int diagonalRows = std::min(blockSize, n - panelBlock * blockSize);
        int failed = 0;
        if (rank == owner) {
            const int ownerLocalBlock = panelBlock / ranks;
            checkCuda(cudaMemset(deviceFailure, 0, sizeof(int)), "clearing device failure flag", rank);
            potrfDiagonalKernel<<<1, blockSize>>>(deviceMatrix, n, blockSize, ownerLocalBlock,
                                                   panelBlock, diagonalRows, deviceFailure);
            checkCuda(cudaGetLastError(), "launching CUDA diagonal factorization", rank);
            checkCuda(cudaMemcpy(&failed, deviceFailure, sizeof(int), cudaMemcpyDeviceToHost),
                      "reading diagonal factorization status", rank);
            std::fill_n(hostDiagonal, static_cast<size_t>(blockSize) * blockSize, 0.0);
            checkCuda(cudaMemcpy2D(hostDiagonal, static_cast<size_t>(blockSize) * sizeof(double),
                                   deviceMatrix + static_cast<size_t>(ownerLocalBlock) * blockSize * n +
                                       panelBlock * blockSize,
                                   static_cast<size_t>(n) * sizeof(double),
                                   static_cast<size_t>(diagonalRows) * sizeof(double), diagonalRows,
                                   cudaMemcpyDeviceToHost),
                      "copying diagonal tile from device", rank);
        }
        checkMpi(MPI_Bcast(&failed, 1, MPI_INT, owner, MPI_COMM_WORLD),
                 "broadcasting diagonal factorization status", rank);
        if (failed != 0) {
            success = false;
            break;
        }
        if (panelBlock + 1 == blocks) {
            continue;
        }
        checkMpi(MPI_Bcast(hostDiagonal, blockSize * blockSize, MPI_DOUBLE, owner, MPI_COMM_WORLD),
                 "broadcasting diagonal tile", rank);
        checkCuda(cudaMemcpy(deviceDiagonal, hostDiagonal,
                             static_cast<size_t>(blockSize) * blockSize * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "copying diagonal tile to device", rank);

        if (localBlocks > 0) {
            trsmPanelKernel<<<localBlocks, blockSize>>>(deviceMatrix, deviceDiagonal, n, blockSize,
                                                         panelBlock, rank, ranks, localBlocks);
            checkCuda(cudaGetLastError(), "launching CUDA panel triangular solve", rank);
        }

        for (int r = 0; r < ranks; ++r) {
            receiveCounts[r] = trailingBlockCount(r, panelBlock, blocks, ranks) * blockSize * blockSize;
            receiveDisplacements[r] = r == 0 ? 0 : receiveDisplacements[r - 1] + receiveCounts[r - 1];
        }
        const int localTrailingBlocks = trailingBlockCount(rank, panelBlock, blocks, ranks);
        const int receiveTotal = receiveDisplacements.back() + receiveCounts.back();
        if (localBlocks > 0) {
            packPanelKernel<<<localBlocks, 256>>>(deviceMatrix, devicePackedPanel, n, blockSize,
                                                   panelBlock, rank, ranks, localBlocks);
            checkCuda(cudaGetLastError(), "launching CUDA panel pack", rank);
        }
        const size_t localPanelElements = static_cast<size_t>(localTrailingBlocks) * blockSize * blockSize;
        if (localPanelElements != 0) {
            checkCuda(cudaMemcpy(localPanel, devicePackedPanel, localPanelElements * sizeof(double),
                                 cudaMemcpyDeviceToHost), "copying local panel from device", rank);
        }
        checkMpi(MPI_Allgatherv(localPanel, static_cast<int>(localPanelElements), MPI_DOUBLE,
                                 receivedPanel, receiveCounts.data(), receiveDisplacements.data(),
                                 MPI_DOUBLE, MPI_COMM_WORLD),
                 "all-gathering distributed panel", rank);
        if (receiveTotal != 0) {
            checkCuda(cudaMemcpy(devicePackedPanel, receivedPanel,
                                 static_cast<size_t>(receiveTotal) * sizeof(double), cudaMemcpyHostToDevice),
                      "copying global panel to device", rank);
        }
        checkCuda(cudaMemcpy(deviceDisplacements, receiveDisplacements.data(),
                             static_cast<size_t>(ranks) * sizeof(int), cudaMemcpyHostToDevice),
                  "copying panel displacements to device", rank);
        if (panelBlock + 1 < blocks) {
            unpackPanelKernel<<<blocks - panelBlock - 1, 256>>>(devicePackedPanel, devicePanel,
                                                                  deviceDisplacements, n, blockSize,
                                                                  panelBlock, blocks, ranks);
            checkCuda(cudaGetLastError(), "launching CUDA panel unpack", rank);
            const dim3 updateThreads(kUpdateTile, kUpdateTile);
            const int tilesPerDimension = (blockSize + kUpdateTile - 1) / kUpdateTile;
            const dim3 updateBlocks(localBlocks, blocks - panelBlock - 1,
                                    tilesPerDimension * tilesPerDimension);
            if (localBlocks > 0) {
                trailingUpdateKernel<<<updateBlocks, updateThreads>>>(deviceMatrix, devicePanel, n, blockSize,
                                                                        panelBlock, rank, ranks, localBlocks);
                checkCuda(cudaGetLastError(), "launching CUDA trailing update", rank);
            }
        }
    }

    if (success) {
        const dim3 zeroThreads(32, 8);
        const dim3 zeroBlocks((n + zeroThreads.x - 1) / zeroThreads.x, localBlocks,
                              (blockSize + zeroThreads.y - 1) / zeroThreads.y);
        if (localBlocks > 0) {
            zeroUpperKernel<<<zeroBlocks, zeroThreads>>>(deviceMatrix, n, blockSize, rank, ranks, localBlocks);
            checkCuda(cudaGetLastError(), "launching CUDA upper-triangle clear", rank);
        }
    }
    checkCuda(cudaDeviceSynchronize(), "synchronizing CUDA factorization", rank);
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "synchronizing after factorization", rank);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    checkMpi(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
             "reducing factorization time", rank);

    if (!success) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed: matrix is not positive definite\n");
        }
        cudaFree(deviceFailure);
        cudaFree(deviceDisplacements);
        cudaFree(devicePanel);
        cudaFree(devicePackedPanel);
        cudaFree(deviceDiagonal);
        cudaFree(deviceMatrix);
        cudaFreeHost(receivedPanel);
        cudaFreeHost(localPanel);
        cudaFreeHost(hostDiagonal);
        MPI_Comm_free(&localComm);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (localElements != 0) {
        checkCuda(cudaMemcpy(hostMatrix.data(), deviceMatrix, localElements * sizeof(double), cudaMemcpyDeviceToHost),
                  "copying factor to host", rank);
    }
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = operations / elapsed / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> globalFactor;
    if (printResults || validate) {
        gatherMatrix(hostMatrix, globalFactor, n, blockSize, blocks, rank, ranks);
    }
    if (printResults && rank == 0) {
        print_results(globalFactor, "CholeskyL");
    }

    int validationPassed = 1;
    if (validate) {
        std::vector<double> globalOriginal;
        gatherMatrix(original, globalOriginal, n, blockSize, blocks, rank, ranks);
        if (rank == 0) {
            std::printf("Validating result...\n");
            validationPassed = validateCholesky(globalFactor, globalOriginal, n) ? 1 : 0;
            std::printf("Validation: %s\n", validationPassed ? "PASSED" : "FAILED");
        }
        checkMpi(MPI_Bcast(&validationPassed, 1, MPI_INT, 0, MPI_COMM_WORLD),
                 "broadcasting validation result", rank);
    }

    checkCuda(cudaFree(deviceFailure), "freeing device failure flag", rank);
    checkCuda(cudaFree(deviceDisplacements), "freeing device panel displacements", rank);
    checkCuda(cudaFree(devicePanel), "freeing device global panel", rank);
    checkCuda(cudaFree(devicePackedPanel), "freeing device packed panel", rank);
    checkCuda(cudaFree(deviceDiagonal), "freeing device diagonal tile", rank);
    checkCuda(cudaFree(deviceMatrix), "freeing device matrix", rank);
    checkCuda(cudaFreeHost(receivedPanel), "freeing pinned gathered panel", rank);
    checkCuda(cudaFreeHost(localPanel), "freeing pinned local panel", rank);
    checkCuda(cudaFreeHost(hostDiagonal), "freeing pinned diagonal tile", rank);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return validationPassed ? EXIT_SUCCESS : EXIT_FAILURE;
}
