#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr size_t kPreferredBlockSize = 32;
constexpr size_t kMinimumBlockSize = 16;
constexpr int kDistributionTag = 100;
constexpr int kGatherTag = 101;

// Matrix row blocks are assigned cyclically.  Unlike a contiguous row
// decomposition this keeps the increasingly expensive lower rows balanced
// across ranks throughout the factorization.
struct RowDistribution {
    size_t n;
    size_t blockSize;
    size_t numBlocks;
    int rank;
    int ranks;

    RowDistribution(size_t matrixSize, int mpiRank, int mpiRanks)
        : n(matrixSize),
          blockSize(selectBlockSize(matrixSize, mpiRanks)),
          numBlocks((matrixSize + blockSize - 1) / blockSize),
          rank(mpiRank),
          ranks(mpiRanks) {}

    static size_t selectBlockSize(size_t matrixSize, int mpiRanks) {
        if (matrixSize == 0) {
            return 1;
        }

        const size_t rankCount = static_cast<size_t>(std::max(mpiRanks, 1));
        const size_t targetBlocks = rankCount > std::numeric_limits<size_t>::max() / 4
                                        ? std::numeric_limits<size_t>::max()
                                        : 4 * rankCount;
        const size_t desired = (matrixSize + targetBlocks - 1) / targetBlocks;
        size_t result = std::min(kPreferredBlockSize,
                                 std::max(kMinimumBlockSize, desired));
        result = std::min(result, matrixSize);

        // Do not strand ranks merely because the preferred cache block is
        // larger than the amount of work available per rank.
        if ((matrixSize + result - 1) / result < rankCount && matrixSize >= rankCount) {
            result = (matrixSize + rankCount - 1) / rankCount;
        }
        return std::max<size_t>(result, 1);
    }

    size_t firstRow(size_t block) const { return block * blockSize; }

    size_t rowsInBlock(size_t block) const {
        return std::min(blockSize, n - firstRow(block));
    }

    int owner(size_t block) const {
        return static_cast<int>(block % static_cast<size_t>(ranks));
    }

    bool owns(size_t block) const { return owner(block) == rank; }

    size_t localRowOffset(size_t block) const {
        // Every block preceding the final (possibly short) block is full.
        return (block / static_cast<size_t>(ranks)) * blockSize;
    }

    size_t localRows(int process) const {
        size_t result = 0;
        for (size_t block = static_cast<size_t>(process); block < numBlocks;
             block += static_cast<size_t>(ranks)) {
            result += rowsInBlock(block);
        }
        return result;
    }

    size_t localRows() const { return localRows(rank); }

    double* block(std::vector<double>& matrix, size_t blockIndex) const {
        return matrix.data() + localRowOffset(blockIndex) * n;
    }

    const double* block(const std::vector<double>& matrix, size_t blockIndex) const {
        return matrix.data() + localRowOffset(blockIndex) * n;
    }
};

bool mpiCountFits(size_t count) {
    return count <= static_cast<size_t>(std::numeric_limits<int>::max());
}

// Generate exactly the same deterministic B as the sequential implementation,
// distribute it, then form only the rows of A owned by this rank.  Each row
// block of B is streamed once to its owner and then once to all ranks, so no
// process needs a full input or output matrix during normal benchmark runs.
void generatePositiveDefiniteMatrix(std::vector<double>& localA,
                                    const RowDistribution& dist) {
    std::vector<double> localB(dist.localRows() * dist.n);
    std::vector<double> generatedBlock;
    unsigned int seed = 42;
    if (dist.rank == 0) {
        generatedBlock.resize(dist.blockSize * dist.n);
    }

    // Generate in global row-major order on rank zero to retain the original
    // rand_r sequence, but send each block immediately instead of retaining B.
    for (size_t block = 0; block < dist.numBlocks; ++block) {
        const size_t count = dist.rowsInBlock(block) * dist.n;
        if (dist.rank == 0) {
            for (size_t i = 0; i < count; ++i) {
                generatedBlock[i] =
                    (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            }
            if (dist.owner(block) == 0) {
                std::copy_n(generatedBlock.data(), count, dist.block(localB, block));
            } else {
                MPI_Send(generatedBlock.data(), static_cast<int>(count), MPI_DOUBLE,
                         dist.owner(block), kDistributionTag, MPI_COMM_WORLD);
            }
        } else if (dist.owns(block)) {
            MPI_Recv(dist.block(localB, block), static_cast<int>(count), MPI_DOUBLE, 0,
                     kDistributionTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    localA.assign(dist.localRows() * dist.n, 0.0);
    std::vector<double> receivedB(dist.blockSize * dist.n);

    for (size_t sourceBlock = 0; sourceBlock < dist.numBlocks; ++sourceBlock) {
        const size_t sourceRows = dist.rowsInBlock(sourceBlock);
        const size_t sourceStart = dist.firstRow(sourceBlock);
        double* bRows = dist.owns(sourceBlock) ? dist.block(localB, sourceBlock)
                                               : receivedB.data();
        MPI_Bcast(bRows, static_cast<int>(sourceRows * dist.n), MPI_DOUBLE,
                  dist.owner(sourceBlock), MPI_COMM_WORLD);

        for (size_t localBlock = 0; localBlock < dist.numBlocks; ++localBlock) {
            if (!dist.owns(localBlock)) {
                continue;
            }
            const size_t rows = dist.rowsInBlock(localBlock);
            const double* ownB = dist.block(localB, localBlock);
            double* ownA = dist.block(localA, localBlock);

            for (size_t i = 0; i < rows; ++i) {
                const double* bi = ownB + i * dist.n;
                double* ai = ownA + i * dist.n;
                for (size_t j = 0; j < sourceRows; ++j) {
                    const double* bj = bRows + j * dist.n;
                    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                    for (size_t k = 0; k < dist.n; ++k) {
                        sum += bi[k] * bj[k];
                    }
                    ai[sourceStart + j] = sum;
                }
            }
        }
    }

    // Add the identity contribution exactly once after all streamed B blocks
    // have contributed to A = B * B^T.
    for (size_t localBlock = 0; localBlock < dist.numBlocks; ++localBlock) {
        if (!dist.owns(localBlock)) {
            continue;
        }
        const size_t rows = dist.rowsInBlock(localBlock);
        const size_t globalStart = dist.firstRow(localBlock);
        double* ownA = dist.block(localA, localBlock);
        for (size_t i = 0; i < rows; ++i) {
            ownA[i * dist.n + globalStart + i] += static_cast<double>(dist.n);
        }
    }
}

bool factorDiagonalBlock(double* block, size_t stride, size_t blockSize,
                         size_t globalStart, size_t& failedDiagonal) {
    for (size_t j = 0; j < blockSize; ++j) {
        double* diagonalRow = block + j * stride;
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += diagonalRow[globalStart + k] * diagonalRow[globalStart + k];
        }

        const double value = diagonalRow[globalStart + j] - sum;
        if (!(value > 0.0)) {
            failedDiagonal = globalStart + j;
            return false;
        }
        diagonalRow[globalStart + j] = std::sqrt(value);

        for (size_t i = j + 1; i < blockSize; ++i) {
            double* row = block + i * stride;
            double product = 0.0;
            for (size_t k = 0; k < j; ++k) {
                product += row[globalStart + k] * diagonalRow[globalStart + k];
            }
            row[globalStart + j] =
                (row[globalStart + j] - product) / diagonalRow[globalStart + j];
        }
    }
    return true;
}

// Blocked right-looking Cholesky.  A compact panel is all-gathered once per
// block; its per-tile transposed layout makes each trailing rank-k update walk
// contiguous memory in its innermost loop.
bool choleskyDecomposition(std::vector<double>& localA, const RowDistribution& dist,
                           size_t& failedDiagonal) {
    std::vector<double> diagonal(dist.blockSize * dist.blockSize);
    std::vector<int> receiveCounts(static_cast<size_t>(dist.ranks));
    std::vector<int> displacements(static_cast<size_t>(dist.ranks));
    std::vector<size_t> panelOffsets(dist.numBlocks);
    std::vector<size_t> processOffsets(static_cast<size_t>(dist.ranks));
    std::vector<double> localPanel;
    std::vector<double> globalPanel;

    for (size_t panelBlock = 0; panelBlock < dist.numBlocks; ++panelBlock) {
        const size_t panelStart = dist.firstRow(panelBlock);
        const size_t panelWidth = dist.rowsInBlock(panelBlock);
        const int diagonalOwner = dist.owner(panelBlock);
        int success = 1;
        unsigned long long failed = 0;

        if (dist.rank == diagonalOwner) {
            size_t localFailure = 0;
            double* diagonalBlock = dist.block(localA, panelBlock);
            success = factorDiagonalBlock(diagonalBlock, dist.n, panelWidth,
                                          panelStart, localFailure)
                          ? 1
                          : 0;
            failed = static_cast<unsigned long long>(localFailure);
            if (success) {
                for (size_t i = 0; i < panelWidth; ++i) {
                    for (size_t j = 0; j < panelWidth; ++j) {
                        diagonal[i * panelWidth + j] =
                            j <= i ? diagonalBlock[i * dist.n + panelStart + j] : 0.0;
                    }
                }
            }
        }

        MPI_Bcast(&success, 1, MPI_INT, diagonalOwner, MPI_COMM_WORLD);
        MPI_Bcast(&failed, 1, MPI_UNSIGNED_LONG_LONG, diagonalOwner, MPI_COMM_WORLD);
        if (!success) {
            failedDiagonal = static_cast<size_t>(failed);
            return false;
        }
        MPI_Bcast(diagonal.data(), static_cast<int>(panelWidth * panelWidth), MPI_DOUBLE,
                  diagonalOwner, MPI_COMM_WORLD);

        // Distributed triangular solve for all blocks below the diagonal.
        for (size_t block = panelBlock + 1; block < dist.numBlocks; ++block) {
            if (!dist.owns(block)) {
                continue;
            }
            const size_t rows = dist.rowsInBlock(block);
            double* matrixBlock = dist.block(localA, block);
            for (size_t i = 0; i < rows; ++i) {
                double* row = matrixBlock + i * dist.n + panelStart;
                for (size_t j = 0; j < panelWidth; ++j) {
                    double product = 0.0;
                    for (size_t k = 0; k < j; ++k) {
                        product += row[k] * diagonal[j * panelWidth + k];
                    }
                    row[j] = (row[j] - product) / diagonal[j * panelWidth + j];
                }
            }
        }

        std::fill(receiveCounts.begin(), receiveCounts.end(), 0);
        for (size_t block = panelBlock + 1; block < dist.numBlocks; ++block) {
            receiveCounts[static_cast<size_t>(dist.owner(block))] +=
                static_cast<int>(dist.rowsInBlock(block) * panelWidth);
        }
        int totalPanelValues = 0;
        for (int process = 0; process < dist.ranks; ++process) {
            displacements[static_cast<size_t>(process)] = totalPanelValues;
            totalPanelValues += receiveCounts[static_cast<size_t>(process)];
        }

        localPanel.resize(
            static_cast<size_t>(receiveCounts[static_cast<size_t>(dist.rank)]));
        size_t packedOffset = 0;
        for (size_t block = panelBlock + 1; block < dist.numBlocks; ++block) {
            if (!dist.owns(block)) {
                continue;
            }
            const size_t rows = dist.rowsInBlock(block);
            const double* matrixBlock = dist.block(localA, block);
            // Transpose the narrow panel tile while packing it.
            for (size_t k = 0; k < panelWidth; ++k) {
                for (size_t i = 0; i < rows; ++i) {
                    localPanel[packedOffset++] = matrixBlock[i * dist.n + panelStart + k];
                }
            }
        }

        globalPanel.resize(static_cast<size_t>(totalPanelValues));
        MPI_Allgatherv(localPanel.data(),
                       receiveCounts[static_cast<size_t>(dist.rank)], MPI_DOUBLE,
                       globalPanel.data(), receiveCounts.data(), displacements.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        for (int process = 0; process < dist.ranks; ++process) {
            processOffsets[static_cast<size_t>(process)] =
                static_cast<size_t>(displacements[static_cast<size_t>(process)]);
        }
        for (size_t block = panelBlock + 1; block < dist.numBlocks; ++block) {
            const size_t owner = static_cast<size_t>(dist.owner(block));
            panelOffsets[block] = processOffsets[owner];
            processOffsets[owner] += dist.rowsInBlock(block) * panelWidth;
        }

        // Each rank updates all lower-triangular tiles in its owned block rows.
        for (size_t rowBlock = panelBlock + 1; rowBlock < dist.numBlocks; ++rowBlock) {
            if (!dist.owns(rowBlock)) {
                continue;
            }
            const size_t rowCount = dist.rowsInBlock(rowBlock);
            const size_t rowStart = dist.firstRow(rowBlock);
            double* matrixBlock = dist.block(localA, rowBlock);
            const double* leftPanel = globalPanel.data() + panelOffsets[rowBlock];

            for (size_t columnBlock = panelBlock + 1; columnBlock <= rowBlock;
                 ++columnBlock) {
                const size_t columnCount = dist.rowsInBlock(columnBlock);
                const size_t columnStart = dist.firstRow(columnBlock);
                const double* rightPanel = globalPanel.data() + panelOffsets[columnBlock];

                if (columnBlock < rowBlock) {
                    for (size_t i = 0; i < rowCount; ++i) {
                        double* result = matrixBlock + i * dist.n + columnStart;
                        for (size_t k = 0; k < panelWidth; ++k) {
                            const double multiplier = leftPanel[k * rowCount + i];
                            const double* rhs = rightPanel + k * columnCount;
#pragma omp simd
                            for (size_t j = 0; j < columnCount; ++j) {
                                result[j] -= multiplier * rhs[j];
                            }
                        }
                    }
                } else {
                    // Only the lower half of a diagonal tile is subsequently used.
                    for (size_t i = 0; i < rowCount; ++i) {
                        double* result = matrixBlock + i * dist.n + rowStart;
                        for (size_t k = 0; k < panelWidth; ++k) {
                            const double multiplier = leftPanel[k * rowCount + i];
                            const double* rhs = leftPanel + k * rowCount;
#pragma omp simd
                            for (size_t j = 0; j <= i; ++j) {
                                result[j] -= multiplier * rhs[j];
                            }
                        }
                    }
                }
            }
        }
    }

    // Match the original result representation: L occupies the lower triangle
    // and every element above it is zero.
    for (size_t block = 0; block < dist.numBlocks; ++block) {
        if (!dist.owns(block)) {
            continue;
        }
        const size_t rows = dist.rowsInBlock(block);
        const size_t start = dist.firstRow(block);
        double* matrixBlock = dist.block(localA, block);
        for (size_t i = 0; i < rows; ++i) {
            double* row = matrixBlock + i * dist.n;
            std::fill(row + start + i + 1, row + dist.n, 0.0);
        }
    }
    return true;
}

bool validateCholesky(const std::vector<double>& localL,
                      const std::vector<double>& localOriginal,
                      const RowDistribution& dist) {
    std::vector<double> receivedRows(dist.blockSize * dist.n);
    double localMaxAbsolute = 0.0;
    double localMaxRelative = 0.0;

    for (size_t sourceBlock = 0; sourceBlock < dist.numBlocks; ++sourceBlock) {
        const size_t sourceRows = dist.rowsInBlock(sourceBlock);
        const size_t sourceStart = dist.firstRow(sourceBlock);
        double* source = dist.owns(sourceBlock)
                             ? const_cast<double*>(dist.block(localL, sourceBlock))
                             : receivedRows.data();
        MPI_Bcast(source, static_cast<int>(sourceRows * dist.n), MPI_DOUBLE,
                  dist.owner(sourceBlock), MPI_COMM_WORLD);

        for (size_t localBlock = 0; localBlock < dist.numBlocks; ++localBlock) {
            if (!dist.owns(localBlock)) {
                continue;
            }
            const size_t localRows = dist.rowsInBlock(localBlock);
            const double* lRows = dist.block(localL, localBlock);
            const double* originalRows = dist.block(localOriginal, localBlock);
            for (size_t i = 0; i < localRows; ++i) {
                const double* li = lRows + i * dist.n;
                const double* original = originalRows + i * dist.n;
                for (size_t j = 0; j < sourceRows; ++j) {
                    const double* lj = source + j * dist.n;
                    double sum = 0.0;
                    const size_t terms = std::min(dist.firstRow(localBlock) + i,
                                                  sourceStart + j) +
                                         1;
#pragma omp simd reduction(+ : sum)
                    for (size_t k = 0; k < terms; ++k) {
                        sum += li[k] * lj[k];
                    }
                    const double error = std::fabs(sum - original[sourceStart + j]);
                    const double relative =
                        error / (std::fabs(original[sourceStart + j]) + 1e-10);
                    localMaxAbsolute = std::max(localMaxAbsolute, error);
                    localMaxRelative = std::max(localMaxRelative, relative);
                }
            }
        }
    }

    double localErrors[2] = {localMaxAbsolute, localMaxRelative};
    double globalErrors[2] = {0.0, 0.0};
    MPI_Allreduce(localErrors, globalErrors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (dist.rank == 0) {
        std::printf("Max absolute error: %.10e\n", globalErrors[0]);
        std::printf("Max relative error: %.10e\n", globalErrors[1]);
        if (globalErrors[1] > 1e-6) {
            std::printf("Validation failed: relative error too large\n");
        }
    }
    return globalErrors[1] <= 1e-6;
}

std::vector<double> gatherRowsToRoot(const std::vector<double>& localMatrix,
                                     const RowDistribution& dist) {
    std::vector<double> globalMatrix;
    if (dist.rank == 0) {
        globalMatrix.resize(dist.n * dist.n);
    }

    if (dist.rank == 0) {
        for (size_t block = 0; block < dist.numBlocks; ++block) {
            const size_t count = dist.rowsInBlock(block) * dist.n;
            double* destination = globalMatrix.data() + dist.firstRow(block) * dist.n;
            if (dist.owner(block) == 0) {
                std::copy_n(dist.block(localMatrix, block), count, destination);
            } else {
                MPI_Recv(destination, static_cast<int>(count), MPI_DOUBLE,
                         dist.owner(block), kGatherTag, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (size_t block = 0; block < dist.numBlocks; ++block) {
            if (dist.owns(block)) {
                const size_t count = dist.rowsInBlock(block) * dist.n;
                MPI_Send(dist.block(localMatrix, block), static_cast<int>(count), MPI_DOUBLE,
                         0, kGatherTag, MPI_COMM_WORLD);
            }
        }
    }
    return globalMatrix;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (errno != 0 || end == argv[i] || *end != '\0' || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max()) {
                argumentsValid = false;
            } else {
                n = static_cast<size_t>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }

    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    const RowDistribution dist(n, rank, ranks);
    const bool dimensionsFit =
        n <= std::numeric_limits<size_t>::max() / n &&
        mpiCountFits(dist.blockSize * n) && mpiCountFits(n * dist.blockSize);
    if (!dimensionsFit) {
        if (rank == 0) {
            std::printf("Error: matrix is too large for this MPI implementation\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("Block size: %zu\n", dist.blockSize);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }

    std::vector<double> localA;
    generatePositiveDefiniteMatrix(localA, dist);
    std::vector<double> localOriginal;
    if (validate) {
        localOriginal = localA;
    }

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    size_t failedDiagonal = 0;
    const bool success = choleskyDecomposition(localA, dist, failedDiagonal);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            std::printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                        failedDiagonal);
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double operations = static_cast<double>(n) * static_cast<double>(n) *
                                  static_cast<double>(n) / 3.0;
        const double gflops = elapsed > 0.0 ? operations / elapsed / 1e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> globalA = gatherRowsToRoot(localA, dist);
        if (rank == 0) {
            print_results(globalA, "CholeskyL");
        }
    }

    int returnCode = 0;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        const bool valid = validateCholesky(localA, localOriginal, dist);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        returnCode = valid ? 0 : 1;
    }

    MPI_Finalize();
    return returnCode;
}
