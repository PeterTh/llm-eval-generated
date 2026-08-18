#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int kRoot = 0;

// Consecutive row blocks are assigned cyclically.  Compared with a simple
// contiguous row split, this keeps useful work on every rank as the trailing
// matrix shrinks.
struct RowDistribution {
    size_t n;
    size_t blockSize;
    size_t blockCount;
    size_t localRows;
    int rank;
    int ranks;

    RowDistribution(size_t matrixSize, size_t rowBlockSize, int mpiRank, int mpiRanks)
        : n(matrixSize),
          blockSize(rowBlockSize),
          blockCount((matrixSize + rowBlockSize - 1) / rowBlockSize),
          localRows(0),
          rank(mpiRank),
          ranks(mpiRanks) {
        for (size_t block = static_cast<size_t>(rank); block < blockCount;
             block += static_cast<size_t>(ranks)) {
            localRows += rowsInBlock(block);
        }
    }

    size_t rowsInBlock(size_t block) const {
        const size_t first = block * blockSize;
        return std::min(blockSize, n - first);
    }

    int ownerOfBlock(size_t block) const {
        return static_cast<int>(block % static_cast<size_t>(ranks));
    }

    int ownerOfRow(size_t row) const {
        return ownerOfBlock(row / blockSize);
    }

    size_t localIndex(size_t row) const {
        const size_t block = row / blockSize;
        return (block / static_cast<size_t>(ranks)) * blockSize + row % blockSize;
    }
};

size_t chooseBlockSize(size_t n, int ranks) {
    if (n == 0) {
        return 1;
    }

    // Aim for at least two blocks per rank for load balance, while capping
    // panels at a cache-friendly width.  Powers of two make the inner kernels
    // and allocation alignment more predictable.
    const size_t processCount = static_cast<size_t>(ranks);
    const size_t desired = (n + 2 * processCount - 1) / (2 * processCount);
    size_t blockSize = 1;
    while (blockSize < desired && blockSize < 64) {
        blockSize *= 2;
    }
    return std::min(blockSize, n);
}

inline double dotProduct(const double* first, const double* second, size_t count) {
    double sum = 0.0;
    for (size_t k = 0; k < count; ++k) {
        sum += first[k] * second[k];
    }
    return sum;
}

// Generate exactly the same B rows as the sequential benchmark on every rank,
// but retain only locally owned rows.  A is formed only in its used lower
// triangle, so both B and A consume O(n^2 / p) memory per rank.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const RowDistribution& dist) {
    const size_t n = dist.n;
    std::vector<double> localB(dist.localRows * n);
    std::vector<double> generatedRow(n);
    unsigned int seed = 42;

    for (size_t globalRow = 0; globalRow < n; ++globalRow) {
        for (size_t k = 0; k < n; ++k) {
            generatedRow[k] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        }
        if (dist.ownerOfRow(globalRow) == dist.rank) {
            const size_t localRow = dist.localIndex(globalRow);
            std::copy(generatedRow.begin(), generatedRow.end(), localB.begin() + localRow * n);
        }
    }

    seed = 42;
    for (size_t column = 0; column < n; ++column) {
        for (size_t k = 0; k < n; ++k) {
            generatedRow[k] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        }

        const size_t firstBlock = column / dist.blockSize;
        for (size_t block = static_cast<size_t>(dist.rank); block < dist.blockCount;
             block += static_cast<size_t>(dist.ranks)) {
            if (block < firstBlock) {
                continue;
            }
            const size_t firstRow = block * dist.blockSize;
            const size_t endRow = firstRow + dist.rowsInBlock(block);
            for (size_t globalRow = std::max(firstRow, column); globalRow < endRow; ++globalRow) {
                const size_t localRow = dist.localIndex(globalRow);
                A[localRow * n + column] =
                    dotProduct(localB.data() + localRow * n, generatedRow.data(), n);
            }
        }
    }

    for (size_t block = static_cast<size_t>(dist.rank); block < dist.blockCount;
         block += static_cast<size_t>(dist.ranks)) {
        const size_t firstRow = block * dist.blockSize;
        const size_t endRow = firstRow + dist.rowsInBlock(block);
        for (size_t globalRow = firstRow; globalRow < endRow; ++globalRow) {
            const size_t localRow = dist.localIndex(globalRow);
            A[localRow * n + globalRow] += static_cast<double>(n);
        }
    }
}

size_t rowsOwnedFromBlock(const RowDistribution& dist, int rank, size_t firstBlock) {
    size_t rows = 0;
    size_t block = firstBlock;
    const size_t rankValue = static_cast<size_t>(rank);
    const size_t processCount = static_cast<size_t>(dist.ranks);
    const size_t remainder = block % processCount;
    if (remainder <= rankValue) {
        block += rankValue - remainder;
    } else {
        block += processCount - (remainder - rankValue);
    }
    for (; block < dist.blockCount; block += processCount) {
        rows += dist.rowsInBlock(block);
    }
    return rows;
}

// Blocked right-looking Cholesky.  A diagonal tile is factored by its owner,
// the distributed panel is assembled with one collective, and every rank then
// updates only its own block rows.  The communication volume is O(n^2), while
// the O(n^3 / p) trailing update is contiguous and vectorizable.
bool choleskyDecomposition(std::vector<double>& A, const RowDistribution& dist,
                           MPI_Comm communicator, size_t& badDiagonal) {
    const size_t n = dist.n;
    const size_t blockSize = dist.blockSize;
    std::vector<int> receiveCounts(static_cast<size_t>(dist.ranks));
    std::vector<int> displacements(static_cast<size_t>(dist.ranks));
    std::vector<double> diagonalBlock(blockSize * blockSize);
    std::vector<double> packedPanel;
    std::vector<double> gatheredPanel;
    std::vector<double> panel;
    packedPanel.reserve(dist.localRows * blockSize);
    gatheredPanel.reserve(n * blockSize);
    panel.reserve(n * blockSize);
    badDiagonal = std::numeric_limits<size_t>::max();

    for (size_t panelBlock = 0; panelBlock < dist.blockCount; ++panelBlock) {
        const size_t panelStart = panelBlock * blockSize;
        const size_t panelWidth = dist.rowsInBlock(panelBlock);
        const size_t nextPanel = panelStart + panelWidth;
        const int owner = dist.ownerOfBlock(panelBlock);

        if (dist.rank == owner) {
            for (size_t j = 0; j < panelWidth; ++j) {
                const size_t globalJ = panelStart + j;
                double* rowJ = A.data() + dist.localIndex(globalJ) * n;
                const double value = rowJ[globalJ] -
                                     dotProduct(rowJ + panelStart, rowJ + panelStart, j);
                if (!(value > 0.0)) {
                    badDiagonal = globalJ;
                    break;
                }
                rowJ[globalJ] = std::sqrt(value);

                for (size_t i = j + 1; i < panelWidth; ++i) {
                    const size_t globalI = panelStart + i;
                    double* rowI = A.data() + dist.localIndex(globalI) * n;
                    rowI[globalJ] =
                        (rowI[globalJ] -
                         dotProduct(rowI + panelStart, rowJ + panelStart, j)) /
                        rowJ[globalJ];
                }
            }
        }

        uint64_t sharedBadDiagonal = static_cast<uint64_t>(badDiagonal);
        MPI_Bcast(&sharedBadDiagonal, 1, MPI_UINT64_T, owner, communicator);
        badDiagonal = static_cast<size_t>(sharedBadDiagonal);
        if (badDiagonal != std::numeric_limits<size_t>::max()) {
            return false;
        }

        if (dist.rank == owner) {
            std::fill(diagonalBlock.begin(), diagonalBlock.end(), 0.0);
            for (size_t i = 0; i < panelWidth; ++i) {
                const size_t globalI = panelStart + i;
                const double* row = A.data() + dist.localIndex(globalI) * n + panelStart;
                std::copy_n(row, i + 1, diagonalBlock.data() + i * panelWidth);
            }
        }
        MPI_Bcast(diagonalBlock.data(), static_cast<int>(panelWidth * panelWidth), MPI_DOUBLE,
                  owner, communicator);

        // Solve L_ik * L_kk^T = A_ik for locally owned rows below the panel.
        for (size_t block = static_cast<size_t>(dist.rank); block < dist.blockCount;
             block += static_cast<size_t>(dist.ranks)) {
            if (block <= panelBlock) {
                continue;
            }
            const size_t firstRow = block * blockSize;
            const size_t endRow = firstRow + dist.rowsInBlock(block);
            for (size_t globalRow = firstRow; globalRow < endRow; ++globalRow) {
                double* row = A.data() + dist.localIndex(globalRow) * n;
                for (size_t j = 0; j < panelWidth; ++j) {
                    const double* diagonalRow = diagonalBlock.data() + j * panelWidth;
                    row[panelStart + j] =
                        (row[panelStart + j] -
                         dotProduct(row + panelStart, diagonalRow, j)) /
                        diagonalRow[j];
                }
            }
        }

        int gatheredElements = 0;
        bool countOverflow = false;
        for (int process = 0; process < dist.ranks; ++process) {
            const size_t elements = rowsOwnedFromBlock(dist, process, panelBlock) * panelWidth;
            if (elements > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                gatheredElements > std::numeric_limits<int>::max() - static_cast<int>(elements)) {
                countOverflow = true;
                break;
            }
            receiveCounts[static_cast<size_t>(process)] = static_cast<int>(elements);
            displacements[static_cast<size_t>(process)] = gatheredElements;
            gatheredElements += static_cast<int>(elements);
        }
        if (countOverflow) {
            if (dist.rank == kRoot) {
                std::fprintf(stderr, "Error: MPI panel exceeds the implementation count limit\n");
            }
            return false;
        }

        packedPanel.resize(
            static_cast<size_t>(receiveCounts[static_cast<size_t>(dist.rank)]));
        size_t packedOffset = 0;
        for (size_t block = panelBlock; block < dist.blockCount; ++block) {
            if (dist.ownerOfBlock(block) != dist.rank) {
                continue;
            }
            const size_t firstRow = block * blockSize;
            const size_t endRow = firstRow + dist.rowsInBlock(block);
            for (size_t globalRow = firstRow; globalRow < endRow; ++globalRow) {
                const double* row = A.data() + dist.localIndex(globalRow) * n + panelStart;
                std::copy_n(row, panelWidth, packedPanel.data() + packedOffset);
                packedOffset += panelWidth;
            }
        }

        gatheredPanel.resize(static_cast<size_t>(gatheredElements));
        MPI_Allgatherv(packedPanel.data(), receiveCounts[static_cast<size_t>(dist.rank)], MPI_DOUBLE,
                       gatheredPanel.data(), receiveCounts.data(), displacements.data(), MPI_DOUBLE,
                       communicator);

        panel.resize((n - panelStart) * panelWidth);
        for (int process = 0; process < dist.ranks; ++process) {
            size_t sourceOffset = static_cast<size_t>(displacements[static_cast<size_t>(process)]);
            for (size_t block = panelBlock; block < dist.blockCount; ++block) {
                if (dist.ownerOfBlock(block) != process) {
                    continue;
                }
                const size_t firstRow = block * blockSize;
                const size_t endRow = firstRow + dist.rowsInBlock(block);
                for (size_t globalRow = firstRow; globalRow < endRow; ++globalRow) {
                    std::copy_n(gatheredPanel.data() + sourceOffset, panelWidth,
                                panel.data() + (globalRow - panelStart) * panelWidth);
                    sourceOffset += panelWidth;
                }
            }
        }

        // Symmetric rank-k update of the local part of the trailing matrix.
        for (size_t block = static_cast<size_t>(dist.rank); block < dist.blockCount;
             block += static_cast<size_t>(dist.ranks)) {
            if (block <= panelBlock) {
                continue;
            }
            const size_t firstRow = block * blockSize;
            const size_t endRow = firstRow + dist.rowsInBlock(block);
            for (size_t globalRow = firstRow; globalRow < endRow; ++globalRow) {
                double* matrixRow = A.data() + dist.localIndex(globalRow) * n;
                const double* panelRow =
                    panel.data() + (globalRow - panelStart) * panelWidth;
                for (size_t column = nextPanel; column <= globalRow; ++column) {
                    const double* panelColumnRow =
                        panel.data() + (column - panelStart) * panelWidth;
                    matrixRow[column] -= dotProduct(panelRow, panelColumnRow, panelWidth);
                }
            }
        }
    }

    // Match the original dense result representation: L below the diagonal and
    // explicit zeros above it.
    for (size_t block = static_cast<size_t>(dist.rank); block < dist.blockCount;
         block += static_cast<size_t>(dist.ranks)) {
        const size_t firstRow = block * blockSize;
        const size_t endRow = firstRow + dist.rowsInBlock(block);
        for (size_t globalRow = firstRow; globalRow < endRow; ++globalRow) {
            double* row = A.data() + dist.localIndex(globalRow) * n;
            std::fill(row + globalRow + 1, row + n, 0.0);
        }
    }
    return true;
}

bool gatherMatrix(const std::vector<double>& localMatrix, const RowDistribution& dist,
                  std::vector<double>& globalMatrix, MPI_Comm communicator) {
    std::vector<int> counts(static_cast<size_t>(dist.ranks));
    std::vector<int> displacements(static_cast<size_t>(dist.ranks));
    int gatheredElements = 0;
    bool overflow = false;
    for (int process = 0; process < dist.ranks; ++process) {
        const size_t elements = rowsOwnedFromBlock(dist, process, 0) * dist.n;
        if (elements > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            gatheredElements > std::numeric_limits<int>::max() - static_cast<int>(elements)) {
            overflow = true;
            break;
        }
        counts[static_cast<size_t>(process)] = static_cast<int>(elements);
        displacements[static_cast<size_t>(process)] = gatheredElements;
        gatheredElements += static_cast<int>(elements);
    }

    int anyOverflow = overflow ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &anyOverflow, 1, MPI_INT, MPI_MAX, communicator);
    if (anyOverflow != 0) {
        if (dist.rank == kRoot) {
            std::fprintf(stderr, "Error: result matrix exceeds the MPI gather count limit\n");
        }
        return false;
    }

    std::vector<double> gathered;
    if (dist.rank == kRoot) {
        gathered.resize(static_cast<size_t>(gatheredElements));
        globalMatrix.assign(dist.n * dist.n, 0.0);
    }
    MPI_Gatherv(localMatrix.data(), counts[static_cast<size_t>(dist.rank)], MPI_DOUBLE,
                gathered.data(), counts.data(), displacements.data(), MPI_DOUBLE, kRoot,
                communicator);

    if (dist.rank == kRoot) {
        for (int process = 0; process < dist.ranks; ++process) {
            size_t sourceOffset = static_cast<size_t>(displacements[static_cast<size_t>(process)]);
            for (size_t block = static_cast<size_t>(process); block < dist.blockCount;
                 block += static_cast<size_t>(dist.ranks)) {
                const size_t firstRow = block * dist.blockSize;
                const size_t rows = dist.rowsInBlock(block);
                std::copy_n(gathered.data() + sourceOffset, rows * dist.n,
                            globalMatrix.data() + firstRow * dist.n);
                sourceOffset += rows * dist.n;
            }
        }
    }
    return true;
}

void completeSymmetricUpperTriangle(std::vector<double>& matrix, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            matrix[i * n + j] = matrix[j * n + i];
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original,
                      size_t n) {
    std::vector<double> reconstructed(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t terms = std::min(i, j) + 1;
            for (size_t k = 0; k < terms; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    double maxError = 0.0;
    double relativeError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = std::fabs(reconstructed[i] - original[i]);
        maxError = std::max(maxError, error);
        const double relative = error / (std::fabs(original[i]) + 1e-10);
        relativeError = std::max(relativeError, relative);
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relativeError);
    if (relativeError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
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

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    uint64_t matrixSize = 512;
    int validate = 0;
    int printResults = 0;
    int parseResult = 0;
    if (rank == kRoot) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                char* end = nullptr;
                const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
                if (end == argv[i] || *end != '\0') {
                    std::printf("Invalid matrix size: %s\n", argv[i]);
                    parseResult = 1;
                    break;
                }
                matrixSize = static_cast<uint64_t>(parsed);
            } else if (std::strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (std::strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (std::strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseResult = 2;
                break;
            } else {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseResult = 1;
                break;
            }
        }
    }

    MPI_Bcast(&parseResult, 1, MPI_INT, kRoot, MPI_COMM_WORLD);
    if (parseResult != 0) {
        MPI_Finalize();
        return parseResult == 2 ? 0 : 1;
    }
    MPI_Bcast(&matrixSize, 1, MPI_UINT64_T, kRoot, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, kRoot, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, kRoot, MPI_COMM_WORLD);

    if (matrixSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        (matrixSize != 0 && matrixSize >
                                static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
                                    matrixSize)) {
        if (rank == kRoot) {
            std::fprintf(stderr, "Error: matrix size is too large for this platform\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t n = static_cast<size_t>(matrixSize);
    const size_t blockSize = chooseBlockSize(n, ranks);
    const RowDistribution distribution(n, blockSize, rank, ranks);

    if (rank == kRoot) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate != 0 ? "enabled" : "disabled");
        std::printf("MPI processes: %d\n", ranks);
        std::printf("Generating positive definite matrix...\n");
    }

    std::vector<double> matrix(distribution.localRows * n, 0.0);
    generatePositiveDefiniteMatrix(matrix, distribution);
    std::vector<double> original;
    if (validate != 0) {
        original = matrix;
    }

    if (rank == kRoot) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    size_t badDiagonal = std::numeric_limits<size_t>::max();
    const bool success =
        choleskyDecomposition(matrix, distribution, MPI_COMM_WORLD, badDiagonal);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, kRoot, MPI_COMM_WORLD);

    if (!success) {
        if (rank == kRoot) {
            if (badDiagonal != std::numeric_limits<size_t>::max()) {
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                            badDiagonal);
            }
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == kRoot) {
        const long long elapsedMilliseconds = static_cast<long long>(elapsed * 1000.0);
        const double operations = static_cast<double>(n) * static_cast<double>(n) *
                                  static_cast<double>(n) / 3.0;
        const double gflops = elapsed > 0.0
                                  ? operations / elapsed / 1e9
                                  : std::numeric_limits<double>::infinity();
        std::printf("Computation time: %lld ms\n", elapsedMilliseconds);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> result;
    if ((validate != 0 || printResults != 0) &&
        !gatherMatrix(matrix, distribution, result, MPI_COMM_WORLD)) {
        MPI_Finalize();
        return 1;
    }

    std::vector<double> originalGlobal;
    if (validate != 0) {
        if (!gatherMatrix(original, distribution, originalGlobal, MPI_COMM_WORLD)) {
            MPI_Finalize();
            return 1;
        }
        if (rank == kRoot) {
            completeSymmetricUpperTriangle(originalGlobal, n);
        }
    }

    int exitCode = 0;
    if (rank == kRoot) {
        if (printResults != 0) {
            print_results(result, "CholeskyL");
        }
        if (validate != 0) {
            std::printf("Validating result...\n");
            if (validateCholesky(result, originalGlobal, n)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, kRoot, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
