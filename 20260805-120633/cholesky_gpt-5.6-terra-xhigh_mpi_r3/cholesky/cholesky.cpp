#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// The matrix is distributed by cyclic row blocks.  Keeping a complete row on
// its owner makes the trailing update contiguous in memory, while cycling the
// blocks balances the progressively larger updates near the bottom of L.

size_t chooseBlockSize(const size_t n, const int ranks) {
    constexpr size_t minBlockSize = 16;
    constexpr size_t maxBlockSize = 128;
    const size_t target = (n + 4 * static_cast<size_t>(ranks) - 1) /
                          (4 * static_cast<size_t>(ranks));
    return std::min(n, std::clamp(target, minBlockSize, maxBlockSize));
}

std::vector<size_t> makeLocalRows(const size_t n, const size_t blockSize,
                                  const int rank, const int ranks) {
    std::vector<size_t> rows;
    const size_t blocks = (n + blockSize - 1) / blockSize;

    for (size_t block = static_cast<size_t>(rank); block < blocks;
         block += static_cast<size_t>(ranks)) {
        const size_t first = block * blockSize;
        const size_t last = std::min(n, first + blockSize);
        for (size_t row = first; row < last; ++row) {
            rows.push_back(row);
        }
    }
    return rows;
}

std::vector<size_t> makeRowsPerRank(const size_t n, const size_t blockSize,
                                    const int ranks) {
    std::vector<size_t> rowCounts(ranks, 0);
    const size_t blocks = (n + blockSize - 1) / blockSize;

    for (size_t block = 0; block < blocks; ++block) {
        const size_t first = block * blockSize;
        rowCounts[block % static_cast<size_t>(ranks)] +=
            std::min(n, first + blockSize) - first;
    }
    return rowCounts;
}

int mpiCount(const size_t count) {
    // Matrices large enough to exceed an MPI int count require more memory
    // than this benchmark can reasonably allocate on a single rank.
    return static_cast<int>(count);
}

void broadcastDoubles(std::vector<double>& data, const int root,
                      const MPI_Comm communicator) {
    const size_t maxCount = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < data.size(); offset += maxCount) {
        const size_t count = std::min(maxCount, data.size() - offset);
        MPI_Bcast(data.data() + offset, mpiCount(count), MPI_DOUBLE, root,
                  communicator);
    }
}

void generateRandomRows(std::vector<double>& rows, unsigned int& seed) {
    for (double& value : rows) {
        value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }
}

void generateDistributedPositiveDefiniteMatrix(
    std::vector<double>& localMatrix, const std::vector<size_t>& localRows,
    const std::vector<size_t>& localIndexByGlobalRow, const size_t n,
    const size_t blockSize, const int rank) {
    // Retain only the random rows owned by this rank.  Broadcasting the source
    // matrix twice (once to distribute rows and once to form B * B^T) avoids a
    // full n-by-n temporary on every rank and keeps memory scalable.
    std::vector<double> localB(localRows.size() * n);
    std::vector<double> randomRows;
    unsigned int seed = 42;

    for (size_t firstRow = 0; firstRow < n; firstRow += blockSize) {
        const size_t rowCount = std::min(blockSize, n - firstRow);
        randomRows.resize(rowCount * n);
        if (rank == 0) {
            generateRandomRows(randomRows, seed);
        }
        broadcastDoubles(randomRows, 0, MPI_COMM_WORLD);

        for (size_t row = 0; row < rowCount; ++row) {
            const size_t globalRow = firstRow + row;
            const size_t localRow = localIndexByGlobalRow[globalRow];
            if (localRow != n) {
                std::copy_n(randomRows.data() + row * n, n,
                            localB.data() + localRow * n);
            }
        }
    }

    std::fill(localMatrix.begin(), localMatrix.end(), 0.0);
    seed = 42;
    for (size_t firstRow = 0; firstRow < n; firstRow += blockSize) {
        const size_t rowCount = std::min(blockSize, n - firstRow);
        randomRows.resize(rowCount * n);
        if (rank == 0) {
            generateRandomRows(randomRows, seed);
        }
        broadcastDoubles(randomRows, 0, MPI_COMM_WORLD);

        for (size_t localRow = 0; localRow < localRows.size(); ++localRow) {
            const double* const bRow = localB.data() + localRow * n;
            double* const aRow = localMatrix.data() + localRow * n + firstRow;
            for (size_t row = 0; row < rowCount; ++row) {
                const double* const bColumnRow = randomRows.data() + row * n;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += bRow[k] * bColumnRow[k];
                }
                aRow[row] = sum;
            }
        }
    }

    for (size_t localRow = 0; localRow < localRows.size(); ++localRow) {
        localMatrix[localRow * n + localRows[localRow]] +=
            static_cast<double>(n);
    }
}

void clearLocalUpperTriangle(std::vector<double>& localMatrix,
                             const std::vector<size_t>& localRows,
                             const size_t n) {
    for (size_t localRow = 0; localRow < localRows.size(); ++localRow) {
        double* const row = localMatrix.data() + localRow * n;
        std::fill(row + localRows[localRow] + 1, row + n, 0.0);
    }
}

bool factorDiagonalBlock(std::vector<double>& localMatrix,
                         const std::vector<size_t>& localIndexByGlobalRow,
                         const size_t n, const size_t panelStart,
                         const size_t panelWidth, std::vector<double>& diagonal,
                         size_t& failedDiagonal) {
    diagonal.assign(panelWidth * panelWidth, 0.0);

    for (size_t i = 0; i < panelWidth; ++i) {
        const size_t globalRow = panelStart + i;
        double* const matrixRow =
            localMatrix.data() + localIndexByGlobalRow[globalRow] * n;

        for (size_t j = 0; j <= i; ++j) {
            double value = matrixRow[panelStart + j];
            for (size_t k = 0; k < j; ++k) {
                value -= matrixRow[panelStart + k] * diagonal[j * panelWidth + k];
            }

            if (i == j) {
                if (value <= 0.0) {
                    failedDiagonal = globalRow;
                    return false;
                }
                value = std::sqrt(value);
            } else {
                value /= diagonal[j * panelWidth + j];
            }

            matrixRow[panelStart + j] = value;
            diagonal[i * panelWidth + j] = value;
        }
    }
    return true;
}

void solveLocalPanelRows(std::vector<double>& localMatrix,
                         const std::vector<size_t>& localRows,
                         const size_t n, const size_t panelStart,
                         const size_t panelWidth,
                         const std::vector<double>& diagonal) {
    const size_t panelEnd = panelStart + panelWidth;
    const auto first = std::lower_bound(localRows.begin(), localRows.end(), panelEnd);

    for (auto rowIt = first; rowIt != localRows.end(); ++rowIt) {
        const size_t localRow = static_cast<size_t>(rowIt - localRows.begin());
        double* const matrixRow = localMatrix.data() + localRow * n;

        for (size_t column = 0; column < panelWidth; ++column) {
            double value = matrixRow[panelStart + column];
            for (size_t k = 0; k < column; ++k) {
                value -= matrixRow[panelStart + k] *
                         diagonal[column * panelWidth + k];
            }
            matrixRow[panelStart + column] =
                value / diagonal[column * panelWidth + column];
        }
    }
}

void exchangePanelColumns(const std::vector<double>& localMatrix,
                          const std::vector<size_t>& localRows, const size_t n,
                          const size_t blockSize, const size_t panelStart,
                          const size_t panelWidth, const int ranks,
                          std::vector<double>& panelColumns) {
    const size_t remainingRows = n - panelStart;
    const auto first = std::lower_bound(localRows.begin(), localRows.end(), panelStart);
    const size_t localFirst = static_cast<size_t>(first - localRows.begin());
    const size_t localPanelRows = localRows.size() - localFirst;

    std::vector<double> localPacked(localPanelRows * panelWidth);
    for (size_t localRow = localFirst, packedRow = 0; localRow < localRows.size();
         ++localRow, ++packedRow) {
        const double* const source = localMatrix.data() + localRow * n + panelStart;
        std::copy_n(source, panelWidth,
                    localPacked.data() + packedRow * panelWidth);
    }

    std::vector<int> counts(ranks);
    std::vector<int> displacements(ranks);
    size_t packedSize = 0;
    const size_t blocks = (n + blockSize - 1) / blockSize;
    for (int rank = 0; rank < ranks; ++rank) {
        size_t rowsOnRank = 0;
        for (size_t block = static_cast<size_t>(rank); block < blocks;
             block += static_cast<size_t>(ranks)) {
            const size_t firstRow = block * blockSize;
            const size_t lastRow = std::min(n, firstRow + blockSize);
            if (lastRow > panelStart) {
                rowsOnRank += lastRow - std::max(firstRow, panelStart);
            }
        }
        counts[rank] = mpiCount(rowsOnRank * panelWidth);
        displacements[rank] = mpiCount(packedSize);
        packedSize += rowsOnRank * panelWidth;
    }

    std::vector<double> packedPanel(packedSize);
    MPI_Allgatherv(localPacked.data(), mpiCount(localPacked.size()), MPI_DOUBLE,
                   packedPanel.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    // Transpose the received panel once.  The trailing update then becomes a
    // sequence of contiguous SAXPY operations, which is substantially faster
    // than evaluating a small dot product for every output element.
    panelColumns.assign(panelWidth * remainingRows, 0.0);
    for (int rank = 0; rank < ranks; ++rank) {
        size_t packedOffset = static_cast<size_t>(displacements[rank]);
        for (size_t block = static_cast<size_t>(rank); block < blocks;
             block += static_cast<size_t>(ranks)) {
            const size_t firstRow = std::max(block * blockSize, panelStart);
            const size_t lastRow = std::min(n, block * blockSize + blockSize);
            for (size_t row = firstRow; row < lastRow; ++row) {
                const double* const source = packedPanel.data() + packedOffset;
                for (size_t column = 0; column < panelWidth; ++column) {
                    panelColumns[column * remainingRows + row - panelStart] =
                        source[column];
                }
                packedOffset += panelWidth;
            }
        }
    }
}

void updateLocalTrailingMatrix(std::vector<double>& localMatrix,
                               const std::vector<size_t>& localRows,
                               const size_t n, const size_t panelStart,
                               const size_t panelWidth,
                               const std::vector<double>& panelColumns) {
    const size_t panelEnd = panelStart + panelWidth;
    const size_t remainingRows = n - panelStart;
    const auto first = std::lower_bound(localRows.begin(), localRows.end(), panelEnd);

    for (auto rowIt = first; rowIt != localRows.end(); ++rowIt) {
        const size_t localRow = static_cast<size_t>(rowIt - localRows.begin());
        const size_t globalRow = *rowIt;
        double* const matrixRow = localMatrix.data() + localRow * n;
        const size_t width = globalRow - panelEnd + 1;
        double* const destination = matrixRow + panelEnd;

        for (size_t column = 0; column < panelWidth; ++column) {
            const double multiplier = matrixRow[panelStart + column];
            const double* const source =
                panelColumns.data() + column * remainingRows + (panelEnd - panelStart);
            for (size_t j = 0; j < width; ++j) {
                destination[j] -= multiplier * source[j];
            }
        }
    }
}

bool distributedCholesky(std::vector<double>& localMatrix,
                         const std::vector<size_t>& localRows,
                         const std::vector<size_t>& localIndexByGlobalRow,
                         const size_t n, const size_t blockSize, const int rank,
                         const int ranks, size_t& failedDiagonal) {
    std::vector<double> diagonal;
    std::vector<double> panelColumns;

    for (size_t panelStart = 0; panelStart < n; panelStart += blockSize) {
        const size_t panelWidth = std::min(blockSize, n - panelStart);
        const int owner = static_cast<int>((panelStart / blockSize) %
                                           static_cast<size_t>(ranks));
        int success = 1;
        unsigned long long failedDiagonalWire = 0;

        if (rank == owner &&
            !factorDiagonalBlock(localMatrix, localIndexByGlobalRow, n, panelStart,
                                 panelWidth, diagonal, failedDiagonal)) {
            success = 0;
        }
        if (rank == owner) {
            failedDiagonalWire = static_cast<unsigned long long>(failedDiagonal);
        }

        MPI_Bcast(&success, 1, MPI_INT, owner, MPI_COMM_WORLD);
        MPI_Bcast(&failedDiagonalWire, 1, MPI_UNSIGNED_LONG_LONG, owner,
                  MPI_COMM_WORLD);
        failedDiagonal = static_cast<size_t>(failedDiagonalWire);
        if (!success) {
            return false;
        }

        if (rank != owner) {
            diagonal.resize(panelWidth * panelWidth);
        }
        MPI_Bcast(diagonal.data(), mpiCount(diagonal.size()), MPI_DOUBLE, owner,
                  MPI_COMM_WORLD);

        solveLocalPanelRows(localMatrix, localRows, n, panelStart, panelWidth,
                            diagonal);
        exchangePanelColumns(localMatrix, localRows, n, blockSize, panelStart,
                             panelWidth, ranks, panelColumns);
        updateLocalTrailingMatrix(localMatrix, localRows, n, panelStart, panelWidth,
                                  panelColumns);
    }
    return true;
}

void gatherDistributedRows(const std::vector<double>& localMatrix, const size_t n,
                           const size_t blockSize,
                           const std::vector<size_t>& rowsPerRank, const int rank,
                           const int ranks, std::vector<double>& globalMatrix) {
    std::vector<int> counts(ranks);
    std::vector<int> displacements(ranks);
    size_t receivedElements = 0;
    for (int process = 0; process < ranks; ++process) {
        counts[process] = mpiCount(rowsPerRank[process] * n);
        displacements[process] = mpiCount(receivedElements);
        receivedElements += rowsPerRank[process] * n;
    }

    std::vector<double> packedRows;
    if (rank == 0) {
        packedRows.resize(receivedElements);
    }
    MPI_Gatherv(localMatrix.data(), mpiCount(localMatrix.size()), MPI_DOUBLE,
                packedRows.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank != 0) {
        return;
    }

    globalMatrix.assign(n * n, 0.0);
    const size_t blocks = (n + blockSize - 1) / blockSize;
    for (int process = 0; process < ranks; ++process) {
        size_t packedRow = static_cast<size_t>(displacements[process]) / n;
        for (size_t block = static_cast<size_t>(process); block < blocks;
             block += static_cast<size_t>(ranks)) {
            const size_t firstRow = block * blockSize;
            const size_t rowCount = std::min(n, firstRow + blockSize) - firstRow;
            std::copy_n(packedRows.data() + packedRow * n, rowCount * n,
                        globalMatrix.data() + firstRow * n);
            packedRow += rowCount;
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& AOrig,
                      const size_t n) {
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
        const double error = std::fabs(reconstructed[i] - AOrig[i]);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (std::fabs(AOrig[i]) + 1e-10));
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool argumentError = false;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
            break;
        } else {
            argumentError = true;
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            break;
        }
    }

    if (showHelp || argumentError || n == 0) {
        if (rank == 0) {
            if (n == 0 && !showHelp && !argumentError) {
                std::printf("Error: Matrix size must be positive\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentError || n == 0 ? 1 : 0;
    }

    const size_t blockSize = chooseBlockSize(n, ranks);
    const std::vector<size_t> localRows = makeLocalRows(n, blockSize, rank, ranks);
    const std::vector<size_t> rowsPerRank = makeRowsPerRank(n, blockSize, ranks);
    std::vector<size_t> localIndexByGlobalRow(n, n);
    for (size_t localRow = 0; localRow < localRows.size(); ++localRow) {
        localIndexByGlobalRow[localRows[localRow]] = localRow;
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("Generating positive definite matrix...\n");
    }

    std::vector<double> localMatrix(localRows.size() * n);
    generateDistributedPositiveDefiniteMatrix(
        localMatrix, localRows, localIndexByGlobalRow, n, blockSize, rank);

    std::vector<double> localOriginal;
    if (validate) {
        localOriginal = localMatrix;
    }
    clearLocalUpperTriangle(localMatrix, localRows, n);

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    size_t failedDiagonal = 0;
    const bool success = distributedCholesky(
        localMatrix, localRows, localIndexByGlobalRow, n, blockSize, rank, ranks,
        failedDiagonal);
    const double elapsed = MPI_Wtime() - start;

    double maximumElapsed = 0.0;
    MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

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
        const long milliseconds = static_cast<long>(maximumElapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", milliseconds);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = operations / maximumElapsed / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> globalMatrix;
    if (printResults || validate) {
        gatherDistributedRows(localMatrix, n, blockSize, rowsPerRank, rank, ranks,
                              globalMatrix);
    }

    if (printResults && rank == 0) {
        print_results(globalMatrix, "CholeskyL");
    }

    bool valid = true;
    if (validate) {
        std::vector<double> globalOriginal;
        gatherDistributedRows(localOriginal, n, blockSize, rowsPerRank, rank, ranks,
                              globalOriginal);
        if (rank == 0) {
            std::printf("Validating result...\n");
            valid = validateCholesky(globalMatrix, globalOriginal, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        int validationStatus = valid ? 1 : 0;
        MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
        valid = validationStatus != 0;
    }

    MPI_Finalize();
    return valid ? 0 : 1;
}
