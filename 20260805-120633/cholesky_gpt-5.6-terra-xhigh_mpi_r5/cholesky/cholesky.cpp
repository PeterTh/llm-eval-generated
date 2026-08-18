#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// A row-block-cyclic distribution balances both the trailing updates and the
// ownership of diagonal blocks.  The block size is reduced when necessary so
// that a small matrix still exposes work to all participating ranks.
size_t chooseBlockSize(const size_t n, const int ranks) {
    constexpr size_t preferredBlockSize = 128;
    constexpr size_t minimumBlockSize = 16;

    if (n <= minimumBlockSize || ranks <= 1) {
        return std::min(n, preferredBlockSize);
    }

    const size_t blocksPerRank = 2;
    const size_t scalableBlockSize = n / (blocksPerRank * static_cast<size_t>(ranks));
    return std::min(preferredBlockSize,
                    std::max(minimumBlockSize, scalableBlockSize));
}

std::vector<std::vector<size_t>> makeRowsByRank(const size_t n,
                                                 const size_t blockSize,
                                                 const int ranks) {
    std::vector<std::vector<size_t>> rowsByRank(static_cast<size_t>(ranks));
    for (size_t row = 0; row < n; ++row) {
        const int owner = static_cast<int>((row / blockSize) % static_cast<size_t>(ranks));
        rowsByRank[static_cast<size_t>(owner)].push_back(row);
    }
    return rowsByRank;
}

// Every rank creates only the B rows it owns.  Rank 0 supplies the rand_r
// state at the start of each global row, preserving the exact original random
// sequence without a rank-local n-by-n copy of B.
void generatePositiveDefiniteMatrix(std::vector<double>& localA,
                                    std::vector<double>& localB,
                                    const std::vector<size_t>& localRows,
                                    const std::vector<std::vector<size_t>>& rowsByRank,
                                    const size_t n,
                                    const int rank,
                                    const int ranks,
                                    std::vector<double>& communicationBuffer) {
    std::vector<unsigned int> rowSeeds(n);
    if (rank == 0) {
        unsigned int seed = 42;
        for (size_t row = 0; row < n; ++row) {
            rowSeeds[row] = seed;
            for (size_t column = 0; column < n; ++column) {
                (void)rand_r(&seed);
            }
        }
    }
    MPI_Bcast(rowSeeds.data(), static_cast<int>(n), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    for (size_t localRow = 0; localRow < localRows.size(); ++localRow) {
        unsigned int seed = rowSeeds[localRows[localRow]];
        double* const bRow = localB.data() + localRow * n;
        for (size_t column = 0; column < n; ++column) {
            bRow[column] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        }
    }

    // Stream B block rows through the ranks.  This computes the same dot
    // products as the original generator while retaining O(n^2 / P) storage.
    for (int source = 0; source < ranks; ++source) {
        const std::vector<size_t>& sourceRows = rowsByRank[static_cast<size_t>(source)];
        const size_t elementCount = sourceRows.size() * n;
        double* const sourceB = source == rank ? localB.data() : communicationBuffer.data();

        MPI_Bcast(sourceB, static_cast<int>(elementCount), MPI_DOUBLE, source, MPI_COMM_WORLD);

        for (size_t localRow = 0; localRow < localRows.size(); ++localRow) {
            const double* const left = localB.data() + localRow * n;
            double* const aRow = localA.data() + localRow * n;
            for (size_t sourceRow = 0; sourceRow < sourceRows.size(); ++sourceRow) {
                const double* const right = sourceB + sourceRow * n;
                double sum = 0.0;
                for (size_t column = 0; column < n; ++column) {
                    sum += left[column] * right[column];
                }
                aRow[sourceRows[sourceRow]] = sum;
            }
        }
    }

    for (size_t localRow = 0; localRow < localRows.size(); ++localRow) {
        localA[localRow * n + localRows[localRow]] += static_cast<double>(n);
    }
}

bool factorDiagonalBlock(std::vector<double>& localA,
                         const std::vector<size_t>& globalToLocal,
                         const size_t start,
                         const size_t blockSize,
                         const size_t n,
                         std::vector<double>& diagonalBlock) {
    diagonalBlock.assign(blockSize * blockSize, 0.0);

    for (size_t i = 0; i < blockSize; ++i) {
        double* const row = localA.data() + globalToLocal[start + i] * n;
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[start + k] * localA[(globalToLocal[start + j] * n) + start + k];
            }
            const double value = row[start + j] - sum;

            if (i == j) {
                if (value <= 0.0) {
                    return false;
                }
                row[start + j] = std::sqrt(value);
            } else {
                row[start + j] = value / localA[(globalToLocal[start + j] * n) + start + j];
            }
            diagonalBlock[i * blockSize + j] = row[start + j];
        }
    }
    return true;
}

void solvePanel(std::vector<double>& localA,
                const std::vector<size_t>& localRows,
                const size_t start,
                const size_t blockSize,
                const size_t next,
                const size_t n,
                const std::vector<double>& diagonalBlock) {
    const auto firstTrailing = std::lower_bound(localRows.begin(), localRows.end(), next);
    for (auto rowIt = firstTrailing; rowIt != localRows.end(); ++rowIt) {
        const size_t localRow = static_cast<size_t>(rowIt - localRows.begin());
        double* const row = localA.data() + localRow * n;
        for (size_t column = 0; column < blockSize; ++column) {
            double sum = 0.0;
            for (size_t k = 0; k < column; ++k) {
                sum += row[start + k] * diagonalBlock[column * blockSize + k];
            }
            const double value = row[start + column] - sum;
            row[start + column] = value / diagonalBlock[column * blockSize + column];
        }
    }
}

// Collect the just-computed panel in rank-major order, then transpose it into
// column-major form.  The latter gives unit-stride accesses in the GEMM-like
// trailing update below.
void gatherPanel(const std::vector<double>& localA,
                 const std::vector<size_t>& localRows,
                 const std::vector<std::vector<size_t>>& rowsByRank,
                 const size_t start,
                 const size_t blockSize,
                 const size_t next,
                 const size_t n,
                 const int ranks,
                 std::vector<double>& gatheredPanel,
                 std::vector<double>& panelByColumn) {
    const auto localFirst = std::lower_bound(localRows.begin(), localRows.end(), next);
    const size_t localOffset = static_cast<size_t>(localFirst - localRows.begin());
    const size_t localCount = localRows.size() - localOffset;

    std::vector<double> packedLocalPanel(localCount * blockSize);
    for (size_t row = 0; row < localCount; ++row) {
        const double* const source = localA.data() + (localOffset + row) * n + start;
        std::memcpy(packedLocalPanel.data() + row * blockSize, source, blockSize * sizeof(double));
    }

    std::vector<int> counts(static_cast<size_t>(ranks));
    std::vector<int> displacements(static_cast<size_t>(ranks));
    size_t gatheredRows = 0;
    for (int process = 0; process < ranks; ++process) {
        const std::vector<size_t>& rows = rowsByRank[static_cast<size_t>(process)];
        const auto first = std::lower_bound(rows.begin(), rows.end(), next);
        const size_t count = static_cast<size_t>(rows.end() - first);
        counts[static_cast<size_t>(process)] = static_cast<int>(count * blockSize);
        displacements[static_cast<size_t>(process)] = static_cast<int>(gatheredRows * blockSize);
        gatheredRows += count;
    }

    gatheredPanel.resize(gatheredRows * blockSize);
    MPI_Allgatherv(packedLocalPanel.data(), static_cast<int>(packedLocalPanel.size()), MPI_DOUBLE,
                   gatheredPanel.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    panelByColumn.assign(blockSize * n, 0.0);
    size_t gatheredRow = 0;
    for (int process = 0; process < ranks; ++process) {
        const std::vector<size_t>& rows = rowsByRank[static_cast<size_t>(process)];
        const auto first = std::lower_bound(rows.begin(), rows.end(), next);
        for (auto rowIt = first; rowIt != rows.end(); ++rowIt, ++gatheredRow) {
            for (size_t column = 0; column < blockSize; ++column) {
                panelByColumn[column * n + *rowIt] = gatheredPanel[gatheredRow * blockSize + column];
            }
        }
    }
}

void updateTrailingMatrix(std::vector<double>& localA,
                          const std::vector<size_t>& localRows,
                          const size_t start,
                          const size_t blockSize,
                          const size_t next,
                          const size_t n,
                          const std::vector<double>& panelByColumn) {
    constexpr size_t updateTileSize = 128;
    const auto firstTrailing = std::lower_bound(localRows.begin(), localRows.end(), next);

    for (auto rowIt = firstTrailing; rowIt != localRows.end(); ++rowIt) {
        const size_t localRow = static_cast<size_t>(rowIt - localRows.begin());
        const size_t globalRow = *rowIt;
        double* const row = localA.data() + localRow * n;

        for (size_t columnStart = next; columnStart <= globalRow; columnStart += updateTileSize) {
            const size_t count = std::min(updateTileSize, globalRow - columnStart + 1);
            std::array<double, updateTileSize> updated{};
            std::memcpy(updated.data(), row + columnStart, count * sizeof(double));

            for (size_t k = 0; k < blockSize; ++k) {
                const double multiplier = row[start + k];
                const double* const panelColumn = panelByColumn.data() + k * n + columnStart;
                for (size_t column = 0; column < count; ++column) {
                    updated[column] -= multiplier * panelColumn[column];
                }
            }
            std::memcpy(row + columnStart, updated.data(), count * sizeof(double));
        }
    }
}

bool choleskyDecompositionMPI(std::vector<double>& localA,
                              const std::vector<size_t>& localRows,
                              const std::vector<size_t>& globalToLocal,
                              const std::vector<std::vector<size_t>>& rowsByRank,
                              const size_t n,
                              const size_t blockSize,
                              const int rank,
                              const int ranks) {
    std::vector<double> diagonalBlock;
    std::vector<double> gatheredPanel;
    std::vector<double> panelByColumn;

    for (size_t start = 0; start < n; start += blockSize) {
        const size_t currentBlockSize = std::min(blockSize, n - start);
        const size_t next = start + currentBlockSize;
        const int owner = static_cast<int>((start / blockSize) % static_cast<size_t>(ranks));

        int success = 1;
        if (rank == owner) {
            success = factorDiagonalBlock(localA, globalToLocal, start, currentBlockSize, n, diagonalBlock) ? 1 : 0;
        } else {
            diagonalBlock.resize(currentBlockSize * currentBlockSize);
        }
        MPI_Bcast(&success, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!success) {
            return false;
        }

        MPI_Bcast(diagonalBlock.data(), static_cast<int>(diagonalBlock.size()), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        solvePanel(localA, localRows, start, currentBlockSize, next, n, diagonalBlock);

        if (next < n) {
            gatherPanel(localA, localRows, rowsByRank, start, currentBlockSize, next, n, ranks,
                        gatheredPanel, panelByColumn);
            updateTrailingMatrix(localA, localRows, start, currentBlockSize, next, n, panelByColumn);
        }
    }

    for (size_t localRow = 0; localRow < localRows.size(); ++localRow) {
        double* const row = localA.data() + localRow * n;
        for (size_t column = localRows[localRow] + 1; column < n; ++column) {
            row[column] = 0.0;
        }
    }
    return true;
}

bool validateCholeskyMPI(const std::vector<double>& localL,
                         const std::vector<double>& localOriginal,
                         const std::vector<size_t>& localRows,
                         const std::vector<std::vector<size_t>>& rowsByRank,
                         const size_t n,
                         const int rank,
                         const int ranks,
                         std::vector<double>& communicationBuffer) {
    double localMaxAbsoluteError = 0.0;
    double localMaxRelativeError = 0.0;

    for (int source = 0; source < ranks; ++source) {
        const std::vector<size_t>& sourceRows = rowsByRank[static_cast<size_t>(source)];
        const size_t elementCount = sourceRows.size() * n;
        double* const broadcastBuffer = source == rank ? const_cast<double*>(localL.data()) : communicationBuffer.data();
        MPI_Bcast(broadcastBuffer, static_cast<int>(elementCount), MPI_DOUBLE, source, MPI_COMM_WORLD);
        const double* const sourceL = broadcastBuffer;

        for (size_t localRow = 0; localRow < localRows.size(); ++localRow) {
            const double* const left = localL.data() + localRow * n;
            const double* const original = localOriginal.data() + localRow * n;
            for (size_t sourceRow = 0; sourceRow < sourceRows.size(); ++sourceRow) {
                const double* const right = sourceL + sourceRow * n;
                double reconstructed = 0.0;
                for (size_t column = 0; column < n; ++column) {
                    reconstructed += left[column] * right[column];
                }
                const double error = std::fabs(reconstructed - original[sourceRows[sourceRow]]);
                const double relativeError = error / (std::fabs(original[sourceRows[sourceRow]]) + 1e-10);
                localMaxAbsoluteError = std::max(localMaxAbsoluteError, error);
                localMaxRelativeError = std::max(localMaxRelativeError, relativeError);
            }
        }
    }

    double maxAbsoluteError = 0.0;
    double maxRelativeError = 0.0;
    MPI_Reduce(&localMaxAbsoluteError, &maxAbsoluteError, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMaxRelativeError, &maxRelativeError, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int valid = 1;
    if (rank == 0) {
        std::printf("Max absolute error: %.10e\n", maxAbsoluteError);
        std::printf("Max relative error: %.10e\n", maxRelativeError);
        if (maxRelativeError > 1e-6) {
            std::printf("Validation failed: relative error too large\n");
            valid = 0;
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
}

std::vector<double> gatherMatrixOnRoot(const std::vector<double>& localA,
                                       const std::vector<std::vector<size_t>>& rowsByRank,
                                       const size_t n,
                                       const int rank,
                                       const int ranks) {
    std::vector<int> counts(static_cast<size_t>(ranks));
    std::vector<int> displacements(static_cast<size_t>(ranks));
    size_t elementOffset = 0;
    for (int process = 0; process < ranks; ++process) {
        counts[static_cast<size_t>(process)] = static_cast<int>(rowsByRank[static_cast<size_t>(process)].size() * n);
        displacements[static_cast<size_t>(process)] = static_cast<int>(elementOffset);
        elementOffset += rowsByRank[static_cast<size_t>(process)].size() * n;
    }

    std::vector<double> gathered;
    if (rank == 0) {
        gathered.resize(n * n);
    }
    MPI_Gatherv(localA.data(), static_cast<int>(localA.size()), MPI_DOUBLE,
                gathered.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        return {};
    }

    std::vector<double> matrix(n * n);
    size_t gatheredRow = 0;
    for (int process = 0; process < ranks; ++process) {
        for (const size_t globalRow : rowsByRank[static_cast<size_t>(process)]) {
            std::memcpy(matrix.data() + globalRow * n, gathered.data() + gatheredRow * n, n * sizeof(double));
            ++gatheredRow;
        }
    }
    return matrix;
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
    int parseSuccess = 1;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' || value == 0 ||
                value > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
                parseSuccess = 0;
            } else {
                n = static_cast<size_t>(value);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            parseSuccess = 0;
        }
    }

    int globallyValidArguments = 0;
    MPI_Allreduce(&parseSuccess, &globallyValidArguments, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!globallyValidArguments) {
        if (rank == 0) {
            std::printf("Invalid command line arguments\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t blockSize = chooseBlockSize(n, ranks);
    const std::vector<std::vector<size_t>> rowsByRank = makeRowsByRank(n, blockSize, ranks);
    const std::vector<size_t>& localRows = rowsByRank[static_cast<size_t>(rank)];
    std::vector<size_t> globalToLocal(n, n);
    for (size_t localRow = 0; localRow < localRows.size(); ++localRow) {
        globalToLocal[localRows[localRow]] = localRow;
    }

    const size_t maxLocalRows = std::max_element(rowsByRank.begin(), rowsByRank.end(),
        [](const std::vector<size_t>& left, const std::vector<size_t>& right) {
            return left.size() < right.size();
        })->size();
    std::vector<double> localA(localRows.size() * n);
    std::vector<double> localB(localRows.size() * n);
    std::vector<double> communicationBuffer(maxLocalRows * n);
    std::vector<double> localOriginal;

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(localA, localB, localRows, rowsByRank, n, rank, ranks, communicationBuffer);
    std::vector<double>().swap(localB);

    if (validate) {
        localOriginal = localA;
    }

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool localSuccess = choleskyDecompositionMPI(localA, localRows, globalToLocal, rowsByRank,
                                                       n, blockSize, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    const double localElapsed = MPI_Wtime() - start;

    int successValue = localSuccess ? 1 : 0;
    int success = 0;
    MPI_Allreduce(&successValue, &success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsed * 1000.0);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        const double gflops = elapsed > 0.0 ? ops / elapsed / 1e9 : 0.0;
        std::printf("Computation time: %ld ms\n", milliseconds);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> result = gatherMatrixOnRoot(localA, rowsByRank, n, rank, ranks);
        if (rank == 0) {
            print_results(result, "CholeskyL");
        }
    }

    int validationSuccess = 1;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        const bool valid = validateCholeskyMPI(localA, localOriginal, localRows, rowsByRank,
                                               n, rank, ranks, communicationBuffer);
        validationSuccess = valid ? 1 : 0;
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    MPI_Finalize();
    return validationSuccess ? 0 : 1;
}
