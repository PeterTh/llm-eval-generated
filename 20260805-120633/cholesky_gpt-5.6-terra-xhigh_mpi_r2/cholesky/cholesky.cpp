#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// The matrix is distributed by contiguous groups of block rows.  A block row
// always belongs to one rank, which makes each diagonal block local while the
// trailing update is evenly distributed across all active ranks.
namespace {

constexpr size_t kBlockSize = 64;

struct RowPartition {
    std::vector<size_t> starts;
    std::vector<size_t> counts;
    std::vector<int> matrixCounts;
    std::vector<int> matrixDisplacements;
    size_t blockCount = 0;
    int activeRanks = 0;
};

bool parseMatrixSize(const char* value, size_t& result) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    result = static_cast<size_t>(parsed);
    return true;
}

RowPartition makeRowPartition(const size_t n, const int rankCount) {
    RowPartition partition;
    partition.blockCount = (n + kBlockSize - 1) / kBlockSize;
    partition.activeRanks = partition.blockCount < static_cast<size_t>(rankCount)
        ? static_cast<int>(partition.blockCount)
        : rankCount;
    partition.starts.resize(rankCount);
    partition.counts.resize(rankCount);
    partition.matrixCounts.resize(rankCount);
    partition.matrixDisplacements.resize(rankCount);

    for (int rank = 0; rank < rankCount; ++rank) {
        size_t start = n;
        size_t end = n;
        if (rank < partition.activeRanks) {
            const size_t firstBlock =
                (static_cast<size_t>(rank) * partition.blockCount + partition.activeRanks - 1) /
                partition.activeRanks;
            const size_t pastLastBlock =
                (static_cast<size_t>(rank + 1) * partition.blockCount + partition.activeRanks - 1) /
                partition.activeRanks;
            start = std::min(n, firstBlock * kBlockSize);
            end = std::min(n, pastLastBlock * kBlockSize);
        }

        partition.starts[rank] = start;
        partition.counts[rank] = end - start;
        partition.matrixCounts[rank] = static_cast<int>(partition.counts[rank] * n);
        partition.matrixDisplacements[rank] = static_cast<int>(start * n);
    }

    return partition;
}

int ownerOfBlock(const size_t block, const RowPartition& partition) {
    return static_cast<int>((block * static_cast<size_t>(partition.activeRanks)) /
                            partition.blockCount);
}

// glibc's rand_r advances this LCG three times per result.  Jumping the state
// lets every rank generate exactly its own contiguous B rows without a root
// process materializing or scattering the full random matrix.
unsigned int advanceLcg(unsigned int state, uint64_t steps) {
    constexpr uint32_t multiplier = 1103515245U;
    constexpr uint32_t increment = 12345U;

    uint32_t accumulatedMultiplier = 1U;
    uint32_t accumulatedIncrement = 0U;
    uint32_t powerMultiplier = multiplier;
    uint32_t powerIncrement = increment;

    while (steps != 0) {
        if ((steps & 1U) != 0) {
            accumulatedIncrement = powerMultiplier * accumulatedIncrement + powerIncrement;
            accumulatedMultiplier *= powerMultiplier;
        }
        powerIncrement = powerMultiplier * powerIncrement + powerIncrement;
        powerMultiplier *= powerMultiplier;
        steps >>= 1U;
    }

    return accumulatedMultiplier * state + accumulatedIncrement;
}

bool factorDiagonalBlock(std::vector<double>& localMatrix, const size_t n,
                         const size_t localRowOffset, const size_t globalStart,
                         const size_t width, size_t& failedDiagonal) {
    double* const block = localMatrix.data() + localRowOffset * n + globalStart;

    for (size_t i = 0; i < width; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double value = block[i * n + j];
            for (size_t k = 0; k < j; ++k) {
                value -= block[i * n + k] * block[j * n + k];
            }

            if (i == j) {
                if (value <= 0.0) {
                    failedDiagonal = globalStart + i;
                    return false;
                }
                block[i * n + j] = std::sqrt(value);
            } else {
                block[i * n + j] = value / block[j * n + j];
            }
        }
    }

    return true;
}

void solvePanelRows(std::vector<double>& localMatrix, const size_t n,
                    const size_t localStart, const size_t localCount,
                    const size_t panelStart, const size_t width,
                    const std::vector<double>& diagonal) {
    const size_t firstRow = std::max(localStart, panelStart + width);
    const size_t localEnd = localStart + localCount;

    for (size_t globalRow = firstRow; globalRow < localEnd; ++globalRow) {
        double* const row = localMatrix.data() + (globalRow - localStart) * n + panelStart;
        for (size_t j = 0; j < width; ++j) {
            double value = row[j];
            for (size_t k = 0; k < j; ++k) {
                value -= row[k] * diagonal[j * width + k];
            }
            row[j] = value / diagonal[j * width + j];
        }
    }
}

void generateDistributedPositiveDefiniteMatrix(std::vector<double>& localMatrix,
                                               std::vector<double>* localOriginal,
                                               const size_t n,
                                               const RowPartition& partition,
                                               const int rank, const int rankCount) {
    const size_t localStart = partition.starts[rank];
    const size_t localRows = partition.counts[rank];
    localMatrix.assign(localRows * n, 0.0);
    if (localOriginal != nullptr) {
        localOriginal->assign(localRows * n, 0.0);
    }

    // Preserve the original benchmark's B values while generating only the
    // local rows.  The LCG state has three transitions for each rand_r call.
    const uint64_t valuesBefore = static_cast<uint64_t>(localStart) * n;
    unsigned int seed = advanceLcg(42U, valuesBefore * 3U);
    std::vector<double> localB(localRows * n);
    for (double& value : localB) {
        value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    size_t maximumRows = 0;
    for (const size_t rows : partition.counts) {
        maximumRows = std::max(maximumRows, rows);
    }
    std::vector<double> currentB(maximumRows * n);
    std::vector<double> receivedB(maximumRows * n);
    if (!localB.empty()) {
        std::copy(localB.begin(), localB.end(), currentB.begin());
    }

    const int nextRank = (rank + 1) % rankCount;
    const int previousRank = (rank + rankCount - 1) % rankCount;
    int currentOwner = rank;

    // Ring B through the ranks.  This is a distributed B * B^T generation:
    // it requires O(n^2) communication and O(n^2 / P) B storage per rank.
    for (int step = 0; step < rankCount; ++step) {
        const size_t sourceStart = partition.starts[currentOwner];
        const size_t sourceRows = partition.counts[currentOwner];
        const size_t sourceEnd = sourceStart + sourceRows;

        for (size_t localRow = 0; localRow < localRows; ++localRow) {
            const size_t globalRow = localStart + localRow;
            const size_t firstColumn = sourceStart;
            const size_t pastLastColumn = std::min(sourceEnd, globalRow + 1);
            const double* const bRow = localB.data() + localRow * n;
            double* const aRow = localMatrix.data() + localRow * n;
            double* const originalRow = localOriginal == nullptr
                ? nullptr
                : localOriginal->data() + localRow * n;

            for (size_t globalColumn = firstColumn; globalColumn < pastLastColumn; ++globalColumn) {
                const double* const bColumn = currentB.data() +
                    (globalColumn - sourceStart) * n;
                double value = 0.0;
#pragma GCC ivdep
                for (size_t k = 0; k < n; ++k) {
                    value += bRow[k] * bColumn[k];
                }
                aRow[globalColumn] = value;
                if (originalRow != nullptr) {
                    originalRow[globalColumn] = value;
                }
            }
        }

        const int sendCount = static_cast<int>(partition.counts[currentOwner] * n);
        const int receiveOwner = (currentOwner + rankCount - 1) % rankCount;
        const int receiveCount = static_cast<int>(partition.counts[receiveOwner] * n);
        MPI_Sendrecv(currentB.data(), sendCount, MPI_DOUBLE, nextRank, 0,
                     receivedB.data(), receiveCount, MPI_DOUBLE, previousRank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        currentB.swap(receivedB);
        currentOwner = receiveOwner;
    }

    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const size_t globalRow = localStart + localRow;
        localMatrix[localRow * n + globalRow] += static_cast<double>(n);
        if (localOriginal != nullptr) {
            (*localOriginal)[localRow * n + globalRow] += static_cast<double>(n);
        }
    }
}

bool distributedCholesky(std::vector<double>& localMatrix, const size_t n,
                         const RowPartition& partition, const int rank,
                         const int rankCount) {
    const size_t localStart = partition.starts[rank];
    const size_t localRows = partition.counts[rank];

    for (size_t block = 0; block < partition.blockCount; ++block) {
        const size_t panelStart = block * kBlockSize;
        const size_t width = std::min(kBlockSize, n - panelStart);
        const int owner = ownerOfBlock(block, partition);
        std::vector<double> diagonal(width * width, 0.0);

        unsigned long long failedDiagonal = static_cast<unsigned long long>(n);
        if (rank == owner) {
            const size_t localOffset = panelStart - localStart;
            size_t failed = n;
            if (!factorDiagonalBlock(localMatrix, n, localOffset, panelStart, width, failed)) {
                failedDiagonal = static_cast<unsigned long long>(failed);
            }

            for (size_t i = 0; i < width; ++i) {
                const double* const source = localMatrix.data() + (localOffset + i) * n + panelStart;
                std::copy_n(source, i + 1, diagonal.data() + i * width);
            }
        }

        unsigned long long globalFailure = 0;
        MPI_Allreduce(&failedDiagonal, &globalFailure, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                      MPI_COMM_WORLD);
        if (globalFailure != static_cast<unsigned long long>(n)) {
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %llu\n",
                       globalFailure);
            }
            return false;
        }

        MPI_Bcast(diagonal.data(), static_cast<int>(width * width), MPI_DOUBLE, owner,
                  MPI_COMM_WORLD);

        if (panelStart + width == n) {
            continue;
        }

        solvePanelRows(localMatrix, n, localStart, localRows, panelStart, width, diagonal);

        std::vector<int> panelCounts(rankCount);
        std::vector<int> panelDisplacements(rankCount);
        for (int process = 0; process < rankCount; ++process) {
            const size_t firstRow = std::max(partition.starts[process], panelStart);
            const size_t processEnd = partition.starts[process] + partition.counts[process];
            const size_t pastLastRow = std::max(firstRow, processEnd);
            panelCounts[process] = static_cast<int>((pastLastRow - firstRow) * width);
            panelDisplacements[process] = static_cast<int>((firstRow - panelStart) * width);
        }

        const size_t localEnd = localStart + localRows;
        const size_t localPanelStart = std::max(localStart, panelStart);
        const size_t localPanelEnd = std::max(localPanelStart, localEnd);
        const size_t localPanelRows = localPanelEnd - localPanelStart;
        std::vector<double> localPanel(localPanelRows * width);
        for (size_t row = 0; row < localPanelRows; ++row) {
            const double* const source = localMatrix.data() +
                (localPanelStart + row - localStart) * n + panelStart;
            std::copy_n(source, width, localPanel.data() + row * width);
        }

        const size_t panelRows = n - panelStart;
        std::vector<double> panel(panelRows * width);
        MPI_Allgatherv(localPanel.data(), static_cast<int>(localPanel.size()), MPI_DOUBLE,
                       panel.data(), panelCounts.data(), panelDisplacements.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        const size_t firstUpdatedRow = std::max(localStart, panelStart + width);
        const size_t firstUpdatedColumn = panelStart + width;

        // Rank-width trailing update.  Keeping the panel rows contiguous makes
        // this inner product vectorizable; it also updates only locally owned
        // rows, so no remote matrix writes or synchronization are required.
        for (size_t globalRow = firstUpdatedRow; globalRow < localEnd; ++globalRow) {
            const double* const panelRow = panel.data() + (globalRow - panelStart) * width;
            double* const matrixRow = localMatrix.data() + (globalRow - localStart) * n;
            for (size_t column = firstUpdatedColumn; column <= globalRow; ++column) {
                const double* const panelColumn = panel.data() + (column - panelStart) * width;
                double update = 0.0;
#pragma GCC ivdep
                for (size_t k = 0; k < width; ++k) {
                    update += panelRow[k] * panelColumn[k];
                }
                matrixRow[column] -= update;
            }
        }
    }

    return true;
}

void gatherRows(const std::vector<double>& localMatrix, std::vector<double>& completeMatrix,
                const size_t n, const RowPartition& partition, const int rank) {
    if (rank == 0) {
        completeMatrix.resize(n * n);
    }
    MPI_Gatherv(localMatrix.data(), static_cast<int>(localMatrix.size()), MPI_DOUBLE,
                rank == 0 ? completeMatrix.data() : nullptr,
                partition.matrixCounts.data(), partition.matrixDisplacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
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
        const double error = fabs(reconstructed[i] - AOrig[i]);
        maxError = std::max(maxError, error);
        const double relative = error / (fabs(AOrig[i]) + 1e-10);
        relError = std::max(relError, relative);
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
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

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int rankCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool parseSucceeded = true;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseMatrixSize(argv[++i], n)) {
                parseSucceeded = false;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseSucceeded = false;
        }
    }

    const size_t maxMpiElements = static_cast<size_t>(std::numeric_limits<int>::max());
    if (n > maxMpiElements / n) {
        parseSucceeded = false;
    }

    if (showHelp || !parseSucceeded) {
        if (rank == 0) {
            if (!parseSucceeded) {
                printf("Invalid matrix size or option\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseSucceeded ? 0 : 1;
    }

    const RowPartition partition = makeRowPartition(n, rankCount);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", rankCount);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    std::vector<double> localMatrix;
    std::vector<double> localOriginal;
    generateDistributedPositiveDefiniteMatrix(localMatrix, validate ? &localOriginal : nullptr,
                                              n, partition, rank, rankCount);

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = distributedCholesky(localMatrix, n, partition, rank, rankCount);
    const double localDuration = MPI_Wtime() - start;

    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const long durationMilliseconds = static_cast<long>(duration * 1000.0);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = duration > 0.0 ? operations / duration / 1e9 : 0.0;
        printf("Computation time: %ld ms\n", durationMilliseconds);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> completeMatrix;
    if (printResults || validate) {
        gatherRows(localMatrix, completeMatrix, n, partition, rank);
    }
    if (printResults && rank == 0) {
        print_results(completeMatrix, "CholeskyL");
    }

    int exitCode = 0;
    if (validate) {
        std::vector<double> completeOriginal;
        gatherRows(localOriginal, completeOriginal, n, partition, rank);
        if (rank == 0) {
            // Only the lower triangle was generated on each rank; the source
            // matrix is symmetric, so restore the original full matrix here.
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < i; ++j) {
                    completeOriginal[j * n + i] = completeOriginal[i * n + j];
                }
            }
            printf("Validating result...\n");
            if (validateCholesky(completeMatrix, completeOriginal, n)) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
