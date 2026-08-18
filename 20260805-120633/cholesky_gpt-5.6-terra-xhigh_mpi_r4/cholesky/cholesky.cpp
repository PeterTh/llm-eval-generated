#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

// A block is deliberately kept small enough that its triangular solve and the
// packed panel stay cache-resident.  The final block in a process's row range
// may be smaller so that diagonal blocks are always owned by one process.
constexpr size_t kBlockSize = 128;

const double* mpiData(const std::vector<double>& values) {
    return values.empty() ? nullptr : values.data();
}

double* mpiData(std::vector<double>& values) {
    return values.empty() ? nullptr : values.data();
}

int mpiCount(const size_t count) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "Error: MPI message exceeds the supported count range\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<int>(count);
}

std::vector<size_t> makeRowOffsets(const size_t n, const int processes) {
    std::vector<size_t> offsets(static_cast<size_t>(processes) + 1, 0);
    const size_t baseRows = n / static_cast<size_t>(processes);
    const size_t extraRows = n % static_cast<size_t>(processes);

    for (int rank = 0; rank < processes; ++rank) {
        offsets[static_cast<size_t>(rank) + 1] = offsets[static_cast<size_t>(rank)] +
            baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    }
    return offsets;
}

std::vector<int> makeElementCounts(const std::vector<size_t>& rowOffsets, const size_t n) {
    const int processes = static_cast<int>(rowOffsets.size()) - 1;
    std::vector<int> counts(processes);
    for (int rank = 0; rank < processes; ++rank) {
        counts[rank] = mpiCount((rowOffsets[static_cast<size_t>(rank) + 1] -
                                 rowOffsets[static_cast<size_t>(rank)]) * n);
    }
    return counts;
}

std::vector<int> makeElementDisplacements(const std::vector<size_t>& rowOffsets, const size_t n) {
    const int processes = static_cast<int>(rowOffsets.size()) - 1;
    std::vector<int> displacements(processes);
    for (int rank = 0; rank < processes; ++rank) {
        displacements[rank] = mpiCount(rowOffsets[static_cast<size_t>(rank)] * n);
    }
    return displacements;
}

// Generate exactly the same seeded B matrix as the original program, then
// circulate its distributed row blocks.  Each rank computes only its lower
// triangular rows of A = B * B^T, avoiding an O(n^2) replica on every rank.
void generatePositiveDefiniteMatrix(std::vector<double>& localA,
                                    const size_t n,
                                    const std::vector<size_t>& rowOffsets,
                                    const int rank,
                                    const int processes) {
    const size_t localFirst = rowOffsets[static_cast<size_t>(rank)];
    const size_t localLast = rowOffsets[static_cast<size_t>(rank) + 1];
    const size_t localRows = localLast - localFirst;
    const std::vector<int> counts = makeElementCounts(rowOffsets, n);
    const std::vector<int> displacements = makeElementDisplacements(rowOffsets, n);

    std::vector<double> rootB;
    if (rank == 0) {
        rootB.resize(n * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i) {
            rootB[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        }
    }

    std::vector<double> localB(localRows * n);
    MPI_Scatterv(mpiData(rootB), counts.data(), displacements.data(), MPI_DOUBLE,
                 mpiData(localB), mpiCount(localB.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // B is no longer needed on the root after the scatter; retaining it would
    // defeat the memory benefit of the distributed representation.
    std::vector<double>().swap(rootB);

    // This rank's B rows are needed throughout the ring, while currentB is
    // the moving row block whose columns of A are being formed.
    std::vector<double> currentB = localB;
    int currentOwner = rank;

    for (int step = 0; step < processes; ++step) {
        const size_t currentFirst = rowOffsets[static_cast<size_t>(currentOwner)];
        const size_t currentLast = rowOffsets[static_cast<size_t>(currentOwner) + 1];

        for (size_t localRow = 0; localRow < localRows; ++localRow) {
            const size_t globalRow = localFirst + localRow;
            const size_t lastColumn = std::min(globalRow + 1, currentLast);
            const double* const bRow = localB.data() + localRow * n;
            double* const aRow = localA.data() + localRow * n;

            for (size_t globalColumn = currentFirst; globalColumn < lastColumn; ++globalColumn) {
                const double* const bColumn = currentB.data() + (globalColumn - currentFirst) * n;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += bRow[k] * bColumn[k];
                }
                aRow[globalColumn] = sum;
            }
        }

        if (step + 1 < processes) {
            const int destination = (rank + 1) % processes;
            const int source = (rank + processes - 1) % processes;
            const int nextOwner = (currentOwner + processes - 1) % processes;
            const size_t nextRows = rowOffsets[static_cast<size_t>(nextOwner) + 1] -
                                    rowOffsets[static_cast<size_t>(nextOwner)];
            std::vector<double> incomingB(nextRows * n);

            MPI_Sendrecv(mpiData(currentB), mpiCount(currentB.size()), MPI_DOUBLE, destination, 0,
                         mpiData(incomingB), mpiCount(incomingB.size()), MPI_DOUBLE, source, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            currentB.swap(incomingB);
            currentOwner = nextOwner;
        }
    }

    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const size_t globalRow = localFirst + localRow;
        localA[localRow * n + globalRow] += static_cast<double>(n);
    }
}

bool factorDiagonalBlock(std::vector<double>& localA,
                         const size_t n,
                         const size_t localFirst,
                         const size_t blockStart,
                         const size_t blockSize,
                         std::vector<double>& diagonalBlock,
                         size_t& failingDiagonal) {
    for (size_t j = 0; j < blockSize; ++j) {
        double* const diagonalRow = localA.data() + (blockStart + j - localFirst) * n;
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            const double value = diagonalRow[blockStart + k];
            sum += value * value;
        }

        const double value = diagonalRow[blockStart + j] - sum;
        if (value <= 0.0) {
            failingDiagonal = blockStart + j;
            return false;
        }
        diagonalRow[blockStart + j] = std::sqrt(value);

        for (size_t i = j + 1; i < blockSize; ++i) {
            double* const row = localA.data() + (blockStart + i - localFirst) * n;
            sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[blockStart + k] * diagonalRow[blockStart + k];
            }
            row[blockStart + j] = (row[blockStart + j] - sum) / diagonalRow[blockStart + j];
        }
    }

    diagonalBlock.assign(blockSize * blockSize, 0.0);
    for (size_t i = 0; i < blockSize; ++i) {
        const double* const row = localA.data() + (blockStart + i - localFirst) * n;
        for (size_t j = 0; j <= i; ++j) {
            diagonalBlock[i * blockSize + j] = row[blockStart + j];
        }
    }
    return true;
}

// Solve L_ik L_kk^T = A_ik for all locally owned rows below the panel.
void solvePanel(std::vector<double>& localA,
                const size_t n,
                const size_t localFirst,
                const size_t localLast,
                const size_t blockStart,
                const size_t blockSize,
                const std::vector<double>& diagonalBlock) {
    const size_t panelEnd = blockStart + blockSize;
    const size_t firstRow = std::max(localFirst, panelEnd);

    for (size_t globalRow = firstRow; globalRow < localLast; ++globalRow) {
        double* const row = localA.data() + (globalRow - localFirst) * n;
        for (size_t j = 0; j < blockSize; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[blockStart + k] * diagonalBlock[j * blockSize + k];
            }
            row[blockStart + j] = (row[blockStart + j] - sum) /
                                  diagonalBlock[j * blockSize + j];
        }
    }
}

// Factor the distributed lower triangle using a right-looking blocked
// Cholesky algorithm.  A rank owns complete contiguous matrix rows, so all
// diagonal blocks are local and the only per-panel communication is a
// broadcast of L_kk followed by an all-gather of the computed L_ik blocks.
bool choleskyDecomposition(std::vector<double>& localA,
                           const size_t n,
                           const std::vector<size_t>& rowOffsets,
                           const int rank,
                           const int processes) {
    const size_t localFirst = rowOffsets[static_cast<size_t>(rank)];
    const size_t localLast = rowOffsets[static_cast<size_t>(rank) + 1];
    std::vector<double> diagonalBlock;
    std::vector<double> packedPanel;
    std::vector<double> panelByRow;
    std::vector<double> panelByColumn;

    int owner = 0;
    for (size_t blockStart = 0; blockStart < n;) {
        while (owner + 1 < processes &&
               blockStart >= rowOffsets[static_cast<size_t>(owner) + 1]) {
            ++owner;
        }

        const size_t blockSize = std::min({kBlockSize,
                                           n - blockStart,
                                           rowOffsets[static_cast<size_t>(owner) + 1] - blockStart});
        const size_t panelEnd = blockStart + blockSize;
        int diagonalIsPositive = 1;
        size_t failingDiagonal = 0;

        if (rank == owner) {
            diagonalIsPositive = factorDiagonalBlock(localA, n, localFirst, blockStart, blockSize,
                                                      diagonalBlock, failingDiagonal) ? 1 : 0;
        }
        MPI_Bcast(&diagonalIsPositive, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!diagonalIsPositive) {
            unsigned long long failingDiagonalMessage = static_cast<unsigned long long>(failingDiagonal);
            MPI_Bcast(&failingDiagonalMessage, 1, MPI_UNSIGNED_LONG_LONG, owner, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                       static_cast<size_t>(failingDiagonalMessage));
            }
            return false;
        }

        if (rank != owner) {
            diagonalBlock.resize(blockSize * blockSize);
        }
        MPI_Bcast(mpiData(diagonalBlock), mpiCount(diagonalBlock.size()), MPI_DOUBLE, owner,
                  MPI_COMM_WORLD);

        solvePanel(localA, n, localFirst, localLast, blockStart, blockSize, diagonalBlock);

        // Pack each rank's contiguous global row range of the panel.  MPI's
        // receive displacements place it directly at globalRow * blockSize.
        const size_t packedFirst = std::max(localFirst, panelEnd);
        const size_t packedRows = localLast > packedFirst ? localLast - packedFirst : 0;
        packedPanel.resize(packedRows * blockSize);
        for (size_t localRow = 0; localRow < packedRows; ++localRow) {
            const double* const source = localA.data() + (packedFirst + localRow - localFirst) * n + blockStart;
            std::copy_n(source, blockSize, packedPanel.data() + localRow * blockSize);
        }

        panelByRow.assign(n * blockSize, 0.0);
        std::vector<int> panelCounts(processes);
        std::vector<int> panelDisplacements(processes);
        for (int process = 0; process < processes; ++process) {
            const size_t processFirst = std::max(rowOffsets[static_cast<size_t>(process)], panelEnd);
            const size_t processLast = rowOffsets[static_cast<size_t>(process) + 1];
            const size_t processRows = processLast > processFirst ? processLast - processFirst : 0;
            panelCounts[process] = mpiCount(processRows * blockSize);
            panelDisplacements[process] = mpiCount(processFirst * blockSize);
        }
        MPI_Allgatherv(mpiData(packedPanel), mpiCount(packedPanel.size()), MPI_DOUBLE,
                       mpiData(panelByRow), panelCounts.data(), panelDisplacements.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // Transpose the packed panel once, allowing each rank's rank-b update
        // to stream both the target row and its source panel data contiguously.
        panelByColumn.resize(blockSize * n);
        for (size_t j = panelEnd; j < n; ++j) {
            for (size_t k = 0; k < blockSize; ++k) {
                panelByColumn[k * n + j] = panelByRow[j * blockSize + k];
            }
        }

        const size_t firstUpdateRow = std::max(localFirst, panelEnd);
        for (size_t globalRow = firstUpdateRow; globalRow < localLast; ++globalRow) {
            double* const target = localA.data() + (globalRow - localFirst) * n + panelEnd;
            const double* const rowPanel = panelByRow.data() + globalRow * blockSize;
            const size_t updateColumns = globalRow - panelEnd + 1;

            for (size_t k = 0; k < blockSize; ++k) {
                const double factor = rowPanel[k];
                const double* const source = panelByColumn.data() + k * n + panelEnd;
                for (size_t j = 0; j < updateColumns; ++j) {
                    target[j] -= factor * source[j];
                }
            }
        }

        blockStart = panelEnd;
    }

    return true;
}

void gatherMatrix(const std::vector<double>& localMatrix,
                  std::vector<double>& rootMatrix,
                  const size_t n,
                  const std::vector<size_t>& rowOffsets,
                  const int rank) {
    const std::vector<int> counts = makeElementCounts(rowOffsets, n);
    const std::vector<int> displacements = makeElementDisplacements(rowOffsets, n);
    if (rank == 0) {
        rootMatrix.resize(n * n);
    }
    MPI_Gatherv(mpiData(localMatrix), mpiCount(localMatrix.size()), MPI_DOUBLE,
                mpiData(rootMatrix), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void restoreSymmetry(std::vector<double>& matrix, const size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            matrix[i * n + j] = matrix[j * n + i];
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& AOrig, const size_t n) {
    std::vector<double> reconstructed(n * n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t last = std::min(i, j);
            for (size_t k = 0; k <= last; ++k) {
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
        const double rel = error / (fabs(AOrig[i]) + 1e-10);
        relError = std::max(relError, rel);
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
    int processes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", processes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    const std::vector<size_t> rowOffsets = makeRowOffsets(n, processes);
    const size_t localRows = rowOffsets[static_cast<size_t>(rank) + 1] - rowOffsets[static_cast<size_t>(rank)];
    std::vector<double> localA(localRows * n, 0.0);
    generatePositiveDefiniteMatrix(localA, n, rowOffsets, rank, processes);

    std::vector<double> localAOrig;
    if (validate) {
        localAOrig = localA;
    }

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(localA, n, rowOffsets, rank, processes);
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

    std::vector<double> rootL;
    if (printResults || validate) {
        gatherMatrix(localA, rootL, n, rowOffsets, rank);
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        const double gflops = ops / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(rootL, "CholeskyL");
        }
    }

    if (validate) {
        std::vector<double> rootAOrig;
        gatherMatrix(localAOrig, rootAOrig, n, rowOffsets, rank);
        if (rank == 0) {
            restoreSymmetry(rootAOrig, n);
            printf("Validating result...\n");
            if (validateCholesky(rootL, rootAOrig, n)) {
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
