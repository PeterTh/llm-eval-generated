#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization.  Keep this formula
// identical to the original benchmark so that distributed initialization has
// exactly the same input matrices.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

size_t ceilDivide(const size_t value, const size_t divisor) noexcept {
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

struct Distribution {
    static constexpr size_t noOffset = std::numeric_limits<size_t>::max();

    size_t N;
    size_t blockSize;
    size_t blockCount;
    int processRows;
    int processCols;
    int rowCoordinate;
    int colCoordinate;

    std::vector<size_t> blockExtent;
    std::vector<size_t> ownedRows;
    std::vector<size_t> ownedCols;

    // Offsets into the packed, tile-major local matrices.  An entry is
    // noOffset when the corresponding tile belongs to another rank.
    std::vector<size_t> aOffsets;
    std::vector<size_t> bOffsets;
    std::vector<size_t> cOffsets;
    size_t aElements = 0;
    size_t bElements = 0;
    size_t cElements = 0;

    Distribution(const size_t matrixSize, const size_t tileSize,
                 const int gridRows, const int gridCols,
                 const int gridRow, const int gridCol)
        : N(matrixSize),
          blockSize(tileSize),
          blockCount(ceilDivide(matrixSize, tileSize)),
          processRows(gridRows),
          processCols(gridCols),
          rowCoordinate(gridRow),
          colCoordinate(gridCol),
          blockExtent(blockCount),
          aOffsets(blockCount * blockCount, noOffset),
          bOffsets(blockCount * blockCount, noOffset),
          cOffsets(blockCount * blockCount, noOffset) {
        for (size_t block = 0; block < blockCount; ++block) {
            blockExtent[block] = std::min(blockSize, N - block * blockSize);
        }

        for (size_t blockRow = 0; blockRow < blockCount; ++blockRow) {
            if (static_cast<int>(blockRow % static_cast<size_t>(processRows)) == rowCoordinate) {
                ownedRows.push_back(blockRow);
            }
        }
        for (size_t blockCol = 0; blockCol < blockCount; ++blockCol) {
            if (static_cast<int>(blockCol % static_cast<size_t>(processCols)) == colCoordinate) {
                ownedCols.push_back(blockCol);
            }
        }

        // A(blockRow, blockK) is owned by (blockRow % processRows,
        // blockK % processCols).
        for (const size_t blockRow : ownedRows) {
            for (size_t blockK = 0; blockK < blockCount; ++blockK) {
                if (static_cast<int>(blockK % static_cast<size_t>(processCols)) != colCoordinate) {
                    continue;
                }
                const size_t index = blockRow * blockCount + blockK;
                aOffsets[index] = aElements;
                aElements += blockExtent[blockRow] * blockExtent[blockK];
            }
        }

        // B(blockK, blockCol) is owned by (blockK % processRows,
        // blockCol % processCols).
        for (size_t blockK = 0; blockK < blockCount; ++blockK) {
            if (static_cast<int>(blockK % static_cast<size_t>(processRows)) != rowCoordinate) {
                continue;
            }
            for (const size_t blockCol : ownedCols) {
                const size_t index = blockK * blockCount + blockCol;
                bOffsets[index] = bElements;
                bElements += blockExtent[blockK] * blockExtent[blockCol];
            }
        }

        // C(blockRow, blockCol) is owned by
        // (blockRow % processRows, blockCol % processCols).
        for (const size_t blockRow : ownedRows) {
            for (const size_t blockCol : ownedCols) {
                const size_t index = blockRow * blockCount + blockCol;
                cOffsets[index] = cElements;
                cElements += blockExtent[blockRow] * blockExtent[blockCol];
            }
        }
    }

    size_t tileIndex(const size_t row, const size_t col) const noexcept {
        return row * blockCount + col;
    }
};

void initializeLocalMatrices(const Distribution& distribution,
                             std::vector<double>& localA,
                             std::vector<double>& localB) {
    const size_t nb = distribution.blockCount;
    const size_t tile = distribution.blockSize;

    for (const size_t blockRow : distribution.ownedRows) {
        for (size_t blockK = 0; blockK < nb; ++blockK) {
            const size_t offset = distribution.aOffsets[distribution.tileIndex(blockRow, blockK)];
            if (offset == Distribution::noOffset) {
                continue;
            }

            const size_t rows = distribution.blockExtent[blockRow];
            const size_t width = distribution.blockExtent[blockK];
            for (size_t i = 0; i < rows; ++i) {
                const size_t globalI = blockRow * tile + i;
                for (size_t k = 0; k < width; ++k) {
                    const size_t globalK = blockK * tile + k;
                    localA[offset + i * width + k] =
                        getPseudoRndValue(distribution.N, globalI, globalK);
                }
            }
        }
    }

    for (size_t blockK = 0; blockK < nb; ++blockK) {
        for (const size_t blockCol : distribution.ownedCols) {
            const size_t offset = distribution.bOffsets[distribution.tileIndex(blockK, blockCol)];
            if (offset == Distribution::noOffset) {
                continue;
            }

            const size_t height = distribution.blockExtent[blockK];
            const size_t cols = distribution.blockExtent[blockCol];
            for (size_t k = 0; k < height; ++k) {
                const size_t globalK = blockK * tile + k;
                for (size_t j = 0; j < cols; ++j) {
                    const size_t globalJ = blockCol * tile + j;
                    localB[offset + k * cols + j] =
                        getPseudoRndValue(distribution.N, globalK, globalJ);
                }
            }
        }
    }
}

void distributedMatrixMultiply(const Distribution& distribution,
                               const MPI_Comm rowCommunicator,
                               const MPI_Comm columnCommunicator,
                               std::vector<double>& localA,
                               std::vector<double>& localB,
                               std::vector<double>& localC) {
    const size_t nb = distribution.blockCount;
    const size_t tile = distribution.blockSize;

    // Each rank keeps one panel for every locally owned block row/column.
    // The storage is sized for the largest K panel, avoiding allocations in
    // the timed loop.  On a broadcast root, the existing local tile is used
    // directly as the MPI buffer; other ranks use these panel buffers.
    std::vector<size_t> aPanelOffsets(distribution.ownedRows.size());
    std::vector<size_t> bPanelOffsets(distribution.ownedCols.size());
    size_t aPanelStorageSize = 0;
    for (size_t index = 0; index < distribution.ownedRows.size(); ++index) {
        aPanelOffsets[index] = aPanelStorageSize;
        aPanelStorageSize += distribution.blockExtent[distribution.ownedRows[index]] * tile;
    }
    size_t bPanelStorageSize = 0;
    for (size_t index = 0; index < distribution.ownedCols.size(); ++index) {
        bPanelOffsets[index] = bPanelStorageSize;
        bPanelStorageSize += distribution.blockExtent[distribution.ownedCols[index]] * tile;
    }

    std::vector<double> aPanelStorage(aPanelStorageSize);
    std::vector<double> bPanelStorage(bPanelStorageSize);
    std::vector<double*> aPanels(distribution.ownedRows.size());
    std::vector<double*> bPanels(distribution.ownedCols.size());

    for (size_t blockK = 0; blockK < nb; ++blockK) {
        const int aRoot = static_cast<int>(blockK % static_cast<size_t>(distribution.processCols));
        const int bRoot = static_cast<int>(blockK % static_cast<size_t>(distribution.processRows));
        const size_t kWidth = distribution.blockExtent[blockK];

        // Broadcast A(blockRow, blockK) along the process row.
        for (size_t rowIndex = 0; rowIndex < distribution.ownedRows.size(); ++rowIndex) {
            const size_t blockRow = distribution.ownedRows[rowIndex];
            const size_t count = distribution.blockExtent[blockRow] * kWidth;
            double* panel = aPanelStorage.data() + aPanelOffsets[rowIndex];
            if (distribution.colCoordinate == aRoot) {
                panel = localA.data() +
                        distribution.aOffsets[distribution.tileIndex(blockRow, blockK)];
            }
            aPanels[rowIndex] = panel;
            MPI_Bcast(panel, static_cast<int>(count), MPI_DOUBLE, aRoot, rowCommunicator);
        }

        // Broadcast B(blockK, blockCol) down the process column.
        for (size_t colIndex = 0; colIndex < distribution.ownedCols.size(); ++colIndex) {
            const size_t blockCol = distribution.ownedCols[colIndex];
            const size_t count = kWidth * distribution.blockExtent[blockCol];
            double* panel = bPanelStorage.data() + bPanelOffsets[colIndex];
            if (distribution.rowCoordinate == bRoot) {
                panel = localB.data() +
                        distribution.bOffsets[distribution.tileIndex(blockK, blockCol)];
            }
            bPanels[colIndex] = panel;
            MPI_Bcast(panel, static_cast<int>(count), MPI_DOUBLE, bRoot, columnCommunicator);
        }

        // The K panels are visited in ascending global K order.  Keeping K as
        // the middle loop means every C element receives its additions in the
        // same order as the original i-j-k implementation, while the inner
        // J loop remains contiguous for efficient vectorized updates.
        for (size_t rowIndex = 0; rowIndex < distribution.ownedRows.size(); ++rowIndex) {
            const size_t blockRow = distribution.ownedRows[rowIndex];
            const size_t rows = distribution.blockExtent[blockRow];
            const double* a = aPanels[rowIndex];

            for (size_t colIndex = 0; colIndex < distribution.ownedCols.size(); ++colIndex) {
                const size_t blockCol = distribution.ownedCols[colIndex];
                const size_t cols = distribution.blockExtent[blockCol];
                const double* b = bPanels[colIndex];
                double* c = localC.data() +
                            distribution.cOffsets[distribution.tileIndex(blockRow, blockCol)];

                for (size_t i = 0; i < rows; ++i) {
                    double* cRow = c + i * cols;
                    const double* aRow = a + i * kWidth;
                    for (size_t k = 0; k < kWidth; ++k) {
                        const double aValue = aRow[k];
                        const double* bRow = b + k * cols;
                        for (size_t j = 0; j < cols; ++j) {
                            cRow[j] += aValue * bRow[j];
                        }
                    }
                }
            }
        }
    }
}

void copyLocalResultToGlobal(const Distribution& distribution,
                             const std::vector<double>& localC,
                             std::vector<double>& globalC) {
    const size_t N = distribution.N;
    for (const size_t blockRow : distribution.ownedRows) {
        const size_t rows = distribution.blockExtent[blockRow];
        for (const size_t blockCol : distribution.ownedCols) {
            const size_t cols = distribution.blockExtent[blockCol];
            const size_t localOffset = distribution.cOffsets[distribution.tileIndex(blockRow, blockCol)];
            for (size_t i = 0; i < rows; ++i) {
                const size_t globalRow = blockRow * distribution.blockSize + i;
                const size_t globalCol = blockCol * distribution.blockSize;
                std::copy_n(localC.data() + localOffset + i * cols, cols,
                            globalC.data() + globalRow * N + globalCol);
            }
        }
    }
}

void gatherResult(const Distribution& distribution,
                  const MPI_Comm cartesianCommunicator,
                  const int cartesianRank,
                  const int processCount,
                  const std::vector<double>& localC,
                  std::vector<double>& globalC) {
    constexpr int resultTag = 947;

    if (cartesianRank == 0) {
        globalC.assign(distribution.N * distribution.N, 0.0);
        copyLocalResultToGlobal(distribution, localC, globalC);

        for (int rank = 1; rank < processCount; ++rank) {
            int coordinates[2] = {0, 0};
            MPI_Cart_coords(cartesianCommunicator, rank, 2, coordinates);

            std::vector<int> blockLengths;
            std::vector<MPI_Aint> displacements;
            size_t expectedElements = 0;

            for (size_t blockRow = static_cast<size_t>(coordinates[0]);
                 blockRow < distribution.blockCount;
                 blockRow += static_cast<size_t>(distribution.processRows)) {
                const size_t rows = distribution.blockExtent[blockRow];
                for (size_t blockCol = static_cast<size_t>(coordinates[1]);
                     blockCol < distribution.blockCount;
                     blockCol += static_cast<size_t>(distribution.processCols)) {
                    const size_t cols = distribution.blockExtent[blockCol];
                    for (size_t i = 0; i < rows; ++i) {
                        blockLengths.push_back(static_cast<int>(cols));
                        const size_t elementOffset =
                            (blockRow * distribution.blockSize + i) * distribution.N +
                            blockCol * distribution.blockSize;
                        displacements.push_back(static_cast<MPI_Aint>(
                            elementOffset * sizeof(double)));
                        expectedElements += cols;
                    }
                }
            }

            if (expectedElements == 0) {
                MPI_Recv(nullptr, 0, MPI_DOUBLE, rank, resultTag,
                         cartesianCommunicator, MPI_STATUS_IGNORE);
                continue;
            }

            MPI_Datatype receiveType = MPI_DATATYPE_NULL;
            MPI_Type_create_hindexed(static_cast<int>(blockLengths.size()),
                                     blockLengths.data(), displacements.data(),
                                     MPI_DOUBLE, &receiveType);
            MPI_Type_commit(&receiveType);
            MPI_Recv(globalC.data(), 1, receiveType, rank, resultTag,
                     cartesianCommunicator, MPI_STATUS_IGNORE);
            MPI_Type_free(&receiveType);
        }
    } else {
        MPI_Send(localC.empty() ? nullptr : localC.data(),
                 static_cast<int>(localC.size()), MPI_DOUBLE, 0, resultTag,
                 cartesianCommunicator);
    }
}

struct ValidationFailure {
    int failed = 0;
    size_t row = 0;
    size_t col = 0;
    double expected = 0.0;
    double actual = 0.0;
    double relativeError = 0.0;
};

bool validateResult(const Distribution& distribution,
                    const MPI_Comm cartesianCommunicator,
                    const int cartesianRank,
                    const std::vector<double>& localC,
                    ValidationFailure& failure) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    const size_t N = distribution.N;

    for (const size_t pointRow : checkPoints) {
        for (const size_t pointCol : checkPoints) {
            const size_t row = pointRow % N;
            const size_t col = pointCol % N;
            const size_t blockRow = row / distribution.blockSize;
            const size_t blockCol = col / distribution.blockSize;
            const int ownerCoordinates[2] = {
                static_cast<int>(blockRow % static_cast<size_t>(distribution.processRows)),
                static_cast<int>(blockCol % static_cast<size_t>(distribution.processCols))};
            int owner = 0;
            MPI_Cart_rank(cartesianCommunicator, ownerCoordinates, &owner);
            if (owner != cartesianRank) {
                continue;
            }

            const size_t cols = distribution.blockExtent[blockCol];
            const size_t localOffset = distribution.cOffsets[distribution.tileIndex(blockRow, blockCol)];
            const double actual = localC[localOffset +
                                         (row - blockRow * distribution.blockSize) * cols +
                                         (col - blockCol * distribution.blockSize)];

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, row, k) *
                            getPseudoRndValue(N, k, col);
            }

            const double relativeError = std::abs((actual - expected) / (expected + 1e-10));
            if (relativeError > 1e-6) {
                failure.failed = 1;
                failure.row = row;
                failure.col = col;
                failure.expected = expected;
                failure.actual = actual;
                failure.relativeError = relativeError;
                return false;
            }
        }
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int parseArguments(const int argc, char** argv, size_t& N,
                   bool& validate, bool& printResults) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max()) {
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            return 2;
        } else {
            return 1;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int gridDimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, gridDimensions);
    const int periods[2] = {0, 0};

    MPI_Comm cartesianCommunicator = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, gridDimensions, periods, 0,
                    &cartesianCommunicator);

    int cartesianRank = 0;
    MPI_Comm_rank(cartesianCommunicator, &cartesianRank);
    int coordinates[2] = {0, 0};
    MPI_Cart_coords(cartesianCommunicator, cartesianRank, 2, coordinates);

    MPI_Comm rowCommunicator = MPI_COMM_NULL;
    MPI_Comm columnCommunicator = MPI_COMM_NULL;
    MPI_Comm_split(cartesianCommunicator, coordinates[0], coordinates[1],
                   &rowCommunicator);
    MPI_Comm_split(cartesianCommunicator, coordinates[1], coordinates[0],
                   &columnCommunicator);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    const int parseStatus = parseArguments(argc, argv, N, validate, printResults);
    if (parseStatus != 0) {
        if (cartesianRank == 0) {
            if (parseStatus == 2) {
                printUsage(argv[0]);
            } else {
                printf("Invalid command line arguments.\n");
                printUsage(argv[0]);
            }
        }
        MPI_Comm_free(&rowCommunicator);
        MPI_Comm_free(&columnCommunicator);
        MPI_Comm_free(&cartesianCommunicator);
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    if (N > std::numeric_limits<size_t>::max() / N) {
        if (cartesianRank == 0) {
            printf("Matrix size is too large.\n");
        }
        MPI_Comm_free(&rowCommunicator);
        MPI_Comm_free(&columnCommunicator);
        MPI_Comm_free(&cartesianCommunicator);
        MPI_Finalize();
        return 1;
    }

    if (cartesianRank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d (%d x %d grid)\n", worldSize,
               gridDimensions[0], gridDimensions[1]);
    }

    const size_t largestGridDimension = static_cast<size_t>(
        std::max(gridDimensions[0], gridDimensions[1]));
    // Keep tiles cache-sized while ensuring enough tiles to use every grid
    // coordinate whenever the matrix is large enough to support it.
    const size_t blockSize = std::min<size_t>(256,
                                              std::max<size_t>(1,
                                                  ceilDivide(N, largestGridDimension)));
    const Distribution distribution(N, blockSize, gridDimensions[0], gridDimensions[1],
                                    coordinates[0], coordinates[1]);

    std::vector<double> localA(distribution.aElements);
    std::vector<double> localB(distribution.bElements);
    std::vector<double> localC(distribution.cElements, 0.0);

    if (cartesianRank == 0) {
        printf("Initializing matrices...\n");
    }
    initializeLocalMatrices(distribution, localA, localB);

    if (cartesianRank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(cartesianCommunicator);
    const auto start = std::chrono::high_resolution_clock::now();
    distributedMatrixMultiply(distribution, rowCommunicator, columnCommunicator,
                              localA, localB, localC);
    const auto end = std::chrono::high_resolution_clock::now();

    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double maximumSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maximumSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               cartesianCommunicator);

    if (cartesianRank == 0) {
        const long long durationMilliseconds = static_cast<long long>(maximumSeconds * 1000.0);
        printf("Computation time: %lld ms\n", durationMilliseconds);
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                               static_cast<double>(N)) /
                              maximumSeconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> globalC;
    if (printResults) {
        gatherResult(distribution, cartesianCommunicator, cartesianRank, worldSize,
                     localC, globalC);
        if (cartesianRank == 0) {
            print_results(globalC, "MatrixC");
        }
    }

    int validationPassed = 1;
    if (validate) {
        if (cartesianRank == 0) {
            printf("Validating result...\n");
        }

        ValidationFailure localFailure;
        validateResult(distribution, cartesianCommunicator, cartesianRank,
                       localC, localFailure);
        ValidationFailure firstFailure;
        std::vector<ValidationFailure> allFailures;
        if (cartesianRank == 0) {
            allFailures.resize(static_cast<size_t>(worldSize));
        }
        MPI_Gather(&localFailure, static_cast<int>(sizeof(ValidationFailure)), MPI_BYTE,
                   cartesianRank == 0 ? allFailures.data() : nullptr,
                   static_cast<int>(sizeof(ValidationFailure)), MPI_BYTE, 0,
                   cartesianCommunicator);

        for (const auto& failure : allFailures) {
            if (failure.failed != 0) {
                firstFailure = failure;
                validationPassed = 0;
                break;
            }
        }
        int globalValidationPassed = 0;
        MPI_Allreduce(&validationPassed, &globalValidationPassed, 1, MPI_INT,
                      MPI_MIN, cartesianCommunicator);
        validationPassed = globalValidationPassed;

        if (cartesianRank == 0) {
            if (validationPassed != 0) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       firstFailure.row, firstFailure.col, firstFailure.expected,
                       firstFailure.actual, firstFailure.relativeError);
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Comm_free(&rowCommunicator);
    MPI_Comm_free(&columnCommunicator);
    MPI_Comm_free(&cartesianCommunicator);
    MPI_Finalize();
    return validationPassed == 0 ? 1 : 0;
}
