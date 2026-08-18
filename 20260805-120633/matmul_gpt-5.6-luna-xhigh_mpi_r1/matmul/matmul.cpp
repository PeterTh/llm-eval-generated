#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization.  Keeping this
// function deterministic lets every rank initialize only the panels it owns.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                    const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

// A balanced block partition.  The first 'remainder' blocks contain one
// additional element, so this also works when N is not divisible by the grid
// dimensions.
size_t blockSize(const size_t n, const int block, const int blockCount) noexcept {
    const size_t base = n / static_cast<size_t>(blockCount);
    const size_t remainder = n % static_cast<size_t>(blockCount);
    return base + (static_cast<size_t>(block) < remainder ? 1 : 0);
}

size_t blockOffset(const size_t n, const int block,
                   const int blockCount) noexcept {
    const size_t base = n / static_cast<size_t>(blockCount);
    const size_t remainder = n % static_cast<size_t>(blockCount);
    return static_cast<size_t>(block) * base +
           std::min(static_cast<size_t>(block), remainder);
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// MPI's classic C bindings use an int count.  Chunking makes the collective
// and point-to-point paths work for large local tiles as well.
constexpr size_t maxMpiCount() noexcept {
    return static_cast<size_t>(std::numeric_limits<int>::max());
}

void broadcastDoubles(double* data, const size_t count, const int root,
                      MPI_Comm communicator) {
    for (size_t offset = 0; offset < count;) {
        const size_t chunk = std::min(maxMpiCount(), count - offset);
        MPI_Bcast(data + offset, static_cast<int>(chunk), MPI_DOUBLE, root,
                  communicator);
        offset += chunk;
    }
}

void sendDoubles(const double* data, const size_t count, const int destination,
                 const int tag, MPI_Comm communicator) {
    for (size_t offset = 0; offset < count;) {
        const size_t chunk = std::min(maxMpiCount(), count - offset);
        MPI_Send(data + offset, static_cast<int>(chunk), MPI_DOUBLE, destination,
                 tag, communicator);
        offset += chunk;
    }
}

void receiveDoubles(double* data, const size_t count, const int source,
                    const int tag, MPI_Comm communicator) {
    for (size_t offset = 0; offset < count;) {
        const size_t chunk = std::min(maxMpiCount(), count - offset);
        MPI_Recv(data + offset, static_cast<int>(chunk), MPI_DOUBLE, source, tag,
                 communicator, MPI_STATUS_IGNORE);
        offset += chunk;
    }
}

void placeTile(std::vector<double>& fullMatrix, const std::vector<double>& tile,
               const size_t N, const int rowBlock, const int columnBlock,
               const int gridRows, const int gridColumns) {
    const size_t rows = blockSize(N, rowBlock, gridRows);
    const size_t columns = blockSize(N, columnBlock, gridColumns);
    const size_t rowStart = blockOffset(N, rowBlock, gridRows);
    const size_t columnStart = blockOffset(N, columnBlock, gridColumns);

    if (rows == 0 || columns == 0) {
        return;
    }

    for (size_t i = 0; i < rows; ++i) {
        std::copy_n(tile.data() + i * columns, columns,
                    fullMatrix.data() + (rowStart + i) * N + columnStart);
    }
}

// Gather rectangular tiles into the global row-major C matrix.  A regular
// MPI_Gatherv cannot describe the gaps between column tiles in row-major
// storage, so the root receives each already-contiguous tile and places it.
void gatherTiles(const std::vector<double>& localMatrix,
                 std::vector<double>& fullMatrix, const size_t N,
                 const int gridRows, const int gridColumns, MPI_Comm cartesian,
                 const int root) {
    int rank = 0;
    int communicatorSize = 0;
    MPI_Comm_rank(cartesian, &rank);
    MPI_Comm_size(cartesian, &communicatorSize);

    if (rank == root) {
        fullMatrix.assign(N * N, 0.0);

        int rootCoordinates[2] = {0, 0};
        MPI_Cart_coords(cartesian, root, 2, rootCoordinates);
        placeTile(fullMatrix, localMatrix, N, rootCoordinates[0],
                  rootCoordinates[1], gridRows, gridColumns);

        std::vector<double> receivedTile;
        for (int source = 0; source < communicatorSize; ++source) {
            if (source == root) {
                continue;
            }

            int coordinates[2] = {0, 0};
            MPI_Cart_coords(cartesian, source, 2, coordinates);
            const size_t rows = blockSize(N, coordinates[0], gridRows);
            const size_t columns = blockSize(N, coordinates[1], gridColumns);
            const size_t tileElements = rows * columns;
            receivedTile.resize(tileElements);
            receiveDoubles(receivedTile.data(), tileElements, source, 31415,
                           cartesian);
            placeTile(fullMatrix, receivedTile, N, coordinates[0],
                      coordinates[1], gridRows, gridColumns);
        }
    } else {
        sendDoubles(localMatrix.data(), localMatrix.size(), root, 31415,
                    cartesian);
    }
}

// The validation uses the same deterministic inputs as the distributed
// computation without requiring the root to retain full copies of A and B.
bool validateResult(const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) *
                            getPseudoRndValue(N, k, j);
            }

            const double actual = C[i * N + j];
            const double relError =
                std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            const unsigned long long parsed =
                std::strtoull(argv[++i], &end, 10);
            if (errno == ERANGE || end == argv[i] || *end != '\0' ||
                parsed > std::numeric_limits<size_t>::max()) {
                parseError = true;
            } else {
                N = static_cast<size_t>(parsed);
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (showHelp || parseError || N == 0 ||
        N > std::numeric_limits<size_t>::max() / N) {
        if (worldRank == 0) {
            if (showHelp && !parseError) {
                printUsage(argv[0]);
            } else if (parseError) {
                printf("Invalid command line arguments.\n");
            } else if (N == 0) {
                printf("Matrix size N must be greater than zero.\n");
            } else {
                printf("Matrix size N is too large.\n");
            }
            if (!showHelp || parseError) {
                printUsage(argv[0]);
            }
        }
        MPI_Finalize();
        return (showHelp && !parseError) ? 0 : 1;
    }

    int worldSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int dimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dimensions);
    const int periods[2] = {0, 0};

    MPI_Comm cartesian = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 1, &cartesian);

    int cartesianRank = 0;
    MPI_Comm_rank(cartesian, &cartesianRank);
    const int origin[2] = {0, 0};
    int root = 0;
    MPI_Cart_rank(cartesian, origin, &root);
    const bool isRoot = cartesianRank == root;

    int coordinates[2] = {0, 0};
    MPI_Cart_coords(cartesian, cartesianRank, 2, coordinates);
    const int rowBlock = coordinates[0];
    const int columnBlock = coordinates[1];

    MPI_Comm rowCommunicator = MPI_COMM_NULL;
    MPI_Comm columnCommunicator = MPI_COMM_NULL;
    // In rowCommunicator, ranks are ordered by column coordinate.  In
    // columnCommunicator, ranks are ordered by row coordinate.
    MPI_Comm_split(cartesian, rowBlock, columnBlock, &rowCommunicator);
    MPI_Comm_split(cartesian, columnBlock, rowBlock, &columnCommunicator);

    if (isRoot) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", worldSize);
        printf("MPI grid: %d x %d\n", dimensions[0], dimensions[1]);
        printf("Initializing matrices...\n");
    }

    const size_t localRows = blockSize(N, rowBlock, dimensions[0]);
    const size_t localColumns =
        blockSize(N, columnBlock, dimensions[1]);
    const size_t rowStart = blockOffset(N, rowBlock, dimensions[0]);
    const size_t columnStart =
        blockOffset(N, columnBlock, dimensions[1]);

    // Use a panel owner for each grid coordinate.  The least common multiple
    // gives both dimensions a compatible, balanced sequence of K panels.
    const size_t panelCount = std::lcm(static_cast<size_t>(dimensions[0]),
                                       static_cast<size_t>(dimensions[1]));
    std::vector<size_t> panelStarts(panelCount);
    std::vector<size_t> panelLengths(panelCount);
    size_t largestPanel = 0;
    for (size_t panel = 0; panel < panelCount; ++panel) {
        panelStarts[panel] = blockOffset(N, static_cast<int>(panel),
                                         static_cast<int>(panelCount));
        panelLengths[panel] = blockSize(N, static_cast<int>(panel),
                                        static_cast<int>(panelCount));
        largestPanel = std::max(largestPanel, panelLengths[panel]);
    }

    std::vector<size_t> aOffsets(panelCount, 0);
    std::vector<size_t> bOffsets(panelCount, 0);
    size_t aStorageElements = 0;
    size_t bStorageElements = 0;
    for (size_t panel = 0; panel < panelCount; ++panel) {
        if (static_cast<int>(panel % static_cast<size_t>(dimensions[1])) ==
            columnBlock) {
            aOffsets[panel] = aStorageElements;
            aStorageElements += localRows * panelLengths[panel];
        }
        if (static_cast<int>(panel % static_cast<size_t>(dimensions[0])) ==
            rowBlock) {
            bOffsets[panel] = bStorageElements;
            bStorageElements += panelLengths[panel] * localColumns;
        }
    }

    std::vector<double> aStorage(aStorageElements);
    std::vector<double> bStorage(bStorageElements);

    // Initialize only the A panels and B panels owned by this rank.  The
    // panel-major layout makes each broadcast buffer contiguous.
    for (size_t panel = 0; panel < panelCount; ++panel) {
        const size_t panelStart = panelStarts[panel];
        const size_t panelLength = panelLengths[panel];

        if (localRows != 0 && panelLength != 0 &&
            static_cast<int>(panel % static_cast<size_t>(dimensions[1])) ==
                columnBlock) {
            double* aPanel = aStorage.data() + aOffsets[panel];
            for (size_t i = 0; i < localRows; ++i) {
                for (size_t k = 0; k < panelLength; ++k) {
                    aPanel[i * panelLength + k] =
                        getPseudoRndValue(N, rowStart + i, panelStart + k);
                }
            }
        }

        if (localColumns != 0 && panelLength != 0 &&
            static_cast<int>(panel % static_cast<size_t>(dimensions[0])) ==
                rowBlock) {
            double* bPanel = bStorage.data() + bOffsets[panel];
            for (size_t k = 0; k < panelLength; ++k) {
                for (size_t j = 0; j < localColumns; ++j) {
                    bPanel[k * localColumns + j] = getPseudoRndValue(
                        N, panelStart + k, columnStart + j);
                }
            }
        }
    }

    std::vector<double> localC(localRows * localColumns, 0.0);
    std::vector<double> aPanel(localRows * largestPanel);
    std::vector<double> bPanel(largestPanel * localColumns);

    if (isRoot) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(cartesian);
    const double start = MPI_Wtime();

    for (size_t panel = 0; panel < panelCount; ++panel) {
        const size_t panelLength = panelLengths[panel];
        const size_t aCount = localRows * panelLength;
        const size_t bCount = panelLength * localColumns;

        double* aBuffer = nullptr;
        if (aCount != 0) {
            if (static_cast<int>(panel % static_cast<size_t>(dimensions[1])) ==
                columnBlock) {
                aBuffer = aStorage.data() + aOffsets[panel];
            } else {
                aBuffer = aPanel.data();
            }
        }

        double* bBuffer = nullptr;
        if (bCount != 0) {
            if (static_cast<int>(panel % static_cast<size_t>(dimensions[0])) ==
                rowBlock) {
                bBuffer = bStorage.data() + bOffsets[panel];
            } else {
                bBuffer = bPanel.data();
            }
        }

        if (dimensions[1] > 1) {
            const int aRoot = static_cast<int>(
                panel % static_cast<size_t>(dimensions[1]));
            broadcastDoubles(aBuffer, aCount, aRoot, rowCommunicator);
        }
        if (dimensions[0] > 1) {
            const int bRoot = static_cast<int>(
                panel % static_cast<size_t>(dimensions[0]));
            broadcastDoubles(bBuffer, bCount, bRoot, columnCommunicator);
        }

        // K is traversed in increasing global order.  The i-k-j ordering
        // keeps A scalar reuse and makes the B and C inner rows contiguous;
        // each C element still receives terms in exactly the original K order.
        if (localRows != 0 && localColumns != 0 && panelLength != 0) {
            for (size_t i = 0; i < localRows; ++i) {
                double* cRow = localC.data() + i * localColumns;
                for (size_t k = 0; k < panelLength; ++k) {
                    const double a = aBuffer[i * panelLength + k];
                    const double* bRow = bBuffer + k * localColumns;
                    for (size_t j = 0; j < localColumns; ++j) {
                        cRow[j] += a * bRow[j];
                    }
                }
            }
        }
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, root,
               cartesian);

    if (isRoot) {
        const long durationMilliseconds =
            static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMilliseconds);

        const double gflops =
            (2.0 * static_cast<double>(N) * static_cast<double>(N) *
             static_cast<double>(N)) /
            (durationMilliseconds / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> fullC;
    if (printResults || validate) {
        gatherTiles(localC, fullC, N, dimensions[0], dimensions[1], cartesian,
                    root);
    }

    int result = 0;
    if (isRoot && printResults) {
        print_results(fullC, "MatrixC");
    }

    if (validate) {
        if (isRoot) {
            printf("Validating result...\n");
            const bool valid = validateResult(fullC, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
        MPI_Bcast(&result, 1, MPI_INT, root, cartesian);
    }

    MPI_Comm_free(&rowCommunicator);
    MPI_Comm_free(&columnCommunicator);
    MPI_Comm_free(&cartesian);
    MPI_Finalize();
    return result;
}
