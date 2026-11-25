#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/hash.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr size_t MAX_BLOCK_SIZE = 64;

struct Distribution {
    size_t nodes;
    size_t blockSize;
    int processes;

    size_t blocks() const {
        return (nodes + blockSize - 1) / blockSize;
    }

    size_t width(size_t block) const {
        return std::min(blockSize, nodes - block * blockSize);
    }

    int owner(size_t block) const {
        return block % processes;
    }

    size_t offset(size_t block) const {
        return (block / processes) * blockSize;
    }

    size_t localSize(int coordinate) const {
        const size_t fullBlocks = nodes / blockSize;
        return ((fullBlocks + processes - 1 - coordinate) / processes) * blockSize
             + (owner(fullBlocks) == coordinate ? nodes % blockSize : 0);
    }

    size_t globalIndex(size_t local, int coordinate) const {
        return ((local / blockSize) * processes + coordinate) * blockSize
             + local % blockSize;
    }
};

// Break up large transfers to stay within the MPI int count interface.
constexpr size_t MPI_CHUNK = 1 << 28;

void sendBand(const unsigned int* data, size_t count, int destination) {
    for (size_t i = 0; i < count; i += MPI_CHUNK) {
        MPI_Send(data + i, std::min(MPI_CHUNK, count - i), MPI_UNSIGNED,
                 destination, 0, MPI_COMM_WORLD);
    }
}

void receiveBand(unsigned int* data, size_t count, int source) {
    for (size_t i = 0; i < count; i += MPI_CHUNK) {
        MPI_Recv(data + i, std::min(MPI_CHUNK, count - i), MPI_UNSIGNED,
                 source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

unsigned int randomDistance(unsigned int& seed) {
    return 1 + static_cast<unsigned int>(
        static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
}

#ifdef __GLIBC__
// glibc rand_r advances its 32-bit LCG three times per call. Compose the
// affine transition by squaring to reach any index in the serial stream.
unsigned int advanceSeed(unsigned int seed, size_t count) {
    uint32_t multiplier = 1103515245u;
    uint32_t increment = 12345u;
    uint32_t state = seed;
    for (size_t steps = count * 3; steps != 0; steps >>= 1) {
        if (steps & 1) {
            state = multiplier * state + increment;
        }
        increment *= multiplier + 1;
        multiplier *= multiplier;
    }
    return state;
}
#endif

void initializeMatrices(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                        const Distribution& rows, const Distribution& cols,
                        int row, int col, [[maybe_unused]] int rank) {
    const size_t localRows = rows.localSize(row);
    const size_t localCols = cols.localSize(col);
    if (localCols == 0 || localRows == 0) {
        return;
    }

#ifdef __GLIBC__
    for (size_t i = 0; i < localRows; ++i) {
        const size_t globalRow = rows.globalIndex(i, row);
        for (size_t j = 0; j < localCols; j += cols.blockSize) {
            const size_t globalCol = cols.globalIndex(j, col);
            unsigned int seed = advanceSeed(42, globalRow * cols.nodes + globalCol);
            const size_t width = std::min(cols.blockSize, localCols - j);
            for (size_t k = 0; k < width; ++k) {
                dist[i * localCols + j + k] = randomDistance(seed);
            }
        }
    }
#else
    // Preserve the host libc's rand_r sequence on platforms without glibc.
    // The root streams one band at a time to the ranks that own it.
    unsigned int seed = 42;
    std::vector<unsigned int> band(rank == 0 ? rows.blockSize * cols.nodes : 0);
    std::vector<unsigned int> packed(rank == 0 ? band.size() : 0);
    for (size_t block = 0; block < rows.blocks(); ++block) {
        const size_t height = rows.width(block);
        if (rank == 0) {
            for (size_t i = 0; i < height * cols.nodes; ++i) {
                band[i] = randomDistance(seed);
            }
            for (int procCol = 0; procCol < cols.processes; ++procCol) {
                const int owner = rows.owner(block) * cols.processes + procCol;
                const size_t stride = cols.localSize(procCol);
                if (stride == 0) {
                    continue;
                }
                auto* dest = owner == 0 ? dist.data() + rows.offset(block) * stride
                                        : packed.data();
                for (size_t j = 0; j < stride; j += cols.blockSize) {
                    const size_t globalCol = cols.globalIndex(j, procCol);
                    const size_t width = std::min(cols.blockSize, stride - j);
                    for (size_t i = 0; i < height; ++i) {
                        std::copy_n(band.data() + i * cols.nodes + globalCol, width,
                                    dest + i * stride + j);
                    }
                }
                if (owner != 0) {
                    sendBand(dest, height * stride, owner);
                }
            }
        } else if (row == rows.owner(block)) {
            receiveBand(dist.data() + rows.offset(block) * localCols, height * localCols, 0);
        }
    }
#endif

    for (size_t i = 0; i < localRows; ++i) {
        const size_t globalRow = rows.globalIndex(i, row);
        std::fill_n(path.data() + i * localCols, localCols, globalRow);
        const size_t block = globalRow / cols.blockSize;
        if (cols.owner(block) == col) {
            const size_t j = cols.offset(block) + globalRow % cols.blockSize;
            dist[i * localCols + j] = 0;
        }
    }
}

inline void relax(unsigned int* __restrict dist, unsigned int* __restrict path,
                  const unsigned int* __restrict pivotRow, size_t width,
                  unsigned int toPivot, unsigned int pivot) {
    for (size_t j = 0; j < width; ++j) {
        const unsigned int candidate = toPivot + pivotRow[j];
        const bool shorter = candidate < dist[j];
        path[j] = shorter ? pivot : path[j];
        dist[j] = shorter ? candidate : dist[j];
    }
}

void broadcastPanels(unsigned int* pivotColumns, size_t columnCount,
                     unsigned int* pivotRows, size_t rowCount, int rowRoot, int colRoot,
                     MPI_Comm rowComm, MPI_Comm colComm) {
    // Both directions can progress together on independent communicators.
    for (size_t i = 0; i < std::max(columnCount, rowCount); i += MPI_CHUNK) {
        MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        if (i < columnCount) {
            MPI_Ibcast(pivotColumns + i, std::min(MPI_CHUNK, columnCount - i),
                       MPI_UNSIGNED, colRoot, rowComm, &requests[0]);
        }
        if (i < rowCount) {
            MPI_Ibcast(pivotRows + i, std::min(MPI_CHUNK, rowCount - i),
                       MPI_UNSIGNED, rowRoot, colComm, &requests[1]);
        }
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const Distribution& rows, const Distribution& cols,
                   int row, int col, MPI_Comm rowComm, MPI_Comm colComm) {
    const size_t localRows = rows.localSize(row);
    const size_t localCols = cols.localSize(col);
    const size_t blockSize = rows.blockSize;
    std::vector<unsigned int> diagonalRows(blockSize * blockSize);
    std::vector<unsigned int> diagonalCols(blockSize * blockSize);
    std::vector<unsigned int> pivotRows(blockSize * localCols);
    std::vector<unsigned int> pivotCols(blockSize * localRows);
    std::vector<unsigned int> outgoingMins(blockSize * ((localCols + blockSize - 1) / blockSize));

    for (size_t block = 0; block < rows.blocks(); ++block) {
        const int rowRoot = rows.owner(block);
        const int colRoot = cols.owner(block);
        const size_t first = block * blockSize;
        const size_t width = rows.width(block);
        const size_t rowOffset = rows.offset(block);
        const size_t colOffset = cols.offset(block);

        // Capture the pivot row and column at each k before proceeding to k+1.
        // Using these snapshots in the later phases preserves strict tie handling.
        if (row == rowRoot && col == colRoot) {
            for (size_t k = 0; k < width; ++k) {
                auto* savedRow = diagonalRows.data() + k * width;
                std::copy_n(dist.data() + (rowOffset + k) * localCols + colOffset,
                            width, savedRow);
                for (size_t i = 0; i < width; ++i) {
                    const size_t offset = (rowOffset + i) * localCols + colOffset;
                    const unsigned int toPivot = dist[offset + k];
                    diagonalCols[k * width + i] = toPivot;
                    relax(dist.data() + offset, path.data() + offset,
                          savedRow, width, toPivot, first + k);
                }
            }
        }

        broadcastPanels(diagonalCols.data(), row == rowRoot ? width * width : 0,
                        diagonalRows.data(), col == colRoot ? width * width : 0,
                        rowRoot, colRoot, rowComm, colComm);

        if (row == rowRoot && localCols != 0) {
            for (size_t j = 0; j < localCols; j += blockSize) {
                if (col == colRoot && j == colOffset) {
                    continue;
                }
                const size_t tileWidth = std::min(blockSize, localCols - j);
                for (size_t k = 0; k < width; ++k) {
                    auto* savedRow = pivotRows.data() + j * width + k * tileWidth;
                    std::copy_n(dist.data() + (rowOffset + k) * localCols + j,
                                tileWidth, savedRow);
                    for (size_t i = 0; i < width; ++i) {
                        const size_t offset = (rowOffset + i) * localCols + j;
                        relax(dist.data() + offset, path.data() + offset, savedRow,
                              tileWidth, diagonalCols[k * width + i], first + k);
                    }
                }
            }
        }

        if (col == colRoot) {
            for (size_t i = 0; i < localRows; i += blockSize) {
                if (row == rowRoot && i == rowOffset) {
                    continue;
                }
                const size_t height = std::min(blockSize, localRows - i);
                for (size_t k = 0; k < width; ++k) {
                    const auto* savedRow = diagonalRows.data() + k * width;
                    auto* savedCol = pivotCols.data() + i * width + k * height;
                    for (size_t ii = 0; ii < height; ++ii) {
                        const size_t offset = (i + ii) * localCols + colOffset;
                        const unsigned int toPivot = dist[offset + k];
                        savedCol[ii] = toPivot;
                        relax(dist.data() + offset, path.data() + offset,
                              savedRow, width, toPivot, first + k);
                    }
                }
            }
        }

        broadcastPanels(pivotCols.data(), width * localRows, pivotRows.data(),
                        width * localCols, rowRoot, colRoot, rowComm, colComm);

        // The minimum in each saved row bounds all candidates for that tile.
        for (size_t j = 0; j < localCols; j += blockSize) {
            if (col == colRoot && j == colOffset) {
                continue;
            }
            const size_t tileWidth = std::min(blockSize, localCols - j);
            for (size_t k = 0; k < width; ++k) {
                const auto* values = pivotRows.data() + j * width + k * tileWidth;
                outgoingMins[j / blockSize * width + k] =
                    *std::min_element(values, values + tileWidth);
            }
        }

        // Apply all k in this pivot block while each destination tile is hot.
        for (size_t i = 0; i < localRows; i += blockSize) {
            if (row == rowRoot && i == rowOffset) {
                continue;
            }
            const size_t height = std::min(blockSize, localRows - i);
            for (size_t j = 0; j < localCols; j += blockSize) {
                if (col == colRoot && j == colOffset) {
                    continue;
                }
                const size_t tileWidth = std::min(blockSize, localCols - j);
                // A row's starting maximum bounds every later distance.
                unsigned int rowMax[MAX_BLOCK_SIZE];
                for (size_t ii = 0; ii < height; ++ii) {
                    const auto* values = dist.data() + (i + ii) * localCols + j;
                    rowMax[ii] = *std::max_element(values, values + tileWidth);
                }
                for (size_t k = 0; k < width; ++k) {
                    const auto* savedRow = pivotRows.data() + j * width + k * tileWidth;
                    const auto* savedCol = pivotCols.data() + i * width + k * height;
                    const unsigned int outgoingMin = outgoingMins[j / blockSize * width + k];
                    for (size_t ii = 0; ii < height; ++ii) {
                        if (savedCol[ii] + outgoingMin < rowMax[ii]) {
                            const size_t offset = (i + ii) * localCols + j;
                            relax(dist.data() + offset, path.data() + offset,
                                  savedRow, tileWidth, savedCol[ii], first + k);
                        }
                    }
                }
            }
        }
    }
}

void printResults(const std::vector<unsigned int>& dist, const Distribution& rows,
                  const Distribution& cols, int row, int col, int rank) {
    const size_t elements = rows.nodes * cols.nodes;
    if (elements == 0) {
        if (rank == 0) {
            printf("=== RESULTS ===\nName: DistanceMatrix\nElements: 0\n=== END RESULTS ===\n");
        }
        return;
    }

    const size_t localCols = cols.localSize(col);
    const size_t indices[] = {0, elements / 4, elements / 2, 3 * elements / 4, elements - 1};
    unsigned int samples[5] = {};
    unsigned int minValue = INF;
    unsigned int maxValue = 0;
    hash digest;
    // Receive one band of rows at a time and hash it in original row-major order.
    std::vector<unsigned int> band(rank == 0 ? rows.blockSize * cols.nodes : 0);
    std::vector<size_t> offsets(rank == 0 ? cols.processes : 0);
    for (size_t block = 0; block < rows.blocks(); ++block) {
        const int procRow = rows.owner(block);
        const size_t height = rows.width(block);
        if (rank != 0) {
            if (row == procRow && localCols != 0) {
                sendBand(dist.data() + rows.offset(block) * localCols, height * localCols, 0);
            }
            continue;
        }

        size_t count = 0;
        for (int procCol = 0; procCol < cols.processes; ++procCol) {
            const int source = procRow * cols.processes + procCol;
            const size_t length = height * cols.localSize(procCol);
            offsets[procCol] = count;
            if (length != 0) {
                if (source == 0) {
                    std::copy_n(dist.data() + rows.offset(block) * localCols,
                                length, band.data() + count);
                } else {
                    receiveBand(band.data() + count, length, source);
                }
            }
            count += length;
        }
        const size_t first = block * rows.blockSize * cols.nodes;
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < cols.blocks(); ++j) {
                const int procCol = cols.owner(j);
                const auto* values = band.data() + offsets[procCol]
                    + i * cols.localSize(procCol) + cols.offset(j);
                const size_t start = first + i * cols.nodes + j * cols.blockSize;
                const size_t width = cols.width(j);
                for (size_t k = 0; k < width; ++k) {
                    const auto value = values[k];
                    minValue = std::min(minValue, value);
                    maxValue = std::max(maxValue, value);
                    digest.add(value);
                }
                for (size_t sample = 0; sample < 5; ++sample) {
                    if (indices[sample] >= start && indices[sample] < start + width) {
                        samples[sample] = values[indices[sample] - start];
                    }
                }
            }
        }
    }
    if (rank == 0) {
        printf("=== RESULTS ===\nName: DistanceMatrix\nElements: %zu\n", elements);
        printf("Min: %u\nMax: %u\n", minValue, maxValue);
        for (size_t i = 0; i < 5; ++i) {
            printf("Sample[%zu]: %u\n", indices[i], samples[i]);
        }
        printf("Hash: %016" PRIx64 "\n=== END RESULTS ===\n", digest.get());
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const Distribution& rows,
                    const Distribution& cols, int row, int col, int rank) {
    const size_t n = rows.nodes;
    const size_t localRows = rows.localSize(row);
    const size_t localCols = cols.localSize(col);
    unsigned int badDiagonal = n;
    for (size_t block = row; block < rows.blocks(); block += rows.processes) {
        if (cols.owner(block) == col) {
            for (size_t i = 0; i < rows.width(block); ++i) {
                if (dist[(rows.offset(block) + i) * localCols + cols.offset(block) + i] != 0) {
                    badDiagonal = std::min(badDiagonal,
                                           static_cast<unsigned int>(block * rows.blockSize + i));
                }
            }
        }
    }
    unsigned int firstBadDiagonal = n;
    MPI_Allreduce(&badDiagonal, &firstBadDiagonal, 1, MPI_UNSIGNED, MPI_MIN, MPI_COMM_WORLD);
    if (firstBadDiagonal != n) {
        if (rank == 0) {
            printf("Validation failed: diagonal element [%u,%u] is not zero\n",
                   firstBadDiagonal, firstBadDiagonal);
        }
        return false;
    }

    // Only rows and columns used by the original sample need to be collected.
    const size_t samples = std::min(n, size_t{10});
    std::vector<unsigned int> edges(2 * samples * n);
    for (size_t i = 0; i < localRows && localCols != 0; ++i) {
        const size_t globalRow = rows.globalIndex(i, row);
        const auto* values = dist.data() + i * localCols;
        for (size_t j = 0; j < localCols; ++j) {
            const size_t globalCol = cols.globalIndex(j, col);
            if (globalRow < samples) {
                edges[globalRow * n + globalCol] = values[j];
            }
            if (globalCol < samples) {
                edges[(samples + globalCol) * n + globalRow] = values[j];
            }
            if (globalRow >= samples && globalCol >= samples) {
                break;
            }
        }
    }
    for (size_t offset = 0; offset < edges.size(); offset += MPI_CHUNK) {
        const int count = std::min(MPI_CHUNK, edges.size() - offset);
        auto* data = edges.data() + offset;
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : data, rank == 0 ? data : nullptr,
                   count, MPI_UNSIGNED, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    int valid = 1;
    if (rank == 0) {
        for (size_t i = 0; i < samples && valid; ++i) {
            for (size_t j = 0; j < samples && valid; ++j) {
                const auto ij = edges[i * n + j];
                for (size_t k = 0; k < n; ++k) {
                    const auto ik = edges[i * n + k];
                    const auto kj = edges[(samples + j) * n + k];
                    if (ik < INF && kj < INF && ik + kj < ij) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                               i, j, k);
                        valid = 0;
                        break;
                    }
                }
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int rank, int processes) {
    size_t numNodes = 512;
    bool validate = false;
    bool reportResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const int input = atoi(argv[++i]);
            if (input < 0) {
                if (rank == 0) {
                    printf("Number of nodes must be nonnegative\n");
                }
                return 1;
            }
            numNodes = input;
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            reportResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    int grid[2] = {};
    MPI_Dims_create(processes, 2, grid);
    const int row = rank / grid[1];
    const int col = rank % grid[1];
    // Fit cache-sized tiles into complete rounds over the longer grid axis.
    const size_t longestAxis = std::max(grid[0], grid[1]);
    const size_t minBlocks = (numNodes + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    const size_t rounds = std::max(size_t{1}, (minBlocks + longestAxis - 1) / longestAxis);
    const size_t tileSlots = rounds * longestAxis;
    const size_t blockSize = std::max(size_t{1}, std::min(numNodes / longestAxis,
                                    (numNodes + tileSlots - 1) / tileSlots));
    const Distribution rows{numNodes, blockSize, grid[0]};
    const Distribution cols{numNodes, blockSize, grid[1]};
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    const size_t localElements = rows.localSize(row) * cols.localSize(col);
    std::vector<unsigned int> dist(localElements);
    std::vector<unsigned int> path(localElements);
    initializeMatrices(dist, path, rows, cols, row, col, rank);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, rows, cols, row, col, rowComm, colComm);
    const double localTime = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&localTime, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000));
        const double operations = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", duration > 0 ? operations / duration / 1e9 : 0.0);
    }
    if (reportResults) {
        printResults(dist, rows, cols, row, col, rank);
    }
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(dist, rows, cols, row, col, rank);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    const int status = runBenchmark(argc, argv, rank, processes);
    MPI_Finalize();
    return status;
}
