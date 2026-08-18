#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int RESULT_TAG = 1701;
constexpr size_t TILE_SIZE = 64;

struct Block {
    size_t begin;
    size_t size;
};

// Divide [0,n) into contiguous, almost equally-sized pieces.
inline Block blockFor(const size_t n, const int coordinate, const int parts) noexcept {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t extra = n % static_cast<size_t>(parts);
    const size_t coord = static_cast<size_t>(coordinate);
    return {coord * base + std::min(coord, extra), base + (coord < extra ? 1u : 0u)};
}

inline int ownerOf(const size_t index, const size_t n, const int parts) noexcept {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t extra = n % static_cast<size_t>(parts);
    const size_t largeRegion = extra * (base + 1);
    if (index < largeRegion) {
        return static_cast<int>(index / (base + 1));
    }
    return static_cast<int>(extra + (index - largeRegion) / base);
}

// rand_r in glibc advances this 32-bit LCG three times per returned value.  The
// jump-ahead operation lets every rank create just its own submatrix while
// producing exactly the same graph as the original seed-42 initialization.
inline uint32_t advanceLcg(uint32_t state, uint64_t steps) noexcept {
    uint32_t accumulatedMultiplier = 1;
    uint32_t accumulatedIncrement = 0;
    uint32_t multiplier = 1103515245u;
    uint32_t increment = 12345u;

    while (steps != 0) {
        if ((steps & 1u) != 0) {
            accumulatedMultiplier *= multiplier;
            accumulatedIncrement = accumulatedIncrement * multiplier + increment;
        }
        increment *= multiplier + 1u;
        multiplier *= multiplier;
        steps >>= 1u;
    }
    return accumulatedMultiplier * state + accumulatedIncrement;
}

inline unsigned int nextRandom(uint32_t& state) noexcept {
    unsigned int result;

    state = state * 1103515245u + 12345u;
    result = (state / 65536u) % 2048u;
    state = state * 1103515245u + 12345u;
    result = (result << 10u) ^ ((state / 65536u) % 1024u);
    state = state * 1103515245u + 12345u;
    result = (result << 10u) ^ ((state / 65536u) % 1024u);
    return result;
}

void initializeLocalDistanceMatrix(std::vector<unsigned int>& dist,
                                   const size_t numNodes,
                                   const Block rows,
                                   const Block columns,
                                   const unsigned int rangeMin,
                                   const unsigned int rangeMax) {
    if (rows.size == 0 || columns.size == 0) {
        return;
    }

    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
#if defined(__GLIBC__)
    const uint64_t firstElement = static_cast<uint64_t>(rows.begin) * numNodes + columns.begin;
    uint32_t state = advanceLcg(42u, 3u * firstElement);

    for (size_t localRow = 0; localRow < rows.size; ++localRow) {
        unsigned int* const row = dist.data() + localRow * columns.size;
        for (size_t localColumn = 0; localColumn < columns.size; ++localColumn) {
            row[localColumn] = rangeMin + static_cast<unsigned int>(
                                              range * nextRandom(state) /
                                              static_cast<double>(RAND_MAX));
        }

        if (localRow + 1 < rows.size) {
            state = advanceLcg(state, 3u * (numNodes - columns.size));
        }
    }
#else
    // rand_r is implementation-defined.  On non-glibc systems, traverse the
    // stream locally so that output remains identical to that platform's
    // original serial program without centralizing the full matrix on rank 0.
    unsigned int state = 42;
    for (size_t globalRow = 0; globalRow < numNodes; ++globalRow) {
        for (size_t globalColumn = 0; globalColumn < numNodes; ++globalColumn) {
            const unsigned int value = rangeMin + static_cast<unsigned int>(
                range * rand_r(&state) / static_cast<double>(RAND_MAX));
            if (globalRow >= rows.begin && globalRow < rows.begin + rows.size &&
                globalColumn >= columns.begin &&
                globalColumn < columns.begin + columns.size) {
                dist[(globalRow - rows.begin) * columns.size +
                     (globalColumn - columns.begin)] = value;
            }
        }
    }
#endif

    const size_t diagonalBegin = std::max(rows.begin, columns.begin);
    const size_t diagonalEnd = std::min(rows.begin + rows.size, columns.begin + columns.size);
    for (size_t i = diagonalBegin; i < diagonalEnd; ++i) {
        dist[(i - rows.begin) * columns.size + (i - columns.begin)] = 0;
    }
}

void relaxDiagonal(std::vector<unsigned int>& diagonal, const size_t blockSize) noexcept {
    for (size_t k = 0; k < blockSize; ++k) {
        const unsigned int* const pivot = diagonal.data() + k * blockSize;
        for (size_t i = 0; i < blockSize; ++i) {
            unsigned int* const row = diagonal.data() + i * blockSize;
            const unsigned int distIK = row[k];
#pragma omp simd
            for (size_t j = 0; j < blockSize; ++j) {
                row[j] = std::min(row[j], distIK + pivot[j]);
            }
        }
    }
}

void relaxRowPanel(std::vector<unsigned int>& panel,
                   const std::vector<unsigned int>& diagonal,
                   const size_t blockSize,
                   const size_t localColumns) noexcept {
    if (localColumns == 0) {
        return;
    }
    for (size_t k = 0; k < blockSize; ++k) {
        const unsigned int* const pivot = panel.data() + k * localColumns;
        for (size_t i = 0; i < blockSize; ++i) {
            unsigned int* const row = panel.data() + i * localColumns;
            const unsigned int distIK = diagonal[i * blockSize + k];
#pragma omp simd
            for (size_t j = 0; j < localColumns; ++j) {
                row[j] = std::min(row[j], distIK + pivot[j]);
            }
        }
    }
}

void relaxColumnPanel(std::vector<unsigned int>& panel,
                      const std::vector<unsigned int>& diagonal,
                      const size_t localRows,
                      const size_t blockSize) noexcept {
    for (size_t k = 0; k < blockSize; ++k) {
        const unsigned int* const pivot = diagonal.data() + k * blockSize;
        for (size_t i = 0; i < localRows; ++i) {
            unsigned int* const row = panel.data() + i * blockSize;
            const unsigned int distIK = row[k];
#pragma omp simd
            for (size_t j = 0; j < blockSize; ++j) {
                row[j] = std::min(row[j], distIK + pivot[j]);
            }
        }
    }
}

void relaxLocalBlock(unsigned int* __restrict__ dist,
                     const unsigned int* __restrict__ rowPanel,
                     const unsigned int* __restrict__ columnPanel,
                     const size_t localRows,
                     const size_t localColumns,
                     const size_t blockSize) noexcept {
    if (localRows == 0 || localColumns == 0) {
        return;
    }
    for (size_t k = 0; k < blockSize; ++k) {
        const unsigned int* const pivot = rowPanel + k * localColumns;
        for (size_t i = 0; i < localRows; ++i) {
            unsigned int* const row = dist + i * localColumns;
            const unsigned int distIK = columnPanel[i * blockSize + k];
#pragma omp simd
            for (size_t j = 0; j < localColumns; ++j) {
                row[j] = std::min(row[j], distIK + pivot[j]);
            }
        }
    }
}

void startLargeBroadcast(std::vector<unsigned int>& buffer,
                         const int root,
                         MPI_Comm communicator,
                         std::vector<MPI_Request>& requests) {
    size_t offset = 0;
    while (offset < buffer.size()) {
        const int count = static_cast<int>(std::min(
            buffer.size() - offset,
            static_cast<size_t>(std::numeric_limits<int>::max())));
        requests.emplace_back();
        MPI_Ibcast(buffer.data() + offset, count, MPI_UNSIGNED, root, communicator,
                   &requests.back());
        offset += static_cast<size_t>(count);
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   const size_t numNodes,
                   const Block rows,
                   const Block columns,
                   const int coordinates[2],
                   const int dimensions[2],
                   const int rank,
                   MPI_Comm cartesianCommunicator,
                   MPI_Comm rowCommunicator,
                   MPI_Comm columnCommunicator) {
    std::vector<unsigned int> diagonal;
    std::vector<unsigned int> rowPanel;
    std::vector<unsigned int> columnPanel;
    diagonal.reserve(TILE_SIZE * TILE_SIZE);
    rowPanel.reserve(TILE_SIZE * columns.size);
    columnPanel.reserve(rows.size * TILE_SIZE);

    for (size_t blockBegin = 0; blockBegin < numNodes;) {
        const size_t k = blockBegin;
        const int rowOwner = ownerOf(k, numNodes, dimensions[0]);
        const int columnOwner = ownerOf(k, numNodes, dimensions[1]);
        const Block ownerRows = blockFor(numNodes, rowOwner, dimensions[0]);
        const Block ownerColumns = blockFor(numNodes, columnOwner, dimensions[1]);
        const size_t blockSize = std::min(
            {TILE_SIZE, numNodes - k, ownerRows.begin + ownerRows.size - k,
             ownerColumns.begin + ownerColumns.size - k});

        diagonal.resize(blockSize * blockSize);
        int pivotCoordinates[2] = {rowOwner, columnOwner};
        int pivotRank;
        MPI_Cart_rank(cartesianCommunicator, pivotCoordinates, &pivotRank);

        if (rank == pivotRank) {
            const size_t localRow = k - rows.begin;
            const size_t localColumn = k - columns.begin;
            for (size_t i = 0; i < blockSize; ++i) {
                std::copy_n(dist.data() + (localRow + i) * columns.size + localColumn,
                            blockSize, diagonal.data() + i * blockSize);
            }
            relaxDiagonal(diagonal, blockSize);
            for (size_t i = 0; i < blockSize; ++i) {
                std::copy_n(diagonal.data() + i * blockSize, blockSize,
                            dist.data() + (localRow + i) * columns.size + localColumn);
            }
        }
        MPI_Bcast(diagonal.data(), static_cast<int>(diagonal.size()), MPI_UNSIGNED,
                  pivotRank, cartesianCommunicator);

        rowPanel.resize(blockSize * columns.size);
        columnPanel.resize(rows.size * blockSize);

        if (coordinates[0] == rowOwner && columns.size != 0) {
            const size_t localRow = k - rows.begin;
            for (size_t i = 0; i < blockSize; ++i) {
                std::copy_n(dist.data() + (localRow + i) * columns.size,
                            columns.size, rowPanel.data() + i * columns.size);
            }
            relaxRowPanel(rowPanel, diagonal, blockSize, columns.size);
            for (size_t i = 0; i < blockSize; ++i) {
                std::copy_n(rowPanel.data() + i * columns.size, columns.size,
                            dist.data() + (localRow + i) * columns.size);
            }
        }

        std::vector<MPI_Request> requests;
        requests.reserve(2);
        if (dimensions[0] > 1) {
            int root;
            MPI_Cart_rank(columnCommunicator, &rowOwner, &root);
            startLargeBroadcast(rowPanel, root, columnCommunicator, requests);
        }

        if (coordinates[1] == columnOwner && rows.size != 0) {
            const size_t localColumn = k - columns.begin;
            for (size_t i = 0; i < rows.size; ++i) {
                for (size_t j = 0; j < blockSize; ++j) {
                    columnPanel[i * blockSize + j] =
                        dist[i * columns.size + localColumn + j];
                }
            }
            relaxColumnPanel(columnPanel, diagonal, rows.size, blockSize);
            for (size_t i = 0; i < rows.size; ++i) {
                for (size_t j = 0; j < blockSize; ++j) {
                    dist[i * columns.size + localColumn + j] =
                        columnPanel[i * blockSize + j];
                }
            }
        }

        if (dimensions[1] > 1) {
            int root;
            MPI_Cart_rank(rowCommunicator, &columnOwner, &root);
            startLargeBroadcast(columnPanel, root, rowCommunicator, requests);
        }
        if (!requests.empty()) {
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        }

        relaxLocalBlock(dist.data(), rowPanel.data(), columnPanel.data(),
                        rows.size, columns.size, blockSize);
        blockBegin += blockSize;
    }
}

void reduceUnsignedSum(const std::vector<unsigned int>& send,
                       std::vector<unsigned int>& receive,
                       const int root,
                       MPI_Comm communicator) {
    size_t offset = 0;
    while (offset < send.size()) {
        const size_t remaining = send.size() - offset;
        const int count = static_cast<int>(std::min(
            remaining, static_cast<size_t>(std::numeric_limits<int>::max())));
        MPI_Reduce(send.data() + offset,
                   receive.empty() ? nullptr : receive.data() + offset,
                   count, MPI_UNSIGNED, MPI_SUM, root, communicator);
        offset += static_cast<size_t>(count);
    }
}

bool validateResult(const std::vector<unsigned int>& dist,
                    const size_t numNodes,
                    const Block rows,
                    const Block columns,
                    const int rank,
                    const int root,
                    MPI_Comm communicator) {
    uint64_t firstBadDiagonal = static_cast<uint64_t>(numNodes);
    const size_t diagonalBegin = std::max(rows.begin, columns.begin);
    const size_t diagonalEnd = std::min(rows.begin + rows.size, columns.begin + columns.size);
    for (size_t i = diagonalBegin; i < diagonalEnd; ++i) {
        if (dist[(i - rows.begin) * columns.size + (i - columns.begin)] != 0) {
            firstBadDiagonal = i;
            break;
        }
    }

    uint64_t globalBadDiagonal = static_cast<uint64_t>(numNodes);
    MPI_Reduce(&firstBadDiagonal, &globalBadDiagonal, 1, MPI_UINT64_T, MPI_MIN,
               root, communicator);

    const size_t samples = std::min(numNodes, static_cast<size_t>(10));
    std::vector<unsigned int> localSampleRows(samples * numNodes, 0);
    std::vector<unsigned int> localSampleColumns(samples * numNodes, 0);

    for (size_t source = 0; source < samples; ++source) {
        if (source >= rows.begin && source < rows.begin + rows.size && columns.size != 0) {
            const unsigned int* const localRow =
                dist.data() + (source - rows.begin) * columns.size;
            std::copy_n(localRow, columns.size,
                        localSampleRows.data() + source * numNodes + columns.begin);
        }
    }
    for (size_t destination = 0; destination < samples; ++destination) {
        if (destination >= columns.begin && destination < columns.begin + columns.size) {
            const size_t localColumn = destination - columns.begin;
            for (size_t i = 0; i < rows.size; ++i) {
                localSampleColumns[destination * numNodes + rows.begin + i] =
                    dist[i * columns.size + localColumn];
            }
        }
    }

    std::vector<unsigned int> sampleRows;
    std::vector<unsigned int> sampleColumns;
    if (rank == root) {
        sampleRows.resize(samples * numNodes);
        sampleColumns.resize(samples * numNodes);
    }
    reduceUnsignedSum(localSampleRows, sampleRows, root, communicator);
    reduceUnsignedSum(localSampleColumns, sampleColumns, root, communicator);

    int valid = 1;
    if (rank == root) {
        if (globalBadDiagonal < numNodes) {
            std::printf("Validation failed: diagonal element [%llu,%llu] is not zero\n",
                        static_cast<unsigned long long>(globalBadDiagonal),
                        static_cast<unsigned long long>(globalBadDiagonal));
            valid = 0;
        }

        for (size_t i = 0; valid != 0 && i < samples; ++i) {
            for (size_t j = 0; valid != 0 && j < samples; ++j) {
                const unsigned int distIJ = sampleRows[i * numNodes + j];
                for (size_t k = 0; k < numNodes; ++k) {
                    const unsigned int distIK = sampleRows[i * numNodes + k];
                    const unsigned int distKJ = sampleColumns[j * numNodes + k];
                    if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                        std::printf(
                            "Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                            i, j, k);
                        valid = 0;
                        break;
                    }
                }
            }
        }
    }

    MPI_Bcast(&valid, 1, MPI_INT, root, communicator);
    return valid != 0;
}

void transferLargeBuffer(const unsigned int* sendBuffer,
                         unsigned int* receiveBuffer,
                         const size_t elements,
                         const int peer,
                         const bool send,
                         MPI_Comm communicator) {
    size_t offset = 0;
    while (offset < elements) {
        const int count = static_cast<int>(std::min(
            elements - offset, static_cast<size_t>(std::numeric_limits<int>::max())));
        if (send) {
            MPI_Send(sendBuffer + offset, count, MPI_UNSIGNED, peer, RESULT_TAG, communicator);
        } else {
            MPI_Recv(receiveBuffer + offset, count, MPI_UNSIGNED, peer, RESULT_TAG,
                     communicator, MPI_STATUS_IGNORE);
        }
        offset += static_cast<size_t>(count);
    }
}

std::vector<unsigned int> gatherResult(const std::vector<unsigned int>& localDist,
                                       const size_t numNodes,
                                       const int dimensions[2],
                                       const int rank,
                                       const int root,
                                       MPI_Comm cartesianCommunicator) {
    int communicatorSize;
    MPI_Comm_size(cartesianCommunicator, &communicatorSize);

    if (rank != root) {
        transferLargeBuffer(localDist.data(), nullptr, localDist.size(), root, true,
                            cartesianCommunicator);
        return {};
    }

    std::vector<unsigned int> result(numNodes * numNodes);
    std::vector<unsigned int> packed;
    for (int source = 0; source < communicatorSize; ++source) {
        int coordinates[2];
        MPI_Cart_coords(cartesianCommunicator, source, 2, coordinates);
        const Block rows = blockFor(numNodes, coordinates[0], dimensions[0]);
        const Block columns = blockFor(numNodes, coordinates[1], dimensions[1]);
        const size_t blockElements = rows.size * columns.size;

        const unsigned int* blockData;
        if (source == root) {
            blockData = localDist.data();
        } else {
            packed.resize(blockElements);
            transferLargeBuffer(nullptr, packed.data(), blockElements, source, false,
                                cartesianCommunicator);
            blockData = packed.data();
        }

        if (columns.size != 0) {
            for (size_t i = 0; i < rows.size; ++i) {
                std::copy_n(blockData + i * columns.size, columns.size,
                            result.data() + (rows.begin + i) * numNodes + columns.begin);
            }
        }
    }

    return result;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank;
    int worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || value > std::numeric_limits<size_t>::max()) {
                argumentsValid = false;
            } else {
                numNodes = static_cast<size_t>(value);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }

    if (numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes)) {
        if (worldRank == 0) {
            std::fprintf(stderr, "Number of nodes is too large for this MPI implementation\n");
        }
        argumentsValid = false;
    }

    if (showHelp || !argumentsValid) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    int dimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dimensions);
    int periods[2] = {0, 0};
    MPI_Comm cartesianCommunicator;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 1, &cartesianCommunicator);

    int rank;
    int coordinates[2];
    MPI_Comm_rank(cartesianCommunicator, &rank);
    MPI_Cart_coords(cartesianCommunicator, rank, 2, coordinates);

    int root = worldRank == 0 ? rank : 0;
    MPI_Allreduce(MPI_IN_PLACE, &root, 1, MPI_INT, MPI_MAX, cartesianCommunicator);

    int keepRowDimension[2] = {0, 1};
    int keepColumnDimension[2] = {1, 0};
    MPI_Comm rowCommunicator;
    MPI_Comm columnCommunicator;
    MPI_Cart_sub(cartesianCommunicator, keepRowDimension, &rowCommunicator);
    MPI_Cart_sub(cartesianCommunicator, keepColumnDimension, &columnCommunicator);

    const Block rows = blockFor(numNodes, coordinates[0], dimensions[0]);
    const Block columns = blockFor(numNodes, coordinates[1], dimensions[1]);

    if (rank == root) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
    }

    std::vector<unsigned int> dist(rows.size * columns.size);
    initializeLocalDistanceMatrix(dist, numNodes, rows, columns, 1, MAX_DISTANCE);

    if (rank == root) {
        std::printf("Computing shortest paths...\n");
    }
    MPI_Barrier(cartesianCommunicator);
    const double start = MPI_Wtime();

    floydWarshall(dist, numNodes, rows, columns, coordinates, dimensions, rank,
                  cartesianCommunicator, rowCommunicator, columnCommunicator);

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, root,
               cartesianCommunicator);

    if (rank == root) {
        const long durationMilliseconds = static_cast<long>(elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", durationMilliseconds);
        const double operations = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gops = operations / elapsed / 1.0e9;
        std::printf("Performance: %.3f GOPS\n", gops);
    }

    if (printResults) {
        std::vector<unsigned int> result = gatherResult(
            dist, numNodes, dimensions, rank, root, cartesianCommunicator);
        if (rank == root) {
            print_results_int(result, "DistanceMatrix");
        }
    }

    bool valid = true;
    if (validate) {
        if (rank == root) {
            std::printf("Validating result...\n");
        }
        valid = validateResult(dist, numNodes, rows, columns, rank, root,
                               cartesianCommunicator);
        if (rank == root) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    MPI_Comm_free(&rowCommunicator);
    MPI_Comm_free(&columnCommunicator);
    MPI_Comm_free(&cartesianCommunicator);
    MPI_Finalize();
    return valid ? 0 : 1;
}
