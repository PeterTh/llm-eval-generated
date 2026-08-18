#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

struct Block {
    size_t begin;
    size_t size;
};

// Divide [0,n) into contiguous, nearly equal blocks.  Keeping the larger
// blocks first also makes finding the owner of an index inexpensive.
inline Block blockFor(const size_t n, const int parts, const int coordinate) {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t extra = n % static_cast<size_t>(parts);
    return {static_cast<size_t>(coordinate) * base +
                std::min(static_cast<size_t>(coordinate), extra),
            base + (static_cast<size_t>(coordinate) < extra ? 1U : 0U)};
}

inline int blockOwner(const size_t index, const size_t n, const int parts) {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t extra = n % static_cast<size_t>(parts);
    const size_t largeRegion = (base + 1U) * extra;
    if (index < largeRegion) {
        return static_cast<int>(index / (base + 1U));
    }
    return static_cast<int>(extra + (index - largeRegion) / base);
}

// rand_r on glibc advances this 32-bit LCG three times per returned value.
// Jumping to the first element of every local row lets all ranks reproduce
// exactly the original seed-42 matrix without storing it on one rank.
unsigned int advanceLcg(unsigned int state, size_t steps) {
    unsigned int accumulatedMultiplier = 1U;
    unsigned int accumulatedIncrement = 0U;
    unsigned int multiplier = 1103515245U;
    unsigned int increment = 12345U;

    while (steps != 0) {
        if ((steps & 1U) != 0) {
            accumulatedMultiplier *= multiplier;
            accumulatedIncrement = accumulatedIncrement * multiplier + increment;
        }
        increment *= multiplier + 1U;
        multiplier *= multiplier;
        steps >>= 1U;
    }
    return accumulatedMultiplier * state + accumulatedIncrement;
}

void initializeLocalDistanceMatrix(std::vector<unsigned int>& dist,
                                   const size_t numNodes,
                                   const Block rows,
                                   const Block columns,
                                   const unsigned int rangeMin,
                                   const unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t localRow = 0; localRow < rows.size; ++localRow) {
        const size_t globalRow = rows.begin + localRow;
        const size_t firstIndex = globalRow * numNodes + columns.begin;
        unsigned int seed = advanceLcg(42U, 3U * firstIndex);
        unsigned int* row = dist.data() + localRow * columns.size;
        for (size_t localColumn = 0; localColumn < columns.size; ++localColumn) {
            row[localColumn] = rangeMin + static_cast<unsigned int>(
                range * rand_r(&seed) / static_cast<double>(RAND_MAX));
        }
        if (globalRow >= columns.begin && globalRow < columns.begin + columns.size) {
            row[globalRow - columns.begin] = 0U;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   const size_t numNodes,
                   const Block rows,
                   const Block columns,
                   const int processRows,
                   const int processColumns,
                   MPI_Comm rowComm,
                   MPI_Comm columnComm) {
    std::vector<unsigned int> columnValues(rows.size);
    std::vector<unsigned int> rowValues(columns.size);

    for (size_t k = 0; k < numNodes; ++k) {
        const int ownerRow = blockOwner(k, numNodes, processRows);
        const int ownerColumn = blockOwner(k, numNodes, processColumns);

        if (k >= columns.begin && k < columns.begin + columns.size) {
            const size_t localK = k - columns.begin;
            for (size_t i = 0; i < rows.size; ++i) {
                columnValues[i] = dist[i * columns.size + localK];
            }
        }
        if (k >= rows.begin && k < rows.begin + rows.size) {
            const size_t localK = k - rows.begin;
            std::copy_n(dist.data() + localK * columns.size,
                        columns.size, rowValues.data());
        }

        MPI_Request requests[2];
        MPI_Ibcast(columnValues.data(), static_cast<int>(columnValues.size()),
                   MPI_UNSIGNED, ownerColumn, rowComm, &requests[0]);
        MPI_Ibcast(rowValues.data(), static_cast<int>(rowValues.size()),
                   MPI_UNSIGNED, ownerRow, columnComm, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        for (size_t i = 0; i < rows.size; ++i) {
            unsigned int* const localRow = dist.data() + i * columns.size;
            const unsigned int distanceToK = columnValues[i];
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
            for (size_t j = 0; j < columns.size; ++j) {
                const unsigned int candidate = distanceToK + rowValues[j];
                localRow[j] = std::min(localRow[j], candidate);
            }
        }
    }
}

std::vector<unsigned int> gatherDistanceMatrix(const std::vector<unsigned int>& localDist,
                                               const size_t numNodes,
                                               const int rank,
                                               const int worldSize,
                                               const int processRows,
                                               const int processColumns) {
    std::vector<unsigned int> fullDist;
    if (rank == 0) {
        fullDist.resize(numNodes * numNodes);
    }

    for (int source = 0; source < worldSize; ++source) {
        const int sourceRow = source / processColumns;
        const int sourceColumn = source % processColumns;
        const Block rows = blockFor(numNodes, processRows, sourceRow);
        const Block columns = blockFor(numNodes, processColumns, sourceColumn);
        const size_t elementCount = rows.size * columns.size;

        if (elementCount > static_cast<size_t>(INT_MAX)) {
            if (rank == 0) {
                std::fprintf(stderr, "A local matrix block exceeds MPI's count limit\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        if (rank == 0) {
            std::vector<unsigned int> received;
            const unsigned int* packed = nullptr;
            if (source == 0) {
                packed = localDist.data();
            } else {
                received.resize(elementCount);
                MPI_Recv(received.data(), static_cast<int>(elementCount), MPI_UNSIGNED,
                         source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                packed = received.data();
            }
            for (size_t i = 0; i < rows.size; ++i) {
                std::copy_n(packed + i * columns.size, columns.size,
                            fullDist.data() + (rows.begin + i) * numNodes + columns.begin);
            }
        } else if (rank == source) {
            MPI_Send(localDist.data(), static_cast<int>(elementCount), MPI_UNSIGNED,
                     0, 1, MPI_COMM_WORLD);
        }
    }
    return fullDist;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[i * numNodes + i] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[i * numNodes + j];
                const unsigned int distIK = dist[i * numNodes + k];
                const unsigned int distKJ = dist[k * numNodes + j];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    std::printf("Validation failed: triangle inequality violated at "
                                "[%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
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
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus;
    }

    int dimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dimensions);
    // Favor splitting rows when a non-square process count has two choices.
    if (dimensions[0] < dimensions[1]) {
        std::swap(dimensions[0], dimensions[1]);
    }
    const int processRows = dimensions[0];
    const int processColumns = dimensions[1];
    const int processRow = rank / processColumns;
    const int processColumn = rank % processColumns;

    MPI_Comm rowComm;
    MPI_Comm columnComm;
    MPI_Comm_split(MPI_COMM_WORLD, processRow, processColumn, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, processColumn, processRow, &columnComm);

    const Block rows = blockFor(numNodes, processRows, processRow);
    const Block columns = blockFor(numNodes, processColumns, processColumn);
    if (rows.size > static_cast<size_t>(INT_MAX) ||
        columns.size > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            std::fprintf(stderr, "A matrix dimension exceeds MPI's count limit\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
    }

    std::vector<unsigned int> dist(rows.size * columns.size);
    initializeLocalDistanceMatrix(dist, numNodes, rows, columns, 1, MAX_DISTANCE);

    if (rank == 0) {
        std::printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, numNodes, rows, columns, processRows, processColumns,
                  rowComm, columnComm);
    const double localDuration = MPI_Wtime() - start;

    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long durationMs = static_cast<long>(duration * 1000.0);
        const double operations = static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes);
        const double gops = duration > 0.0 ? operations / duration / 1.0e9 : 0.0;
        std::printf("Computation time: %ld ms\n", durationMs);
        std::printf("Performance: %.3f GOPS\n", gops);
    }

    int result = 0;
    if (printResults || validate) {
        std::vector<unsigned int> fullDist = gatherDistanceMatrix(
            dist, numNodes, rank, worldSize, processRows, processColumns);
        if (rank == 0) {
            if (printResults) {
                print_results_int(fullDist, "DistanceMatrix");
            }
            if (validate) {
                std::printf("Validating result...\n");
                const bool valid = validateResult(fullDist, numNodes);
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                result = valid ? 0 : 1;
            }
        }
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&columnComm);
    MPI_Finalize();
    return result;
}
