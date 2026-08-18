#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// Each rank owns a contiguous block of source vertices.  In the column-major
// layout used here this makes the values for one destination contiguous.
void floydWarshallMPI(std::vector<unsigned int>& dist,
                      std::vector<unsigned int>& path,
                      const size_t numNodes,
                      const size_t firstSource,
                      const size_t localSources,
                      const std::vector<int>& sourceOffsets) {
    std::vector<unsigned int> pivot(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = static_cast<int>(
            std::upper_bound(sourceOffsets.begin(), sourceOffsets.end(), static_cast<int>(k)) -
            sourceOffsets.begin()) - 1;
        if (k >= firstSource && k < firstSource + localSources) {
            const size_t localK = k - firstSource;
            for (size_t j = 0; j < numNodes; ++j) {
                pivot[j] = dist[j * localSources + localK];
            }
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        if (localSources == 0) {
            continue;
        }
        const unsigned int* const distanceToPivot = dist.data() + k * localSources;
        for (size_t j = 0; j < numNodes; ++j) {
            // The k-th destination is unchanged because its pivot distance is zero.
            if (j == k) {
                continue;
            }
            const unsigned int pivotToDestination = pivot[j];
            unsigned int* const distanceColumn = dist.data() + j * localSources;
            unsigned int* const pathColumn = path.data() + j * localSources;
            for (size_t localI = 0; localI < localSources; ++localI) {
                const unsigned int newDist = distanceToPivot[localI] + pivotToDestination;
                if (newDist < distanceColumn[localI]) {
                    distanceColumn[localI] = newDist;
                    pathColumn[localI] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    
    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", 
                               i, j, k);
                        return false;
                    }
                }
            }
        }
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
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
    int exitCode = 0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
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
            exitCode = 1;
        }
    }
    if (numNodes == 0 || numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        numNodes * ((numNodes + static_cast<size_t>(worldSize) - 1) / static_cast<size_t>(worldSize))
            > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            printf("Number of nodes is outside the supported MPI count range\n");
        }
        exitCode = 1;
    }
    if (exitCode != 0) {
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    std::vector<int> sourceCounts(worldSize);
    std::vector<int> sourceOffsets(worldSize);
    std::vector<int> elementCounts(worldSize);
    std::vector<int> elementOffsets(worldSize);
    const size_t baseSources = numNodes / static_cast<size_t>(worldSize);
    const size_t extraSources = numNodes % static_cast<size_t>(worldSize);
    int packedOffset = 0;
    int sourceOffset = 0;
    for (int process = 0; process < worldSize; ++process) {
        const size_t localCount = baseSources + (static_cast<size_t>(process) < extraSources ? 1 : 0);
        sourceCounts[process] = static_cast<int>(localCount);
        sourceOffsets[process] = sourceOffset;
        elementCounts[process] = static_cast<int>(numNodes * localCount);
        elementOffsets[process] = packedOffset;
        sourceOffset += sourceCounts[process];
        packedOffset += elementCounts[process];
    }

    const size_t localSources = static_cast<size_t>(sourceCounts[rank]);
    const size_t firstSource = static_cast<size_t>(sourceOffsets[rank]);
    std::vector<unsigned int> localDist(numNodes * localSources);
    std::vector<unsigned int> localPath(numNodes * localSources);
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t localI = 0; localI < localSources; ++localI) {
            localPath[j * localSources + localI] = static_cast<unsigned int>(firstSource + localI);
        }
    }

    std::vector<unsigned int> packedDistance;
    if (rank == 0) {
        std::vector<unsigned int> initialDistance(numNodes * numNodes);
        initializeDistanceMatrix(initialDistance, numNodes, 1, MAX_DISTANCE);
        packedDistance.resize(numNodes * numNodes);
        for (int process = 0; process < worldSize; ++process) {
            const size_t processSources = static_cast<size_t>(sourceCounts[process]);
            const size_t processFirst = static_cast<size_t>(sourceOffsets[process]);
            unsigned int* const destination = packedDistance.data() + elementOffsets[process];
            for (size_t j = 0; j < numNodes; ++j) {
                std::memcpy(destination + j * processSources,
                            initialDistance.data() + j * numNodes + processFirst,
                            processSources * sizeof(unsigned int));
            }
        }
    }
    MPI_Scatterv(rank == 0 ? packedDistance.data() : nullptr, elementCounts.data(), elementOffsets.data(),
                 MPI_UNSIGNED, localDist.data(), elementCounts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshallMPI(localDist, localPath, numNodes, firstSource, localSources, sourceOffsets);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
        printf("Computation time: %lld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = elapsed > 0.0 ? ops / elapsed / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    if (printResults || validate) {
        if (rank == 0) {
            packedDistance.resize(numNodes * numNodes);
        }
        MPI_Gatherv(localDist.data(), elementCounts[rank], MPI_UNSIGNED,
                    rank == 0 ? packedDistance.data() : nullptr, elementCounts.data(), elementOffsets.data(),
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<unsigned int> dist(numNodes * numNodes);
            for (int process = 0; process < worldSize; ++process) {
                const size_t processSources = static_cast<size_t>(sourceCounts[process]);
                const size_t processFirst = static_cast<size_t>(sourceOffsets[process]);
                const unsigned int* const source = packedDistance.data() + elementOffsets[process];
                for (size_t j = 0; j < numNodes; ++j) {
                    std::memcpy(dist.data() + j * numNodes + processFirst,
                                source + j * processSources, processSources * sizeof(unsigned int));
                }
            }
            if (printResults) {
                print_results_int(dist, "DistanceMatrix");
            }
            if (validate) {
                printf("Validating result...\n");
                if (validateResult(dist, numNodes)) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
