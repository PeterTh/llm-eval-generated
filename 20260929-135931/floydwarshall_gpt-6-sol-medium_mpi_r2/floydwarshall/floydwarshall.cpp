#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
                              const size_t rowStart, unsigned int& seed,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < dist.size(); ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    for (size_t i = 0; i < dist.size() / numNodes; ++i) {
        dist[idx2(rowStart + i, i, numNodes)] = 0;
    }
}

// Divide each matrix dimension into contiguous blocks, including empty blocks
// when there are more processes than nodes along a dimension.
size_t blockStart(int coordinate, int processes, size_t n) {
    return n * static_cast<size_t>(coordinate) / static_cast<size_t>(processes);
}

int blockOwner(size_t index, int processes, size_t n) {
    return static_cast<int>(((index + 1) * static_cast<size_t>(processes) - 1) / n);
}

void copyBlock(const std::vector<unsigned int>& matrix, std::vector<unsigned int>& block,
               size_t rowStart, size_t colStart, size_t rows, size_t cols, size_t n) {
    for (size_t i = 0; i < rows; ++i) {
        std::copy_n(matrix.data() + (rowStart + i) * n + colStart, cols,
                    block.data() + i * cols);
    }
}

void placeBlock(const std::vector<unsigned int>& block, std::vector<unsigned int>& matrix,
                size_t rowStart, size_t colStart, size_t rows, size_t cols, size_t n) {
    for (size_t i = 0; i < rows; ++i) {
        std::copy_n(block.data() + i * cols, cols,
                    matrix.data() + (rowStart + i) * n + colStart);
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   size_t n, size_t rowStart, size_t colStart, size_t rows, size_t cols,
                   int rowProcesses, int colProcesses, MPI_Comm rowComm, MPI_Comm colComm) {
    std::vector<unsigned int> pivotRow(cols);
    std::vector<unsigned int> pivotCol(rows);

    for (size_t k = 0; k < n; ++k) {
        const int rowOwner = blockOwner(k, rowProcesses, n);
        const int colOwner = blockOwner(k, colProcesses, n);
        if (cols != 0 && k >= rowStart && k < rowStart + rows) {
            std::copy_n(dist.data() + (k - rowStart) * cols, cols, pivotRow.data());
        }
        if (k >= colStart && k < colStart + cols) {
            const size_t localCol = k - colStart;
            for (size_t i = 0; i < rows; ++i) {
                pivotCol[i] = dist[i * cols + localCol];
            }
        }
        MPI_Request broadcasts[2];
        MPI_Ibcast(pivotRow.data(), static_cast<int>(cols), MPI_UNSIGNED,
                   rowOwner, colComm, &broadcasts[0]);
        MPI_Ibcast(pivotCol.data(), static_cast<int>(rows), MPI_UNSIGNED,
                   colOwner, rowComm, &broadcasts[1]);
        MPI_Waitall(2, broadcasts, MPI_STATUSES_IGNORE);

        for (size_t i = 0; i < rows && cols != 0; ++i) {
            unsigned int* const distanceRow = dist.data() + i * cols;
            unsigned int* const pathRow = path.data() + i * cols;
            const unsigned int distanceIK = pivotCol[i];
            for (size_t j = 0; j < cols; ++j) {
                const unsigned int candidate = distanceIK + pivotRow[j];
                if (candidate < distanceRow[j]) {
                    distanceRow[j] = candidate;
                    pathRow[j] = static_cast<unsigned int>(k);
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
    int worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
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

    // MPI counts are int; this also bounds every submatrix message.
    if (numNodes == 0 || numNodes > static_cast<size_t>(std::sqrt(INT_MAX))) {
        if (rank == 0) fprintf(stderr, "Number of nodes must be between 1 and %d\n",
                               static_cast<int>(std::sqrt(INT_MAX)));
        MPI_Finalize();
        return 1;
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dims);
    const int rowCoordinate = rank / dims[1];
    const int colCoordinate = rank % dims[1];
    const size_t rowStart = blockStart(rowCoordinate, dims[0], numNodes);
    const size_t colStart = blockStart(colCoordinate, dims[1], numNodes);
    const size_t rows = blockStart(rowCoordinate + 1, dims[0], numNodes) - rowStart;
    const size_t cols = blockStart(colCoordinate + 1, dims[1], numNodes) - colStart;
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(MPI_COMM_WORLD, rowCoordinate, colCoordinate, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, colCoordinate, rowCoordinate, &colComm);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> fullDist;
    std::vector<unsigned int> dist(rows * cols);
    std::vector<unsigned int> path(rows * cols);
    for (size_t i = 0; i < rows && cols != 0; ++i) {
        std::fill_n(path.data() + i * cols, cols, static_cast<unsigned int>(rowStart + i));
    }

    // Root packs each rectangular block so compute ranks store only their own data.
    if (rank == 0) {
        unsigned int seed = 42;
        for (int r = 0; r < dims[0]; ++r) {
            const size_t rs = blockStart(r, dims[0], numNodes);
            const size_t nr = blockStart(r + 1, dims[0], numNodes) - rs;
            std::vector<unsigned int> strip(nr * numNodes);
            initializeDistanceMatrix(strip, numNodes, rs, seed, 1, MAX_DISTANCE);
            for (int c = 0; c < dims[1]; ++c) {
                const int dest = r * dims[1] + c;
                const size_t cs = blockStart(c, dims[1], numNodes);
                const size_t nc = blockStart(c + 1, dims[1], numNodes) - cs;
                if (nr == 0 || nc == 0) continue;
                if (dest == 0) {
                    copyBlock(strip, dist, 0, cs, nr, nc, numNodes);
                } else {
                    std::vector<unsigned int> block(nr * nc);
                    copyBlock(strip, block, 0, cs, nr, nc, numNodes);
                    MPI_Send(block.data(), static_cast<int>(block.size()), MPI_UNSIGNED,
                             dest, 0, MPI_COMM_WORLD);
                }
            }
        }
        printf("Computing shortest paths...\n");
    } else if (!dist.empty()) {
        MPI_Recv(dist.data(), static_cast<int>(dist.size()), MPI_UNSIGNED,
                 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, rowStart, colStart, rows, cols,
                  dims[0], dims[1], rowComm, colComm);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / seconds / 1e9);
    }

    if (printResults || validate) {
        if (rank == 0) {
            fullDist.resize(numNodes * numNodes);
            for (int source = 0; source < worldSize; ++source) {
                const int r = source / dims[1];
                const int c = source % dims[1];
                const size_t rs = blockStart(r, dims[0], numNodes);
                const size_t cs = blockStart(c, dims[1], numNodes);
                const size_t nr = blockStart(r + 1, dims[0], numNodes) - rs;
                const size_t nc = blockStart(c + 1, dims[1], numNodes) - cs;
                if (nr == 0 || nc == 0) continue;
                if (source == 0) {
                    placeBlock(dist, fullDist, rs, cs, nr, nc, numNodes);
                } else {
                    std::vector<unsigned int> block(nr * nc);
                    MPI_Recv(block.data(), static_cast<int>(block.size()), MPI_UNSIGNED,
                             source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    placeBlock(block, fullDist, rs, cs, nr, nc, numNodes);
                }
            }
        } else if (!dist.empty()) {
            MPI_Send(dist.data(), static_cast<int>(dist.size()), MPI_UNSIGNED,
                     0, 1, MPI_COMM_WORLD);
        }
    }

    int result = 0;
    if (rank == 0) {
        if (printResults) print_results_int(fullDist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(fullDist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Finalize();
    return result;
}
