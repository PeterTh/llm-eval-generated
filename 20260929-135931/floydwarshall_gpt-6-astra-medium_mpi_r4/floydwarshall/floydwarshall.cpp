#include <algorithm>
#include <cerrno>
#include <climits>
#include <limits>
#include <exception>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Contiguous, nearly equal blocks; empty blocks are supported as well.
size_t blockStart(size_t n, int coordinate, int parts) {
    return n * static_cast<size_t>(coordinate) / static_cast<size_t>(parts);
}

int blockOwner(size_t k, size_t n, int parts) {
    return static_cast<int>(((k + 1) * static_cast<size_t>(parts) - 1) / n);
}

// Chunk large transfers so the matrix size is not limited by MPI's int counts.
void transferBlock(unsigned int* data, size_t count, int peer, bool send,
                   MPI_Comm comm) {
    while (count != 0) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) {
            MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, comm);
        } else {
            MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, comm, MPI_STATUS_IGNORE);
        }
        data += chunk;
        count -= chunk;
    }
}

void initializeMatrices(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path, size_t n,
                        int rank, const int dims[2], const int coords[2],
                        MPI_Comm grid) {
    const size_t rowStart = blockStart(n, coords[0], dims[0]);
    const size_t rows = blockStart(n, coords[0] + 1, dims[0]) - rowStart;
    const size_t cols = blockStart(n, coords[1] + 1, dims[1]) -
                        blockStart(n, coords[1], dims[1]);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            path[i * cols + j] = static_cast<unsigned int>(rowStart + i);
        }
    }

    if (rank != 0) {
        transferBlock(dist.data(), dist.size(), 0, false, grid);
        return;
    }

    // Preserve the original rand_r stream exactly. Generate just one process
    // row at a time, packed by destination, instead of a replicated matrix.
    unsigned int seed = 42;
    std::vector<unsigned int> slab;
    for (int r = 0; r < dims[0]; ++r) {
        const size_t first = blockStart(n, r, dims[0]);
        const size_t height = blockStart(n, r + 1, dims[0]) - first;
        slab.resize(height * n);
        for (size_t i = 0; i < height; ++i) {
            for (int c = 0; c < dims[1]; ++c) {
                const size_t begin = blockStart(n, c, dims[1]);
                const size_t width = blockStart(n, c + 1, dims[1]) - begin;
                for (size_t j = 0; j < width; ++j) {
                    unsigned int value = 1 + static_cast<unsigned int>(
                        static_cast<double>(MAX_DISTANCE) * rand_r(&seed) /
                        static_cast<double>(RAND_MAX));
                    if (first + i == begin + j) value = 0;
                    slab[begin * height + i * width + j] = value;
                }
            }
        }
        for (int c = 0; c < dims[1]; ++c) {
            const size_t begin = blockStart(n, c, dims[1]);
            const size_t count = height * (blockStart(n, c + 1, dims[1]) - begin);
            if (count == 0) continue;
            int destination, coordinates[2] = {r, c};
            MPI_Cart_rank(grid, coordinates, &destination);
            unsigned int* block = slab.data() + begin * height;
            if (destination == 0) {
                std::copy_n(block, count, dist.data());
            } else {
                transferBlock(block, count, destination, true, grid);
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, size_t n,
                   const int dims[2], const int coords[2],
                   MPI_Comm rowComm, MPI_Comm colComm,
                   std::vector<unsigned int>& pivotRow,
                   std::vector<unsigned int>& pivotColumn) {
    const size_t rowStart = blockStart(n, coords[0], dims[0]);
    const size_t colStart = blockStart(n, coords[1], dims[1]);
    const size_t rows = pivotColumn.size(), cols = pivotRow.size();
    for (size_t k = 0; k < n; ++k) {
        const int ownerRow = blockOwner(k, n, dims[0]);
        const int ownerCol = blockOwner(k, n, dims[1]);
        if (coords[0] == ownerRow && cols != 0) {
            std::copy_n(dist.data() + (k - rowStart) * cols, cols, pivotRow.data());
        }
        if (coords[1] == ownerCol) {
            for (size_t i = 0; i < rows; ++i) {
                pivotColumn[i] = dist[i * cols + k - colStart];
            }
        }
        // Independent row/column broadcasts overlap. The unchanged zero
        // diagonal makes these snapshots equivalent to the in-place kernel.
        MPI_Request requests[2];
        MPI_Ibcast(pivotRow.data(), static_cast<int>(cols), MPI_UNSIGNED,
                   ownerRow, colComm, &requests[0]);
        MPI_Ibcast(pivotColumn.data(), static_cast<int>(rows), MPI_UNSIGNED,
                   ownerCol, rowComm, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        if (cols == 0) continue;
        for (size_t i = 0; i < rows; ++i) {
            if (rowStart + i == k) continue;
            const unsigned int dik = pivotColumn[i];
            unsigned int* d = dist.data() + i * cols;
            unsigned int* p = path.data() + i * cols;
            for (size_t j = 0; j < cols; ++j) {
                const unsigned int candidate = dik + pivotRow[j];
                const bool improved = candidate < d[j];
                // Conditional assignments allow SIMD without changing strict
                // improvement / intermediate-vertex semantics.
                d[j] = improved ? candidate : d[j];
                p[j] = improved ? static_cast<unsigned int>(k) : p[j];
            }
        }
    }
}

std::vector<unsigned int> gatherDistances(std::vector<unsigned int>& local,
                                         size_t n, int rank, int size,
                                         const int dims[2], MPI_Comm grid) {
    if (rank != 0) {
        transferBlock(local.data(), local.size(), 0, true, grid);
        return {};
    }
    std::vector<unsigned int> result(n * n), buffer;
    for (int source = 0; source < size; ++source) {
        int coords[2];
        MPI_Cart_coords(grid, source, 2, coords);
        const size_t firstRow = blockStart(n, coords[0], dims[0]);
        const size_t firstCol = blockStart(n, coords[1], dims[1]);
        const size_t rows = blockStart(n, coords[0] + 1, dims[0]) - firstRow;
        const size_t cols = blockStart(n, coords[1] + 1, dims[1]) - firstCol;
        if (rows == 0 || cols == 0) continue;
        unsigned int* block = local.data();
        if (source != 0) {
            buffer.resize(rows * cols);
            transferBlock(buffer.data(), buffer.size(), source, false, grid);
            block = buffer.data();
        }
        for (size_t i = 0; i < rows; ++i) {
            std::copy_n(block + i * cols, cols,
                        result.data() + (firstRow + i) * n + firstCol);
        }
    }
    return result;
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    int status = 0;
    bool stop = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long parsed = strtoull(arg, &end, 10);
            if (errno != 0 || end == arg || *end != '\0' || *arg == '-' ||
                parsed > INT_MAX || (parsed != 0 &&
                parsed > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / parsed)) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", arg);
                status = 1;
                stop = true;
                break;
            }
            numNodes = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            stop = true;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
            stop = true;
            break;
        }
    }
    if (stop) {
        MPI_Finalize();
        return status;
    }

    try {
        int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2];
        MPI_Dims_create(size, 2, dims);
        MPI_Comm grid, rowComm, colComm;
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
        MPI_Cart_coords(grid, rank, 2, coords);
        // Rank within each subcommunicator equals the corresponding coordinate.
        MPI_Comm_split(grid, coords[0], coords[1], &rowComm);
        MPI_Comm_split(grid, coords[1], coords[0], &colComm);
        const size_t rows = blockStart(numNodes, coords[0] + 1, dims[0]) -
                            blockStart(numNodes, coords[0], dims[0]);
        const size_t cols = blockStart(numNodes, coords[1] + 1, dims[1]) -
                            blockStart(numNodes, coords[1], dims[1]);
        std::vector<unsigned int> dist(rows * cols), path(rows * cols);
        std::vector<unsigned int> pivotRow(cols), pivotColumn(rows);
        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
        }
        initializeMatrices(dist, path, numNodes, rank, dims, coords, grid);
        if (rank == 0) printf("Computing shortest paths...\n");
        MPI_Barrier(grid);
        const double start = MPI_Wtime();
        floydWarshall(dist, path, numNodes, dims, coords, rowComm, colComm,
                      pivotRow, pivotColumn);
        const double elapsed = MPI_Wtime() - start;
        double duration = 0;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
        if (rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
            const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", duration > 0 ? ops / duration / 1e9 : 0.0);
        }
        if (printResults || validate) {
            auto result = gatherDistances(dist, numNodes, rank, size, dims, grid);
            if (rank == 0) {
                if (printResults) print_results_int(result, "DistanceMatrix");
                if (validate) {
                    printf("Validating result...\n");
                    const bool valid = validateResult(result, numNodes);
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    status = valid ? 0 : 1;
                }
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, grid);
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
        MPI_Comm_free(&grid);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
