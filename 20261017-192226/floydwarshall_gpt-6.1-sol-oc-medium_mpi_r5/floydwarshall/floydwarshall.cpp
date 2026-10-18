#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
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

size_t blockSize(size_t n, int parts, int coordinate) {
    return n / parts + (static_cast<size_t>(coordinate) < n % parts);
}

size_t blockStart(size_t n, int parts, int coordinate) {
    return (n / parts) * coordinate + std::min(n % parts, static_cast<size_t>(coordinate));
}

int blockOwner(size_t k, size_t n, int parts) {
    const size_t q = n / parts, r = n % parts;
    const size_t split = (q + 1) * r;
    return static_cast<int>(k < split ? k / (q + 1) : r + (k - split) / q);
}

// Chunk point-to-point transfers so a local matrix may exceed MPI's int count.
void transfer(unsigned int* data, size_t count, int peer, bool send, MPI_Comm comm) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, comm);
        else MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, comm, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

void initializeMatrices(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                        size_t n, const int* dims, const int* coords, int rank, MPI_Comm comm) {
    const size_t rows = blockSize(n, dims[0], coords[0]);
    const size_t cols = blockSize(n, dims[1], coords[1]);
    const size_t firstRow = blockStart(n, dims[0], coords[0]);
    for (size_t i = 0; i < rows && cols != 0; ++i)
        std::fill_n(path.data() + i * cols, cols, static_cast<unsigned int>(firstRow + i));

    if (rank != 0) {
        transfer(dist.data(), dist.size(), 0, false, comm);
        return;
    }

    // Generate the exact original rand_r stream, one process-row slab at a time.
    // No rank needs a replicated global matrix during initialization or compute.
    unsigned int seed = 42;
    for (int r = 0; r < dims[0]; ++r) {
        const size_t nr = blockSize(n, dims[0], r);
        const size_t start = blockStart(n, dims[0], r);
        std::vector<unsigned int> slab(nr * n);
        for (auto& value : slab)
            value = 1 + static_cast<unsigned int>(static_cast<double>(MAX_DISTANCE) *
                                                rand_r(&seed) / static_cast<double>(RAND_MAX));
        for (size_t i = 0; i < nr; ++i) slab[i * n + start + i] = 0;
        for (int c = 0; c < dims[1]; ++c) {
            const size_t nc = blockSize(n, dims[1], c);
            const size_t offset = blockStart(n, dims[1], c);
            int coordinates[2] = {r, c}, target;
            MPI_Cart_rank(comm, coordinates, &target);
            std::vector<unsigned int> packed(nr * nc);
            for (size_t i = 0; i < nr && nc != 0; ++i)
                std::copy_n(slab.data() + i * n + offset, nc, packed.data() + i * nc);
            if (target == 0) dist.swap(packed);
            else transfer(packed.data(), packed.size(), target, true, comm);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   size_t n, const int* dims, const int* coords,
                   MPI_Comm rowComm, MPI_Comm colComm) {
    const size_t rows = blockSize(n, dims[0], coords[0]);
    const size_t cols = blockSize(n, dims[1], coords[1]);
    const size_t firstRow = blockStart(n, dims[0], coords[0]);
    const size_t firstCol = blockStart(n, dims[1], coords[1]);
    std::vector<unsigned int> pivotRow(cols), pivotCol(rows);
    for (size_t k = 0; k < n; ++k) {
        const int ownerRow = blockOwner(k, n, dims[0]);
        const int ownerCol = blockOwner(k, n, dims[1]);
        if (coords[0] == ownerRow && cols != 0)
            std::copy_n(dist.data() + (k - firstRow) * cols, cols, pivotRow.data());
        if (coords[1] == ownerCol)
            for (size_t i = 0; i < rows; ++i) pivotCol[i] = dist[i * cols + k - firstCol];

        MPI_Request requests[2];
        MPI_Ibcast(pivotRow.data(), static_cast<int>(cols), MPI_UNSIGNED,
                   ownerRow, colComm, &requests[0]);
        MPI_Ibcast(pivotCol.data(), static_cast<int>(rows), MPI_UNSIGNED,
                   ownerCol, rowComm, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        // The diagonal is zero and all weights are nonnegative, so pivot row
        // and column values cannot change in this iteration. Snapshotting them
        // preserves both strict-improvement tie handling and the original paths.
        for (size_t i = 0; i < rows; ++i) {
            const unsigned int dik = pivotCol[i];
            if (firstRow + i == k || cols == 0) continue;
            unsigned int* distance = dist.data() + i * cols;
            unsigned int* intermediate = path.data() + i * cols;
            for (size_t j = 0; j < cols; ++j) {
                const unsigned int candidate = dik + pivotRow[j];
                if (candidate < distance[j]) {
                    distance[j] = candidate;
                    intermediate[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

std::vector<unsigned int> gatherMatrix(std::vector<unsigned int>& local, size_t n,
                                      const int* dims, int rank, MPI_Comm comm) {
    if (rank != 0) {
        transfer(local.data(), local.size(), 0, true, comm);
        return {};
    }
    std::vector<unsigned int> result(n * n);
    for (int r = 0; r < dims[0]; ++r) {
        const size_t nr = blockSize(n, dims[0], r), row = blockStart(n, dims[0], r);
        for (int c = 0; c < dims[1]; ++c) {
            const size_t nc = blockSize(n, dims[1], c), col = blockStart(n, dims[1], c);
            int coordinates[2] = {r, c}, source;
            MPI_Cart_rank(comm, coordinates, &source);
            std::vector<unsigned int> packed(source == 0 ? 0 : nr * nc);
            if (source != 0) transfer(packed.data(), packed.size(), source, false, comm);
            const auto& block = source == 0 ? local : packed;
            for (size_t i = 0; i < nr && nc != 0; ++i)
                std::copy_n(block.data() + i * nc, nc, result.data() + (row + i) * n + col);
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
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* text = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(text, &end, 10);
            if (errno || text[0] == '-' || end == text || *end != '\0' ||
                parsed > INT_MAX ||
                (parsed != 0 && parsed > std::numeric_limits<size_t>::max() /
                                          sizeof(unsigned int) / parsed)) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", text);
                MPI_Finalize();
                return 1;
            }
            numNodes = static_cast<size_t>(parsed);
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
    
    int dims[2] = {0, 0}, coords[2], periods[2] = {0, 0};
    MPI_Dims_create(processes, 2, dims);
    MPI_Comm grid, rowComm, colComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
    MPI_Cart_coords(grid, rank, 2, coords);
    // Split keys make communicator ranks equal to their grid coordinates.
    MPI_Comm_split(grid, coords[0], coords[1], &rowComm);
    MPI_Comm_split(grid, coords[1], coords[0], &colComm);

    int status = 0;
    try {
        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI processes: %d (%d x %d grid)\n", processes, dims[0], dims[1]);
            printf("Initializing graph...\n");
        }
        const size_t localSize = blockSize(numNodes, dims[0], coords[0]) *
                                 blockSize(numNodes, dims[1], coords[1]);
        std::vector<unsigned int> dist(localSize), path(localSize);
        initializeMatrices(dist, path, numNodes, dims, coords, rank, grid);

        if (rank == 0) printf("Computing shortest paths...\n");
        MPI_Barrier(grid);
        const double start = MPI_Wtime();
        floydWarshall(dist, path, numNodes, dims, coords, rowComm, colComm);
        const double elapsed = MPI_Wtime() - start;
        double seconds = 0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
        if (rank == 0) {
            printf("Computation time: %.3f ms\n", seconds * 1000.0);
            const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
        }

        // Reconstruction is outside the timed kernel and needed only for the
        // original global validation and order-dependent result hash.
        if (printResults || validate) {
            auto result = gatherMatrix(dist, numNodes, dims, rank, grid);
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
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(grid, 1);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, grid);
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return status;
}
