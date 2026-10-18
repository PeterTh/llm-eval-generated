#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
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

// Balanced contiguous partitions, including empty partitions for tiny graphs.
int blockSize(int n, int parts, int coordinate) {
    return n / parts + (coordinate < n % parts);
}

int blockStart(int n, int parts, int coordinate) {
    return coordinate * (n / parts) + std::min(coordinate, n % parts);
}

int blockOwner(int k, int n, int parts) {
    const int base = n / parts;
    const int remainder = n % parts;
    const long long wideEnd = (static_cast<long long>(base) + 1) * remainder;
    return k < wideEnd ? k / (base + 1) : remainder + (k - wideEnd) / base;
}

// Transfer directly between the original row-major layout and distributed tiles.
// Derived datatypes avoid packing a second full matrix on rank zero. Chunked
// local transfers also avoid the MPI int-count limit for large matrix blocks.
void transferMatrix(std::vector<unsigned int>& global,
                    std::vector<unsigned int>& local, int n,
                    const int dims[2], MPI_Comm grid, bool gather) {
    int rank, processes;
    MPI_Comm_rank(grid, &rank);
    MPI_Comm_size(grid, &processes);
    for (int peer = 0; peer < processes; ++peer) {
        if (rank != 0 && rank != peer) continue;
        int coords[2];
        MPI_Cart_coords(grid, peer, 2, coords);
        const int rows = blockSize(n, dims[0], coords[0]);
        const int cols = blockSize(n, dims[1], coords[1]);
        if (rows == 0 || cols == 0) continue;
        const int firstRow = blockStart(n, dims[0], coords[0]);
        const int firstCol = blockStart(n, dims[1], coords[1]);
        if (peer == 0) {
            for (int i = 0; i < rows; ++i) {
                auto* whole = global.data() + size_t(firstRow + i) * n + firstCol;
                auto* tile = local.data() + size_t(i) * cols;
                if (gather) std::copy_n(tile, cols, whole);
                else std::copy_n(whole, cols, tile);
            }
        } else {
            // A row slab has at most INT_MAX elements even when the complete
            // tile does not; use the largest slabs supported by MPI counts.
            const int slabRows = INT_MAX / cols;
            for (int first = 0; first < rows;) {
                const int height = std::min(slabRows, rows - first);
                if (rank == 0) {
                    MPI_Datatype slab;
                    MPI_Type_vector(height, cols, n, MPI_UNSIGNED, &slab);
                    MPI_Type_commit(&slab);
                    auto* data = global.data() + size_t(firstRow + first) * n + firstCol;
                    if (gather) MPI_Recv(data, 1, slab, peer, 0, grid, MPI_STATUS_IGNORE);
                    else MPI_Send(data, 1, slab, peer, 0, grid);
                    MPI_Type_free(&slab);
                } else {
                    auto* data = local.data() + size_t(first) * cols;
                    if (gather) MPI_Send(data, height * cols, MPI_UNSIGNED, 0, 0, grid);
                    else MPI_Recv(data, height * cols, MPI_UNSIGNED, 0, 0, grid, MPI_STATUS_IGNORE);
                }
                first += height;
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, int n,
                   const int dims[2], const int coords[2],
                   MPI_Comm rowComm, MPI_Comm colComm) {
    const int rows = blockSize(n, dims[0], coords[0]);
    const int cols = blockSize(n, dims[1], coords[1]);
    const int firstRow = blockStart(n, dims[0], coords[0]);
    const int firstCol = blockStart(n, dims[1], coords[1]);
    std::vector<unsigned int> pivotRow(cols), pivotCol(rows);
    for (int k = 0; k < n; ++k) {
        const int ownerRow = blockOwner(k, n, dims[0]);
        const int ownerCol = blockOwner(k, n, dims[1]);
        if (coords[0] == ownerRow && cols != 0)
            std::copy_n(dist.data() + size_t(k - firstRow) * cols, cols, pivotRow.data());
        if (coords[1] == ownerCol && cols != 0)
            for (int i = 0; i < rows; ++i)
                pivotCol[i] = dist[size_t(i) * cols + k - firstCol];

        // Independent collectives can progress concurrently. Each rank receives
        // only its segment of the pivot row and column, never the full matrix.
        MPI_Request requests[2];
        MPI_Ibcast(pivotRow.data(), cols, MPI_UNSIGNED, ownerRow, colComm, &requests[0]);
        MPI_Ibcast(pivotCol.data(), rows, MPI_UNSIGNED, ownerCol, rowComm, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        if (cols == 0) continue;
        for (int i = 0; i < rows; ++i) {
            const unsigned int dik = pivotCol[i];
            // Positive weights and a zero diagonal make the pivot row invariant.
            if (dik == 0) continue;
            unsigned int* __restrict__ d = dist.data() + size_t(i) * cols;
            unsigned int* __restrict__ p = path.data() + size_t(i) * cols;
            const unsigned int* __restrict__ row = pivotRow.data();
            for (int j = 0; j < cols; ++j) {
                const unsigned int candidate = dik + row[j];
                const bool improve = candidate < d[j];
                p[j] = improve ? static_cast<unsigned int>(k) : p[j];
                d[j] = improve ? candidate : d[j];
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

int runBenchmark(int argc, char** argv) {
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || parsed > INT_MAX ||
                (parsed && parsed > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / parsed)) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", value);
                return 1;
            }
            numNodes = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2];
    MPI_Dims_create(processes, 2, dims);
    MPI_Comm grid, rowComm, colComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
    MPI_Cart_coords(grid, rank, 2, coords);
    MPI_Comm_split(grid, coords[0], coords[1], &rowComm);
    MPI_Comm_split(grid, coords[1], coords[0], &colComm);
    const int n = static_cast<int>(numNodes);
    const int rows = blockSize(n, dims[0], coords[0]);
    const int cols = blockSize(n, dims[1], coords[1]);
    std::vector<unsigned int> dist(size_t(rows) * cols);
    std::vector<unsigned int> path(dist.size());
    std::vector<unsigned int> global;
    
    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
        global.resize(numNodes * numNodes);
        initializeDistanceMatrix(global, numNodes, 1, MAX_DISTANCE);
    }
    transferMatrix(global, dist, n, dims, grid, false);
    // Reproduce the original path matrix's global source-row indices.
    for (int i = 0; cols != 0 && i < rows; ++i)
        std::fill_n(path.data() + size_t(i) * cols, cols,
                    static_cast<unsigned int>(blockStart(n, dims[0], coords[0]) + i));
    std::vector<unsigned int>().swap(global);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(grid);
    const double start = MPI_Wtime();
    
    floydWarshall(dist, path, n, dims, coords, rowComm, colComm);
    
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));

        // Floyd-Warshall has O(n³) complexity; use the slowest rank's time.
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = seconds > 0 ? ops / seconds / 1e9 : 0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    if (printResults || validate) {
        if (rank == 0) global.resize(numNodes * numNodes);
        transferMatrix(global, dist, n, dims, grid, true);
    }
    
    // Print results for external validation (integer hash-based)
    if (rank == 0 && printResults) {
        print_results_int(global, "DistanceMatrix");
    }
    
    // Validation
    int result = 0;
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(global, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }
    
    MPI_Bcast(&result, 1, MPI_INT, 0, grid);
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&grid);
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int result = 1;
    try {
        result = runBenchmark(argc, argv);
    } catch (const std::exception& error) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
