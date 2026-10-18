#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <exception>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Contiguous, balanced partitions, including empty partitions when P > N.
size_t partitionStart(size_t n, int parts, int coordinate) {
    return (n / parts) * coordinate + std::min(n % parts, size_t(coordinate));
}

int partitionOwner(size_t k, size_t n, int parts) {
    const size_t small = n / parts;
    const size_t largeEnd = (small + 1) * (n % parts);
    return k < largeEnd ? int(k / (small + 1))
                        : int(n % parts + (k - largeEnd) / small);
}

void initializeMatrices(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path, size_t n,
                        size_t firstRow, size_t rows, size_t firstCol, size_t cols,
                        int rank, MPI_Comm grid) {
    // Preserve the exact rand_r stream without replicating the global matrix.
    // A seed per row lets each tile generate its own entries independently.
    std::vector<unsigned int> seeds(n);
    if (rank == 0) {
        unsigned int seed = 42;
        for (size_t i = 0; i < n; ++i) {
            seeds[i] = seed;
            if (i + 1 < n) {
                for (size_t j = 0; j < n; ++j) (void)rand_r(&seed);
            }
        }
    }
    MPI_Bcast(seeds.data(), int(n), MPI_UNSIGNED, 0, grid);
    for (size_t i = 0; i < rows; ++i) {
        unsigned int seed = seeds[firstRow + i];
        for (size_t j = 0; j < firstCol; ++j) (void)rand_r(&seed);
        for (size_t j = 0; j < cols; ++j) {
            const unsigned int value = 1 + (unsigned int)(
                double(MAX_DISTANCE) * rand_r(&seed) / double(RAND_MAX));
            dist[i * cols + j] = firstRow + i == firstCol + j ? 0 : value;
            path[i * cols + j] = static_cast<unsigned int>(firstRow + i);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, size_t n,
                   size_t firstRow, size_t rows, size_t firstCol, size_t cols,
                   const int* dims, const int* coords,
                   MPI_Comm rowComm, MPI_Comm colComm) {
    std::vector<unsigned int> pivotRow(std::max(cols, size_t(1)));
    std::vector<unsigned int> pivotCol(std::max(rows, size_t(1)));
    for (size_t k = 0; k < n; ++k) {
        const int ownerRow = partitionOwner(k, n, dims[0]);
        const int ownerCol = partitionOwner(k, n, dims[1]);
        if (coords[0] == ownerRow && cols != 0) {
            std::copy_n(dist.data() + (k - firstRow) * cols, cols, pivotRow.data());
        }
        if (coords[1] == ownerCol) {
            for (size_t i = 0; i < rows; ++i) pivotCol[i] = dist[i * cols + k - firstCol];
        }
        // Independent collectives can progress together on the two grid axes.
        MPI_Request requests[2];
        MPI_Ibcast(pivotRow.data(), int(cols), MPI_UNSIGNED, ownerRow, colComm, &requests[0]);
        MPI_Ibcast(pivotCol.data(), int(rows), MPI_UNSIGNED, ownerCol, rowComm, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        // Contiguous inner loops permit SIMD and reuse each pivot-row segment
        // from cache. Positive weights and a zero diagonal keep pivots stable.
        constexpr size_t tileWidth = 256;
        for (size_t jj = 0; jj < cols; jj += tileWidth) {
            const size_t end = std::min(jj + tileWidth, cols);
            for (size_t i = 0; i < rows; ++i) {
                const unsigned int dik = pivotCol[i];
                if (dik == 0) continue;
                unsigned int* d = dist.data() + i * cols;
                unsigned int* p = path.data() + i * cols;
                for (size_t j = jj; j < end; ++j) {
                    const unsigned int candidate = dik + pivotRow[j];
                    const bool improved = candidate < d[j];
                    p[j] = improved ? static_cast<unsigned int>(k) : p[j];
                    d[j] = improved ? candidate : d[j];
                }
            }
        }
    }
}

// Gather only for the original root-only validation/output. Receive directly
// into the global row-major layout, without packing another full matrix.
std::vector<unsigned int> gatherDistance(const std::vector<unsigned int>& local,
                                        size_t n, const int* dims, int rank,
                                        int processes, MPI_Comm grid) {
    std::vector<unsigned int> global;
    if (rank == 0) {
        global.resize(n * n);
        for (int source = 0; source < processes; ++source) {
            int coords[2];
            MPI_Cart_coords(grid, source, 2, coords);
            const size_t r = partitionStart(n, dims[0], coords[0]);
            const size_t c = partitionStart(n, dims[1], coords[1]);
            const size_t rows = partitionStart(n, dims[0], coords[0] + 1) - r;
            const size_t cols = partitionStart(n, dims[1], coords[1] + 1) - c;
            if (rows == 0 || cols == 0) continue;
            if (source == 0) {
                for (size_t i = 0; i < rows; ++i)
                    std::copy_n(local.data() + i * cols, cols, global.data() + (r + i) * n + c);
            } else {
                MPI_Datatype tile;
                MPI_Type_vector(int(rows), int(cols), int(n), MPI_UNSIGNED, &tile);
                MPI_Type_commit(&tile);
                MPI_Recv(global.data() + r * n + c, 1, tile, source, 0, grid, MPI_STATUS_IGNORE);
                MPI_Type_free(&tile);
            }
        }
    } else if (!local.empty()) {
        MPI_Send(local.data(), int(local.size()), MPI_UNSIGNED, 0, 0, grid);
    }
    return global;
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

int main(int argc, char** argv) try {
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
            char* end = nullptr;
            const char* value = argv[++i];
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || parsed > INT_MAX ||
                parsed > std::numeric_limits<size_t>::max() / std::max(parsed, 1ULL)) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", value);
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
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    int dims[2] = {0, 0}, coords[2], periods[2] = {0, 0};
    MPI_Dims_create(processes, 2, dims);
    MPI_Comm grid, rowComm, colComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
    MPI_Cart_coords(grid, rank, 2, coords);
    MPI_Comm_split(grid, coords[0], coords[1], &rowComm);
    MPI_Comm_split(grid, coords[1], coords[0], &colComm);
    const size_t firstRow = partitionStart(numNodes, dims[0], coords[0]);
    const size_t firstCol = partitionStart(numNodes, dims[1], coords[1]);
    const size_t rows = partitionStart(numNodes, dims[0], coords[0] + 1) - firstRow;
    const size_t cols = partitionStart(numNodes, dims[1], coords[1] + 1) - firstCol;
    int tooLarge = rows * cols > INT_MAX, anyTooLarge;
    MPI_Allreduce(&tooLarge, &anyTooLarge, 1, MPI_INT, MPI_MAX, grid);
    if (anyTooLarge) {
        if (rank == 0) fprintf(stderr, "Local matrix exceeds MPI count limit; use more ranks.\n");
        MPI_Abort(grid, 1);
    }
    std::vector<unsigned int> dist(rows * cols);
    std::vector<unsigned int> path(rows * cols);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeMatrices(dist, path, numNodes, firstRow, rows, firstCol, cols, rank, grid);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(grid);
    const double start = MPI_Wtime();
    
    floydWarshall(dist, path, numNodes, firstRow, rows, firstCol, cols,
                  dims, coords, rowComm, colComm);
    
    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));

        // Floyd-Warshall performs O(n³) relaxation attempts.
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = duration > 0 ? ops / duration / 1e9 : 0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> global;
    if (printResults || validate) global = gatherDistance(dist, numNodes, dims, rank, processes, grid);
    
    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(global, "DistanceMatrix");
    }
    
    // Validation
    int status = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(global, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }
    
    MPI_Bcast(&status, 1, MPI_INT, 0, grid);
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return status;
} catch (const std::exception& error) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, error.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
}
