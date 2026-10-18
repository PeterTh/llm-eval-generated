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

// Contiguous, balanced partitions, including empty blocks when ranks exceed n.
size_t blockStart(size_t n, int parts, int part) {
    return (n / parts) * part + std::min(n % parts, static_cast<size_t>(part));
}

// Generate the original rand_r stream exactly once. Only a process-row strip
// is materialized on rank zero; all other ranks allocate just their own tile.
void initializeMatrices(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path, size_t n,
                        const int dims[2], int rank) {
    unsigned int seed = 42;
    for (int r = 0; r < dims[0]; ++r) {
        const size_t first = blockStart(n, dims[0], r);
        const size_t rows = blockStart(n, dims[0], r + 1) - first;
        std::vector<unsigned int> strip;
        if (rank == 0) {
            strip.resize(rows * n);
            for (auto& value : strip) {
                value = 1 + (unsigned int)(static_cast<double>(MAX_DISTANCE) *
                                          rand_r(&seed) / (double)RAND_MAX);
            }
            for (size_t i = 0; i < rows; ++i) {
                strip[i * n + first + i] = 0;
            }
        }
        for (int c = 0; c < dims[1]; ++c) {
            const int target = r * dims[1] + c;
            const size_t col = blockStart(n, dims[1], c);
            const size_t cols = blockStart(n, dims[1], c + 1) - col;
            if (rows == 0 || cols == 0) continue;
            if (rank == 0) {
                if (target == 0) {
                    for (size_t i = 0; i < rows; ++i)
                        std::copy_n(strip.data() + i * n + col, cols,
                                    dist.data() + i * cols);
                } else {
                    MPI_Datatype tile;
                    MPI_Type_vector(static_cast<int>(rows), static_cast<int>(cols),
                                    static_cast<int>(n), MPI_UNSIGNED, &tile);
                    MPI_Type_commit(&tile);
                    MPI_Send(strip.data() + col, 1, tile, target, 0, MPI_COMM_WORLD);
                    MPI_Type_free(&tile);
                }
            } else if (rank == target) {
                MPI_Recv(dist.data(), static_cast<int>(dist.size()), MPI_UNSIGNED,
                         0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    const int r = rank / dims[1];
    const size_t first = blockStart(n, dims[0], r);
    const size_t rows = blockStart(n, dims[0], r + 1) - first;
    const size_t cols = blockStart(n, dims[1], rank % dims[1] + 1) -
                        blockStart(n, dims[1], rank % dims[1]);
    if (cols != 0)
        for (size_t i = 0; i < rows; ++i)
            std::fill_n(path.data() + i * cols, cols, static_cast<unsigned int>(first + i));
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, size_t n,
                   const int dims[2], int rank, MPI_Comm rowComm, MPI_Comm colComm) {
    const int r = rank / dims[1], c = rank % dims[1];
    const size_t firstRow = blockStart(n, dims[0], r);
    const size_t firstCol = blockStart(n, dims[1], c);
    const size_t rows = blockStart(n, dims[0], r + 1) - firstRow;
    const size_t cols = blockStart(n, dims[1], c + 1) - firstCol;
    std::vector<unsigned int> pivotRow(cols), pivotCol(rows);
    int ownerRow = 0, ownerCol = 0;
    for (size_t k = 0; k < n; ++k) {
        while (k >= blockStart(n, dims[0], ownerRow + 1)) ++ownerRow;
        while (k >= blockStart(n, dims[1], ownerCol + 1)) ++ownerCol;
        if (r == ownerRow && cols != 0)
            std::copy_n(dist.data() + (k - firstRow) * cols, cols, pivotRow.data());
        if (c == ownerCol)
            for (size_t i = 0; i < rows; ++i)
                pivotCol[i] = dist[i * cols + k - firstCol];

        // Independent broadcasts on orthogonal communicators can progress together.
        MPI_Request requests[2];
        MPI_Ibcast(pivotRow.data(), static_cast<int>(cols), MPI_UNSIGNED,
                   ownerRow, colComm, &requests[0]);
        MPI_Ibcast(pivotCol.data(), static_cast<int>(rows), MPI_UNSIGNED,
                   ownerCol, rowComm, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        if (cols == 0) continue;
        for (size_t i = 0; i < rows; ++i) {
            const unsigned int dik = pivotCol[i];
            // Positive edge weights and a zero diagonal make this row invariant.
            if (firstRow + i == k) continue;
            unsigned int* const distances = dist.data() + i * cols;
            unsigned int* const paths = path.data() + i * cols;
            for (size_t j = 0; j < cols; ++j) {
                const unsigned int candidate = dik + pivotRow[j];
                if (candidate < distances[j]) {
                    distances[j] = candidate;
                    paths[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

// Reconstruction is only needed for the existing validation and hash output.
std::vector<unsigned int> gatherDistances(const std::vector<unsigned int>& local,
                                        size_t n, const int dims[2], int rank) {
    std::vector<unsigned int> result(rank == 0 ? n * n : 0);
    for (int r = 0; r < dims[0]; ++r) {
        const size_t row = blockStart(n, dims[0], r);
        const size_t rows = blockStart(n, dims[0], r + 1) - row;
        for (int c = 0; c < dims[1]; ++c) {
            const int source = r * dims[1] + c;
            const size_t col = blockStart(n, dims[1], c);
            const size_t cols = blockStart(n, dims[1], c + 1) - col;
            if (rows == 0 || cols == 0) continue;
            if (rank == 0) {
                if (source == 0) {
                    for (size_t i = 0; i < rows; ++i)
                        std::copy_n(local.data() + i * cols, cols,
                                    result.data() + (row + i) * n + col);
                } else {
                    MPI_Datatype tile;
                    MPI_Type_vector(static_cast<int>(rows), static_cast<int>(cols),
                                    static_cast<int>(n), MPI_UNSIGNED, &tile);
                    MPI_Type_commit(&tile);
                    MPI_Recv(result.data() + row * n + col, 1, tile, source, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Type_free(&tile);
                }
            } else if (rank == source) {
                MPI_Send(local.data(), static_cast<int>(local.size()), MPI_UNSIGNED,
                         0, 1, MPI_COMM_WORLD);
            }
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

int runBenchmark(int argc, char** argv, int rank, int processes) {
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* value = argv[++i];
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end != '\0' ||
                parsed > static_cast<unsigned long long>(INT_MAX)) {
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
    
    int dims[2] = {0, 0};
    MPI_Dims_create(processes, 2, dims);
    const size_t maxRows = blockStart(numNodes, dims[0], 1);
    const size_t maxCols = blockStart(numNodes, dims[1], 1);
    if ((maxCols != 0 && maxRows > static_cast<size_t>(INT_MAX) / maxCols) ||
        (numNodes != 0 && numNodes >
         std::numeric_limits<size_t>::max() / sizeof(unsigned int) / numNodes)) {
        if (rank == 0) fprintf(stderr, "Matrix exceeds supported MPI or allocation size\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(MPI_COMM_WORLD, rank / dims[1], rank % dims[1], &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, rank % dims[1], rank / dims[1], &colComm);
    const size_t rows = blockStart(numNodes, dims[0], rank / dims[1] + 1) -
                        blockStart(numNodes, dims[0], rank / dims[1]);
    const size_t cols = blockStart(numNodes, dims[1], rank % dims[1] + 1) -
                        blockStart(numNodes, dims[1], rank % dims[1]);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<unsigned int> dist(rows * cols);
    std::vector<unsigned int> path(rows * cols);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeMatrices(dist, path, numNodes, dims, rank);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, dims, rank, rowComm, colComm);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration * 1000.0);

        // Calculate operations per second; Floyd-Warshall has O(n³) complexity.
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = duration > 0 ? ops / duration / 1e9 : 0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> result;
    if (printResults || validate)
        result = gatherDistances(dist, numNodes, dims, rank);
    
    // Print results for external validation (integer hash-based)
    if (rank == 0 && printResults) {
        print_results_int(result, "DistanceMatrix");
    }
    
    // Validation
    int status = 0;
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(result, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Finalize();
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    try {
        return runBenchmark(argc, argv, rank, processes);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
