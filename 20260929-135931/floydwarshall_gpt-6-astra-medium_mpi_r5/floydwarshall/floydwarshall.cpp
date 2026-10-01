#include <algorithm>
#include <cerrno>
#include <climits>
#include <exception>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Only the MPI C interface is used; omit obsolete MPI C++ bindings.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// A contiguous 2-D block per rank. Communicator ranks match grid coordinates.
struct Grid {
    int rank, dims[2] = {0, 0}, row, col;
    size_t n, firstRow, firstCol, rows, cols;
    MPI_Comm rowComm, colComm;

    explicit Grid(size_t nodes) : n(nodes) {
        int ranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        MPI_Dims_create(ranks, 2, dims);
        row = rank / dims[1];
        col = rank % dims[1];
        firstRow = boundary(row, 0);
        firstCol = boundary(col, 1);
        rows = boundary(row + 1, 0) - firstRow;
        cols = boundary(col + 1, 1) - firstCol;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
    }

    size_t boundary(int coordinate, int axis) const {
        return n * static_cast<size_t>(coordinate) / dims[axis];
    }

    ~Grid() {
        // On allocation failure, let main abort immediately; other ranks may
        // still be inside communication and cannot join collective cleanup.
        if (std::uncaught_exceptions()) return;
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
};

// Chunk point-to-point transfers to avoid MPI's int count limit.
void transfer(unsigned int* data, size_t count, int peer, bool send) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send)
            MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD);
        else
            MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

void initializeMatrices(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path, const Grid& g) {
    if (g.cols)
        for (size_t i = 0; i < g.rows; ++i)
            std::fill_n(path.data() + i * g.cols, g.cols,
                        static_cast<unsigned int>(g.firstRow + i));

    if (g.rank != 0) {
        transfer(dist.data(), dist.size(), 0, false);
        return;
    }

    // Generate the exact original rand_r sequence once. Only one process-row
    // stripe is staged, rather than a replicated global matrix on every rank.
    unsigned int seed = 42;
    for (int r = 0; r < g.dims[0]; ++r) {
        size_t first = g.boundary(r, 0);
        size_t rows = g.boundary(r + 1, 0) - first;
        std::vector<unsigned int> stripe(rows * g.n);
        for (auto& value : stripe)
            value = 1 + static_cast<unsigned int>(
                static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
        for (size_t i = 0; i < rows; ++i)
            stripe[i * g.n + first + i] = 0;
        for (int c = 0; c < g.dims[1]; ++c) {
            size_t firstCol = g.boundary(c, 1);
            size_t cols = g.boundary(c + 1, 1) - firstCol;
            std::vector<unsigned int> block(rows * cols);
            if (cols)
                for (size_t i = 0; i < rows; ++i)
                    std::copy_n(stripe.data() + i * g.n + firstCol, cols,
                                block.data() + i * cols);
            int destination = r * g.dims[1] + c;
            if (destination == 0)
                dist.swap(block);
            else
                transfer(block.data(), block.size(), destination, true);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                  std::vector<unsigned int>& path, const Grid& g) {
    std::vector<unsigned int> pivotRow(g.cols), pivotCol(g.rows);
    for (size_t k = 0; k < g.n; ++k) {
        // This ownership formula also handles empty blocks when ranks > nodes.
        int ownerRow = static_cast<int>(((k + 1) * g.dims[0] - 1) / g.n);
        int ownerCol = static_cast<int>(((k + 1) * g.dims[1] - 1) / g.n);
        if (g.row == ownerRow && g.cols)
            std::copy_n(dist.data() + (k - g.firstRow) * g.cols,
                        g.cols, pivotRow.data());
        if (g.col == ownerCol)
            for (size_t i = 0; i < g.rows; ++i)
                pivotCol[i] = dist[i * g.cols + k - g.firstCol];

        // Independent broadcasts can progress together on the two grid axes.
        MPI_Request requests[2];
        MPI_Ibcast(pivotRow.data(), static_cast<int>(g.cols), MPI_UNSIGNED,
                   ownerRow, g.colComm, &requests[0]);
        MPI_Ibcast(pivotCol.data(), static_cast<int>(g.rows), MPI_UNSIGNED,
                   ownerCol, g.rowComm, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        // Positive edge weights and zero diagonals keep the pivot row/column
        // unchanged. Snapshotting them preserves the original strict updates.
        if (!g.cols) continue;
        for (size_t i = 0; i < g.rows; ++i) {
            if (g.firstRow + i == k) continue;
            unsigned int* distance = dist.data() + i * g.cols;
            unsigned int* intermediate = path.data() + i * g.cols;
            const unsigned int dik = pivotCol[i];
            for (size_t j = 0; j < g.cols; ++j) {
                const unsigned int candidate = dik + pivotRow[j];
                const bool improved = candidate < distance[j];
                intermediate[j] = improved ? static_cast<unsigned int>(k) : intermediate[j];
                distance[j] = improved ? candidate : distance[j];
            }
        }
    }
}

std::vector<unsigned int> gatherDistance(std::vector<unsigned int>& dist,
                                        const Grid& g) {
    if (g.rank != 0) {
        transfer(dist.data(), dist.size(), 0, true);
        return {};
    }
    std::vector<unsigned int> result(g.n * g.n);
    for (int r = 0; r < g.dims[0]; ++r) {
        size_t firstRow = g.boundary(r, 0);
        size_t rows = g.boundary(r + 1, 0) - firstRow;
        for (int c = 0; c < g.dims[1]; ++c) {
            size_t firstCol = g.boundary(c, 1);
            size_t cols = g.boundary(c + 1, 1) - firstCol;
            int source = r * g.dims[1] + c;
            std::vector<unsigned int> received;
            if (source) {
                received.resize(rows * cols);
                transfer(received.data(), received.size(), source, false);
            }
            const auto& block = source ? received : dist;
            if (cols)
                for (size_t i = 0; i < rows; ++i)
                    std::copy_n(block.data() + i * cols, cols,
                                result.data() + (firstRow + i) * g.n + firstCol);
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

int runBenchmark(int argc, char** argv, int rank) {
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            errno = 0;
            unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value == end || *end || *value == '-' || parsed > INT_MAX ||
                (parsed && parsed > std::numeric_limits<size_t>::max() /
                                     sizeof(unsigned int) / parsed)) {
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

    Grid grid(numNodes);
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> dist(grid.rows * grid.cols);
    std::vector<unsigned int> path(dist.size());
    initializeMatrices(dist, path, grid);
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    floydWarshall(dist, path, grid);
    double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", duration > 0 ? ops / duration / 1e9 : 0);
    }

    int status = 0;
    if (printResults || validate) {
        auto result = gatherDistance(dist, grid);
        if (rank == 0) {
            if (printResults) print_results_int(result, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(result, numNodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int status;
    try {
        status = runBenchmark(argc, argv, rank);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
