#include <algorithm>
#include <cerrno>
#include <climits>
#include <exception>
#include <limits>
#include <mpi.h>
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

// A rectangular, balanced block on a two-dimensional process grid.
struct Block {
    size_t rowBegin, rowEnd, colBegin, colEnd;
    size_t rows() const { return rowEnd - rowBegin; }
    size_t cols() const { return colEnd - colBegin; }
};

Block blockForRank(size_t n, int rank, const int* dims) {
    const size_t r = rank / dims[1], c = rank % dims[1];
    return {n * r / dims[0], n * (r + 1) / dims[0],
            n * c / dims[1], n * (c + 1) / dims[1]};
}

// Stream bounded-size slabs through rank zero. This preserves rand_r's exact
// original order without replicating the matrix or the random-number work.
// MPI counts and displacements remain within the classic MPI int limit.
void transferMatrix(std::vector<unsigned int>& local,
                    std::vector<unsigned int>& global, size_t n,
                    const Block& block, const int* dims, int rank, int ranks,
                    MPI_Comm grid, bool initialize) {
    if (n == 0) return;
    const size_t slabRows = std::max<size_t>(1,
        std::min<size_t>(1048576 / n, static_cast<size_t>(INT_MAX) / n));
    std::vector<int> counts(ranks), offsets(ranks);
    std::vector<unsigned int> packed, received;
    unsigned int seed = 42;
    for (size_t first = 0; first < n; first += slabRows) {
        const size_t last = std::min(n, first + slabRows);
        size_t total = 0;
        for (int r = 0; r < ranks; ++r) {
            const Block b = blockForRank(n, r, dims);
            const size_t begin = std::max(first, b.rowBegin);
            const size_t end = std::min(last, b.rowEnd);
            counts[r] = static_cast<int>((end > begin ? end - begin : 0) * b.cols());
            offsets[r] = static_cast<int>(total);
            total += counts[r];
        }
        const size_t begin = std::max(first, block.rowBegin);
        received.resize(counts[rank]);
        if (rank == 0) packed.resize(total);
        if (initialize) {
            if (rank == 0) {
                // Pack directly into each rank's contiguous block in the slab.
                for (size_t i = first; i < last; ++i) {
                    const int ownerRow = static_cast<int>(((i + 1) * dims[0] - 1) / n);
                    for (int c = 0; c < dims[1]; ++c) {
                        const int owner = ownerRow * dims[1] + c;
                        const Block b = blockForRank(n, owner, dims);
                        size_t pos = offsets[owner] +
                            (i - std::max(first, b.rowBegin)) * b.cols();
                        for (size_t j = b.colBegin; j < b.colEnd; ++j) {
                            const unsigned int value = 1 + static_cast<unsigned int>(
                                static_cast<double>(MAX_DISTANCE) * rand_r(&seed) /
                                static_cast<double>(RAND_MAX));
                            packed[pos++] = i == j ? 0 : value;
                        }
                    }
                }
            }
            MPI_Scatterv(packed.data(), counts.data(), offsets.data(), MPI_UNSIGNED,
                         received.data(), counts[rank], MPI_UNSIGNED, 0, grid);
            if (counts[rank] != 0) {
                std::copy(received.begin(), received.end(),
                          local.begin() + (begin - block.rowBegin) * block.cols());
            }
        } else {
            if (counts[rank] != 0) {
                std::copy_n(local.begin() + (begin - block.rowBegin) * block.cols(),
                            counts[rank], received.begin());
            }
            MPI_Gatherv(received.data(), counts[rank], MPI_UNSIGNED,
                        packed.data(), counts.data(), offsets.data(), MPI_UNSIGNED, 0, grid);
            if (rank == 0) {
                for (int r = 0; r < ranks; ++r) {
                    const Block b = blockForRank(n, r, dims);
                    const size_t rowStart = std::max(first, b.rowBegin);
                    const size_t rowStop = std::min(last, b.rowEnd);
                    for (size_t i = rowStart; i < rowStop; ++i) {
                        std::copy_n(packed.begin() + offsets[r] + (i - rowStart) * b.cols(),
                                    b.cols(), global.begin() + i * n + b.colBegin);
                    }
                }
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, size_t n,
                   const Block& block, const int* dims, int rank,
                   MPI_Comm rowComm, MPI_Comm colComm) {
    const int processRow = rank / dims[1], processCol = rank % dims[1];
    const size_t rows = block.rows(), cols = block.cols();
    std::vector<unsigned int> pivotRow(cols), pivotCol(rows);
    for (size_t k = 0; k < n; ++k) {
        const int ownerRow = static_cast<int>(((k + 1) * dims[0] - 1) / n);
        const int ownerCol = static_cast<int>(((k + 1) * dims[1] - 1) / n);
        if (processRow == ownerRow && cols != 0) {
            std::copy_n(dist.data() + (k - block.rowBegin) * cols,
                        cols, pivotRow.data());
        }
        if (processCol == ownerCol) {
            for (size_t i = 0; i < rows; ++i)
                pivotCol[i] = dist[i * cols + k - block.colBegin];
        }
        // Independent row/column collectives can progress concurrently.
        MPI_Request requests[2];
        MPI_Ibcast(pivotRow.data(), static_cast<int>(cols), MPI_UNSIGNED,
                   ownerRow, colComm, &requests[0]);
        MPI_Ibcast(pivotCol.data(), static_cast<int>(rows), MPI_UNSIGNED,
                   ownerCol, rowComm, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        if (cols == 0) continue;
        const unsigned int intermediate = static_cast<unsigned int>(k);
        for (size_t i = 0; i < rows; ++i) {
            const unsigned int dik = pivotCol[i];
            // The diagonal is zero and weights are positive, so this row
            // cannot change. All other updates preserve strict tie handling.
            if (block.rowBegin + i == k) continue;
            unsigned int* d = dist.data() + i * cols;
            unsigned int* p = path.data() + i * cols;
            const unsigned int* pivot = pivotRow.data();
            for (size_t j = 0; j < cols; ++j) {
                const unsigned int candidate = dik + pivot[j];
                if (candidate < d[j]) {
                    d[j] = candidate;
                    p[j] = intermediate;
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    int exitCode = 0;
    bool stop = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            errno = 0;
            const unsigned long long value = strtoull(arg, &end, 10);
            if (errno != 0 || end == arg || *end != '\0' || arg[0] == '-' ||
                value > static_cast<unsigned long long>(INT_MAX) ||
                value > std::numeric_limits<size_t>::max() / sizeof(unsigned int) /
                        std::max<unsigned long long>(1, value)) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", arg);
                exitCode = 1;
                stop = true;
                break;
            }
            numNodes = static_cast<size_t>(value);
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
            exitCode = 1;
            stop = true;
            break;
        }
    }
    if (stop) {
        MPI_Finalize();
        return exitCode;
    }

    try {
        int dims[2] = {0, 0};
        MPI_Dims_create(ranks, 2, dims);
        const int periods[2] = {0, 0};
        MPI_Comm grid, rowComm, colComm;
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
        // Explicit keys make communicator ranks equal to grid coordinates.
        MPI_Comm_split(grid, rank / dims[1], rank % dims[1], &rowComm);
        MPI_Comm_split(grid, rank % dims[1], rank / dims[1], &colComm);
        const Block block = blockForRank(numNodes, rank, dims);
        std::vector<unsigned int> dist(block.rows() * block.cols());
        std::vector<unsigned int> path(dist.size());
        std::vector<unsigned int> global;
        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
        }
        transferMatrix(dist, global, numNodes, block, dims, rank, ranks, grid, true);
        for (size_t i = 0; block.cols() != 0 && i < block.rows(); ++i) {
            std::fill_n(path.begin() + i * block.cols(), block.cols(),
                        static_cast<unsigned int>(block.rowBegin + i));
        }
        if (rank == 0) printf("Computing shortest paths...\n");
        MPI_Barrier(grid);
        const double start = MPI_Wtime();
        floydWarshall(dist, path, numNodes, block, dims, rank, rowComm, colComm);
        const double elapsed = MPI_Wtime() - start;
        double duration = 0;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
        if (rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
            const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", duration > 0 ? ops / duration / 1e9 : 0.0);
        }
        // Normal benchmark runs never gather or replicate the global matrix.
        if (validate || printResults) {
            if (rank == 0) global.resize(numNodes * numNodes);
            transferMatrix(dist, global, numNodes, block, dims, rank, ranks, grid, false);
            if (rank == 0) {
                if (printResults) print_results_int(global, "DistanceMatrix");
                if (validate) {
                    printf("Validating result...\n");
                    const bool valid = validateResult(global, numNodes);
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    exitCode = valid ? 0 : 1;
                }
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, grid);
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
        MPI_Comm_free(&grid);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return exitCode;
}
