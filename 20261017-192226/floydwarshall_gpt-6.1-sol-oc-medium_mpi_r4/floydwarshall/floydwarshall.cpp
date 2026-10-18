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

// Balanced, contiguous blocks, including empty blocks when there are more
// process rows/columns than vertices.
size_t blockStart(size_t n, int blocks, int coordinate) {
    return (n / blocks) * coordinate + std::min(n % blocks, size_t(coordinate));
}

int blockOwner(size_t k, size_t n, int blocks) {
    const size_t base = n / blocks;
    const size_t remainder = n % blocks;
    const size_t boundary = (base + 1) * remainder;
    return k < boundary ? int(k / (base + 1))
                        : int(remainder + (k - boundary) / base);
}

// Chunk point-to-point transfers so matrix blocks need not fit an MPI int count.
void transfer(unsigned int* data, size_t count, int peer, bool send, MPI_Comm comm) {
    while (count) {
        const int chunk = int(std::min(count, size_t(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, comm);
        else MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, comm, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

// Row leaders pass the RNG state in global row order, preserving rand_r's exact
// sequence without redundant generation. Each leader scatters its own strip;
// these transfers overlap generation of the next strip on the next leader.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              const int dims[2], const int coords[2],
                              int rank, MPI_Comm grid) {
    const size_t rows = blockStart(n, dims[0], coords[0] + 1) -
                        blockStart(n, dims[0], coords[0]);
    const size_t cols = blockStart(n, dims[1], coords[1] + 1) -
                        blockStart(n, dims[1], coords[1]);
    int leaderCoords[2] = {coords[0], 0}, leader;
    MPI_Cart_rank(grid, leaderCoords, &leader);
    if (coords[1] != 0) {
        transfer(dist.data(), rows * cols, leader, false, grid);
        return;
    }
    unsigned int seed = 42;
    if (coords[0] > 0) {
        int previousCoords[2] = {coords[0] - 1, 0}, previous;
        MPI_Cart_rank(grid, previousCoords, &previous);
        MPI_Recv(&seed, 1, MPI_UNSIGNED, previous, 1, grid, MPI_STATUS_IGNORE);
    }
    std::vector<unsigned int> strip(rows * n), packed;
    for (auto& value : strip)
        value = 1 + (unsigned int)(double(MAX_DISTANCE) * rand_r(&seed) / (double)RAND_MAX);
    if (coords[0] + 1 < dims[0]) {
        int nextCoords[2] = {coords[0] + 1, 0}, next;
        MPI_Cart_rank(grid, nextCoords, &next);
        MPI_Send(&seed, 1, MPI_UNSIGNED, next, 1, grid);
    }
    const size_t first = blockStart(n, dims[0], coords[0]);
    for (size_t i = 0; i < rows; ++i) strip[i * n + first + i] = 0;
    for (int c = 0; c < dims[1]; ++c) {
        const size_t start = blockStart(n, dims[1], c);
        const size_t width = blockStart(n, dims[1], c + 1) - start;
        packed.resize(rows * width);
        for (size_t i = 0; i < rows && width; ++i)
            std::copy_n(strip.data() + i * n + start, width, packed.data() + i * width);
        int targetCoords[2] = {coords[0], c}, target;
        MPI_Cart_rank(grid, targetCoords, &target);
        if (target == rank) std::copy(packed.begin(), packed.end(), dist.begin());
        else transfer(packed.data(), packed.size(), target, true, grid);
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                  size_t n, const int dims[2], const int coords[2],
                  MPI_Comm rowComm, MPI_Comm colComm) {
    const size_t rowStart = blockStart(n, dims[0], coords[0]);
    const size_t colStart = blockStart(n, dims[1], coords[1]);
    const size_t rows = blockStart(n, dims[0], coords[0] + 1) - rowStart;
    const size_t cols = blockStart(n, dims[1], coords[1] + 1) - colStart;
    std::vector<unsigned int> pivotRow(cols), pivotCol(rows);
    for (size_t k = 0; k < n; ++k) {
        const int ownerRow = blockOwner(k, n, dims[0]);
        const int ownerCol = blockOwner(k, n, dims[1]);
        if (coords[0] == ownerRow && cols)
            std::copy_n(dist.data() + (k - rowStart) * cols, cols, pivotRow.data());
        if (coords[1] == ownerCol)
            for (size_t i = 0; i < rows; ++i) pivotCol[i] = dist[i * cols + k - colStart];
        // Independent collectives on orthogonal communicators can progress together.
        MPI_Request requests[2];
        MPI_Ibcast(pivotRow.data(), int(cols), MPI_UNSIGNED, ownerRow, colComm, &requests[0]);
        MPI_Ibcast(pivotCol.data(), int(rows), MPI_UNSIGNED, ownerCol, rowComm, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        for (size_t i = 0; i < rows && cols; ++i) {
            const unsigned int dik = pivotCol[i];
            if (dik == 0) continue; // The diagonal pivot row cannot improve itself.
            unsigned int* __restrict__ distances = dist.data() + i * cols;
            unsigned int* __restrict__ paths = path.data() + i * cols;
            const unsigned int* __restrict__ pivot = pivotRow.data();
            for (size_t j = 0; j < cols; ++j) {
                const unsigned int candidate = dik + pivot[j];
                if (candidate < distances[j]) {
                    distances[j] = candidate;
                    paths[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

std::vector<unsigned int> gatherDistances(std::vector<unsigned int>& local, size_t n,
                                        const int dims[2], int rank, MPI_Comm grid) {
    if (rank != 0) {
        transfer(local.data(), local.size(), 0, true, grid);
        return {};
    }
    std::vector<unsigned int> result(n * n), packed;
    for (int r = 0; r < dims[0]; ++r) {
        const size_t first = blockStart(n, dims[0], r);
        const size_t height = blockStart(n, dims[0], r + 1) - first;
        for (int c = 0; c < dims[1]; ++c) {
            const size_t start = blockStart(n, dims[1], c);
            const size_t width = blockStart(n, dims[1], c + 1) - start;
            int sourceCoords[2] = {r, c}, source;
            MPI_Cart_rank(grid, sourceCoords, &source);
            packed.resize(height * width);
            if (source != 0) transfer(packed.data(), packed.size(), source, false, grid);
            const auto& block = source == 0 ? local : packed;
            for (size_t i = 0; i < height && width; ++i)
                std::copy_n(block.data() + i * width, width, result.data() + (first + i) * n + start);
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
            char* end = nullptr;
            const char* value = argv[++i];
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end ||
                parsed > INT_MAX ||
                parsed > std::numeric_limits<size_t>::max() / sizeof(unsigned int) /
                             std::max(1ULL, parsed)) {
                if (rank == 0) printf("Invalid number of nodes: %s\n", value);
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
    
    int status = 0;
    try {
        int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2];
        MPI_Dims_create(processes, 2, dims);
        MPI_Comm grid, rowComm, colComm;
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
        MPI_Cart_coords(grid, rank, 2, coords);
        MPI_Comm_split(grid, coords[0], coords[1], &rowComm);
        MPI_Comm_split(grid, coords[1], coords[0], &colComm);
        const size_t rowStart = blockStart(numNodes, dims[0], coords[0]);
        const size_t colStart = blockStart(numNodes, dims[1], coords[1]);
        const size_t rows = blockStart(numNodes, dims[0], coords[0] + 1) - rowStart;
        const size_t cols = blockStart(numNodes, dims[1], coords[1] + 1) - colStart;
        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
        }
        std::vector<unsigned int> dist(rows * cols), path(rows * cols);
        initializeDistanceMatrix(dist, numNodes, dims, coords, rank, grid);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                path[i * cols + j] = static_cast<unsigned int>(colStart + j);

        if (rank == 0) printf("Computing shortest paths...\n");
        MPI_Barrier(grid);
        const double start = MPI_Wtime();
        floydWarshall(dist, path, numNodes, dims, coords, rowComm, colComm);
        const double elapsed = MPI_Wtime() - start;
        double duration = 0;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
        if (rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
            const double ops = double(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", duration > 0 ? ops / duration / 1e9 : 0.0);
        }

        // A full matrix is needed only for the original output/validation routines.
        if (printResults || validate) {
            auto result = gatherDistances(dist, numNodes, dims, rank, grid);
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
