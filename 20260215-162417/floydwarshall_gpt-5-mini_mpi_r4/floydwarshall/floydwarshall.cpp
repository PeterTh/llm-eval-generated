#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Original column-major index calculation for flattened 2D array
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

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

// Distributed Floyd-Warshall using row-block distribution (each rank owns a set of rows)
// The flattened storage used by the benchmark is column-major; for communication
// we pack rows into contiguous buffers (row-major blocks) for each rank.
long floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Compute row distribution
    const size_t base_rows = numNodes / world_size;
    const size_t rem = numNodes % world_size;

    auto rows_for_rank = [&](int r) -> size_t {
        return r < (int)rem ? base_rows + 1 : base_rows;
    };

    auto row_start_for_rank = [&](int r) -> size_t {
        if (r < (int)rem) return (size_t)r * (base_rows + 1);
        return rem * (base_rows + 1) + (size_t)(r - rem) * base_rows;
    };

    const size_t local_rows = rows_for_rank(world_rank);
    const size_t local_row_start = row_start_for_rank(world_rank);

    // Root packs and scatters rows (rows packed in contiguous blocks: row-major per-row)
    std::vector<int> sendcounts(world_size), displs(world_size);
    std::vector<unsigned int> sendbuf_dist; // packed row-major blocks
    std::vector<unsigned int> sendbuf_path;

    if (world_rank == 0) {
        // Build packed send buffers arranged as concatenated row blocks for each rank
        sendbuf_dist.reserve(numNodes * numNodes);
        sendbuf_path.reserve(numNodes * numNodes);
        int offset = 0;
        for (int r = 0; r < world_size; ++r) {
            size_t rs = row_start_for_rank(r);
            size_t lr = rows_for_rank(r);
            sendcounts[r] = static_cast<int>(lr * numNodes);
            displs[r] = offset;
            offset += sendcounts[r];

            for (size_t i = 0; i < lr; ++i) {
                const size_t global_row = rs + i;
                for (size_t j = 0; j < numNodes; ++j) {
                    // original dist stored column-major at idx2(row, col)
                    sendbuf_dist.push_back(dist[idx2(global_row, j, numNodes)]);
                    sendbuf_path.push_back(path[idx2(global_row, j, numNodes)]);
                }
            }
        }
    }

    // Allocate local storage (rows packed row-major: local_rows x numNodes)
    std::vector<unsigned int> local_dist;
    std::vector<unsigned int> local_path;
    local_dist.resize(local_rows * numNodes);
    local_path.resize(local_rows * numNodes);

    // Scatter packed rows
    MPI_Scatterv(world_rank == 0 ? sendbuf_dist.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_dist.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(world_rank == 0 ? sendbuf_path.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_path.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Buffer for k-th row (numNodes entries)
    std::vector<unsigned int> kth_row(numNodes);

    // Synchronize and start timing on rank 0
    MPI_Barrier(MPI_COMM_WORLD);
    long duration_ms = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> tstart;
    if (world_rank == 0) tstart = std::chrono::high_resolution_clock::now();

    // Main distributed Floyd-Warshall
    for (size_t k = 0; k < numNodes; ++k) {
        // Determine owner of row k
        int owner;
        if (k < (base_rows + 1) * rem) {
            owner = static_cast<int>(k / (base_rows + 1));
        } else {
            owner = static_cast<int>(rem + (k - (base_rows + 1) * rem) / base_rows);
        }

        // If owner holds the row, pack it into kth_row
        if (owner == world_rank) {
            size_t local_k = k - local_row_start;
            unsigned int* src = &local_dist[local_k * numNodes];
            std::memcpy(kth_row.data(), src, sizeof(unsigned int) * numNodes);
        }

        // Broadcast kth row from owner to all ranks
        MPI_Bcast(kth_row.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Update local rows using kth_row
        for (size_t i_local = 0; i_local < local_rows; ++i_local) {
            unsigned int distIK = local_dist[i_local * numNodes + k];
            // If distIK is INF then skip inner loop to avoid overflow and unnecessary ops
            if (distIK >= INF) continue;

            unsigned int* row_ptr = &local_dist[i_local * numNodes];
            unsigned int* path_ptr = &local_path[i_local * numNodes];

            for (size_t j = 0; j < numNodes; ++j) {
                unsigned int distKJ = kth_row[j];
                if (distKJ >= INF) continue;
                unsigned int newDist = distIK + distKJ;
                if (newDist < row_ptr[j]) {
                    row_ptr[j] = newDist;
                    path_ptr[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }

    // Finish timing
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        auto tend = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count();
    }

    // Gather updated rows back to root (packed row-major blocks)
    std::vector<unsigned int> recvbuf_dist;
    std::vector<unsigned int> recvbuf_path;
    if (world_rank == 0) {
        recvbuf_dist.resize(numNodes * numNodes);
        recvbuf_path.resize(numNodes * numNodes);
    }

    MPI_Gatherv(local_dist.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED,
                world_rank == 0 ? recvbuf_dist.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_path.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED,
                world_rank == 0 ? recvbuf_path.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    // Unpack recvbuf back into original column-major layout on root
    if (world_rank == 0) {
        int offset = 0;
        for (int r = 0; r < world_size; ++r) {
            size_t rs = row_start_for_rank(r);
            size_t lr = rows_for_rank(r);
            for (size_t i = 0; i < lr; ++i) {
                size_t global_row = rs + i;
                for (size_t j = 0; j < numNodes; ++j) {
                    dist[idx2(global_row, j, numNodes)] = recvbuf_dist[offset + i * numNodes + j];
                    path[idx2(global_row, j, numNodes)] = recvbuf_path[offset + i * numNodes + j];
                }
            }
            offset += static_cast<int>(lr * numNodes);
        }
    }

    return duration_ms;
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
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse the same)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (world_rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    // Root initializes full matrices; other ranks keep empty vectors
    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;
    if (world_rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    if (world_rank == 0) printf("Computing shortest paths...\n");

    long duration_ms = floydWarshall(dist, path, numNodes);

    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);
            
            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
