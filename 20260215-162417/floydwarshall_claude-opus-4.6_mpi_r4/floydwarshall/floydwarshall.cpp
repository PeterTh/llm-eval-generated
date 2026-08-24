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

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    
    // 2. Triangle inequality check on a sample
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks)
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
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Row-block distribution: each process owns a contiguous block of rows
    // Row i is stored contiguously at dist[i*numNodes .. (i+1)*numNodes-1]
    // (since dist[idx2(j, i, n)] = i*n + j)
    size_t base_rows = numNodes / (size_t)nprocs;
    size_t remainder = numNodes % (size_t)nprocs;

    std::vector<int> row_counts(nprocs), row_offsets(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        row_counts[r] = (int)(base_rows + ((size_t)r < remainder ? 1 : 0));
        row_offsets[r] = (int)((size_t)r * base_rows + std::min((size_t)r, remainder));
    }

    size_t local_rows = (size_t)row_counts[rank];

    // Sendcounts/displacements in elements for Scatterv/Gatherv
    std::vector<int> sendcounts(nprocs), displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        sendcounts[r] = row_counts[r] * (int)numNodes;
        displs[r] = row_offsets[r] * (int)numNodes;
    }

    // Allocate local row blocks
    std::vector<unsigned int> local_dist(local_rows * numNodes);
    std::vector<unsigned int> local_path(local_rows * numNodes);

    // Rank 0 initializes full matrices, then scatters
    {
        std::vector<unsigned int> dist, path;
        if (rank == 0) {
            printf("Initializing graph...\n");
            dist.resize(numNodes * numNodes);
            path.resize(numNodes * numNodes);
            initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
            initializePathMatrix(path, numNodes);
        }

        MPI_Scatterv(rank == 0 ? dist.data() : nullptr,
                      sendcounts.data(), displs.data(), MPI_UNSIGNED,
                      local_dist.data(), (int)(local_rows * numNodes), MPI_UNSIGNED,
                      0, MPI_COMM_WORLD);
        MPI_Scatterv(rank == 0 ? path.data() : nullptr,
                      sendcounts.data(), displs.data(), MPI_UNSIGNED,
                      local_path.data(), (int)(local_rows * numNodes), MPI_UNSIGNED,
                      0, MPI_COMM_WORLD);
    } // dist, path freed here

    // Precompute owner rank for each global row k
    std::vector<int> owner_rank(numNodes);
    for (int r = 0; r < nprocs; ++r) {
        for (int i = 0; i < row_counts[r]; ++i) {
            owner_rank[row_offsets[r] + i] = r;
        }
    }

    // Buffer for broadcasting row k of the distance matrix
    std::vector<unsigned int> row_k_buf(numNodes);

    if (rank == 0) printf("Computing shortest paths...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Floyd-Warshall: for each intermediate node k, broadcast row k
    // then each process updates its local rows
    for (size_t k = 0; k < numNodes; ++k) {
        int owner = owner_rank[k];

        // Owner copies its local row k into the broadcast buffer
        if (rank == owner) {
            size_t lk = k - (size_t)row_offsets[owner];
            memcpy(row_k_buf.data(), &local_dist[lk * numNodes],
                   numNodes * sizeof(unsigned int));
        }

        MPI_Bcast(row_k_buf.data(), (int)numNodes, MPI_UNSIGNED,
                  owner, MPI_COMM_WORLD);

        // Update local rows
        for (size_t li = 0; li < local_rows; ++li) {
            unsigned int* __restrict__ row_dist = &local_dist[li * numNodes];
            unsigned int* __restrict__ row_path = &local_path[li * numNodes];
            const unsigned int dist_ik = row_dist[k];

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = dist_ik + row_k_buf[j];
                if (newDist < row_dist[j]) {
                    row_dist[j] = newDist;
                    row_path[j] = (unsigned int)k;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG,
               MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather distance matrix to rank 0 for validation/output
    std::vector<unsigned int> dist;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
    }
    MPI_Gatherv(local_dist.data(), (int)(local_rows * numNodes), MPI_UNSIGNED,
                rank == 0 ? dist.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);

        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
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
