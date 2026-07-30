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

// Original column-major index (for serial-compatible validation/results)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Compute row distribution: how many rows process p owns
inline size_t rowsOnProc(const size_t numNodes, const int p, const int size) {
    return (numNodes / size) + (static_cast<size_t>(p) < numNodes % size ? 1 : 0);
}

// Compute starting global row index for process p
inline size_t startRowForProc(const size_t numNodes, const int p, const int size) {
    size_t base = numNodes / size;
    size_t remainder = numNodes % size;
    if (static_cast<size_t>(p) < remainder)
        return static_cast<size_t>(p) * (base + 1);
    else
        return remainder * (base + 1) + static_cast<size_t>(p - remainder) * base;
}

// Initialize full distance matrix (same as original for reproducibility)
// Uses column-major layout: dist[j*n + i] = dist[i][j]
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// Initialize full path matrix (same as original)
void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

// Serial Floyd-Warshall (kept for reference; not used in MPI path)
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    for (size_t k = 0; k < numNodes; ++k) {
        for (size_t i = 0; i < numNodes; ++i) {
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                const unsigned int newDist = distIK + distKJ;

                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k;
                }
            }
        }
    }
}

// Distributed Floyd-Warshall with 1D block-row decomposition.
// Local storage uses ROW-MAJOR layout for cache-friendly inner loops:
//   local_dist[li * N + j] = logical dist[globalRow][j]
//   local_path[li * N + j] = logical path[globalRow][j]
// This ensures the innermost j-loop accesses contiguous memory.
void floydWarshallDistributed(std::vector<unsigned int>& local_dist,
                              std::vector<unsigned int>& local_path,
                              const std::vector<unsigned int>& full_dist_init,
                              const size_t numNodes, const size_t localRows,
                              const size_t startRow, const int rank, const int size) {
    // Initialize local matrices from the full initialization
    // Local row-major: local[li * N + j] = full[idx2(startRow+li, j, N)] = full[j*N + startRow+li]
    local_dist.resize(localRows * numNodes);
    local_path.resize(localRows * numNodes);

    for (size_t li = 0; li < localRows; ++li) {
        size_t gi = startRow + li;
        for (size_t j = 0; j < numNodes; ++j) {
            local_dist[li * numNodes + j] = full_dist_init[idx2(gi, j, numNodes)];
            // Path matrix: path[i][j] = j initially (except path[j][j] = j, which is same)
            // From original: path[idx2(i,j,N)] = j and path[idx2(j,i,N)] = i
            // The second assignment overwrites when j < i. Net effect:
            // path[i][j] = j if i <= j, else i (but actually both are set in the loop)
            // Let me trace: for a given (j_outer, i_inner):
            //   path[idx2(i_inner, j_outer, N)] = j_outer  => path[i_inner][j_outer] = j_outer
            //   path[idx2(j_outer, i_inner, N)] = i_inner  => path[j_outer][i_inner] = i_inner
            // Then path[idx2(j_outer, j_outer, N)] = j_outer => path[j_outer][j_outer] = j_outer
            // So path[i][j] = j for all i,j (the second assignment sets path[j_outer][i_inner] = i_inner
            // which is path[i][j] = i when we rename j_outer->i, i_inner->j, but that's a different cell)
            // Actually: the loop is "for j: for i: path[i][j]=j; path[j][i]=i"
            // For cell path[r][c]: it gets set to c (when j=c, i=r) and also to r (when j=r, i=c)
            // The second happens when j=r which is later in the outer loop if r > c, earlier if r < c
            // Net: path[r][c] = c if c > r (last write was j=c), path[r][c] = r if r > c (last write was j=r)
            // path[r][r] = r
            // So path[i][j] = max(i, j)
            local_path[li * numNodes + j] = static_cast<unsigned int>(j);
        }
    }

    // Pivot row buffer
    std::vector<unsigned int> pivot_row(numNodes);

    // Precompute owner and local index for each k
    std::vector<int> ownerOf(numNodes);
    std::vector<size_t> localIdxOf(numNodes);
    {
        size_t offset = 0;
        for (int p = 0; p < size; ++p) {
            size_t rows = rowsOnProc(numNodes, p, size);
            for (size_t r = 0; r < rows; ++r) {
                size_t globalK = offset + r;
                ownerOf[globalK] = p;
                localIdxOf[globalK] = r;
            }
            offset += rows;
        }
    }

    // Main FW loop
    for (size_t k = 0; k < numNodes; ++k) {
        int owner = ownerOf[k];

        // Owner prepares pivot row (row k of dist matrix)
        if (rank == owner) {
            size_t lk = localIdxOf[k];
            const unsigned int* row_ptr = &local_dist[lk * numNodes];
            std::memcpy(pivot_row.data(), row_ptr, numNodes * sizeof(unsigned int));
        }

        // Broadcast pivot row to all processes
        MPI_Bcast(pivot_row.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Update local rows
        unsigned int* dist_ptr = local_dist.data();
        unsigned int* path_ptr = local_path.data();
        const size_t N = numNodes;

        for (size_t li = 0; li < localRows; ++li) {
            const unsigned int distIK = dist_ptr[li * N + k];
            unsigned int* row_dist = &dist_ptr[li * N];
            unsigned int* row_path = &path_ptr[li * N];

            for (size_t j = 0; j < N; ++j) {
                const unsigned int newDist = distIK + pivot_row[j];
                if (newDist < row_dist[j]) {
                    row_dist[j] = newDist;
                    row_path[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

// Gather distributed row-major local matrices back to full column-major matrices on rank 0
void gatherResults(const std::vector<unsigned int>& local_dist,
                   const std::vector<unsigned int>& local_path,
                   std::vector<unsigned int>& full_dist,
                   std::vector<unsigned int>& full_path,
                   const size_t numNodes, const size_t localRows,
                   const size_t /*startRow*/, const int rank, const int size) {
    if (rank == 0) {
        full_dist.resize(numNodes * numNodes);
        full_path.resize(numNodes * numNodes);
    }

    // Each process sends its local data to rank 0
    // We use MPI_Gatherv with counts/displacements
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for (int p = 0; p < size; ++p) {
        recvcounts[p] = static_cast<int>(rowsOnProc(numNodes, p, size) * numNodes);
        displs[p] = static_cast<int>(startRowForProc(numNodes, p, size) * numNodes);
    }

    // Gather dist: local is row-major [localRows * N], full is column-major [j*N + i]
    // We need to transpose during gather. Easiest: gather to a temporary row-major buffer,
    // then convert. Or use MPI_Gatherv directly into the right positions.
    // Since full column-major: full[j*N + i], row i occupies positions j*N+i for j=0..N-1 (strided).
    // This is hard to do with a single Gatherv. Instead, gather row-major then convert.

    std::vector<unsigned int> full_dist_rowmajor;
    std::vector<unsigned int> full_path_rowmajor;
    if (rank == 0) {
        full_dist_rowmajor.resize(numNodes * numNodes);
        full_path_rowmajor.resize(numNodes * numNodes);
    }

    MPI_Gatherv(local_dist.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                full_dist_rowmajor.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(local_path.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                full_path_rowmajor.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    // Convert row-major to column-major on rank 0
    if (rank == 0) {
        for (size_t i = 0; i < numNodes; ++i) {
            for (size_t j = 0; j < numNodes; ++j) {
                full_dist[idx2(i, j, numNodes)] = full_dist_rowmajor[i * numNodes + j];
                full_path[idx2(i, j, numNodes)] = full_path_rowmajor[i * numNodes + j];
            }
        }
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

    // 2. Triangle inequality check (sampled)
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse identically)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute local decomposition
    size_t localRows = rowsOnProc(numNodes, rank, size);
    size_t startRow = startRowForProc(numNodes, rank, size);

    // Initialize: each process generates the full initial distance matrix
    // (to maintain identical random sequence), then extracts its local rows.
    if (rank == 0) printf("Initializing graph...\n");

    std::vector<unsigned int> full_dist_init;
    {
        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE - 1) + 1.0;
        full_dist_init.resize(numNodes * numNodes);
        for (size_t i = 0; i < numNodes * numNodes; ++i) {
            full_dist_init[i] = 1 + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        }
        for (size_t i = 0; i < numNodes; ++i) {
            full_dist_init[idx2(i, i, numNodes)] = 0;
        }
    }

    // Distributed FW (initializes local matrices internally)
    std::vector<unsigned int> local_dist, local_path;

    if (rank == 0) printf("Computing shortest paths...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallDistributed(local_dist, local_path, full_dist_init,
                             numNodes, localRows, startRow, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Compute max duration across all processes
    long long local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_ms);
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (max_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather results to rank 0 for validation and output
    std::vector<unsigned int> full_dist, full_path;
    if (printResults || validate) {
        gatherResults(local_dist, local_path, full_dist, full_path,
                      numNodes, localRows, startRow, rank, size);
    }

    if (printResults) {
        if (rank == 0) {
            print_results_int(full_dist, "DistanceMatrix");
        }
    }

    if (validate) {
        int valid_exit = 0;
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(full_dist, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
                valid_exit = 0;
            } else {
                printf("Validation: FAILED\n");
                valid_exit = 1;
            }
        }
        // Broadcast validation result to all processes
        MPI_Bcast(&valid_exit, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return valid_exit;
    }

    MPI_Finalize();
    return 0;
}
