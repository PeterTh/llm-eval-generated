#include <algorithm>
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

struct RowDecomp {
    size_t start_row;
    size_t local_rows;
};

static inline RowDecomp decompose_rows(const size_t nrows, const int rank, const int size) {
    const size_t p = static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = nrows / p;
    const size_t rem = nrows % p;
    const size_t local = base + (r < rem ? 1 : 0);
    const size_t start = r * base + (r < rem ? r : rem);
    return {start, local};
}

static inline int owner_of_row(const size_t row, const size_t nrows, const int size) {
    const size_t p = static_cast<size_t>(size);
    const size_t base = nrows / p;
    const size_t rem = nrows % p;
    const size_t cutoff = (base + 1) * rem;
    if (row < cutoff) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(rem + (row - cutoff) / base);
}

// Fast jump-ahead for the rand_r() seed state assuming glibc/POSIX LCG update.
// glibc rand_r advances the seed by 3 LCG steps (a*s + c) per call.
static inline unsigned int advance_rand_r_seed(unsigned int seed, uint64_t calls) {
    constexpr uint32_t a = 1103515245u;
    constexpr uint32_t c = 12345u;

    // Effective transition per rand_r call: s' = A*s + C (mod 2^32)
    const uint32_t A = static_cast<uint32_t>(static_cast<uint64_t>(a) * a % 0x100000000ULL * a % 0x100000000ULL);
    const uint32_t C = static_cast<uint32_t>((static_cast<uint64_t>(c) * ((static_cast<uint64_t>(a) * a + a + 1u) & 0xFFFFFFFFu)) & 0xFFFFFFFFu);

    uint32_t mul = 1u;
    uint32_t add = 0u;
    uint32_t bmul = A;
    uint32_t badd = C;

    while (calls) {
        if (calls & 1u) {
            // (bmul, badd) ∘ (mul, add)
            add = static_cast<uint32_t>(static_cast<uint64_t>(bmul) * add + badd);
            mul = static_cast<uint32_t>(static_cast<uint64_t>(bmul) * mul);
        }
        // square base: (bmul, badd) ∘ (bmul, badd)
        badd = static_cast<uint32_t>(static_cast<uint64_t>(bmul) * badd + badd);
        bmul = static_cast<uint32_t>(static_cast<uint64_t>(bmul) * bmul);
        calls >>= 1u;
    }

    return static_cast<uint32_t>(static_cast<uint64_t>(mul) * seed + add);
}

static inline void initialize_local_matrices(std::vector<unsigned int>& dist_local,
                                             std::vector<unsigned int>& path_local,
                                             const size_t numNodes,
                                             const RowDecomp& decomp) {
    const unsigned int rangeMin = 1;
    const unsigned int rangeMax = MAX_DISTANCE;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const uint64_t start_index = static_cast<uint64_t>(decomp.start_row) * static_cast<uint64_t>(numNodes);
    unsigned int seed = advance_rand_r_seed(42u, start_index);

    const size_t local_elems = decomp.local_rows * numNodes;
    for (size_t t = 0; t < local_elems; ++t) {
        dist_local[t] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }

    // Set local diagonal elements to 0
    for (size_t li = 0; li < decomp.local_rows; ++li) {
        const size_t gi = decomp.start_row + li;
        dist_local[li * numNodes + gi] = 0;
    }

    // Path init: predecessor of (i -> j) starts as i (matches original init for this layout)
    for (size_t li = 0; li < decomp.local_rows; ++li) {
        const size_t gi = decomp.start_row + li;
        unsigned int* prow = path_local.data() + li * numNodes;
        std::fill(prow, prow + numNodes, static_cast<unsigned int>(gi));
        prow[gi] = static_cast<unsigned int>(gi);
    }
}

static inline void floyd_warshall_mpi(std::vector<unsigned int>& dist_local,
                                     std::vector<unsigned int>& path_local,
                                     const size_t numNodes,
                                     const RowDecomp& decomp,
                                     const int rank,
                                     const int size) {
    std::vector<unsigned int> row_k(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = owner_of_row(k, numNodes, size);
        if (rank == owner) {
            const size_t lk = k - decomp.start_row;
            const unsigned int* src = dist_local.data() + lk * numNodes;
            std::memcpy(row_k.data(), src, numNodes * sizeof(unsigned int));
        }

        MPI_Bcast(row_k.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t li = 0; li < decomp.local_rows; ++li) {
            unsigned int* __restrict__ row_i = dist_local.data() + li * numNodes;
            unsigned int* __restrict__ path_i = path_local.data() + li * numNodes;

            const unsigned int ik = row_i[k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int cand = ik + row_k[j];
                const unsigned int dij = row_i[j];
                if (cand < dij) {
                    row_i[j] = cand;
                    path_i[j] = static_cast<unsigned int>(k);
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint64_t numNodes_u64 = 512;
    int validate_i = 0;
    int printResults_i = 0;
    int showHelp_i = 0;
    int parseOk_i = 1;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes_u64 = static_cast<uint64_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_i = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_i = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp_i = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseOk_i = 0;
                break;
            }
        }
    }

    MPI_Bcast(&parseOk_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parseOk_i) {
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(&showHelp_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (showHelp_i) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    MPI_Bcast(&numNodes_u64, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_i, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const size_t numNodes = static_cast<size_t>(numNodes_u64);
    const bool validate = (validate_i != 0);
    const bool printResults = (printResults_i != 0);

    const RowDecomp decomp = decompose_rows(numNodes, rank, size);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate local matrices (row-block distribution)
    std::vector<unsigned int> dist_local(decomp.local_rows * numNodes);
    std::vector<unsigned int> path_local(decomp.local_rows * numNodes);

    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initialize_local_matrices(dist_local, path_local, numNodes, decomp);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    floyd_warshall_mpi(dist_local, path_local, numNodes, decomp, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double local_s = t1 - t0;

    double max_s = 0.0;
    MPI_Reduce(&local_s, &max_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int exitCode = 0;

    if (rank == 0) {
        const long ms = static_cast<long>(max_s * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        const double ops = static_cast<double>(numNodes) * static_cast<double>(numNodes) * static_cast<double>(numNodes);
        const double gops = ops / max_s / 1e9;
        printf("Performance: %.3f GOPS\n", gops);
    }

    // Gather for external results printing / validation to preserve identical behavior
    std::vector<unsigned int> dist_global;
    std::vector<int> recvcounts;
    std::vector<int> displs;

    if (printResults || validate) {
        if (rank == 0) {
            dist_global.resize(numNodes * numNodes);
            recvcounts.resize(static_cast<size_t>(size));
            displs.resize(static_cast<size_t>(size));
            for (int r = 0; r < size; ++r) {
                const RowDecomp d = decompose_rows(numNodes, r, size);
                const size_t cnt = d.local_rows * numNodes;
                recvcounts[static_cast<size_t>(r)] = static_cast<int>(cnt);
                displs[static_cast<size_t>(r)] = static_cast<int>(d.start_row * numNodes);
            }
        }

        unsigned int* recvbuf = (rank == 0) ? dist_global.data() : nullptr;
        int* rc = (rank == 0) ? recvcounts.data() : nullptr;
        int* dsp = (rank == 0) ? displs.data() : nullptr;

        MPI_Gatherv(dist_local.data(), static_cast<int>(decomp.local_rows * numNodes), MPI_UNSIGNED,
                    recvbuf, rc, dsp, MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);

        if (rank == 0 && printResults) {
            print_results_int(dist_global, "DistanceMatrix");
        }

        if (rank == 0 && validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(dist_global, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
