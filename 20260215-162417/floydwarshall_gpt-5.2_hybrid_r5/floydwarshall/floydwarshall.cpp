#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array (column-major: index = col*n + row)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

static inline void cuda_check(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
        std::fflush(stderr);
        std::exit(2);
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

// ------------------------- Initialization (rank 0) -------------------------

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

// ------------------------- MPI row distribution helpers -------------------------

static inline void row_range_for_rank(const int r, const int world, const size_t n,
                                      size_t& startRow, size_t& numRows) {
    const size_t base = n / (size_t)world;
    const size_t rem = n % (size_t)world;
    numRows = base + ((size_t)r < rem ? 1 : 0);
    startRow = base * (size_t)r + std::min((size_t)r, rem);
}

// ------------------------- CUDA kernel (per-k update) -------------------------

static constexpr int TX = 32;
static constexpr int TY = 8;

__global__ void fw_update_k(unsigned int* __restrict__ dist,
                            unsigned int* __restrict__ path,
                            const unsigned int* __restrict__ rowK,
                            int N, int k, int localRows) {
    __shared__ unsigned int colK[TY];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = (int)blockIdx.y * TY + ty;
    const int col = (int)blockIdx.x * TX + tx;

    if (tx == 0) {
        colK[ty] = (row < localRows) ? dist[row * N + k] : 0;
    }
    __syncthreads();

    if (row < localRows && col < N) {
        const unsigned int cand = rowK[col] + colK[ty];
        const int idx = row * N + col;
        const unsigned int cur = dist[idx];
        if (cand < cur) {
            dist[idx] = cand;
            path[idx] = (unsigned int)k;
        }
    }
}

// ------------------------- Main -------------------------

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    // Unconditionally require a CUDA device.
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found; this benchmark requires CUDA.\n");
        }
        MPI_Finalize();
        return 2;
    }
    CUDA_CHECK(cudaSetDevice(rank % devCount));

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", world);
        printf("OpenMP threads (rank 0): %d\n", omp_get_max_threads());
        printf("CUDA devices visible: %d\n", devCount);
    }

    // Determine local row range (destination indices) for this rank.
    size_t rowStart = 0, localRows = 0;
    row_range_for_rank(rank, world, numNodes, rowStart, localRows);

    // Precompute row ranges for all ranks and owner-of-row mapping.
    std::vector<size_t> allRowStarts(world), allRowCounts(world);
    for (int r = 0; r < world; ++r) {
        row_range_for_rank(r, world, numNodes, allRowStarts[r], allRowCounts[r]);
    }

    std::vector<int> rowOwner(numNodes);
    for (int r = 0; r < world; ++r) {
        const size_t rs = allRowStarts[r];
        const size_t re = rs + allRowCounts[r];
        for (size_t i = rs; i < re; ++i) {
            rowOwner[i] = r;
        }
    }

    // Allocate local matrices in row-major (localRows x numNodes)
    std::vector<unsigned int> dist_local(localRows * numNodes);
    std::vector<unsigned int> path_local(localRows * numNodes);

    // Rank 0 initializes global column-major matrices, then packs/scatters.
    std::vector<unsigned int> dist_global;
    std::vector<unsigned int> path_global;
    std::vector<unsigned int> sendPackedDist;
    std::vector<unsigned int> sendPackedPath;

    std::vector<int> scatCounts(world), scatDispls(world);
    if (rank == 0) {
        dist_global.resize(numNodes * numNodes);
        path_global.resize(numNodes * numNodes);

        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist_global, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path_global, numNodes);

        // Prepare Scatterv counts/displs (in elements)
        int disp = 0;
        for (int r = 0; r < world; ++r) {
            const size_t cnt = allRowCounts[r] * numNodes;
            scatCounts[r] = (int)cnt;
            scatDispls[r] = disp;
            disp += (int)cnt;
        }

        sendPackedDist.resize((size_t)disp);
        sendPackedPath.resize((size_t)disp);

        // Pack from global column-major into per-rank row-major buffers.
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < world; ++r) {
            const size_t rs = allRowStarts[r];
            const size_t rn = allRowCounts[r];
            const size_t outBase = (size_t)scatDispls[r];

            for (size_t rr = 0; rr < rn; ++rr) {
                const size_t gRow = rs + rr;
                unsigned int* outD = &sendPackedDist[outBase + rr * numNodes];
                unsigned int* outP = &sendPackedPath[outBase + rr * numNodes];

                for (size_t c = 0; c < numNodes; ++c) {
                    outD[c] = dist_global[idx2(gRow, c, numNodes)];
                    outP[c] = path_global[idx2(gRow, c, numNodes)];
                }
            }
        }
    } else {
        std::fill(scatCounts.begin(), scatCounts.end(), 0);
        std::fill(scatDispls.begin(), scatDispls.end(), 0);
    }

    MPI_Scatterv(rank == 0 ? sendPackedDist.data() : nullptr,
                 scatCounts.data(), scatDispls.data(), MPI_UNSIGNED,
                 dist_local.data(), (int)(localRows * numNodes), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? sendPackedPath.data() : nullptr,
                 scatCounts.data(), scatDispls.data(), MPI_UNSIGNED,
                 path_local.data(), (int)(localRows * numNodes), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Allocate device buffers.
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, dist_local.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_path, path_local.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_dist, dist_local.data(), dist_local.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path_local.data(), path_local.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Device + pinned host buffers for the broadcast row k.
    unsigned int* d_rowK = nullptr;
    unsigned int* h_rowK = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rowK, numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaHostAlloc(&h_rowK, numNodes * sizeof(unsigned int), cudaHostAllocDefault));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    const double t0 = MPI_Wtime();

    for (int k = 0; k < (int)numNodes; ++k) {
        const int owner = rowOwner[(size_t)k];

        if (rank == owner) {
            const size_t localK = (size_t)k - rowStart;
            CUDA_CHECK(cudaMemcpy(h_rowK, d_dist + localK * numNodes,
                                  numNodes * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }

        MPI_Bcast(h_rowK, (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(d_rowK, h_rowK, numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice));

        if (localRows > 0) {
            dim3 block(TX, TY);
            dim3 grid((unsigned int)((numNodes + TX - 1) / TX),
                      (unsigned int)((localRows + TY - 1) / TY));
            fw_update_k<<<grid, block>>>(d_dist, d_path, d_rowK, (int)numNodes, k, (int)localRows);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double localTime = t1 - t0;

    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = (long)llround(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double ops = (double)numNodes * (double)numNodes * (double)numNodes;
        const double gops = ops / maxTime / 1e9;
        printf("Performance: %.3f GOPS\n", gops);
    }

    // Gather dist back to rank 0 for printing/validation (row-major segments).
    CUDA_CHECK(cudaMemcpy(dist_local.data(), d_dist, dist_local.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<int> gathCounts(world), gathDispls(world);
    std::vector<unsigned int> recvPacked;
    if (rank == 0) {
        int disp = 0;
        for (int r = 0; r < world; ++r) {
            const size_t cnt = allRowCounts[r] * numNodes;
            gathCounts[r] = (int)cnt;
            gathDispls[r] = disp;
            disp += (int)cnt;
        }
        recvPacked.resize((size_t)disp);
    }

    MPI_Gatherv(dist_local.data(), (int)(localRows * numNodes), MPI_UNSIGNED,
                rank == 0 ? recvPacked.data() : nullptr,
                gathCounts.data(), gathDispls.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Unpack back to global column-major for identical output semantics.
        dist_global.assign(numNodes * numNodes, 0);
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < world; ++r) {
            const size_t rs = allRowStarts[r];
            const size_t rn = allRowCounts[r];
            const size_t inBase = (size_t)gathDispls[r];

            for (size_t rr = 0; rr < rn; ++rr) {
                const size_t gRow = rs + rr;
                const unsigned int* inD = &recvPacked[inBase + rr * numNodes];
                for (size_t c = 0; c < numNodes; ++c) {
                    dist_global[idx2(gRow, c, numNodes)] = inD[c];
                }
            }
        }

        if (printResults) {
            print_results_int(dist_global, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool ok = validateResult(dist_global, numNodes);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            if (!ok) {
                CUDA_CHECK(cudaFreeHost(h_rowK));
                CUDA_CHECK(cudaFree(d_rowK));
                CUDA_CHECK(cudaFree(d_dist));
                CUDA_CHECK(cudaFree(d_path));
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_rowK));
    CUDA_CHECK(cudaFree(d_rowK));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));

    MPI_Finalize();
    return 0;
}
