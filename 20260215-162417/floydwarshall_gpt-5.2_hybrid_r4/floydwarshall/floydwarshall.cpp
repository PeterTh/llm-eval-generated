#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array (column-major by source index)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

static inline void compute_row_partition(const size_t n, const int size, const int rank,
                                        size_t& startRow, size_t& localRows) {
    const size_t p = static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = n / p;
    const size_t rem = n % p;

    if (r < rem) {
        localRows = base + 1;
        startRow = r * (base + 1);
    } else {
        localRows = base;
        startRow = rem * (base + 1) + (r - rem) * base;
    }
}

static inline int owner_of_row(const size_t k, const size_t n, const int size) {
    const size_t p = static_cast<size_t>(size);
    const size_t base = n / p;
    const size_t rem = n % p;

    if (base == 0) {
        return static_cast<int>(k); // first n ranks own one row each
    }

    const size_t first_block_rows = (base + 1) * rem;
    if (k < first_block_rows) {
        return static_cast<int>(k / (base + 1));
    }
    return static_cast<int>(rem + (k - first_block_rows) / base);
}

static inline void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                                           const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // Keep this loop serial to preserve deterministic rand_r sequence.
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

static inline void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = static_cast<unsigned int>(j);
            path[idx2(j, i, numNodes)] = static_cast<unsigned int>(i);
        }
        path[idx2(j, j, numNodes)] = static_cast<unsigned int>(j);
    }
}

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t _e = (call);                                                         \
        if (_e != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

constexpr int CUDA_TILE = 256;

__global__ void fw_update_kernel(unsigned int* __restrict__ dist,
                                 unsigned int* __restrict__ path,
                                 const unsigned int* __restrict__ row_k,
                                 int n, int localRows, int k) {
    const int i = static_cast<int>(blockIdx.x) * CUDA_TILE + static_cast<int>(threadIdx.x);
    const int r = static_cast<int>(blockIdx.y);

    __shared__ unsigned int djk;
    __shared__ unsigned int srow[CUDA_TILE];

    if (threadIdx.x == 0) {
        djk = dist[r * n + k];
    }
    if (i < n) {
        srow[threadIdx.x] = row_k[i];
    }
    __syncthreads();

    if (i < n && r < localRows) {
        const unsigned int oldVal = dist[r * n + i];
        const unsigned int newVal = srow[threadIdx.x] + djk;
        if (newVal < oldVal) {
            dist[r * n + i] = newVal;
            path[r * n + i] = static_cast<unsigned int>(k);
        }
    }
}

static inline bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks

    // 1. Diagonal should be zero
    int diag_ok = 1;
#pragma omp parallel for reduction(&&:diag_ok)
    for (size_t i = 0; i < numNodes; ++i) {
        diag_ok = diag_ok && (dist[idx2(i, i, numNodes)] == 0);
    }
    if (!diag_ok) {
        for (size_t i = 0; i < numNodes; ++i) {
            if (dist[idx2(i, i, numNodes)] != 0) {
                printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
                break;
            }
        }
        return false;
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

static inline void printUsage(const char* progName) {
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

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(atoi(argv[++i]));
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

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    // Partition rows (destination index j) across MPI ranks
    size_t startRow = 0;
    size_t localRows = 0;
    compute_row_partition(numNodes, size, rank, startRow, localRows);

    const size_t localElems = localRows * numNodes;

    // Prepare counts/displacements for scatter/gather (in elements)
    std::vector<int> counts(size, 0), displs(size, 0);
    for (int r = 0; r < size; ++r) {
        size_t s = 0, lr = 0;
        compute_row_partition(numNodes, size, r, s, lr);
        const size_t elems = lr * numNodes;
        if (elems > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) {
                fprintf(stderr, "Problem size too large for MPI counts\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        counts[r] = static_cast<int>(elems);
        displs[r] = static_cast<int>(s * numNodes);
    }

    // Root initializes full matrices exactly like the original code, then scatters rows.
    std::vector<unsigned int> dist_full;
    std::vector<unsigned int> path_full;
    if (rank == 0) {
        dist_full.resize(numNodes * numNodes);
        path_full.resize(numNodes * numNodes);

        // Touch pages in parallel to reduce first-touch overhead (does not affect determinism).
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < dist_full.size(); ++i) {
            dist_full[i] = 0;
            path_full[i] = 0;
        }

        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist_full, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path_full, numNodes);
    }

    std::vector<unsigned int> dist_local(localElems);
    std::vector<unsigned int> path_local(localElems);

    MPI_Scatterv(rank == 0 ? dist_full.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 dist_local.data(), static_cast<int>(localElems), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? path_full.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 path_local.data(), static_cast<int>(localElems), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // CUDA setup: map rank to a GPU on the node.
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_row_k = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dist), localElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_path), localElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_row_k), numNodes * sizeof(unsigned int)));

    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_dist, dist_local.data(), localElems * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_path, path_local.data(), localElems * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    unsigned int* row_host = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&row_host), numNodes * sizeof(unsigned int)));

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int n_i = static_cast<int>(numNodes);
    const int localRows_i = static_cast<int>(localRows);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = owner_of_row(k, numNodes, size);

        // Owner extracts row k from device (after all prior updates) and broadcasts it.
        if (rank == owner) {
            const size_t local_k = k - startRow;
            if (local_k >= localRows) {
                fprintf(stderr, "Internal error: owner does not have row k\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            const unsigned int* d_row_src = d_dist + local_k * numNodes;
            CUDA_CHECK(cudaMemcpyAsync(row_host, d_row_src, numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        MPI_Bcast(row_host, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpyAsync(d_row_k, row_host, numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));

        if (localRows > 0) {
            dim3 block(CUDA_TILE, 1, 1);
            dim3 grid((static_cast<unsigned int>(numNodes) + CUDA_TILE - 1) / CUDA_TILE,
                      static_cast<unsigned int>(localRows), 1);
            fw_update_kernel<<<grid, block, 0, stream>>>(d_dist, d_path, d_row_k, n_i, localRows_i, static_cast<int>(k));
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    const double local_duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration_ms);
        double ops = static_cast<double>(numNodes) * static_cast<double>(numNodes) * static_cast<double>(numNodes);
        double gops = ops / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gops);
    }

    // Copy back local dist for optional gather/output/validation
    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpyAsync(dist_local.data(), d_dist, localElems * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<unsigned int> dist_gathered;
    if (rank == 0 && (printResults || validate)) {
        dist_gathered.resize(numNodes * numNodes);
    }

    if (printResults || validate) {
        MPI_Gatherv(dist_local.data(), static_cast<int>(localElems), MPI_UNSIGNED,
                    rank == 0 ? dist_gathered.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        if (printResults) {
            print_results_int(dist_gathered, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            bool ok = validateResult(dist_gathered, numNodes);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");

            CUDA_CHECK(cudaFreeHost(row_host));
            CUDA_CHECK(cudaFree(d_row_k));
            CUDA_CHECK(cudaFree(d_path));
            CUDA_CHECK(cudaFree(d_dist));
            CUDA_CHECK(cudaStreamDestroy(stream));
            MPI_Finalize();
            return ok ? 0 : 1;
        }
    }

    CUDA_CHECK(cudaFreeHost(row_host));
    CUDA_CHECK(cudaFree(d_row_k));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Finalize();
    return 0;
}
