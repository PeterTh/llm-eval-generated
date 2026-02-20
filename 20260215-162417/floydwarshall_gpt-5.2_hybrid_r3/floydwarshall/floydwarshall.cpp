#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err == cudaSuccess) return;
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

#define CUDA_CHECK(x) cudaCheck((x), __FILE__, __LINE__)

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

static inline void rowPartition(const size_t n, const int worldSize, const int rank,
                                size_t& rowStart, size_t& rowCount) {
    const size_t ws = static_cast<size_t>(worldSize);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = n / ws;
    const size_t rem = n % ws;
    rowCount = base + ((r < rem) ? 1 : 0);
    rowStart = base * r + std::min(r, rem);
}

static inline int ownerOfRow(const size_t n, const int worldSize, const size_t k) {
    const size_t ws = static_cast<size_t>(worldSize);
    const size_t base = n / ws;
    const size_t rem = n % ws;
    if (base == 0) return static_cast<int>(k); // first n ranks own 1 row
    const size_t cut = (base + 1) * rem;
    if (k < cut) return static_cast<int>(k / (base + 1));
    return static_cast<int>(rem + (k - cut) / base);
}

__global__ void fw_update_kernel(unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ path,
                                const unsigned int* __restrict__ rowk,
                                int n,
                                int localRows,
                                int k) {
    const int j = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
    const int i = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) +
                  static_cast<int>(threadIdx.y);
    if (i >= localRows || j >= n) return;

    const unsigned int dik = dist[i * n + k];
    const unsigned int dkj = rowk[j];
    const unsigned int old = dist[i * n + j];
    const unsigned int nd = dik + dkj;

    if (nd < old) {
        dist[i * n + j] = nd;
        path[i * n + j] = static_cast<unsigned int>(k);
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

    // 2. Triangle inequality (sampled)
    const size_t limit = std::min(numNodes, static_cast<size_t>(10));
    int ok = 1;
#pragma omp parallel for collapse(2) reduction(&:ok)
    for (size_t i = 0; i < limit; ++i) {
        for (size_t j = 0; j < limit; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) ok = 0;
                }
            }
        }
    }

    if (!ok) {
        printf("Validation failed: triangle inequality violated in sample\n");
        return false;
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

struct Options {
    size_t numNodes = 512;
    int validate = 0;
    int printResults = 0;
    int ok = 1;
    int exitCode = 0;
};

static Options parseOptionsRoot(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            opt.numNodes = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            opt.validate = 1;
        } else if (strcmp(argv[i], "-r") == 0) {
            opt.printResults = 1;
        } else if (strcmp(argv[i], "-h") == 0) {
            opt.ok = 0;
            opt.exitCode = 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            opt.ok = 0;
            opt.exitCode = 1;
        }
    }
    return opt;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Force OpenMP runtime initialization unconditionally.
    int omp_dummy = 0;
#pragma omp parallel for reduction(+:omp_dummy)
    for (int i = 0; i < 128; ++i) omp_dummy += i;
    if (omp_dummy < 0) printf("%d\n", omp_dummy);

    Options opt;
    if (rank == 0) {
        opt = parseOptionsRoot(argc, argv);
        if (!opt.ok) {
            printUsage(argv[0]);
        }
    }

    MPI_Bcast(&opt.ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&opt.exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!opt.ok) {
        MPI_Finalize();
        return opt.exitCode;
    }

    // Broadcast options
    unsigned long long n64 = (rank == 0) ? static_cast<unsigned long long>(opt.numNodes) : 0ULL;
    MPI_Bcast(&n64, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    opt.numNodes = static_cast<size_t>(n64);
    MPI_Bcast(&opt.validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&opt.printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const size_t numNodes = opt.numNodes;

    // Choose GPU device per rank (simple round-robin).
    int deviceCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", opt.validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", worldSize);
    }

    // Compute row decomposition and scatter counts/displacements.
    size_t rowStart = 0, rowCount = 0;
    rowPartition(numNodes, worldSize, rank, rowStart, rowCount);

    std::vector<int> sendCounts(worldSize, 0), displs(worldSize, 0);
#pragma omp parallel for
    for (int r = 0; r < worldSize; ++r) {
        size_t rs = 0, rc = 0;
        rowPartition(numNodes, worldSize, r, rs, rc);
        const size_t elems = rc * numNodes;
        sendCounts[r] = static_cast<int>(elems);
        displs[r] = static_cast<int>(rs * numNodes);
    }

    const size_t localElemsSz = rowCount * numNodes;
    const int localElems = static_cast<int>(localElemsSz);

    std::vector<unsigned int> dist_h(localElemsSz);
    std::vector<unsigned int> path_h(localElemsSz);

    std::vector<unsigned int> dist_full;
    std::vector<unsigned int> path_full;

    if (rank == 0) {
        dist_full.resize(numNodes * numNodes);
        path_full.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist_full, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path_full, numNodes);
    }

    MPI_Scatterv(rank == 0 ? dist_full.data() : nullptr, sendCounts.data(), displs.data(),
                 MPI_UNSIGNED, dist_h.data(), localElems, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? path_full.data() : nullptr, sendCounts.data(), displs.data(),
                 MPI_UNSIGNED, path_h.data(), localElems, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Device allocations
    unsigned int* dist_d = nullptr;
    unsigned int* path_d = nullptr;
    CUDA_CHECK(cudaMalloc(&dist_d, localElemsSz * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&path_d, localElemsSz * sizeof(unsigned int)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMemcpyAsync(dist_d, dist_h.data(), localElemsSz * sizeof(unsigned int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(path_d, path_h.data(), localElemsSz * sizeof(unsigned int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Row buffer for MPI broadcast
    unsigned int* rowk_h = nullptr;
    CUDA_CHECK(cudaMallocHost(&rowk_h, numNodes * sizeof(unsigned int)));
    unsigned int* rowk_d = nullptr;
    CUDA_CHECK(cudaMalloc(&rowk_d, numNodes * sizeof(unsigned int)));

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    const dim3 block(32, 8);
    const unsigned int gridX = static_cast<unsigned int>((numNodes + block.x - 1) / block.x);
    const unsigned int gridY = std::max(1U, static_cast<unsigned int>((rowCount + block.y - 1) / block.y));
    const dim3 grid(gridX, gridY);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow(numNodes, worldSize, k);
        const size_t bytes = numNodes * sizeof(unsigned int);

        const unsigned int* rowk_dev = rowk_d;
        if (rank == owner) {
            const size_t localK = k - rowStart;
            rowk_dev = dist_d + localK * numNodes;
            CUDA_CHECK(cudaMemcpyAsync(rowk_h, rowk_dev, bytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        MPI_Bcast(rowk_h, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        if (rank != owner) {
            CUDA_CHECK(cudaMemcpyAsync(rowk_d, rowk_h, bytes, cudaMemcpyHostToDevice, stream));
        }

        fw_update_kernel<<<grid, block, 0, stream>>>(dist_d, path_d, rowk_dev,
                                                     static_cast<int>(numNodes),
                                                     static_cast<int>(rowCount),
                                                     static_cast<int>(k));
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    double maxTime = 0.0;
    const double localTime = t1 - t0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double ops = static_cast<double>(numNodes) * static_cast<double>(numNodes) *
                           static_cast<double>(numNodes);
        const double gops = ops / maxTime / 1e9;
        printf("Performance: %.3f GOPS\n", gops);
    }

    const int needResults = (opt.validate || opt.printResults) ? 1 : 0;

    if (needResults) {
        CUDA_CHECK(cudaMemcpyAsync(dist_h.data(), dist_d, localElemsSz * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (rank == 0) dist_full.resize(numNodes * numNodes);
        MPI_Gatherv(dist_h.data(), localElems, MPI_UNSIGNED,
                    rank == 0 ? dist_full.data() : nullptr,
                    sendCounts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        if (rank == 0 && opt.printResults) {
            print_results_int(dist_full, "DistanceMatrix");
        }

        if (rank == 0 && opt.validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(dist_full, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            CUDA_CHECK(cudaFreeHost(rowk_h));
            CUDA_CHECK(cudaFree(rowk_d));
            CUDA_CHECK(cudaFree(dist_d));
            CUDA_CHECK(cudaFree(path_d));
            CUDA_CHECK(cudaStreamDestroy(stream));
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    CUDA_CHECK(cudaFreeHost(rowk_h));
    CUDA_CHECK(cudaFree(rowk_d));
    CUDA_CHECK(cudaFree(dist_d));
    CUDA_CHECK(cudaFree(path_d));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Finalize();
    return 0;
}
