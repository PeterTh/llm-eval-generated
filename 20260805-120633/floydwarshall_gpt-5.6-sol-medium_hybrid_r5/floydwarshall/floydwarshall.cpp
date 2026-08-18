#include <algorithm>
#include <climits>
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
constexpr int TILE = 32;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call) do {                                                     \
    const cudaError_t error_ = (call);                                             \
    if (error_ != cudaSuccess) {                                                   \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                     cudaGetErrorString(error_));                                  \
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                   \
    }                                                                              \
} while (0)

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t n,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    // Preserve the benchmark's deterministic rand_r sequence exactly.
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < n * n; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(
            range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) dist[idx2(i, i, n)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t row = 0; row < n; ++row) {
        std::fill_n(path.data() + row * n, n, static_cast<unsigned int>(row));
    }
}

__global__ void diagonalKernel(unsigned int* dist, unsigned int* path,
                               int n, int localBase, int k0, int kb) {
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int localRow = k0 - localBase + y;
    const size_t pos = static_cast<size_t>(localRow) * n + k0 + x;
    tile[y][x] = (x < kb && y < kb) ? dist[pos] : INF;
    __syncthreads();

    for (int k = 0; k < kb; ++k) {
        const unsigned int candidate = tile[y][k] + tile[k][x];
        __syncthreads();
        if (x < kb && y < kb && candidate < tile[y][x]) {
            tile[y][x] = candidate;
            path[pos] = static_cast<unsigned int>(k0 + k);
        }
        __syncthreads();
    }
    if (x < kb && y < kb) dist[pos] = tile[y][x];
}

__global__ void pivotRowKernel(unsigned int* dist, unsigned int* path,
                               const unsigned int* diagonal, int n,
                               int localBase, int k0, int kb) {
    __shared__ unsigned int diag[TILE][TILE + 1];
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int col0 = blockIdx.x * TILE;
    if (col0 == k0) return;
    const int localRow = k0 - localBase + y;
    const int col = col0 + x;
    const size_t pos = static_cast<size_t>(localRow) * n + col;
    diag[y][x] = (x < kb && y < kb) ? diagonal[y * TILE + x] : INF;
    tile[y][x] = (y < kb && col < n) ? dist[pos] : INF;
    __syncthreads();

    for (int k = 0; k < kb; ++k) {
        const unsigned int candidate = diag[y][k] + tile[k][x];
        __syncthreads();
        if (y < kb && col < n && candidate < tile[y][x]) {
            tile[y][x] = candidate;
            path[pos] = static_cast<unsigned int>(k0 + k);
        }
        __syncthreads();
    }
    if (y < kb && col < n) dist[pos] = tile[y][x];
}

__global__ void pivotColumnKernel(unsigned int* dist, unsigned int* path,
                                  const unsigned int* diagonal, int n,
                                  int localBase, int localRows, int k0, int kb) {
    __shared__ unsigned int diag[TILE][TILE + 1];
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int localRow = blockIdx.x * TILE + y;
    const int globalRow0 = localBase + blockIdx.x * TILE;
    if (globalRow0 == k0) return;
    const size_t pos = static_cast<size_t>(localRow) * n + k0 + x;
    diag[y][x] = (x < kb && y < kb) ? diagonal[y * TILE + x] : INF;
    tile[y][x] = (localRow < localRows && x < kb) ? dist[pos] : INF;
    __syncthreads();

    for (int k = 0; k < kb; ++k) {
        const unsigned int candidate = tile[y][k] + diag[k][x];
        __syncthreads();
        if (localRow < localRows && x < kb && candidate < tile[y][x]) {
            tile[y][x] = candidate;
            path[pos] = static_cast<unsigned int>(k0 + k);
        }
        __syncthreads();
    }
    if (localRow < localRows && x < kb) dist[pos] = tile[y][x];
}

__global__ void remainderKernel(unsigned int* dist, unsigned int* path,
                                const unsigned int* pivotRows, int n,
                                int localBase, int localRows, int k0, int kb) {
    __shared__ unsigned int column[TILE][TILE + 1];
    __shared__ unsigned int row[TILE][TILE + 1];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int col0 = blockIdx.x * TILE;
    const int localRow0 = blockIdx.y * TILE;
    const int globalRow0 = localBase + localRow0;
    if (col0 == k0 || globalRow0 == k0) return;
    const int col = col0 + x;
    const int localRow = localRow0 + y;

    column[y][x] = (localRow < localRows && x < kb)
        ? dist[static_cast<size_t>(localRow) * n + k0 + x] : INF;
    row[y][x] = (y < kb && col < n)
        ? pivotRows[static_cast<size_t>(y) * n + col] : INF;
    __syncthreads();

    if (localRow < localRows && col < n) {
        const size_t pos = static_cast<size_t>(localRow) * n + col;
        unsigned int best = dist[pos];
        unsigned int bestK = 0;
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            if (k < kb) {
                const unsigned int candidate = column[y][k] + row[k][x];
                if (candidate < best) {
                    best = candidate;
                    bestK = static_cast<unsigned int>(k0 + k);
                }
            }
        }
        if (best < dist[pos]) {
            dist[pos] = best;
            path[pos] = bestK;
        }
    }
}

struct Distribution {
    std::vector<int> rows;
    std::vector<int> rowOffsets;
    std::vector<int> counts;
    std::vector<int> displacements;
};

Distribution makeDistribution(int n, int ranks) {
    Distribution d;
    d.rows.resize(ranks);
    d.rowOffsets.resize(ranks);
    d.counts.resize(ranks);
    d.displacements.resize(ranks);
    const int blockRows = (n + TILE - 1) / TILE;
    const int base = blockRows / ranks;
    const int extra = blockRows % ranks;
    int firstBlock = 0;
    for (int r = 0; r < ranks; ++r) {
        const int blocks = base + (r < extra ? 1 : 0);
        const int firstRow = std::min(n, firstBlock * TILE);
        const int endRow = std::min(n, (firstBlock + blocks) * TILE);
        d.rowOffsets[r] = firstRow;
        d.rows[r] = endRow - firstRow;
        const long long count = static_cast<long long>(d.rows[r]) * n;
        const long long displacement = static_cast<long long>(firstRow) * n;
        if (count > INT_MAX || displacement > INT_MAX) {
            std::fprintf(stderr, "Matrix is too large for MPI Scatterv/Gatherv counts\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        d.counts[r] = static_cast<int>(count);
        d.displacements[r] = static_cast<int>(displacement);
        firstBlock += blocks;
    }
    return d;
}

int ownerOfRow(const Distribution& d, int row) {
    for (int r = 0; r < static_cast<int>(d.rows.size()); ++r) {
        if (row >= d.rowOffsets[r] && row < d.rowOffsets[r] + d.rows[r]) return r;
    }
    return 0;
}

void floydWarshall(std::vector<unsigned int>& localDist,
                   std::vector<unsigned int>& localPath, int n,
                   const Distribution& distribution, int rank) {
    const int localRows = distribution.rows[rank];
    const int localBase = distribution.rowOffsets[rank];
    const size_t localElements = static_cast<size_t>(localRows) * n;
    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    unsigned int *deviceDiagonal = nullptr, *devicePivotRows = nullptr;
    unsigned int *hostDiagonal = nullptr, *hostPivotRows = nullptr;

    CUDA_CHECK(cudaMalloc(&deviceDist, std::max<size_t>(1, localElements) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePath, std::max<size_t>(1, localElements) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&deviceDiagonal, TILE * TILE * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePivotRows, static_cast<size_t>(TILE) * n * sizeof(unsigned int)));
    CUDA_CHECK(cudaHostAlloc(&hostDiagonal, TILE * TILE * sizeof(unsigned int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hostPivotRows, static_cast<size_t>(TILE) * n * sizeof(unsigned int), cudaHostAllocDefault));
    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpy(deviceDist, localDist.data(), localElements * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devicePath, localPath.data(), localElements * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    const dim3 threads(TILE, TILE);
    const int columnTiles = (n + TILE - 1) / TILE;
    const int localRowTiles = (localRows + TILE - 1) / TILE;
    for (int k0 = 0; k0 < n; k0 += TILE) {
        const int kb = std::min(TILE, n - k0);
        const int owner = ownerOfRow(distribution, k0);
        if (rank == owner) {
            diagonalKernel<<<1, threads>>>(deviceDist, devicePath, n, localBase, k0, kb);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy2D(hostDiagonal, TILE * sizeof(unsigned int),
                                    deviceDist + static_cast<size_t>(k0 - localBase) * n + k0,
                                    static_cast<size_t>(n) * sizeof(unsigned int),
                                    kb * sizeof(unsigned int), kb, cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(hostDiagonal, TILE * TILE, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(deviceDiagonal, hostDiagonal, TILE * TILE * sizeof(unsigned int), cudaMemcpyHostToDevice));

        if (rank == owner) {
            pivotRowKernel<<<columnTiles, threads>>>(deviceDist, devicePath, deviceDiagonal,
                                                    n, localBase, k0, kb);
            CUDA_CHECK(cudaGetLastError());
        }
        if (localRowTiles > 0) {
            pivotColumnKernel<<<localRowTiles, threads>>>(deviceDist, devicePath, deviceDiagonal,
                                                         n, localBase, localRows, k0, kb);
            CUDA_CHECK(cudaGetLastError());
        }
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(hostPivotRows,
                                  deviceDist + static_cast<size_t>(k0 - localBase) * n,
                                  static_cast<size_t>(kb) * n * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(hostPivotRows, TILE * n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(devicePivotRows, hostPivotRows,
                              static_cast<size_t>(TILE) * n * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        if (localRowTiles > 0) {
            remainderKernel<<<dim3(columnTiles, localRowTiles), threads>>>(
                deviceDist, devicePath, devicePivotRows, n, localBase, localRows, k0, kb);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpy(localDist.data(), deviceDist, localElements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localPath.data(), devicePath, localElements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFreeHost(hostPivotRows));
    CUDA_CHECK(cudaFreeHost(hostDiagonal));
    CUDA_CHECK(cudaFree(devicePivotRows));
    CUDA_CHECK(cudaFree(deviceDiagonal));
    CUDA_CHECK(cudaFree(devicePath));
    CUDA_CHECK(cudaFree(deviceDist));
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t n) {
    int valid = 1;
#pragma omp parallel for reduction(&:valid) schedule(static)
    for (size_t i = 0; i < n; ++i) valid &= (dist[idx2(i, i, n)] == 0);
    if (!valid) {
        std::printf("Validation failed: a diagonal element is not zero\n");
        return false;
    }
    const size_t sample = std::min(n, static_cast<size_t>(10));
#pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (size_t i = 0; i < sample; ++i) {
        for (size_t j = 0; j < sample; ++j) {
            for (size_t k = 0; k < n; ++k) {
                const unsigned int ij = dist[idx2(j, i, n)];
                const unsigned int ik = dist[idx2(k, i, n)];
                const unsigned int kj = dist[idx2(j, k, n)];
                if (ik < INF && kj < INF && ik + kj < ij) valid = 0;
            }
        }
    }
    if (!valid) std::printf("Validation failed: triangle inequality violated\n");
    return valid != 0;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t numNodes = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (numNodes == 0 || numNodes > static_cast<size_t>(INT_MAX) ||
        numNodes * numNodes > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) std::fprintf(stderr, "-n must be positive and the matrix must fit MPI integer counts\n");
        MPI_Finalize();
        return 1;
    }
    const int n = static_cast<int>(numNodes);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices are available\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\nValidation: %s\n", numNodes,
                    validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), %d OpenMP thread(s), CUDA (32x32 tiles)\n",
                    ranks, omp_get_max_threads());
        std::printf("Initializing graph...\n");
    }

    const Distribution distribution = makeDistribution(n, ranks);
    std::vector<unsigned int> dist, path;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }
    std::vector<unsigned int> localDist(distribution.counts[rank]);
    std::vector<unsigned int> localPath(distribution.counts[rank]);
    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, distribution.counts.data(),
                 distribution.displacements.data(), MPI_UNSIGNED, localDist.data(),
                 distribution.counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? path.data() : nullptr, distribution.counts.data(),
                 distribution.displacements.data(), MPI_UNSIGNED, localPath.data(),
                 distribution.counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) std::printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(localDist, localPath, n, distribution, rank);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (validate || printResults) {
        MPI_Gatherv(localDist.data(), distribution.counts[rank], MPI_UNSIGNED,
                    rank == 0 ? dist.data() : nullptr, distribution.counts.data(),
                    distribution.displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }
    int returnCode = 0;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        std::printf("Performance: %.3f GOPS\n", ops / seconds / 1.0e9);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(dist, numNodes);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            returnCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return returnCode;
}
