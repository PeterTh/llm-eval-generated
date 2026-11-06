#include <algorithm>
#include <cerrno>
#include <climits>
#include <mpi.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
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

// Each MPI rank owns whole 32-row tiles.  The padded leading dimension is a
// multiple of 32, so every thread in a warp accesses contiguous elements.
constexpr int FW_TILE = 32;

__global__ void diagonal_kernel(unsigned int* matrix, size_t n, size_t row,
                                size_t pivot) {
    __shared__ unsigned int tile[FW_TILE][FW_TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    for (int r = y; r < FW_TILE; r += 8)
        tile[r][x] = matrix[(row + r) * n + pivot * FW_TILE + x];
    __syncthreads();
    for (int k = 0; k < FW_TILE; ++k) {
        for (int r = y; r < FW_TILE; r += 8) {
            unsigned int candidate = tile[r][k] + tile[k][x];
            if (candidate < tile[r][x]) tile[r][x] = candidate;
        }
        __syncthreads();
    }
    for (int r = y; r < FW_TILE; r += 8)
        matrix[(row + r) * n + pivot * FW_TILE + x] = tile[r][x];
}

__global__ void pivot_row_kernel(unsigned int* matrix, size_t n, size_t row,
                                size_t pivot) {
    const size_t col_tile = blockIdx.x;
    if (col_tile == pivot) return;
    __shared__ unsigned int diagonal[FW_TILE][FW_TILE];
    __shared__ unsigned int target[FW_TILE][FW_TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    for (int r = y; r < FW_TILE; r += 8) {
        diagonal[r][x] = matrix[(row + r) * n + pivot * FW_TILE + x];
        target[r][x] = matrix[(row + r) * n + col_tile * FW_TILE + x];
    }
    __syncthreads();
    for (int k = 0; k < FW_TILE; ++k) {
        for (int r = y; r < FW_TILE; r += 8) {
            unsigned int candidate = diagonal[r][k] + target[k][x];
            if (candidate < target[r][x]) target[r][x] = candidate;
        }
        __syncthreads();
    }
    for (int r = y; r < FW_TILE; r += 8)
        matrix[(row + r) * n + col_tile * FW_TILE + x] = target[r][x];
}

__global__ void column_kernel(unsigned int* matrix, const unsigned int* pivot_rows,
                              size_t n, size_t pivot, int skip_local_tile) {
    const size_t row_tile = blockIdx.x;
    if (static_cast<int>(row_tile) == skip_local_tile) return;
    __shared__ unsigned int diagonal[FW_TILE][FW_TILE];
    __shared__ unsigned int column[FW_TILE][FW_TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    for (int r = y; r < FW_TILE; r += 8) {
        diagonal[r][x] = pivot_rows[r * n + pivot * FW_TILE + x];
        column[r][x] = matrix[(row_tile * FW_TILE + r) * n + pivot * FW_TILE + x];
    }
    __syncthreads();
    for (int k = 0; k < FW_TILE; ++k) {
        for (int r = y; r < FW_TILE; r += 8) {
            unsigned int candidate = column[r][k] + diagonal[k][x];
            if (candidate < column[r][x]) column[r][x] = candidate;
        }
        __syncthreads();
    }
    for (int r = y; r < FW_TILE; r += 8)
        matrix[(row_tile * FW_TILE + r) * n + pivot * FW_TILE + x] = column[r][x];
}

__global__ void remaining_kernel(unsigned int* matrix, const unsigned int* pivot_rows,
                                 size_t n, size_t pivot, int skip_local_tile) {
    const size_t col_tile = blockIdx.x, row_tile = blockIdx.y;
    if (col_tile == pivot || static_cast<int>(row_tile) == skip_local_tile) return;
    __shared__ unsigned int column[FW_TILE][FW_TILE];
    __shared__ unsigned int row[FW_TILE][FW_TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    unsigned int values[4];
    for (int t = 0; t < 4; ++t) {
        int r = y + 8 * t;
        column[r][x] = matrix[(row_tile * FW_TILE + r) * n + pivot * FW_TILE + x];
        row[r][x] = pivot_rows[r * n + col_tile * FW_TILE + x];
        values[t] = matrix[(row_tile * FW_TILE + r) * n + col_tile * FW_TILE + x];
    }
    __syncthreads();
    for (int k = 0; k < FW_TILE; ++k) {
        unsigned int right = row[k][x];
        for (int t = 0; t < 4; ++t) {
            unsigned int candidate = column[y + 8 * t][k] + right;
            values[t] = min(values[t], candidate);
        }
    }
    for (int t = 0; t < 4; ++t)
        matrix[(row_tile * FW_TILE + y + 8 * t) * n + col_tile * FW_TILE + x] = values[t];
}

void launch_diagonal(unsigned int* matrix, size_t n, size_t local_pivot_row,
                     size_t pivot_tile) {
    diagonal_kernel<<<1, dim3(32, 8)>>>(matrix, n, local_pivot_row, pivot_tile);
}

void launch_pivot_row(unsigned int* matrix, size_t n, size_t local_pivot_row,
                      size_t pivot_tile) {
    pivot_row_kernel<<<n / FW_TILE, dim3(32, 8)>>>(matrix, n, local_pivot_row, pivot_tile);
}

void launch_column(unsigned int* matrix, const unsigned int* pivot_rows, size_t n,
                   size_t local_tile_rows, int skip_local_tile, size_t pivot_tile) {
    if (local_tile_rows)
        column_kernel<<<local_tile_rows, dim3(32, 8)>>>(matrix, pivot_rows, n, pivot_tile, skip_local_tile);
}

void launch_remaining(unsigned int* matrix, const unsigned int* pivot_rows, size_t n,
                      size_t local_tile_rows, int skip_local_tile, size_t pivot_tile) {
    if (local_tile_rows)
        remaining_kernel<<<dim3(n / FW_TILE, local_tile_rows), dim3(32, 8)>>>(matrix, pivot_rows, n, pivot_tile, skip_local_tile);
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

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// MPI counts are int, while a matrix can contain more than INT_MAX elements.
static void transfer(unsigned int* data, size_t count, int peer, int tag, bool send) {
    while (count != 0) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t numNodes = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            errno = 0;
            unsigned long long value = strtoull(arg, &end, 10);
            if (errno || *arg == '-' || *arg == '\0' || *end != '\0' || value > INT_MAX - 31) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
            numNodes = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");
    MPI_Comm_free(&localComm);

    constexpr size_t tile = 32;
    const size_t n = (numNodes + tile - 1) / tile * tile;
    const size_t tiles = n / tile;
    auto tileStart = [=](int r) -> size_t { return tiles * static_cast<size_t>(r) / ranks; };
    const size_t firstTile = tileStart(rank);
    const size_t localTiles = tileStart(rank + 1) - firstTile;
    const size_t localElements = localTiles * tile * n;
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    std::vector<unsigned int> matrix;
    std::vector<unsigned int> local(localElements);
    if (rank == 0) {
        matrix.resize(n * n);
        #pragma omp parallel for schedule(static) if(n * n > 1048576)
        for (size_t i = 0; i < n * n; ++i) matrix[i] = INF;
        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE);
        // rand_r is stateful: retain the original row-major random stream.
        for (size_t i = 0; i < numNodes; ++i)
            for (size_t j = 0; j < numNodes; ++j)
                matrix[i * n + j] = 1 + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
        #pragma omp parallel for schedule(static) if(n > 1024)
        for (size_t i = 0; i < n; ++i) matrix[i * n + i] = 0;
        if (localElements) std::copy_n(matrix.data(), localElements, local.data());
        for (int r = 1; r < ranks; ++r) {
            const size_t offset = tileStart(r) * tile * n;
            const size_t count = (tileStart(r + 1) - tileStart(r)) * tile * n;
            if (count) transfer(matrix.data() + offset, count, r, 10, true);
        }
    } else if (localElements) transfer(local.data(), localElements, 0, 10, false);

    unsigned int *deviceMatrix = nullptr, *devicePivot = nullptr, *pivot = nullptr;
    const size_t pivotElements = tile * n;
    checkCuda(cudaMalloc(&deviceMatrix, std::max(localElements, size_t(1)) * sizeof(unsigned int)), "cudaMalloc matrix");
    checkCuda(cudaMalloc(&devicePivot, std::max(pivotElements, size_t(1)) * sizeof(unsigned int)), "cudaMalloc pivot");
    checkCuda(cudaMallocHost(&pivot, std::max(pivotElements, size_t(1)) * sizeof(unsigned int)), "cudaMallocHost pivot");
    if (localElements) checkCuda(cudaMemcpy(deviceMatrix, local.data(), localElements * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy matrix to GPU");
    std::vector<unsigned int>().swap(local);
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (size_t k = 0; k < tiles; ++k) {
        // Partition boundaries are contiguous and may contain empty ranks.
        int owner = static_cast<int>(((k + 1) * static_cast<size_t>(ranks) - 1) / tiles);
        int skip = -1;
        if (rank == owner) {
            skip = static_cast<int>(k - firstTile);
            launch_diagonal(deviceMatrix, n, static_cast<size_t>(skip) * tile, k);
            launch_pivot_row(deviceMatrix, n, static_cast<size_t>(skip) * tile, k);
            checkCuda(cudaGetLastError(), "pivot kernels");
            if (ranks > 1)
                checkCuda(cudaMemcpy(pivot, deviceMatrix + static_cast<size_t>(skip) * tile * n,
                                     pivotElements * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy pivot from GPU");
        }
        for (size_t offset = 0; ranks > 1 && offset < pivotElements;) {
            int count = static_cast<int>(std::min(pivotElements - offset, static_cast<size_t>(INT_MAX)));
            MPI_Bcast(pivot + offset, count, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            offset += count;
        }
        if (rank != owner)
            checkCuda(cudaMemcpy(devicePivot, pivot, pivotElements * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy pivot to GPU");
        const unsigned int* pivotOnDevice = rank == owner ? deviceMatrix + static_cast<size_t>(skip) * tile * n : devicePivot;
        if (localTiles) {
            launch_column(deviceMatrix, pivotOnDevice, static_cast<int>(n), static_cast<int>(localTiles), skip, static_cast<int>(k));
            launch_remaining(deviceMatrix, pivotOnDevice, static_cast<int>(n), static_cast<int>(localTiles), skip, static_cast<int>(k));
            checkCuda(cudaGetLastError(), "update kernels");
        }
    }
    checkCuda(cudaDeviceSynchronize(), "finish computation");
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        long long milliseconds = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", milliseconds);
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / maxElapsed / 1e9);
    }

    if (validate || printResults) {
        local.resize(localElements);
        if (localElements) checkCuda(cudaMemcpy(local.data(), deviceMatrix, localElements * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy result from GPU");
        if (rank == 0) {
            if (localElements) std::copy_n(local.data(), localElements, matrix.data());
            for (int r = 1; r < ranks; ++r) {
                size_t offset = tileStart(r) * tile * n;
                size_t count = (tileStart(r + 1) - tileStart(r)) * tile * n;
                if (count) transfer(matrix.data() + offset, count, r, 11, false);
            }
        } else if (localElements) transfer(local.data(), localElements, 0, 11, true);
    }
    int valid = 1;
    if (rank == 0 && (validate || printResults)) {
        std::vector<unsigned int> dist(numNodes * numNodes);
        #pragma omp parallel for schedule(static) if(numNodes > 1024)
        for (size_t i = 0; i < numNodes; ++i)
            std::copy_n(matrix.data() + i * n, numNodes, dist.data() + i * numNodes);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            valid = validateResult(dist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFreeHost(pivot);
    cudaFree(devicePivot);
    cudaFree(deviceMatrix);
    MPI_Finalize();
    return valid ? 0 : 1;
}
