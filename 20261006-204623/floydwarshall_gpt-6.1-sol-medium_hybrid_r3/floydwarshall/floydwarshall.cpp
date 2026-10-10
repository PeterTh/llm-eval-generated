#include <algorithm>
#include <cerrno>
#include <climits>
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
constexpr int TILE = 32;
inline constexpr size_t idx2(size_t i, size_t j, size_t n) { return j * n + i; }

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// Snapshot each pivot row at exactly its original k step. Each column tile
// independently advances the panel, using a replicated panel square.
__global__ void preparePanel(unsigned* dist, unsigned* path, unsigned* panel, const unsigned* panelColumns,
                             size_t stride, size_t localPivot, size_t pivot, int width) {
    __shared__ unsigned left[TILE][TILE];
    __shared__ unsigned right[TILE][TILE];
    const int x = threadIdx.x;
    const size_t column = size_t(blockIdx.x) * TILE + x;
    unsigned pv[4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        int y = threadIdx.y + a * 8;
        left[y][x] = panelColumns[(localPivot + y) * TILE + x];
        right[y][x] = dist[(localPivot + y) * stride + column];
        pv[a] = path[(localPivot + y) * stride + column];
    }
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        if (threadIdx.y == 0) panel[size_t(k) * stride + column] = right[k][x];
        unsigned ik[4], kj = right[k][x], lk = left[k][x];
        #pragma unroll
        for (int a = 0; a < 4; ++a) ik[a] = left[threadIdx.y + a * 8][k];
        __syncthreads();
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            int y = threadIdx.y + a * 8;
            unsigned candidate = ik[a] + kj;
            if (candidate < right[y][x]) {
                right[y][x] = candidate;
                pv[a] = unsigned(pivot + k);
            }
            left[y][x] = min(left[y][x], ik[a] + lk);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        int y = threadIdx.y + a * 8;
        dist[(localPivot + y) * stride + column] = right[y][x];
        path[(localPivot + y) * stride + column] = pv[a];
    }
}

// Maintain each row's panel columns in shared memory. The broadcast contains
// time-ordered snapshots, so strict improvements and path pivots match serial FW.
__global__ void updateRows(unsigned* dist, unsigned* path, const unsigned* panel, const unsigned* panelColumns,
                           size_t stride, size_t firstRow, size_t pivot, int width) {
    const size_t rowBase = size_t(blockIdx.y) * TILE;
    if (firstRow + rowBase == pivot) return;
    __shared__ unsigned left[TILE][TILE];
    __shared__ unsigned right[TILE][TILE];
    __shared__ unsigned square[TILE][TILE];
    int x = threadIdx.x;
    size_t column = size_t(blockIdx.x) * TILE + x;
    unsigned values[4], paths[4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        int y = threadIdx.y + a * 8;
        size_t offset = (rowBase + y) * stride;
        left[y][x] = panelColumns[(rowBase + y) * TILE + x];
        right[y][x] = panel[size_t(y) * stride + column];
        square[y][x] = panel[size_t(y) * stride + pivot + x];
        values[a] = dist[offset + column];
        paths[a] = path[offset + column];
    }
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        unsigned ik[4];
        #pragma unroll
        for (int a = 0; a < 4; ++a) ik[a] = left[threadIdx.y + a * 8][k];
        __syncthreads();
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            int y = threadIdx.y + a * 8;
            unsigned candidate = ik[a] + right[k][x];
            if (candidate < values[a]) {
                values[a] = candidate;
                paths[a] = unsigned(pivot + k);
            }
            left[y][x] = min(left[y][x], ik[a] + square[k][x]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        size_t offset = (rowBase + threadIdx.y + a * 8) * stride + column;
        dist[offset] = values[a];
        path[offset] = paths[a];
    }
}

// Chunk collectives to avoid MPI's signed-int count limit. Host staging makes
// this work with ordinary MPI installations as well as CUDA-aware MPI.
void broadcastPanel(unsigned* data, size_t count, int owner) {
    for (size_t offset = 0; offset < count;) {
        int chunk = int(std::min(count - offset, size_t(INT_MAX)));
        MPI_Bcast(data + offset, chunk, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        offset += chunk;
    }
}
void transferRows(unsigned* data, size_t count, int peer, bool send) {
    for (size_t offset = 0; offset < count;) {
        int chunk = int(std::min(count - offset, size_t(INT_MAX)));
        if (send) MPI_Send(data + offset, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data + offset, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        offset += chunk;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t n) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (size_t i = 0; i < n; ++i) valid &= dist[i * n + i] == 0;
    const size_t sample = std::min(n, size_t(10));
    #pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (size_t i = 0; i < sample; ++i)
        for (size_t j = 0; j < sample; ++j)
            for (size_t k = 0; k < n; ++k) {
                unsigned ik = dist[i * n + k], kj = dist[k * n + j];
                if (ik < INF && kj < INF) valid &= ik + kj >= dist[i * n + j];
            }
    return valid != 0;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* value = argv[++i];
            unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || *value == '-' || end == value || *end || parsed > UINT_MAX) {
                if (!rank) fprintf(stderr, "Invalid number of nodes: %s\n", value);
                MPI_Finalize(); return 1;
            }
            numNodes = size_t(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(node, &localRank);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) {
        if (!rank) fprintf(stderr, "This benchmark requires a CUDA GPU on every rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&node);
    size_t blocks = (numNodes + TILE - 1) / TILE;
    size_t stride = blocks * TILE;
    if (stride && stride > std::numeric_limits<size_t>::max() / stride / sizeof(unsigned)) {
        if (!rank) fprintf(stderr, "Matrix size overflows address space.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    auto blockStart = [=](int r) { return blocks / ranks * r + std::min(size_t(r), blocks % ranks); };
    size_t first = blockStart(rank) * TILE;
    size_t rows = (blockStart(rank + 1) - blockStart(rank)) * TILE;
    size_t count = rows * stride;
    std::vector<unsigned> dist;
    std::vector<unsigned> local(count, INF), localPath(count);
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n",
               numNodes, validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        dist.resize(numNodes * numNodes);
        unsigned seed = 42;
        for (size_t i = 0; i < dist.size(); ++i)
            dist[i] = 1 + (unsigned)(double(MAX_DISTANCE) * rand_r(&seed) / double(RAND_MAX));
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numNodes; ++i) dist[i * numNodes + i] = 0;
        // Pack one rank at a time: no replicated global matrix on worker ranks.
        for (int r = 0; r < ranks; ++r) {
            size_t start = blockStart(r) * TILE;
            size_t nr = (blockStart(r + 1) - blockStart(r)) * TILE;
            std::vector<unsigned> packed(r == 0 ? 0 : nr * stride, INF);
            unsigned* target = r == 0 ? local.data() : packed.data();
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < nr; ++i) {
                if (start + i < numNodes)
                    std::copy_n(dist.data() + (start + i) * numNodes, numNodes, target + i * stride);
                target[i * stride + start + i] = 0;
            }
            if (r) transferRows(target, nr * stride, r, true);
        }
        if (!validate && !printResults) std::vector<unsigned>().swap(dist);
    } else transferRows(local.data(), count, 0, false);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i)
        std::fill_n(localPath.data() + i * stride, stride, unsigned(first + i));
    unsigned *deviceDist = nullptr, *devicePath = nullptr, *devicePanel = nullptr, *hostPanel = nullptr, *deviceColumns = nullptr;
    cudaStream_t stream;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (count) {
        CUDA(cudaMalloc(&deviceDist, count * sizeof(unsigned)));
        CUDA(cudaMalloc(&devicePath, count * sizeof(unsigned)));
        CUDA(cudaMalloc(&deviceColumns, rows * TILE * sizeof(unsigned)));
        CUDA(cudaMemcpyAsync(deviceDist, local.data(), count * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
        CUDA(cudaMemcpyAsync(devicePath, localPath.data(), count * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
    }
    if (stride) {
        CUDA(cudaMalloc(&devicePanel, TILE * stride * sizeof(unsigned)));
        CUDA(cudaMemsetAsync(devicePanel, 0, TILE * stride * sizeof(unsigned), stream));
        CUDA(cudaMallocHost(&hostPanel, TILE * stride * sizeof(unsigned)));
    }
    CUDA(cudaStreamSynchronize(stream));
    std::vector<unsigned>().swap(localPath);
    if (!rank) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    int owner = 0;
    for (size_t b = 0; b < blocks; ++b) {
        while (b >= blockStart(owner + 1)) ++owner;
        size_t pivot = b * TILE;
        int width = int(std::min(size_t(TILE), numNodes - pivot));
        // Freeze panel columns before any tile writes to the matrix.
        if (rows) CUDA(cudaMemcpy2DAsync(deviceColumns, TILE * sizeof(unsigned),
            deviceDist + pivot, stride * sizeof(unsigned), TILE * sizeof(unsigned), rows,
            cudaMemcpyDeviceToDevice, stream));
        if (rank == owner) {
            preparePanel<<<unsigned(blocks), dim3(32, 8), 0, stream>>>(
                deviceDist, devicePath, devicePanel, deviceColumns, stride, pivot - first, pivot, width);
            CUDA(cudaGetLastError());
            CUDA(cudaMemcpyAsync(hostPanel, devicePanel, width * stride * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
            CUDA(cudaStreamSynchronize(stream));
        }
        broadcastPanel(hostPanel, width * stride, owner);
        if (rank != owner && rows)
            CUDA(cudaMemcpyAsync(devicePanel, hostPanel, width * stride * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
        if (rows) {
            updateRows<<<dim3(unsigned(blocks), unsigned(rows / TILE)), dim3(32, 8), 0, stream>>>(
                deviceDist, devicePath, devicePanel, deviceColumns, stride, first, pivot, width);
            CUDA(cudaGetLastError());
        }
        // Fence before host staging reuse; MPI never accesses an in-flight copy.
        CUDA(cudaStreamSynchronize(stream));
    }
    double elapsed = MPI_Wtime() - start, seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %lld ms\n", (long long)(seconds * 1000));
        double ops = double(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0);
    }
    int success = 1;
    if (validate || printResults) {
        if (count) CUDA(cudaMemcpy(local.data(), deviceDist, count * sizeof(unsigned), cudaMemcpyDeviceToHost));
        if (!rank) {
            for (int r = 0; r < ranks; ++r) {
                size_t startRow = blockStart(r) * TILE;
                size_t nr = (blockStart(r + 1) - blockStart(r)) * TILE;
                std::vector<unsigned> received(r ? nr * stride : 0);
                unsigned* source = r ? received.data() : local.data();
                if (r) transferRows(source, nr * stride, r, false);
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < nr; ++i)
                    if (startRow + i < numNodes)
                        std::copy_n(source + i * stride, numNodes, dist.data() + (startRow + i) * numNodes);
            }
            if (printResults) print_results_int(dist, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                success = validateResult(dist, numNodes);
                printf("Validation: %s\n", success ? "PASSED" : "FAILED");
            }
        } else transferRows(local.data(), count, 0, true);
    }
    MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA(cudaFree(deviceColumns)); CUDA(cudaFree(deviceDist)); CUDA(cudaFree(devicePath)); CUDA(cudaFree(devicePanel));
    CUDA(cudaFreeHost(hostPanel)); CUDA(cudaStreamDestroy(stream));
    MPI_Finalize();
    return success ? 0 : 1;
}
