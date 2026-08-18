#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000U;
constexpr unsigned int MAX_DISTANCE = 200U;
constexpr int TILE = 32;

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

#define CUDA_CHECK(call) do { \
    const cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, 2); \
    } \
} while (0)

// Matrices on the device are row-major and contain only this rank's rows.
// Keeping complete B-row pivots separately makes the only inter-rank traffic one
// broadcast per blocked Floyd-Warshall iteration.
__global__ void pivotKernel(unsigned int* d, unsigned int* p, int P, int row0, int kb) {
    __shared__ unsigned int sd[TILE][TILE], sp[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    int c = kb * TILE + x;
    int local = kb * TILE + y;
    sd[y][x] = d[local * P + c]; sp[y][x] = p[local * P + c];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        unsigned int candidate = sd[y][k] + sd[k][x];
        if (candidate < sd[y][x]) { sd[y][x] = candidate; sp[y][x] = kb * TILE + k; }
        __syncthreads();
    }
    d[local * P + c] = sd[y][x]; p[local * P + c] = sp[y][x];
}

__global__ void pivotRowKernel(unsigned int* pivot, unsigned int* pivotPath, int P, int kb) {
    int c = blockIdx.x * TILE + threadIdx.x, b = threadIdx.y;
    if (c >= P) return;
    for (int k = 0; k < TILE; ++k) {
        unsigned int candidate = pivot[b * P + kb * TILE + k] + pivot[k * P + c];
        if (candidate < pivot[b * P + c]) { pivot[b * P + c] = candidate; pivotPath[b * P + c] = kb * TILE + k; }
    }
}

__global__ void pivotColumnKernel(unsigned int* d, unsigned int* p, const unsigned int* pivot,
                                  int P, int rows, int row0, int kb) {
    int b = threadIdx.x, li = blockIdx.x * TILE + threadIdx.y;
    if (li >= rows || row0 + li / TILE == kb) return;
    int col = kb * TILE + b;
    for (int k = 0; k < TILE; ++k) {
        unsigned int candidate = d[li * P + kb * TILE + k] + pivot[k * P + col];
        if (candidate < d[li * P + col]) { d[li * P + col] = candidate; p[li * P + col] = kb * TILE + k; }
    }
}

__global__ void remainingKernel(unsigned int* d, unsigned int* p, const unsigned int* pivot,
                                int P, int rows, int row0, int kb) {
    int j = blockIdx.x * TILE + threadIdx.x, li = blockIdx.y * TILE + threadIdx.y;
    if (j >= P || li >= rows || j / TILE == kb || row0 + li / TILE == kb) return;
    for (int k = 0; k < TILE; ++k) {
        unsigned int candidate = d[li * P + kb * TILE + k] + pivot[k * P + j];
        if (candidate < d[li * P + j]) { d[li * P + j] = candidate; p[li * P + j] = kb * TILE + k; }
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n) {
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        dist[i] = 1U + static_cast<unsigned int>(MAX_DISTANCE * rand_r(&seed) / static_cast<double>(RAND_MAX));
    for (size_t i = 0; i < n; ++i) dist[idx2(i, i, n)] = 0;
}

bool validateResult(const std::vector<unsigned int>& d, size_t n) {
    for (size_t i = 0; i < n; ++i) if (d[idx2(i, i, n)] != 0) return false;
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k)
                if (d[idx2(k,i,n)] < INF && d[idx2(j,k,n)] < INF &&
                    d[idx2(k,i,n)] + d[idx2(j,k,n)] < d[idx2(j,i,n)]) return false;
    return true;
}

void printUsage(const char* p) { std::printf("Usage: %s [-n nodes] [-v] [-r] [-h]\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) results = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max() / TILE)) {
        if (!rank) std::fprintf(stderr, "Invalid node count\n"); MPI_Finalize(); return 1;
    }
    const int blocks = (static_cast<int>(n) + TILE - 1) / TILE;
    const int blocksPerRank = (blocks + ranks - 1) / ranks;
    const int totalBlocks = blocksPerRank * ranks, P = totalBlocks * TILE;
    const int localRows = blocksPerRank * TILE, row0 = rank * localRows;
    if (static_cast<long long>(P) * P > std::numeric_limits<int>::max()) {
        if (!rank) std::fprintf(stderr, "Matrix exceeds MPI count limits\n"); MPI_Finalize(); return 1;
    }

    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(rank % devices));
    if (!rank) std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nMPI ranks: %d, CUDA tile: %d\n", n, ranks, TILE);

    std::vector<unsigned int> local(static_cast<size_t>(localRows) * P), localPath(static_cast<size_t>(localRows) * P);
    // Only rank zero materializes the original column-major input.  It packs it
    // once into MPI's block-row distribution, avoiding an O(n^2) replica on
    // every accelerator rank.
    std::vector<unsigned int> packed;
    if (!rank) {
        std::vector<unsigned int> initial(n * n);
        initializeDistanceMatrix(initial, n);
        packed.assign(static_cast<size_t>(P) * P, INF);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < P; ++i) for (int j = 0; j < P; ++j)
            packed[static_cast<size_t>(i)*P+j] = i == j ? 0 :
                (i < static_cast<int>(n) && j < static_cast<int>(n) ? initial[idx2(i,j,n)] : INF);
    }
    MPI_Scatter(packed.data(), localRows*P, MPI_UNSIGNED, local.data(), localRows*P,
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
    for (int li = 0; li < localRows; ++li) for (int j = 0; j < P; ++j) {
        localPath[static_cast<size_t>(li)*P+j] = j;
    }
    unsigned int *d = nullptr, *dp = nullptr, *pivot = nullptr, *pivotPath = nullptr;
    size_t localBytes = local.size() * sizeof(unsigned int), pivotBytes = static_cast<size_t>(TILE)*P*sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&d, localBytes)); CUDA_CHECK(cudaMalloc(&dp, localBytes));
    CUDA_CHECK(cudaMalloc(&pivot, pivotBytes)); CUDA_CHECK(cudaMalloc(&pivotPath, pivotBytes));
    CUDA_CHECK(cudaMemcpy(d, local.data(), localBytes, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dp, localPath.data(), localBytes, cudaMemcpyHostToDevice));
    std::vector<unsigned int> hostPivot(static_cast<size_t>(TILE)*P), hostPivotPath(static_cast<size_t>(TILE)*P);
    MPI_Barrier(MPI_COMM_WORLD); auto start = std::chrono::high_resolution_clock::now();
    dim3 threads(TILE,TILE); dim3 colGrid((localRows+TILE-1)/TILE), rowGrid((P+TILE-1)/TILE);
    for (int kb = 0; kb < totalBlocks; ++kb) {
        int owner = kb / blocksPerRank;
        if (rank == owner) {
            pivotKernel<<<1,threads>>>(d, dp, P, row0, kb); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            int offset = (kb*TILE-row0)*P;
            CUDA_CHECK(cudaMemcpy(hostPivot.data(), d + offset, pivotBytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hostPivotPath.data(), dp + offset, pivotBytes, cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(hostPivot.data(), static_cast<int>(TILE*P), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        MPI_Bcast(hostPivotPath.data(), static_cast<int>(TILE*P), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(pivot, hostPivot.data(), pivotBytes, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(pivotPath, hostPivotPath.data(), pivotBytes, cudaMemcpyHostToDevice));
        if (rank == owner) { pivotRowKernel<<<rowGrid,threads>>>(pivot,pivotPath,P,kb); CUDA_CHECK(cudaGetLastError()); }
        pivotColumnKernel<<<colGrid,threads>>>(d,dp,pivot,P,localRows,row0,kb); CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (rank == owner) {
            // The completed pivot row is also part of this rank's permanent
            // block-row storage and is needed by later pivot iterations.
            int offset = (kb*TILE-row0)*P;
            CUDA_CHECK(cudaMemcpy(d + offset,pivot,pivotBytes,cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(dp + offset,pivotPath,pivotBytes,cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(hostPivot.data(),pivot,pivotBytes,cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hostPivotPath.data(),pivotPath,pivotBytes,cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(hostPivot.data(), static_cast<int>(TILE*P), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        MPI_Bcast(hostPivotPath.data(), static_cast<int>(TILE*P), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(pivot,hostPivot.data(),pivotBytes,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(pivotPath,hostPivotPath.data(),pivotBytes,cudaMemcpyHostToDevice));
        remainingKernel<<<dim3((P+TILE-1)/TILE,(localRows+TILE-1)/TILE),threads>>>(d,dp,pivot,P,localRows,row0,kb); CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize()); auto end = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(local.data(),d,localBytes,cudaMemcpyDeviceToHost));
    std::vector<unsigned int> all; if (!rank) all.resize(static_cast<size_t>(P)*P);
    MPI_Gather(local.data(), localRows*P, MPI_UNSIGNED, all.data(), localRows*P, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(d)); CUDA_CHECK(cudaFree(dp)); CUDA_CHECK(cudaFree(pivot)); CUDA_CHECK(cudaFree(pivotPath));
    int exitCode = 0;
    if (!rank) {
        std::vector<unsigned int> output(n*n);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) output[idx2(i,j,n)] = all[i*P+j];
        const double seconds = std::chrono::duration<double>(end-start).count();
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GOPS\n", seconds*1e3, (double(n)*n*n)/seconds/1e9);
        if (results) print_results_int(output,"DistanceMatrix");
        if (validate && !validateResult(output,n)) { std::printf("Validation: FAILED\n"); exitCode=1; }
        else if (validate) std::printf("Validation: PASSED\n");
    }
    MPI_Bcast(&exitCode,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return exitCode;
}
