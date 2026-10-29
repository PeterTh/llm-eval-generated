#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { int e = (call); if (e != MPI_SUCCESS) { \
    char msg[MPI_MAX_ERROR_STRING]; int len; MPI_Error_string(e, msg, &len); \
    fprintf(stderr, "MPI error: %.*s\n", len, msg); MPI_Abort(MPI_COMM_WORLD, e); } } while (0)

// Each block updates one 32x32 tile. The left and upper tiles are held in
// shared memory; when either is the output tile, refresh it after each k.
__global__ void updateTiles(unsigned int* matrix, const unsigned int* pivotRows,
                            int n, int firstTile, int pivot, int mode) {
    int tileRow, tileCol;
    if (mode == 0) { tileRow = pivot; tileCol = pivot; }
    else if (mode == 1) { tileRow = pivot; tileCol = blockIdx.x; if (tileCol == pivot) return; }
    else if (mode == 2) { tileRow = firstTile + blockIdx.x; tileCol = pivot; if (tileRow == pivot) return; }
    else { tileRow = firstTile + blockIdx.y; tileCol = blockIdx.x;
           if (tileRow == pivot || tileCol == pivot) return; }
    __shared__ unsigned int left[TILE][TILE], upper[TILE][TILE], out[TILE][TILE];
    const int x = threadIdx.x;
    for (int y = threadIdx.y; y < TILE; y += blockDim.y) {
        int localRow = (tileRow - firstTile) * TILE + y;
        out[y][x] = matrix[(size_t)localRow * n + tileCol * TILE + x];
        left[y][x] = matrix[(size_t)localRow * n + pivot * TILE + x];
        upper[y][x] = pivotRows[(size_t)y * n + tileCol * TILE + x];
    }
    __syncthreads();
    if (mode == 3) {
        unsigned int values[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) values[t] = out[threadIdx.y + t * 8][x];
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            unsigned int b = upper[k][x];
            #pragma unroll
            for (int t = 0; t < 4; ++t)
                values[t] = min(values[t], left[threadIdx.y + t * 8][k] + b);
        }
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int y = threadIdx.y + t * 8;
            matrix[(size_t)((tileRow - firstTile) * TILE + y) * n + tileCol * TILE + x] = values[t];
        }
        return;
    }
    for (int k = 0; k < TILE; ++k) {
        for (int y = threadIdx.y; y < TILE; y += blockDim.y) {
            unsigned int candidate = left[y][k] + upper[k][x];
            if (candidate < out[y][x]) out[y][x] = candidate;
        }
        __syncthreads();
        for (int y = threadIdx.y; y < TILE; y += blockDim.y) {
            if (tileCol == pivot) left[y][x] = out[y][x];
            if (tileRow == pivot) upper[y][x] = out[y][x];
        }
        __syncthreads();
    }
    for (int y = threadIdx.y; y < TILE; y += blockDim.y)
        matrix[(size_t)((tileRow - firstTile) * TILE + y) * n + tileCol * TILE + x] = out[y][x];
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[i * n + i] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k)
                if (dist[i*n+k] < INF && dist[k*n+j] < INF &&
                    dist[i*n+k] + dist[k*n+j] < dist[i*n+j]) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i,j,k);
                    return false;
                }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) { if (!rank) fprintf(stderr, "MPI does not provide required thread support\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    size_t n = 512; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end; const char* arg = argv[++i]; unsigned long long v = strtoull(arg, &end, 10);
            if (*arg == '-' || !*arg || *end || v == 0 || v > INT_MAX - TILE) {
                if (!rank) fprintf(stderr, "Invalid node count: %s\n", arg); MPI_Finalize(); return 1;
            }
            n = static_cast<size_t>(v);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    int tiles = (n + TILE - 1) / TILE, padded = tiles * TILE;
    // MPI's standard collective counts and displacements are int.
    if ((size_t)padded * padded > INT_MAX) { if (!rank) fprintf(stderr, "Matrix exceeds MPI count limit\n"); MPI_Finalize(); return 1; }
    MPI_Comm localComm; MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
    int localRank; MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    int devices; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    std::vector<int> counts(ranks), displs(ranks), starts(ranks + 1);
    for (int r = 0; r <= ranks; ++r) starts[r] = (int)((long long)tiles * r / ranks);
    for (int r = 0; r < ranks; ++r) { counts[r] = (starts[r+1]-starts[r])*TILE*padded; displs[r] = starts[r]*TILE*padded; }
    int localTiles = starts[rank+1] - starts[rank];
    std::vector<unsigned int> full, local(counts[rank]), pivotHost((size_t)TILE*padded);
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\nInitializing graph...\n", n, validate ? "enabled" : "disabled");
        full.assign((size_t)padded*padded, INF);
        // Preserve the original serial rand_r sequence exactly.
        unsigned int seed = 42;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                full[i*(size_t)padded+j] = 1 + (unsigned int)(double(MAX_DISTANCE) * rand_r(&seed) / double(RAND_MAX));
        #pragma omp parallel for
        for (int i = 0; i < padded; ++i) full[(size_t)i*padded+i] = 0;
    }
    MPI_CHECK(MPI_Scatterv(full.data(), counts.data(), displs.data(), MPI_UNSIGNED, local.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    unsigned int *deviceMatrix, *devicePivot;
    CUDA_CHECK(cudaMalloc(&deviceMatrix, std::max(size_t(1), local.size()) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePivot, pivotHost.size() * sizeof(unsigned int)));
    if (!local.empty()) CUDA_CHECK(cudaMemcpy(deviceMatrix, local.data(), local.size()*sizeof(unsigned int), cudaMemcpyHostToDevice));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (!rank) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    dim3 threads(TILE, 8);
    for (int p = 0; p < tiles; ++p) {
        int owner = int(((long long)(p + 1) * ranks - 1) / tiles);
        if (rank == owner) {
            unsigned int* row = deviceMatrix + (size_t)(p-starts[rank])*TILE*padded;
            updateTiles<<<1, threads>>>(deviceMatrix, row, padded, starts[rank], p, 0);
            updateTiles<<<tiles, threads>>>(deviceMatrix, row, padded, starts[rank], p, 1);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(pivotHost.data(), row, pivotHost.size()*sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        MPI_CHECK(MPI_Bcast(pivotHost.data(), (int)pivotHost.size(), MPI_UNSIGNED, owner, MPI_COMM_WORLD));
        if (localTiles) {
            CUDA_CHECK(cudaMemcpy(devicePivot, pivotHost.data(), pivotHost.size()*sizeof(unsigned int), cudaMemcpyHostToDevice));
            updateTiles<<<localTiles, threads>>>(deviceMatrix, devicePivot, padded, starts[rank], p, 2);
            updateTiles<<<dim3(tiles, localTiles), threads>>>(deviceMatrix, devicePivot, padded, starts[rank], p, 3);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    double localSeconds = std::chrono::duration<double>(end-start).count();
    double seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!local.empty()) CUDA_CHECK(cudaMemcpy(local.data(), deviceMatrix, local.size()*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    MPI_CHECK(MPI_Gatherv(local.data(), counts[rank], MPI_UNSIGNED, full.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    int status = 0;
    if (!rank) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(seconds));
        printf("Computation time: %ld ms\n", duration.count());
        printf("Performance: %.3f GOPS\n", double(n)*n*n / seconds / 1e9);
        if (printResults || validate) {
            std::vector<unsigned int> dist(n*n);
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) std::copy_n(full.data()+i*(size_t)padded, n, dist.data()+i*n);
            if (printResults) print_results_int(dist, "DistanceMatrix");
            if (validate) { printf("Validating result...\n"); bool valid = validateResult(dist, n); printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); status = valid ? 0 : 1; }
        }
    }
    CUDA_CHECK(cudaFree(deviceMatrix)); CUDA_CHECK(cudaFree(devicePivot));
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Comm_free(&localComm)); MPI_CHECK(MPI_Finalize()); return status;
}
