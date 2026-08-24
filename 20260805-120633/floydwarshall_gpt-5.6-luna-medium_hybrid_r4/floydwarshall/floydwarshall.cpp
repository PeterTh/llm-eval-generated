#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t n,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    // Keep the reference generator and traversal order unchanged.
    for (size_t i = 0; i < n * n; ++i)
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), n)] = 0;
}

__global__ void floydWarshallKernel(unsigned int* __restrict__ dist,
                                    unsigned int* __restrict__ path,
                                    const unsigned int* __restrict__ pivotColumn,
                                    const size_t n, const size_t localN,
                                    const size_t localOffset, const size_t k) {
    const size_t localI = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t j = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (localI >= localN || j >= n) return;

    // Layout is [destination][owned source], so neighboring threads access
    // neighboring source columns. pivotColumn is a snapshot for this k.
    const size_t out = j * localN + localI;
    const unsigned int candidate = dist[k * localN + localI] + pivotColumn[j];
    if (candidate < dist[out]) {
        dist[out] = candidate;
        path[out] = static_cast<unsigned int>(k);
    }
    (void)n;
    (void)localOffset;
}

inline void cudaCheck(const cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    const size_t sample = std::min(n, static_cast<size_t>(10));
    for (size_t i = 0; i < sample; ++i)
        for (size_t j = 0; j < sample; ++j)
            for (size_t k = 0; k < n; ++k) {
                const unsigned int ij = dist[idx2(j, i, n)];
                const unsigned int ik = dist[idx2(k, i, n)];
                const unsigned int kj = dist[idx2(j, k, n)];
                if (ik < INF && kj < INF && ik + kj < ij) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
    return true;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>  Number of nodes (default: 512)\n"
                "  -v       Enable validation\n  -r       Print results\n  -h       Show help\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI implementation lacks MPI_THREAD_FUNNELED\n");
        MPI_Finalize();
        return 1;
    }

    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max()) || world > static_cast<int>(n)) {
        if (rank == 0) std::fprintf(stderr, "Invalid node/rank count\n");
        MPI_Finalize();
        return 1;
    }

    // Select a device by node-local MPI rank, allowing multiple MPI ranks/node.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");

    const size_t base = n / static_cast<size_t>(world), rem = n % static_cast<size_t>(world);
    const size_t localN = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t localOffset = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t localElements = n * localN;
    std::vector<int> counts(world), displs(world);
    for (int r = 0; r < world; ++r) {
        const size_t rn = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        const size_t off = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
        if (rn * n > static_cast<size_t>(std::numeric_limits<int>::max()) || off * n > static_cast<size_t>(std::numeric_limits<int>::max())) MPI_Abort(MPI_COMM_WORLD, 2);
        counts[r] = static_cast<int>(rn * n);
        displs[r] = static_cast<int>(off * n); // used only as documentation for ownership
    }

    std::vector<unsigned int> global;
    std::vector<unsigned int> packed;
    if (rank == 0) {
        global.resize(n * n);
        initializeDistanceMatrix(global, n, 1, MAX_DISTANCE);
        packed.resize(n * n);
        #pragma omp parallel for schedule(static)
        for (long long r = 0; r < world; ++r) {
            const size_t rn = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            const size_t off = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
            for (size_t j = 0; j < n; ++j)
                std::memcpy(packed.data() + off * n + j * rn, global.data() + j * n + off, rn * sizeof(unsigned int));
        }
    }
    std::vector<unsigned int> dist(localElements), path(localElements);
    MPI_Scatterv(rank == 0 ? packed.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 dist.data(), static_cast<int>(localElements), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static)
    for (long long x = 0; x < static_cast<long long>(localElements); ++x)
        path[static_cast<size_t>(x)] = static_cast<unsigned int>(localOffset + static_cast<size_t>(x) % localN);

    unsigned int *dDist = nullptr, *dPath = nullptr, *dPivot = nullptr;
    cudaCheck(cudaMalloc(&dDist, localElements * sizeof(unsigned int)), "cudaMalloc(dist)");
    cudaCheck(cudaMalloc(&dPath, localElements * sizeof(unsigned int)), "cudaMalloc(path)");
    cudaCheck(cudaMalloc(&dPivot, n * sizeof(unsigned int)), "cudaMalloc(pivot)");
    cudaCheck(cudaMemcpy(dDist, dist.data(), localElements * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy dist");
    cudaCheck(cudaMemcpy(dPath, path.data(), localElements * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy path");
    std::vector<unsigned int> pivot(n);
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned int>((localN + block.x - 1) / block.x), static_cast<unsigned int>((n + block.y - 1) / block.y));
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t k = 0; k < n; ++k) {
        size_t ownerOffset = 0;
        int kOwner = 0;
        for (int r = 0; r < world; ++r) {
            const size_t rn = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            if (k < ownerOffset + rn) { kOwner = r; break; }
            ownerOffset += rn;
        }
        if (rank == kOwner) cudaCheck(cudaMemcpy2D(pivot.data(), sizeof(unsigned int), dDist + (k - ownerOffset), localN * sizeof(unsigned int), sizeof(unsigned int), n, cudaMemcpyDeviceToHost), "copy pivot");
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, kOwner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(dPivot, pivot.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy pivot to device");
        floydWarshallKernel<<<grid, block>>>(dDist, dPath, dPivot, n, localN, localOffset, k);
        cudaCheck(cudaGetLastError(), "kernel launch");
    }
    cudaCheck(cudaDeviceSynchronize(), "kernel completion");
    const auto end = std::chrono::high_resolution_clock::now();
    const double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double ms = 0.0;
    MPI_Reduce(&localMs, &ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(dist.data(), dDist, localElements * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy result");
    cudaFree(dPivot); cudaFree(dPath); cudaFree(dDist);

    std::vector<unsigned int> gathered(rank == 0 ? n * n : 0);
    MPI_Gatherv(dist.data(), static_cast<int>(localElements), MPI_UNSIGNED,
                rank == 0 ? gathered.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        // Gatherv receives packed column blocks; restore the reference layout.
        for (int r = 0; r < world; ++r) { const size_t rn = base + (static_cast<size_t>(r) < rem ? 1 : 0); const size_t off = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem); for (size_t j = 0; j < n; ++j) std::memcpy(global.data() + j * n + off, gathered.data() + off * n + j * rn, rn * sizeof(unsigned int)); }
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GOPS\n", ms, static_cast<double>(n) * n * n / (ms * 1.0e6));
        if (printResults) print_results_int(global, "DistanceMatrix");
        if (validate) { std::printf("Validating result...\nValidation: %s\n", validateResult(global, n) ? "PASSED" : "FAILED"); }
    }
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return 0;
}
