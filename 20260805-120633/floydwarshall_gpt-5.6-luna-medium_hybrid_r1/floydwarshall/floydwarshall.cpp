#include <algorithm>
#include <chrono>
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

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t n,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    // Keep the original generator and traversal order so the benchmark's input is unchanged.
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < n * n; ++i)
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), n)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t n) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long j = 0; j < static_cast<long long>(n); ++j) {
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            path[idx2(static_cast<size_t>(i), static_cast<size_t>(j), n)] =
                static_cast<unsigned int>(j);
            path[idx2(static_cast<size_t>(j), static_cast<size_t>(i), n)] =
                static_cast<unsigned int>(i);
        }
    }
}

__global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path,
                                    const unsigned int* pivot, size_t localRows,
                                    size_t n, size_t firstRow, size_t k) {
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t localRow = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (localRow >= localRows || col >= n) return;

    const size_t offset = localRow * n + col;
    const unsigned int candidate = dist[localRow * n + k] + pivot[col];
    if (candidate < dist[offset]) {
        dist[offset] = candidate;
        path[offset] = static_cast<unsigned int>(k);
    }
    (void)firstRow; // Retained in the interface to make row ownership explicit at the call site.
}

[[noreturn]] void cudaFailure(const char* operation, cudaError_t error, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    MPI_Abort(comm, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t error, const char* operation, MPI_Comm comm) {
    if (error != cudaSuccess) cudaFailure(operation, error, comm);
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -n <num>     Number of nodes in the graph (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    const size_t sample = std::min(n, static_cast<size_t>(10));
    int valid = 1;
    #pragma omp parallel for collapse(2) schedule(static) reduction(&:valid)
    for (long long i = 0; i < static_cast<long long>(sample); ++i) {
        for (long long j = 0; j < static_cast<long long>(sample); ++j) {
            for (size_t k = 0; k < n; ++k) {
                const unsigned int a = dist[idx2(static_cast<size_t>(k), static_cast<size_t>(i), n)];
                const unsigned int b = dist[idx2(static_cast<size_t>(j), k, n)];
                if (a < INF && b < INF && a + b < dist[idx2(static_cast<size_t>(j), static_cast<size_t>(i), n)])
                    valid = 0;
            }
        }
    }
    if (!valid) std::printf("Validation failed: triangle inequality violated\n");
    return valid != 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    size_t n = 512;
    bool validate = false, printResults = false, parseOk = true;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else parseOk = false;
    }
    if (!parseOk || n == 0) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount", MPI_COMM_WORLD);
    if (devices == 0) cudaFailure("selecting a CUDA device", cudaErrorNoDevice, MPI_COMM_WORLD);
    checkCuda(cudaSetDevice(rank % devices), "cudaSetDevice", MPI_COMM_WORLD);

    const size_t base = n / static_cast<size_t>(world);
    const size_t remainder = n % static_cast<size_t>(world);
    const size_t firstRow = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    const size_t localRows = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    std::vector<int> counts(world), displacements(world);
    for (int r = 0; r < world; ++r) {
        const size_t rows = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
        counts[r] = static_cast<int>(rows * n);
        displacements[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder)) * n);
    }

    std::vector<unsigned int> globalDist, globalPath;
    if (rank == 0) {
        globalDist.resize(n * n);
        globalPath.resize(n * n);
        initializeDistanceMatrix(globalDist, n, 1, MAX_DISTANCE);
        initializePathMatrix(globalPath, n);
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n",
                    n, validate ? "enabled" : "disabled");
    }
    std::vector<unsigned int> dist(localRows * n), path(localRows * n), pivot(n);
    MPI_Scatterv(globalDist.data(), counts.data(), displacements.data(), MPI_UNSIGNED,
                 dist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(globalPath.data(), counts.data(), displacements.data(), MPI_UNSIGNED,
                 path.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    globalDist.clear(); globalPath.clear();

    unsigned int *dDist = nullptr, *dPath = nullptr, *dPivot = nullptr;
    checkCuda(cudaMalloc(&dDist, dist.size() * sizeof(unsigned int)), "cudaMalloc(dist)", MPI_COMM_WORLD);
    checkCuda(cudaMalloc(&dPath, path.size() * sizeof(unsigned int)), "cudaMalloc(path)", MPI_COMM_WORLD);
    checkCuda(cudaMalloc(&dPivot, n * sizeof(unsigned int)), "cudaMalloc(pivot)", MPI_COMM_WORLD);
    checkCuda(cudaMemcpy(dDist, dist.data(), dist.size() * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy dist", MPI_COMM_WORLD);
    checkCuda(cudaMemcpy(dPath, path.data(), path.size() * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy path", MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const dim3 block(32, 8);
    const dim3 grid((n + block.x - 1) / block.x, (localRows + block.y - 1) / block.y);
    for (size_t k = 0; k < n; ++k) {
        int pivotOwner = 0;
        for (int r = 0; r < world; ++r) {
            const size_t rFirst = static_cast<size_t>(displacements[r]) / n;
            const size_t rRows = static_cast<size_t>(counts[r]) / n;
            if (k >= rFirst && k < rFirst + rRows) { pivotOwner = r; break; }
        }
        if (rank == pivotOwner) {
            // The pivot row was updated by the preceding CUDA iteration; make
            // that device-resident row available to MPI for the next broadcast.
            if (k == firstRow)
                std::copy_n(dist.data(), n, pivot.data());
            else
                checkCuda(cudaMemcpy(pivot.data(), dDist + (k - firstRow) * n,
                                     n * sizeof(unsigned int), cudaMemcpyDeviceToHost),
                          "copy pivot row", MPI_COMM_WORLD);
        }
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, pivotOwner, MPI_COMM_WORLD);
        checkCuda(cudaMemcpy(dPivot, pivot.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy pivot", MPI_COMM_WORLD);
        floydWarshallKernel<<<grid, block>>>(dDist, dPath, dPivot, localRows, n, firstRow, k);
        checkCuda(cudaGetLastError(), "launch Floyd-Warshall kernel", MPI_COMM_WORLD);
        checkCuda(cudaDeviceSynchronize(), "synchronize Floyd-Warshall kernel", MPI_COMM_WORLD);
    }
    checkCuda(cudaMemcpy(dist.data(), dDist, dist.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy result", MPI_COMM_WORLD);
    checkCuda(cudaFree(dDist), "free dist", MPI_COMM_WORLD);
    checkCuda(cudaFree(dPath), "free path", MPI_COMM_WORLD);
    checkCuda(cudaFree(dPivot), "free pivot", MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const long long localMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long elapsedMs = 0;
    MPI_Reduce(&localMs, &elapsedMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        globalDist.resize(n * n);
    }
    MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED, globalDist.data(), counts.data(), displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %lld ms\nPerformance: %.3f GOPS\n", elapsedMs,
                    static_cast<double>(n) * n * n / (std::max(1LL, elapsedMs) / 1000.0) / 1e9);
        if (printResults) print_results_int(globalDist, "DistanceMatrix");
        bool valid = true;
        if (validate) {
            std::printf("Validating result...\n");
            valid = validateResult(globalDist, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        const int result = (validate && !valid) ? 1 : 0;
        MPI_Finalize();
        return result;
    }
    MPI_Finalize();
    return 0;
}
