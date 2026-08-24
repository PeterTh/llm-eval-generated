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

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] void fail(const char* what, int rank) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, what);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(cudaError_t status, const char* what, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s: %s\n", rank, what, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

void checkMpi(int status, const char* what, int rank) {
    if (status != MPI_SUCCESS) {
        char error[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, error, &length);
        std::fprintf(stderr, "Rank %d: %s: %.*s\n", rank, what, length, error);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

// The local representation is source-major: local[source * n + destination].
// This makes a source-row shard contiguous on the GPU; conversion is done only at I/O.
__global__ void relaxPivot(unsigned int* __restrict__ dist,
                           unsigned int* __restrict__ path,
                           const unsigned int* __restrict__ pivot,
                           size_t localRows, size_t n, size_t k) {
    const size_t destination = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t source = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (source >= localRows || destination >= n) return;

    const size_t offset = source * n + destination;
    const unsigned int candidate = dist[source * n + k] + pivot[destination];
    if (candidate < dist[offset]) {
        dist[offset] = candidate;
        path[offset] = static_cast<unsigned int>(k);
    }
}

void initializeGlobal(std::vector<unsigned int>& dist, size_t n) {
    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE);
    for (size_t offset = 0; offset < n * n; ++offset) {
        dist[offset] = 1U + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i)
        dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), n)] = 0;
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    bool valid = true;
#pragma omp parallel for schedule(static) reduction(&:valid)
    for (ptrdiff_t ii = 0; ii < static_cast<ptrdiff_t>(n); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        if (dist[idx2(i, i, n)] != 0) valid = false;
    }
    if (!valid) return false;

    const size_t sample = std::min(n, size_t{10});
#pragma omp parallel for collapse(2) schedule(static) reduction(&:valid)
    for (ptrdiff_t ii = 0; ii < static_cast<ptrdiff_t>(sample); ++ii) {
        for (ptrdiff_t jj = 0; jj < static_cast<ptrdiff_t>(sample); ++jj) {
            const size_t i = static_cast<size_t>(ii), j = static_cast<size_t>(jj);
            for (size_t k = 0; k < n; ++k) {
                const unsigned int dik = dist[idx2(k, i, n)];
                const unsigned int dkj = dist[idx2(j, k, n)];
                if (dik < INF && dkj < INF && dik + dkj < dist[idx2(j, i, n)]) valid = false;
            }
        }
    }
    return valid;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -n <num>     Number of nodes (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results\n  -h           Show this help\n");
}

int main(int argc, char** argv) {
    checkMpi(MPI_Init(&argc, &argv), "MPI_Init", 0);
    int rank = 0, ranks = 1;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank", rank);
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size", rank);

    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max())) fail("node count must be in 1..INT_MAX", rank);

    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount", rank);
    if (devices == 0) fail("no CUDA device available", rank);
    checkCuda(cudaSetDevice(rank % devices), "cudaSetDevice", rank);

    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    const size_t localRows = base + (static_cast<size_t>(rank) < extra);
    const size_t firstRow = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    if (localRows == 0) fail("number of MPI ranks exceeds node count", rank);

    std::vector<unsigned int> localDist(localRows * n), localPath(localRows * n), pivot(n);
    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        globalDist.resize(n * n);
        initializeGlobal(globalDist, n);
    }

    // Scatter source rows explicitly because the legacy public layout is transposed.
    if (rank == 0) {
        for (int target = 0; target < ranks; ++target) {
            const size_t rows = base + (static_cast<size_t>(target) < extra);
            const size_t begin = static_cast<size_t>(target) * base + std::min(static_cast<size_t>(target), extra);
            std::vector<unsigned int> packed(rows * n);
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(rows); ++i)
                for (size_t j = 0; j < n; ++j) packed[static_cast<size_t>(i) * n + j] = globalDist[idx2(j, begin + static_cast<size_t>(i), n)];
            if (target == 0) localDist.swap(packed);
            else checkMpi(MPI_Send(packed.data(), static_cast<int>(packed.size()), MPI_UNSIGNED, target, 0, MPI_COMM_WORLD), "MPI_Send initial rows", rank);
        }
    } else checkMpi(MPI_Recv(localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Recv initial rows", rank);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(localRows); ++i)
        for (size_t j = 0; j < n; ++j) localPath[static_cast<size_t>(i) * n + j] = static_cast<unsigned int>(firstRow + static_cast<size_t>(i));

    unsigned int *deviceDist = nullptr, *devicePath = nullptr, *devicePivot = nullptr;
    checkCuda(cudaMalloc(&deviceDist, localDist.size() * sizeof(unsigned int)), "cudaMalloc dist", rank);
    checkCuda(cudaMalloc(&devicePath, localPath.size() * sizeof(unsigned int)), "cudaMalloc path", rank);
    checkCuda(cudaMalloc(&devicePivot, n * sizeof(unsigned int)), "cudaMalloc pivot", rank);
    checkCuda(cudaMemcpy(deviceDist, localDist.data(), localDist.size() * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy dist to device", rank);
    checkCuda(cudaMemcpy(devicePath, localPath.data(), localPath.size() * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy path to device", rank);

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nMPI ranks: %d, CUDA devices/rank host: %d, OpenMP threads: %d\nValidation: %s\nComputing shortest paths...\n", n, ranks, devices, omp_get_max_threads(), validate ? "enabled" : "disabled");
    }
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier", rank);
    const auto start = std::chrono::steady_clock::now();
    const dim3 block(32, 8);
    const dim3 grid((n + block.x - 1) / block.x, (localRows + block.y - 1) / block.y);
    for (size_t k = 0; k < n; ++k) {
        const size_t largeRows = extra * (base + 1);
        const int owner = static_cast<int>(k < largeRows ? k / (base + 1) : extra + (k - largeRows) / base);
        if (rank == owner) {
            const size_t ownerFirst = static_cast<size_t>(owner) * base + std::min(static_cast<size_t>(owner), extra);
            checkCuda(cudaMemcpy(pivot.data(), deviceDist + (k - ownerFirst) * n, n * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy pivot from device", rank);
        }
        checkMpi(MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD), "MPI_Bcast pivot", rank);
        checkCuda(cudaMemcpy(devicePivot, pivot.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy pivot to device", rank);
        relaxPivot<<<grid, block>>>(deviceDist, devicePath, devicePivot, localRows, n, k);
        checkCuda(cudaGetLastError(), "launch relaxPivot", rank);
    }
    checkCuda(cudaDeviceSynchronize(), "synchronize computation", rank);
    const auto end = std::chrono::steady_clock::now();
    const long long localMilliseconds = static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    long long milliseconds = 0;
    checkMpi(MPI_Reduce(&localMilliseconds, &milliseconds, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD), "MPI_Reduce computation time", rank);

    checkCuda(cudaMemcpy(localDist.data(), deviceDist, localDist.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy result from device", rank);
    checkCuda(cudaFree(devicePivot), "free pivot", rank);
    checkCuda(cudaFree(devicePath), "free path", rank);
    checkCuda(cudaFree(deviceDist), "free dist", rank);

    int exitCode = 0;
    if (rank == 0) {
        for (size_t i = 0; i < localRows; ++i)
            for (size_t j = 0; j < n; ++j) globalDist[idx2(j, i, n)] = localDist[i * n + j];
        for (int source = 1; source < ranks; ++source) {
            const size_t rows = base + (static_cast<size_t>(source) < extra);
            const size_t begin = static_cast<size_t>(source) * base + std::min(static_cast<size_t>(source), extra);
            std::vector<unsigned int> packed(rows * n);
            checkMpi(MPI_Recv(packed.data(), static_cast<int>(packed.size()), MPI_UNSIGNED, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Recv result rows", rank);
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(rows); ++i)
                for (size_t j = 0; j < n; ++j) globalDist[idx2(j, begin + static_cast<size_t>(i), n)] = packed[static_cast<size_t>(i) * n + j];
        }
        std::printf("Computation time: %lld ms\nPerformance: %.3f GOPS\n", static_cast<long long>(milliseconds), static_cast<double>(n) * n * n / (static_cast<double>(milliseconds) / 1000.0) / 1e9);
        if (printResults) print_results_int(globalDist, "DistanceMatrix");
        if (validate) {
            const bool valid = validateResult(globalDist, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    } else checkMpi(MPI_Send(localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD), "MPI_Send result rows", rank);

    checkMpi(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast exit code", rank);
    MPI_Finalize();
    return exitCode;
}
