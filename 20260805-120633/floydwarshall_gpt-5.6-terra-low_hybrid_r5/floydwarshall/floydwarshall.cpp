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

// The externally visible format used by the original benchmark is column major.
inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

// localDist is conventional row-major: local source row i, destination j.
// A full pivot row is replicated on every GPU for the duration of one k step.
__global__ void relaxPivot(unsigned int* localDist, unsigned int* localPath,
                           const unsigned int* pivot, size_t rows, size_t n,
                           size_t k) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t i = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (i >= rows || j >= n) return;
    const size_t pos = i * n + j;
    const unsigned int throughK = localDist[i * n + k] + pivot[j];
    if (throughK < localDist[pos]) {
        localDist[pos] = throughK;
        localPath[pos] = static_cast<unsigned int>(k);
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              unsigned int rangeMin, unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t p = 0; p < n * n; ++p)
        dist[p] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    for (size_t i = 0; i < n; ++i) dist[idx2(i, i, n)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, size_t n) {
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            path[idx2(j, i, n)] = i; // equivalent final value of the original initialization
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) if (dist[idx2(i, i, n)] != 0) return false;
    const size_t sample = std::min(n, size_t{10});
    for (size_t i = 0; i < sample; ++i)
        for (size_t j = 0; j < sample; ++j)
            for (size_t k = 0; k < n; ++k)
                if (dist[idx2(k, i, n)] < INF && dist[idx2(j, k, n)] < INF &&
                    dist[idx2(k, i, n)] + dist[idx2(j, k, n)] < dist[idx2(j, i, n)]) return false;
    return true;
}

static int ownerOf(size_t k, size_t n, int ranks) {
    const size_t base = n / static_cast<size_t>(ranks), extra = n % static_cast<size_t>(ranks);
    const size_t large = (base + 1) * extra;
    return k < large ? static_cast<int>(k / (base + 1))
                     : static_cast<int>(extra + (k - large) / base);
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num>     Number of nodes (default: 512)\n  -v           Enable validation\n  -r           Print results\n  -h           Show help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (!n || n > static_cast<size_t>(std::numeric_limits<int>::max())) { if (!rank) std::fprintf(stderr, "Invalid node count\n"); MPI_Finalize(); return 1; }

    int devices = 0; cudaCheck(cudaGetDeviceCount(&devices), "querying CUDA devices");
    if (!devices) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(rank % devices), "selecting CUDA device");

    const size_t base = n / ranks, extra = n % ranks;
    const size_t localRows = base + (static_cast<size_t>(rank) < extra);
    const size_t firstRow = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) { const size_t rows = base + (static_cast<size_t>(r) < extra); counts[r] = static_cast<int>(rows * n); displs[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra)) * n); }

    std::vector<unsigned int> canonicalDist, canonicalPath, packedDist, packedPath;
    int exitCode = 0;
    if (!rank) {
        canonicalDist.resize(n * n); canonicalPath.resize(n * n);
        initializeDistanceMatrix(canonicalDist, n, 1, MAX_DISTANCE); initializePathMatrix(canonicalPath, n);
        packedDist.resize(n * n); packedPath.resize(n * n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) { packedDist[i*n+j] = canonicalDist[idx2(j,i,n)]; packedPath[i*n+j] = canonicalPath[idx2(j,i,n)]; }
    }
    std::vector<unsigned int> localDist(localRows * n), localPath(localRows * n), pivot(n);
    MPI_Scatterv(rank ? nullptr : packedDist.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localRows*n), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank ? nullptr : packedPath.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                 localPath.data(), static_cast<int>(localRows*n), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    unsigned int *dDist, *dPath, *dPivot;
    const size_t allocation = std::max<size_t>(1, localRows * n);
    cudaCheck(cudaMalloc(&dDist, allocation * sizeof(*dDist)), "allocating distance matrix");
    cudaCheck(cudaMalloc(&dPath, allocation * sizeof(*dPath)), "allocating path matrix");
    cudaCheck(cudaMalloc(&dPivot, n * sizeof(*dPivot)), "allocating pivot row");
    cudaCheck(cudaMemcpy(dDist, localDist.data(), localRows*n*sizeof(*dDist), cudaMemcpyHostToDevice), "uploading distances");
    cudaCheck(cudaMemcpy(dPath, localPath.data(), localRows*n*sizeof(*dPath), cudaMemcpyHostToDevice), "uploading paths");

    if (!rank) std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nMPI ranks: %d; CUDA GPUs/rank: 1; OpenMP threads: %d\nComputing shortest paths...\n", n, ranks, omp_get_max_threads());
    MPI_Barrier(MPI_COMM_WORLD); const double start = MPI_Wtime();
    const dim3 block(32, 8), grid((n + block.x - 1) / block.x, (localRows + block.y - 1) / block.y);
    for (size_t k = 0; k < n; ++k) {
        const int owner = ownerOf(k, n, ranks);
        // Only the owner transfers the newly computed pivot row to its host.
        // This avoids a full device/host matrix round trip on every k iteration.
        if (rank == owner)
            cudaCheck(cudaMemcpy(pivot.data(), dDist + (k - firstRow) * n,
                                 n * sizeof(unsigned int), cudaMemcpyDeviceToHost),
                      "downloading pivot row");
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(dPivot, pivot.data(), n*sizeof(*dPivot), cudaMemcpyHostToDevice), "uploading pivot row");
        if (localRows) {
            relaxPivot<<<grid, block>>>(dDist, dPath, dPivot, localRows, n, k);
            cudaCheck(cudaGetLastError(), "launching Floyd-Warshall kernel");
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "synchronizing final computation");
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(localDist.data(), dDist, localRows*n*sizeof(*dDist), cudaMemcpyDeviceToHost), "downloading final distances");
    MPI_Gatherv(localDist.data(), static_cast<int>(localRows*n), MPI_UNSIGNED, rank ? nullptr : packedDist.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (!rank) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) canonicalDist[idx2(j,i,n)] = packedDist[i*n+j];
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GOPS\n", maxElapsed*1000.0, (double(n)*n*n)/maxElapsed/1e9);
        if (printResults) print_results_int(canonicalDist, "DistanceMatrix");
        if (validate) {
            const bool valid = validateResult(canonicalDist, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    cudaFree(dPivot); cudaFree(dPath); cudaFree(dDist);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
