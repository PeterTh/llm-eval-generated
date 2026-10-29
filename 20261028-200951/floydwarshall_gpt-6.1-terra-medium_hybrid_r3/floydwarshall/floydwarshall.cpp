#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { int e = (call); if (e != MPI_SUCCESS) { \
    char s[MPI_MAX_ERROR_STRING]; int len; MPI_Error_string(e, s, &len); \
    fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, len, s); \
    MPI_Abort(MPI_COMM_WORLD, e); } } while (0)

// Each rank owns complete, contiguous source rows.  The pivot row is
// unchanged during its iteration because all diagonal entries are zero.
__global__ void update(unsigned int* dist, unsigned int* path,
                       const unsigned int* pivot, size_t n, size_t rows, size_t k) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < n) {
        size_t pos = i * n + j;
        unsigned int candidate = dist[i * n + k] + pivot[j];
        if (candidate < dist[pos]) { dist[pos] = candidate; path[pos] = (unsigned int)k; }
    }
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            unsigned long long value = strtoull(argv[++i], &end, 10);
            if (*argv[i] == '-' || *argv[i] == '\0' || *end || value == 0 || value > INT_MAX || value > (unsigned long long)INT_MAX / value) {
                if (!rank) fprintf(stderr, "Invalid node count: %s\n", argv[i]);
                MPI_Finalize(); return 1;
            }
            n = (size_t)value;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    MPI_Comm local;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
    int localRank, devices;
    MPI_CHECK(MPI_Comm_rank(local, &localRank));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_CHECK(MPI_Comm_free(&local));

    std::vector<int> counts(ranks), offsets(ranks), rowOffsets(ranks + 1);
    for (int r = 0; r < ranks; ++r) {
        size_t rows = n / ranks + ((size_t)r < n % ranks);
        rowOffsets[r + 1] = rowOffsets[r] + (int)rows;
        counts[r] = (int)(rows * n); offsets[r] = (int)((size_t)rowOffsets[r] * n);
    }
    size_t rows = rowOffsets[rank + 1] - rowOffsets[rank];
    size_t elements = rows * n;
    std::vector<unsigned int> full, host(elements), paths(elements);
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\nInitializing graph...\n", n, validate ? "enabled" : "disabled");
        full.resize(n * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i)
            full[i] = 1 + (unsigned int)(double(MAX_DISTANCE) * rand_r(&seed) / (double)RAND_MAX);
        for (size_t i = 0; i < n; ++i) full[i * n + i] = 0;
    }
    MPI_CHECK(MPI_Scatterv(full.data(), counts.data(), offsets.data(), MPI_UNSIGNED,
                           host.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < n; ++j) paths[i * n + j] = (unsigned int)j;

    unsigned int *dDist, *dPath, *dPivot, *pivot;
    size_t bytes = std::max<size_t>(elements, 1) * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&dDist, bytes)); CUDA_CHECK(cudaMalloc(&dPath, bytes));
    CUDA_CHECK(cudaMalloc(&dPivot, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(&pivot, n * sizeof(unsigned int)));
    if (elements) {
        CUDA_CHECK(cudaMemcpy(dDist, host.data(), elements * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, paths.data(), elements * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if (!rank) printf("Computing shortest paths...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();
    int owner = 0;
    dim3 block(32, 8), grid((unsigned)((n + 31) / 32), (unsigned)((rows + 7) / 8));
    for (size_t k = 0; k < n; ++k) {
        while (k >= (size_t)rowOffsets[owner + 1]) ++owner;
        if (rank == owner)
            CUDA_CHECK(cudaMemcpy(pivot, dDist + (k - rowOffsets[rank]) * n, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        MPI_CHECK(MPI_Bcast(pivot, (int)n, MPI_UNSIGNED, owner, MPI_COMM_WORLD));
        CUDA_CHECK(cudaMemcpy(dPivot, pivot, n * sizeof(unsigned int), cudaMemcpyHostToDevice));
        if (rows) { update<<<grid, block>>>(dDist, dPath, dPivot, n, rows, k); CUDA_CHECK(cudaGetLastError()); }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    if (elements) CUDA_CHECK(cudaMemcpy(host.data(), dDist, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (validate || printResults)
        MPI_CHECK(MPI_Gatherv(host.data(), counts[rank], MPI_UNSIGNED, full.data(), counts.data(), offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    int valid = 1;
    if (!rank) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", (long)duration.count());
        double seconds = std::chrono::duration<double>(end - start).count();
        printf("Performance: %.3f GOPS\n", (double)n * n * n / seconds / 1e9);
        if (printResults) print_results_int(full, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            #pragma omp parallel for reduction(&:valid) schedule(static)
            for (size_t i = 0; i < n; ++i) if (full[i * n + i] != 0) valid = 0;
            #pragma omp parallel for reduction(&:valid) schedule(static)
            for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
                for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
                    for (size_t k = 0; k < n; ++k) {
                        unsigned int a = full[i * n + k], b = full[k * n + j];
                        if (a < INF && b < INF && a + b < full[i * n + j]) valid = 0;
                    }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CUDA_CHECK(cudaFreeHost(pivot)); CUDA_CHECK(cudaFree(dPivot)); CUDA_CHECK(cudaFree(dPath)); CUDA_CHECK(cudaFree(dDist));
    MPI_CHECK(MPI_Finalize());
    return valid ? 0 : 1;
}
