#include <algorithm>
#include <chrono>
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

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

__global__ void fw_step(unsigned int* dist, unsigned int* path, size_t n, size_t k,
                        size_t firstRow, size_t rows) {
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = rows * n;
    if (p >= count) return;
    size_t i = firstRow + p / n;
    size_t j = p % n;
    size_t ij = j * n + i;
    unsigned int candidate = dist[k * n + i] + dist[j * n + k];
    if (candidate < dist[ij]) {
        dist[ij] = candidate;
        path[ij] = static_cast<unsigned int>(k);
    }
}

static void initialize(std::vector<unsigned int>& d, std::vector<unsigned int>& p, size_t n) {
    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE);
    #pragma omp parallel for schedule(static)
    for (long long x = 0; x < static_cast<long long>(n * n); ++x) {
        // The original deterministic stream is retained by generating it below on one thread.
        (void)x;
    }
    for (size_t x = 0; x < n * n; ++x)
        d[x] = 1 + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        d[idx2(i, i, n)] = 0;
        for (size_t j = 0; j < n; ++j) {
            p[idx2(static_cast<size_t>(i), j, n)] = static_cast<unsigned int>(j);
            p[idx2(j, static_cast<size_t>(i), n)] = static_cast<unsigned int>(i);
        }
    }
}

static bool validate(const std::vector<unsigned int>& d, size_t n) {
    for (size_t i = 0; i < n; ++i) if (d[idx2(i, i, n)] != 0) return false;
    bool ok = true;
    #pragma omp parallel for collapse(2) reduction(&:ok) schedule(static)
    for (int i = 0; i < static_cast<int>(std::min(n, size_t(10))); ++i)
        for (int j = 0; j < static_cast<int>(std::min(n, size_t(10))); ++j)
            for (size_t k = 0; k < n; ++k)
                if (d[idx2(k, i, n)] < INF && d[idx2(j, k, n)] < INF &&
                    d[idx2(k, i, n)] + d[idx2(j, k, n)] < d[idx2(j, i, n)]) ok = false;
    return ok;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512; bool doValidate = false, printResults = false;
    for (int a = 1; a < argc; ++a) {
        if (!strcmp(argv[a], "-n") && a + 1 < argc) n = strtoull(argv[++a], nullptr, 10);
        else if (!strcmp(argv[a], "-v")) doValidate = true;
        else if (!strcmp(argv[a], "-r")) printResults = true;
        else if (!strcmp(argv[a], "-h")) { if (!rank) printf("Usage: %s [-n nodes] [-v] [-r]\n", argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) printf("Unknown option: %s\n", argv[a]); MPI_Finalize(); return 1; }
    }
    std::vector<unsigned int> d(n*n), path(n*n);
    if (!rank) initialize(d, path, n);
    MPI_Bcast(d.data(), static_cast<int>(d.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), static_cast<int>(path.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) { size_t b=n*r/ranks, e=n*(r+1)/ranks; counts[r]=static_cast<int>((e-b)*n); displs[r]=static_cast<int>(b*n); }
    size_t first = n*rank/ranks, rows = n*(rank+1)/ranks-first;
    unsigned int *dd=nullptr, *pp=nullptr;
    cudaMalloc(&dd, d.size()*sizeof(unsigned int)); cudaMalloc(&pp, path.size()*sizeof(unsigned int));
    cudaMemcpy(dd,d.data(),d.size()*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(pp,path.data(),path.size()*sizeof(unsigned int),cudaMemcpyHostToDevice);
    MPI_Barrier(MPI_COMM_WORLD); double start=MPI_Wtime();
    for (size_t k=0; k<n; ++k) {
        size_t work=rows*n; fw_step<<<(work+255)/256,256>>>(dd,pp,n,k,first,rows);
        cudaDeviceSynchronize();
        cudaMemcpy(d.data()+first*n,dd+first*n,work*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        cudaMemcpy(path.data()+first*n,pp+first*n,work*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,d.data(),counts.data(),displs.data(),MPI_UNSIGNED,MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,path.data(),counts.data(),displs.data(),MPI_UNSIGNED,MPI_COMM_WORLD);
        cudaMemcpy(dd,d.data(),d.size()*sizeof(unsigned int),cudaMemcpyHostToDevice);
        cudaMemcpy(pp,path.data(),path.size()*sizeof(unsigned int),cudaMemcpyHostToDevice);
    }
    double elapsed=MPI_Wtime()-start; cudaFree(dd); cudaFree(pp);
    if (!rank) { printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nComputation time: %.3f ms\nPerformance: %.3f GOPS\n",n,elapsed*1000,(double)n*n*n/elapsed/1e9); if(printResults) print_results_int(d,"DistanceMatrix"); if(doValidate) printf("Validation: %s\n",validate(d,n)?"PASSED":"FAILED"); }
    MPI_Finalize(); return 0;
}
