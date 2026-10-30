#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

__host__ __device__ inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

__global__ void fw_kernel(unsigned int* dist, unsigned int* path, size_t n, size_t k) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    const unsigned int distIK = dist[idx2(k, i, n)];
    const unsigned int distKJ = dist[idx2(j, k, n)];
    if (distIK >= INF || distKJ >= INF) return;
    const unsigned int newDist = distIK + distKJ;
    unsigned int* p = &dist[idx2(j, i, n)];
    if (newDist < *p) {
        *p = newDist;
        path[idx2(j, i, n)] = (unsigned int)k;
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    unsigned int *d_dist, *d_path;
    size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    cudaMalloc(&d_dist, bytes);
    cudaMalloc(&d_path, bytes);
    cudaMemcpy(d_dist, dist.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_path, path.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((numNodes + 15) / 16, (numNodes + 15) / 16);
    for (size_t k = 0; k < numNodes; ++k) {
        fw_kernel<<<grid, block>>>(d_dist, d_path, numNodes, k);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(dist.data(), d_dist, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(path.data(), d_path, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dist);
    cudaFree(d_path);
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            }
        }
    }
    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    int v = validate, r = printResults;
    MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&r, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = v != 0; printResults = r != 0;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    if (rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    floydWarshall(dist, path, numNodes);
    auto end = std::chrono::high_resolution_clock::now();
    MPI_Bcast(dist.data(), (int)(numNodes*numNodes), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), (int)(numNodes*numNodes), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);
            if (valid) printf("Validation: PASSED\n");
            else printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();
    return 0;
}
