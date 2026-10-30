#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

__global__ void fw_kernel(unsigned int* dist, unsigned int* path, size_t n, size_t k, size_t i_start, size_t i_end) {
    size_t i = i_start + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= i_end) return;
    for (size_t j = 0; j < n; ++j) {
        unsigned int distIJ = dist[j * n + i];
        unsigned int distIK = dist[k * n + i];
        unsigned int distKJ = dist[j * n + k];
        unsigned int newDist = distIK + distKJ;
        if (newDist < distIJ) {
            dist[j * n + i] = newDist;
            path[j * n + i] = (unsigned int)k;
        }
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    #pragma omp parallel for
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        unsigned int s = seed;
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&s) / (double)RAND_MAX);
    }
    // deterministic recompute to match original
    unsigned int seed2 = 42;
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed2) / (double)RAND_MAX);
    }
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    #pragma omp parallel for collapse(2)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = (unsigned int)j;
            path[idx2(j, i, numNodes)] = (unsigned int)i;
        }
    }
    for (size_t j = 0; j < numNodes; ++j) {
        path[idx2(j, j, numNodes)] = (unsigned int)j;
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t chunk = numNodes / size;
    size_t start = rank * chunk;
    size_t end = (rank == size - 1) ? numNodes : start + chunk;

    // device buffers
    unsigned int *d_dist, *d_path;
    cudaMalloc(&d_dist, numNodes * numNodes * sizeof(unsigned int));
    cudaMalloc(&d_path, numNodes * numNodes * sizeof(unsigned int));
    cudaMemcpy(d_dist, dist.data(), numNodes * numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_path, path.data(), numNodes * numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);

    for (size_t k = 0; k < numNodes; ++k) {
        // broadcast row k not needed since all have full device copy updated via gather
        int threads = 256;
        int blocks = (int)((end - start + threads - 1) / threads);
        fw_kernel<<<blocks, threads>>>(d_dist, d_path, numNodes, k, start, end);
        cudaDeviceSynchronize();

        // gather updated rows back to host for this rank
        cudaMemcpy(dist.data(), d_dist, numNodes * numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(path.data(), d_path, numNodes * numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        // sync across ranks: each rank updated subset; combine by replacing rows
        // All ranks have full arrays; use Allgatherv on row chunks? Simpler: Allreduce max? Values are mins; use MPI_MIN for dist
        // But path must follow dist; instead gather rows from each rank.
        // We'll do MPI_Allgatherv on dist and path row-chunks.
        // Build counts/displs by rows (each row n ints)
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            size_t s = r * chunk;
            size_t e = (r == size - 1) ? numNodes : s + chunk;
            counts[r] = (int)((e - s) * numNodes);
            displs[r] = (int)(s * numNodes);
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       dist.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       path.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                       MPI_COMM_WORLD);

        // refresh device
        cudaMemcpy(d_dist, dist.data(), numNodes * numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_path, path.data(), numNodes * numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }

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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

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

    // ensure all ranks same init
    MPI_Bcast(dist.data(), (int)(numNodes*numNodes), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), (int)(numNodes*numNodes), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    floydWarshall(dist, path, numNodes);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);
            if (valid) { printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); }
        }
    }

    MPI_Finalize();
    return 0;
}
