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

constexpr unsigned int INF = 1000000000u;
constexpr unsigned int MAX_DISTANCE = 200u;

// Index calculation for flattened 2D array (row-major: idx = i * n + j)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return i * n + j;
}

// CUDA error check
#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); MPI_Abort(MPI_COMM_WORLD, -1); } } while(0)

// Kernel: update local rows [startRow, startRow+numRows)
__global__ void fw_update(unsigned int* dist, const unsigned int* row_k, size_t n, size_t startRow, size_t numRows, size_t k) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = numRows * n;
    if (tid >= total) return;
    size_t local_i = tid / n;
    size_t j = tid % n;
    size_t i = startRow + local_i;
    size_t idx = i * n + j;

    unsigned int dik = dist[i * n + k];
    unsigned int dkj = row_k[j];
    if (dik < INF && dkj < INF) {
        unsigned int newDist = dik + dkj;
        if (newDist < dist[idx]) {
            dist[idx] = newDist;
        }
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42u;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    size_t total = numNodes * numNodes;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; ++i) {
        unsigned int s = seed + (unsigned int)(i);
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&s) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            path[idx2(i, j, numNodes)] = j;
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            if (MPI::COMM_WORLD.Get_rank() == 0)
                printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality: sample check accelerated with OpenMP
    size_t sampleN = std::min(numNodes, static_cast<size_t>(10));
    for (size_t i = 0; i < sampleN; ++i) {
        for (size_t j = 0; j < sampleN; ++j) {
            #pragma omp parallel for schedule(static)
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(i, j, numNodes)];
                const unsigned int distIK = dist[idx2(i, k, numNodes)];
                const unsigned int distKJ = dist[idx2(k, j, numNodes)];

                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        if (MPI::COMM_WORLD.Get_rank() == 0)
                            printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                        // early exit (note: races possible but acceptable for validation)
                        exit(1);
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 prints help)
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Floyd-Warshall Hybrid (MPI+OpenMP+CUDA) Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Ranks: %d, Threads: %d\n", size, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate host matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1u, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    // Prepare MPI distribution of rows
    std::vector<int> counts(size), displs(size);
    size_t base = numNodes / size;
    int rem = numNodes % size;
    for (int r = 0; r < size; ++r) {
        counts[r] = static_cast<int>(base + (r < rem ? 1 : 0));
        displs[r] = (r == 0) ? 0 : displs[r-1] + counts[r-1];
    }
    size_t localRows = counts[rank];
    size_t startRow = displs[rank];

    // Initialize CUDA device
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    int device = rank % (deviceCount ? deviceCount : 1);
    if (deviceCount > 0) CUDA_CHECK(cudaSetDevice(device));

    // Allocate device memory for entire matrix and row buffer
    unsigned int* dist_d = nullptr;
    unsigned int* rowk_d = nullptr;
    size_t matrixBytes = numNodes * numNodes * sizeof(unsigned int);
    size_t rowBytes = numNodes * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc(&dist_d, matrixBytes));
    CUDA_CHECK(cudaMalloc(&rowk_d, rowBytes));
    CUDA_CHECK(cudaMemcpy(dist_d, dist.data(), matrixBytes, cudaMemcpyHostToDevice));

    std::vector<unsigned int> rowk(numNodes);

    if (rank == 0) printf("Computing shortest paths (hybrid)...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Main k loop with MPI row broadcasts and CUDA updates for local rows
    for (size_t k = 0; k < numNodes; ++k) {
        // Find owner of row k
        int owner = 0;
        for (int r = 0; r < size; ++r) {
            if (k >= static_cast<size_t>(displs[r]) && k < static_cast<size_t>(displs[r] + counts[r])) { owner = r; break; }
        }

        if (rank == owner) {
            // copy row k from device to host
            unsigned int* src = dist_d + (k * numNodes);
            CUDA_CHECK(cudaMemcpy(rowk.data(), src, rowBytes, cudaMemcpyDeviceToHost));
        }

        // Broadcast rowk from owner to all ranks
        MPI_Bcast(rowk.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // copy rowk to device
        CUDA_CHECK(cudaMemcpy(rowk_d, rowk.data(), rowBytes, cudaMemcpyHostToDevice));

        // Launch CUDA kernel to update local rows
        size_t total = localRows * numNodes;
        if (total > 0) {
            const int block = 256;
            int grid = static_cast<int>((total + block - 1) / block);
            fw_update<<<grid, block>>>(dist_d, rowk_d, numNodes, startRow, localRows, k);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(dist.data(), dist_d, matrixBytes, cudaMemcpyDeviceToHost));

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) print_results_int(dist, "DistanceMatrix");
    }

    // Validation (only rank 0 performs final validation)
    if (validate) {
        bool valid = true;
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(dist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        // Ensure all ranks exit with same code
        int code = valid ? 0 : 1;
        MPI_Bcast(&code, 1, MPI_INT, 0, MPI_COMM_WORLD);
        CUDA_CHECK(cudaFree(dist_d));
        CUDA_CHECK(cudaFree(rowk_d));
        MPI_Finalize();
        return code;
    }

    CUDA_CHECK(cudaFree(dist_d));
    CUDA_CHECK(cudaFree(rowk_d));

    MPI_Finalize();
    return 0;
}
