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

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: update all local rows for one k iteration
__global__ void floydWarshallKernel(
    unsigned int* __restrict__ d_dist,
    unsigned int* __restrict__ d_path,
    const unsigned int* __restrict__ d_row_k,
    const int n, const unsigned int k, const int num_local_rows)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_i >= num_local_rows || j >= n) return;

    const unsigned int dist_ik = d_dist[local_i * n + k];
    const unsigned int dist_kj = d_row_k[j];
    const unsigned int new_dist = dist_ik + dist_kj;
    const int idx = local_i * n + j;

    if (new_dist < d_dist[idx]) {
        d_dist[idx] = new_dist;
        d_path[idx] = k;
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    bool valid = true;

    // 1. Diagonal should be zero
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            #pragma omp atomic write
            valid = false;
        }
    }
    if (!valid) return false;

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    #pragma omp parallel for collapse(2) schedule(dynamic)
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
                        #pragma omp atomic write
                        valid = false;
                    }
                }
            }
        }
    }

    return valid;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Assign GPU to this MPI rank
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d, OpenMP threads: %d, CUDA devices: %d\n",
               num_ranks, omp_get_max_threads(), num_devices);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Block row distribution across MPI ranks
    size_t base_rows = numNodes / num_ranks;
    size_t extra = numNodes % num_ranks;

    std::vector<int> row_counts(num_ranks);
    std::vector<int> row_offsets(num_ranks);
    for (int r = 0; r < num_ranks; r++) {
        row_counts[r] = (int)base_rows + (r < (int)extra ? 1 : 0);
        row_offsets[r] = (r == 0) ? 0 : row_offsets[r - 1] + row_counts[r - 1];
    }

    int my_row_start = row_offsets[rank];
    int my_num_rows = row_counts[rank];
    size_t local_size = (size_t)my_num_rows * numNodes;

    // All ranks generate the same initial distance matrix (deterministic seed)
    if (rank == 0) printf("Initializing graph...\n");

    std::vector<unsigned int> full_dist(numNodes * numNodes);
    initializeDistanceMatrix(full_dist, numNodes, 1, MAX_DISTANCE);

    // Extract local rows and initialize local path matrix with OpenMP
    std::vector<unsigned int> h_local_dist(local_size);
    std::vector<unsigned int> h_local_path(local_size);

    #pragma omp parallel for schedule(static)
    for (int li = 0; li < my_num_rows; li++) {
        int gi = my_row_start + li;
        for (size_t j = 0; j < numNodes; j++) {
            h_local_dist[(size_t)li * numNodes + j] = full_dist[(size_t)gi * numNodes + j];
            h_local_path[(size_t)li * numNodes + j] = (unsigned int)gi;
        }
    }

    full_dist.clear();
    full_dist.shrink_to_fit();

    // Allocate GPU memory and copy local data
    unsigned int *d_dist = nullptr, *d_path = nullptr, *d_row_k = nullptr;
    if (local_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_dist, local_size * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_path, local_size * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(d_dist, h_local_dist.data(),
                              local_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_path, h_local_path.data(),
                              local_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_row_k, numNodes * sizeof(unsigned int)));

    // Pinned host buffer for row k (faster H2D/D2H transfers)
    unsigned int* h_row_k;
    CUDA_CHECK(cudaMallocHost(&h_row_k, numNodes * sizeof(unsigned int)));

    // CUDA launch configuration
    dim3 blockDim(32, 8);
    dim3 gridDim(((int)numNodes + blockDim.x - 1) / blockDim.x,
                 (my_num_rows + blockDim.y - 1) / blockDim.y);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Main Floyd-Warshall loop
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t k = 0; k < numNodes; k++) {
        // Determine which rank owns row k
        int owner;
        if (k < extra * (base_rows + 1))
            owner = (int)(k / (base_rows + 1));
        else
            owner = (int)(extra + (k - extra * (base_rows + 1)) / base_rows);

        // Owner copies row k from GPU to pinned host buffer
        if (rank == owner) {
            int local_k = (int)k - my_row_start;
            CUDA_CHECK(cudaMemcpy(h_row_k, d_dist + (size_t)local_k * numNodes,
                                  numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }

        // Broadcast row k to all ranks
        MPI_Bcast(h_row_k, (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Async copy row k to GPU and launch kernel
        CUDA_CHECK(cudaMemcpyAsync(d_row_k, h_row_k,
                                    numNodes * sizeof(unsigned int),
                                    cudaMemcpyHostToDevice, stream));

        if (my_num_rows > 0) {
            floydWarshallKernel<<<gridDim, blockDim, 0, stream>>>(
                d_dist, d_path, d_row_k, (int)numNodes, (unsigned int)k, my_num_rows);
        }

        // Must synchronize before next iteration (row k+1 may be modified)
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG_INT,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Copy distance results back from GPU
    if (local_size > 0) {
        CUDA_CHECK(cudaMemcpy(h_local_dist.data(), d_dist,
                              local_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    // Gather all rows to rank 0
    std::vector<int> gather_counts(num_ranks), gather_displs(num_ranks);
    for (int r = 0; r < num_ranks; r++) {
        gather_counts[r] = row_counts[r] * (int)numNodes;
        gather_displs[r] = row_offsets[r] * (int)numNodes;
    }

    std::vector<unsigned int> gathered_dist;
    if (rank == 0) gathered_dist.resize(numNodes * numNodes);

    MPI_Gatherv(h_local_dist.data(), (int)local_size, MPI_UNSIGNED,
                rank == 0 ? gathered_dist.data() : nullptr,
                gather_counts.data(), gather_displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    // Print results for external validation (integer hash-based)
    if (rank == 0 && printResults) {
        print_results_int(gathered_dist, "DistanceMatrix");
    }

    // Validation
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(gathered_dist, numNodes);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            if (d_dist) CUDA_CHECK(cudaFree(d_dist));
            if (d_path) CUDA_CHECK(cudaFree(d_path));
            CUDA_CHECK(cudaFree(d_row_k));
            CUDA_CHECK(cudaFreeHost(h_row_k));
            CUDA_CHECK(cudaStreamDestroy(stream));
            MPI_Finalize();
            return 1;
        }
    }

    // Cleanup
    if (d_dist) CUDA_CHECK(cudaFree(d_dist));
    if (d_path) CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_row_k));
    CUDA_CHECK(cudaFreeHost(h_row_k));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Finalize();
    return 0;
}
