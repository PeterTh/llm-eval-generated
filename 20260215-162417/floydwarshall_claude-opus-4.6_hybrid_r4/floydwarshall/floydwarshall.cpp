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
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: each thread updates one (local_i, j) element for a given k
__global__ void fw_kernel(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          const unsigned int* __restrict__ row_k,
                          const int n, const int k, const int local_rows) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_i < local_rows && j < n) {
        unsigned int dik = dist[local_i * n + k];
        unsigned int dkj = __ldg(&row_k[j]);
        unsigned int newDist = dik + dkj;

        if (newDist < dist[local_i * n + j]) {
            dist[local_i * n + j] = newDist;
            path[local_i * n + j] = static_cast<unsigned int>(k);
        }
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

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality (sampled, parallelized with OpenMP)
    bool valid = true;
    #pragma omp parallel for collapse(2) reduction(&& : valid)
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        valid = false;
                    }
                }
            }
        }
    }

    if (!valid) {
        printf("Validation failed: triangle inequality violated\n");
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
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

    const int N = static_cast<int>(numNodes);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d, OpenMP threads: %d\n", nprocs, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribute rows across MPI ranks
    const int base_rows = N / nprocs;
    const int extra = N % nprocs;

    std::vector<int> all_rows(nprocs), all_offsets(nprocs);
    std::vector<int> send_counts(nprocs), send_displs(nprocs);

    for (int r = 0; r < nprocs; r++) {
        all_rows[r] = base_rows + (r < extra ? 1 : 0);
        all_offsets[r] = (r > 0) ? (all_offsets[r - 1] + all_rows[r - 1]) : 0;
        send_counts[r] = all_rows[r] * N;
        send_displs[r] = all_offsets[r] * N;
    }

    const int my_rows = all_rows[rank];

    // Precompute k -> owner rank mapping
    std::vector<int> k_owner(N);
    for (int r = 0; r < nprocs; r++) {
        for (int i = 0; i < all_rows[r]; i++) {
            k_owner[all_offsets[r] + i] = r;
        }
    }

    // Initialize full matrices on rank 0
    std::vector<unsigned int> dist, path;
    if (rank == 0) {
        dist.resize(static_cast<size_t>(N) * N);
        path.resize(static_cast<size_t>(N) * N);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    // Local host buffers for this rank's row block
    std::vector<unsigned int> local_dist(static_cast<size_t>(my_rows) * N);
    std::vector<unsigned int> local_path(static_cast<size_t>(my_rows) * N);

    // Scatter matrices from rank 0 to all ranks
    MPI_Scatterv(rank == 0 ? dist.data() : nullptr,
                 send_counts.data(), send_displs.data(), MPI_UNSIGNED,
                 local_dist.data(), my_rows * N, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? path.data() : nullptr,
                 send_counts.data(), send_displs.data(), MPI_UNSIGNED,
                 local_path.data(), my_rows * N, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // CUDA setup - assign GPU based on rank
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

    const size_t local_bytes = static_cast<size_t>(my_rows) * N * sizeof(unsigned int);
    const size_t row_bytes = static_cast<size_t>(N) * sizeof(unsigned int);

    unsigned int *d_dist = nullptr, *d_path = nullptr, *d_row_k = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, local_bytes));
    CUDA_CHECK(cudaMalloc(&d_path, local_bytes));
    CUDA_CHECK(cudaMalloc(&d_row_k, row_bytes));

    CUDA_CHECK(cudaMemcpy(d_dist, local_dist.data(), local_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, local_path.data(), local_bytes, cudaMemcpyHostToDevice));

    // Pinned host memory for fast row transfers
    unsigned int* h_row_k = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_row_k, row_bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Kernel launch configuration
    dim3 block(32, 8);
    dim3 grid((N + block.x - 1) / block.x,
              (my_rows + block.y - 1) / block.y);

    if (rank == 0) printf("Computing shortest paths...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < N; k++) {
        const int owner = k_owner[k];

        // Owner extracts row k from GPU
        if (rank == owner) {
            const int local_k = k - all_offsets[owner];
            CUDA_CHECK(cudaMemcpyAsync(h_row_k, d_dist + local_k * N,
                                        row_bytes, cudaMemcpyDeviceToHost, stream));
        }

        // Synchronize stream: ensures previous kernel and D2H transfer are complete
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Broadcast row k from owner to all ranks
        MPI_Bcast(h_row_k, N, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Copy row k to GPU and launch update kernel
        CUDA_CHECK(cudaMemcpyAsync(d_row_k, h_row_k, row_bytes,
                                    cudaMemcpyHostToDevice, stream));

        if (my_rows > 0) {
            fw_kernel<<<grid, block, 0, stream>>>(d_dist, d_path, d_row_k, N, k, my_rows);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Copy results back from GPU
    CUDA_CHECK(cudaMemcpy(local_dist.data(), d_dist, local_bytes, cudaMemcpyDeviceToHost));

    // Gather results on rank 0
    MPI_Gatherv(local_dist.data(), my_rows * N, MPI_UNSIGNED,
                rank == 0 ? dist.data() : nullptr,
                send_counts.data(), send_displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Cleanup CUDA resources
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(h_row_k));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_row_k));

    int ret = 0;
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());

        // Floyd-Warshall has O(n^3) complexity
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }

    MPI_Finalize();
    return ret;
}
