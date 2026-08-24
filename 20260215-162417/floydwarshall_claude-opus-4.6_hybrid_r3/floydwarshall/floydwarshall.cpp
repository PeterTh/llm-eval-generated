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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Column-major index: column j starts at offset j*n
inline size_t idx2(size_t i, size_t j, size_t n) {
    return j * n + i;
}

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;

// CUDA kernel: updates local columns for a given intermediate node k.
// Uses shared memory to reduce redundant global reads.
__global__ __launch_bounds__(BLOCK_X * BLOCK_Y)
void fw_kernel(unsigned int* __restrict__ d_dist,
               unsigned int* __restrict__ d_path,
               const unsigned int* __restrict__ d_col_k,
               int N, int k, int num_local_cols) {
    __shared__ unsigned int s_col_k[BLOCK_X];
    __shared__ unsigned int s_distIK[BLOCK_Y];

    int j  = blockIdx.x * BLOCK_X + threadIdx.x;
    int lc = blockIdx.y * BLOCK_Y + threadIdx.y;

    if (threadIdx.y == 0 && j < N)
        s_col_k[threadIdx.x] = d_col_k[j];
    if (threadIdx.x == 0 && lc < num_local_cols)
        s_distIK[threadIdx.y] = d_dist[lc * N + k];
    __syncthreads();

    if (j >= N || lc >= num_local_cols) return;

    unsigned int newDist = s_distIK[threadIdx.y] + s_col_k[threadIdx.x];
    int idx = lc * N + j;
    unsigned int distIJ = d_dist[idx];

    if (newDist < distIJ) {
        d_dist[idx] = newDist;
        d_path[idx] = (unsigned int)k;
    }
}

// Deterministic init — must remain sequential for reproducible rand_r sequence
void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t numNodes,
                              unsigned int rangeMin, unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// path[col * N + row] = col  (direct predecessor is the source)
void initializePathMatrix(std::vector<unsigned int>& path, size_t numNodes) {
    #pragma omp parallel for schedule(static)
    for (size_t col = 0; col < numNodes; ++col) {
        for (size_t row = 0; row < numNodes; ++row) {
            path[col * numNodes + row] = (unsigned int)col;
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, size_t numNodes) {
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t numNodes = 512;
    bool validate   = false;
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

    // Assign one GPU per rank (round-robin)
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d, GPUs: %d, OpenMP threads: %d\n",
               nprocs, num_devices, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Every rank initialises the full matrix (deterministic, same seed)
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    // Distribute columns across MPI ranks (blocked)
    int N = (int)numNodes;
    int base_cols = N / nprocs;
    int rem       = N % nprocs;
    int col_start      = rank * base_cols + std::min(rank, rem);
    int num_local_cols  = base_cols + (rank < rem ? 1 : 0);

    // Extract local columns (OpenMP-parallel memcpy)
    size_t local_elems = (size_t)num_local_cols * N;
    std::vector<unsigned int> local_dist(local_elems);
    std::vector<unsigned int> local_path(local_elems);

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < num_local_cols; ++c) {
        int gc = col_start + c;
        memcpy(&local_dist[(size_t)c * N], &dist[(size_t)gc * N], N * sizeof(unsigned int));
        memcpy(&local_path[(size_t)c * N], &path[(size_t)gc * N], N * sizeof(unsigned int));
    }

    // Free full matrices — will gather on rank 0 after computation
    dist.clear(); dist.shrink_to_fit();
    path.clear(); path.shrink_to_fit();

    // GPU allocation
    unsigned int *d_dist = nullptr, *d_path = nullptr, *d_col_k = nullptr;
    size_t local_bytes = local_elems * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&d_dist, local_bytes));
    CUDA_CHECK(cudaMalloc(&d_path, local_bytes));
    CUDA_CHECK(cudaMalloc(&d_col_k, N * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_dist, local_dist.data(), local_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, local_path.data(), local_bytes, cudaMemcpyHostToDevice));

    // Pinned host buffer for column-k broadcast (faster H2D / D2H)
    unsigned int* col_k_buf = nullptr;
    CUDA_CHECK(cudaMallocHost(&col_k_buf, N * sizeof(unsigned int)));

    // Precompute column ownership
    std::vector<int> col_owner(N);
    std::vector<int> col_local_off(N);
    {
        int cs = 0;
        for (int r = 0; r < nprocs; ++r) {
            int nc = base_cols + (r < rem ? 1 : 0);
            for (int c = cs; c < cs + nc; ++c) {
                col_owner[c]     = r;
                col_local_off[c] = c - cs;
            }
            cs += nc;
        }
    }

    // Kernel grid
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BLOCK_X - 1) / BLOCK_X,
              (num_local_cols + BLOCK_Y - 1) / BLOCK_Y);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing shortest paths...\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < N; ++k) {
        int owner = col_owner[k];

        // Owner extracts column k from GPU
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(col_k_buf, d_dist + (size_t)col_local_off[k] * N,
                                  N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }

        MPI_Bcast(col_k_buf, N, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Async H2D + kernel launch on stream
        CUDA_CHECK(cudaMemcpyAsync(d_col_k, col_k_buf,
                                   N * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));

        if (num_local_cols > 0) {
            fw_kernel<<<grid, block, 0, stream>>>(d_dist, d_path, d_col_k,
                                                   N, k, num_local_cols);
        }

        // Must sync before next iteration's D2H copy
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    long local_duration_ms = duration.count();
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);
        double ops = (double)N * N * N;
        double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Copy results back
    CUDA_CHECK(cudaMemcpy(local_dist.data(), d_dist, local_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_col_k));
    CUDA_CHECK(cudaFreeHost(col_k_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Gather all columns on rank 0
    std::vector<int> recvcounts(nprocs), displs(nprocs);
    int disp = 0;
    for (int r = 0; r < nprocs; ++r) {
        int nc = base_cols + (r < rem ? 1 : 0);
        recvcounts[r] = nc * N;
        displs[r] = disp;
        disp += recvcounts[r];
    }

    std::vector<unsigned int> full_dist;
    if (rank == 0) full_dist.resize((size_t)N * N);

    MPI_Gatherv(local_dist.data(), num_local_cols * N, MPI_UNSIGNED,
                rank == 0 ? full_dist.data() : nullptr,
                recvcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (printResults) {
            print_results_int(full_dist, "DistanceMatrix");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full_dist, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
