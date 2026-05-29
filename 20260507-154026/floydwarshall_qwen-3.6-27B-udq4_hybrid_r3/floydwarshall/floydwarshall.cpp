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

// Column-major index (preserves original layout): D[i][j] at flat[j*n + i]
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

/* ------------------------------------------------------------------ */
/*  CUDA kernel – column-major layout, column decomposition           */
/*  flat[j*n + i] stores D[i][j].  Each rank owns columns             */
/*  [j_start .. j_start+local_cols-1].  col_k[i] = D[i][k]            */
/*  is broadcast from MPI each k-iteration.                           */
/* ------------------------------------------------------------------ */
__global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path,
                                     const unsigned int* col_k,
                                     unsigned int n,
                                     unsigned int local_cols,
                                     unsigned int k)
{
    unsigned int j_local = blockIdx.x * blockDim.x + threadIdx.x;  // 0 .. local_cols-1
    unsigned int i       = blockIdx.y * blockDim.y + threadIdx.y;  // 0 .. n-1

    if (i < n && j_local < local_cols) {
        // D[i][j] = dist[j_local*n + i]
        // D[i][k] = col_k[i]          (broadcast)
        // D[k][j] = dist[j_local*n + k]
        unsigned int new_dist = col_k[i] + __ldg(&dist[j_local * n + k]);
        if (new_dist < dist[j_local * n + i]) {
            dist[j_local * n + i] = new_dist;
            path[j_local * n + i] = k;
        }
    }
}

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _err = call;                                           \
        if (_err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %d at %s:%d: %s\n",               \
                    _err, __FILE__, __LINE__, cudaGetErrorString(_err));   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                  \
        }                                                                  \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Initialise the distance matrix (sequential fill, then diagonal).   */
/* ------------------------------------------------------------------ */
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                               const unsigned int rangeMin, const unsigned int rangeMax)
{
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin +
            (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    for (size_t i = 0; i < numNodes; ++i)
        dist[idx2(i, i, numNodes)] = 0;
}

/* ------------------------------------------------------------------ */
/*  Initialise the path matrix: path[i][j] = j  (OpenMP parallel)      */
/* ------------------------------------------------------------------ */
void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < numNodes; ++i)
        for (size_t j = 0; j < numNodes; ++j)
            path[idx2(i, j, numNodes)] = (unsigned int)j;
}

/* ------------------------------------------------------------------ */
/*  Validate the result (OpenMP parallel for the triangle check).      */
/* ------------------------------------------------------------------ */
bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes)
{
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    size_t limit = std::min(numNodes, static_cast<size_t>(10));
    bool violated = false;

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t i = 0; i < limit; ++i)
        for (size_t j = 0; j < limit; ++j)
            for (size_t k = 0; k < numNodes; ++k) {
                unsigned int dist_ij = dist[idx2(i, j, numNodes)];
                unsigned int dist_ik = dist[idx2(i, k, numNodes)];
                unsigned int dist_kj = dist[idx2(k, j, numNodes)];

                if (dist_ik < INF && dist_kj < INF) {
                    if (dist_ik + dist_kj < dist_ij)
                        violated = true;
                }
            }

    if (violated) {
        printf("Validation failed: triangle inequality violated\n");
        return false;
    }
    return true;
}

/* ================================================================== */
/*  main – Hybrid MPI (column decomp) + OpenMP + CUDA                  */
/* ================================================================== */
int main(int argc, char** argv)
{
    /* ---- MPI init ---- */
    int rank, num_ranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    /* ---- Parse arguments (rank 0) ---- */
    size_t numNodes = 512;
    int validate = 0, printResults = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = (size_t)atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("  -n <num>  Number of nodes (default: 512)\n");
                printf("  -v        Enable validation\n");
                printf("  -r        Print results\n");
                printf("  -h        Show help\n");
                MPI_Finalize();
                return 0;
            }
        }
    }

    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate,   1, MPI_INT,               0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT,             0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Hybrid: MPI (%d ranks) + OpenMP + CUDA\n", num_ranks);
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    /* ---- 1-D column decomposition (contiguous in column-major) ---- */
    size_t cols_per_rank = numNodes / num_ranks;
    if (numNodes % num_ranks != 0 && rank == 0)
        printf("Warning: numNodes not evenly divisible by num_ranks\n");

    size_t j_start    = rank * cols_per_rank;
    size_t local_size = cols_per_rank * numNodes;

    /* ---- Host buffers ---- */
    std::vector<unsigned int> local_dist(local_size);
    std::vector<unsigned int> local_path(local_size);

    /* ---- Initialise on rank 0, Scatterv to all ranks ---- */
    if (rank == 0) {
        printf("Initializing graph...\n");
        std::vector<unsigned int> full_dist(numNodes * numNodes);
        std::vector<unsigned int> full_path(numNodes * numNodes);

        initializeDistanceMatrix(full_dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(full_path, numNodes);

        std::vector<int> recvcounts(num_ranks, (int)local_size);
        std::vector<MPI_Offset> displs(num_ranks);
        for (int r = 0; r < num_ranks; ++r)
            displs[r] = r * cols_per_rank * numNodes;

        MPI_Scatterv(full_dist.data(), recvcounts.data(), displs.data(),
                     MPI_UNSIGNED, local_dist.data(), (int)local_size,
                     MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        MPI_Scatterv(full_path.data(), recvcounts.data(), displs.data(),
                     MPI_UNSIGNED, local_path.data(), (int)local_size,
                     MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED,
                     local_dist.data(), (int)local_size, MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED,
                     local_path.data(), (int)local_size, MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    }

    /* ---- CUDA setup ---- */
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    int cuda_device = rank % std::max(device_count, 1);
    CUDA_CHECK(cudaSetDevice(cuda_device));

    unsigned int *d_dist  = nullptr,
                 *d_path  = nullptr,
                 *d_col_k = nullptr;

    CUDA_CHECK(cudaMalloc(&d_dist,  local_size * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_path,  local_size * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_col_k, numNodes * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_dist, local_dist.data(),
                          local_size * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, local_path.data(),
                          local_size * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    /* ---- Floyd-Warshall main loop ---- */
    if (rank == 0) printf("Computing shortest paths...\n");

    unsigned int* h_col_k = new unsigned int[numNodes];

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = std::chrono::high_resolution_clock::now();

    dim3 block(16, 16);
    dim3 grid((cols_per_rank + block.x - 1) / block.x,
              (numNodes + block.y - 1) / block.y);

    for (size_t k = 0; k < numNodes; ++k) {
        /* Determine which rank owns column k */
        int k_owner   = (int)(k / cols_per_rank);
        size_t local_k = k - (size_t)k_owner * cols_per_rank;

        /* Extract column k from local data (only the owner does this) */
        /* In column-major: column k is at flat[k*n + i] for i in [0,n) */
        /* In local data: column k starts at local_k * n */
        if (rank == k_owner)
            for (size_t i = 0; i < numNodes; ++i)
                h_col_k[i] = local_dist[local_k * numNodes + i];

        /* Broadcast column k to every rank */
        MPI_Bcast(h_col_k, (int)numNodes, MPI_UNSIGNED, k_owner, MPI_COMM_WORLD);

        /* Transfer column k to GPU */
        CUDA_CHECK(cudaMemcpy(d_col_k, h_col_k,
                              numNodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        /* Launch kernel – each thread updates one (i, j_local) pair */
        floydWarshallKernel<<<grid, block>>>(d_dist, d_path, d_col_k,
                                              (unsigned int)numNodes,
                                              (unsigned int)cols_per_rank,
                                              (unsigned int)k);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    delete[] h_col_k;

    /* ---- Copy results back from GPU ---- */
    CUDA_CHECK(cudaMemcpy(local_dist.data(), d_dist,
                          local_size * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(local_path.data(), d_path,
                          local_size * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    /* ---- Gather full matrix to rank 0 ---- */
    std::vector<unsigned int> full_dist, full_path;
    if (rank == 0) {
        full_dist.resize(numNodes * numNodes);
        full_path.resize(numNodes * numNodes);
    }

    std::vector<int> recvcounts(num_ranks, (int)local_size);
    std::vector<MPI_Offset> displs(num_ranks);
    for (int r = 0; r < num_ranks; ++r)
        displs[r] = r * cols_per_rank * numNodes;

    MPI_Gatherv(local_dist.data(), (int)local_size, MPI_UNSIGNED,
                full_dist.data(), recvcounts.data(), displs.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_path.data(), (int)local_size, MPI_UNSIGNED,
                full_path.data(), recvcounts.data(), displs.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    /* ---- Performance reporting (rank 0) ---- */
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
        printf("Computation time: %ld ms\n", duration.count());

        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / ((double)duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults)
            print_results_int(full_dist, "DistanceMatrix");

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full_dist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) {
                CUDA_CHECK(cudaFree(d_dist));
                CUDA_CHECK(cudaFree(d_path));
                CUDA_CHECK(cudaFree(d_col_k));
                MPI_Finalize();
                return 1;
            }
        }
    }

    /* ---- Cleanup ---- */
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_col_k));

    MPI_Finalize();
    return 0;
}
