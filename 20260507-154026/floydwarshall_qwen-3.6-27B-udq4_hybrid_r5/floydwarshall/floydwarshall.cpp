#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// idx2(i, j, n) = j*n + i  (column-major layout in the original code).
// The FW loop calls idx2(j, i, n) = i*n + j, which gives row-major access.
// The flat array is therefore effectively row-major.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

/* ------------------------------------------------------------------ */
/*  CUDA kernel – 2-D tile over (local_row, column).                  */
/*  Shared memory caches row_k to reduce global loads.                */
/* ------------------------------------------------------------------ */
__global__ void fw_kernel(unsigned int* dist, unsigned int* path,
                          const unsigned int* row_k,
                          size_t n, size_t local_rows, unsigned int k)
{
    extern __shared__ unsigned int srow[];

    // cooperative load of row_k into shared memory
    for (size_t j = threadIdx.x; j < n; j += blockDim.x)
        srow[j] = row_k[j];
    __syncthreads();

    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < local_rows && j < n) {
        unsigned int d_ik = dist[i * n + k];
        unsigned int d_kj = srow[j];
        unsigned int d_ij = dist[i * n + j];
        unsigned int nd   = d_ik + d_kj;
        if (nd < d_ij) {
            dist[i * n + j] = nd;
            path[i * n + j] = k;
        }
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

    // 2. Triangle inequality (sampled)
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

int main(int argc, char** argv)
{
    /* ---- MPI init ---- */
    MPI_Init(&argc, reinterpret_cast<char***>(&argv));
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- Parse arguments (rank 0) ---- */
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            const char* arg = reinterpret_cast<const char*>(argv[i]);
            if (strcmp(arg, "-n") == 0 && i + 1 < argc)
                numNodes = static_cast<size_t>(atoi(reinterpret_cast<const char*>(argv[++i])));
            else if (strcmp(arg, "-v") == 0) validate = true;
            else if (strcmp(arg, "-r") == 0) printResults = true;
            else if (strcmp(arg, "-h") == 0) { printUsage(reinterpret_cast<const char*>(argv[0])); MPI_Finalize(); return 0; }
            else { printf("Unknown option: %s\n", reinterpret_cast<const char*>(argv[i])); printUsage(reinterpret_cast<const char*>(argv[0])); MPI_Finalize(); return 1; }
        }
    }

    /* broadcast parameters */
    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    { int v = validate;   MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD); validate = v; }
    { int v = printResults; MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD); printResults = v; }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    /* ---- Row distribution (1-D block across rows) ---- */
    size_t rows_per_rank = numNodes / size;
    size_t remainder     = numNodes % size;
    size_t local_rows    = rows_per_rank + (rank < static_cast<int>(remainder) ? 1 : 0);
    size_t local_row_start =
        static_cast<size_t>(rank) * rows_per_rank + std::min(static_cast<size_t>(rank), remainder);

    std::vector<int> rank_starts(size), rank_counts(size);
    {
        size_t cum = 0;
        for (int r = 0; r < size; ++r) {
            rank_counts[r] = static_cast<int>(rows_per_rank + (r < static_cast<int>(remainder) ? 1 : 0));
            rank_starts[r] = static_cast<int>(cum);
            cum += rank_counts[r];
        }
    }

    /* ---- Local matrices (row-major, same layout as original flat array) ---- */
    std::vector<unsigned int> local_dist(local_rows * numNodes);
    std::vector<unsigned int> local_path(local_rows * numNodes);

    if (rank == 0) printf("Initializing graph...\n");

    // Reproduce the original initialization exactly.
    // The original fills dist[0..N*N-1] sequentially with rand_r.
    // dist[g] with g = i*n + j stores element (i, j) in row-major.
    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE - 1) + 1.0;
    for (size_t g = 0; g < numNodes * numNodes; ++g) {
        unsigned int val = 1 + static_cast<unsigned int>(range * rand_r(&seed) / (double)RAND_MAX);
        size_t i = g / numNodes, j = g % numNodes;
        if (i >= local_row_start && i < local_row_start + local_rows)
            local_dist[(i - local_row_start) * numNodes + j] = val;
    }

    // Zero diagonal: dist[i*n + i] = 0  (global row i, global col i)
    for (size_t il = 0; il < local_rows; ++il) {
        size_t gi = local_row_start + il;  // global row index
        if (gi < numNodes)
            local_dist[il * numNodes + gi] = 0;
    }

    // Path init: path[i*n + j] = i  (row-major: path[row][col] = row)
#ifndef __CUDACC__
    #pragma omp parallel for collapse(2)
#endif
    for (size_t il = 0; il < local_rows; ++il)
        for (size_t j = 0; j < numNodes; ++j)
            local_path[il * numNodes + j] = static_cast<unsigned int>(local_row_start + il);

    /* ---- GPU memory ---- */
    unsigned int *d_dist = nullptr, *d_path = nullptr, *d_row_k = nullptr;
    cudaMalloc(&d_dist,  local_rows * numNodes * sizeof(unsigned int));
    cudaMalloc(&d_path,  local_rows * numNodes * sizeof(unsigned int));
    cudaMalloc(&d_row_k, numNodes * sizeof(unsigned int));
    cudaMemcpy(d_dist, local_dist.data(), local_rows * numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_path, local_path.data(), local_rows * numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);

    std::vector<unsigned int> h_row_k(numNodes);

    if (rank == 0) printf("Computing shortest paths...\n");

    /* ---- Floyd-Warshall main loop (MPI broadcast + CUDA kernel) ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t k = 0; k < numNodes; ++k) {
        // find rank owning row k
        int k_rank = 0, k_local = 0;
        for (int r = 0; r < size; ++r) {
            if (k < static_cast<size_t>(rank_starts[r] + rank_counts[r])) {
                k_rank  = r;
                k_local = static_cast<int>(k - rank_starts[r]);
                break;
            }
        }

        // owner copies its row-k from GPU into the broadcast buffer
        if (rank == k_rank && local_rows > 0) {
            cudaMemcpy(h_row_k.data(), d_dist + k_local * numNodes,
                       numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        }

        // broadcast to all ranks
        MPI_Bcast(h_row_k.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  k_rank, MPI_COMM_WORLD);

        // transfer row_k to GPU
        cudaMemcpy(d_row_k, h_row_k.data(),
                   numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);

        // launch kernel for local rows
        if (local_rows > 0) {
            dim3 block(16, 16);
            dim3 grid((numNodes   + block.x - 1) / block.x,
                      (local_rows + block.y - 1) / block.y);
            fw_kernel<<<grid, block, numNodes * sizeof(unsigned int)>>>(
                d_dist, d_path, d_row_k, numNodes, local_rows,
                static_cast<unsigned int>(k));
            cudaDeviceSynchronize();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- Copy results back ---- */
    cudaMemcpy(local_dist.data(), d_dist,
               local_rows * numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(local_path.data(), d_path,
               local_rows * numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    /* ---- Gather to rank 0 ---- */
    std::vector<unsigned int> full_dist, full_path;
    if (rank == 0) {
        full_dist.resize(numNodes * numNodes);
        full_path.resize(numNodes * numNodes);
    }

    std::vector<int> recvcounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        recvcounts[r] = rank_counts[r] * static_cast<int>(numNodes);
        displs[r]     = rank_starts[r] * static_cast<int>(numNodes);
    }

    MPI_Gatherv(local_dist.data(),  static_cast<int>(local_rows * numNodes), MPI_UNSIGNED,
                full_dist.data(),   recvcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_path.data(),  static_cast<int>(local_rows * numNodes), MPI_UNSIGNED,
                full_path.data(),   recvcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    /* ---- Post-processing on rank 0 ---- */
    int retcode = 0;
    if (rank == 0) {
        // full_dist is already in the same row-major layout as the original flat array.
        // idx2(j, i, n) = i*n + j gives row-major access, so validateResult works directly.
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults)
            print_results_int(full_dist, "DistanceMatrix");

        if (validate) {
            printf("Validating result...\n");
            retcode = validateResult(full_dist, numNodes) ? 0 : 1;
            printf("Validation: %s\n", retcode == 0 ? "PASSED" : "FAILED");
        }
    }

    MPI_Bcast(&retcode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* ---- Cleanup ---- */
    cudaFree(d_dist); cudaFree(d_path); cudaFree(d_row_k);
    MPI_Finalize();
    return retcode;
}
