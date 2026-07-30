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
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 16;

// ---------------------------------------------------------------------------
// CUDA error checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                          \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Original column-major indexing: dist[col][row] = col * n + row
// ---------------------------------------------------------------------------
inline constexpr size_t idx2(const size_t i, const size_t j,
                             const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------
// CUDA kernel – one thread per (local_row, column) element
// Local data is stored row‑major: [local_row * n + col]
// ---------------------------------------------------------------------------
__global__ void fwKernel(unsigned int* __restrict__ local_dist,
                         unsigned int* __restrict__ local_path,
                         const unsigned int* __restrict__ pivot_row, int k,
                         int n, int local_n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column
    int i = blockIdx.y * blockDim.y + threadIdx.y; // local row

    if (i < local_n && j < n) {
        unsigned int ik = local_dist[i * n + k]; // dist[i][k]
        unsigned int kj = pivot_row[j];          // dist[k][j]
        unsigned int ij = local_dist[i * n + j]; // dist[i][j]

        unsigned int newDist = ik + kj;
        if (newDist < ij) {
            local_dist[i * n + j] = newDist;
            local_path[i * n + j] = static_cast<unsigned int>(k);
        }
    }
}

// ---------------------------------------------------------------------------
// MPI distribution helpers – O(1)
// ---------------------------------------------------------------------------
inline size_t computeLocalN(size_t N, int rank, int numRanks) {
    size_t base  = N / static_cast<size_t>(numRanks);
    size_t rem   = N % static_cast<size_t>(numRanks);
    return base + (static_cast<size_t>(rank) < rem ? 1 : 0);
}

inline size_t computeStart(int rank, size_t N, int numRanks) {
    size_t base  = N / static_cast<size_t>(numRanks);
    size_t rem   = N % static_cast<size_t>(numRanks);
    if (static_cast<size_t>(rank) < rem)
        return static_cast<size_t>(rank) * (base + 1);
    return rem * (base + 1) +
           (static_cast<size_t>(rank) - rem) * base;
}

inline int computeOwner(size_t k, size_t N, int numRanks) {
    size_t base = N / static_cast<size_t>(numRanks);
    size_t rem  = N % static_cast<size_t>(numRanks);
    size_t firstLarge = rem * (base + 1);
    if (k < firstLarge)
        return static_cast<int>(k / (base + 1));
    return static_cast<int>(rem + (k - firstLarge) / base);
}

// ---------------------------------------------------------------------------
// Initialisation – deterministic on every rank, O(N²)
// ---------------------------------------------------------------------------
void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                              size_t numNodes,
                              unsigned int rangeMin, unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range =
        static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i)
        dist[i] = rangeMin +
                  static_cast<unsigned int>(range * rand_r(&seed) /
                                            static_cast<double>(RAND_MAX));

    // Diagonal → 0
#pragma omp parallel for
    for (size_t i = 0; i < numNodes; ++i)
        dist[idx2(i, i, numNodes)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, size_t numNodes) {
#pragma omp parallel for
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = static_cast<unsigned int>(j);
            path[idx2(j, i, numNodes)] = static_cast<unsigned int>(i);
        }
        path[idx2(j, j, numNodes)] = static_cast<unsigned int>(j);
    }
}

// ---------------------------------------------------------------------------
// Validation – OpenMP-parallelised, runs on rank 0 only
// ---------------------------------------------------------------------------
bool validateResult(const std::vector<unsigned int>& dist, size_t numNodes) {
    bool valid = true;

    // 1. Diagonal must be zero
#pragma omp parallel for reduction(&& : valid)
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
#pragma omp critical
            {
                printf("Validation failed: diagonal [%zu,%zu] != 0\n", i, i);
                valid = false;
            }
        }
    }
    if (!valid) return false;

    // 2. Triangle inequality on a sample
    size_t limit = std::min(numNodes, static_cast<size_t>(10));
#pragma omp parallel for collapse(2) reduction(&& : valid)
    for (size_t i = 0; i < limit; ++i) {
        for (size_t j = 0; j < limit; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                unsigned int dIJ = dist[idx2(j, i, numNodes)];
                unsigned int dIK = dist[idx2(k, i, numNodes)];
                unsigned int dKJ = dist[idx2(j, k, numNodes)];
                if (dIK < INF && dKJ < INF && dIK + dKJ < dIJ) {
#pragma omp critical
                    {
                        printf("Validation: triangle inequality "
                               "violated at [%zu,%zu,%zu]\n",
                               i, j, k);
                        valid = false;
                    }
                }
            }
        }
    }
    return valid;
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes (default: 512)\n");
    printf("  -v           Validate result\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           This help\n");
}

// ---------------------------------------------------------------------------
// Main – hybrid MPI / OpenMP / CUDA
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int numRanks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // ---- CUDA device selection (round‑robin) ---------------------------------
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    int cudaDevice = numDevices > 0 ? rank % numDevices : 0;
    if (numDevices > 0) CUDA_CHECK(cudaSetDevice(cudaDevice));

    // ---- Parse arguments on rank 0, then broadcast ---------------------------
    size_t numNodes = 512;
    bool   validate     = false;
    bool   printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(atol(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Bcast(&numNodes,     1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    int valInt = validate ? 1 : 0;
    MPI_Bcast(&valInt,       1, MPI_INT,           0, MPI_COMM_WORLD);
    validate = (valInt != 0);
    int prInt = printResults ? 1 : 0;
    MPI_Bcast(&prInt,        1, MPI_INT,           0, MPI_COMM_WORLD);
    printResults = (prInt != 0);

    size_t localN     = computeLocalN(numNodes, rank, numRanks);
    size_t localStart = computeStart(rank, numNodes, numRanks);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI processes:   %d\n", numRanks);
        printf("OMP threads:     %d\n", omp_get_max_threads());
        printf("CUDA device:     %d / %d found\n", cudaDevice, numDevices);
        printf("Validation:      %s\n", validate ? "enabled" : "disabled");
    }

    // ---- Allocate & initialise the full matrix (deterministic) ----------------
    std::vector<unsigned int> fullDist(numNodes * numNodes);
    std::vector<unsigned int> fullPath(numNodes * numNodes);

    initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(fullPath, numNodes);

    // ---- Pinned host buffers for GPU transfer --------------------------------
    unsigned int *h_dist = nullptr, *h_path = nullptr, *h_pivot = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_dist),
                             localN * numNodes * sizeof(unsigned int),
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_path),
                             localN * numNodes * sizeof(unsigned int),
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_pivot),
                             numNodes * sizeof(unsigned int),
                             cudaHostAllocPortable));

    // Convert local rows from column‑major → row‑major for the GPU
#pragma omp parallel for
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t li = 0; li < localN; ++li) {
            size_t gr = localStart + li;
            h_dist[li * numNodes + j] = fullDist[j * numNodes + gr];
            h_path[li * numNodes + j] = fullPath[j * numNodes + gr];
        }
    }

    // ---- GPU allocations ----------------------------------------------------
    unsigned int *d_dist, *d_path, *d_pivot;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dist),
                          localN * numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_path),
                          localN * numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pivot),
                          numNodes * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_dist, h_dist,
                          localN * numNodes * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, h_path,
                          localN * numNodes * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing shortest paths...\n");

    auto tstart = std::chrono::high_resolution_clock::now();

    // ---- Kernel launch configuration ----------------------------------------
    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim((numNodes + blockDim.x - 1) / blockDim.x,
                 (localN  + blockDim.y - 1) / blockDim.y);

    // ---- Main Floyd‑Warshall loop – each k is sequential --------------------
    for (size_t k = 0; k < numNodes; ++k) {
        int owner = computeOwner(k, numNodes, numRanks);

        // Owner extracts the pivot row from its GPU memory
        if (rank == owner) {
            size_t off = k - computeStart(owner, numNodes, numRanks);
            CUDA_CHECK(cudaMemcpy(h_pivot,
                                  d_dist + off * numNodes,
                                  numNodes * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }

        // Broadcast pivot row to all ranks
        MPI_Bcast(h_pivot, static_cast<int>(numNodes),
                  MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Copy pivot row to GPU
        CUDA_CHECK(cudaMemcpy(d_pivot, h_pivot,
                              numNodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        // Launch kernel
        fwKernel<<<gridDim, blockDim>>>(d_dist, d_path, d_pivot,
                                        static_cast<int>(k),
                                        static_cast<int>(numNodes),
                                        static_cast<int>(localN));

        // Check for launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[%d] Kernel error at k=%zu: %s\n",
                    rank, k, cudaGetErrorString(err));
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto tend  = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart);

    // ---- Copy results back to host -----------------------------------------
    CUDA_CHECK(cudaMemcpy(h_dist, d_dist,
                          localN * numNodes * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_path, d_path,
                          localN * numNodes * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    // ---- Performance report (rank 0 only) ----------------------------------
    if (rank == 0) {
        printf("Computation time: %ld ms\n",
               static_cast<long>(duration.count()));
        double ops    = static_cast<double>(numNodes) *
                        static_cast<double>(numNodes) *
                        static_cast<double>(numNodes);
        double gflops = ops / (static_cast<double>(duration.count()) / 1000.0)
                        / 1.0e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // ---- Gather results to rank 0 for output / validation -------------------
    if (rank == 0) {
        // Write own rows back into fullDist / fullPath (column‑major)
#pragma omp parallel for
        for (size_t j = 0; j < numNodes; ++j) {
            for (size_t li = 0; li < localN; ++li) {
                size_t gr = localStart + li;
                fullDist[j * numNodes + gr] = h_dist[li * numNodes + j];
                fullPath[j * numNodes + gr] = h_path[li * numNodes + j];
            }
        }

        // Receive from other ranks
        std::vector<unsigned int> recvBuf;
        for (int r = 1; r < numRanks; ++r) {
            size_t rn = computeLocalN(numNodes, r, numRanks);
            size_t rs = computeStart(r, numNodes, numRanks);
            recvBuf.resize(rn * numNodes);
            MPI_Recv(recvBuf.data(), static_cast<int>(rn * numNodes),
                     MPI_UNSIGNED, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (size_t j = 0; j < numNodes; ++j) {
                for (size_t li = 0; li < rn; ++li) {
                    fullDist[j * numNodes + rs + li] =
                        recvBuf[li * numNodes + j];
                    fullPath[j * numNodes + rs + li] =
                        recvBuf[li * numNodes + j];
                }
            }
        }
    } else {
        MPI_Send(h_dist, static_cast<int>(localN * numNodes),
                 MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
    }

    // ---- Print results (rank 0 only) ----------------------------------------
    if (rank == 0 && printResults)
        print_results_int(fullDist, "DistanceMatrix");

    // ---- Validation (rank 0 only) -------------------------------------------
    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool ok = validateResult(fullDist, numNodes);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            exitCode = ok ? 0 : 1;
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // ---- Clean-up -----------------------------------------------------------
    CUDA_CHECK(cudaFreeHost(h_dist));
    CUDA_CHECK(cudaFreeHost(h_path));
    CUDA_CHECK(cudaFreeHost(h_pivot));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_pivot));

    MPI_Finalize();
    return exitCode;
}
