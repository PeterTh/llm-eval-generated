#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>

#include <mpi.h>
#include <omp.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000u;
constexpr unsigned int MAX_DISTANCE = 200u;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#ifdef USE_CUDA
static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}
#endif

// Initialize on rank 0 then broadcast
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax, int rank) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    unsigned int seed = 42 + rank; // different seed per rank if used locally

    if (rank == 0) {
        for (size_t i = 0; i < numNodes * numNodes; ++i) {
            dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        }

        for (size_t i = 0; i < numNodes; ++i) {
            dist[idx2(i, i, numNodes)] = 0;
        }
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes, int rank) {
    if (rank == 0) {
        for (size_t j = 0; j < numNodes; ++j) {
            for (size_t i = 0; i < numNodes; ++i) {
                path[idx2(i, j, numNodes)] = j;
                path[idx2(j, i, numNodes)] = i;
            }
            path[idx2(j, j, numNodes)] = j;
        }
    }
}

#ifdef USE_CUDA
// CUDA kernel: each thread handles one (i,j) pair within the local i range
extern "C" __global__ void fw_kernel(unsigned int* dist, unsigned int* path, const size_t n, const size_t k, const size_t iStart, const size_t iCount) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = iCount * n;
    if (idx >= total) return;
    const size_t local_i = idx / n;
    const size_t j = idx % n;
    const size_t i = iStart + local_i;

    unsigned int distIJ = dist[j * n + i]; // idx2(i,j,n) = j*n + i
    unsigned int distIK = dist[k * n + i]; // idx2(k,i,n)
    unsigned int distKJ = dist[j * n + k]; // idx2(j,k,n)

    unsigned int newDist = INF;
    if (distIK < INF && distKJ < INF) {
        newDist = distIK + distKJ;
    }

    if (newDist < distIJ) {
        dist[j * n + i] = newDist;
        path[j * n + i] = (unsigned int)k;
    }
}
#else
// CPU fallback kernel using OpenMP
static void fw_kernel_cpu(std::vector<unsigned int>& dist, std::vector<unsigned int>& path, const size_t n, const size_t k, const size_t iStart, const size_t iCount) {
    #pragma omp parallel for
    for (size_t local_i = 0; local_i < iCount; ++local_i) {
        size_t i = iStart + local_i;
        for (size_t j = 0; j < n; ++j) {
            unsigned int distIJ = dist[idx2(i, j, n)];
            unsigned int distIK = dist[idx2(k, i, n)];
            unsigned int distKJ = dist[idx2(j, k, n)];

            unsigned int newDist = INF;
            if (distIK < INF && distKJ < INF) newDist = distIK + distKJ;
            if (newDist < distIJ) {
                dist[idx2(i, j, n)] = newDist;
                path[idx2(i, j, n)] = (unsigned int)k;
            }
        }
    }
}
#endif

// MPI reduction to pick (dist, path) pair with minimum distance
void minDistPath_op(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype) {
    uint64_t* in = static_cast<uint64_t*>(invec);
    uint64_t* inout = static_cast<uint64_t*>(inoutvec);
    for (int i = 0; i < *len; ++i) {
        uint64_t a = in[i];
        uint64_t b = inout[i];
        uint32_t dist_a = (uint32_t)(a >> 32);
        uint32_t dist_b = (uint32_t)(b >> 32);
        if (dist_a < dist_b) {
            inout[i] = a;
        } else if (dist_a == dist_b) {
            uint32_t path_a = (uint32_t)(a & 0xFFFFFFFFu);
            uint32_t path_b = (uint32_t)(b & 0xFFFFFFFFu);
            if (path_a < path_b) inout[i] = a;
        }
    }
}

void floydWarshallHybrid(std::vector<unsigned int>& dist, 
                         std::vector<unsigned int>& path, 
                         const size_t numNodes, MPI_Comm comm) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Determine i-range for this rank
    size_t base = numNodes / size;
    size_t rem = numNodes % size;
    size_t iStart = rank < (int)rem ? rank * (base + 1) : rank * base + rem;
    size_t iCount = (rank < (int)rem) ? (base + 1) : base;

#ifdef USE_CUDA
    // Device allocations
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    size_t bytes = sizeof(unsigned int) * numNodes * numNodes;
    cudaCheck(cudaMalloc(&d_dist, bytes), "alloc d_dist");
    cudaCheck(cudaMalloc(&d_path, bytes), "alloc d_path");
#endif

    // Prepare MPI reduction op for uint64_t pairs
    MPI_Op minOp;
    MPI_Op_create((MPI_User_function*)minDistPath_op, /*commute=*/1, &minOp);

    std::vector<uint64_t> localCombined(numNodes * numNodes);
    std::vector<uint64_t> globalCombined(numNodes * numNodes);

    // Main k loop
    for (size_t k = 0; k < numNodes; ++k) {
#ifdef USE_CUDA
        // Copy host arrays to device
        size_t bytes = sizeof(unsigned int) * numNodes * numNodes;
        cudaCheck(cudaMemcpy(d_dist, dist.data(), bytes, cudaMemcpyHostToDevice), "H2D dist");
        cudaCheck(cudaMemcpy(d_path, path.data(), bytes, cudaMemcpyHostToDevice), "H2D path");

        // Launch kernel to update local i range
        size_t total = iCount * numNodes;
        const int block = 256;
        int grid = (int)((total + block - 1) / block);
        fw_kernel<<<grid, block>>>(d_dist, d_path, numNodes, k, iStart, iCount);
        cudaCheck(cudaGetLastError(), "kernel launch");
        cudaCheck(cudaDeviceSynchronize(), "kernel sync");

        // Copy device results back to host (only need local i range, but copy full arrays for simplicity)
        std::vector<unsigned int> updatedDist(numNodes * numNodes);
        std::vector<unsigned int> updatedPath(numNodes * numNodes);
        cudaCheck(cudaMemcpy(updatedDist.data(), d_dist, bytes, cudaMemcpyDeviceToHost), "D2H dist");
        cudaCheck(cudaMemcpy(updatedPath.data(), d_path, bytes, cudaMemcpyDeviceToHost), "D2H path");

        // Build localCombined = (dist<<32) | path
        #pragma omp parallel for
        for (size_t idx = 0; idx < numNodes * numNodes; ++idx) {
            uint32_t dval = updatedDist[idx];
            uint32_t pval = updatedPath[idx];
            localCombined[idx] = (static_cast<uint64_t>(dval) << 32) | static_cast<uint64_t>(pval);
        }
#else
        // CPU fallback: run kernel on host for local i range
        fw_kernel_cpu(dist, path, numNodes, k, iStart, iCount);

        #pragma omp parallel for
        for (size_t idx = 0; idx < numNodes * numNodes; ++idx) {
            uint32_t dval = dist[idx];
            uint32_t pval = path[idx];
            localCombined[idx] = (static_cast<uint64_t>(dval) << 32) | static_cast<uint64_t>(pval);
        }
#endif

        // Allreduce with custom op to get minimal distances and corresponding paths
        MPI_Allreduce(localCombined.data(), globalCombined.data(), (int)(numNodes * numNodes), MPI_UNSIGNED_LONG_LONG, minOp, comm);

        // Unpack globalCombined back into dist and path (host arrays)
        #pragma omp parallel for
        for (size_t idx = 0; idx < numNodes * numNodes; ++idx) {
            uint64_t val = globalCombined[idx];
            dist[idx] = (uint32_t)(val >> 32);
            path[idx] = (uint32_t)(val & 0xFFFFFFFFu);
        }
    }

    MPI_Op_free(&minOp);
#ifdef USE_CUDA
    cudaCheck(cudaFree(d_dist), "free d_dist");
    cudaCheck(cudaFree(d_path), "free d_path");
#endif
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
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
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
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
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // Parse command line arguments (only rank 0 prints help/errors)
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
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices (host)
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    // Initialize on rank 0 and broadcast
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE, rank);
    initializePathMatrix(path, numNodes, rank);

    MPI_Bcast(dist.data(), (int)(numNodes * numNodes), MPI_UNSIGNED, 0, comm);
    MPI_Bcast(path.data(), (int)(numNodes * numNodes), MPI_UNSIGNED, 0, comm);

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    floydWarshallHybrid(dist, path, numNodes, comm);

    double t1 = MPI_Wtime();
    double duration_ms = (t1 - t0) * 1000.0;

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration_ms);
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
