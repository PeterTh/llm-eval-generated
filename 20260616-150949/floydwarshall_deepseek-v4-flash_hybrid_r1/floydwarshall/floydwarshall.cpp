#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

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

// Index calculation for flattened 2D array (column-major: dist[i][j] = j * n + i)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
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

// ---------------------------------------------------------------------------
// CUDA kernel: processes local rows for a single Floyd-Warshall iteration k
// Each thread handles one (localRow, column) pair.
// Local storage: column-major with localRows elements per column.
// Grid layout:  blockIdx.x = row-group,   blockIdx.y = column
//               threadIdx.x = local-row within block
// Coalescing: adjacent threads access adjacent rows within same column
// ---------------------------------------------------------------------------
__global__ void floydWarshallKernel(unsigned int* __restrict__ dist,
                                     unsigned int* __restrict__ path,
                                     const unsigned int* __restrict__ kRow,
                                     size_t k, size_t localRows, size_t numNodes) {
    size_t lr = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y;

    if (lr >= localRows) return;

    // dist[i][k] for local row i is at column k, local row lr
    unsigned int dist_ik = dist[k * localRows + lr];
    // dist[k][j] from the broadcast row buffer
    unsigned int dist_kj = kRow[j];

    size_t offset = j * localRows + lr;
    unsigned int dist_ij = dist[offset];

    unsigned int newDist = dist_ik + dist_kj;
    if (newDist < dist_ij) {
        dist[offset] = newDist;
        path[offset] = k;
    }
}

// ---------------------------------------------------------------------------
// Hybrid Floyd-Warshall: MPI distributes rows, OpenMP + CUDA compute locally
// ---------------------------------------------------------------------------
void floydWarshallHybrid(std::vector<unsigned int>& dist,
                          std::vector<unsigned int>& path,
                          size_t numNodes, int rank, int numRanks) {
    // --- Row distribution among MPI ranks ---
    size_t rowsPerRank = (numNodes + numRanks - 1) / numRanks;
    size_t localStart = rank * rowsPerRank;
    size_t localEnd   = std::min(localStart + rowsPerRank, numNodes);
    size_t localRows  = localEnd - localStart;

    if (localRows == 0) {
        return;
    }

    // --- Extract local rows from the full matrix ---
    // Local storage preserves column-major: element (lr, col) at col * localRows + lr
    std::vector<unsigned int> distLocal(localRows * numNodes);
    std::vector<unsigned int> pathLocal(localRows * numNodes);

    // Gather the local portion from the full (non-distributed) matrix.
    // The local buffer packs each full column's range [localStart, localEnd)
    // contiguously, making each column localRows deep.
    #pragma omp parallel for
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t lr = 0; lr < localRows; ++lr) {
            size_t i  = localStart + lr;
            distLocal[lr + j * localRows] = dist[idx2(j, i, numNodes)];
            pathLocal[lr + j * localRows] = path[idx2(j, i, numNodes)];
        }
    }

    // --- GPU memory allocation ---
    unsigned int *d_dist, *d_path, *d_kRow;
    cudaMalloc(&d_dist, localRows * numNodes * sizeof(unsigned int));
    cudaMalloc(&d_path, localRows * numNodes * sizeof(unsigned int));
    cudaMalloc(&d_kRow, numNodes * sizeof(unsigned int));
    cudaMemcpy(d_dist, distLocal.data(),
               localRows * numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_path, pathLocal.data(),
               localRows * numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // --- Kernel launch configuration ---
    dim3 blockDim(256);
    dim3 gridDim((localRows + 255) / 256, numNodes);

    // Host buffer for the MPI-broadcast row
    std::vector<unsigned int> kRowBuf(numNodes);

    // --- Pre-load row 0 on the owning rank ---
    size_t owner0 = 0 / rowsPerRank;
    if (rank == (int)owner0) {
        // cudaMemcpy2D copies strided row: 1-wide, numNodes-tall,
        // source pitch = localRows * sizeof(unsigned int).
        cudaMemcpy2D(kRowBuf.data(), sizeof(unsigned int),
                     d_dist, localRows * sizeof(unsigned int),
                     sizeof(unsigned int), numNodes,
                     cudaMemcpyDeviceToHost);
    }

    // --- Main Floyd-Warshall loop ---
    for (size_t k = 0; k < numNodes; ++k) {
        int owner = static_cast<int>(k / rowsPerRank);

        // 1. Broadcast row k from owning rank to all ranks
        MPI_Bcast(kRowBuf.data(), numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // 2. Copy the broadcast row to GPU memory
        cudaMemcpy(d_kRow, kRowBuf.data(),
                   numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);

        // 3. Launch CUDA kernel for this k-iteration
        floydWarshallKernel<<<gridDim, blockDim>>>(
            d_dist, d_path, d_kRow, k, localRows, numNodes);
        cudaDeviceSynchronize();

        // 4. Pre-fetch the *next* row (k+1) if we own it
        size_t nextK = k + 1;
        if (nextK < numNodes) {
            size_t nextOwner = nextK / rowsPerRank;
            if (rank == (int)nextOwner) {
                size_t lrOffset = nextK - nextOwner * rowsPerRank;
                cudaMemcpy2D(kRowBuf.data(), sizeof(unsigned int),
                             d_dist + lrOffset, localRows * sizeof(unsigned int),
                             sizeof(unsigned int), numNodes,
                             cudaMemcpyDeviceToHost);
            }
        }
    }

    // --- Copy results back to host ---
    cudaMemcpy(distLocal.data(), d_dist,
               localRows * numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pathLocal.data(), d_path,
               localRows * numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_dist);
    cudaFree(d_path);
    cudaFree(d_kRow);

    // --- Gather results back to rank 0 ---
    if (rank == 0) {
        // Copy our own portion
        #pragma omp parallel for
        for (size_t j = 0; j < numNodes; ++j) {
            for (size_t lr = 0; lr < localRows; ++lr) {
                size_t i = localStart + lr;
                dist[idx2(j, i, numNodes)] = distLocal[lr + j * localRows];
                path[idx2(j, i, numNodes)] = pathLocal[lr + j * localRows];
            }
        }

        // Receive from other ranks
        for (int r = 1; r < numRanks; ++r) {
            size_t rStart = r * rowsPerRank;
            size_t rEnd   = std::min(rStart + rowsPerRank, numNodes);
            size_t rRows  = rEnd - rStart;

            std::vector<unsigned int> recvDist(rRows * numNodes);
            std::vector<unsigned int> recvPath(rRows * numNodes);

            MPI_Recv(recvDist.data(), rRows * numNodes, MPI_UNSIGNED,
                     r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recvPath.data(), rRows * numNodes, MPI_UNSIGNED,
                     r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (size_t j = 0; j < numNodes; ++j) {
                for (size_t lr = 0; lr < rRows; ++lr) {
                    size_t i = rStart + lr;
                    dist[idx2(j, i, numNodes)] = recvDist[lr + j * rRows];
                    path[idx2(j, i, numNodes)] = recvPath[lr + j * rRows];
                }
            }
        }
    } else {
        MPI_Send(distLocal.data(), localRows * numNodes, MPI_UNSIGNED,
                 0, 0, MPI_COMM_WORLD);
        MPI_Send(pathLocal.data(), localRows * numNodes, MPI_UNSIGNED,
                 0, 1, MPI_COMM_WORLD);
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    
    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                // Check for overflow before addition
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Set CUDA device: round-robin across available GPUs
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices > 0) {
        cudaSetDevice(rank % numDevices);
    } else {
        if (rank == 0) {
            printf("ERROR: No CUDA-capable device found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 parses, then broadcasts)
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
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast parsed parameters to all ranks
    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of MPI ranks: %d\n", numRanks);
        printf("CUDA devices available: %d\n", numDevices);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Parallelization: MPI + OpenMP + CUDA\n");
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // All ranks allocate the full matrix (same seed = identical values)
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing shortest paths (MPI + OpenMP + CUDA)...\n");
    }

    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallHybrid(dist, path, numNodes, rank, numRanks);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        double ops = (double)numNodes * numNodes * numNodes;
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
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
