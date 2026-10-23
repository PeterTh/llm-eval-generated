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

inline void rowRange(const size_t n, const int rank, const int size,
                     size_t& r0, size_t& r1) noexcept {
    r0 = (n * static_cast<size_t>(rank)) / static_cast<size_t>(size);
    r1 = (n * static_cast<size_t>(rank + 1)) / static_cast<size_t>(size);
}

static void checkCuda(const cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
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
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(numNodes); ++i) {
        dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t rows,
                          const size_t r0, const size_t numNodes) {
    #pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r) {
        const unsigned int value = static_cast<unsigned int>(r0 + static_cast<size_t>(r));
        unsigned int* row = path.data() + static_cast<size_t>(r) * numNodes;
        std::fill(row, row + numNodes, value);
    }
}

__global__ void fwUpdateScalarKernel(unsigned int* __restrict__ dist,
                                     unsigned int* __restrict__ path,
                                     const unsigned int* __restrict__ pivotRow,
                                     const int n, const int rows,
                                     const unsigned int k) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= n) {
        return;
    }

    const size_t base = static_cast<size_t>(i) * static_cast<size_t>(n);
    const unsigned int distIJ = dist[base + j];
    const unsigned int distIK = dist[base + k];
    const unsigned int newDist = distIK + pivotRow[j];

    if (newDist < distIJ) {
        dist[base + j] = newDist;
        path[base + j] = k;
    }
}

__global__ void fwUpdateVecKernel(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const unsigned int* __restrict__ pivotRow,
                                  const int n, const int rows,
                                  const unsigned int k) {
    __shared__ uint4 sPivot[32];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n4 = n >> 2;
    const int j4 = blockIdx.x * blockDim.x + tx;

    if (ty == 0 && tx < 32 && j4 < n4) {
        sPivot[tx] = reinterpret_cast<const uint4*>(pivotRow)[j4];
    }
    __syncthreads();

    const int i = blockIdx.y * blockDim.y + ty;
    if (i >= rows || j4 >= n4) {
        return;
    }

    const size_t base = static_cast<size_t>(i) * static_cast<size_t>(n);
    uint4* distRow = reinterpret_cast<uint4*>(dist + base);
    uint4* pathRow = reinterpret_cast<uint4*>(path + base);

    uint4 distIJ = distRow[j4];
    const uint4 pivot = sPivot[tx];
    const unsigned int distIK = dist[base + k];

    const unsigned int nx = distIK + pivot.x;
    const unsigned int ny = distIK + pivot.y;
    const unsigned int nz = distIK + pivot.z;
    const unsigned int nw = distIK + pivot.w;

    const bool cx = nx < distIJ.x;
    const bool cy = ny < distIJ.y;
    const bool cz = nz < distIJ.z;
    const bool cw = nw < distIJ.w;

    if (cx) distIJ.x = nx;
    if (cy) distIJ.y = ny;
    if (cz) distIJ.z = nz;
    if (cw) distIJ.w = nw;

    if (cx || cy || cz || cw) {
        uint4 pathIJ = pathRow[j4];
        if (cx) pathIJ.x = k;
        if (cy) pathIJ.y = k;
        if (cz) pathIJ.z = k;
        if (cw) pathIJ.w = k;
        distRow[j4] = distIJ;
        pathRow[j4] = pathIJ;
    }
}

__global__ void fwUpdateDistScalarKernel(unsigned int* __restrict__ dist,
                                        const unsigned int* __restrict__ pivotRow,
                                        const int n, const int rows,
                                        const unsigned int k) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= n) {
        return;
    }

    const size_t base = static_cast<size_t>(i) * static_cast<size_t>(n);
    const unsigned int distIJ = dist[base + j];
    const unsigned int distIK = dist[base + k];
    const unsigned int newDist = distIK + pivotRow[j];

    if (newDist < distIJ) {
        dist[base + j] = newDist;
    }
}

__global__ void fwUpdateDistVecKernel(unsigned int* __restrict__ dist,
                                      const unsigned int* __restrict__ pivotRow,
                                      const int n, const int rows,
                                      const unsigned int k) {
    __shared__ uint4 sPivot[32];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n4 = n >> 2;
    const int j4 = blockIdx.x * blockDim.x + tx;

    if (ty == 0 && tx < 32 && j4 < n4) {
        sPivot[tx] = reinterpret_cast<const uint4*>(pivotRow)[j4];
    }
    __syncthreads();

    const int i = blockIdx.y * blockDim.y + ty;
    if (i >= rows || j4 >= n4) {
        return;
    }

    const size_t base = static_cast<size_t>(i) * static_cast<size_t>(n);
    uint4* distRow = reinterpret_cast<uint4*>(dist + base);

    uint4 distIJ = distRow[j4];
    const uint4 pivot = sPivot[tx];
    const unsigned int distIK = dist[base + k];

    const unsigned int nx = distIK + pivot.x;
    const unsigned int ny = distIK + pivot.y;
    const unsigned int nz = distIK + pivot.z;
    const unsigned int nw = distIK + pivot.w;

    const bool cx = nx < distIJ.x;
    const bool cy = ny < distIJ.y;
    const bool cz = nz < distIJ.z;
    const bool cw = nw < distIJ.w;

    if (cx) distIJ.x = nx;
    if (cy) distIJ.y = ny;
    if (cz) distIJ.z = nz;
    if (cw) distIJ.w = nw;

    if (cx || cy || cz || cw) {
        distRow[j4] = distIJ;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    int badDiagonal = 0;
    #pragma omp parallel for schedule(static) reduction(| : badDiagonal)
    for (long long i = 0; i < static_cast<long long>(numNodes); ++i) {
        if (dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), numNodes)] != 0) {
            badDiagonal = 1;
        }
    }
    if (badDiagonal) {
        printf("Validation failed: a diagonal element is not zero\n");
        return false;
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    const size_t sample = std::min(numNodes, static_cast<size_t>(10));
    int violation = 0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(| : violation)
    for (long long i = 0; i < static_cast<long long>(sample); ++i) {
        for (long long j = 0; j < static_cast<long long>(sample); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(static_cast<size_t>(j),
                                                      static_cast<size_t>(i), numNodes)];
                const unsigned int distIK = dist[idx2(k, static_cast<size_t>(i), numNodes)];
                const unsigned int distKJ = dist[idx2(static_cast<size_t>(j), k, numNodes)];

                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        violation = 1;
                    }
                }
            }
        }
    }
    if (violation) {
        printf("Validation failed: triangle inequality violated\n");
        return false;
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool badArgs = false;
    bool help = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            badArgs = true;
        }
    }

    if (help) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    if (badArgs) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: this build requires at least one CUDA device\n");
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice");

    std::vector<int> localWorldRanks(localSize);
    MPI_Allgather(&rank, 1, MPI_INT, localWorldRanks.data(), 1, MPI_INT, localComm);
    std::vector<int> worldToLocal(size, -1);
    for (int i = 0; i < localSize; ++i) {
        worldToLocal[localWorldRanks[i]] = i;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    size_t r0 = 0;
    size_t r1 = 0;
    rowRange(numNodes, rank, size, r0, r1);
    const size_t localRows = r1 - r0;
    const size_t localCount = localRows * numNodes;

    std::vector<int> counts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        size_t a0 = 0;
        size_t a1 = 0;
        rowRange(numNodes, r, size, a0, a1);
        counts[r] = static_cast<int>((a1 - a0) * numNodes);
        displs[r] = static_cast<int>(a0 * numNodes);
    }

    std::vector<unsigned int> fullDist;
    if (rank == 0) {
        fullDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
    }

    std::vector<unsigned int> dist(localCount);
    MPI_Scatterv(rank == 0 ? fullDist.data() : nullptr, counts.data(), displs.data(),
                 MPI_UNSIGNED, dist.data(), static_cast<int>(localCount), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    std::vector<unsigned int> path(localCount);
    initializePathMatrix(path, localRows, r0, numNodes);

    size_t blockSize = 1;
    if (numNodes > 0) {
        blockSize = std::min<size_t>(32, std::max<size_t>(8, numNodes / 64));
    }
    const size_t bMax = std::min(blockSize, numNodes);

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_pivotBlock = nullptr;
    if (localCount > 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_dist),
                             localCount * sizeof(unsigned int)), "cudaMalloc dist");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_path),
                             localCount * sizeof(unsigned int)), "cudaMalloc path");
        checkCuda(cudaMemcpy(d_dist, dist.data(), localCount * sizeof(unsigned int),
                             cudaMemcpyHostToDevice), "copy dist to device");
        checkCuda(cudaMemcpy(d_path, path.data(), localCount * sizeof(unsigned int),
                             cudaMemcpyHostToDevice), "copy path to device");
    }
    if (size > 1 && bMax > 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_pivotBlock),
                             bMax * numNodes * sizeof(unsigned int)),
                  "cudaMalloc pivot block");
    }

    unsigned int* h_block = nullptr;
    if (size > 1 && bMax > 0) {
        checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&h_block),
                                bMax * numNodes * sizeof(unsigned int),
                                cudaHostAllocDefault), "cudaHostAlloc block");
    }

    const bool useShm = (localSize == size) && (size > 1) && (bMax > 0);
    unsigned int* shmBase = nullptr;
    MPI_Win shmWin = MPI_WIN_NULL;
    std::vector<unsigned int*> shmSlots;
    if (useShm) {
        const MPI_Aint shmBytes = static_cast<MPI_Aint>(
            2 * bMax * numNodes * sizeof(unsigned int));
        const int err = MPI_Win_allocate_shared(shmBytes, static_cast<int>(sizeof(unsigned int)),
                                                MPI_INFO_NULL, localComm, &shmBase, &shmWin);
        if (err != MPI_SUCCESS) {
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        shmSlots.resize(localSize);
        for (int i = 0; i < localSize; ++i) {
            MPI_Aint slotSize = 0;
            int dispUnit = 0;
            unsigned int* slot = nullptr;
            MPI_Win_shared_query(shmWin, i, &slotSize, &dispUnit, &slot);
            shmSlots[i] = slot;
        }
    }

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    const int blockX = 32;
    const int blockY = 8;
    const bool useVector = ((numNodes % 4) == 0);
    const int n4 = static_cast<int>(numNodes / 4);
    const dim3 block(blockX, blockY);

    size_t blockCounter = 0;
    for (int o = 0; o < size; ++o) {
        size_t o0 = 0;
        size_t o1 = 0;
        rowRange(numNodes, o, size, o0, o1);

        for (size_t k0 = o0; k0 < o1; k0 += blockSize) {
            const size_t k1 = std::min(k0 + blockSize, o1);
            const size_t bl = k1 - k0;
            const size_t blockCount = bl * numNodes;

            if (size > 1 && bl > 0) {
                if (rank == o) {
                    checkCuda(cudaMemcpy(h_block, d_dist + (k0 - r0) * numNodes,
                                         blockCount * sizeof(unsigned int),
                                         cudaMemcpyDeviceToHost),
                              "copy pivot block to host");
                }

                if (useShm) {
                    const size_t parity = blockCounter & 1;
                    const size_t offset = parity * bMax * numNodes;
                    if (rank == o) {
                        std::memcpy(shmSlots[localRank] + offset, h_block,
                                    blockCount * sizeof(unsigned int));
                    }
                    MPI_Win_sync(shmWin);
                    MPI_Barrier(localComm);
                    if (rank != o) {
                        std::memcpy(h_block, shmSlots[worldToLocal[o]] + offset,
                                    blockCount * sizeof(unsigned int));
                    }
                } else {
                    MPI_Bcast(h_block, static_cast<int>(blockCount), MPI_UNSIGNED, o,
                              MPI_COMM_WORLD);
                }

                if (rank != o) {
                    checkCuda(cudaMemcpy(d_pivotBlock, h_block,
                                         blockCount * sizeof(unsigned int),
                                         cudaMemcpyHostToDevice),
                              "copy pivot block to device");
                }
            }

            for (size_t l = 0; l < bl; ++l) {
                const size_t k = k0 + l;
                const unsigned int* pivotDevice = nullptr;
                bool advanceCopy = false;

                if (size == 1 || rank == o) {
                    pivotDevice = d_dist + (k - r0) * numNodes;
                } else {
                    pivotDevice = d_pivotBlock + l * numNodes;
                    advanceCopy = true;
                }

                if (localRows > 0) {
                    if (useVector) {
                        const dim3 grid((n4 + blockX - 1) / blockX,
                                        (static_cast<int>(localRows) + blockY - 1) / blockY);
                        fwUpdateVecKernel<<<grid, block>>>(
                            d_dist, d_path, pivotDevice, static_cast<int>(numNodes),
                            static_cast<int>(localRows), static_cast<unsigned int>(k));
                    } else {
                        const dim3 grid((static_cast<int>(numNodes) + blockX - 1) / blockX,
                                        (static_cast<int>(localRows) + blockY - 1) / blockY);
                        fwUpdateScalarKernel<<<grid, block>>>(
                            d_dist, d_path, pivotDevice, static_cast<int>(numNodes),
                            static_cast<int>(localRows), static_cast<unsigned int>(k));
                    }
                }

                if (advanceCopy && bl > 0) {
                    if (useVector) {
                        const dim3 grid((n4 + blockX - 1) / blockX,
                                        (static_cast<int>(bl) + blockY - 1) / blockY);
                        fwUpdateDistVecKernel<<<grid, block>>>(
                            d_pivotBlock, pivotDevice, static_cast<int>(numNodes),
                            static_cast<int>(bl), static_cast<unsigned int>(k));
                    } else {
                        const dim3 grid((static_cast<int>(numNodes) + blockX - 1) / blockX,
                                        (static_cast<int>(bl) + blockY - 1) / blockY);
                        fwUpdateDistScalarKernel<<<grid, block>>>(
                            d_pivotBlock, pivotDevice, static_cast<int>(numNodes),
                            static_cast<int>(bl), static_cast<unsigned int>(k));
                    }
                }
            }

            ++blockCounter;
        }
    }

    checkCuda(cudaDeviceSynchronize(), "final kernel synchronize");

    const auto end = std::chrono::high_resolution_clock::now();
    const double localMs =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    double maxMs = 0.0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long long durationMs = static_cast<long long>(maxMs);
        printf("Computation time: %lld ms\n", durationMs);

        const double ops = static_cast<double>(numNodes) * static_cast<double>(numNodes) *
                           static_cast<double>(numNodes);
        const double seconds = static_cast<double>(durationMs) / 1000.0;
        const double gflops = ops / seconds / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    if (printResults || validate) {
        if (rank == 0) {
            fullDist.resize(numNodes * numNodes);
        }
        checkCuda(cudaMemcpy(dist.data(), d_dist, localCount * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost), "copy dist to host");
        MPI_Gatherv(dist.data(), static_cast<int>(localCount), MPI_UNSIGNED,
                    rank == 0 ? fullDist.data() : nullptr, counts.data(), displs.data(),
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    int rc = 0;
    if (rank == 0) {
        if (printResults) {
            print_results_int(fullDist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(fullDist, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }

    if (h_block != nullptr) {
        cudaFreeHost(h_block);
    }
    if (d_pivotBlock != nullptr) {
        cudaFree(d_pivotBlock);
    }
    if (d_path != nullptr) {
        cudaFree(d_path);
    }
    if (d_dist != nullptr) {
        cudaFree(d_dist);
    }
    if (shmWin != MPI_WIN_NULL) {
        MPI_Win_free(&shmWin);
    }
    if (localComm != MPI_COMM_NULL) {
        MPI_Comm_free(&localComm);
    }

    MPI_Finalize();
    return rc;
}
