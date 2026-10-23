#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err__));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

#define MPI_CHECK(call)                                                         \
    do {                                                                        \
        int err__ = (call);                                                     \
        if (err__ != MPI_SUCCESS) {                                             \
            char msg__[MPI_MAX_ERROR_STRING];                                   \
            int len__ = 0;                                                      \
            MPI_Error_string(err__, msg__, &len__);                             \
            fprintf(stderr, "MPI error %s:%d: %.*s\n", __FILE__, __LINE__,      \
                    len__, msg__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kBlockZ = 4;

__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const int nx, const int ny,
                              const int nzGlobal,
                              const int z0,
                              const int localZStart,
                              const int localZCount) {
    const int groupStart = localZStart + static_cast<int>(blockIdx.z) * kBlockZ;
    const int remaining = localZStart + localZCount - groupStart;
    if (remaining <= 0) {
        return;
    }
    const int computed = remaining < kBlockZ ? remaining : kBlockZ;

    constexpr int sharedX = kBlockX + 2;
    constexpr int sharedY = kBlockY + 2;
    constexpr int sharedPlane = sharedX * sharedY;
    __shared__ Real tile[(kBlockZ + 2) * sharedPlane];

    const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    const int numThreads = static_cast<int>(blockDim.x * blockDim.y);
    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);

    const int xBase = static_cast<int>(blockIdx.x) * kBlockX;
    const int yBase = static_cast<int>(blockIdx.y) * kBlockY;
    const int tileZCount = computed + 2;
    const int tileElems = tileZCount * sharedPlane;

    for (int e = tid; e < tileElems; e += numThreads) {
        const int sx = e % sharedX;
        const int rem = e / sharedX;
        const int sy = rem % sharedY;
        const int dz = rem / sharedY;

        const int gx = xBase + sx;
        const int gy = yBase + sy;
        const int localZ = groupStart - 1 + dz;

        Real value = static_cast<Real>(0);
        if (gx < nx && gy < ny) {
            const size_t idx = static_cast<size_t>(localZ) * plane
                             + static_cast<size_t>(gy) * static_cast<size_t>(nx)
                             + static_cast<size_t>(gx);
            value = input[idx];
        }
        tile[e] = value;
    }
    __syncthreads();

    const int x = xBase + static_cast<int>(threadIdx.x) + 1;
    const int y = yBase + static_cast<int>(threadIdx.y) + 1;
    if (x >= nx - 1 || y >= ny - 1) {
        return;
    }

    const int sx = static_cast<int>(threadIdx.x) + 1;
    const int sy = static_cast<int>(threadIdx.y) + 1;

    for (int dz = 0; dz < computed; ++dz) {
        const int localZ = groupStart + dz;
        const int globalZ = z0 + localZ - 1;
        if (globalZ <= 0 || globalZ >= nzGlobal - 1) {
            continue;
        }

        const Real* t = tile + (dz + 1) * sharedPlane;
        const int centerIdx = sy * sharedX + sx;

        const Real center = t[centerIdx];
        const Real left = t[centerIdx - 1];
        const Real right = t[centerIdx + 1];
        const Real front = t[centerIdx - sharedX];
        const Real back = t[centerIdx + sharedX];
        const Real bottom = t[centerIdx - sharedPlane];
        const Real top = t[centerIdx + sharedPlane];

        const size_t idx = static_cast<size_t>(localZ) * plane
                         + static_cast<size_t>(y) * static_cast<size_t>(nx)
                         + static_cast<size_t>(x);
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

void initializeLocalGrid(std::vector<Real>& grid,
                         const size_t nx, const size_t ny,
                         const size_t nzLocal, const size_t z0,
                         const size_t nz) {
    const size_t plane = nx * ny;
    const int lzCount = static_cast<int>(nzLocal + 2);
    const int yCount = static_cast<int>(ny);
    const int xCount = static_cast<int>(nx);

    #pragma omp parallel for collapse(3) schedule(static)
    for (int lz = 0; lz < lzCount; ++lz) {
        for (int y = 0; y < yCount; ++y) {
            for (int x = 0; x < xCount; ++x) {
                const long long globalZ = static_cast<long long>(z0)
                                        + static_cast<long long>(lz) - 1;
                const size_t localIdx = static_cast<size_t>(lz) * plane
                                      + static_cast<size_t>(y) * nx
                                      + static_cast<size_t>(x);
                if (globalZ >= 0 && static_cast<size_t>(globalZ) < nz) {
                    const size_t globalIdx = static_cast<size_t>(globalZ) * plane
                                           + static_cast<size_t>(y) * nx
                                           + static_cast<size_t>(x);
                    grid[localIdx] = static_cast<Real>(globalIdx % 19);
                } else {
                    grid[localIdx] = static_cast<Real>(0);
                }
            }
        }
    }
}

void exchangeHalos(Real* deviceGrid,
                   const size_t nzLocal,
                   const int prevRank,
                   const int nextRank,
                   Real* sendLower,
                   Real* sendUpper,
                   Real* recvLower,
                   Real* recvUpper,
                   const size_t plane,
                   MPI_Comm comm) {
    const size_t planeBytes = plane * sizeof(Real);
    const int planeCount = static_cast<int>(plane);

    if (prevRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(sendLower, deviceGrid + plane, planeBytes,
                              cudaMemcpyDeviceToHost));
    }
    if (nextRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(sendUpper, deviceGrid + nzLocal * plane, planeBytes,
                              cudaMemcpyDeviceToHost));
    }

    // Send my lower plane to prev; receive my upper halo from next.
    MPI_CHECK(MPI_Sendrecv(sendLower, planeCount, MPI_DOUBLE, prevRank, 11,
                           recvUpper, planeCount, MPI_DOUBLE, nextRank, 11,
                           comm, MPI_STATUS_IGNORE));

    // Send my upper plane to next; receive my lower halo from prev.
    MPI_CHECK(MPI_Sendrecv(sendUpper, planeCount, MPI_DOUBLE, nextRank, 12,
                           recvLower, planeCount, MPI_DOUBLE, prevRank, 12,
                           comm, MPI_STATUS_IGNORE));

    if (prevRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(deviceGrid, recvLower, planeBytes,
                              cudaMemcpyHostToDevice));
    }
    if (nextRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(deviceGrid + (nzLocal + 1) * plane, recvUpper,
                              planeBytes, cudaMemcpyHostToDevice));
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    const long long count = static_cast<long long>(grid.size());

    // 1. No NaN or Inf values
    bool hasInvalid = false;
    #pragma omp parallel for reduction(||:hasInvalid) schedule(static)
    for (long long i = 0; i < count; ++i) {
        const Real val = grid[static_cast<size_t>(i)];
        if (std::isnan(val) || std::isinf(val)) {
            hasInvalid = true;
        }
    }
    if (hasInvalid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (long long i = 0; i < count; ++i) {
        const Real val = grid[static_cast<size_t>(i)];
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank = 0;
    int numRanks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &localComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA device found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    const size_t baseZ = nz / static_cast<size_t>(numRanks);
    const size_t remainderZ = nz % static_cast<size_t>(numRanks);
    const size_t z0 = static_cast<size_t>(rank) * baseZ
                    + std::min(static_cast<size_t>(rank), remainderZ);
    const size_t nzLocal = baseZ
                         + (static_cast<size_t>(rank) < remainderZ ? 1 : 0);
    const int activeRanks = static_cast<int>(std::min<size_t>(nz,
                                      static_cast<size_t>(numRanks)));
    const bool active = rank < activeRanks;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        fflush(stdout);
    }

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const size_t localElems = active ? plane * (nzLocal + 2) : 0;
    const size_t localOwnedElems = active ? plane * nzLocal : 0;

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    Real* sendLower = nullptr;
    Real* sendUpper = nullptr;
    Real* recvLower = nullptr;
    Real* recvUpper = nullptr;

    if (active) {
        std::vector<Real> hostGrid(localElems);
        initializeLocalGrid(hostGrid, nx, ny, nzLocal, z0, nz);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid1),
                              localElems * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid2),
                              localElems * sizeof(Real)));
        CUDA_CHECK(cudaMemcpy(deviceGrid1, hostGrid.data(),
                              localElems * sizeof(Real),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceGrid2, hostGrid.data(),
                              localElems * sizeof(Real),
                              cudaMemcpyHostToDevice));
        std::vector<Real>().swap(hostGrid);

        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&sendLower),
                                  plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&sendUpper),
                                  plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&recvLower),
                                  plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&recvUpper),
                                  plane * sizeof(Real)));
    }

    const int prevRank = (active && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int nextRank = (active && rank + 1 < activeRanks) ? rank + 1
                                                            : MPI_PROC_NULL;

    Real* deviceInput = deviceGrid1;
    Real* deviceOutput = deviceGrid2;

    auto start = std::chrono::high_resolution_clock::now();

    if (active && nx > 2 && ny > 2 && nz > 2) {
        const size_t interiorStart = std::max<size_t>(z0, 1);
        const size_t interiorEnd = std::min<size_t>(z0 + nzLocal - 1, nz - 2);

        int localZStart = 1;
        int localZCount = 0;
        if (interiorEnd >= interiorStart) {
            localZStart = static_cast<int>(interiorStart - z0 + 1);
            localZCount = static_cast<int>(interiorEnd - interiorStart + 1);
        }

        const dim3 block(kBlockX, kBlockY, 1);
        const dim3 grid((static_cast<unsigned int>(nx - 2) + kBlockX - 1) / kBlockX,
                        (static_cast<unsigned int>(ny - 2) + kBlockY - 1) / kBlockY,
                        (static_cast<unsigned int>(localZCount) + kBlockZ - 1) / kBlockZ);

        for (int iter = 0; iter < iterations; ++iter) {
            if (prevRank != MPI_PROC_NULL || nextRank != MPI_PROC_NULL) {
                exchangeHalos(deviceInput, nzLocal, prevRank, nextRank,
                              sendLower, sendUpper, recvLower, recvUpper,
                              plane, MPI_COMM_WORLD);
            }

            if (localZCount > 0) {
                stencilKernel<<<grid, block>>>(deviceInput, deviceOutput,
                                               static_cast<int>(nx),
                                               static_cast<int>(ny),
                                               static_cast<int>(nz),
                                               static_cast<int>(z0),
                                               localZStart, localZCount);
                CUDA_CHECK(cudaGetLastError());
            }

            std::swap(deviceInput, deviceOutput);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long long localMs = static_cast<long long>(duration.count());
    long long globalMs = localMs;
    MPI_CHECK(MPI_Reduce(&localMs, &globalMs, 1, MPI_LONG_LONG, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    std::vector<Real> fullGrid;
    std::vector<Real> ownedGrid;
    if (printResults || validate) {
        if (active) {
            ownedGrid.resize(localOwnedElems);
            CUDA_CHECK(cudaMemcpy(ownedGrid.data(),
                                  deviceInput + plane,
                                  localOwnedElems * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
        }

        std::vector<int> counts(static_cast<size_t>(numRanks), 0);
        std::vector<int> displs(static_cast<size_t>(numRanks), 0);
        for (int r = 0; r < numRanks; ++r) {
            const size_t base = nz / static_cast<size_t>(numRanks);
            const size_t rem = nz % static_cast<size_t>(numRanks);
            const size_t rz0 = static_cast<size_t>(r) * base
                             + std::min(static_cast<size_t>(r), rem);
            const size_t rnz = base
                             + (static_cast<size_t>(r) < rem ? 1 : 0);
            counts[static_cast<size_t>(r)] = static_cast<int>(plane * rnz);
            displs[static_cast<size_t>(r)] = static_cast<int>(plane * rz0);
        }

        if (rank == 0) {
            fullGrid.resize(gridSize);
        }
        MPI_CHECK(MPI_Gatherv(active ? ownedGrid.data() : nullptr,
                              static_cast<int>(localOwnedElems), MPI_DOUBLE,
                              rank == 0 ? fullGrid.data() : nullptr,
                              counts.data(), displs.data(), MPI_DOUBLE,
                              0, MPI_COMM_WORLD));
    }

    int returnCode = 0;

    if (rank == 0) {
        printf("Computation time: %lld ms\n", globalMs);

        // Calculate performance metrics
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2))
                                 * static_cast<double>(iterations);
        const double mcups = cellUpdates / (static_cast<double>(globalMs) / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(fullGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            if (validateResult(fullGrid, nx, ny, nz)) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                returnCode = 1;
            }
        }
        fflush(stdout);
    }

    MPI_CHECK(MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD));

    if (active) {
        CUDA_CHECK(cudaFreeHost(sendLower));
        CUDA_CHECK(cudaFreeHost(sendUpper));
        CUDA_CHECK(cudaFreeHost(recvLower));
        CUDA_CHECK(cudaFreeHost(recvUpper));
        CUDA_CHECK(cudaFree(deviceGrid1));
        CUDA_CHECK(cudaFree(deviceGrid2));
    }

    MPI_CHECK(MPI_Comm_free(&localComm));
    MPI_CHECK(MPI_Finalize());

    return returnCode;
}
