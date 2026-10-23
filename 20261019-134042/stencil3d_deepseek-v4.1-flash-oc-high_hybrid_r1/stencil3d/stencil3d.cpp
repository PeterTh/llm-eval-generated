#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

static void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), file, line);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) checkCuda((call), __FILE__, __LINE__)

struct ZPartition {
    size_t start;
    size_t end;

    size_t count() const { return end - start; }
};

static ZPartition partitionZ(const size_t nz, const int rank, const int nranks) {
    const size_t p = static_cast<size_t>(nranks);
    const size_t base = nz / p;
    const size_t rem = nz % p;
    const size_t r = static_cast<size_t>(rank);

    size_t count;
    size_t start;
    if (r < rem) {
        count = base + 1;
        start = r * count;
    } else {
        count = base;
        start = rem * (base + 1) + (r - rem) * base;
    }
    return {start, start + count};
}

static int ownerOfZ(const size_t z, const size_t nz, const int nranks) {
    for (int r = 0; r < nranks; ++r) {
        const ZPartition part = partitionZ(nz, r, nranks);
        if (z >= part.start && z < part.end) {
            return r;
        }
    }
    return -1;
}

// 7-point stencil.  Only real (non-ghost) local planes are written.  Global
// boundaries are copied so that the double-buffered arrays always contain a
// complete, valid grid, exactly like the reference implementation.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx,
                              const size_t ny,
                              const size_t nz,
                              const size_t zStart,
                              const size_t localZBase,
                              const size_t localZCount,
                              const size_t plane) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t lz = localZBase + static_cast<size_t>(blockIdx.z);

    if (x >= nx || y >= ny || lz >= localZCount) {
        return;
    }

    const size_t gz = zStart + lz;
    const size_t dst = (lz + 1) * plane + y * nx + x;

    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
        gz == 0 || gz + 1 == nz) {
        output[dst] = input[dst];
        return;
    }

    const Real center = input[dst];
    const Real left = input[dst - 1];
    const Real right = input[dst + 1];
    const Real front = input[dst - nx];
    const Real back = input[dst + nx];
    const Real bottom = input[dst - plane];
    const Real top = input[dst + plane];

    output[dst] = (center + left + right + front + back + bottom + top) / 7.0;
}

static void launchStencil(const Real* input,
                          Real* output,
                          const size_t nx,
                          const size_t ny,
                          const size_t nz,
                          const size_t zStart,
                          const size_t localZCount) {
    if (localZCount == 0) {
        return;
    }

    const size_t plane = nx * ny;
    const dim3 block(32, 8, 1);
    const dim3 gridXY((nx + block.x - 1) / block.x,
                      (ny + block.y - 1) / block.y,
                      1);
    const size_t maxGridZ = 65535;

    for (size_t zBase = 0; zBase < localZCount; zBase += maxGridZ) {
        const size_t zChunk = (localZCount - zBase < maxGridZ)
                                  ? (localZCount - zBase)
                                  : maxGridZ;
        const dim3 grid(gridXY.x, gridXY.y, static_cast<unsigned int>(zChunk));
        stencilKernel<<<grid, block>>>(input, output, nx, ny, nz, zStart,
                                       zBase, localZCount, plane);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Exchange one z-plane with each neighbouring rank.  The communication is
// non-blocking so that the lower and upper halo transfers can overlap.
static void exchangeHalos(Real* deviceGrid,
                          const size_t localZCount,
                          const size_t plane,
                          const size_t zStart,
                          const size_t zEnd,
                          const size_t nz,
                          const int nranks,
                          Real* sendLower,
                          Real* recvLower,
                          Real* sendUpper,
                          Real* recvUpper) {
    if (localZCount == 0) {
        return;
    }

    const int planeCount = static_cast<int>(plane);
    constexpr int tagFirstPlane = 10;
    constexpr int tagLastPlane = 11;

    MPI_Request requests[4];
    int requestCount = 0;

    if (zStart > 0) {
        const int lower = ownerOfZ(zStart - 1, nz, nranks);
        CUDA_CHECK(cudaMemcpy(sendLower, deviceGrid + plane,
                              plane * sizeof(Real), cudaMemcpyDeviceToHost));
        MPI_Irecv(recvLower, planeCount, MPI_DOUBLE, lower, tagLastPlane,
                  MPI_COMM_WORLD, &requests[requestCount++]);
        MPI_Isend(sendLower, planeCount, MPI_DOUBLE, lower, tagFirstPlane,
                  MPI_COMM_WORLD, &requests[requestCount++]);
    }

    if (zEnd < nz) {
        const int upper = ownerOfZ(zEnd, nz, nranks);
        CUDA_CHECK(cudaMemcpy(sendUpper, deviceGrid + localZCount * plane,
                              plane * sizeof(Real), cudaMemcpyDeviceToHost));
        MPI_Irecv(recvUpper, planeCount, MPI_DOUBLE, upper, tagFirstPlane,
                  MPI_COMM_WORLD, &requests[requestCount++]);
        MPI_Isend(sendUpper, planeCount, MPI_DOUBLE, upper, tagLastPlane,
                  MPI_COMM_WORLD, &requests[requestCount++]);
    }

    if (requestCount > 0) {
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    }

    if (zStart > 0) {
        CUDA_CHECK(cudaMemcpy(deviceGrid, recvLower, plane * sizeof(Real),
                              cudaMemcpyHostToDevice));
    }
    if (zEnd < nz) {
        CUDA_CHECK(cudaMemcpy(deviceGrid + (localZCount + 1) * plane,
                              recvUpper, plane * sizeof(Real),
                              cudaMemcpyHostToDevice));
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Bind each rank to a GPU on its node.  Local ranks are used so that a
    // multi-GPU node is fully utilised before devices are shared.
    int localRank = 0;
    int localSize = 1;
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &localComm);
    if (localComm != MPI_COMM_NULL) {
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_size(localComm, &localSize);
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (getenv("OMP_NUM_THREADS") == nullptr) {
        const int threads = std::max(1, omp_get_num_procs() / std::max(1, localSize));
        omp_set_num_threads(threads);
    }

    if (localComm != MPI_COMM_NULL) {
        MPI_Comm_free(&localComm);
    }

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

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t plane = nx * ny;
    const size_t gridSize = nx * ny * nz;
    const ZPartition part = partitionZ(nz, rank, nranks);
    const size_t zStart = part.start;
    const size_t zEnd = part.end;
    const size_t localZCount = part.count();

    // Double-buffered local grids with one ghost plane on each side in z.
    const size_t localAlloc = (localZCount + 2) * plane;
    Real* dGrid[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dGrid[0]),
                          localAlloc * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dGrid[1]),
                          localAlloc * sizeof(Real)));

    // Initialize the real part of the local domain.  The global linear index
    // exactly matches the reference implementation, independent of the
    // number of ranks.
    if (localZCount > 0) {
        std::vector<Real> initial(localZCount * plane);
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t lz = 0; lz < localZCount; ++lz) {
            for (size_t rem = 0; rem < plane; ++rem) {
                const size_t globalIdx = (zStart + lz) * plane + rem;
                initial[lz * plane + rem] = static_cast<Real>(globalIdx % 19);
            }
        }
        CUDA_CHECK(cudaMemcpy(dGrid[0] + plane, initial.data(),
                              localZCount * plane * sizeof(Real),
                              cudaMemcpyHostToDevice));
    }

    // Pinned host staging buffers for the two halo planes.
    Real* sendLower = nullptr;
    Real* recvLower = nullptr;
    Real* sendUpper = nullptr;
    Real* recvUpper = nullptr;
    if (plane > 0) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&sendLower),
                                  plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&recvLower),
                                  plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&sendUpper),
                                  plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&recvUpper),
                                  plane * sizeof(Real)));
    }

    if (rank == 0) {
        printf("Initializing grid...\n");
        printf("Running stencil computation...\n");
    }

    // Run stencil iterations
    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    int inputBuffer = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        Real* input = dGrid[inputBuffer];
        Real* output = dGrid[1 - inputBuffer];

        exchangeHalos(input, localZCount, plane, zStart, zEnd, nz, nranks,
                      sendLower, recvLower, sendUpper, recvUpper);
        launchStencil(input, output, nx, ny, nz, zStart, localZCount);

        inputBuffer = 1 - inputBuffer;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - startTime;

    double maxElapsed = elapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    const long long durationMs =
        static_cast<long long>(std::llround(maxElapsed * 1000.0));

    if (rank == 0) {
        printf("Computation time: %lld ms\n", durationMs);
        const double cellUpdates =
            static_cast<double>((nx > 2 ? nx - 2 : 0) *
                                (ny > 2 ? ny - 2 : 0) *
                                (nz > 2 ? nz - 2 : 0)) *
            static_cast<double>(iterations);
        const double mcups =
            cellUpdates / (static_cast<double>(durationMs) / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const Real* finalDeviceGrid = dGrid[inputBuffer];

    if (printResults || validate) {
        std::vector<Real> localFinal(localZCount * plane);
        if (localZCount > 0) {
            CUDA_CHECK(cudaMemcpy(localFinal.data(), finalDeviceGrid + plane,
                                  localZCount * plane * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
        }

        std::vector<int> counts(nranks, 0);
        std::vector<int> displacements(nranks, 0);
        for (int r = 0; r < nranks; ++r) {
            const ZPartition p = partitionZ(nz, r, nranks);
            counts[r] = static_cast<int>(p.count() * plane);
            displacements[r] = static_cast<int>(p.start * plane);
        }

        std::vector<Real> globalFinal;
        if (rank == 0) {
            globalFinal.resize(gridSize);
        }

        MPI_Gatherv(localFinal.data(), counts[rank], MPI_DOUBLE,
                    rank == 0 ? globalFinal.data() : nullptr, counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(globalFinal, "Grid");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(globalFinal, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }

                CUDA_CHECK(cudaFreeHost(sendLower));
                CUDA_CHECK(cudaFreeHost(recvLower));
                CUDA_CHECK(cudaFreeHost(sendUpper));
                CUDA_CHECK(cudaFreeHost(recvUpper));
                CUDA_CHECK(cudaFree(dGrid[0]));
                CUDA_CHECK(cudaFree(dGrid[1]));
                MPI_Finalize();
                return valid ? 0 : 1;
            }
        }
    }

    CUDA_CHECK(cudaFreeHost(sendLower));
    CUDA_CHECK(cudaFreeHost(recvLower));
    CUDA_CHECK(cudaFreeHost(sendUpper));
    CUDA_CHECK(cudaFreeHost(recvUpper));
    CUDA_CHECK(cudaFree(dGrid[0]));
    CUDA_CHECK(cudaFree(dGrid[1]));

    MPI_Finalize();
    return 0;
}
