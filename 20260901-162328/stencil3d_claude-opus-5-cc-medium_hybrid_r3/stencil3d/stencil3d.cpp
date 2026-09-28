#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        const cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,        \
                    __LINE__);                                                                     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                          \
        }                                                                                          \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize the slab of planes [z0, z0 + nzLocal) owned by this rank.
// The values match the original global initialization: grid[idx] = (idx % 19) * 1.0
void initializeGrid(std::vector<Real>& slab, const size_t nx, const size_t ny,
                    const size_t z0, const size_t nzLocal) {
    const size_t plane = nx * ny;
#pragma omp parallel for schedule(static)
    for (size_t k = 0; k < nzLocal; ++k) {
        const size_t globalBase = (z0 + k) * plane;
        Real* dst = slab.data() + k * plane;
        for (size_t i = 0; i < plane; ++i) {
            dst[i] = ((globalBase + i) % 19) * 1.0;
        }
    }
}

// 7-point stencil computation on a slab of z-planes.
// The device buffers hold nzLocal owned planes at local plane indices 1..nzLocal, surrounded by
// one halo plane on each side (local plane indices 0 and nzLocal+1). Local plane k corresponds to
// the global plane z0 + k - 1. Interior points are averaged, boundary points of the global grid
// are copied, exactly as in the serial reference implementation.
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              const int nx, const int ny, const size_t nz, const size_t z0,
                              const int kBegin, const int kEnd) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const bool xyBoundary = (x == 0 || x == nx - 1 || y == 0 || y == ny - 1);

    size_t off = static_cast<size_t>(kBegin) * plane + static_cast<size_t>(y) * nx + x;

    // Rolling registers along z: only one new plane value is loaded per iteration.
    Real bottom = input[off - plane];
    Real center = input[off];

    for (int k = kBegin; k <= kEnd; ++k, off += plane) {
        const Real top = input[off + plane];
        const size_t gz = z0 + static_cast<size_t>(k) - 1;

        if (xyBoundary || gz == 0 || gz == nz - 1) {
            output[off] = center;
        } else {
            const Real left = input[off - 1];
            const Real right = input[off + 1];
            const Real front = input[off - nx];
            const Real back = input[off + nx];
            output[off] = (center + left + right + front + back + bottom + top) / 7.0;
        }

        bottom = center;
        center = top;
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    int bad = 0;
#pragma omp parallel for schedule(static) reduction(| : bad)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real val = grid[i];
        if (std::isnan(val) || std::isinf(val)) {
            bad = 1;
        }
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for schedule(static) reduction(min : minVal) reduction(max : maxVal)
    for (size_t i = 0; i < grid.size(); ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
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

// Bind this rank to one of the GPUs available on its node.
static void selectDevice(const int worldRank) {
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(nodeRank % deviceCount));
}

int main(int argc, char** argv) {
    int mpiProvided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiProvided);
    (void)mpiProvided;

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool isRoot = (worldRank == 0);

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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (isRoot) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    selectDevice(worldRank);

    // ---- 1D domain decomposition along z ------------------------------------------------------
    // Ranks that would receive no plane at all stay idle during the computation (they only take
    // part in the final gather with a zero-sized contribution).
    const int activeRanks = static_cast<int>(std::min<size_t>(nz, static_cast<size_t>(worldSize)));
    const bool active = (worldRank < activeRanks);

    size_t z0 = 0;
    size_t nzLocal = 0;
    if (active) {
        const size_t base = nz / static_cast<size_t>(activeRanks);
        const size_t rem = nz % static_cast<size_t>(activeRanks);
        const size_t r = static_cast<size_t>(worldRank);
        nzLocal = base + (r < rem ? 1 : 0);
        z0 = r * base + std::min(r, rem);
    }

    const int lowerNeighbor = (active && worldRank > 0) ? worldRank - 1 : MPI_PROC_NULL;
    const int upperNeighbor = (active && worldRank + 1 < activeRanks) ? worldRank + 1 : MPI_PROC_NULL;

    // ---- allocations --------------------------------------------------------------------------
    const size_t slabPlanes = nzLocal + 2;  // owned planes + one halo plane on each side
    const size_t slabElems = slabPlanes * plane;

    Real* dA = nullptr;
    Real* dB = nullptr;
    Real* hSendLow = nullptr;
    Real* hSendHigh = nullptr;
    Real* hRecvLow = nullptr;
    Real* hRecvHigh = nullptr;
    cudaStream_t computeStream = nullptr;
    cudaStream_t haloStream = nullptr;
    cudaEvent_t computeDone = nullptr;
    cudaEvent_t haloDone = nullptr;

    if (active) {
        CUDA_CHECK(cudaMalloc(&dA, slabElems * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&dB, slabElems * sizeof(Real)));
        CUDA_CHECK(cudaMemset(dA, 0, slabElems * sizeof(Real)));
        CUDA_CHECK(cudaMemset(dB, 0, slabElems * sizeof(Real)));
        CUDA_CHECK(cudaHostAlloc(&hSendLow, plane * sizeof(Real), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&hSendHigh, plane * sizeof(Real), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&hRecvLow, plane * sizeof(Real), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&hRecvHigh, plane * sizeof(Real), cudaHostAllocDefault));
        CUDA_CHECK(cudaStreamCreate(&computeStream));
        CUDA_CHECK(cudaStreamCreate(&haloStream));
        CUDA_CHECK(cudaEventCreateWithFlags(&computeDone, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&haloDone, cudaEventDisableTiming));
    }

    // Initialize
    if (isRoot) printf("Initializing grid...\n");
    if (active) {
        std::vector<Real> hostSlab(nzLocal * plane);
        initializeGrid(hostSlab, nx, ny, z0, nzLocal);
        CUDA_CHECK(cudaMemcpy(dA + plane, hostSlab.data(), nzLocal * plane * sizeof(Real),
                              cudaMemcpyHostToDevice));
    }

    // ---- stencil iterations -------------------------------------------------------------------
    if (isRoot) printf("Running stencil computation...\n");

    const dim3 block(32, 8, 1);
    const dim3 grid((static_cast<unsigned>(nx) + block.x - 1) / block.x,
                    (static_cast<unsigned>(ny) + block.y - 1) / block.y, 1);
    const int kLast = static_cast<int>(nzLocal);

    auto start = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    for (int iter = 0; iter < iterations; ++iter) {
        if (!active) break;

        Real* in = (iter % 2 == 0) ? dA : dB;
        Real* out = (iter % 2 == 0) ? dB : dA;

        // Stage the two planes that neighbors need, and exchange them.
        CUDA_CHECK(cudaMemcpyAsync(hSendLow, in + plane, plane * sizeof(Real),
                                   cudaMemcpyDeviceToHost, haloStream));
        CUDA_CHECK(cudaMemcpyAsync(hSendHigh, in + static_cast<size_t>(kLast) * plane,
                                   plane * sizeof(Real), cudaMemcpyDeviceToHost, haloStream));
        CUDA_CHECK(cudaStreamSynchronize(haloStream));

        MPI_Request requests[4];
        MPI_Irecv(hRecvLow, static_cast<int>(plane), MPI_DOUBLE, lowerNeighbor, 0, MPI_COMM_WORLD,
                  &requests[0]);
        MPI_Irecv(hRecvHigh, static_cast<int>(plane), MPI_DOUBLE, upperNeighbor, 1, MPI_COMM_WORLD,
                  &requests[1]);
        MPI_Isend(hSendLow, static_cast<int>(plane), MPI_DOUBLE, lowerNeighbor, 1, MPI_COMM_WORLD,
                  &requests[2]);
        MPI_Isend(hSendHigh, static_cast<int>(plane), MPI_DOUBLE, upperNeighbor, 0, MPI_COMM_WORLD,
                  &requests[3]);

        // Planes that do not depend on the halos can be updated while the exchange is in flight.
        if (kLast >= 3) {
            stencilKernel<<<grid, block, 0, computeStream>>>(in, out, static_cast<int>(nx),
                                                             static_cast<int>(ny), nz, z0, 2,
                                                             kLast - 1);
            CUDA_CHECK(cudaGetLastError());
        }

        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

        CUDA_CHECK(cudaMemcpyAsync(in, hRecvLow, plane * sizeof(Real),
                                   cudaMemcpyHostToDevice, haloStream));
        CUDA_CHECK(cudaMemcpyAsync(in + static_cast<size_t>(kLast + 1) * plane, hRecvHigh,
                                   plane * sizeof(Real), cudaMemcpyHostToDevice, haloStream));

        stencilKernel<<<grid, block, 0, haloStream>>>(in, out, static_cast<int>(nx),
                                                      static_cast<int>(ny), nz, z0, 1, 1);
        CUDA_CHECK(cudaGetLastError());
        if (kLast >= 2) {
            stencilKernel<<<grid, block, 0, haloStream>>>(in, out, static_cast<int>(nx),
                                                          static_cast<int>(ny), nz, z0, kLast,
                                                          kLast);
            CUDA_CHECK(cudaGetLastError());
        }

        // Both streams must finish before the next iteration reads `out`.
        CUDA_CHECK(cudaEventRecord(computeDone, computeStream));
        CUDA_CHECK(cudaStreamWaitEvent(haloStream, computeDone, 0));
        CUDA_CHECK(cudaEventRecord(haloDone, haloStream));
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, haloDone, 0));
    }

    if (active) {
        CUDA_CHECK(cudaStreamSynchronize(haloStream));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (isRoot) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ---- gather the final grid on the root ----------------------------------------------------
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        const Real* dFinal = (iterations % 2 == 0) ? dA : dB;
        std::vector<Real> hostSlab(active ? nzLocal * plane : 0);
        if (active) {
            CUDA_CHECK(cudaMemcpy(hostSlab.data(), dFinal + plane, nzLocal * plane * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
        }

        std::vector<int> counts(worldSize, 0);
        std::vector<int> displs(worldSize, 0);
        const int localCount = static_cast<int>(nzLocal * plane);
        MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (isRoot) {
            finalGrid.resize(gridSize);
            int offset = 0;
            for (int r = 0; r < worldSize; ++r) {
                displs[r] = offset;
                offset += counts[r];
            }
        }
        MPI_Gatherv(hostSlab.data(), localCount, MPI_DOUBLE, finalGrid.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (isRoot) {
        // Print results for external validation
        if (printResults) {
            print_results(finalGrid, "Grid");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (active) {
        CUDA_CHECK(cudaEventDestroy(computeDone));
        CUDA_CHECK(cudaEventDestroy(haloDone));
        CUDA_CHECK(cudaStreamDestroy(computeStream));
        CUDA_CHECK(cudaStreamDestroy(haloStream));
        CUDA_CHECK(cudaFreeHost(hSendLow));
        CUDA_CHECK(cudaFreeHost(hSendHigh));
        CUDA_CHECK(cudaFreeHost(hRecvLow));
        CUDA_CHECK(cudaFreeHost(hRecvHigh));
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
    }

    MPI_Finalize();
    return exitCode;
}
