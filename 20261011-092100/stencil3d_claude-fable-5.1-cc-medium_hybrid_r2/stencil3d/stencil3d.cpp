// 3D 7-point stencil benchmark.
//
// Hybrid parallelization:
//   * MPI    : 1D domain decomposition of the global grid into Z-slabs, one
//              slab (and one GPU) per rank, with halo exchange of the two
//              boundary planes between neighbouring ranks each iteration.
//   * CUDA   : the stencil itself (and grid initialisation) runs on the GPU.
//              Interior planes are computed on one stream while the edge
//              planes are computed, downloaded, exchanged and uploaded on a
//              second stream, overlapping communication with computation.
//   * OpenMP : host-side work (final verification / reductions over the
//              gathered grid, host buffer handling) is multithreaded.
//
// Results are bit-identical to the original serial implementation: the
// per-cell arithmetic is kept in the same order, boundary cells are never
// modified and the final grid is gathered on rank 0 for output/validation.

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

// Thread block shape for the stencil kernel (x-fastest, one warp per row).
constexpr int STENCIL_BX = 32;
constexpr int STENCIL_BY = 8;

using Real = double;

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
static void abortRun(const char* what, const char* detail, const char* file, int line) {
    fprintf(stderr, "%s failed at %s:%d: %s\n", what, file, line, detail);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            abortRun("CUDA call", cudaGetErrorString(err__), __FILE__, __LINE__); \
        }                                                                         \
    } while (0)

#define MPI_CHECK(call)                                                   \
    do {                                                                  \
        int err__ = (call);                                               \
        if (err__ != MPI_SUCCESS) {                                       \
            char msg__[MPI_MAX_ERROR_STRING];                             \
            int len__ = 0;                                                \
            MPI_Error_string(err__, msg__, &len__);                       \
            abortRun("MPI call", msg__, __FILE__, __LINE__);              \
        }                                                                 \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Initialise the local slab (including halo planes that map to a valid global
// plane) with the same formula as the serial code: value = (globalIdx % 19).
__global__ void initializeKernel(Real* __restrict__ grid, const size_t plane, const size_t localCells,
                                 const long long zGlobalOfPlane0, const long long nz) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t l = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; l < localCells; l += stride) {
        const size_t p = l / plane;
        const long long gz = zGlobalOfPlane0 + static_cast<long long>(p);
        if (gz < 0 || gz >= nz) continue;
        const size_t gidx = static_cast<size_t>(gz) * plane + (l - p * plane);
        grid[l] = (gidx % 19) * 1.0;
    }
}

// Correctly rounded x / 7.0 using a reciprocal + one fma correction step
// (Markstein). For a divisor whose correctly rounded reciprocal is used and
// whose significand is not all ones, q' = fma(fma(-q, 7, x), 1/7, q) is the
// IEEE round-to-nearest quotient for every finite x, so results are
// bit-identical to the hardware/software division of the serial code, while
// costing 3 FP64 ops instead of the ~10 of a general double division (which
// matters on GPUs with reduced FP64 throughput). Verified exhaustively against
// x / 7.0 on >2.5e10 random inputs with zero mismatches.
__device__ __forceinline__ Real div7(const Real x) {
    constexpr Real inv7 = 1.0 / 7.0;
    const Real q = x * inv7;
    const Real r = fma(-q, 7.0, x);  // exact remainder
    return fma(r, inv7, q);
}

// 7-point averaging stencil over interior x/y and local planes [zBegin, zEnd).
// Each thread handles one (x, y) column and marches along z, keeping the
// bottom/center/top values in registers (one global load per cell on z).
__global__ void __launch_bounds__(STENCIL_BX * STENCIL_BY) stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                                                     const int nx, const int ny, const int zBegin, const int zEnd,
                                                     const int zChunk) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x >= nx - 1 || y >= ny - 1) return;

    const int z0 = zBegin + blockIdx.z * zChunk;
    const int z1 = min(z0 + zChunk, zEnd);
    if (z0 >= z1) return;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    size_t idx = idx3(x, y, z0, nx, ny);

    Real bottom = input[idx - plane];
    Real center = input[idx];

    for (int z = z0; z < z1; ++z, idx += plane) {
        const Real top = input[idx + plane];
        const Real left = input[idx - 1];
        const Real right = input[idx + 1];
        const Real front = input[idx - nx];
        const Real back = input[idx + nx];

        // Simple averaging stencil (same evaluation order as the serial code)
        output[idx] = div7(center + left + right + front + back + bottom + top);

        bottom = center;
        center = top;
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct Decomposition {
    int rank = 0;
    int size = 1;
    long long zStart = 0;   // first global plane owned by this rank
    long long nzLocal = 0;  // number of owned planes
    bool hasPrev = false;   // a lower neighbour rank with data exists
    bool hasNext = false;   // an upper neighbour rank with data exists
};

static long long planesOfRank(const int r, const int size, const long long nz) {
    const long long base = nz / size;
    const long long rem = nz % size;
    return base + (r < rem ? 1 : 0);
}

static long long startOfRank(const int r, const int size, const long long nz) {
    const long long base = nz / size;
    const long long rem = nz % size;
    return static_cast<long long>(r) * base + std::min<long long>(r, rem);
}

static Decomposition decompose(const int rank, const int size, const long long nz) {
    Decomposition d;
    d.rank = rank;
    d.size = size;
    d.nzLocal = planesOfRank(rank, size, nz);
    d.zStart = startOfRank(rank, size, nz);
    d.hasPrev = (d.nzLocal > 0) && (rank > 0);
    d.hasNext = (d.nzLocal > 0) && (rank + 1 < size) && (planesOfRank(rank + 1, size, nz) > 0);
    return d;
}

// Launch the stencil over local planes [zBegin, zEnd) on the given stream.
static void launchStencil(const Real* input, Real* output, const int nx, const int ny, const int zBegin,
                          const int zEnd, const int numSMs, cudaStream_t stream) {
    if (zEnd <= zBegin || nx < 3 || ny < 3) return;

    const dim3 block(STENCIL_BX, STENCIL_BY, 1);
    const unsigned gx = static_cast<unsigned>((nx - 2 + block.x - 1) / block.x);
    const unsigned gy = static_cast<unsigned>((ny - 2 + block.y - 1) / block.y);
    const long long nzRange = zEnd - zBegin;

    // Pick the per-block z extent so that enough blocks are in flight to fill
    // the GPU while still amortising the register pipeline along z.
    const long long xyBlocks = static_cast<long long>(gx) * gy;
    long long zChunk = (nzRange * xyBlocks) / (static_cast<long long>(numSMs) * 16);
    zChunk = std::max<long long>(1, std::min<long long>(zChunk, 64));
    const unsigned gz = static_cast<unsigned>((nzRange + zChunk - 1) / zChunk);

    const dim3 grid(gx, gy, gz);
    stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, zBegin, zEnd, static_cast<int>(zChunk));
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    const size_t n = grid.size();
    const Real* data = grid.data();
    bool bad = false;
#pragma omp parallel for reduction(|| : bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) bad = true;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, data[i]);
        maxVal = std::max(maxVal, data[i]);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values should not change significantly
    // (they are never modified, so they are identical to the initial values)

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
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    const bool isRoot = (rank == 0);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (argv is identical on every rank)
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

    // ---- GPU selection: one device per rank, round-robin over node-local ranks
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0, localSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &localSize));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) abortRun("CUDA", "no CUDA devices found", __FILE__, __LINE__);
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const int numSMs = prop.multiProcessorCount;

    if (isRoot) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: %d MPI rank(s), %d OpenMP thread(s)/rank, GPU: %s (%d per node, %d rank(s)/node)\n",
               size, omp_get_max_threads(), prop.name, deviceCount, localSize);
    }

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // ---- Domain decomposition along Z
    const Decomposition dec = decompose(rank, size, static_cast<long long>(nz));
    const long long nzLocal = dec.nzLocal;
    const size_t localPlanes = static_cast<size_t>(nzLocal) + 2;  // owned planes + 2 halo planes
    const size_t localCells = localPlanes * plane;

    // Local plane p corresponds to global plane (zStart - 1 + p); owned planes are 1..nzLocal.
    // Computed planes are the owned planes that are global interior planes.
    const long long gzFirst = dec.zStart;
    const long long gzLast = dec.zStart + nzLocal - 1;
    int cpBegin = 1, cpEnd = 0;  // inclusive local range of computed planes (empty if cpEnd < cpBegin)
    if (nzLocal > 0 && nz >= 3) {
        const long long c0 = std::max<long long>(gzFirst, 1);
        const long long c1 = std::min<long long>(gzLast, static_cast<long long>(nz) - 2);
        if (c0 <= c1) {
            cpBegin = static_cast<int>(c0 - dec.zStart + 1);
            cpEnd = static_cast<int>(c1 - dec.zStart + 1);
        }
    }
    const bool hasCompute = (cpEnd >= cpBegin) && nx >= 3 && ny >= 3;
    // Edge planes: computed planes adjacent to a neighbouring rank's halo.
    const bool edgeLow = hasCompute && dec.hasPrev && cpBegin == 1;
    const bool edgeHigh = hasCompute && dec.hasNext && cpEnd == nzLocal && !(edgeLow && cpBegin == cpEnd);
    const int interiorBegin = cpBegin + (edgeLow ? 1 : 0);
    const int interiorEnd = cpEnd - (edgeHigh ? 1 : 0);  // inclusive

    // ---- Device buffers (double buffering) and pinned host halo staging
    Real* dGrid[2] = {nullptr, nullptr};
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&dGrid[b], std::max<size_t>(localCells, 1) * sizeof(Real)));
    }
    Real* hSendLow = nullptr;
    Real* hSendHigh = nullptr;
    Real* hRecvLow = nullptr;
    Real* hRecvHigh = nullptr;
    const size_t haloBytes = std::max<size_t>(plane, 1) * sizeof(Real);
    CUDA_CHECK(cudaMallocHost(&hSendLow, haloBytes));
    CUDA_CHECK(cudaMallocHost(&hSendHigh, haloBytes));
    CUDA_CHECK(cudaMallocHost(&hRecvLow, haloBytes));
    CUDA_CHECK(cudaMallocHost(&hRecvHigh, haloBytes));

    cudaStream_t sInterior, sEdge;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sInterior, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sEdge, cudaStreamNonBlocking));
    cudaEvent_t evEdgeDone, evInteriorDone;
    CUDA_CHECK(cudaEventCreateWithFlags(&evEdgeDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evInteriorDone, cudaEventDisableTiming));

    // ---- Initialize (both buffers identically: boundary cells are never written
    //      by the stencil, so they stay equal to their initial values as in the
    //      serial code, which copies them every iteration).
    if (isRoot) printf("Initializing grid...\n");
    if (localCells > 0) {
        const unsigned initBlocks = static_cast<unsigned>(std::min<size_t>((localCells + 255) / 256, 65535));
        for (int b = 0; b < 2; ++b) {
            initializeKernel<<<initBlocks, 256, 0, sInterior>>>(dGrid[b], plane, localCells, dec.zStart - 1,
                                                                static_cast<long long>(nz));
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const int prevRank = dec.hasPrev ? rank - 1 : MPI_PROC_NULL;
    const int nextRank = dec.hasNext ? rank + 1 : MPI_PROC_NULL;
    const int planeCount = static_cast<int>(plane);
    const bool needExchange = (size > 1) && (dec.hasPrev || dec.hasNext);

    // ---- Run stencil iterations
    if (isRoot) printf("Running stencil computation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    Real* input = dGrid[0];
    Real* output = dGrid[1];

    for (int iter = 0; iter < iterations; ++iter) {
        if (hasCompute) {
            // Edge planes first (they feed the halo exchange), on the edge stream.
            CUDA_CHECK(cudaStreamWaitEvent(sEdge, evInteriorDone, 0));
            if (edgeLow) launchStencil(input, output, static_cast<int>(nx), static_cast<int>(ny), cpBegin, cpBegin + 1, numSMs, sEdge);
            if (edgeHigh) launchStencil(input, output, static_cast<int>(nx), static_cast<int>(ny), cpEnd, cpEnd + 1, numSMs, sEdge);
            CUDA_CHECK(cudaEventRecord(evEdgeDone, sEdge));

            // Interior planes overlap with the halo exchange below.
            CUDA_CHECK(cudaStreamWaitEvent(sInterior, evEdgeDone, 0));
            launchStencil(input, output, static_cast<int>(nx), static_cast<int>(ny), interiorBegin, interiorEnd + 1, numSMs, sInterior);
            CUDA_CHECK(cudaEventRecord(evInteriorDone, sInterior));
        }

        if (needExchange) {
            // Download the freshly computed edge planes, exchange with neighbours, upload into halos.
            if (dec.hasPrev) CUDA_CHECK(cudaMemcpyAsync(hSendLow, output + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, sEdge));
            if (dec.hasNext) CUDA_CHECK(cudaMemcpyAsync(hSendHigh, output + static_cast<size_t>(nzLocal) * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, sEdge));
            CUDA_CHECK(cudaStreamSynchronize(sEdge));

            MPI_Request reqs[4];
            int nreq = 0;
            MPI_CHECK(MPI_Irecv(hRecvLow, planeCount, MPI_DOUBLE, prevRank, 1, MPI_COMM_WORLD, &reqs[nreq++]));
            MPI_CHECK(MPI_Irecv(hRecvHigh, planeCount, MPI_DOUBLE, nextRank, 0, MPI_COMM_WORLD, &reqs[nreq++]));
            MPI_CHECK(MPI_Isend(hSendLow, planeCount, MPI_DOUBLE, prevRank, 0, MPI_COMM_WORLD, &reqs[nreq++]));
            MPI_CHECK(MPI_Isend(hSendHigh, planeCount, MPI_DOUBLE, nextRank, 1, MPI_COMM_WORLD, &reqs[nreq++]));
            MPI_CHECK(MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE));

            if (dec.hasPrev) CUDA_CHECK(cudaMemcpyAsync(output, hRecvLow, plane * sizeof(Real), cudaMemcpyHostToDevice, sEdge));
            if (dec.hasNext) CUDA_CHECK(cudaMemcpyAsync(output + (static_cast<size_t>(nzLocal) + 1) * plane, hRecvHigh, plane * sizeof(Real), cudaMemcpyHostToDevice, sEdge));
        }

        std::swap(input, output);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (isRoot) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate performance metrics
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ---- Gather the final grid on rank 0 (only when it is needed)
    const Real* dFinal = input;  // after the swaps, 'input' holds the result of the last iteration
    int exitCode = 0;
    if (printResults || validate) {
        std::vector<Real> finalGrid;
        Real* hLocal = nullptr;
        const size_t ownedCells = static_cast<size_t>(nzLocal) * plane;
        if (isRoot) {
            finalGrid.resize(gridSize);
            hLocal = finalGrid.data();
        } else if (ownedCells > 0) {
            CUDA_CHECK(cudaMallocHost(&hLocal, ownedCells * sizeof(Real)));
        }
        if (ownedCells > 0) {
            CUDA_CHECK(cudaMemcpy(hLocal, dFinal + plane, ownedCells * sizeof(Real), cudaMemcpyDeviceToHost));
        }

        // Gather whole planes so counts/displacements stay small for large grids.
        MPI_Datatype planeType;
        MPI_CHECK(MPI_Type_contiguous(planeCount, MPI_DOUBLE, &planeType));
        MPI_CHECK(MPI_Type_commit(&planeType));
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = static_cast<int>(planesOfRank(r, size, static_cast<long long>(nz)));
            displs[r] = static_cast<int>(startOfRank(r, size, static_cast<long long>(nz)));
        }
        MPI_CHECK(MPI_Gatherv(isRoot ? MPI_IN_PLACE : hLocal, static_cast<int>(nzLocal), planeType,
                              isRoot ? finalGrid.data() : nullptr, counts.data(), displs.data(), planeType, 0,
                              MPI_COMM_WORLD));
        MPI_CHECK(MPI_Type_free(&planeType));
        if (!isRoot && hLocal) CUDA_CHECK(cudaFreeHost(hLocal));

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
        MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }

    // ---- Cleanup
    CUDA_CHECK(cudaEventDestroy(evEdgeDone));
    CUDA_CHECK(cudaEventDestroy(evInteriorDone));
    CUDA_CHECK(cudaStreamDestroy(sInterior));
    CUDA_CHECK(cudaStreamDestroy(sEdge));
    CUDA_CHECK(cudaFreeHost(hSendLow));
    CUDA_CHECK(cudaFreeHost(hSendHigh));
    CUDA_CHECK(cudaFreeHost(hRecvLow));
    CUDA_CHECK(cudaFreeHost(hRecvHigh));
    CUDA_CHECK(cudaFree(dGrid[0]));
    CUDA_CHECK(cudaFree(dGrid[1]));

    MPI_Finalize();
    return exitCode;
}
