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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        const cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

constexpr int BX = 32;
constexpr int BY = 8;

// Local slab layout: nLocal owned z-planes plus one halo plane below (local z = 0)
// and one above (local z = nLocal + 1). Local plane lz corresponds to global plane
// z0 + lz - 1.

// Initialize all local planes (including halos) that map to valid global planes.
__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t nx, const size_t ny,
                                     const size_t nz, const long long z0, const int nLocalPlanes) {
    const size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * (size_t)blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    for (int lz = blockIdx.z; lz < nLocalPlanes; lz += gridDim.z) {
        const long long gz = z0 + lz - 1;
        if (gz < 0 || gz >= (long long)nz) continue;
        const size_t gidx = idx3(x, y, (size_t)gz, nx, ny);
        grid[idx3(x, y, (size_t)lz, nx, ny)] = (gidx % 19) * 1.0;
    }
}

// Correctly rounded s / 7.0 (bit-identical to IEEE division): reciprocal multiply followed
// by one FMA-based correction step (Markstein). Much cheaper than a full FP64 division on
// GPUs with low FP64 throughput. Falls back to true division near underflow/overflow,
// where the correction step is not guaranteed to be exact.
__device__ __noinline__ Real divideBy7Slow(const Real s) { return s / 7.0; }

__device__ __forceinline__ Real divideBy7(const Real s) {
    constexpr Real r = 1.0 / 7.0;
    // Biased exponent check with integer ops (cheap compared to FP64 compares):
    // fast path for 2^-959 <= |s| < 2^941; zero, subnormals, Inf and NaN take the slow path.
    const unsigned biasedExp = ((unsigned)__double2hiint(s) >> 20) & 0x7ffu;
    if (biasedExp - 64u >= 1900u) return divideBy7Slow(s);
    Real q = s * r;
    const Real e = fma(-q, 7.0, s);
    q = fma(e, r, q);
    return q;
}

// 7-point stencil over local planes [lzBegin, lzEnd). Each thread owns an (x, y) column
// and marches through zPerBlock planes, keeping the z-neighbors in registers.
__global__ void __launch_bounds__(BX * BY)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const size_t nx, const size_t ny, const size_t nz, const long long z0,
              const int lzBegin, const int lzEnd, const int zPerBlock) {
    const size_t x = blockIdx.x * (size_t)BX + threadIdx.x;
    const size_t y = blockIdx.y * (size_t)BY + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const int zs = lzBegin + blockIdx.z * zPerBlock;
    const int ze = min(zs + zPerBlock, lzEnd);
    if (zs >= ze) return;

    const size_t plane = nx * ny;
    const size_t col = y * nx + x;
    const bool xyBoundary = (x == 0 || x == nx - 1 || y == 0 || y == ny - 1);

    if (xyBoundary) {
        // Copy boundary values
        for (int lz = zs; lz < ze; ++lz) {
            const size_t idx = (size_t)lz * plane + col;
            output[idx] = input[idx];
        }
        return;
    }

    size_t idx = (size_t)zs * plane + col;
    Real bottom = input[idx - plane];
    Real center = input[idx];
    for (int lz = zs; lz < ze; ++lz, idx += plane) {
        const Real top = input[idx + plane];
        const long long gz = z0 + lz - 1;
        if (gz == 0 || gz == (long long)nz - 1) {
            // Copy boundary values
            output[idx] = center;
        } else {
            const Real left = input[idx - 1];
            const Real right = input[idx + 1];
            const Real front = input[idx - nx];
            const Real back = input[idx + nx];
            // Simple averaging stencil (same summation order as the reference)
            output[idx] = divideBy7(center + left + right + front + back + bottom + top);
        }
        bottom = center;
        center = top;
    }
}

struct Launcher {
    size_t nx, ny, nz;
    long long z0;
    unsigned gx = 1, gy = 1;
    int targetBlocks = 1;

    void launch(const Real* in, Real* out, int lzBegin, int lzEnd, cudaStream_t s) const {
        const int n = lzEnd - lzBegin;
        if (n <= 0) return;
        const long long xyBlocks = (long long)gx * gy;
        int chunks = (int)std::max<long long>(1, (targetBlocks + xyBlocks - 1) / xyBlocks);
        chunks = std::min({chunks, n, 65535});
        const int zPer = (n + chunks - 1) / chunks;
        chunks = (n + zPer - 1) / zPer;
        stencilKernel<<<dim3(gx, gy, chunks), dim3(BX, BY, 1), 0, s>>>(in, out, nx, ny, nz, z0,
                                                                       lzBegin, lzEnd, zPer);
    }
};

// Distributed sanity checks on the locally owned part of the grid (OpenMP on the host,
// combined across ranks with MPI).
bool validateResult(const Real* local, const size_t localCount, MPI_Comm comm, const bool isRoot) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    int bad = 0;
    #pragma omp parallel for reduction(|:bad) schedule(static)
    for (size_t i = 0; i < localCount; ++i) {
        if (std::isnan(local[i]) || std::isinf(local[i])) bad = 1;
    }
    int anyBad = 0;
    MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_LOR, comm);
    if (anyBad) {
        if (isRoot) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = INFINITY;
    Real maxVal = -INFINITY;
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t i = 0; i < localCount; ++i) {
        minVal = std::min(minVal, local[i]);
        maxVal = std::max(maxVal, local[i]);
    }
    Real gMin = 0.0, gMax = 0.0;
    MPI_Allreduce(&minVal, &gMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&maxVal, &gMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (isRoot) printf("Value range: [%.6f, %.6f]\n", gMin, gMax);

    // After averaging, values should be somewhat bounded
    if (gMax > 1e6 || gMin < -1e6) {
        if (isRoot) printf("Validation failed: values out of expected range\n");
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int worldRank = 0, worldSize = 1;
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

    // ---- Domain decomposition: contiguous z-slabs across ranks ----
    const size_t plane = nx * ny;
    const size_t base = nz / worldSize;
    const size_t rem = nz % worldSize;
    const int nLocal = (int)(base + ((size_t)worldRank < rem ? 1 : 0));
    const long long z0 = (long long)((size_t)worldRank * base + std::min<size_t>(worldRank, rem));

    // Ranks without planes (more ranks than z-planes) sit out of the computation
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, nLocal > 0 ? 0 : MPI_UNDEFINED, worldRank, &comm);
    const bool active = (nLocal > 0);

    // ---- Device selection: one GPU per node-local rank ----
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_free(&nodeComm);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(nodeRank % numDevices));

    int rank = 0, size = 1, lower = MPI_PROC_NULL, upper = MPI_PROC_NULL;
    if (active) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        if (rank > 0) lower = rank - 1;
        if (rank < size - 1) upper = rank + 1;
    }

    const int localPlanes = active ? nLocal + 2 : 0;  // with halos
    const size_t localSize = (size_t)localPlanes * plane;
    const size_t planeBytes = plane * sizeof(Real);

    // Allocate grids (double buffering)
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    Real* h_sendLo = nullptr;
    Real* h_sendHi = nullptr;
    Real* h_recvLo[2] = {nullptr, nullptr};  // double-buffered: the H2D copy from one
    Real* h_recvHi[2] = {nullptr, nullptr};  // may still be in flight next iteration
    cudaStream_t sBoundary = nullptr, sInterior = nullptr;
    cudaEvent_t evBoundary = nullptr, evInterior = nullptr, evSendLo = nullptr, evSendHi = nullptr;

    int numSMs = 1;
    {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
    }

    Launcher launcher{nx, ny, nz, z0};
    launcher.gx = (unsigned)((nx + BX - 1) / BX);
    launcher.gy = (unsigned)((ny + BY - 1) / BY);
    launcher.targetBlocks = numSMs * 16;

    if (active) {
        CUDA_CHECK(cudaMalloc(&d_grid1, localSize * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_grid2, localSize * sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_grid2, 0, localSize * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_sendLo, planeBytes));
        CUDA_CHECK(cudaMallocHost(&h_sendHi, planeBytes));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMallocHost(&h_recvLo[b], planeBytes));
            CUDA_CHECK(cudaMallocHost(&h_recvHi[b], planeBytes));
        }
        int prioLeast = 0, prioGreatest = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLeast, &prioGreatest));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sBoundary, cudaStreamNonBlocking, prioGreatest));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sInterior, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&evBoundary, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evInterior, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evSendLo, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evSendHi, cudaEventDisableTiming));
    }

    // Force (lazy) module loading of the stencil kernel outside the timed region
    {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, stencilKernel));
    }

    // Initialize (including halo planes, which are a function of the global index)
    if (isRoot) printf("Initializing grid...\n");
    if (active) {
        const dim3 blk(BX, BY, 1);
        const dim3 grd(launcher.gx, launcher.gy, (unsigned)std::min(localPlanes, 65535));
        initializeGridKernel<<<grd, blk, 0, sInterior>>>(d_grid1, nx, ny, nz, z0, localPlanes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Run stencil iterations
    if (isRoot) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (active) {
        const bool hasLower = (lower != MPI_PROC_NULL);
        const bool hasUpper = (upper != MPI_PROC_NULL);
        const bool exchange = hasLower || hasUpper;
        MPI_Request reqs[4];

        for (int iter = 0; iter < iterations; ++iter) {
            const Real* in = (iter % 2 == 0) ? d_grid1 : d_grid2;
            Real* out = (iter % 2 == 0) ? d_grid2 : d_grid1;
            Real* recvLo = h_recvLo[iter % 2];
            Real* recvHi = h_recvHi[iter % 2];

            if (exchange) {
                // Owned planes adjacent to neighbors first (on a high-priority stream), so
                // their halos can be exchanged while the interior is being computed.
                int nreq = 0;
                if (hasLower) MPI_Irecv(recvLo, (int)plane, MPI_DOUBLE, lower, 1, comm, &reqs[nreq++]);
                if (hasUpper) MPI_Irecv(recvHi, (int)plane, MPI_DOUBLE, upper, 0, comm, &reqs[nreq++]);

                launcher.launch(in, out, 1, 2, sBoundary);
                if (hasLower) {
                    CUDA_CHECK(cudaMemcpyAsync(h_sendLo, out + plane, planeBytes,
                                               cudaMemcpyDeviceToHost, sBoundary));
                    CUDA_CHECK(cudaEventRecord(evSendLo, sBoundary));
                }
                if (nLocal > 1) launcher.launch(in, out, nLocal, nLocal + 1, sBoundary);
                if (hasUpper) {
                    CUDA_CHECK(cudaMemcpyAsync(h_sendHi, out + (size_t)nLocal * plane, planeBytes,
                                               cudaMemcpyDeviceToHost, sBoundary));
                    CUDA_CHECK(cudaEventRecord(evSendHi, sBoundary));
                }
                launcher.launch(in, out, 2, nLocal, sInterior);

                if (hasLower) {
                    CUDA_CHECK(cudaEventSynchronize(evSendLo));
                    MPI_Isend(h_sendLo, (int)plane, MPI_DOUBLE, lower, 0, comm, &reqs[nreq++]);
                }
                if (hasUpper) {
                    CUDA_CHECK(cudaEventSynchronize(evSendHi));
                    MPI_Isend(h_sendHi, (int)plane, MPI_DOUBLE, upper, 1, comm, &reqs[nreq++]);
                }

                // Upload each halo as soon as it arrives
                const int nrecv = (hasLower ? 1 : 0) + (hasUpper ? 1 : 0);
                for (int done = 0; done < nrecv; ++done) {
                    int which = MPI_UNDEFINED;
                    MPI_Waitany(nrecv, reqs, &which, MPI_STATUS_IGNORE);
                    const bool fromLower = hasLower && which == 0;
                    if (fromLower)
                        CUDA_CHECK(cudaMemcpyAsync(out, recvLo, planeBytes,
                                                   cudaMemcpyHostToDevice, sBoundary));
                    else
                        CUDA_CHECK(cudaMemcpyAsync(out + (size_t)(nLocal + 1) * plane, recvHi,
                                                   planeBytes, cudaMemcpyHostToDevice, sBoundary));
                }
                MPI_Waitall(nreq - nrecv, reqs + nrecv, MPI_STATUSES_IGNORE);

                // Both streams must be done with this iteration before the next starts
                CUDA_CHECK(cudaEventRecord(evInterior, sInterior));
                CUDA_CHECK(cudaStreamWaitEvent(sBoundary, evInterior, 0));
                CUDA_CHECK(cudaEventRecord(evBoundary, sBoundary));
                CUDA_CHECK(cudaStreamWaitEvent(sInterior, evBoundary, 0));
            } else {
                launcher.launch(in, out, 1, nLocal + 1, sInterior);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long durationMs = duration.count();
    MPI_Bcast(&durationMs, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (isRoot) {
        printf("Computation time: %lld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Bring the locally owned planes of the final grid back to the host
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    const size_t ownedCount = active ? (size_t)nLocal * plane : 0;
    std::vector<Real> localResult;
    if ((printResults || validate) && active) {
        localResult.resize(ownedCount);
        CUDA_CHECK(cudaMemcpy(localResult.data(), d_final + plane, ownedCount * sizeof(Real),
                              cudaMemcpyDeviceToHost));
    }

    // Print results for external validation
    if (printResults && active) {
        MPI_Datatype planeType;
        MPI_Type_contiguous((int)plane, MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);
        std::vector<int> counts, displs;
        std::vector<Real> finalGrid;
        if (rank == 0) {
            counts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                counts[r] = (int)(base + ((size_t)r < rem ? 1 : 0));
                displs[r] = (int)((size_t)r * base + std::min<size_t>(r, rem));
            }
            finalGrid.resize(nx * ny * nz);
        }
        MPI_Gatherv(localResult.data(), nLocal, planeType, finalGrid.data(), counts.data(),
                    displs.data(), planeType, 0, comm);
        MPI_Type_free(&planeType);
        if (rank == 0) print_results(finalGrid, "Grid");
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (isRoot) printf("Validating result...\n");
        int valid = 1;
        if (active) valid = validateResult(localResult.data(), ownedCount, comm, isRoot) ? 1 : 0;
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (valid) {
            if (isRoot) printf("Validation: PASSED\n");
            exitCode = 0;
        } else {
            if (isRoot) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    if (active) {
        CUDA_CHECK(cudaEventDestroy(evBoundary));
        CUDA_CHECK(cudaEventDestroy(evInterior));
        CUDA_CHECK(cudaEventDestroy(evSendLo));
        CUDA_CHECK(cudaEventDestroy(evSendHi));
        CUDA_CHECK(cudaStreamDestroy(sBoundary));
        CUDA_CHECK(cudaStreamDestroy(sInterior));
        CUDA_CHECK(cudaFreeHost(h_sendLo));
        CUDA_CHECK(cudaFreeHost(h_sendHi));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaFreeHost(h_recvLo[b]));
            CUDA_CHECK(cudaFreeHost(h_recvHi[b]));
        }
        CUDA_CHECK(cudaFree(d_grid1));
        CUDA_CHECK(cudaFree(d_grid2));
        MPI_Comm_free(&comm);
    }

    MPI_Finalize();
    return exitCode;
}
