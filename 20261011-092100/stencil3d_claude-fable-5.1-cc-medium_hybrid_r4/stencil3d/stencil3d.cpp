// 3D 7-point stencil benchmark: hybrid MPI + OpenMP + CUDA implementation.
//
// Parallelization strategy
//   * MPI   : 1D slab decomposition of the global grid along the z axis.  Each
//             rank owns a contiguous range of z planes plus one halo plane on
//             each side.  Halo planes are exchanged every iteration with
//             non-blocking point-to-point messages, staged through pinned host
//             buffers so no CUDA-aware MPI is required.
//   * CUDA  : Each rank drives one GPU (selected by node-local rank).  The
//             stencil is evaluated by a register-tiled kernel that marches
//             along z with a 2D thread tile in x/y.  The two planes that must
//             be sent to neighbours are computed first on a dedicated stream so
//             their transfer and the MPI exchange overlap the interior update.
//   * OpenMP: All host-side bulk work (grid initialisation, local validation
//             reductions, assembling the gathered grid) is multi-threaded.
//
// Semantics are identical to the original serial code: interior cells are
// replaced by the 7-point average (same summation order), boundary cells keep
// their initial values, and the final grid / statistics match bit for bit.

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

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

#define MPI_CHECK(call)                                                                    \
    do {                                                                                   \
        int err__ = (call);                                                                \
        if (err__ != MPI_SUCCESS) {                                                        \
            fprintf(stderr, "MPI error %s at %s:%d (code %d)\n", #call, __FILE__, __LINE__, \
                    err__);                                                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Host: initialise the locally owned planes (global z in [zs, ze)) into a local
// buffer that has one halo plane below the owned planes (local plane k holds
// global plane zs - 1 + k).  Uses the same formula as the original code
// evaluated on the *global* linear index so the values are identical.
// ---------------------------------------------------------------------------
static void initializeGrid(std::vector<Real>& local, const size_t nx, const size_t ny,
                           const size_t zs, const size_t nzl) {
    const size_t plane = nx * ny;
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t k = 0; k < nzl; ++k) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t gz = zs + k;
            Real* row = local.data() + (k + 1) * plane + y * nx;
            const size_t base = gz * plane + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = base + x;
                row[x] = (idx % 19) * 1.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel: 7-point averaging stencil on local planes [kBegin, kEnd).
//
// Thread block = TX x TY tile in x/y; each thread marches along z over a chunk
// of KZ planes keeping the below/centre/above values in registers, so every
// input element is loaded from global memory (L2/L1 served) roughly once.
// Boundary cells (global x/y/z on the outer faces) are never written: the
// output buffer already holds their (immutable) values.
// ---------------------------------------------------------------------------
constexpr int TX = 32;
constexpr int TY = 8;
constexpr int KZ = 32;

__global__ void __launch_bounds__(TX * TY)
stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
              const int nx, const int ny, const int nz,
              const int zs,           // global z of local plane 1
              const int kBegin,       // first local plane to update (>= 1)
              const int kEnd)         // one past last local plane to update
{
    const int x = blockIdx.x * TX + threadIdx.x;
    const int y = blockIdx.y * TY + threadIdx.y;
    if (x < 1 || x >= nx - 1 || y < 1 || y >= ny - 1) return;

    int k0 = kBegin + blockIdx.z * KZ;
    int k1 = min(k0 + KZ, kEnd);
    // Clip against the global z boundaries: planes with gz == 0 / nz-1 are kept.
    if (zs - 1 + k0 == 0) ++k0;
    if (zs - 1 + (k1 - 1) == nz - 1) --k1;
    if (k0 >= k1) return;

    const size_t plane = (size_t)nx * ny;
    const Real* __restrict__ p = in + (size_t)k0 * plane + (size_t)y * nx + x;
    Real* __restrict__ q = out + (size_t)k0 * plane + (size_t)y * nx + x;

    Real below  = __ldg(p - plane);
    Real center = __ldg(p);

    for (int k = k0; k < k1; ++k) {
        const Real top    = __ldg(p + plane);
        const Real left   = __ldg(p - 1);
        const Real right  = __ldg(p + 1);
        const Real front  = __ldg(p - nx);
        const Real back   = __ldg(p + nx);
        // Same association order as the serial reference.
        *q = (center + left + right + front + back + below + top) / 7.0;
        below  = center;
        center = top;
        p += plane;
        q += plane;
    }
}

static void launchStencil(const Real* in, Real* out, int nx, int ny, int nz, int zs,
                          int kBegin, int kEnd, cudaStream_t stream) {
    if (kEnd <= kBegin) return;
    const dim3 block(TX, TY, 1);
    const dim3 grid((nx + TX - 1) / TX, (ny + TY - 1) / TY, (kEnd - kBegin + KZ - 1) / KZ);
    stencilKernel<<<grid, block, 0, stream>>>(in, out, nx, ny, nz, zs, kBegin, kEnd);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Validation: distributed version of the original checks.  Each rank scans its
// owned planes with OpenMP; results are combined with MPI reductions.
// ---------------------------------------------------------------------------
static bool validateResult(const std::vector<Real>& local, const size_t nx, const size_t ny,
                           const size_t nzl, MPI_Comm comm, int rank) {
    const size_t plane = nx * ny;
    const Real* data = local.data() + plane;  // skip lower halo plane
    const size_t n = plane * nzl;

    int bad = 0;
    Real minVal = (n > 0) ? data[0] : HUGE_VAL;
    Real maxVal = (n > 0) ? data[0] : -HUGE_VAL;
#pragma omp parallel for reduction(|| : bad) reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Real v = data[i];
        if (std::isnan(v) || std::isinf(v)) bad = 1;
        minVal = std::min(minVal, v);
        maxVal = std::max(maxVal, v);
    }

    int gbad = 0;
    Real gmin = 0, gmax = 0;
    MPI_CHECK(MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_MAX, comm));
    MPI_CHECK(MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, comm));
    MPI_CHECK(MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm));

    if (gbad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (rank == 0) printf("Value range: [%.6f, %.6f]\n", gmin, gmax);
    if (gmax > 1e6 || gmin < -1e6) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

static void printUsage(const char* progName) {
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
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, nranks = 1;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));
    MPI_CHECK(MPI_Comm_size(comm, &nranks));

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

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // ---- GPU selection: one device per rank, by node-local rank -------------
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&nodeComm));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(comm, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(0));  // establish context now (outside the timed region)

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n", nranks,
               omp_get_max_threads(), deviceCount);
    }

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;

    // ---- 1D slab decomposition along z ---------------------------------------
    std::vector<int> zStart(nranks + 1);
    for (int r = 0; r <= nranks; ++r) zStart[r] = (int)((nz * (size_t)r) / (size_t)nranks);
    const int zs = zStart[rank];
    const int nzl = zStart[rank + 1] - zs;   // number of owned planes (may be 0)

    // Neighbours are the nearest ranks that own at least one plane.
    int down = MPI_PROC_NULL, up = MPI_PROC_NULL;
    for (int r = rank - 1; r >= 0; --r)
        if (zStart[r + 1] - zStart[r] > 0) { down = r; break; }
    for (int r = rank + 1; r < nranks; ++r)
        if (zStart[r + 1] - zStart[r] > 0) { up = r; break; }
    if (nzl == 0) { down = MPI_PROC_NULL; up = MPI_PROC_NULL; }

    // Local storage: owned planes + one halo plane on each side.
    const size_t localPlanes = (size_t)nzl + 2;
    const size_t localSize = localPlanes * plane;

    // ---- Host allocation + initialisation (OpenMP) ----------------------------
    if (rank == 0) printf("Initializing grid...\n");
    std::vector<Real> hostGrid(localSize, 0.0);
    initializeGrid(hostGrid, nx, ny, (size_t)zs, (size_t)nzl);

    // ---- Device allocation --------------------------------------------------
    Real *dA = nullptr, *dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, localSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&dB, localSize * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(dA, hostGrid.data(), localSize * sizeof(Real), cudaMemcpyHostToDevice));

    // Pinned staging buffers for halo exchange (send lower/upper, recv lower/upper).
    Real *sendLo = nullptr, *sendHi = nullptr, *recvLo = nullptr, *recvHi = nullptr;
    CUDA_CHECK(cudaMallocHost(&sendLo, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&sendHi, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recvLo, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recvHi, plane * sizeof(Real)));

    cudaStream_t sBoundary, sInterior;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sBoundary, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sInterior, cudaStreamNonBlocking));
    cudaEvent_t evLo, evHi;
    CUDA_CHECK(cudaEventCreateWithFlags(&evLo, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evHi, cudaEventDisableTiming));

    // Initial halo exchange so both neighbours' boundary planes are present
    // before the first iteration (and the boundary copies of the output buffer
    // hold the correct, immutable outer-face values).
    auto exchangeHalos = [&](Real* buf, cudaStream_t s) {
        if (nzl == 0) return;
        MPI_Request req[4];
        int nreq = 0;
        if (down != MPI_PROC_NULL)
            MPI_CHECK(MPI_Irecv(recvLo, (int)plane, MPI_DOUBLE, down, 1, comm, &req[nreq++]));
        if (up != MPI_PROC_NULL)
            MPI_CHECK(MPI_Irecv(recvHi, (int)plane, MPI_DOUBLE, up, 0, comm, &req[nreq++]));
        CUDA_CHECK(cudaMemcpyAsync(sendLo, buf + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(sendHi, buf + (size_t)nzl * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        if (down != MPI_PROC_NULL)
            MPI_CHECK(MPI_Isend(sendLo, (int)plane, MPI_DOUBLE, down, 0, comm, &req[nreq++]));
        if (up != MPI_PROC_NULL)
            MPI_CHECK(MPI_Isend(sendHi, (int)plane, MPI_DOUBLE, up, 1, comm, &req[nreq++]));
        MPI_CHECK(MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE));
        if (down != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(buf, recvLo, plane * sizeof(Real), cudaMemcpyHostToDevice, s));
        if (up != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(buf + (size_t)(nzl + 1) * plane, recvHi, plane * sizeof(Real), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
    };
    exchangeHalos(dA, sBoundary);
    // Boundary cells are never written by the kernel, so seed the second buffer
    // with the full initial state (outer faces + halos) once.
    CUDA_CHECK(cudaMemcpy(dB, dA, localSize * sizeof(Real), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Plane ranges: boundary planes (k = 1 and k = nzl) must be sent to the
    // neighbours; the interior planes (2 .. nzl-1) are overlapped with that.
    const bool hasLo = (nzl >= 1);
    const bool hasHi = (nzl >= 2);              // when nzl == 1 the plane is shared
    const int intBegin = 2, intEnd = nzl;       // [2, nzl) == planes 2..nzl-1

    // ---- Run stencil iterations ----------------------------------------------
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_CHECK(MPI_Barrier(comm));
    auto start = std::chrono::high_resolution_clock::now();

    Real* in = dA;
    Real* out = dB;
    for (int iter = 0; iter < iterations; ++iter) {
        if (nzl > 0) {
            MPI_Request req[4];
            int nreq = 0;
            // Post receives for next halos early.
            if (down != MPI_PROC_NULL)
                MPI_CHECK(MPI_Irecv(recvLo, (int)plane, MPI_DOUBLE, down, 1, comm, &req[nreq++]));
            if (up != MPI_PROC_NULL)
                MPI_CHECK(MPI_Irecv(recvHi, (int)plane, MPI_DOUBLE, up, 0, comm, &req[nreq++]));

            // 1. Boundary planes first (own stream) + D2H of each as soon as done.
            if (hasLo) {
                launchStencil(in, out, (int)nx, (int)ny, (int)nz, zs, 1, 2, sBoundary);
                if (down != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(sendLo, out + plane, plane * sizeof(Real),
                                               cudaMemcpyDeviceToHost, sBoundary));
                CUDA_CHECK(cudaEventRecord(evLo, sBoundary));
            }
            if (hasHi) {
                launchStencil(in, out, (int)nx, (int)ny, (int)nz, zs, nzl, nzl + 1, sBoundary);
            }
            if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(sendHi, out + (size_t)nzl * plane, plane * sizeof(Real),
                                           cudaMemcpyDeviceToHost, sBoundary));
            CUDA_CHECK(cudaEventRecord(evHi, sBoundary));

            // 2. Interior planes concurrently on the second stream.
            launchStencil(in, out, (int)nx, (int)ny, (int)nz, zs, intBegin, intEnd, sInterior);

            // 3. Send boundary planes as soon as they land in host memory.
            if (down != MPI_PROC_NULL) {
                CUDA_CHECK(cudaEventSynchronize(evLo));
                MPI_CHECK(MPI_Isend(sendLo, (int)plane, MPI_DOUBLE, down, 0, comm, &req[nreq++]));
            }
            if (up != MPI_PROC_NULL) {
                CUDA_CHECK(cudaEventSynchronize(evHi));
                MPI_CHECK(MPI_Isend(sendHi, (int)plane, MPI_DOUBLE, up, 1, comm, &req[nreq++]));
            }

            // 4. Receive halos and upload into the output buffer (next input).
            MPI_CHECK(MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE));
            if (down != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(out, recvLo, plane * sizeof(Real),
                                           cudaMemcpyHostToDevice, sBoundary));
            if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(out + (size_t)(nzl + 1) * plane, recvHi,
                                           plane * sizeof(Real), cudaMemcpyHostToDevice, sBoundary));

            // 5. Join streams before the next iteration reads `out`.
            CUDA_CHECK(cudaStreamSynchronize(sBoundary));
            CUDA_CHECK(cudaStreamSynchronize(sInterior));
        }
        std::swap(in, out);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(comm));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localElapsedMs = duration.count();
    long maxElapsedMs = 0;
    MPI_CHECK(MPI_Reduce(&localElapsedMs, &maxElapsedMs, 1, MPI_LONG, MPI_MAX, 0, comm));

    // After the loop `in` points at the buffer holding the final state
    // (same parity rule as the original: even iterations -> grid1).
    Real* dFinal = in;
    CUDA_CHECK(cudaMemcpy(hostGrid.data(), dFinal, localSize * sizeof(Real), cudaMemcpyDeviceToHost));

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxElapsedMs);
        // Calculate performance metrics
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (maxElapsedMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ---- Print results for external validation (gather full grid to rank 0) --
    if (printResults) {
        std::vector<Real> finalGrid;
        std::vector<int> counts(nranks), displs(nranks);
        // MPI counts are ints; gather plane-by-plane chunks if a slab exceeds INT_MAX.
        bool fits = true;
        for (int r = 0; r < nranks; ++r) {
            const size_t cnt = (size_t)(zStart[r + 1] - zStart[r]) * plane;
            const size_t off = (size_t)zStart[r] * plane;
            if (cnt > (size_t)0x7fffffff || off > (size_t)0x7fffffff) fits = false;
            counts[r] = (int)cnt;
            displs[r] = (int)off;
        }
        if (rank == 0) finalGrid.resize(gridSize);
        if (fits) {
            MPI_CHECK(MPI_Gatherv(hostGrid.data() + plane, (int)((size_t)nzl * plane), MPI_DOUBLE,
                                  finalGrid.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm));
        } else {
            // Fallback: one message per plane.
            if (rank == 0) {
#pragma omp parallel for schedule(static)
                for (int k = 0; k < nzl; ++k)
                    std::memcpy(finalGrid.data() + (size_t)(zs + k) * plane,
                                hostGrid.data() + (size_t)(k + 1) * plane, plane * sizeof(Real));
                for (int r = 1; r < nranks; ++r)
                    for (int k = zStart[r]; k < zStart[r + 1]; ++k)
                        MPI_CHECK(MPI_Recv(finalGrid.data() + (size_t)k * plane, (int)plane, MPI_DOUBLE,
                                           r, 100 + k, comm, MPI_STATUS_IGNORE));
            } else {
                for (int k = 0; k < nzl; ++k)
                    MPI_CHECK(MPI_Send(hostGrid.data() + (size_t)(k + 1) * plane, (int)plane, MPI_DOUBLE,
                                       0, 100 + zs + k, comm));
            }
        }
        if (rank == 0) print_results(finalGrid, "Grid");
    }

    // ---- Validation ------------------------------------------------------------
    int exitCode = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(hostGrid, nx, ny, (size_t)nzl, comm, rank);
        if (rank == 0) {
            if (valid) printf("Validation: PASSED\n");
            else       printf("Validation: FAILED\n");
        }
        exitCode = valid ? 0 : 1;
    }

    // ---- Cleanup ---------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(evLo));
    CUDA_CHECK(cudaEventDestroy(evHi));
    CUDA_CHECK(cudaStreamDestroy(sBoundary));
    CUDA_CHECK(cudaStreamDestroy(sInterior));
    CUDA_CHECK(cudaFreeHost(sendLo));
    CUDA_CHECK(cudaFreeHost(sendHi));
    CUDA_CHECK(cudaFreeHost(recvLo));
    CUDA_CHECK(cudaFreeHost(recvHi));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));

    MPI_Finalize();
    return exitCode;
}
