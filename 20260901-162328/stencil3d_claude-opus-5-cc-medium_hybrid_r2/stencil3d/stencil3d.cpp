// 3D 7-point stencil benchmark -- hybrid MPI + OpenMP + CUDA implementation.
//
// Parallelization strategy:
//   * MPI    : 1D slab decomposition along Z across ranks (one rank per GPU).
//              Halo planes are exchanged every iteration, overlapped with the
//              computation of the slab interior. Node-local neighbours trade
//              planes through an MPI-3 shared-memory window, so the GPUs DMA
//              straight into and out of the memory the neighbour reads and no
//              host-side message copy is needed; off-node neighbours fall back
//              to point-to-point sends out of pinned staging buffers.
//   * CUDA   : the stencil itself runs on the GPU. A register-blocked Z-sweep
//              kernel handles the bulk of the slab, a thin kernel handles the
//              two boundary planes that have to be communicated first.
//   * OpenMP : host-side work (result assembly and the validation reductions)
//              is threaded.
//
// All results are bit-identical to the original serial implementation.

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

#define CUDA_CHECK(call)                                                                                     \
    do {                                                                                                     \
        const cudaError_t err_ = (call);                                                                     \
        if (err_ != cudaSuccess) {                                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__);       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                    \
        }                                                                                                    \
    } while (0)

// Stencil kernel tiling: a 64x4 thread tile keeps the X halo overhead low while
// staying at 256 threads, and each block sweeps ZCHUNK planes along Z.
#define STENCIL_BX 64
#define STENCIL_BY 4
#define STENCIL_ZCHUNK 32

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Correctly rounded division by 7 built from FMAs (Markstein correction).
// Consumer-class GPUs run IEEE double division at a small fraction of their FMA
// rate, which makes the plain "/ 7.0" the bottleneck of this memory-bound
// stencil. The reciprocal estimate followed by an exact residual correction
// returns the same bit pattern as the hardware divide.
__device__ __forceinline__ Real div7(const Real x) {
    const Real r = 1.0 / 7.0;
    const Real q = x * r;
    const Real e = __fma_rn(-7.0, q, x);  // exact residual x - 7*q
    return __fma_rn(e, r, q);
}

// grid[x, y, lz] with lz in [0, nzLocal+1]; lz == 0 and lz == nzLocal+1 are halo
// planes. The global Z coordinate of local plane lz is z0 + lz - 1.
__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t nx, const size_t ny, const size_t nz,
                                     const size_t z0, const size_t nzLocal) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t lz = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || lz >= nzLocal + 2) return;

    const size_t local = lz * (nx * ny) + y * nx + x;
    // Halo planes outside of the global domain are never read; keep them at 0.
    Real value = 0.0;
    if (lz + z0 >= 1 && z0 + lz - 1 < nz) {
        const size_t gz = z0 + lz - 1;
        const size_t gidx = gz * (nx * ny) + y * nx + x;
        value = static_cast<Real>(gidx % 19) * 1.0;
    }
    grid[local] = value;
}

// Computes the local planes [lzBegin, lzEnd] (all of which are guaranteed to be
// globally interior) with a register-blocked sweep along Z: each thread walks a
// column keeping the three active planes in registers, so every input point is
// fetched once. blockIdx.z splits the range into chunks of ZCHUNK planes, which
// gives the GPU enough concurrent blocks to saturate memory.
__global__ void __launch_bounds__(STENCIL_BX* STENCIL_BY)
    stencilSweepKernel(const Real* __restrict__ input, Real* __restrict__ output, const size_t nx, const size_t ny,
                       const int lzBegin, const int lzEnd) {
    const size_t x = blockIdx.x * STENCIL_BX + threadIdx.x;
    const size_t y = blockIdx.y * STENCIL_BY + threadIdx.y;

    const int zBegin = lzBegin + static_cast<int>(blockIdx.z) * STENCIL_ZCHUNK;
    int zEnd = zBegin + STENCIL_ZCHUNK - 1;
    if (zEnd > lzEnd) zEnd = lzEnd;
    if (zBegin > lzEnd || x < 1 || x >= nx - 1 || y < 1 || y >= ny - 1) return;

    const size_t plane = nx * ny;
    size_t idx = static_cast<size_t>(zBegin) * plane + y * nx + x;

    Real below = input[idx - plane];
    Real center = input[idx];

    for (int lz = zBegin; lz <= zEnd; ++lz, idx += plane) {
        const Real above = input[idx + plane];
        const Real left = input[idx - 1];
        const Real right = input[idx + 1];
        const Real front = input[idx - nx];
        const Real back = input[idx + nx];
        output[idx] = div7(center + left + right + front + back + below + above);
        below = center;
        center = above;
    }
}

// Computes exactly the two communication-critical planes of the slab.
__global__ void __launch_bounds__(STENCIL_BX* STENCIL_BY)
    stencilEdgeKernel(const Real* __restrict__ input, Real* __restrict__ output, const size_t nx, const size_t ny,
                      const size_t nz, const size_t z0, const int lzLow, const int lzHigh) {
    const size_t x = blockIdx.x * STENCIL_BX + threadIdx.x;
    const size_t y = blockIdx.y * STENCIL_BY + threadIdx.y;
    if (x < 1 || x >= nx - 1 || y < 1 || y >= ny - 1) return;

    const int lz = (blockIdx.z == 0) ? lzLow : lzHigh;
    const size_t gz = z0 + static_cast<size_t>(lz) - 1;
    if (gz < 1 || gz >= nz - 1) return;

    const size_t plane = nx * ny;
    const size_t idx = static_cast<size_t>(lz) * plane + y * nx + x;
    output[idx] = div7(input[idx] + input[idx - 1] + input[idx + 1] + input[idx - nx] + input[idx + nx] +
                       input[idx - plane] + input[idx + plane]);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    const size_t n = grid.size();
    int bad = 0;
    #pragma omp parallel for reduction(|:bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) bad = 1;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // --- Bind each rank to a GPU (round robin over the node-local ranks) ---
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, nodeSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &nodeSize);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // --- Slab decomposition along Z ---
    const size_t base = nz / static_cast<size_t>(nranks);
    const size_t rem = nz % static_cast<size_t>(nranks);
    const size_t nzLocal = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t z0 = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);

    // Ranks beyond the number of planes own nothing; they still take part in the
    // collectives but have no halo neighbours.
    const int activeRanks = (int)std::min(static_cast<size_t>(nranks), nz);
    const bool amActive = (rank < activeRanks);
    const int prev = (amActive && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next = (amActive && rank + 1 < activeRanks) ? rank + 1 : MPI_PROC_NULL;

    const size_t localCells = (nzLocal + 2) * plane;

    Real* d_a = nullptr;
    Real* d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, localCells * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_b, localCells * sizeof(Real)));

    // Pinned staging buffers for the halo exchange (MPI is not CUDA-aware here).
    Real* h_sendLo = nullptr;
    Real* h_sendHi = nullptr;
    Real* h_recvLo = nullptr;
    Real* h_recvHi = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_sendLo, plane * sizeof(Real), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_sendHi, plane * sizeof(Real), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recvLo, plane * sizeof(Real), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recvHi, plane * sizeof(Real), cudaHostAllocDefault));

    // --- Node-local halo exchange through an MPI-3 shared-memory window ---
    // For neighbours that live on the same node the GPU downloads its boundary
    // plane straight into shared memory that the neighbour uploads from, which
    // removes the two host-side copies MPI would otherwise perform. Two slots
    // per direction (indexed by the iteration parity) keep a rank from
    // overwriting a plane its neighbour has not uploaded yet.
    std::vector<int> worldOfNode(nodeSize);
    MPI_Allgather(&rank, 1, MPI_INT, worldOfNode.data(), 1, MPI_INT, nodeComm);
    int prevNodeRank = -1, nextNodeRank = -1;
    for (int i = 0; i < nodeSize; ++i) {
        if (prev != MPI_PROC_NULL && worldOfNode[i] == prev) prevNodeRank = i;
        if (next != MPI_PROC_NULL && worldOfNode[i] == next) nextNodeRank = i;
    }

    const size_t slotBytes = plane * sizeof(Real);
    MPI_Info shInfo;
    MPI_Info_create(&shInfo);
    MPI_Info_set(shInfo, "alloc_shared_noncontig", "false");
    MPI_Win shWin = MPI_WIN_NULL;
    Real* shSelf = nullptr;
    MPI_Win_allocate_shared((MPI_Aint)(4 * slotBytes), (int)sizeof(Real), shInfo, nodeComm, &shSelf, &shWin);
    MPI_Info_free(&shInfo);

    auto shBaseOf = [&](int nodeRank) -> Real* {
        MPI_Aint sz = 0;
        int du = 0;
        Real* p = nullptr;
        MPI_Win_shared_query(shWin, nodeRank, &sz, &du, &p);
        return p;
    };
    Real* const shPrev = (prevNodeRank >= 0) ? shBaseOf(prevNodeRank) : nullptr;
    Real* const shNext = (nextNodeRank >= 0) ? shBaseOf(nextNodeRank) : nullptr;

    // Page-lock the whole node segment so the GPU can DMA in and out of it at
    // full speed. Failure is not fatal, the copies just fall back to pageable.
    Real* shRegBase = shBaseOf(0);
    size_t shRegBytes = 0;
    {
        MPI_Aint lastSz = 0;
        int du = 0;
        Real* lastBase = nullptr;
        MPI_Win_shared_query(shWin, nodeSize - 1, &lastSz, &du, &lastBase);
        shRegBytes = (size_t)((char*)lastBase - (char*)shRegBase) + (size_t)lastSz;
        if (cudaHostRegister(shRegBase, shRegBytes, cudaHostRegisterDefault) != cudaSuccess) {
            cudaGetLastError();
            shRegBytes = 0;
        }
    }
    MPI_Win_lock_all(MPI_MODE_NOCHECK, shWin);

    // The per-iteration node barrier is only needed if somebody on this node
    // actually exchanges through the window.
    int usesShared = (shPrev != nullptr || shNext != nullptr) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &usesShared, 1, MPI_INT, MPI_MAX, nodeComm);
    const bool nodeSyncNeeded = (usesShared != 0);

    cudaStream_t sEdge, sHalo, sBulk;
    CUDA_CHECK(cudaStreamCreate(&sEdge));
    CUDA_CHECK(cudaStreamCreate(&sHalo));
    CUDA_CHECK(cudaStreamCreate(&sBulk));
    cudaEvent_t evEdge, evSendLo, evSendHi;
    CUDA_CHECK(cudaEventCreateWithFlags(&evEdge, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evSendLo, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evSendHi, cudaEventDisableTiming));

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    {
        const dim3 blk(32, 4, 2);
        const dim3 grd((unsigned)((nx + blk.x - 1) / blk.x), (unsigned)((ny + blk.y - 1) / blk.y),
                       (unsigned)((nzLocal + 2 + blk.z - 1) / blk.z));
        initializeGridKernel<<<grd, blk>>>(d_a, nx, ny, nz, z0, nzLocal);
        CUDA_CHECK(cudaGetLastError());
        // Boundary cells are only ever copied from input to output, so they never
        // change. Seeding both buffers with the initial state makes the per
        // iteration boundary copy unnecessary.
        CUDA_CHECK(cudaMemcpy(d_b, d_a, localCells * sizeof(Real), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Local plane range owned by this rank: [1, nzLocal]. The two planes at the
    // ends of the slab feed the neighbours' halos and are computed first; the
    // remaining planes [2, nzLocal-1] are always globally interior.
    const int lzLow = 1;
    const int lzHigh = (int)nzLocal;
    const bool hasInterior = (nzLocal > 2);

    // Launch configuration for the stencil kernels.
    const dim3 sblk(STENCIL_BX, STENCIL_BY, 1);
    const unsigned gx = (unsigned)((nx + STENCIL_BX - 1) / STENCIL_BX);
    const unsigned gy = (unsigned)((ny + STENCIL_BY - 1) / STENCIL_BY);
    const unsigned bulkPlanes = hasInterior ? (unsigned)(lzHigh - 1 - (lzLow + 1) + 1) : 1u;
    const dim3 sgrd(gx, gy, (bulkPlanes + STENCIL_ZCHUNK - 1) / STENCIL_ZCHUNK);
    const dim3 egrd(gx, gy, 2);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        const Real* in = (iter % 2 == 0) ? d_a : d_b;
        Real* out = (iter % 2 == 0) ? d_b : d_a;

        // Shared-window slots for this iteration: [parity][lo|hi].
        Real* const myLoOut = shSelf + (size_t)((iter & 1) * 2 + 0) * plane;
        Real* const myHiOut = shSelf + (size_t)((iter & 1) * 2 + 1) * plane;

        // 1) Post the receives for off-node neighbours up front.
        MPI_Request reqs[4];
        int nreq = 0;
        if (prev != MPI_PROC_NULL && shPrev == nullptr) {
            MPI_Irecv(h_recvLo, (int)plane, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (next != MPI_PROC_NULL && shNext == nullptr) {
            MPI_Irecv(h_recvHi, (int)plane, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        }

        if (nzLocal > 0) {
            // 2) The two planes the neighbours need, on their own streams so the
            //    two downloads run independently and each send can start as soon
            //    as its plane has landed on the host.
            stencilEdgeKernel<<<egrd, sblk, 0, sEdge>>>(in, out, nx, ny, nz, z0, lzLow, lzHigh);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(evEdge, sEdge));
            CUDA_CHECK(cudaStreamWaitEvent(sHalo, evEdge, 0));
            if (prev != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(shPrev ? myLoOut : h_sendLo, out + (size_t)lzLow * plane,
                                           plane * sizeof(Real), cudaMemcpyDeviceToHost, sEdge));
                CUDA_CHECK(cudaEventRecord(evSendLo, sEdge));
            }
            if (next != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(shNext ? myHiOut : h_sendHi, out + (size_t)lzHigh * plane,
                                           plane * sizeof(Real), cudaMemcpyDeviceToHost, sHalo));
                CUDA_CHECK(cudaEventRecord(evSendHi, sHalo));
            }

            // 3) Bulk of the slab, overlapped with the halo exchange below.
            if (hasInterior) {
                stencilSweepKernel<<<sgrd, sblk, 0, sBulk>>>(in, out, nx, ny, lzLow + 1, lzHigh - 1);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // 4) Halo exchange, pipelined with the downloads.
        if (prev != MPI_PROC_NULL) CUDA_CHECK(cudaEventSynchronize(evSendLo));
        if (prev != MPI_PROC_NULL && shPrev == nullptr) {
            MPI_Isend(h_sendLo, (int)plane, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (next != MPI_PROC_NULL) CUDA_CHECK(cudaEventSynchronize(evSendHi));
        if (next != MPI_PROC_NULL && shNext == nullptr) {
            MPI_Isend(h_sendHi, (int)plane, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        // Publish the shared-window writes and wait until the node-local
        // neighbours have published theirs.
        if (nodeSyncNeeded) {
            MPI_Win_sync(shWin);
            MPI_Barrier(nodeComm);
            MPI_Win_sync(shWin);
        }
        if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        if (nzLocal > 0) {
            if (prev != MPI_PROC_NULL) {
                // The neighbour below exports its high plane in the same slot.
                const Real* src = shPrev ? (shPrev + (size_t)((iter & 1) * 2 + 1) * plane) : h_recvLo;
                CUDA_CHECK(cudaMemcpyAsync(out, src, plane * sizeof(Real), cudaMemcpyHostToDevice, sEdge));
            }
            if (next != MPI_PROC_NULL) {
                const Real* src = shNext ? (shNext + (size_t)((iter & 1) * 2 + 0) * plane) : h_recvHi;
                CUDA_CHECK(cudaMemcpyAsync(out + (size_t)(nzLocal + 1) * plane, src, plane * sizeof(Real),
                                           cudaMemcpyHostToDevice, sHalo));
            }
            CUDA_CHECK(cudaStreamSynchronize(sEdge));
            CUDA_CHECK(cudaStreamSynchronize(sHalo));
            CUDA_CHECK(cudaStreamSynchronize(sBulk));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // --- Gather the final grid on rank 0 (only when it is actually needed) ---
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        const Real* result = (iterations % 2 == 0) ? d_a : d_b;

        std::vector<Real> localSlab(nzLocal * plane);
        if (nzLocal > 0) {
            CUDA_CHECK(cudaMemcpy(localSlab.data(), result + plane, nzLocal * plane * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
        }

        if (rank == 0) {
            finalGrid.resize(gridSize);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < nzLocal * plane; ++i) finalGrid[i] = localSlab[i];

            // Messages can exceed INT_MAX elements, so receive in chunks.
            const size_t chunk = 64u * 1024u * 1024u;
            for (int r = 1; r < nranks; ++r) {
                const size_t rnz = base + (static_cast<size_t>(r) < rem ? 1 : 0);
                const size_t roff = (base * static_cast<size_t>(r) + std::min(static_cast<size_t>(r), rem)) * plane;
                size_t remaining = rnz * plane;
                size_t done = 0;
                while (remaining > 0) {
                    const size_t n = std::min(remaining, chunk);
                    MPI_Recv(finalGrid.data() + roff + done, (int)n, MPI_DOUBLE, r, 2, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    remaining -= n;
                    done += n;
                }
            }
        } else {
            const size_t chunk = 64u * 1024u * 1024u;
            size_t remaining = nzLocal * plane;
            size_t done = 0;
            while (remaining > 0) {
                const size_t n = std::min(remaining, chunk);
                MPI_Send(localSlab.data() + done, (int)n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                remaining -= n;
                done += n;
            }
        }
    }

    int exitCode = 0;
    if (rank == 0) {
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

    MPI_Win_unlock_all(shWin);
    if (shRegBytes > 0) CUDA_CHECK(cudaHostUnregister(shRegBase));
    MPI_Win_free(&shWin);
    MPI_Comm_free(&nodeComm);

    CUDA_CHECK(cudaEventDestroy(evEdge));
    CUDA_CHECK(cudaEventDestroy(evSendLo));
    CUDA_CHECK(cudaEventDestroy(evSendHi));
    CUDA_CHECK(cudaStreamDestroy(sEdge));
    CUDA_CHECK(cudaStreamDestroy(sHalo));
    CUDA_CHECK(cudaStreamDestroy(sBulk));
    CUDA_CHECK(cudaFreeHost(h_sendLo));
    CUDA_CHECK(cudaFreeHost(h_sendHi));
    CUDA_CHECK(cudaFreeHost(h_recvLo));
    CUDA_CHECK(cudaFreeHost(h_recvHi));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    MPI_Finalize();
    return exitCode;
}
