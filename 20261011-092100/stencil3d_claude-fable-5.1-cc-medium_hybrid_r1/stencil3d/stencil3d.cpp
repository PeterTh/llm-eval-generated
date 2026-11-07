// 3D 7-point stencil benchmark -- hybrid MPI + OpenMP + CUDA implementation.
//
// Parallelization strategy
//   * MPI    : 1D domain decomposition of the interior z-planes across ranks.
//              Each rank owns a contiguous slab of planes plus one halo plane
//              on either side.  Halo planes are exchanged every iteration.
//              Neighbours on different nodes exchange through pinned host
//              staging buffers with non-blocking MPI.  Neighbours on the same
//              node exchange directly GPU-to-GPU through CUDA IPC mappings of
//              each other's buffers (peer DMA, no host staging); MPI then only
//              carries tiny synchronisation tokens.
//   * CUDA   : each rank drives one GPU (node-local rank modulo #GPUs).  The
//              stencil is evaluated by a register-sliding-window kernel (one
//              thread per (x,y) column, marching along z).  The two boundary
//              planes of the slab are computed first on a dedicated stream so
//              that their exchange overlaps the bulk interior kernel running
//              on a second stream.
//   * OpenMP : host side work (grid initialisation, result assembly,
//              validation) is thread parallel.
//
// Semantics are identical to the original serial code: interior points are
// replaced by the 7-point average, boundary points keep their initial value.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// ----------------------------------------------------------------------------
// Error handling helpers
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

#define MPI_CHECK(call)                                                         \
    do {                                                                        \
        int err__ = (call);                                                     \
        if (err__ != MPI_SUCCESS) {                                             \
            fprintf(stderr, "MPI error %s at %s:%d (code %d)\n", #call,        \
                    __FILE__, __LINE__, err__);                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ----------------------------------------------------------------------------
// CUDA kernel: 7-point stencil on local planes [zBegin, zEnd) of a slab.
// One thread per (x,y) column; each block handles a chunk of zChunk planes and
// keeps bottom/center/top in registers so each input element is loaded from
// global memory ~once along z (x/y neighbours are served by L1/L2).
// ----------------------------------------------------------------------------
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
              const int nx, const int ny,
              const int zBegin, const int zEnd, const int zChunk) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x + 1;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y + 1;
    if (x >= nx - 1 || y >= ny - 1) return;

    const int z0 = zBegin + blockIdx.z * zChunk;
    const int z1 = min(z0 + zChunk, zEnd);
    if (z0 >= z1) return;

    const size_t plane = (size_t)nx * (size_t)ny;
    size_t idx = (size_t)z0 * plane + (size_t)y * nx + x;

    Real bottom = __ldg(&in[idx - plane]);
    Real center = __ldg(&in[idx]);

    for (int z = z0; z < z1; ++z) {
        const Real top   = __ldg(&in[idx + plane]);
        const Real left  = __ldg(&in[idx - 1]);
        const Real right = __ldg(&in[idx + 1]);
        const Real front = __ldg(&in[idx - nx]);
        const Real back  = __ldg(&in[idx + nx]);

        // Simple averaging stencil (same expression/order as the original)
        out[idx] = (center + left + right + front + back + bottom + top) / 7.0;

        bottom = center;
        center = top;
        idx += plane;
    }
}

static void launchStencil(const Real* in, Real* out, int nx, int ny,
                          int zBegin, int zEnd, cudaStream_t stream) {
    const int nz = zEnd - zBegin;
    if (nz <= 0 || nx < 3 || ny < 3) return;

    const unsigned gx = (unsigned)((nx - 2 + BLOCK_X - 1) / BLOCK_X);
    const unsigned gy = (unsigned)((ny - 2 + BLOCK_Y - 1) / BLOCK_Y);

    // Pick the z-chunk so that we have plenty of blocks to fill the GPU while
    // keeping the register sliding window effective.
    const long xyBlocks = (long)gx * gy;
    const long targetBlocks = 4096;
    int zChunk = (int)std::max<long>(1, std::min<long>(64, (xyBlocks * nz) / targetBlocks));
    zChunk = std::min(zChunk, nz);
    const unsigned gz = (unsigned)((nz + zChunk - 1) / zChunk);

    dim3 grid(gx, gy, gz);
    dim3 block(BLOCK_X, BLOCK_Y);
    stencilKernel<<<grid, block, 0, stream>>>(in, out, nx, ny, zBegin, zEnd, zChunk);
    CUDA_CHECK(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Host helpers
// ----------------------------------------------------------------------------

// Initial value of the global grid cell with linear index idx (as in the
// original initializeGrid).
inline Real initialValue(const size_t idx) noexcept {
    return (idx % 19) * 1.0;
}

// Fill host buffer `slab` with the initial values of global planes
// [gzFirst, gzFirst + nPlanes).
void initializeSlab(Real* slab, const size_t nx, const size_t ny,
                    const size_t gzFirst, const size_t nPlanes) {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 0; lz < nPlanes; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t gz = gzFirst + lz;
            Real* row = slab + idx3(0, y, lz, nx, ny);
            const size_t base = idx3(0, y, gz, nx, ny);
            for (size_t x = 0; x < nx; ++x) {
                row[x] = initialValue(base + x);
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values; 2. min/max
    const size_t n = grid.size();
    bool bad = false;
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for reduction(|| : bad) reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Real val = grid[i];
        if (std::isnan(val) || std::isinf(val)) bad = true;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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

// Interior planes [1, nz-2] are distributed in contiguous blocks over ranks.
struct Decomp {
    size_t start;  // first global interior plane owned (valid if count > 0)
    size_t count;  // number of interior planes owned
};

static Decomp decompose(const size_t nInterior, const int size, const int rank) {
    const size_t base = nInterior / (size_t)size;
    const size_t rem = nInterior % (size_t)size;
    Decomp d;
    d.count = base + ((size_t)rank < rem ? 1 : 0);
    d.start = 1 + (size_t)rank * base + std::min((size_t)rank, rem);
    return d;
}

// ----------------------------------------------------------------------------
// Halo neighbour description
// ----------------------------------------------------------------------------
enum MsgTag : int { TAG_HALO_LO = 0, TAG_HALO_HI = 1, TAG_IPC_INFO = 2, TAG_IPC_OK = 3,
                    TAG_DONE = 4, TAG_PUSHED = 5 };

struct IpcInfo {
    cudaIpcMemHandle_t mem[2];   // handles of dGrid1 / dGrid2
    cudaIpcEventHandle_t event;  // "pushed" event handle
    int device;                  // owner's CUDA device ordinal
};

struct Neighbor {
    int rank = MPI_PROC_NULL;
    bool ipc = false;
    int sendPlane = 0;        // local plane index of the boundary plane we provide
    int recvPlane = 0;        // local plane index of the halo plane we receive into
    int tagSend = 0, tagRecv = 0;
    // host-staged path
    Real* hSend = nullptr;
    Real* hRecv = nullptr;
    cudaEvent_t evCopied = nullptr;
    MPI_Request sendReq = MPI_REQUEST_NULL, recvReq = MPI_REQUEST_NULL;
    // CUDA IPC path
    Real* nbrBuf[2] = {nullptr, nullptr};   // neighbour's dGrid1 / dGrid2 mapped here
    size_t nbrHaloOffset = 0;               // element offset of "our" halo plane in nbr buffer
    cudaEvent_t myPushed = nullptr;         // we record after pushing into neighbour
    cudaEvent_t nbrPushed = nullptr;        // neighbour's event (opened via IPC)
    MPI_Request doneReq = MPI_REQUEST_NULL, pushedReq = MPI_REQUEST_NULL;

    bool valid() const { return rank != MPI_PROC_NULL; }
};

static void sendToken(int dest, int tag, MPI_Request& req) {
    if (req != MPI_REQUEST_NULL) MPI_CHECK(MPI_Wait(&req, MPI_STATUS_IGNORE));
    MPI_CHECK(MPI_Isend(nullptr, 0, MPI_BYTE, dest, tag, MPI_COMM_WORLD, &req));
}

static void recvToken(int src, int tag) {
    MPI_CHECK(MPI_Recv(nullptr, 0, MPI_BYTE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    const bool root = (rank == 0);

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // ---- Node topology & GPU selection (node-local rank modulo #devices) ----
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0, localSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &localSize));
    int nodeId = rank;  // node identified by the smallest world rank on it
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &nodeId, 1, MPI_INT, MPI_MIN, nodeComm));
    MPI_CHECK(MPI_Comm_free(&nodeComm));
    std::vector<int> nodeOfRank(size);
    MPI_CHECK(MPI_Allgather(&nodeId, 1, MPI_INT, nodeOfRank.data(), 1, MPI_INT, MPI_COMM_WORLD));

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices <= 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % numDevices;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(0));  // establish context now (outside timing)

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: %d MPI rank(s), %d rank(s)/node, %d GPU(s)/node, %d OpenMP thread(s)/rank\n",
               size, localSize, numDevices, omp_get_max_threads());
    }

    const size_t gridSize = nx * ny * nz;
    const size_t planeSize = nx * ny;
    const size_t planeBytes = planeSize * sizeof(Real);
    const bool hasInterior = (nx >= 3 && ny >= 3 && nz >= 3);
    const size_t nInterior = hasInterior ? nz - 2 : 0;

    // ---- Domain decomposition along z ----
    const Decomp me = decompose(nInterior, size, rank);
    const size_t nloc = me.count;
    const int lzFirst = 1;         // first owned plane (local index)
    const int lzLast = (int)nloc;  // last owned plane (local index)

    // Neighbours: nearest lower/higher rank that owns at least one plane.
    Neighbor nbrs[2];  // [0] = lower z neighbour, [1] = upper z neighbour
    if (nloc > 0) {
        for (int r = rank - 1; r >= 0; --r) {
            if (decompose(nInterior, size, r).count > 0) { nbrs[0].rank = r; break; }
        }
        for (int r = rank + 1; r < size; ++r) {
            if (decompose(nInterior, size, r).count > 0) { nbrs[1].rank = r; break; }
        }
    }
    nbrs[0].sendPlane = lzFirst;  nbrs[0].recvPlane = 0;
    nbrs[0].tagSend = TAG_HALO_LO; nbrs[0].tagRecv = TAG_HALO_HI;
    nbrs[1].sendPlane = lzLast;   nbrs[1].recvPlane = lzLast + 1;
    nbrs[1].tagSend = TAG_HALO_HI; nbrs[1].tagRecv = TAG_HALO_LO;

    // Local slab: planes [me.start-1, me.start+nloc] -> nloc+2 planes.
    const size_t slabPlanes = (nloc > 0) ? nloc + 2 : 0;
    const size_t slabSize = slabPlanes * planeSize;

    // ---- Allocate host slab (pinned) and device buffers (double buffering) ----
    Real* hostSlab = nullptr;
    Real* dGrid[2] = {nullptr, nullptr};
    // Halo stream gets the highest priority so boundary work is scheduled
    // ahead of the (long-running) interior kernel.
    cudaStream_t sComp, sHalo;
    int prioLeast = 0, prioGreatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLeast, &prioGreatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sComp, cudaStreamNonBlocking, prioLeast));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sHalo, cudaStreamNonBlocking, prioGreatest));

    if (root) printf("Initializing grid...\n");
    if (nloc > 0) {
        CUDA_CHECK(cudaMallocHost((void**)&hostSlab, slabSize * sizeof(Real)));
        CUDA_CHECK(cudaMalloc((void**)&dGrid[0], slabSize * sizeof(Real)));
        CUDA_CHECK(cudaMalloc((void**)&dGrid[1], slabSize * sizeof(Real)));

        // Initialize (OpenMP) and upload. Both buffers get the full initial
        // slab so that boundary cells (never written by the kernel) keep their
        // initial values in either buffer -- equivalent to the original's
        // per-iteration boundary copy.
        initializeSlab(hostSlab, nx, ny, me.start - 1, slabPlanes);
        CUDA_CHECK(cudaMemcpy(dGrid[0], hostSlab, slabSize * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dGrid[1], hostSlab, slabSize * sizeof(Real), cudaMemcpyHostToDevice));
    }

    // ---- Set up halo exchange with each neighbour ----
    // Try CUDA IPC for same-node neighbours; fall back to host staging.
    for (Neighbor& nb : nbrs) {
        if (!nb.valid()) continue;
        bool wantIpc = (nodeOfRank[nb.rank] == nodeOfRank[rank]);
        bool myOk = false;
        if (wantIpc) {
            IpcInfo mine{}, theirs{};
            mine.device = device;
            cudaError_t e = cudaSuccess;
            if (e == cudaSuccess) e = cudaIpcGetMemHandle(&mine.mem[0], dGrid[0]);
            if (e == cudaSuccess) e = cudaIpcGetMemHandle(&mine.mem[1], dGrid[1]);
            if (e == cudaSuccess) e = cudaEventCreateWithFlags(&nb.myPushed, cudaEventDisableTiming | cudaEventInterprocess);
            if (e == cudaSuccess) e = cudaIpcGetEventHandle(&mine.event, nb.myPushed);
            (void)cudaGetLastError();
            if (e != cudaSuccess)
                fprintf(stderr, "Rank %d: cannot export CUDA IPC handles (%s); using host-staged halo exchange with rank %d\n",
                        rank, cudaGetErrorString(e), nb.rank);
            int haveHandles = (e == cudaSuccess) ? 1 : 0;
            int peerHave = 0;
            MPI_CHECK(MPI_Sendrecv(&haveHandles, 1, MPI_INT, nb.rank, TAG_IPC_OK,
                                   &peerHave, 1, MPI_INT, nb.rank, TAG_IPC_OK,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            if (haveHandles && peerHave) {
                MPI_CHECK(MPI_Sendrecv(&mine, (int)sizeof(IpcInfo), MPI_BYTE, nb.rank, TAG_IPC_INFO,
                                       &theirs, (int)sizeof(IpcInfo), MPI_BYTE, nb.rank, TAG_IPC_INFO,
                                       MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                // Map the neighbour's buffers into our address space (requires
                // peer access between the two devices, or the same device).
                e = cudaIpcOpenMemHandle((void**)&nb.nbrBuf[0], theirs.mem[0], cudaIpcMemLazyEnablePeerAccess);
                if (e == cudaSuccess) e = cudaIpcOpenMemHandle((void**)&nb.nbrBuf[1], theirs.mem[1], cudaIpcMemLazyEnablePeerAccess);
                if (e == cudaSuccess) e = cudaIpcOpenEventHandle(&nb.nbrPushed, theirs.event);
                (void)cudaGetLastError();
                myOk = (e == cudaSuccess);
                if (!myOk)
                    fprintf(stderr, "Rank %d: cannot open CUDA IPC handles of rank %d (%s); using host-staged halo exchange\n",
                            rank, nb.rank, cudaGetErrorString(e));
            }
        }
        int ok = myOk ? 1 : 0, peerOk = 0;
        if (wantIpc) {
            MPI_CHECK(MPI_Sendrecv(&ok, 1, MPI_INT, nb.rank, TAG_IPC_OK,
                                   &peerOk, 1, MPI_INT, nb.rank, TAG_IPC_OK,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
        nb.ipc = wantIpc && ok && peerOk;
        if (nb.ipc) {
            // Where does our boundary plane land in the neighbour's slab?
            const size_t nbrCount = decompose(nInterior, size, nb.rank).count;
            nb.nbrHaloOffset = (&nb == &nbrs[0]) ? (nbrCount + 1) * planeSize   // we are their upper neighbour
                                                 : 0;                            // we are their lower neighbour
        } else {
            // Release any partially opened IPC resources and set up host staging.
            if (nb.nbrBuf[0]) { cudaIpcCloseMemHandle(nb.nbrBuf[0]); nb.nbrBuf[0] = nullptr; }
            if (nb.nbrBuf[1]) { cudaIpcCloseMemHandle(nb.nbrBuf[1]); nb.nbrBuf[1] = nullptr; }
            if (nb.nbrPushed) { cudaEventDestroy(nb.nbrPushed); nb.nbrPushed = nullptr; }
            if (nb.myPushed) { cudaEventDestroy(nb.myPushed); nb.myPushed = nullptr; }
            (void)cudaGetLastError();
            CUDA_CHECK(cudaMallocHost((void**)&nb.hSend, planeBytes));
            CUDA_CHECK(cudaMallocHost((void**)&nb.hRecv, planeBytes));
            CUDA_CHECK(cudaEventCreateWithFlags(&nb.evCopied, cudaEventDisableTiming));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (root) {
        int ipcCount = 0, stagedCount = 0;
        for (const Neighbor& nb : nbrs) if (nb.valid()) (nb.ipc ? ipcCount : stagedCount)++;
        int tot[2] = {ipcCount, stagedCount};
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, tot, 2, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
        printf("Halo exchange: %d GPU-direct (CUDA IPC) link(s), %d host-staged MPI link(s)\n", tot[0] / 2, tot[1] / 2);
    } else {
        int ipcCount = 0, stagedCount = 0;
        for (const Neighbor& nb : nbrs) if (nb.valid()) (nb.ipc ? ipcCount : stagedCount)++;
        int tot[2] = {ipcCount, stagedCount};
        MPI_CHECK(MPI_Reduce(tot, nullptr, 2, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
    }

    Real* dIn = dGrid[0];
    Real* dOut = dGrid[1];

    // IPC neighbours may write into our dOut of iteration 0 right away.
    for (Neighbor& nb : nbrs)
        if (nb.valid() && nb.ipc) sendToken(nb.rank, TAG_DONE, nb.doneReq);

    // Run stencil iterations
    if (root) printf("Running stencil computation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (nloc > 0) {
            const int outIdx = (iter & 1) ^ 1;  // dOut == dGrid[outIdx] on every rank

            // 0) Post receives for host-staged neighbours as early as possible.
            for (Neighbor& nb : nbrs)
                if (nb.valid() && !nb.ipc)
                    MPI_CHECK(MPI_Irecv(nb.hRecv, (int)planeSize, MPI_DOUBLE, nb.rank, nb.tagRecv,
                                        MPI_COMM_WORLD, &nb.recvReq));

            // 1) Boundary planes of the slab first (high-priority halo stream),
            //    followed by their device-to-host copies for host-staged
            //    neighbours.  All of this is enqueued before the big interior
            //    kernel so the block scheduler does not starve it.
            launchStencil(dIn, dOut, (int)nx, (int)ny, lzFirst, lzFirst + 1, sHalo);
            if (nloc > 1) launchStencil(dIn, dOut, (int)nx, (int)ny, lzLast, lzLast + 1, sHalo);
            for (Neighbor& nb : nbrs) {
                if (!nb.valid() || nb.ipc) continue;
                CUDA_CHECK(cudaMemcpyAsync(nb.hSend, dOut + (size_t)nb.sendPlane * planeSize, planeBytes,
                                           cudaMemcpyDeviceToHost, sHalo));
                CUDA_CHECK(cudaEventRecord(nb.evCopied, sHalo));
            }

            // 2) Interior planes overlap with the exchange (compute stream).
            launchStencil(dIn, dOut, (int)nx, (int)ny, lzFirst + 1, lzLast, sComp);

            // 3a) IPC neighbours: push our boundary plane straight into their
            //     dOut halo plane once they are done reading that buffer.
            for (Neighbor& nb : nbrs) {
                if (!nb.valid() || !nb.ipc) continue;
                recvToken(nb.rank, TAG_DONE);
                CUDA_CHECK(cudaMemcpyAsync(nb.nbrBuf[outIdx] + nb.nbrHaloOffset,
                                           dOut + (size_t)nb.sendPlane * planeSize, planeBytes,
                                           cudaMemcpyDefault, sHalo));
                CUDA_CHECK(cudaEventRecord(nb.myPushed, sHalo));
                sendToken(nb.rank, TAG_PUSHED, nb.pushedReq);
            }

            // 3b) Host-staged neighbours: send as soon as the plane is on the host.
            for (Neighbor& nb : nbrs) {
                if (!nb.valid() || nb.ipc) continue;
                CUDA_CHECK(cudaEventSynchronize(nb.evCopied));
                MPI_CHECK(MPI_Isend(nb.hSend, (int)planeSize, MPI_DOUBLE, nb.rank, nb.tagSend,
                                    MPI_COMM_WORLD, &nb.sendReq));
            }
            for (Neighbor& nb : nbrs) {
                if (!nb.valid() || nb.ipc) continue;
                MPI_CHECK(MPI_Wait(&nb.recvReq, MPI_STATUS_IGNORE));
                CUDA_CHECK(cudaMemcpyAsync(dOut + (size_t)nb.recvPlane * planeSize, nb.hRecv, planeBytes,
                                           cudaMemcpyHostToDevice, sHalo));
            }
            for (Neighbor& nb : nbrs) {
                if (!nb.valid() || nb.ipc) continue;
                MPI_CHECK(MPI_Wait(&nb.sendReq, MPI_STATUS_IGNORE));
            }

            // 3c) IPC neighbours: wait until their push into our dOut has landed.
            for (Neighbor& nb : nbrs) {
                if (!nb.valid() || !nb.ipc) continue;
                recvToken(nb.rank, TAG_PUSHED);
                CUDA_CHECK(cudaEventSynchronize(nb.nbrPushed));
            }

            // 4) Everything must be complete before the next iteration reads dOut.
            CUDA_CHECK(cudaStreamSynchronize(sHalo));
            CUDA_CHECK(cudaStreamSynchronize(sComp));

            // 5) dIn is no longer read: IPC neighbours may push into it next iteration.
            for (Neighbor& nb : nbrs)
                if (nb.valid() && nb.ipc) sendToken(nb.rank, TAG_DONE, nb.doneReq);
        }
        std::swap(dIn, dOut);
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double localMs =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    double maxMs = 0.0;
    MPI_CHECK(MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    const long durationMs = (long)maxMs;

    // Drain the trailing DONE tokens so no messages are left in flight.
    for (Neighbor& nb : nbrs) {
        if (!nb.valid() || !nb.ipc) continue;
        recvToken(nb.rank, TAG_DONE);
        if (nb.doneReq != MPI_REQUEST_NULL) MPI_CHECK(MPI_Wait(&nb.doneReq, MPI_STATUS_IGNORE));
        if (nb.pushedReq != MPI_REQUEST_NULL) MPI_CHECK(MPI_Wait(&nb.pushedReq, MPI_STATUS_IGNORE));
    }

    if (root) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ---- Gather final grid on rank 0 (only when needed) ----
    // After the loop dIn holds the most recent result (== grid1 for even
    // iteration counts, grid2 for odd -- same as the original).
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        if (nloc > 0) {
            CUDA_CHECK(cudaMemcpy(hostSlab + planeSize, dIn + planeSize,
                                  nloc * planeBytes, cudaMemcpyDeviceToHost));
        }

        MPI_Datatype planeType;
        MPI_CHECK(MPI_Type_contiguous((int)planeSize, MPI_DOUBLE, &planeType));
        MPI_CHECK(MPI_Type_commit(&planeType));

        std::vector<int> counts, displs;
        if (root) {
            finalGrid.resize(gridSize);
            counts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                const Decomp d = decompose(nInterior, size, r);
                counts[r] = (int)d.count;
                displs[r] = (d.count > 0) ? (int)d.start : 0;
            }
            // Global boundary planes never change: regenerate them directly.
            if (nz >= 1) initializeSlab(finalGrid.data(), nx, ny, 0, 1);
            if (nz >= 2) initializeSlab(finalGrid.data() + (nz - 1) * planeSize, nx, ny, nz - 1, 1);
            if (!hasInterior && nz > 2) {
                // nx or ny < 3: nothing is ever updated.
                initializeSlab(finalGrid.data() + planeSize, nx, ny, 1, nz - 2);
            }
        }
        MPI_CHECK(MPI_Gatherv(nloc > 0 ? hostSlab + planeSize : nullptr, (int)nloc, planeType,
                              root ? finalGrid.data() : nullptr,
                              root ? counts.data() : nullptr,
                              root ? displs.data() : nullptr,
                              planeType, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Type_free(&planeType));
    }

    int exitCode = 0;
    if (root) {
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

    // ---- Cleanup (neighbours must not free buffers we still have mapped) ----
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    for (Neighbor& nb : nbrs) {
        if (!nb.valid()) continue;
        if (nb.ipc) {
            CUDA_CHECK(cudaIpcCloseMemHandle(nb.nbrBuf[0]));
            CUDA_CHECK(cudaIpcCloseMemHandle(nb.nbrBuf[1]));
            CUDA_CHECK(cudaEventDestroy(nb.nbrPushed));
            CUDA_CHECK(cudaEventDestroy(nb.myPushed));
        } else {
            CUDA_CHECK(cudaFreeHost(nb.hSend));
            CUDA_CHECK(cudaFreeHost(nb.hRecv));
            CUDA_CHECK(cudaEventDestroy(nb.evCopied));
        }
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDA_CHECK(cudaStreamDestroy(sComp));
    CUDA_CHECK(cudaStreamDestroy(sHalo));
    if (nloc > 0) {
        CUDA_CHECK(cudaFree(dGrid[0]));
        CUDA_CHECK(cudaFree(dGrid[1]));
        CUDA_CHECK(cudaFreeHost(hostSlab));
    }

    MPI_Finalize();
    return exitCode;
}
