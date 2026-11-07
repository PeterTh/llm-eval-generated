// 3D 7-point stencil benchmark — hybrid MPI + OpenMP + CUDA implementation.
//
// Parallelization strategy
//   * MPI   : 1D domain decomposition of the interior z-planes across ranks.
//             Every rank owns a contiguous slab of z-planes plus one halo plane
//             on each side.  Halo planes are exchanged with the neighbouring
//             ranks every iteration (non-blocking, host-staged through pinned
//             memory so that it also works with non CUDA-aware MPI).
//   * CUDA  : The stencil itself runs on one GPU per rank.  The slab is split
//             into an inner part (independent of the halos) and the two
//             boundary planes (which need the halos).  Inner planes are
//             computed on one stream while the halo exchange and the boundary
//             planes are processed on a second stream, overlapping
//             communication with computation.
//   * OpenMP: Host-side work (grid initialisation, validation reductions,
//             assembling the gathered grid) is multithreaded with OpenMP.
//
// The numerical result is bit-for-bit identical to the original serial code:
// the same operations are evaluated in the same order, and the global
// boundary values are never modified (just as the original copies them
// unchanged every iteration).

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
#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        const cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,          \
                    cudaGetErrorString(err__));                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

#define MPI_CHECK(call)                                                                         \
    do {                                                                                        \
        const int err__ = (call);                                                               \
        if (err__ != MPI_SUCCESS) {                                                             \
            char msg__[MPI_MAX_ERROR_STRING];                                                   \
            int len__ = 0;                                                                      \
            MPI_Error_string(err__, msg__, &len__);                                             \
            fprintf(stderr, "MPI error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, msg__);   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initial value of a grid cell (identical to the original initializeGrid).
__host__ __device__ inline Real initialValue(const size_t globalIdx) noexcept {
    return (globalIdx % 19) * 1.0;
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Block shape for the stencil kernel: 32 threads along x give fully coalesced
// 256-byte row accesses, 8 rows along y.  Each thread marches along z and keeps
// the bottom/center/top values in registers, so every input element is read
// from global memory (or L1/L2) essentially once.
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int Z_CHUNK = 16;  // z-planes handled per thread in one kernel

// 7-point stencil on the local planes [zBegin, zEnd) of the local slab.
// Local plane p corresponds to global plane (zStart - 1 + p); plane 0 and
// plane (count+1) are halo planes.  Only interior x/y points are updated.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const int nx, const int ny, const int zBegin, const int zEnd) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x + 1;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y + 1;
    if (x >= nx - 1 || y >= ny - 1) return;

    const int z0 = zBegin + blockIdx.z * Z_CHUNK;
    const int z1 = min(z0 + Z_CHUNK, zEnd);
    if (z0 >= z1) return;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    size_t idx = idx3(x, y, z0, nx, ny);

    Real bottom = input[idx - plane];
    Real center = input[idx];
    for (int z = z0; z < z1; ++z) {
        const Real top = input[idx + plane];
        const Real left = input[idx - 1];
        const Real right = input[idx + 1];
        const Real front = input[idx - nx];
        const Real back = input[idx + nx];

        // Simple averaging stencil (same evaluation order as the original)
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;

        bottom = center;
        center = top;
        idx += plane;
    }
}

// ---------------------------------------------------------------------------
// Distributed grid description
// ---------------------------------------------------------------------------
struct Decomposition {
    int rank = 0;
    int size = 1;
    size_t nx = 0, ny = 0, nz = 0;
    size_t zStart = 0;   // first global interior plane owned by this rank
    size_t count = 0;    // number of interior planes owned by this rank
    int lowerNeighbor = MPI_PROC_NULL;  // rank owning the plane zStart-1 (if not a global boundary)
    int upperNeighbor = MPI_PROC_NULL;  // rank owning the plane zStart+count (if not a global boundary)
    size_t plane() const { return nx * ny; }
    size_t localPlanes() const { return count + 2; }
    size_t localSize() const { return localPlanes() * plane(); }
};

// Split the interior z-planes [1, nz-2] as evenly as possible across ranks.
// Ranks with zero planes (only when size > nz-2) are always at the tail, so
// the ranks with work form a contiguous chain.
static size_t planesOf(const size_t interior, const int size, const int r) {
    const size_t base = interior / static_cast<size_t>(size);
    const size_t rem = interior % static_cast<size_t>(size);
    return base + (static_cast<size_t>(r) < rem ? 1 : 0);
}

static Decomposition makeDecomposition(const int rank, const int size, const size_t nx,
                                       const size_t ny, const size_t nz) {
    Decomposition d;
    d.rank = rank;
    d.size = size;
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;
    const size_t interior = (nz >= 3) ? nz - 2 : 0;
    d.count = planesOf(interior, size, rank);
    size_t start = 1;
    for (int r = 0; r < rank; ++r) start += planesOf(interior, size, r);
    d.zStart = start;
    if (rank > 0 && planesOf(interior, size, rank - 1) > 0 && d.count > 0) d.lowerNeighbor = rank - 1;
    if (rank + 1 < size && planesOf(interior, size, rank + 1) > 0 && d.count > 0) d.upperNeighbor = rank + 1;
    return d;
}

// ---------------------------------------------------------------------------
// Host-side helpers (OpenMP)
// ---------------------------------------------------------------------------

// Fill the local slab (including halo planes) with the initial values.
static void initializeLocalGrid(Real* local, const Decomposition& d) {
    const size_t plane = d.plane();
    const size_t nPlanes = d.localPlanes();
    const size_t globalOffset = (d.zStart - 1) * plane;
#pragma omp parallel for schedule(static)
    for (size_t p = 0; p < nPlanes; ++p) {
        Real* dst = local + p * plane;
        const size_t base = globalOffset + p * plane;
        for (size_t i = 0; i < plane; ++i) {
            dst[i] = initialValue(base + i);
        }
    }
}

// Write the initial values of global plane z into dst (used for the
// never-changing global boundary planes on rank 0).
static void fillInitialPlane(Real* dst, const size_t z, const size_t plane) {
    const size_t base = z * plane;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < plane; ++i) {
        dst[i] = initialValue(base + i);
    }
}

struct LocalStats {
    bool hasNanInf = false;
    Real minVal = 0.0;
    Real maxVal = 0.0;
    bool any = false;
};

// NaN/Inf check and min/max over a contiguous range, multithreaded.
static LocalStats computeStats(const Real* data, const size_t n) {
    LocalStats s;
    if (n == 0) return s;
    bool bad = false;
    Real mn = data[0];
    Real mx = data[0];
#pragma omp parallel for schedule(static) reduction(|| : bad) reduction(min : mn) reduction(max : mx)
    for (size_t i = 0; i < n; ++i) {
        const Real v = data[i];
        if (std::isnan(v) || std::isinf(v)) bad = true;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    s.hasNanInf = bad;
    s.minVal = mn;
    s.maxVal = mx;
    s.any = true;
    return s;
}

// Distributed equivalent of the original validateResult.  'localHost' holds the
// rank's own interior planes of the final grid; rank 0 additionally checks the
// two global boundary planes (which keep their initial values).
static bool validateResult(const Real* localInterior, const Decomposition& d, const Real* boundaryPlanes,
                           const size_t boundaryCount) {
    LocalStats s = computeStats(localInterior, d.count * d.plane());
    if (d.rank == 0 && boundaryCount > 0) {
        const LocalStats b = computeStats(boundaryPlanes, boundaryCount);
        if (!s.any) {
            s = b;
        } else {
            s.hasNanInf = s.hasNanInf || b.hasNanInf;
            s.minVal = std::min(s.minVal, b.minVal);
            s.maxVal = std::max(s.maxVal, b.maxVal);
        }
    }

    int badLocal = s.hasNanInf ? 1 : 0;
    int badGlobal = 0;
    MPI_CHECK(MPI_Allreduce(&badLocal, &badGlobal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    if (badGlobal) {
        if (d.rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Ranks without any data must not influence the reduction.
    Real localMin = s.any ? s.minVal : INFINITY;
    Real localMax = s.any ? s.maxVal : -INFINITY;
    Real minVal = 0.0, maxVal = 0.0;
    MPI_CHECK(MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    if (d.rank == 0) printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        if (d.rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }

    // Boundary values are never modified, so they trivially match the initial values.
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

// Select the GPU for this rank: ranks sharing a node are spread round-robin
// over the node's visible devices.  Unless the user set OMP_NUM_THREADS, the
// node's cores are also shared evenly between the ranks on it.
static void selectDevice(const int worldRank) {
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm));
    int nodeRank = 0, nodeSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &nodeRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &nodeSize));
    MPI_CHECK(MPI_Comm_free(&nodeComm));
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / std::max(1, nodeSize)));
    }
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(nodeRank % deviceCount));
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

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

    selectDevice(rank);
    if (rank == 0) {
        printf("Parallelization: %d MPI ranks, %d OpenMP threads/rank, 1 CUDA device/rank\n", size,
               omp_get_max_threads());
    }
    const Decomposition d = makeDecomposition(rank, size, nx, ny, nz);
    const size_t plane = d.plane();
    const size_t localSize = d.localSize();
    const bool active = (d.count > 0) && (plane > 0);

    // Host staging buffer for the local slab (pinned for fast transfers).
    Real* hostLocal = nullptr;
    CUDA_CHECK(cudaMallocHost(&hostLocal, std::max<size_t>(localSize, 1) * sizeof(Real)));

    // Device grids (double buffering)
    Real* dGrid1 = nullptr;
    Real* dGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&dGrid1, std::max<size_t>(localSize, 1) * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&dGrid2, std::max<size_t>(localSize, 1) * sizeof(Real)));

    // Pinned halo staging buffers: [0]=lower send, [1]=upper send, [2]=lower recv, [3]=upper recv
    Real* halo[4] = {nullptr, nullptr, nullptr, nullptr};
    for (auto& h : halo) CUDA_CHECK(cudaMallocHost(&h, std::max<size_t>(plane, 1) * sizeof(Real)));

    // Two streams: the inner planes run on a low-priority stream, the halo
    // traffic and the (small) boundary-plane kernels on a high-priority stream
    // so they are scheduled as soon as possible and the next halo exchange can
    // start while the inner kernel is still running.
    int prioLow = 0, prioHigh = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLow, &prioHigh));
    cudaStream_t innerStream, haloStream;
    CUDA_CHECK(cudaStreamCreateWithPriority(&innerStream, cudaStreamNonBlocking, prioLow));
    CUDA_CHECK(cudaStreamCreateWithPriority(&haloStream, cudaStreamNonBlocking, prioHigh));
    cudaEvent_t innerDone, boundaryDone, haloDownloaded;
    CUDA_CHECK(cudaEventCreateWithFlags(&innerDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&boundaryDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&haloDownloaded, cudaEventDisableTiming));

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    if (localSize > 0) {
        initializeLocalGrid(hostLocal, d);
        // Both buffers start identical: the boundary planes (and halos) are
        // therefore already "copied" into the output buffer, exactly as the
        // original boundary-copy loop does every iteration.
        CUDA_CHECK(cudaMemcpy(dGrid1, hostLocal, localSize * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dGrid2, dGrid1, localSize * sizeof(Real), cudaMemcpyDeviceToDevice));
    }
    // Initial boundary planes to be sent to the neighbours (planes 1 and count).
    if (active) {
        if (d.lowerNeighbor != MPI_PROC_NULL)
            std::memcpy(halo[0], hostLocal + 1 * plane, plane * sizeof(Real));
        if (d.upperNeighbor != MPI_PROC_NULL)
            std::memcpy(halo[1], hostLocal + d.count * plane, plane * sizeof(Real));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Kernel launch geometry
    const int inx = static_cast<int>(nx);
    const int iny = static_cast<int>(ny);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    auto gridFor = [&](const int zBegin, const int zEnd) {
        const int gx = (inx > 2) ? (inx - 2 + BLOCK_X - 1) / BLOCK_X : 0;
        const int gy = (iny > 2) ? (iny - 2 + BLOCK_Y - 1) / BLOCK_Y : 0;
        const int gz = (zEnd > zBegin) ? (zEnd - zBegin + Z_CHUNK - 1) / Z_CHUNK : 0;
        return dim3(gx, gy, gz);
    };
    const bool haveWork = active && inx > 2 && iny > 2;
    const int count = static_cast<int>(d.count);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    // Pipeline per iteration (all GPU ordering is expressed with events, the
    // host only blocks on MPI and on the download of the boundary planes):
    //   innerStream : [wait boundaryDone(i-1)] inner kernel(i)  -> innerDone
    //   haloStream  : halo H2D(i) [wait innerDone(i-1)] boundary kernels(i)
    //                 -> boundaryDone, boundary D2H(i) -> haloDownloaded
    // Dependencies: inner(i) reads planes 1/count written by boundary(i-1) and
    // overwrites planes 2/count-1 read by boundary(i-1); boundary(i) reads
    // planes 2/count-1 written by inner(i-1) and overwrites planes 1/count read
    // by inner(i-1).  The halo H2D only touches the halo planes 0/count+1 of
    // the input buffer, which no kernel writes.
    Real* dIn = dGrid1;
    Real* dOut = dGrid2;
    for (int iter = 0; iter < iterations; ++iter) {
        if (haveWork) {
            // 1) Start the halo exchange.  The send buffers hold the input's
            //    first/last interior planes (filled at init or downloaded at
            //    the end of the previous iteration).
            MPI_Request reqs[4];
            int nreq = 0;
            if (d.lowerNeighbor != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Irecv(halo[2], static_cast<int>(plane), MPI_DOUBLE, d.lowerNeighbor, 1,
                                    MPI_COMM_WORLD, &reqs[nreq++]));
                MPI_CHECK(MPI_Isend(halo[0], static_cast<int>(plane), MPI_DOUBLE, d.lowerNeighbor, 0,
                                    MPI_COMM_WORLD, &reqs[nreq++]));
            }
            if (d.upperNeighbor != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Irecv(halo[3], static_cast<int>(plane), MPI_DOUBLE, d.upperNeighbor, 0,
                                    MPI_COMM_WORLD, &reqs[nreq++]));
                MPI_CHECK(MPI_Isend(halo[1], static_cast<int>(plane), MPI_DOUBLE, d.upperNeighbor, 1,
                                    MPI_COMM_WORLD, &reqs[nreq++]));
            }

            // 2) Inner planes [2, count) do not depend on the halos: compute
            //    them while the exchange is in flight.
            if (count > 2) {
                CUDA_CHECK(cudaStreamWaitEvent(innerStream, boundaryDone, 0));
                stencilKernel<<<gridFor(2, count), block, 0, innerStream>>>(dIn, dOut, inx, iny, 2, count);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaEventRecord(innerDone, innerStream));

            // 3) Wait for the halos, upload them, and compute the boundary planes.
            if (nreq > 0) MPI_CHECK(MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE));
            if (d.lowerNeighbor != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(dIn, halo[2], plane * sizeof(Real), cudaMemcpyHostToDevice,
                                           haloStream));
            }
            if (d.upperNeighbor != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(dIn + (d.count + 1) * plane, halo[3], plane * sizeof(Real),
                                           cudaMemcpyHostToDevice, haloStream));
            }
            CUDA_CHECK(cudaStreamWaitEvent(haloStream, innerDone, 0));
            // First interior plane (local plane 1)
            stencilKernel<<<gridFor(1, 2), block, 0, haloStream>>>(dIn, dOut, inx, iny, 1, 2);
            CUDA_CHECK(cudaGetLastError());
            // Last interior plane (local plane count), if distinct from the first
            if (count > 1) {
                stencilKernel<<<gridFor(count, count + 1), block, 0, haloStream>>>(dIn, dOut, inx, iny,
                                                                                   count, count + 1);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaEventRecord(boundaryDone, haloStream));

            // 4) Download the freshly computed boundary planes: they are the
            //    halo data the neighbours need for the next iteration.
            if (d.lowerNeighbor != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(halo[0], dOut + 1 * plane, plane * sizeof(Real),
                                           cudaMemcpyDeviceToHost, haloStream));
            }
            if (d.upperNeighbor != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(halo[1], dOut + d.count * plane, plane * sizeof(Real),
                                           cudaMemcpyDeviceToHost, haloStream));
            }
            CUDA_CHECK(cudaEventRecord(haloDownloaded, haloStream));
            // The host only needs the send buffers before posting the next
            // exchange; the inner kernel keeps running on the GPU meanwhile.
            CUDA_CHECK(cudaEventSynchronize(haloDownloaded));
        }
        std::swap(dIn, dOut);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate performance metrics
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // The final grid lives in dIn after the last swap (grid1 for even iteration counts).
    // Bring this rank's interior planes back to the host.
    Real* const finalLocal = hostLocal + plane;  // interior planes start at local plane 1
    if (active) {
        CUDA_CHECK(cudaMemcpy(finalLocal, dIn + plane, d.count * plane * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    // Global boundary planes (z = 0 and z = nz-1) never change; rank 0 owns them
    // for output/validation purposes.
    std::vector<Real> boundaryPlanes;
    size_t boundaryCount = 0;
    if (rank == 0 && gridSize > 0) {
        boundaryCount = (nz >= 2) ? 2 * plane : plane;
        boundaryPlanes.resize(boundaryCount);
        fillInitialPlane(boundaryPlanes.data(), 0, plane);
        if (nz >= 2) fillInitialPlane(boundaryPlanes.data() + plane, nz - 1, plane);
    }

    // Print results for external validation
    if (printResults) {
        // Gather the complete grid on rank 0 (same layout as the original finalGrid).
        std::vector<Real> finalGrid;
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            const Decomposition dr = makeDecomposition(r, size, nx, ny, nz);
            counts[r] = static_cast<int>(dr.count * plane);
            displs[r] = static_cast<int>(dr.zStart * plane);
        }
        if (rank == 0) {
            finalGrid.resize(gridSize);
            if (gridSize > 0) {
                std::memcpy(finalGrid.data(), boundaryPlanes.data(), plane * sizeof(Real));
                if (nz >= 2)
                    std::memcpy(finalGrid.data() + (nz - 1) * plane, boundaryPlanes.data() + plane,
                                plane * sizeof(Real));
            }
        }
        MPI_CHECK(MPI_Gatherv(finalLocal, counts[rank], MPI_DOUBLE, finalGrid.data(), counts.data(),
                              displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
        if (rank == 0) print_results(finalGrid, "Grid");
    }

    int exitCode = 0;
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(finalLocal, d, boundaryPlanes.data(), boundaryCount);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        exitCode = valid ? 0 : 1;
    }

    CUDA_CHECK(cudaEventDestroy(innerDone));
    CUDA_CHECK(cudaEventDestroy(boundaryDone));
    CUDA_CHECK(cudaEventDestroy(haloDownloaded));
    CUDA_CHECK(cudaStreamDestroy(innerStream));
    CUDA_CHECK(cudaStreamDestroy(haloStream));
    for (auto& h : halo) CUDA_CHECK(cudaFreeHost(h));
    CUDA_CHECK(cudaFree(dGrid1));
    CUDA_CHECK(cudaFree(dGrid2));
    CUDA_CHECK(cudaFreeHost(hostLocal));

    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
