// Hybrid MPI + OpenMP + CUDA 3D 7-point stencil benchmark.
//
// - MPI: the grid is split into contiguous slabs along Z, one per rank (ranks beyond
//   nz stay idle). Halo planes travel through pinned host memory (MPI is not assumed
//   to be CUDA-aware); between ranks of the same node they are copied straight into
//   the neighbour's pinned MPI-3 shared-memory slots and only signalled via MPI.
// - CUDA: each rank drives one GPU (topology-aware choice). Per sweep the two slab-edge
//   planes are computed first on a high-priority stream and shipped while the interior
//   planes are computed concurrently on a second stream.
// - OpenMP: the host cores (pinned next to the rank's GPU) compute the top part of the
//   slab concurrently with the GPU when a short calibration shows that this pays off,
//   and parallelize initialization and validation.
//
// Results are bit-identical to the sequential reference (same summation order, IEEE
// division semantics).

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int Z_CHUNK = 16;

// Correctly rounded s / 7.0 (bit-identical to IEEE division): reciprocal multiply
// followed by one FMA residual correction (Markstein). FP64 division is a long
// instruction sequence and dominates the kernel cost on GPUs with low FP64 rate.
__device__ __forceinline__ Real div7(const Real s) {
    constexpr Real r = 1.0 / 7.0;
    const Real q = s * r;
    const Real e = fma(-q, 7.0, s);
    return fma(e, r, q);
}

// Initialize local planes [0, nzLocal+2) (including ghosts) with the global formula.
__global__ void initKernel(Real* __restrict__ grid, const size_t nx, const size_t ny, const size_t nz,
                           const long long zGlobalOfLocal0, const size_t nzAlloc) {
    const size_t planeSize = nx * ny;
    const size_t total = planeSize * nzAlloc;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const long long gz = zGlobalOfLocal0 + (long long)(i / planeSize);
        if (gz < 0 || gz >= (long long)nz) continue;
        const size_t gidx = (size_t)gz * planeSize + (i % planeSize);
        grid[i] = (gidx % 19) * 1.0;
    }
}

// Compute local planes [zBegin, zEnd). Local plane lz corresponds to global plane
// lz + zOffset. Global boundary points are copied, interior points averaged.
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
stencilKernel(const Real* __restrict__ in, Real* __restrict__ out, const int nx, const int ny,
              const int nz, const int zBegin, const int zEnd, const int zOffset) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const int z0 = zBegin + blockIdx.z * Z_CHUNK;
    const int z1 = min(z0 + Z_CHUNK, zEnd);
    if (z0 >= z1) return;

    const size_t plane = (size_t)nx * ny;
    size_t idx = (size_t)z0 * plane + (size_t)y * nx + x;
    const bool xyBoundary = (x == 0 || x == nx - 1 || y == 0 || y == ny - 1);

    if (xyBoundary) {
        for (int z = z0; z < z1; ++z, idx += plane) out[idx] = in[idx];
        return;
    }

    Real bottom = in[idx - plane];
    Real center = in[idx];
    for (int z = z0; z < z1; ++z, idx += plane) {
        const Real top = in[idx + plane];
        const int gz = z + zOffset;
        if (gz == 0 || gz == nz - 1) {
            out[idx] = center;
        } else {
            const Real left = in[idx - 1];
            const Real right = in[idx + 1];
            const Real front = in[idx - nx];
            const Real back = in[idx + nx];
            // Same summation order as the reference implementation
            out[idx] = div7(center + left + right + front + back + bottom + top);
        }
        bottom = center;
        center = top;
    }
}

static void launchStencil(const Real* in, Real* out, int nx, int ny, int nz, int zBegin, int zEnd,
                          int zOffset, cudaStream_t stream) {
    if (zEnd <= zBegin) return;
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((nx + BLOCK_X - 1) / BLOCK_X, (ny + BLOCK_Y - 1) / BLOCK_Y,
              (zEnd - zBegin + Z_CHUNK - 1) / Z_CHUNK);
    stencilKernel<<<grid, block, 0, stream>>>(in, out, nx, ny, nz, zBegin, zEnd, zOffset);
    CUDA_CHECK(cudaGetLastError());
}

// Host (OpenMP) version of the sweep over local planes [zBegin, zEnd); same semantics
// and summation order as the device kernel.
static void cpuStencil(const Real* __restrict__ in, Real* __restrict__ out, const size_t nx, const size_t ny,
                       const size_t nz, const int zBegin, const int zEnd, const int zOffset) {
    if (zEnd <= zBegin) return;
    const size_t plane = nx * ny;
    const long long nRows = (long long)(zEnd - zBegin) * (long long)ny;
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < nRows; ++r) {
        const int z = zBegin + (int)(r / (long long)ny);
        const size_t y = (size_t)(r % (long long)ny);
        const long long gz = (long long)z + zOffset;
        const size_t base = (size_t)z * plane + y * nx;
        const Real* c = in + base;
        Real* o = out + base;
        if (gz == 0 || gz == (long long)nz - 1 || y == 0 || y == ny - 1 || nx < 3) {
            std::memcpy(o, c, nx * sizeof(Real));
            continue;
        }
        const Real* f = c - nx;
        const Real* b = c + nx;
        const Real* d = c - plane;
        const Real* t = c + plane;
        o[0] = c[0];
#pragma omp simd
        for (size_t x = 1; x < nx - 1; ++x) {
            o[x] = (c[x] + c[x - 1] + c[x + 1] + f[x] + b[x] + d[x] + t[x]) / 7.0;
        }
        o[nx - 1] = c[nx - 1];
    }
}

// Parse a Linux cpulist ("0-63,128-191") into a cpu set
static bool readCpuList(const char* path, cpu_set_t* set) {
    CPU_ZERO(set);
    FILE* f = fopen(path, "r");
    if (!f) return false;
    char buf[4096];
    const bool ok = fgets(buf, sizeof(buf), f) != nullptr;
    fclose(f);
    if (!ok) return false;
    for (char* p = buf; *p && *p != '\n';) {
        char* end = nullptr;
        const long a = strtol(p, &end, 10);
        if (end == p) break;
        long b = a;
        p = end;
        if (*p == '-') {
            b = strtol(p + 1, &end, 10);
            p = end;
        }
        for (long c = a; c <= b && c < CPU_SETSIZE; ++c) CPU_SET((int)c, set);
        if (*p == ',') ++p;
    }
    return CPU_COUNT(set) > 0;
}

// Pin this rank's OpenMP threads to physical cores (one hardware thread per core).
// Ranks of a node that share the same affinity mask (e.g. unbound, or bound per socket
// by the launcher) split that mask's cores among themselves: each gets an even share of
// the cores local to its GPU's NUMA domain (so the host thread and pinned staging
// buffers sit next to the GPU) plus a share of the cores local to none of their GPUs.
// User-provided OMP_PROC_BIND/OMP_PLACES take precedence.
static void bindHostThreads(MPI_Comm nodeComm, const int device) {
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr) return;
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);

    // [0]: affinity mask, [1]: CPUs local to the rank's GPU
    cpu_set_t info[2];
    CPU_ZERO(&info[0]);
    CPU_ZERO(&info[1]);
    const int haveMask = sched_getaffinity(0, sizeof(cpu_set_t), &info[0]) == 0 ? 1 : 0;
    char busId[32] = {0};
    if (cudaDeviceGetPCIBusId(busId, sizeof(busId), device) == cudaSuccess) {
        for (char* c = busId; *c; ++c) *c = (char)tolower(*c);
        char path[128];
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/local_cpulist", busId);
        if (!readCpuList(path, &info[1])) CPU_ZERO(&info[1]);
    }
    std::vector<cpu_set_t> all(2 * (size_t)localSize);
    MPI_Allgather(info, 2 * sizeof(cpu_set_t), MPI_BYTE, all.data(), 2 * sizeof(cpu_set_t), MPI_BYTE, nodeComm);
    int allHaveMask = 0;
    MPI_Allreduce(&haveMask, &allHaveMask, 1, MPI_INT, MPI_MIN, nodeComm);
    if (!allHaveMask) return;
    const cpu_set_t& mask = info[0];
    const cpu_set_t& gpuLocal = info[1];

    auto physicalCores = [&](auto inSet) {
        std::vector<int> cores;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &mask) || !inSet(c)) continue;
            int first = c;
            char path[128];
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
            if (FILE* f = fopen(path, "r")) {
                int a = c;
                if (fscanf(f, "%d", &a) == 1 && a >= 0 && a < CPU_SETSIZE && CPU_ISSET(a, &mask)) first = a;
                fclose(f);
            }
            if (first == c) cores.push_back(c);
        }
        return cores;
    };
    auto share = [](const std::vector<int>& v, int idx, int n) {
        return std::vector<int>(v.begin() + v.size() * idx / n, v.begin() + v.size() * (idx + 1) / n);
    };

    // Ranks sharing my mask; among them, those sharing my GPU-local CPU set
    std::vector<int> group;
    int gIdx = 0, gpuIdx = 0, gpuN = 0;
    for (int r = 0; r < localSize; ++r) {
        if (!CPU_EQUAL(&all[2 * r], &mask)) continue;
        if (r < localRank) ++gIdx;
        group.push_back(r);
        if (CPU_COUNT(&gpuLocal) > 0 && CPU_EQUAL(&all[2 * r + 1], &gpuLocal)) {
            if (r < localRank) ++gpuIdx;
            ++gpuN;
        }
    }
    const int gN = (int)group.size();

    std::vector<int> cores;
    if (gpuN > 0) cores = share(physicalCores([&](int c) { return CPU_ISSET(c, &gpuLocal) != 0; }), gpuIdx, gpuN);
    const std::vector<int> spare = share(physicalCores([&](int c) {
                                             for (int r : group)
                                                 if (CPU_ISSET(c, &all[2 * r + 1])) return false;
                                             return true;
                                         }),
                                         gIdx, gN);
    cores.insert(cores.end(), spare.begin(), spare.end());
    if (cores.empty()) cores = share(physicalCores([](int) { return true; }), gIdx, gN);
    if (cores.empty()) {
        // Fewer cores than ranks in the mask: at least keep the host threads apart
        const std::vector<int> pool = physicalCores([](int) { return true; });
        if (pool.empty()) return;
        cores.push_back(pool[(size_t)gIdx % pool.size()]);
    }
    if (getenv("OMP_NUM_THREADS") == nullptr) omp_set_num_threads((int)cores.size());

    // The calling (master) thread lands on the first, GPU-local core
#pragma omp parallel
    {
        const int t = omp_get_thread_num();
        const int n = omp_get_num_threads();
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cores[(size_t)t * cores.size() / n], &one);
        pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    const size_t n = grid.size();
    const Real* g = grid.data();

    // 1. No NaN or Inf values
    int bad = 0;
#pragma omp parallel for reduction(| : bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(g[i]) || std::isinf(g[i])) bad = 1;
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
        minVal = std::min(minVal, g[i]);
        maxVal = std::max(maxVal, g[i]);
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
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool root = (worldRank == 0);

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

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t gridSize = nx * ny * nz;
    const size_t planeSize = nx * ny;

    // ---- Domain decomposition along Z (only ranks that get at least one plane) ----
    const int nActive = (int)std::min<size_t>((size_t)worldSize, std::max<size_t>(nz, 1));
    const bool active = worldRank < nActive;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &comm);

    std::vector<int> planeCounts(nActive), planeDispls(nActive);
    for (int r = 0, off = 0; r < nActive; ++r) {
        planeCounts[r] = (int)(nz / nActive + ((size_t)r < nz % nActive ? 1 : 0));
        planeDispls[r] = off;
        off += planeCounts[r];
    }

    // ---- GPU selection (round-robin over node-local devices) and host thread placement ----
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        int nDev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&nDev));
        if (nDev < 1) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Every rank runs the same greedy assignment: least-used device that is local to the
        // rank's affinity mask (falls back to least-used overall; plain round-robin when unbound)
        int localSize = 1;
        MPI_Comm_size(nodeComm, &localSize);
        cpu_set_t myMask;
        CPU_ZERO(&myMask);
        if (sched_getaffinity(0, sizeof(myMask), &myMask) != 0)
            for (int c = 0; c < CPU_SETSIZE; ++c) CPU_SET(c, &myMask);
        std::vector<cpu_set_t> masks(localSize);
        MPI_Allgather(&myMask, sizeof(cpu_set_t), MPI_BYTE, masks.data(), sizeof(cpu_set_t), MPI_BYTE, nodeComm);
        std::vector<cpu_set_t> devLocal(nDev);
        for (int d = 0; d < nDev; ++d) {
            CPU_ZERO(&devLocal[d]);
            char busId[32] = {0};
            if (cudaDeviceGetPCIBusId(busId, sizeof(busId), d) == cudaSuccess) {
                for (char* c = busId; *c; ++c) *c = (char)tolower(*c);
                char path[128];
                snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/local_cpulist", busId);
                readCpuList(path, &devLocal[d]);
            }
        }
        std::vector<int> used(nDev, 0);
        int device = 0;
        for (int r = 0; r <= localRank; ++r) {
            int best = -1;
            for (int pass = 0; pass < 2 && best < 0; ++pass) {
                for (int d = 0; d < nDev; ++d) {
                    cpu_set_t both;
                    CPU_AND(&both, &masks[r], &devLocal[d]);
                    if (pass == 0 && CPU_COUNT(&both) == 0) continue;
                    if (best < 0 || used[d] < used[best]) best = d;
                }
            }
            ++used[best];
            if (r == localRank) device = best;
        }
        CUDA_CHECK(cudaSetDevice(device));
        bindHostThreads(nodeComm, device);
        MPI_Comm_free(&nodeComm);
    }

    MPI_Datatype planeType;
    MPI_Type_contiguous((int)planeSize, MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    int nzLocal = 0, zStart = 0, up = MPI_PROC_NULL, down = MPI_PROC_NULL;
    int cpuPlanes = 0;  // top cpuPlanes owned planes are computed by the host (OpenMP)
    // Device and host copies of the local slab (local planes 0..nzLocal+1, ghosts included).
    // Planes [1, nzLocal-cpuPlanes] live on the GPU, the rest on the host.
    Real* dA = nullptr;
    Real* dB = nullptr;
    Real* hA = nullptr;
    Real* hB = nullptr;
    Real* hSend = nullptr;  // pinned [lower edge | upper edge]
    Real* hRecv = nullptr;  // pinned, 2 parities x [lower ghost | upper ghost]
    cudaStream_t sEdge = nullptr, sInner = nullptr;
    cudaEvent_t evEdge = nullptr, evInner = nullptr;
    bool finalInA = true;

    Real* hRegA = nullptr;  // page-aligned pinned windows of the host buffers
    Real* hRegB = nullptr;

    // (Re)create the host part for a given CPU share: host buffers are only touched for
    // local planes [g, nzLocal+1] (first touch by the threads that compute them), and
    // only the two GPU/host interface planes are pinned for async transfers.
    auto prepareHost = [&](const int cpu) {
        if (hA) {
            CUDA_CHECK(cudaHostUnregister(hRegA));
            CUDA_CHECK(cudaHostUnregister(hRegB));
            std::free(hA);
            std::free(hB);
            hA = hB = nullptr;
        }
        if (cpu <= 0) return;
        const size_t P = planeSize;
        const size_t pageSz = (size_t)sysconf(_SC_PAGESIZE);
        const size_t bytes = (P * (size_t)(nzLocal + 2) * sizeof(Real) + pageSz - 1) / pageSz * pageSz;
        hA = static_cast<Real*>(std::aligned_alloc(pageSz, bytes));
        hB = static_cast<Real*>(std::aligned_alloc(pageSz, bytes));
        if (!hA || !hB) {
            fprintf(stderr, "Host allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        const int g = nzLocal - cpu;
        const long long z0 = (long long)zStart - 1;
        auto fillRows = [&](const int zBegin, const int zEnd) {  // same row partition as cpuStencil
            const long long nRows = (long long)(zEnd - zBegin) * (long long)ny;
#pragma omp parallel for schedule(static)
            for (long long r = 0; r < nRows; ++r) {
                const size_t z = (size_t)(zBegin + r / (long long)ny);
                const size_t y = (size_t)(r % (long long)ny);
                const long long gz = z0 + (long long)z;
                const size_t base = z * P + y * nx;
                for (size_t x = 0; x < nx; ++x) {
                    const size_t gidx = (size_t)gz * P + y * nx + x;
                    hA[base + x] = (gz >= 0 && gz < (long long)nz) ? (gidx % 19) * 1.0 : 0.0;
                    hB[base + x] = 0.0;
                }
            }
        };
        fillRows(g + 1, nzLocal + 1);
        fillRows(g, g + 1);
        fillRows(nzLocal + 1, nzLocal + 2);

        auto pinWindow = [&](Real* base) {
            const uintptr_t b = (uintptr_t)(base + (size_t)g * P) / pageSz * pageSz;
            const uintptr_t e = std::min<uintptr_t>((uintptr_t)(base + (size_t)(g + 2) * P + pageSz - 1) / pageSz * pageSz,
                                                    (uintptr_t)base + bytes);
            CUDA_CHECK(cudaHostRegister((void*)b, e - b, cudaHostRegisterDefault));
            return (Real*)b;
        };
        hRegA = pinWindow(hA);
        hRegB = pinWindow(hB);
    };

    auto initialize = [&](const int cpu) {
        // Local plane 0 is the global plane zStart-1
        initKernel<<<1024, 256, 0, sInner>>>(dA, nx, ny, nz, (long long)zStart - 1, (size_t)nzLocal + 2);
        CUDA_CHECK(cudaGetLastError());
        prepareHost(cpu);
        CUDA_CHECK(cudaStreamSynchronize(sInner));
    };

    // Intra-node halo slots: every rank exposes pinned shared-memory slots
    // [parity][0: from lower neighbour | 1: from upper neighbour]; a same-node neighbour
    // copies its edge plane straight into them (D2H) and only a zero-byte MPI message
    // signals completion. Inter-node neighbours use staged MPI messages instead.
    MPI_Comm shmComm = MPI_COMM_NULL;
    MPI_Win shmWin = MPI_WIN_NULL;
    Real* mySlots = nullptr;
    Real* downSlots = nullptr;  // lower neighbour's slots, if it shares the node
    Real* upSlots = nullptr;    // upper neighbour's slots, if it shares the node
    std::vector<void*> registered;

    // Run `iters` sweeps starting from (dA, hA) with the given CPU share; the result
    // ends up in A if iters is even, in B otherwise.
    auto runIterations = [&](const int iters, const int cpu) {
        const int inx = (int)nx, iny = (int)ny, inz = (int)nz;
        const int zOffset = zStart - 1;  // global z of local plane lz is lz + zOffset
        const int g = nzLocal - cpu;     // GPU owns local planes [1, g], host owns [g+1, nzLocal]
        const size_t P = planeSize;
        const size_t pb = P * sizeof(Real);

        // A link whose lower side computes its top plane on the host uses plain MPI
        int cpuDown = 0;
        MPI_Sendrecv(&cpu, 1, MPI_INT, up, 2, &cpuDown, 1, MPI_INT, down, 2, comm, MPI_STATUS_IGNORE);
        const bool shmDown = downSlots != nullptr && cpuDown == 0;
        const bool shmUp = upSlots != nullptr && cpu == 0;

        Real* dIn = dA;
        Real* dOut = dB;
        Real* hIn = hA;
        Real* hOut = hB;
        for (int iter = 0; iter < iters; ++iter) {
            const size_t par = (size_t)(iter & 1) * 2 * P;
            Real* rbuf = hRecv + par;
            Real* fromDown = shmDown ? mySlots + par : rbuf;
            Real* fromUp = shmUp ? mySlots + par + P : (cpu > 0 ? hOut + (size_t)(nzLocal + 1) * P : rbuf + P);

            // GPU: edge planes first so their halos can travel while the interior is computed
            launchStencil(dIn, dOut, inx, iny, inz, 1, 2, zOffset, sEdge);
            if (g > 1) launchStencil(dIn, dOut, inx, iny, inz, g, g + 1, zOffset, sEdge);
            if (down != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(shmDown ? downSlots + par + P : hSend, dOut + P, pb,
                                           cudaMemcpyDeviceToHost, sEdge));
            if (cpu > 0)
                CUDA_CHECK(cudaMemcpyAsync(hOut + (size_t)g * P, dOut + (size_t)g * P, pb, cudaMemcpyDeviceToHost, sEdge));
            else if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(shmUp ? upSlots + par : hSend + P, dOut + (size_t)g * P, pb,
                                           cudaMemcpyDeviceToHost, sEdge));
            launchStencil(dIn, dOut, inx, iny, inz, 2, g, zOffset, sInner);

            MPI_Request reqs[4];
            int nr = 0;
            if (down != MPI_PROC_NULL) MPI_Irecv(fromDown, shmDown ? 0 : 1, planeType, down, 0, comm, &reqs[nr++]);
            if (up != MPI_PROC_NULL) MPI_Irecv(fromUp, shmUp ? 0 : 1, planeType, up, 1, comm, &reqs[nr++]);

            // Host: top plane first (it is sent upwards), then the rest of the host part
            if (cpu > 0) {
                if (up != MPI_PROC_NULL) {
                    cpuStencil(hIn, hOut, nx, ny, nz, nzLocal, nzLocal + 1, zOffset);
                    MPI_Isend(hOut + (size_t)nzLocal * P, 1, planeType, up, 0, comm, &reqs[nr++]);
                    cpuStencil(hIn, hOut, nx, ny, nz, g + 1, nzLocal, zOffset);
                } else {
                    cpuStencil(hIn, hOut, nx, ny, nz, g + 1, nzLocal + 1, zOffset);
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(sEdge));
            if (cpu == 0 && up != MPI_PROC_NULL) MPI_Isend(hSend + P, shmUp ? 0 : 1, planeType, up, 0, comm, &reqs[nr++]);
            if (down != MPI_PROC_NULL) MPI_Isend(hSend, shmDown ? 0 : 1, planeType, down, 1, comm, &reqs[nr++]);
            MPI_Waitall(nr, reqs, MPI_STATUSES_IGNORE);

            // Refresh the device ghost planes of the freshly written buffer
            if (down != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(dOut, fromDown, pb, cudaMemcpyHostToDevice, sEdge));
            if (cpu > 0)
                CUDA_CHECK(cudaMemcpyAsync(dOut + (size_t)(g + 1) * P, hOut + (size_t)(g + 1) * P, pb, cudaMemcpyHostToDevice, sEdge));
            else if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(dOut + (size_t)(g + 1) * P, fromUp, pb, cudaMemcpyHostToDevice, sEdge));

            // Both streams must finish this sweep before the next one starts
            CUDA_CHECK(cudaEventRecord(evEdge, sEdge));
            CUDA_CHECK(cudaEventRecord(evInner, sInner));
            CUDA_CHECK(cudaStreamWaitEvent(sInner, evEdge, 0));
            CUDA_CHECK(cudaStreamWaitEvent(sEdge, evInner, 0));
            std::swap(dIn, dOut);
            std::swap(hIn, hOut);
        }
        CUDA_CHECK(cudaStreamSynchronize(sEdge));
        CUDA_CHECK(cudaStreamSynchronize(sInner));
    };

    if (active) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        nzLocal = planeCounts[rank];
        zStart = planeDispls[rank];
        if (rank > 0) down = rank - 1;
        if (rank < nActive - 1) up = rank + 1;

        const size_t localElems = planeSize * (size_t)(nzLocal + 2);
        const size_t localBytes = localElems * sizeof(Real);
        CUDA_CHECK(cudaMalloc(&dA, localBytes));
        CUDA_CHECK(cudaMalloc(&dB, localBytes));
        CUDA_CHECK(cudaMemset(dA, 0, localBytes));
        CUDA_CHECK(cudaMemset(dB, 0, localBytes));
        CUDA_CHECK(cudaMallocHost(&hSend, 2 * planeSize * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&hRecv, 4 * planeSize * sizeof(Real)));
        // Halo work gets priority so its blocks are scheduled ahead of the interior sweep
        int prioLow = 0, prioHigh = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLow, &prioHigh));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sEdge, cudaStreamNonBlocking, prioHigh));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sInner, cudaStreamNonBlocking, prioLow));
        CUDA_CHECK(cudaEventCreateWithFlags(&evEdge, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evInner, cudaEventDisableTiming));

        // Shared-memory halo slots for same-node neighbours
        {
            MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shmComm);
            const size_t pageSz = (size_t)sysconf(_SC_PAGESIZE);
            const size_t slotBytes = (4 * planeSize * sizeof(Real) + pageSz - 1) / pageSz * pageSz;
            MPI_Info info;
            MPI_Info_create(&info);
            MPI_Info_set(info, "alloc_shared_noncontig", "true");
            MPI_Win_allocate_shared((MPI_Aint)slotBytes, sizeof(Real), info, shmComm, &mySlots, &shmWin);
            MPI_Info_free(&info);
            std::memset(mySlots, 0, slotBytes);

            MPI_Group worldGroup, shmGroup;
            MPI_Comm_group(comm, &worldGroup);
            MPI_Comm_group(shmComm, &shmGroup);
            auto peerSlots = [&](int nbr) -> Real* {
                if (nbr == MPI_PROC_NULL) return nullptr;
                int shmRank = MPI_UNDEFINED;
                MPI_Group_translate_ranks(worldGroup, 1, &nbr, shmGroup, &shmRank);
                if (shmRank == MPI_UNDEFINED) return nullptr;
                MPI_Aint sz = 0;
                int du = 0;
                Real* ptr = nullptr;
                MPI_Win_shared_query(shmWin, shmRank, &sz, &du, &ptr);
                return ((size_t)sz >= 4 * planeSize * sizeof(Real)) ? ptr : nullptr;
            };
            downSlots = peerSlots(down);
            upSlots = peerSlots(up);
            MPI_Group_free(&worldGroup);
            MPI_Group_free(&shmGroup);
            // Pin the node's whole window (all ranks' slots) once in this process
            int shmSize = 1;
            MPI_Comm_size(shmComm, &shmSize);
            uintptr_t lo = UINTPTR_MAX, hi = 0;
            for (int r = 0; r < shmSize; ++r) {
                MPI_Aint sz = 0;
                int du = 0;
                Real* ptr = nullptr;
                MPI_Win_shared_query(shmWin, r, &sz, &du, &ptr);
                lo = std::min(lo, (uintptr_t)ptr);
                hi = std::max(hi, (uintptr_t)ptr + (uintptr_t)sz);
            }
            lo = lo / pageSz * pageSz;
            hi = (hi + pageSz - 1) / pageSz * pageSz;
            CUDA_CHECK(cudaHostRegister((void*)lo, hi - lo, cudaHostRegisterDefault));
            registered.push_back((void*)lo);
        }

        if (root) printf("Initializing grid...\n");
        initialize(0);
    }

    // ---- GPU/CPU work split: measure both and balance (not part of the timed region) ----
    if (active && iterations > 0) {
        const int inx = (int)nx, iny = (int)ny, inz = (int)nz;
        const int zOffset = zStart - 1;
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        auto gpuTime = [&](int z0, int z1) {  // seconds per sweep over local planes [z0, z1)
            launchStencil(dA, dB, inx, iny, inz, z0, z1, zOffset, sInner);
            CUDA_CHECK(cudaEventRecord(t0, sInner));
            const int reps = 3;
            for (int r = 0; r < reps; ++r) launchStencil(dA, dB, inx, iny, inz, z0, z1, zOffset, sInner);
            CUDA_CHECK(cudaEventRecord(t1, sInner));
            CUDA_CHECK(cudaEventSynchronize(t1));
            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
            return ms * 1e-3 / reps;
        };
        auto cpuTime = [&](int z0, int z1) {
            cpuStencil(hA, hB, nx, ny, nz, z0, z1, zOffset);
            const double s = MPI_Wtime();
            const int reps = 3;
            for (int r = 0; r < reps; ++r) cpuStencil(hA, hB, nx, ny, nz, z0, z1, zOffset);
            return (MPI_Wtime() - s) / reps;
        };

        // Fixed number of rounds on every rank: the measurements are barrier-synchronized
        int cpu = 0;
        int trial = std::max(1, nzLocal / 5);
        for (int round = 0; round < 3; ++round) {
            const bool measure = nzLocal > 1 && trial > 0;
            const int g = nzLocal - trial;
            if (measure) prepareHost(trial);
            MPI_Barrier(comm);  // all ranks of a node measure concurrently (shared CPUs)
            const double tg = measure ? gpuTime(1, g + 1) / g : 0.0;
            MPI_Barrier(comm);
            const double tc = measure ? cpuTime(g + 1, nzLocal + 1) / trial : 0.0;
            if (measure) {
                const double share = (1.0 / tc) / (1.0 / tc + 1.0 / tg);
                cpu = std::min(nzLocal - 1, (int)std::floor(share * nzLocal));
                trial = cpu;
            }
        }
        CUDA_CHECK(cudaEventDestroy(t0));
        CUDA_CHECK(cudaEventDestroy(t1));

        // The model ignores cache and transfer effects: pick the best of a few candidate
        // shares (including GPU-only) using short real runs
        int anyCpu = 0;
        MPI_Allreduce(&cpu, &anyCpu, 1, MPI_INT, MPI_MAX, comm);
        if (anyCpu > 0) {
            const int nCand = 5;  // cpu >> k for k = 0..3, then GPU-only
            auto candidate = [&](int k) { return k < nCand - 1 ? (cpu >> k) : 0; };
            const int trialIters = 8;
            int best = nCand - 1;
            double bestTime = 1e300;
            for (int k = 0; k < nCand; ++k) {
                double t = 1e300;
                for (int rep = 0; rep < 2; ++rep) {
                    initialize(candidate(k));
                    MPI_Barrier(comm);
                    const double s = MPI_Wtime();
                    runIterations(trialIters, candidate(k));
                    MPI_Barrier(comm);
                    t = std::min(t, MPI_Wtime() - s);
                }
                if (t < bestTime) {
                    bestTime = t;
                    best = k;
                }
            }
            // Timings are barrier-delimited, so rank 0's choice holds for everyone
            MPI_Bcast(&best, 1, MPI_INT, 0, comm);
            cpuPlanes = candidate(best);
            initialize(cpuPlanes);  // restore the initial state
        } else {
            prepareHost(0);
        }
    }

    // Run stencil iterations
    if (root) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (active) {
        runIterations(iterations, cpuPlanes);
        finalInA = (iterations % 2 == 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int exitCode = 0;
    if ((printResults || validate) && active) {
        // Assemble the final grid on rank 0
        Real* dFinal = finalInA ? dA : dB;
        Real* hFinal = finalInA ? hA : hB;
        const int g = nzLocal - cpuPlanes;
        std::vector<Real> finalGrid;
        if (root) finalGrid.resize(gridSize);

        Real* hLocal = nullptr;
        CUDA_CHECK(cudaMallocHost(&hLocal, std::max<size_t>(1, planeSize * nzLocal) * sizeof(Real)));
        CUDA_CHECK(cudaMemcpy(hLocal, dFinal + planeSize, planeSize * g * sizeof(Real), cudaMemcpyDeviceToHost));
        if (cpuPlanes > 0)
            std::memcpy(hLocal + planeSize * g, hFinal + planeSize * (g + 1), planeSize * cpuPlanes * sizeof(Real));
        MPI_Gatherv(hLocal, nzLocal, planeType, root ? finalGrid.data() : nullptr, planeCounts.data(),
                    planeDispls.data(), planeType, 0, comm);
        CUDA_CHECK(cudaFreeHost(hLocal));

        if (root) {
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
    }

    MPI_Type_free(&planeType);
    if (active) {
        CUDA_CHECK(cudaEventDestroy(evEdge));
        CUDA_CHECK(cudaEventDestroy(evInner));
        CUDA_CHECK(cudaStreamDestroy(sEdge));
        CUDA_CHECK(cudaStreamDestroy(sInner));
        CUDA_CHECK(cudaFreeHost(hSend));
        CUDA_CHECK(cudaFreeHost(hRecv));
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        prepareHost(0);
        for (void* r : registered) CUDA_CHECK(cudaHostUnregister(r));
        MPI_Win_free(&shmWin);
        MPI_Comm_free(&shmComm);
        MPI_Comm_free(&comm);
    }
    MPI_Finalize();
    return exitCode;
}
