#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <vector>

#include <sched.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t err_ = (call);                                                   \
        if (err_ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),    \
                    __FILE__, __LINE__);                                             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                            \
        }                                                                            \
    } while (0)

// Allocate pinned host memory on the NUMA node local to the given GPU: the calling
// thread is temporarily bound to the GPU's local CPUs (from Linux sysfs) so the pages
// are first-touched there, then the original (launcher-provided) binding is restored.
static Real* allocPinnedNearGpu(const size_t bytes, const int device) {
    cpu_set_t oldSet, gpuSet;
    bool rebound = false;
    char busId[32] = {0};
    if (sched_getaffinity(0, sizeof(oldSet), &oldSet) == 0 &&
        cudaDeviceGetPCIBusId(busId, sizeof(busId), device) == cudaSuccess) {
        for (char* p = busId; *p; ++p) *p = (char)tolower(*p);
        char path[128];
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/local_cpulist", busId);
        if (FILE* f = fopen(path, "r")) {
            char buf[4096];
            if (fgets(buf, sizeof(buf), f)) {
                CPU_ZERO(&gpuSet);
                const char* s = buf;
                while (*s) {
                    char* endp;
                    const long lo = strtol(s, &endp, 10);
                    if (endp == s) break;
                    long hi = lo;
                    s = endp;
                    if (*s == '-') {
                        hi = strtol(s + 1, &endp, 10);
                        s = endp;
                    }
                    for (long c = lo; c <= hi && c < CPU_SETSIZE; ++c) CPU_SET((int)c, &gpuSet);
                    if (*s != ',') break;
                    ++s;
                }
                rebound = CPU_COUNT(&gpuSet) > 0 && sched_setaffinity(0, sizeof(gpuSet), &gpuSet) == 0;
            }
            fclose(f);
        }
    }
    Real* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, bytes));
    memset(ptr, 0, bytes);
    if (rebound) sched_setaffinity(0, sizeof(oldSet), &oldSet);
    return ptr;
}

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int Z_CHUNK = 16;

// Initialize the local slab (owned planes plus halo planes) on the device.
// Local plane lz corresponds to global plane gz0 + lz.
__global__ void initKernel(Real* __restrict__ grid, const size_t nx, const size_t ny,
                           const long gz0, const int nPlanes) {
    const size_t plane = nx * ny;
    const size_t total = plane * (size_t)nPlanes;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const size_t gidx = (size_t)gz0 * plane + i;
        grid[i] = (gidx % 19) * 1.0;
    }
}

// Correctly rounded s / 7.0 without the (slow on many GPUs) FP64 division:
// q = RN(s * RN(1/7)) is faithful (RN(1/7) has relative error 2^-54), so by
// Markstein's theorem one FMA-based correction yields exactly RN(s / 7).
__device__ __forceinline__ Real div7(const Real s) {
    constexpr Real inv7 = 1.0 / 7.0;
    const Real q = __dmul_rn(s, inv7);
    const Real r = __fma_rn(-q, 7.0, s);
    return __fma_rn(r, inv7, q);
}

// 7-point stencil over local planes [zlo, zhi] (inclusive), interior x/y only.
// Each thread marches along z, keeping the bottom/center/top values in registers.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
              const int nx, const int ny, const int zlo, const int zhi) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x + 1;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y + 1;
    if (x >= nx - 1 || y >= ny - 1) return;

    const int zs = zlo + blockIdx.z * Z_CHUNK;
    if (zs > zhi) return;
    const int ze = min(zs + Z_CHUNK - 1, zhi);

    const size_t sx = 1, sy = (size_t)nx, sz = (size_t)nx * ny;
    size_t idx = (size_t)zs * sz + (size_t)y * sy + x;

    Real bottom = __ldg(&in[idx - sz]);
    Real center = __ldg(&in[idx]);
    for (int z = zs; z <= ze; ++z) {
        const Real top = __ldg(&in[idx + sz]);
        const Real left = __ldg(&in[idx - sx]);
        const Real right = __ldg(&in[idx + sx]);
        const Real front = __ldg(&in[idx - sy]);
        const Real back = __ldg(&in[idx + sy]);
        // Same summation order as the reference implementation
        out[idx] = div7((((((center + left) + right) + front) + back) + bottom) + top);
        bottom = center;
        center = top;
        idx += sz;
    }
}

// Update the two (distinct) boundary planes zA and zB of the slab in one launch;
// zB < 0 disables the second plane.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
stencilPlanesKernel(const Real* __restrict__ in, Real* __restrict__ out,
                    const int nx, const int ny, const int zA, const int zB) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x + 1;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y + 1;
    const int z = (blockIdx.z == 0) ? zA : zB;
    if (x >= nx - 1 || y >= ny - 1 || z < 0) return;

    const size_t sy = (size_t)nx, sz = (size_t)nx * ny;
    const size_t idx = (size_t)z * sz + (size_t)y * sy + x;
    const Real center = __ldg(&in[idx]);
    const Real left = __ldg(&in[idx - 1]);
    const Real right = __ldg(&in[idx + 1]);
    const Real front = __ldg(&in[idx - sy]);
    const Real back = __ldg(&in[idx + sy]);
    const Real bottom = __ldg(&in[idx - sz]);
    const Real top = __ldg(&in[idx + sz]);
    out[idx] = div7((((((center + left) + right) + front) + back) + bottom) + top);
}

static void launchPlanes(const Real* in, Real* out, const size_t nx, const size_t ny,
                         int zA, int zB, cudaStream_t stream) {
    if (zA < 0) std::swap(zA, zB);
    if (zA < 0 || nx < 3 || ny < 3) return;
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((unsigned)((nx - 2 + BLOCK_X - 1) / BLOCK_X),
              (unsigned)((ny - 2 + BLOCK_Y - 1) / BLOCK_Y), zB < 0 ? 1u : 2u);
    stencilPlanesKernel<<<grid, block, 0, stream>>>(in, out, (int)nx, (int)ny, zA, zB);
}

static void launchStencil(const Real* in, Real* out, const size_t nx, const size_t ny,
                          const int zlo, const int zhi, cudaStream_t stream) {
    if (zlo > zhi || nx < 3 || ny < 3) return;
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((unsigned)((nx - 2 + BLOCK_X - 1) / BLOCK_X),
              (unsigned)((ny - 2 + BLOCK_Y - 1) / BLOCK_Y),
              (unsigned)((zhi - zlo + 1 + Z_CHUNK - 1) / Z_CHUNK));
    stencilKernel<<<grid, block, 0, stream>>>(in, out, (int)nx, (int)ny, zlo, zhi);
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    const size_t n = grid.size();
    const Real* g = grid.data();

    // 1. No NaN or Inf values
    int bad = 0;
    #pragma omp parallel for reduction(|:bad) schedule(static)
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
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
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
    const size_t plane = nx * ny;

    // ---- Domain decomposition: 1D slabs along z; at most one rank per plane ----
    const int nActive = (int)std::min<size_t>((size_t)worldSize, std::max<size_t>(nz, 1));
    const bool active = worldRank < nActive;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &comm);

    // Plane counts / offsets for all active ranks (needed by root for gathering)
    std::vector<int> planeCounts(nActive), planeOffsets(nActive);
    for (int r = 0; r < nActive; ++r) {
        const size_t base = nz / nActive, rem = nz % nActive;
        planeCounts[r] = (int)(base + ((size_t)r < rem ? 1 : 0));
        planeOffsets[r] = (int)(r * base + std::min<size_t>((size_t)r, rem));
    }

    // ---- GPU selection: round-robin over node-local ranks ----
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_free(&nodeComm);
    int nDev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nDev));
    if (nDev == 0) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(nodeRank % nDev));

    int rank = 0, nRanks = 1, nl = 0;
    long gz0 = 0;          // global z of local plane 1 (first owned plane)
    int lower = MPI_PROC_NULL, upper = MPI_PROC_NULL;
    Real* d_a = nullptr;
    Real* d_b = nullptr;
    Real* h_halo = nullptr;  // pinned: sendLo, sendHi, recvLo, recvHi
    cudaStream_t sInt = nullptr, sBnd = nullptr;
    MPI_Datatype planeType = MPI_DATATYPE_NULL;

    if (active) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &nRanks);
        nl = planeCounts[rank];
        gz0 = planeOffsets[rank];
        lower = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        upper = (rank < nRanks - 1) ? rank + 1 : MPI_PROC_NULL;

        MPI_Type_contiguous((int)plane, MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);

        // Local slab: halo plane 0, owned planes 1..nl, halo plane nl+1
        const size_t localElems = plane * (size_t)(nl + 2);
        CUDA_CHECK(cudaMalloc(&d_a, localElems * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_b, localElems * sizeof(Real)));
        h_halo = allocPinnedNearGpu(4 * plane * sizeof(Real), nodeRank % nDev);
        // Boundary/halo work gets the highest priority so it is not delayed by the bulk update
        int prioLow = 0, prioHigh = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLow, &prioHigh));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sInt, cudaStreamNonBlocking, prioLow));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sBnd, cudaStreamNonBlocking, prioHigh));
    }

    // ---- Initialize ----
    if (root) printf("Initializing grid...\n");
    if (active) {
        // Initialize the planes that exist globally (halo planes beyond the domain stay unused).
        const long gFirst = std::max<long>(gz0 - 1, 0);
        const long gLast = std::min<long>(gz0 + nl, (long)nz - 1);
        const int nInit = (int)(gLast - gFirst + 1);
        const size_t offset = (size_t)(gFirst - (gz0 - 1)) * plane;
        const size_t total = plane * (size_t)nInit;
        const int threads = 256;
        const int blocks = (int)std::min<size_t>((total + threads - 1) / threads, 65535 * 16);
        if (total > 0) {
            initKernel<<<blocks, threads>>>(d_a + offset, nx, ny, gFirst, nInit);
            CUDA_CHECK(cudaGetLastError());
        }
        // Boundary values are copied unchanged every iteration, so both buffers
        // start out identical and only interior points are ever rewritten.
        CUDA_CHECK(cudaMemcpy(d_b, d_a, plane * (size_t)(nl + 2) * sizeof(Real), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Local plane range that is part of the global interior (z in [1, nz-2])
    int zIntLo = 1, zIntHi = 0;
    if (active && nz >= 3) {
        zIntLo = (int)std::max<long>(1, 1 - gz0 + 1);              // local of global z=1
        zIntHi = (int)std::min<long>(nl, (long)nz - 2 - gz0 + 1);  // local of global z=nz-2
    }

    // ---- Run stencil iterations ----
    if (root) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (active) {
        Real* sendLo = h_halo;
        Real* sendHi = h_halo + plane;
        Real* recvLo = h_halo + 2 * plane;
        Real* recvHi = h_halo + 3 * plane;
        const size_t planeBytes = plane * sizeof(Real);
        const bool hasLo = (lower != MPI_PROC_NULL);
        const bool hasHi = (upper != MPI_PROC_NULL);

        // Boundary planes (adjacent to halos) and interior planes of this slab
        const bool firstIsInt = (zIntLo <= 1 && 1 <= zIntHi);
        const bool lastIsInt = (nl > 1 && zIntLo <= nl && nl <= zIntHi);
        const int innerLo = std::max(zIntLo, 2);
        const int innerHi = std::min(zIntHi, nl - 1);

        Real* in = d_a;
        Real* out = d_b;
        for (int iter = 0; iter < iterations; ++iter) {
            // Planes next to the halos first, so their exchange overlaps the bulk update
            launchPlanes(in, out, nx, ny, firstIsInt ? 1 : -1, lastIsInt ? nl : -1, sBnd);
            launchStencil(in, out, nx, ny, innerLo, innerHi, sInt);

            if (nRanks > 1) {
                if (hasLo) CUDA_CHECK(cudaMemcpyAsync(sendLo, out + plane, planeBytes, cudaMemcpyDeviceToHost, sBnd));
                if (hasHi) CUDA_CHECK(cudaMemcpyAsync(sendHi, out + (size_t)nl * plane, planeBytes, cudaMemcpyDeviceToHost, sBnd));
                CUDA_CHECK(cudaStreamSynchronize(sBnd));

                // Receives occupy slots 0/1 so each halo upload can start as soon as it arrives
                MPI_Request reqs[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
                if (hasLo) MPI_Irecv(recvLo, 1, planeType, lower, 1, comm, &reqs[0]);
                if (hasHi) MPI_Irecv(recvHi, 1, planeType, upper, 0, comm, &reqs[1]);
                if (hasLo) MPI_Isend(sendLo, 1, planeType, lower, 0, comm, &reqs[2]);
                if (hasHi) MPI_Isend(sendHi, 1, planeType, upper, 1, comm, &reqs[3]);
                for (int pending = (int)hasLo + (int)hasHi; pending > 0; --pending) {
                    int which = MPI_UNDEFINED;
                    MPI_Waitany(2, reqs, &which, MPI_STATUS_IGNORE);
                    if (which == 0) {
                        CUDA_CHECK(cudaMemcpyAsync(out, recvLo, planeBytes, cudaMemcpyHostToDevice, sBnd));
                    } else if (which == 1) {
                        CUDA_CHECK(cudaMemcpyAsync(out + (size_t)(nl + 1) * plane, recvHi, planeBytes, cudaMemcpyHostToDevice, sBnd));
                    }
                }
                MPI_Waitall(2, &reqs[2], MPI_STATUSES_IGNORE);
            }
            CUDA_CHECK(cudaStreamSynchronize(sBnd));
            CUDA_CHECK(cudaStreamSynchronize(sInt));
            std::swap(in, out);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
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
    if (printResults || validate) {
        // Gather the final grid on the root rank
        std::vector<Real> finalGrid;
        if (active) {
            const Real* d_final = (iterations % 2 == 0) ? d_a : d_b;
            if (root) {
                finalGrid.resize(gridSize);
                CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final + plane, (size_t)nl * plane * sizeof(Real), cudaMemcpyDeviceToHost));
                MPI_Gatherv(MPI_IN_PLACE, 0, planeType, finalGrid.data(), planeCounts.data(),
                            planeOffsets.data(), planeType, 0, comm);
            } else {
                std::vector<Real> local((size_t)nl * plane);
                CUDA_CHECK(cudaMemcpy(local.data(), d_final + plane, local.size() * sizeof(Real), cudaMemcpyDeviceToHost));
                MPI_Gatherv(local.data(), nl, planeType, nullptr, nullptr, nullptr, planeType, 0, comm);
            }
        }

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
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (active) {
        CUDA_CHECK(cudaStreamDestroy(sInt));
        CUDA_CHECK(cudaStreamDestroy(sBnd));
        CUDA_CHECK(cudaFreeHost(h_halo));
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        MPI_Type_free(&planeType);
        MPI_Comm_free(&comm);
    }

    MPI_Finalize();
    return exitCode;
}
