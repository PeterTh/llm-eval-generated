#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <sys/syscall.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call)                                                                                   \
    do {                                                                                                   \
        const cudaError_t err_ = (call);                                                                    \
        if (err_ != cudaSuccess) {                                                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__);      \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                   \
        }                                                                                                   \
    } while (0)

// Correctly rounded x / 7.0 without the (very slow on consumer GPUs) FP64 divider.
// Markstein refinement: with a correctly rounded reciprocal the FMA-based residual
// correction reproduces the IEEE round-to-nearest quotient exactly.
__device__ __forceinline__ Real divideBySeven(const Real x) {
    constexpr Real r = 1.0 / 7.0;
    const Real q = x * r;
    const Real e = __fma_rn(-7.0, q, x);  // exact residual
    return __fma_rn(e, r, q);
}

// 7-point stencil kernel. Computes the local z-planes [zbeg, zend) of the halo-padded
// local sub-grid; x/y boundaries of the global domain are never written.
// Each thread marches over zChunk planes, keeping the z-neighbours in registers.
// The summation order matches the original scalar code exactly, so results are bit-identical.
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              const int nx, const int ny, const int zbeg, const int zend,
                              const int zChunk) {
    const int x = 1 + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = 1 + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= nx - 1 || y >= ny - 1) return;

    const int z0 = zbeg + static_cast<int>(blockIdx.z) * zChunk;
    if (z0 >= zend) return;
    const int z1 = min(z0 + zChunk, zend);

    const size_t slice = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t base = static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);

    Real bottom = input[base + static_cast<size_t>(z0 - 1) * slice];
    Real center = input[base + static_cast<size_t>(z0) * slice];

    for (int z = z0; z < z1; ++z) {
        const size_t idx = base + static_cast<size_t>(z) * slice;
        const Real top = input[idx + slice];
        const Real left = input[idx - 1];
        const Real right = input[idx + 1];
        const Real front = input[idx - nx];
        const Real back = input[idx + nx];

        // Simple averaging stencil
        output[idx] = divideBySeven(center + left + right + front + back + bottom + top);

        bottom = center;
        center = top;
    }
}

// Initialize the halo-padded local slab; planes outside the global domain are zeroed.
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz,
                    const size_t zStart, const size_t nzLocal) {
    const size_t plane = nx * ny;
#pragma omp parallel for schedule(static)
    for (size_t lz = 0; lz < nzLocal + 2; ++lz) {
        const size_t gz = lz + zStart;  // global z is lz + zStart - 1, offset by one for the halo
        if (gz == 0 || gz > nz) {       // ghost plane outside the global domain
            std::fill_n(grid.data() + lz * plane, plane, Real(0));
            continue;
        }
        const size_t z = gz - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[lz * plane + y * nx + x] = (idx % 19) * 1.0;
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    bool bad = false;
#pragma omp parallel for schedule(static) reduction(|| : bad)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real val = grid[i];
        bad = bad || std::isnan(val) || std::isinf(val);
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

// NUMA node this process is currently running on (-1 if it cannot be determined).
static int cpuNumaNode() {
    unsigned cpu = 0, node = 0;
    if (syscall(SYS_getcpu, &cpu, &node, nullptr) != 0) return -1;
    return static_cast<int>(node);
}

// NUMA node the given CUDA device is attached to (-1 if unknown).
static int pciNumaNode(const int device) {
    char busId[32] = {0};
    if (cudaDeviceGetPCIBusId(busId, sizeof(busId), device) != cudaSuccess) return -1;
    for (char* c = busId; *c; ++c) *c = static_cast<char>(tolower(static_cast<unsigned char>(*c)));
    char path[96];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", busId);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int node = -1;
    if (fscanf(f, "%d", &node) != 1) node = -1;
    fclose(f);
    return node;
}

// Collect the distributed slabs into the full grid on rank 0.
static void gatherGrid(const std::vector<Real>& local, std::vector<Real>& full, const size_t plane,
                       const std::vector<size_t>& zStart, const std::vector<size_t>& nzLocal,
                       const int rank, const int numRanks) {
    constexpr size_t kChunk = 1u << 24;  // elements per MPI message
    if (rank == 0) {
        if (nzLocal[0] > 0) {
            std::copy_n(local.data() + plane, nzLocal[0] * plane, full.data() + zStart[0] * plane);
        }
        for (int r = 1; r < numRanks; ++r) {
            size_t remaining = nzLocal[r] * plane;
            Real* dst = full.data() + zStart[r] * plane;
            while (remaining > 0) {
                const size_t n = std::min(remaining, kChunk);
                MPI_Recv(dst, static_cast<int>(n), MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                dst += n;
                remaining -= n;
            }
        }
    } else {
        size_t remaining = nzLocal[rank] * plane;
        const Real* src = local.data() + plane;
        while (remaining > 0) {
            const size_t n = std::min(remaining, kChunk);
            MPI_Send(src, static_cast<int>(n), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            src += n;
            remaining -= n;
        }
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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
        printf("MPI ranks: %d, OpenMP threads: %d\n", numRanks, omp_get_max_threads());
    }

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // Bind one GPU per rank
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);

    // Pick a GPU attached to the NUMA node this rank runs on: host staging of the halos
    // goes over PCIe, and crossing the socket interconnect costs a large part of that
    // bandwidth. Every local rank runs the same greedy assignment, so the result is unique.
    std::vector<int> deviceNuma(deviceCount);
    for (int d = 0; d < deviceCount; ++d) {
        deviceNuma[d] = pciNumaNode(d);
    }
    std::vector<int> rankNuma(localSize);
    const int myNuma = cpuNumaNode();
    MPI_Allgather(&myNuma, 1, MPI_INT, rankNuma.data(), 1, MPI_INT, nodeComm);
    MPI_Comm_free(&nodeComm);

    int device = localRank % deviceCount;
    {
        std::vector<char> taken(deviceCount, 0);
        for (int lr = 0; lr < std::min(localSize, deviceCount); ++lr) {
            int pick = -1;
            for (int d = 0; d < deviceCount; ++d) {  // prefer a free NUMA-local device
                if (!taken[d] && deviceNuma[d] == rankNuma[lr] && rankNuma[lr] >= 0) { pick = d; break; }
            }
            for (int d = 0; pick < 0 && d < deviceCount; ++d) {
                if (!taken[d]) pick = d;
            }
            taken[pick] = 1;
            if (lr == localRank) device = pick;
        }
    }
    CUDA_CHECK(cudaSetDevice(device));

    // 1D block decomposition along z
    std::vector<size_t> zStart(numRanks), nzLocal(numRanks);
    {
        const size_t base = nz / numRanks;
        const size_t rem = nz % numRanks;
        size_t off = 0;
        for (int r = 0; r < numRanks; ++r) {
            nzLocal[r] = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            zStart[r] = off;
            off += nzLocal[r];
        }
    }
    const size_t zs = zStart[rank];
    const size_t nzLoc = nzLocal[rank];
    const size_t localPlanes = nzLoc + 2;  // one halo plane on each side
    const bool active = nzLoc > 0;

    // Neighbours (skipping any empty ranks)
    int left = MPI_PROC_NULL, right = MPI_PROC_NULL;
    if (active) {
        for (int r = rank - 1; r >= 0; --r) {
            if (nzLocal[r] > 0) { left = r; break; }
        }
        for (int r = rank + 1; r < numRanks; ++r) {
            if (nzLocal[r] > 0) { right = r; break; }
        }
    }

    // Initialize (host, OpenMP) and upload to the device
    if (rank == 0) printf("Initializing grid...\n");
    std::vector<Real> hostGrid(active ? localPlanes * plane : 0);
    if (active) initializeGrid(hostGrid, nx, ny, nz, zs, nzLoc);

    Real* dGrid[2] = {nullptr, nullptr};
    Real* hHalo[4] = {nullptr, nullptr, nullptr, nullptr};  // sendLeft, sendRight, recvLeft, recvRight
    cudaStream_t streamEdge = nullptr, streamInner = nullptr;
    if (active) {
        const size_t bytes = localPlanes * plane * sizeof(Real);
        CUDA_CHECK(cudaMalloc(&dGrid[0], bytes));
        CUDA_CHECK(cudaMalloc(&dGrid[1], bytes));
        // Both buffers start from the initial grid: the original code re-copies the domain
        // boundary from input to output every iteration, so boundary values stay constant.
        CUDA_CHECK(cudaMemcpy(dGrid[0], hostGrid.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dGrid[1], hostGrid.data(), bytes, cudaMemcpyHostToDevice));
        for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaMallocHost(&hHalo[i], plane * sizeof(Real)));
        CUDA_CHECK(cudaStreamCreate(&streamEdge));
        CUDA_CHECK(cudaStreamCreate(&streamInner));
    }

    // Local plane range to compute: global z in [1, nz-1) mapped to local indices
    int computeLo = 0, computeHi = 0;
    if (active && nz >= 3 && nx >= 3 && ny >= 3) {
        computeLo = static_cast<int>(std::max<size_t>(zs, 1) - zs + 1);
        computeHi = static_cast<int>(std::min<size_t>(zs + nzLoc, nz - 1) - zs + 1);
        if (computeHi < computeLo) computeHi = computeLo;
    }
    // Planes whose values have to be shipped to the neighbours are computed first
    const int edgeLo = (computeLo <= 1 && 1 < computeHi) ? 1 : -1;
    const int edgeHi = (static_cast<int>(nzLoc) != edgeLo && computeLo <= static_cast<int>(nzLoc) &&
                        static_cast<int>(nzLoc) < computeHi)
                           ? static_cast<int>(nzLoc)
                           : -1;
    const int innerLo = (edgeLo > 0) ? std::max(computeLo, 2) : computeLo;
    const int innerHi = (edgeHi > 0) ? std::min(computeHi, static_cast<int>(nzLoc)) : computeHi;

    const dim3 block(32, 8, 1);
    const dim3 gridXY(nx > 2 ? (static_cast<unsigned>(nx - 2) + block.x - 1) / block.x : 0,
                      ny > 2 ? (static_cast<unsigned>(ny - 2) + block.y - 1) / block.y : 0, 1);

    // Pick the number of z-planes per block so that the whole GPU stays saturated:
    // longer z-marches amortize the loads better, but must not starve the SMs of blocks.
    int smCount = 1;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    const size_t targetBlocks = static_cast<size_t>(smCount) * 8;
    const size_t blocksXY = std::max<size_t>(1, static_cast<size_t>(gridXY.x) * gridXY.y);
    const int zChunk = static_cast<int>(std::clamp<size_t>(
        (static_cast<size_t>(std::max(innerHi - innerLo, 1)) * blocksXY) / targetBlocks, 1, 32));

    auto launch = [&](const Real* in, Real* out, int zbeg, int zend, cudaStream_t s) {
        if (zend <= zbeg || gridXY.x == 0 || gridXY.y == 0) return;
        const dim3 g(gridXY.x, gridXY.y, (static_cast<unsigned>(zend - zbeg) + zChunk - 1) / zChunk);
        stencilKernel<<<g, block, 0, s>>>(in, out, static_cast<int>(nx), static_cast<int>(ny), zbeg, zend,
                                          zChunk);
    };

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    for (int iter = 0; iter < iterations; ++iter) {
        if (!active) break;
        const Real* in = dGrid[iter % 2];
        Real* out = dGrid[(iter + 1) % 2];

        // 1. Compute the planes the neighbours need and stage them on the host
        if (edgeLo > 0) launch(in, out, edgeLo, edgeLo + 1, streamEdge);
        if (edgeHi > 0) launch(in, out, edgeHi, edgeHi + 1, streamEdge);
        if (left != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(hHalo[0], out + plane, plane * sizeof(Real),
                                       cudaMemcpyDeviceToHost, streamEdge));
        }
        if (right != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(hHalo[1], out + nzLoc * plane, plane * sizeof(Real),
                                       cudaMemcpyDeviceToHost, streamEdge));
        }
        CUDA_CHECK(cudaStreamSynchronize(streamEdge));

        // 2. Exchange halos; the interior planes do not depend on them and are computed
        //    concurrently with the transfer.
        MPI_Request reqs[4];
        int nreq = 0;
        if (left != MPI_PROC_NULL) {
            MPI_Irecv(hHalo[2], static_cast<int>(plane), MPI_DOUBLE, left, 1, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Isend(hHalo[0], static_cast<int>(plane), MPI_DOUBLE, left, 2, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (right != MPI_PROC_NULL) {
            MPI_Irecv(hHalo[3], static_cast<int>(plane), MPI_DOUBLE, right, 2, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Isend(hHalo[1], static_cast<int>(plane), MPI_DOUBLE, right, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        }

        launch(in, out, innerLo, innerHi, streamInner);

        if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        // 3. Push the received halo planes back to the device
        if (left != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(out, hHalo[2], plane * sizeof(Real), cudaMemcpyHostToDevice,
                                       streamEdge));
        }
        if (right != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(out + (nzLoc + 1) * plane, hHalo[3], plane * sizeof(Real),
                                       cudaMemcpyHostToDevice, streamEdge));
        }
        CUDA_CHECK(cudaStreamSynchronize(streamEdge));
        CUDA_CHECK(cudaStreamSynchronize(streamInner));
    }
    if (active) CUDA_CHECK(cudaGetLastError());
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

    int exitCode = 0;
    if (printResults || validate) {
        // Bring the final grid back to rank 0
        if (active) {
            CUDA_CHECK(cudaMemcpy(hostGrid.data(), dGrid[iterations % 2], localPlanes * plane * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
        }
        std::vector<Real> finalGrid(rank == 0 ? gridSize : 0);
        gatherGrid(hostGrid, finalGrid, plane, zStart, nzLocal, rank, numRanks);

        if (rank == 0) {
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
        CUDA_CHECK(cudaStreamDestroy(streamEdge));
        CUDA_CHECK(cudaStreamDestroy(streamInner));
        for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaFreeHost(hHalo[i]));
        CUDA_CHECK(cudaFree(dGrid[0]));
        CUDA_CHECK(cudaFree(dGrid[1]));
    }

    MPI_Finalize();
    return exitCode;
}
