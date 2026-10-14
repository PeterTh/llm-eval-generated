#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// Thread block shape for the 2D (x, y) mapping; each thread sweeps the z-range.
constexpr unsigned int kBlockX = 32;
constexpr unsigned int kBlockY = 8;

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(err_));                                      \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
    } while (0)

// Layout of one rank's z-slab. Local plane 0 and plane nzl+1 are ghost planes
// holding the neighbours' boundary planes; planes 1..nzl are owned.
struct Domain {
    size_t nx, ny, nz;  // global grid size
    size_t plane;       // nx * ny, elements per xy-plane
    size_t zStart;      // global z index of owned plane 1
    size_t nzl;         // number of owned planes
    int up, down;       // neighbour ranks (higher / lower z) or MPI_PROC_NULL
};

// Pinned host buffers used to stage halo planes between device and MPI.
struct Staging {
    Real *sendLo, *sendHi, *recvLo, *recvHi;
};

// Initialize owned planes with the same values as the serial version, using global indices.
__global__ void initKernel(Real* __restrict__ grid, const size_t nx, const size_t ny,
                           const size_t zStart, const size_t nzl) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t plane = nx * ny;
    const size_t base = y * nx + x;
    for (size_t k = 1; k <= nzl; ++k) {
        const size_t idx = (zStart + k - 1) * plane + base;
        grid[k * plane + base] = static_cast<Real>(idx % 19);
    }
}

// 7-point stencil over local planes [kBegin, kEnd). Each thread owns one (x, y)
// column and sweeps z, keeping the planes below and above in registers.
// Boundary cells and global z-boundary planes are copied, as in the serial version.
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
                              const size_t nx, const size_t ny, const size_t nz,
                              const size_t zStart, const size_t kBegin, const size_t kEnd) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t plane = nx * ny;
    const size_t base = y * nx + x;

    if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1) {
        for (size_t k = kBegin; k < kEnd; ++k) {
            out[k * plane + base] = in[k * plane + base];
        }
        return;
    }

    Real below = in[(kBegin - 1) * plane + base];
    Real cur = in[kBegin * plane + base];
    for (size_t k = kBegin; k < kEnd; ++k) {
        const size_t i = k * plane + base;
        const Real above = in[i + plane];
        const size_t gz = zStart + k - 1;

        // Same summation order as the serial version, so results are bit-identical.
        if (gz == 0 || gz == nz - 1) {
            out[i] = cur;
        } else {
            out[i] = (cur + in[i - 1] + in[i + 1] + in[i - nx] + in[i + nx] + below + above) / 7.0;
        }

        below = cur;
        cur = above;
    }
}

void launchInit(const Domain& d, Real* grid, cudaStream_t s) {
    const dim3 block(kBlockX, kBlockY);
    const dim3 blocks((d.nx + kBlockX - 1) / kBlockX, (d.ny + kBlockY - 1) / kBlockY);
    initKernel<<<blocks, block, 0, s>>>(grid, d.nx, d.ny, d.zStart, d.nzl);
    CUDA_CHECK(cudaGetLastError());
}

void launchStencil(const Domain& d, const Real* in, Real* out, size_t kBegin, size_t kEnd, cudaStream_t s) {
    const dim3 block(kBlockX, kBlockY);
    const dim3 blocks((d.nx + kBlockX - 1) / kBlockX, (d.ny + kBlockY - 1) / kBlockY);
    stencilKernel<<<blocks, block, 0, s>>>(in, out, d.nx, d.ny, d.nz, d.zStart, kBegin, kEnd);
    CUDA_CHECK(cudaGetLastError());
}

// Exchange boundary planes with the neighbouring slabs and fill this rank's ghost planes.
// Must be called after the owned boundary planes of `buf` have been written on stream `s`.
void exchangeHalos(const Domain& d, const Staging& st, Real* buf, cudaStream_t s) {
    const size_t bytes = d.plane * sizeof(Real);
    const int count = static_cast<int>(d.plane);

    if (d.down != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(st.sendLo, buf + d.plane, bytes, cudaMemcpyDeviceToHost, s));
    }
    if (d.up != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(st.sendHi, buf + d.nzl * d.plane, bytes, cudaMemcpyDeviceToHost, s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));

    MPI_Sendrecv(st.sendHi, count, MPI_DOUBLE, d.up, 0,
                 st.recvLo, count, MPI_DOUBLE, d.down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(st.sendLo, count, MPI_DOUBLE, d.down, 1,
                 st.recvHi, count, MPI_DOUBLE, d.up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (d.down != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(buf, st.recvLo, bytes, cudaMemcpyHostToDevice, s));
    }
    if (d.up != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(buf + (d.nzl + 1) * d.plane, st.recvHi, bytes, cudaMemcpyHostToDevice, s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// Number of z-planes owned by a rank, and the global index of its first plane.
size_t planesOf(const size_t nz, const int nranks, const int rank) {
    return nz / nranks + (static_cast<size_t>(rank) < nz % nranks ? 1 : 0);
}

size_t startOf(const size_t nz, const int nranks, const int rank) {
    const size_t r = static_cast<size_t>(rank);
    return r * (nz / nranks) + std::min(r, nz % nranks);
}

// Sanity checks as in the serial version, computed over the owned planes of every rank.
// Returns the same result on all ranks; only rank 0 prints.
bool validateResult(const std::vector<Real>& owned, const int rank) {
    const Real inf = std::numeric_limits<Real>::infinity();
    long long nonFinite = 0;
    Real minVal = inf;
    Real maxVal = -inf;
    const size_t n = owned.size();

#pragma omp parallel
    {
        long long tNonFinite = 0;
        Real tMin = inf;
        Real tMax = -inf;
#pragma omp for nowait
        for (size_t i = 0; i < n; ++i) {
            const Real val = owned[i];
            if (!std::isfinite(val)) ++tNonFinite;
            tMin = std::min(tMin, val);
            tMax = std::max(tMax, val);
        }
#pragma omp critical
        {
            nonFinite += tNonFinite;
            minVal = std::min(minVal, tMin);
            maxVal = std::max(maxVal, tMax);
        }
    }

    long long globalNonFinite = 0;
    Real globalMin = 0.0, globalMax = 0.0;
    MPI_Allreduce(&nonFinite, &globalNonFinite, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // 1. No NaN or Inf values
    if (globalNonFinite > 0) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    if (rank == 0) printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);

    // 2. Values should be reasonable (bounded)
    if (globalMax > 1e6 || globalMin < -1e6) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }

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

int run(int argc, char** argv) {
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool root = rank == 0;

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
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
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

    if (nx == 0 || ny == 0 || nz == 0) {
        if (root) fprintf(stderr, "Error: grid dimensions must be positive\n");
        return 1;
    }
    if (nx * ny > static_cast<size_t>(INT_MAX)) {
        if (root) fprintf(stderr, "Error: nx * ny exceeds the MPI message count limit\n");
        return 1;
    }

    // One GPU per rank: ranks on the same node are assigned round-robin to the local devices.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Error: no CUDA device available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // 1D slab decomposition along z: planes are contiguous in memory, so halos are whole planes.
    // If there are fewer planes than ranks, the surplus ranks own no planes and take no part
    // in the exchange.
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(nranks)));
    Domain d{};
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;
    d.plane = nx * ny;
    d.zStart = startOf(nz, nranks, rank);
    d.nzl = planesOf(nz, nranks, rank);
    d.up = (rank + 1 < activeRanks) ? rank + 1 : MPI_PROC_NULL;
    d.down = (rank > 0 && rank < activeRanks) ? rank - 1 : MPI_PROC_NULL;

    const size_t gridSize = nx * ny * nz;
    const size_t localElems = (d.nzl + 2) * d.plane;

    // Allocate grids (double buffering) on the device
    Real* dGrid[2];
    for (auto& g : dGrid) {
        CUDA_CHECK(cudaMalloc(&g, localElems * sizeof(Real)));
        CUDA_CHECK(cudaMemset(g, 0, localElems * sizeof(Real)));
    }

    Staging st{};
    CUDA_CHECK(cudaMallocHost(&st.sendLo, d.plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&st.sendHi, d.plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&st.recvLo, d.plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&st.recvHi, d.plane * sizeof(Real)));

    // Edge planes run on one stream so their halos can be exchanged while the interior
    // planes run on the other stream.
    cudaStream_t sEdge, sInterior;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sEdge, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sInterior, cudaStreamNonBlocking));

    // Initialize
    if (root) printf("Initializing grid...\n");
    launchInit(d, dGrid[0], sEdge);
    CUDA_CHECK(cudaStreamSynchronize(sEdge));
    exchangeHalos(d, st, dGrid[0], sEdge);

    // Run stencil iterations
    if (root) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        const Real* in = dGrid[iter % 2];
        Real* out = dGrid[(iter + 1) % 2];

        // Boundary planes first, so their halos can be sent while the interior is computed.
        if (d.nzl > 0) launchStencil(d, in, out, 1, 2, sEdge);
        if (d.nzl > 1) launchStencil(d, in, out, d.nzl, d.nzl + 1, sEdge);
        if (d.nzl > 2) launchStencil(d, in, out, 2, d.nzl, sInterior);

        exchangeHalos(d, st, out, sEdge);
        CUDA_CHECK(cudaStreamSynchronize(sInterior));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // The slowest rank determines the run time.
    const double localSec = std::chrono::duration<double>(end - start).count();
    double maxSec = 0.0;
    MPI_Allreduce(&localSec, &maxSec, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const long durationMs = static_cast<long>(maxSec * 1000.0);

    if (root) printf("Computation time: %ld ms\n", durationMs);

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;  // Million cell updates per second
    if (root) printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Final grid, matching the double-buffer parity of the serial version
    const Real* finalDev = (iterations % 2 == 0) ? dGrid[0] : dGrid[1];
    std::vector<Real> owned;
    if (printResults || validate) {
        owned.resize(d.nzl * d.plane);
        if (d.nzl > 0) {
            CUDA_CHECK(cudaMemcpy(owned.data(), finalDev + d.plane, owned.size() * sizeof(Real), cudaMemcpyDeviceToHost));
        }
    }

    // Print results for external validation: gather the slabs in global z order on rank 0
    if (printResults) {
        std::vector<Real> full;
        if (root) full.resize(gridSize);

        MPI_Datatype planeType;
        MPI_Type_contiguous(static_cast<int>(d.plane), MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);

        std::vector<int> counts(nranks), displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            counts[r] = static_cast<int>(planesOf(nz, nranks, r));
            displs[r] = static_cast<int>(startOf(nz, nranks, r));
        }
        MPI_Gatherv(owned.data(), static_cast<int>(d.nzl), planeType,
                    full.data(), counts.data(), displs.data(), planeType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&planeType);

        if (root) print_results(full, "Grid");
    }

    // Validation
    int rc = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        const bool valid = validateResult(owned, rank);

        if (valid) {
            if (root) printf("Validation: PASSED\n");
        } else {
            if (root) printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    CUDA_CHECK(cudaStreamDestroy(sEdge));
    CUDA_CHECK(cudaStreamDestroy(sInterior));
    CUDA_CHECK(cudaFreeHost(st.sendLo));
    CUDA_CHECK(cudaFreeHost(st.sendHi));
    CUDA_CHECK(cudaFreeHost(st.recvLo));
    CUDA_CHECK(cudaFreeHost(st.recvHi));
    CUDA_CHECK(cudaFree(dGrid[0]));
    CUDA_CHECK(cudaFree(dGrid[1]));

    return rc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = run(argc, argv);
    MPI_Finalize();
    return rc;
}
