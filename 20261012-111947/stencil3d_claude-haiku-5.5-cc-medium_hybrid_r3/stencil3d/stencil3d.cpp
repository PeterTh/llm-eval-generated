#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Global grid dimensions
struct Grid3 {
    size_t nx, ny, nz;
};

// Contiguous z-slab owned by one MPI rank
struct Slab {
    size_t firstPlane;  // global z index of the first owned plane
    size_t planes;      // number of owned planes
};

// Ranks beyond nz own nothing and only take part in the collectives
Slab slabFor(const int rank, const int nRanks, const size_t nz) {
    const size_t active = std::min<size_t>(static_cast<size_t>(nRanks), nz);
    const size_t r = static_cast<size_t>(rank);
    if (r >= active) return {0, 0};
    const size_t base = nz / active;
    const size_t rem = nz % active;
    return {r * base + std::min(r, rem), base + (r < rem ? 1 : 0)};
}

// Host-side initialization of the owned planes (OpenMP parallel)
void initializeSlab(std::vector<Real>& slab, const size_t firstPlane, const size_t nPlanes, const size_t nx, const size_t ny) {
    const size_t planeSize = nx * ny;
    const size_t base = firstPlane * planeSize;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nPlanes * planeSize; ++i) {
        slab[i] = ((base + i) % 19) * 1.0;
    }
}

// 7-point stencil on one or more local planes. Local plane 0 and nPlanes+1 are halo (ghost) planes.
// Global boundary cells are copied, as in the original code.
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out, const Grid3 g,
                              const size_t globalZ0, const size_t firstLocal, const size_t nPlanes) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= g.nx) return;
    for (size_t p = blockIdx.z; p < nPlanes; p += gridDim.z) {
        const size_t z = firstLocal + p;
        const size_t gz = globalZ0 + z - 1;  // global z coordinate of local plane z
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < g.ny; y += gridDim.y * blockDim.y) {
            const size_t idx = idx3(x, y, z, g.nx, g.ny);
            if (x == 0 || x == g.nx - 1 || y == 0 || y == g.ny - 1 || gz == 0 || gz == g.nz - 1) {
                out[idx] = in[idx];
                continue;
            }

            const Real center = in[idx];
            const Real left = in[idx3(x-1, y, z, g.nx, g.ny)];
            const Real right = in[idx3(x+1, y, z, g.nx, g.ny)];
            const Real front = in[idx3(x, y-1, z, g.nx, g.ny)];
            const Real back = in[idx3(x, y+1, z, g.nx, g.ny)];
            const Real bottom = in[idx3(x, y, z-1, g.nx, g.ny)];
            const Real top = in[idx3(x, y, z+1, g.nx, g.ny)];

            // Simple averaging stencil (same operation order as the CPU original)
            out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
        }
    }
}

void launchStencil(cudaStream_t stream, const Real* in, Real* out, const Grid3& g,
                   const size_t globalZ0, const size_t firstLocal, const size_t nPlanes) {
    if (nPlanes == 0 || g.nx == 0 || g.ny == 0) return;
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned int>((g.nx + block.x - 1) / block.x),
                    static_cast<unsigned int>(std::min<size_t>((g.ny + block.y - 1) / block.y, 65535)),
                    static_cast<unsigned int>(std::min<size_t>(nPlanes, 65535)));
    stencilKernel<<<grid, block, 0, stream>>>(in, out, g, globalZ0, firstLocal, nPlanes);
    CUDA_CHECK(cudaGetLastError());
}

// Pinned host staging buffers for the halo exchange (one plane each)
struct Halo {
    Real* sendDown;
    Real* sendUp;
    Real* recvDown;
    Real* recvUp;
};

// One sweep over this rank's slab. Interior planes are computed while the
// boundary planes are exchanged with the neighbours; the boundary planes follow.
void sweep(Real* in, Real* out, const Grid3& g, const Slab& s, const int down, const int up,
           const Halo& h, cudaStream_t sInt, cudaStream_t sBnd) {
    const size_t plane = g.nx * g.ny;
    const size_t nl = s.planes;
    const size_t bytes = plane * sizeof(Real);
    const int count = static_cast<int>(plane);

    // Planes 2..nl-1 only read owned data
    if (nl > 2) launchStencil(sInt, in, out, g, s.firstPlane, 2, nl - 2);

    // Stage owned boundary planes 1 and nl to the host
    if (down != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(h.sendDown, in + plane, bytes, cudaMemcpyDeviceToHost, sBnd));
    if (up != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(h.sendUp, in + nl * plane, bytes, cudaMemcpyDeviceToHost, sBnd));
    CUDA_CHECK(cudaStreamSynchronize(sBnd));

    MPI_Sendrecv(h.sendDown, count, MPI_DOUBLE, down, 0, h.recvUp, count, MPI_DOUBLE, up, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h.sendUp, count, MPI_DOUBLE, up, 1, h.recvDown, count, MPI_DOUBLE, down, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Ghost planes go back to the device
    if (up != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(in + (nl + 1) * plane, h.recvUp, bytes, cudaMemcpyHostToDevice, sBnd));
    if (down != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(in, h.recvDown, bytes, cudaMemcpyHostToDevice, sBnd));

    // Boundary planes need the halos
    launchStencil(sBnd, in, out, g, s.firstPlane, 1, 1);
    if (nl > 1) launchStencil(sBnd, in, out, g, s.firstPlane, nl, 1);

    CUDA_CHECK(cudaStreamSynchronize(sInt));
    CUDA_CHECK(cudaStreamSynchronize(sBnd));
}

// Distributed validation over all ranks' owned values (OpenMP scan per rank)
bool validateResult(const Real* values, const size_t count, const bool root) {
    int nonFinite = 0;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    #pragma omp parallel
    {
        int threadNonFinite = 0;
        Real threadMin = std::numeric_limits<Real>::infinity();
        Real threadMax = -std::numeric_limits<Real>::infinity();
        #pragma omp for nowait
        for (size_t i = 0; i < count; ++i) {
            const Real val = values[i];
            if (!std::isfinite(val)) threadNonFinite = 1;
            threadMin = std::min(threadMin, val);
            threadMax = std::max(threadMax, val);
        }
        #pragma omp critical
        {
            nonFinite |= threadNonFinite;
            localMin = std::min(localMin, threadMin);
            localMax = std::max(localMax, threadMax);
        }
    }

    int globalNonFinite = 0;
    Real minVal = 0.0;
    Real maxVal = 0.0;
    MPI_Allreduce(&nonFinite, &globalNonFinite, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // 1. No NaN or Inf values
    if (globalNonFinite) {
        if (root) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    if (root) printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        if (root) printf("Validation failed: values out of expected range\n");
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
    int rank = 0;
    int nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
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

    const Grid3 g{nx, ny, nz};
    const size_t planeSize = nx * ny;
    const size_t gridSize = nx * ny * nz;

    // One GPU per rank (round-robin over the GPUs on the node); OpenMP threads share the node's cores
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int nDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    if (nDevices == 0) {
        fprintf(stderr, "No CUDA devices found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % nDevices));
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const Slab slab = slabFor(rank, nRanks, nz);
    const size_t nl = slab.planes;
    const bool hasLower = nl > 0 && slab.firstPlane > 0;
    const bool hasUpper = nl > 0 && slab.firstPlane + nl < nz;
    const int down = hasLower ? rank - 1 : MPI_PROC_NULL;
    const int up = hasUpper ? rank + 1 : MPI_PROC_NULL;

    // Device double buffers with one ghost plane on each side; host staging for init/results
    std::vector<Real> host(nl * planeSize);
    Real* dev[2] = {nullptr, nullptr};
    Halo halo{nullptr, nullptr, nullptr, nullptr};
    cudaStream_t sInt = nullptr;
    cudaStream_t sBnd = nullptr;
    if (nl > 0) {
        const size_t devBytes = (nl + 2) * planeSize * sizeof(Real);
        for (Real*& d : dev) {
            CUDA_CHECK(cudaMalloc(&d, devBytes));
            CUDA_CHECK(cudaMemset(d, 0, devBytes));
        }
        CUDA_CHECK(cudaMallocHost(&halo.sendDown, planeSize * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&halo.sendUp, planeSize * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&halo.recvDown, planeSize * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&halo.recvUp, planeSize * sizeof(Real)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sInt, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sBnd, cudaStreamNonBlocking));
    }

    // Initialize
    if (root) printf("Initializing grid...\n");
    initializeSlab(host, slab.firstPlane, nl, nx, ny);
    if (nl > 0) {
        CUDA_CHECK(cudaMemcpy(dev[0] + planeSize, host.data(), nl * planeSize * sizeof(Real), cudaMemcpyHostToDevice));
    }

    // Run stencil iterations
    if (root) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    Real* cur = dev[0];
    Real* next = dev[1];
    for (int iter = 0; iter < iterations; ++iter) {
        if (nl > 0) sweep(cur, next, g, slab, down, up, halo, sInt, sBnd);
        std::swap(cur, next);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // The slowest rank determines the time
    long long localMs = duration.count();
    long long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %lld ms\n", maxMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (maxMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // `cur` now holds the final grid, as in the original (grid1 for even iteration counts)
    const bool needHost = printResults || validate;
    if (needHost && nl > 0) {
        CUDA_CHECK(cudaMemcpy(host.data(), cur + planeSize, nl * planeSize * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    // Print results for external validation (gathered in z order on rank 0)
    if (printResults) {
        std::vector<Real> finalGrid;
        std::vector<int> counts;
        std::vector<int> displs;
        if (root) {
            finalGrid.resize(gridSize);
            for (int r = 0; r < nRanks; ++r) {
                const Slab s = slabFor(r, nRanks, nz);
                counts.push_back(static_cast<int>(s.planes * planeSize));
                displs.push_back(static_cast<int>(s.firstPlane * planeSize));
            }
        }
        MPI_Gatherv(host.data(), static_cast<int>(nl * planeSize), MPI_DOUBLE,
                    finalGrid.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (root) print_results(finalGrid, "Grid");
    }

    int status = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        bool valid = validateResult(host.data(), nl * planeSize, root);
        if (root) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        status = valid ? 0 : 1;
    }

    if (nl > 0) {
        CUDA_CHECK(cudaStreamDestroy(sInt));
        CUDA_CHECK(cudaStreamDestroy(sBnd));
        CUDA_CHECK(cudaFreeHost(halo.sendDown));
        CUDA_CHECK(cudaFreeHost(halo.sendUp));
        CUDA_CHECK(cudaFreeHost(halo.recvDown));
        CUDA_CHECK(cudaFreeHost(halo.recvUp));
        CUDA_CHECK(cudaFree(dev[0]));
        CUDA_CHECK(cudaFree(dev[1]));
    }

    MPI_Finalize();
    return status;
}
