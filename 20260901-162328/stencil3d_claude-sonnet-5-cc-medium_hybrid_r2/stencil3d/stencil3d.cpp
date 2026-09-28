#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,               \
                    cudaGetErrorString(err__));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

// 3D index calculation (host + device)
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize the portion of the grid owned by this rank into a compact
// (ghost-free) host buffer of size nx*ny*local_nz. Values match the original
// single-rank formula exactly because they only depend on the *global* index.
void initializeGridLocal(std::vector<Real>& localGrid, const size_t nx, const size_t ny,
                          const size_t local_nz, const size_t z_start) {
    #pragma omp parallel for schedule(static)
    for (size_t lz = 0; lz < local_nz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalZ = z_start + lz;
                const size_t globalIdx = idx3(x, y, globalZ, nx, ny);
                const size_t localIdx = idx3(x, y, lz, nx, ny);
                localGrid[localIdx] = (globalIdx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation for a single z-plane (identified by local
// buffer z-index `bz`). Handles global boundary copy-through exactly like the
// original single-process implementation.
__global__ void stencilPlaneKernel(const Real* __restrict__ input, Real* __restrict__ output,
                                    const size_t nx, const size_t ny, const size_t bz,
                                    const size_t globalZ, const size_t nzGlobal) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t idx = idx3(x, y, bz, nx, ny);

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || globalZ == 0 || globalZ == nzGlobal - 1) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left = input[idx - 1];
    const Real right = input[idx + 1];
    const Real front = input[idx - nx];
    const Real back = input[idx + nx];
    const Real bottom = input[idx - nx * ny];
    const Real top = input[idx + nx * ny];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

// 7-point stencil computation for a contiguous range of interior z-planes
// that are guaranteed not to touch a global z-boundary (only x/y boundaries
// need to be checked).
__global__ void stencilInteriorKernel(const Real* __restrict__ input, Real* __restrict__ output,
                                       const size_t nx, const size_t ny,
                                       const size_t bzFirst, const size_t numPlanes) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t zOff = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || zOff >= numPlanes) return;

    const size_t bz = bzFirst + zOff;
    const size_t idx = idx3(x, y, bz, nx, ny);

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left = input[idx - 1];
    const Real right = input[idx + 1];
    const Real front = input[idx - nx];
    const Real back = input[idx + nx];
    const Real bottom = input[idx - nx * ny];
    const Real top = input[idx + nx * ny];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    bool badFound = false;
    #pragma omp parallel for reduction(|| : badFound) schedule(static)
    for (size_t i = 0; i < grid.size(); ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            badFound = true;
        }
    }
    if (badFound) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) schedule(static)
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Bind this rank to a GPU: round-robin over the GPUs visible on its node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0, nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int myDevice = nodeRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(myDevice));

    // Give each rank a fair share of the node's CPU cores for its OpenMP
    // team (avoids oversubscribing when several ranks share a node).
    const int hwThreads = std::max(1u, std::thread::hardware_concurrency());
    omp_set_num_threads(std::max(1, hwThreads / nodeSize));

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (argv is identical across ranks under mpirun)
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
        printf("3D Stencil Benchmark (MPI + OpenMP + CUDA hybrid)\n");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs visible/node: %d\n",
               numRanks, omp_get_max_threads(), deviceCount);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // 1D domain decomposition along Z: block distribution with remainder
    // spread across the first ranks.
    const size_t base = nz / static_cast<size_t>(numRanks);
    const size_t rem = nz % static_cast<size_t>(numRanks);
    const size_t local_nz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t z_start = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);

    const int downRank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int upRank = (rank < numRanks - 1) ? rank + 1 : MPI_PROC_NULL;

    const size_t planeElems = nx * ny;
    const size_t localBufElems = planeElems * (local_nz + 2);

    // Host staging buffer used for initialization and for gathering the
    // final result on rank 0.
    std::vector<Real> hostLocal(local_nz * planeElems);
    if (rank == 0) printf("Initializing grid...\n");
    initializeGridLocal(hostLocal, nx, ny, local_nz, z_start);

    // Device double buffers (with 2 ghost planes each).
    Real* d_bufA = nullptr;
    Real* d_bufB = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bufA, localBufElems * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_bufB, localBufElems * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_bufA, 0, localBufElems * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_bufB, 0, localBufElems * sizeof(Real)));

    if (local_nz > 0) {
        CUDA_CHECK(cudaMemcpy(d_bufA + planeElems, hostLocal.data(),
                               local_nz * planeElems * sizeof(Real), cudaMemcpyHostToDevice));
    }

    // Pinned host staging buffers for halo exchange (fast D2H/H2D transfers).
    Real* h_sendDown = nullptr; // this rank's bottom-owned plane -> down neighbor
    Real* h_sendUp = nullptr;   // this rank's top-owned plane -> up neighbor
    Real* h_recvBelow = nullptr; // ghost plane below, from down neighbor
    Real* h_recvAbove = nullptr; // ghost plane above, from up neighbor
    CUDA_CHECK(cudaHostAlloc(&h_sendDown, planeElems * sizeof(Real), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_sendUp, planeElems * sizeof(Real), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recvBelow, planeElems * sizeof(Real), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recvAbove, planeElems * sizeof(Real), cudaHostAllocDefault));

    cudaStream_t computeStream, haloStream;
    CUDA_CHECK(cudaStreamCreate(&computeStream));
    CUDA_CHECK(cudaStreamCreate(&haloStream));

    dim3 block2D(16, 16, 1);
    dim3 grid2D((static_cast<unsigned>(nx) + block2D.x - 1) / block2D.x,
                (static_cast<unsigned>(ny) + block2D.y - 1) / block2D.y, 1);

    auto haloExchange = [&](Real* d_in) {
        if (local_nz == 0) {
            MPI_Sendrecv(nullptr, 0, MPI_DOUBLE, MPI_PROC_NULL, 0, nullptr, 0, MPI_DOUBLE, MPI_PROC_NULL, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            return;
        }
        // Stage owned boundary planes from device to host.
        CUDA_CHECK(cudaMemcpyAsync(h_sendDown, d_in + planeElems, planeElems * sizeof(Real),
                                    cudaMemcpyDeviceToHost, haloStream));
        CUDA_CHECK(cudaMemcpyAsync(h_sendUp, d_in + local_nz * planeElems, planeElems * sizeof(Real),
                                    cudaMemcpyDeviceToHost, haloStream));
        CUDA_CHECK(cudaStreamSynchronize(haloStream));

        MPI_Sendrecv(h_sendDown, static_cast<int>(planeElems), MPI_DOUBLE, downRank, 0,
                     h_recvAbove, static_cast<int>(planeElems), MPI_DOUBLE, upRank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(h_sendUp, static_cast<int>(planeElems), MPI_DOUBLE, upRank, 1,
                     h_recvBelow, static_cast<int>(planeElems), MPI_DOUBLE, downRank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (downRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(d_in, h_recvBelow, planeElems * sizeof(Real),
                                        cudaMemcpyHostToDevice, haloStream));
        }
        if (upRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(d_in + (local_nz + 1) * planeElems, h_recvAbove,
                                        planeElems * sizeof(Real), cudaMemcpyHostToDevice, haloStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(haloStream));
    };

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    Real* d_in = d_bufA;
    Real* d_out = d_bufB;

    for (int iter = 0; iter < iterations; ++iter) {
        // Interior planes (buffer z-index in [2, local_nz-1], i.e. owned
        // planes that are not adjacent to a ghost plane) never depend on
        // halo data, so launch them immediately and overlap the halo
        // exchange (host-staged D2H/MPI/H2D) with this compute on the GPU.
        if (local_nz >= 3) {
            const size_t numInterior = local_nz - 2;
            dim3 gridInterior(grid2D.x, grid2D.y, static_cast<unsigned>(numInterior));
            stencilInteriorKernel<<<gridInterior, block2D, 0, computeStream>>>(
                d_in, d_out, nx, ny, /*bzFirst=*/2, numInterior);
        }

        haloExchange(d_in);

        // Boundary-adjacent owned planes need the freshly exchanged ghosts.
        if (local_nz >= 1) {
            const size_t globalZ0 = z_start + 0;
            stencilPlaneKernel<<<grid2D, block2D, 0, haloStream>>>(
                d_in, d_out, nx, ny, /*bz=*/1, globalZ0, nz);
        }
        if (local_nz >= 2) {
            const size_t globalZLast = z_start + (local_nz - 1);
            stencilPlaneKernel<<<grid2D, block2D, 0, haloStream>>>(
                d_in, d_out, nx, ny, /*bz=*/local_nz, globalZLast, nz);
        }

        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        CUDA_CHECK(cudaStreamSynchronize(haloStream));

        std::swap(d_in, d_out);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate performance metrics
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy the final owned data back to the host and gather it on rank 0.
    if (local_nz > 0) {
        CUDA_CHECK(cudaMemcpy(hostLocal.data(), d_in + planeElems,
                               local_nz * planeElems * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    std::vector<Real> fullGrid;
    std::vector<int> recvCounts, displs;
    if (rank == 0) {
        fullGrid.resize(nx * ny * nz);
        recvCounts.resize(numRanks);
        displs.resize(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            const size_t rBase = nz / static_cast<size_t>(numRanks);
            const size_t rRem = nz % static_cast<size_t>(numRanks);
            const size_t rLocalNz = rBase + (static_cast<size_t>(r) < rRem ? 1 : 0);
            const size_t rZStart = static_cast<size_t>(r) * rBase + std::min(static_cast<size_t>(r), rRem);
            recvCounts[r] = static_cast<int>(rLocalNz * planeElems);
            displs[r] = static_cast<int>(rZStart * planeElems);
        }
    }
    MPI_Gatherv(hostLocal.data(), static_cast<int>(local_nz * planeElems), MPI_DOUBLE,
                rank == 0 ? fullGrid.data() : nullptr,
                rank == 0 ? recvCounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print results for external validation
    if (rank == 0) {
        if (printResults) {
            print_results(fullGrid, "Grid");
        }
    }

    int validationResult = 0; // 0 = pass/not-run, 1 = fail
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(fullGrid, nx, ny, nz);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        validationResult = valid ? 0 : 1;
    }

    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(haloStream));
    CUDA_CHECK(cudaFreeHost(h_sendDown));
    CUDA_CHECK(cudaFreeHost(h_sendUp));
    CUDA_CHECK(cudaFreeHost(h_recvBelow));
    CUDA_CHECK(cudaFreeHost(h_recvAbove));
    CUDA_CHECK(cudaFree(d_bufA));
    CUDA_CHECK(cudaFree(d_bufB));

    MPI_Bcast(&validationResult, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return validate ? validationResult : 0;
}
