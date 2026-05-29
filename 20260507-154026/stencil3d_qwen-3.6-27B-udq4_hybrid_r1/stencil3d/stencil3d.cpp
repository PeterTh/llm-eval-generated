#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// ──────────────────────────────────────────────────────────────
// 3D index calculation
// ──────────────────────────────────────────────────────────────
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ──────────────────────────────────────────────────────────────
// CUDA kernel: 7-point stencil on interior points
// ──────────────────────────────────────────────────────────────
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x >= 1 && x < nx - 1 && y >= 1 && y < ny - 1 && z >= 1 && z < nz - 1) {
        const size_t idx = z * (nx * ny) + y * nx + x;
        const Real center = input[idx];
        const Real left   = input[idx - 1];
        const Real right  = input[idx + 1];
        const Real front  = input[idx - nx];
        const Real back   = input[idx + nx];
        const Real bottom = input[idx - nx * ny];
        const Real top    = input[idx + nx * ny];
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

// ──────────────────────────────────────────────────────────────
// CUDA kernel: copy boundary values
// ──────────────────────────────────────────────────────────────
__global__ void copyBoundaryKernel(const Real* __restrict__ input,
                                    Real* __restrict__ output,
                                    const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
        const size_t idx = z * (nx * ny) + y * nx + x;
        output[idx] = input[idx];
    }
}

// ──────────────────────────────────────────────────────────────
// Launch configuration helpers
// ──────────────────────────────────────────────────────────────
static dim3 getBlockDim(const size_t nx, const size_t ny, const size_t nz) {
    // Balance threads: prefer more threads in X (fastest changing)
    // Max 256 threads per block; keep z small for occupancy
    dim3 block;
    block.x = std::min(static_cast<size_t>(32), nx);
    block.y = std::min(static_cast<size_t>(16), ny);
    block.z = std::min(static_cast<size_t>(4), nz);
    // Ensure total <= 1024 (sm_35+)
    while (block.x * block.y * block.z > 1024) {
        if (block.z > 1) block.z /= 2;
        else if (block.y > 1) block.y /= 2;
        else break;
    }
    return block;
}

static dim3 getGridDim(const size_t nx, const size_t ny, const size_t nz,
                       const dim3& block) {
    dim3 grid;
    grid.x = (nx + block.x - 1) / block.x;
    grid.y = (ny + block.y - 1) / block.y;
    grid.z = (nz + block.z - 1) / block.z;
    return grid;
}

// ──────────────────────────────────────────────────────────────
// CPU: initialize grid (OpenMP parallel)
// ──────────────────────────────────────────────────────────────
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t nz) {
#pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = static_cast<Real>((idx % 19) * 1.0);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────
// CPU: stencil iteration (OpenMP parallel) – used for halo/boundary
// ──────────────────────────────────────────────────────────────
void stencilIterationCPU(const std::vector<Real>& input,
                         std::vector<Real>& output,
                         const size_t nx, const size_t ny, const size_t nz) {
    // Interior stencil
#pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 1; z < nz - 1; ++z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const Real center = input[idx];
                const Real left   = input[idx - 1];
                const Real right  = input[idx + 1];
                const Real front  = input[idx - nx];
                const Real back   = input[idx + nx];
                const Real bottom = input[idx - nx * ny];
                const Real top    = input[idx + nx * ny];
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }

    // Copy boundary values
#pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
                    z == 0 || z == nz - 1) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Validation (OpenMP parallel reduction)
// ──────────────────────────────────────────────────────────────
bool validateResult(const std::vector<Real>& grid) {
    // 1. No NaN or Inf values
    bool hasBad = false;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < grid.size(); ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
#pragma omp atomic write
            hasBad = true;
        }
    }
    if (hasBad) {
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

    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

// ──────────────────────────────────────────────────────────────
// Usage
// ──────────────────────────────────────────────────────────────
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

// ──────────────────────────────────────────────────────────────
// main – hybrid MPI + CUDA + OpenMP
// ──────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // Parse args BEFORE MPI_Init so we can show help without MPI
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // ── Initialise MPI ────────────────────────────────────────
    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // ── Domain decomposition along Z ──────────────────────────
    // Each rank owns nz_local interior Z planes plus 2 ghost layers
    size_t halo = 1;
    size_t nz_interior = nz - 2 * halo;  // exclude global boundaries
    size_t nz_local_base = nz_interior / numRanks;
    size_t nz_local_rem  = nz_interior % numRanks;

    // Rank i gets nz_local_base + (i < nz_local_rem ? 1 : 0) interior Z planes
    size_t nz_local_interior = nz_local_base + (static_cast<size_t>(rank) < nz_local_rem ? 1 : 0);
    // Plus 2 ghost layers (top and bottom)
    size_t nz_local = nz_local_interior + 2 * halo;

    // Offset of this rank's first interior Z plane in the global grid (excluding
    // the global bottom boundary at z=0)
    size_t offset = 0;
    for (int r = 0; r < rank; ++r) {
        offset += nz_local_base + (static_cast<size_t>(r) < nz_local_rem ? 1 : 0);
    }
    // Global index of first interior Z plane for this rank
    size_t z_start_global = 1 + offset;  // +1 for bottom global boundary

    // Local arrays (with ghost layers)
    size_t localSize = nx * ny * nz_local;
    std::vector<Real> grid1(localSize);
    std::vector<Real> grid2(localSize);

    // ── Determine neighbor ranks ──────────────────────────────
    int rankBelow = (rank == 0)    ? MPI_PROC_NULL : rank - 1;
    int rankAbove = (rank == numRanks - 1) ? MPI_PROC_NULL : rank + 1;

    // Sizes for halo send/recv: nx * ny * halo
    size_t haloSize = nx * ny * halo;
    std::vector<Real> sendBufTop(haloSize);
    std::vector<Real> sendBufBot(haloSize);
    std::vector<Real> recvBufTop(haloSize);
    std::vector<Real> recvBufBot(haloSize);

    // ── Print configuration on rank 0 ─────────────────────────
    if (rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+CUDA+OpenMP)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d\n", numRanks);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ── Initialise GPU ────────────────────────────────────────
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        if (rank == 0) printf("No CUDA device found – falling back to CPU (OpenMP)\n");
    }
    // Select GPU: one GPU per rank, round-robin
    int gpuId = rank % std::max(deviceCount, 1);
    cudaSetDevice(gpuId);

    // Allocate device memory (double-buffered)
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    bool useGPU = (deviceCount > 0);
    if (useGPU) {
        cudaMalloc(&d_grid1, localSize * sizeof(Real));
        cudaMalloc(&d_grid2, localSize * sizeof(Real));
    }

    // ── Initialize grid ───────────────────────────────────────
    if (rank == 0) printf("Initializing grid...\n");

    // Fill local grid: map local indices to global indices
    for (size_t lz = 0; lz < nz_local; ++lz) {
        size_t gz = z_start_global - halo + lz;  // global Z
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t lidx = lz * (nx * ny) + y * nx + x;
                size_t gidx = gz * (nx * ny) + y * nx + x;
                grid1[lidx] = static_cast<Real>((gidx % 19) * 1.0);
                grid2[lidx] = static_cast<Real>((gidx % 19) * 1.0);
            }
        }
    }

    // Copy to GPU
    if (useGPU) {
        cudaMemcpy(d_grid1, grid1.data(), localSize * sizeof(Real),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_grid2, grid2.data(), localSize * sizeof(Real),
                   cudaMemcpyHostToDevice);
    }

    // ── Timing ────────────────────────────────────────────────
    if (rank == 0) printf("Running stencil computation...\n");

    auto start = std::chrono::high_resolution_clock::now();

    dim3 block = getBlockDim(nx, ny, nz_local);
    dim3 grid  = getGridDim(nx, ny, nz_local, block);

    // Streams for overlapping halo exchange with computation
    cudaStream_t streamTop = 0, streamBot = 0;
    if (useGPU) {
        cudaStreamCreate(&streamTop);
        cudaStreamCreate(&streamBot);
    }

    for (int iter = 0; iter < iterations; ++iter) {
        // ── Stencil on GPU ────────────────────────────────────
        if (useGPU) {
            const Real* d_in  = (iter % 2 == 0) ? d_grid1 : d_grid2;
            Real*       d_out = (iter % 2 == 0) ? d_grid2 : d_grid1;

            // Launch stencil kernel (interior only)
            stencilKernel<<<grid, block, 0, 0>>>(d_in, d_out, nx, ny, nz_local);

            // Launch boundary copy kernel
            copyBoundaryKernel<<<grid, block, 0, 0>>>(d_in, d_out, nx, ny, nz_local);

            cudaDeviceSynchronize();
        } else {
            // CPU fallback with OpenMP
            const std::vector<Real>* pIn  = (iter % 2 == 0) ? &grid1 : &grid2;
            std::vector<Real>*       pOut = (iter % 2 == 0) ? &grid2 : &grid1;
            stencilIterationCPU(*pIn, *pOut, nx, ny, nz_local);
        }

        // ── Halo exchange (non-blocking) ──────────────────────
        std::vector<Real>*       pOut = (iter % 2 == 0) ? &grid2 : &grid1;

        // Extract top halo (z = nz_local - 1 .. nz_local - halo)
        for (size_t i = 0; i < haloSize; ++i) {
            sendBufTop[i] = (*pOut)[(nz_local - halo) * (nx * ny) + i];
        }
        // Extract bottom halo (z = 0 .. halo - 1)
        for (size_t i = 0; i < haloSize; ++i) {
            sendBufBot[i] = (*pOut)[i];
        }

        MPI_Request reqs[4];
        // Send top halo to rank above, receive from rank above
        MPI_Isend(sendBufTop.data(), haloSize, MPI_DOUBLE, rankAbove, 0,
                  MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recvBufTop.data(), haloSize, MPI_DOUBLE, rankAbove, 0,
                  MPI_COMM_WORLD, &reqs[1]);
        // Send bottom halo to rank below, receive from rank below
        MPI_Isend(sendBufBot.data(), haloSize, MPI_DOUBLE, rankBelow, 0,
                  MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(recvBufBot.data(), haloSize, MPI_DOUBLE, rankBelow, 0,
                  MPI_COMM_WORLD, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // Insert received halos into output (skip MPI_PROC_NULL to preserve global boundaries)
        if (rankAbove != MPI_PROC_NULL) {
            for (size_t i = 0; i < haloSize; ++i) {
                (*pOut)[(nz_local - halo) * (nx * ny) + i] = recvBufTop[i];
            }
        }
        if (rankBelow != MPI_PROC_NULL) {
            for (size_t i = 0; i < haloSize; ++i) {
                (*pOut)[i] = recvBufBot[i];
            }
        }

        // Copy updated halos to GPU (only if we have a real neighbor)
        if (useGPU) {
            Real* d_out = (iter % 2 == 0) ? d_grid2 : d_grid1;
            if (rankAbove != MPI_PROC_NULL) {
                cudaMemcpyAsync(d_out + (nz_local - halo) * (nx * ny),
                                recvBufTop.data(),
                                haloSize * sizeof(Real),
                                cudaMemcpyHostToDevice, streamTop);
            }
            if (rankBelow != MPI_PROC_NULL) {
                cudaMemcpyAsync(d_out,
                                recvBufBot.data(),
                                haloSize * sizeof(Real),
                                cudaMemcpyHostToDevice, streamBot);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (useGPU) {
        cudaStreamDestroy(streamTop);
        cudaStreamDestroy(streamBot);
    }

    // ── Gather final results on rank 0 ────────────────────────
    // Determine which local grid holds the final result
    const std::vector<Real>& localFinal = (iterations % 2 == 0) ? grid2 : grid1;

    // Gather: each rank sends its local data to rank 0
    // The gathered buffer must be large enough for ALL local data (including ghost layers)
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks);
    size_t totalGathered = 0;
    for (int r = 0; r < numRanks; ++r) {
        size_t nz_loc_int = nz_local_base + (static_cast<size_t>(r) < nz_local_rem ? 1 : 0);
        size_t nz_loc     = nz_loc_int + 2 * halo;
        recvCounts[r] = static_cast<int>(nx * ny * nz_loc);
        displs[r]     = static_cast<int>(totalGathered);
        totalGathered += nx * ny * nz_loc;
    }

    std::vector<Real> gathered;
    if (rank == 0) {
        gathered.resize(totalGathered);
    }

    std::vector<Real> sendVec(localSize);
    for (size_t i = 0; i < localSize; ++i) {
        sendVec[i] = localFinal[i];
    }

    MPI_Gatherv(sendVec.data(), static_cast<int>(localSize), MPI_DOUBLE,
                gathered.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Reconstruct global grid from gathered data on rank 0.
    // Each rank's data includes 2 ghost layers (top and bottom) that overlap
    // with neighbors. We extract only the interior planes plus global boundaries.
    std::vector<Real> fullGrid;
    if (rank == 0) {
        fullGrid.resize(nx * ny * nz);

        // Bottom global boundary (z=0) comes from rank 0's bottom ghost layer
        std::memcpy(fullGrid.data(),
                    gathered.data(),
                    nx * ny * sizeof(Real));

        // Top global boundary (z=nz-1) comes from last rank's top ghost layer
        {
            int lastRank = numRanks - 1;
            size_t lastRankDataStart = 0;
            for (int r = 0; r < lastRank; ++r) {
                size_t nz_i = nz_local_base + (static_cast<size_t>(r) < nz_local_rem ? 1 : 0);
                lastRankDataStart += (nz_i + 2 * halo) * nx * ny;
            }
            size_t lastRankNzInt = nz_local_base + (static_cast<size_t>(lastRank) < nz_local_rem ? 1 : 0);
            size_t lastRankNzLocal = lastRankNzInt + 2 * halo;
            std::memcpy(fullGrid.data() + (nz - 1) * (nx * ny),
                        gathered.data() + lastRankDataStart + (lastRankNzLocal - halo) * (nx * ny),
                        nx * ny * sizeof(Real));
        }

        // Interior planes: for each rank, copy nz_loc_int planes starting after
        // the bottom ghost layer
        size_t globalZ = 1;
        for (int r = 0; r < numRanks; ++r) {
            size_t nz_loc_int = nz_local_base + (static_cast<size_t>(r) < nz_local_rem ? 1 : 0);
            size_t rankDataStart = 0;
            for (int rr = 0; rr < r; ++rr) {
                size_t nz_i = nz_local_base + (static_cast<size_t>(rr) < nz_local_rem ? 1 : 0);
                rankDataStart += (nz_i + 2 * halo) * nx * ny;
            }
            size_t interiorStart = rankDataStart + halo * (nx * ny);
            for (size_t lz = 0; lz < nz_loc_int; ++lz) {
                std::memcpy(fullGrid.data() + globalZ * (nx * ny),
                            gathered.data() + interiorStart + lz * (nx * ny),
                            nx * ny * sizeof(Real));
                ++globalZ;
            }
        }
    }

    // ── Performance report ────────────────────────────────────
    double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double secs = duration.count() / 1000.0;
    double mcups = (secs > 0) ? cellUpdates / secs / 1e6 : 0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ── Results output ────────────────────────────────────────
    if (printResults && rank == 0) {
        print_results(fullGrid, "Grid");
    }

    // ── Validation ────────────────────────────────────────────
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(fullGrid);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // ── Cleanup ───────────────────────────────────────────────
    if (useGPU) {
        cudaFree(d_grid1);
        cudaFree(d_grid2);
    }

    MPI_Finalize();
    return 0;
}
