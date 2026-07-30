#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// ---------------------------------------------------------------------------
// CUDA error checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(ans) do {                                                 \
    cudaError_t _err = (ans);                                                \
    if (_err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(_err));                \
        MPI_Abort(MPI_COMM_WORLD, 1);                                        \
    }                                                                        \
} while (0)

// ---------------------------------------------------------------------------
// 3-D index helper (host only)
// ---------------------------------------------------------------------------
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                              const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// CUDA kernel: 7-point stencil on interior points
// ---------------------------------------------------------------------------
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx,
                              const size_t ny,
                              const size_t nz_local,
                              const size_t nx_ny) {
    // x, y, z are local 1-based interior indices
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const size_t z = blockIdx.z + 1;          // blockDim.z == 1

    if (x >= nx - 1 || y >= ny - 1 || z > nz_local) return;

    const size_t idx = z * nx_ny + y * nx + x;

    const Real center = input[idx];
    const Real left   = input[idx - 1];
    const Real right  = input[idx + 1];
    const Real front  = input[idx - nx];
    const Real back   = input[idx + nx];
    const Real bottom = input[idx - nx_ny];
    const Real top    = input[idx + nx_ny];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

// ---------------------------------------------------------------------------
// CUDA kernel: copy boundary values (X/Y boundaries and global Z boundaries)
// ---------------------------------------------------------------------------
__global__ void boundaryKernel(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               const size_t nx,
                               const size_t ny,
                               const size_t nz_local,
                               const size_t nx_ny,
                               const bool isBottom,
                               const bool isTop) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z + 1;

    if (x >= nx || y >= ny || z > nz_local) return;

    const size_t idx = z * nx_ny + y * nx + x;

    // X and Y boundaries (all Z)
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
        output[idx] = input[idx];
    }

    // Global Z bottom boundary (rank 0 only)
    if (isBottom && z == 1) {
        output[idx] = input[idx];
    }

    // Global Z top boundary (last rank only)
    if (isTop && z == nz_local) {
        output[idx] = input[idx];
    }
}

// ---------------------------------------------------------------------------
// Compute domain decomposition along Z
// ---------------------------------------------------------------------------
static void computeDecomp(const size_t nz, const int rank, const int size,
                          size_t& z_start, size_t& nz_local) {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t extra = nz % static_cast<size_t>(size);
    if (static_cast<size_t>(rank) < extra) {
        nz_local = base + 1;
        z_start = static_cast<size_t>(rank) * nz_local;
    } else {
        nz_local = base;
        z_start = extra * (base + 1) +
                  static_cast<size_t>(rank - static_cast<int>(extra)) * base;
    }
}

// ---------------------------------------------------------------------------
// Initialize local slab (owned cells only, not halos)
// ---------------------------------------------------------------------------
static void initializeLocalGrid(std::vector<Real>& grid,
                                 const size_t nx, const size_t ny,
                                 const size_t nz_local, const size_t z_start) {
    const size_t nx_ny = nx * ny;
    #pragma omp parallel for
    for (int z_local = 0; z_local < static_cast<int>(nz_local); ++z_local) {
        const size_t global_z = z_start + static_cast<size_t>(z_local);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_idx = global_z * nx_ny + y * nx + x;
                grid[(static_cast<size_t>(z_local) + 1) * nx_ny + y * nx + x] =
                    static_cast<Real>((global_idx % 19));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Validation (called on rank 0 with full gathered grid)
// ---------------------------------------------------------------------------
static bool validateResult(const std::vector<Real>& grid,
                           const size_t nx, const size_t ny, const size_t nz) {
    const size_t n = nx * ny * nz;

    // 1. Check for NaN / Inf
    bool foundBad = false;
    #pragma omp parallel for reduction(||:foundBad)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            foundBad = true;
        }
    }
    if (foundBad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Value range
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < n; ++i) {
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

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void printUsage(const char* progName) {
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

// ===========================================================================
int main(int argc, char** argv) {
    // -----------------------------------------------------------------------
    // MPI initialisation
    // -----------------------------------------------------------------------
    MPI_Init(&argc, &argv);
    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // -----------------------------------------------------------------------
    // Parse arguments (duplicated on every rank)
    // -----------------------------------------------------------------------
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int    iterations = 10;
    bool   validate = false;
    bool   printResults = false;

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
            if (mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // -----------------------------------------------------------------------
    // OpenMP
    // -----------------------------------------------------------------------
    const int ompThreads = omp_get_max_threads();

    // -----------------------------------------------------------------------
    // CUDA device selection (round-robin across MPI ranks)
    // -----------------------------------------------------------------------
    int nGpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nGpus));
    if (nGpus == 0) {
        fprintf(stderr, "[rank %d] No CUDA devices available\n", mpiRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int cudaDev = mpiRank % nGpus;
    CUDA_CHECK(cudaSetDevice(cudaDev));

    cudaDeviceProp devProp;
    CUDA_CHECK(cudaGetDeviceProperties(&devProp, cudaDev));

    // -----------------------------------------------------------------------
    // Domain decomposition (Z dimension)
    // -----------------------------------------------------------------------
    size_t z_start = 0, nz_local = 0;
    computeDecomp(nz, mpiRank, mpiSize, z_start, nz_local);

    // Pre-compute decomposition for all ranks (needed for gather)
    std::vector<size_t> all_nz_local(static_cast<size_t>(mpiSize));
    std::vector<size_t> all_z_start(static_cast<size_t>(mpiSize));
    for (int r = 0; r < mpiSize; ++r)
        computeDecomp(nz, r, mpiSize, all_z_start[static_cast<size_t>(r)],
                      all_nz_local[static_cast<size_t>(r)]);

    // -----------------------------------------------------------------------
    // Print banner (rank 0)
    // -----------------------------------------------------------------------
    if (mpiRank == 0) {
        printf("3D Stencil Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI ranks: %d\n", mpiSize);
        printf("OpenMP threads: %d\n", ompThreads);
        printf("CUDA device: %s (SM %d.%d, %d SMs, %zu GB)\n",
               devProp.name, devProp.major, devProp.minor,
               devProp.multiProcessorCount,
               devProp.totalGlobalMem / (1024UL * 1024UL * 1024UL));
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Local slab Z: %zu planes\n", nz_local);
        fflush(stdout);
    }

    // If this rank has nothing to do, participate in MPI only
    const bool hasWork = (nz_local > 0 && nx > 2 && ny > 2);

    // -----------------------------------------------------------------------
    // Host memory layout: [halo_bottom | owned cells | halo_top]
    // Each slab: nx * ny * (nz_local + 2)  (2 extra for halos)
    // -----------------------------------------------------------------------
    const size_t nx_ny   = nx * ny;
    const size_t slabVol = nx_ny * (nz_local + 2);   // host vector size
    const size_t planeSz = nx_ny;                     // bytes: planeSz * sizeof(Real)

    std::vector<Real> h_grid(slabVol, Real{0});

    // Initialise owned cells
    if (hasWork) {
        initializeLocalGrid(h_grid, nx, ny, nz_local, z_start);
    }

    // -----------------------------------------------------------------------
    // Initial halo exchange (fill halos with neighbour data)
    // -----------------------------------------------------------------------
    {
        MPI_Status st;
        int left  = (mpiRank > 0)          ? mpiRank - 1 : MPI_PROC_NULL;
        int right = (mpiRank < mpiSize - 1) ? mpiRank + 1 : MPI_PROC_NULL;

        // Exchange with left neighbour: send my bottom plane, recv into my bottom halo
        MPI_Sendrecv(
            hasWork ? &h_grid[idx3(0, 0, 1, nx, ny)] : nullptr,
            static_cast<int>(planeSz), MPI_DOUBLE, left,  0,
            hasWork ? &h_grid[idx3(0, 0, 0, nx, ny)] : nullptr,
            static_cast<int>(planeSz), MPI_DOUBLE, left,  1,
            MPI_COMM_WORLD, &st);

        // Exchange with right neighbour: send my top plane, recv into my top halo
        MPI_Sendrecv(
            hasWork ? &h_grid[idx3(0, 0, nz_local, nx, ny)] : nullptr,
            static_cast<int>(planeSz), MPI_DOUBLE, right, 1,
            hasWork ? &h_grid[idx3(0, 0, nz_local + 1, nx, ny)] : nullptr,
            static_cast<int>(planeSz), MPI_DOUBLE, right, 0,
            MPI_COMM_WORLD, &st);
    }

    // -----------------------------------------------------------------------
    // CUDA memory allocation
    // -----------------------------------------------------------------------
    Real *d_in  = nullptr, *d_out = nullptr;
    if (hasWork) {
        const size_t devBytes = slabVol * sizeof(Real);
        CUDA_CHECK(cudaMalloc(&d_in,  devBytes));
        CUDA_CHECK(cudaMalloc(&d_out, devBytes));

        // Copy initial grid (incl. halos) to device
        CUDA_CHECK(cudaMemcpy(d_in, h_grid.data(), devBytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, devBytes));
    }

    // -----------------------------------------------------------------------
    // Stencil iteration loop
    // -----------------------------------------------------------------------
    if (mpiRank == 0) {
        printf("Running stencil computation...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double wallStart = MPI_Wtime();

    // CUDA kernel launch configuration
    const dim3 blockSz(16, 16, 1);
    const dim3 stencilGrid(
        hasWork ? ((nx - 2) + 15) / 16 : 1,
        hasWork ? ((ny - 2) + 15) / 16 : 1,
        hasWork ? nz_local : 1);
    const dim3 boundaryGrid(
        hasWork ? (nx + 15) / 16 : 1,
        hasWork ? (ny + 15) / 16 : 1,
        hasWork ? nz_local : 1);

    const bool isBottomRank = (mpiRank == 0);
    const bool isTopRank    = (mpiRank == mpiSize - 1);

    for (int iter = 0; iter < iterations; ++iter) {
        if (!hasWork) {
            // No local work; still need to synchronise with peers so that
            // every rank finishes the loop at roughly the same time.
            MPI_Barrier(MPI_COMM_WORLD);
            continue;
        }

        // ---- 1. Launch stencil kernel ------------------------------------
        stencilKernel<<<stencilGrid, blockSz>>>(d_in, d_out,
                                                 nx, ny, nz_local, nx_ny);
        CUDA_CHECK(cudaGetLastError());

        // ---- 2. Copy boundary values -------------------------------------
        boundaryKernel<<<boundaryGrid, blockSz>>>(d_in, d_out,
                                                   nx, ny, nz_local, nx_ny,
                                                   isBottomRank, isTopRank);
        CUDA_CHECK(cudaGetLastError());

        // ---- 3. Copy halo planes to host for MPI exchange -----------------
        // Bottom send plane  (z_local = 1)
        CUDA_CHECK(cudaMemcpy(&h_grid[idx3(0, 0, 1, nx, ny)],
                               &d_out[idx3(0, 0, 1, nx, ny)],
                               planeSz * sizeof(Real),
                               cudaMemcpyDeviceToHost));
        // Top send plane (z_local = nz_local)
        CUDA_CHECK(cudaMemcpy(&h_grid[idx3(0, 0, nz_local, nx, ny)],
                               &d_out[idx3(0, 0, nz_local, nx, ny)],
                               planeSz * sizeof(Real),
                               cudaMemcpyDeviceToHost));

        // ---- 4. MPI halo exchange -----------------------------------------
        {
            MPI_Status st;
            int left  = (mpiRank > 0)          ? mpiRank - 1 : MPI_PROC_NULL;
            int right = (mpiRank < mpiSize - 1) ? mpiRank + 1 : MPI_PROC_NULL;

            MPI_Sendrecv(
                &h_grid[idx3(0, 0, 1, nx, ny)],
                static_cast<int>(planeSz), MPI_DOUBLE, left,  0,
                &h_grid[idx3(0, 0, 0, nx, ny)],
                static_cast<int>(planeSz), MPI_DOUBLE, left,  1,
                MPI_COMM_WORLD, &st);

            MPI_Sendrecv(
                &h_grid[idx3(0, 0, nz_local, nx, ny)],
                static_cast<int>(planeSz), MPI_DOUBLE, right, 1,
                &h_grid[idx3(0, 0, nz_local + 1, nx, ny)],
                static_cast<int>(planeSz), MPI_DOUBLE, right, 0,
                MPI_COMM_WORLD, &st);
        }

        // ---- 5. Copy received halos back to device (into d_out) ----------
        CUDA_CHECK(cudaMemcpy(&d_out[idx3(0, 0, 0, nx, ny)],
                               &h_grid[idx3(0, 0, 0, nx, ny)],
                               planeSz * sizeof(Real),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_out[idx3(0, 0, nz_local + 1, nx, ny)],
                               &h_grid[idx3(0, 0, nz_local + 1, nx, ny)],
                               planeSz * sizeof(Real),
                               cudaMemcpyHostToDevice));

        // ---- 6. Swap buffers ----------------------------------------------
        std::swap(d_in, d_out);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const double wallEnd = MPI_Wtime();
    const double elapsed = wallEnd - wallStart;

    // Reduce time across ranks (max – wall time of the slowest rank)
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const size_t nzTotal = nz;
    // -----------------------------------------------------------------------
    // Copy final result to host
    // -----------------------------------------------------------------------
    // After the loop, d_in always holds the latest result (the swap at the
    // end of each iteration ensures this).
    Real* d_final = d_in;
    std::vector<Real> h_result;
    if (hasWork) {
        h_result.resize(slabVol);
        CUDA_CHECK(cudaMemcpy(h_result.data(), d_final, slabVol * sizeof(Real),
                              cudaMemcpyDeviceToHost));
    }

    // -----------------------------------------------------------------------
    // Gather full grid on rank 0 for validation / result printing
    // -----------------------------------------------------------------------
    std::vector<Real> fullGrid;
    if (mpiRank == 0) {
        fullGrid.resize(nx * ny * nzTotal);
    }

    // Build gather counts / displacements (in elements = planeSz * nz_local)
    std::vector<int> recvCounts(static_cast<size_t>(mpiSize));
    std::vector<int> recvDispls(static_cast<size_t>(mpiSize));
    if (mpiRank == 0) {
        for (int r = 0; r < mpiSize; ++r) {
            const size_t rnz = all_nz_local[static_cast<size_t>(r)];
            recvCounts[static_cast<size_t>(r)] =
                static_cast<int>(rnz * nx_ny);
            recvDispls[static_cast<size_t>(r)] =
                static_cast<int>(all_z_start[static_cast<size_t>(r)] * nx_ny);
        }
    }

    // Each rank packs its owned cells (skip halos) into a send buffer
    std::vector<Real> sendBuf;
    if (hasWork) {
        sendBuf.resize(nz_local * nx_ny);
        #pragma omp parallel for
        for (size_t z = 0; z < nz_local; ++z) {
            std::memcpy(&sendBuf[z * nx_ny],
                        &h_result[(z + 1) * nx_ny],
                        nx_ny * sizeof(Real));
        }
    }

    MPI_Gatherv(
        hasWork ? sendBuf.data() : nullptr,
        hasWork ? static_cast<int>(nz_local * nx_ny) : 0,
        MPI_DOUBLE,
        mpiRank == 0 ? fullGrid.data() : nullptr,
        recvCounts.data(),
        recvDispls.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // Report – rank 0 only
    // -----------------------------------------------------------------------
    if (mpiRank == 0) {
        printf("Computation time: %.3f s\n", maxElapsed);

        const double cellUpdates =
            static_cast<double>((nx - 2) * (ny - 2) * (nzTotal - 2)) *
            static_cast<double>(iterations);
        const double mcups =
            cellUpdates / maxElapsed / 1.0e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
        fflush(stdout);

        if (printResults) {
            print_results(fullGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            fflush(stdout);
            const bool valid = validateResult(fullGrid, nx, ny, nzTotal);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            fflush(stdout);
        }
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    if (hasWork) {
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }
    MPI_Finalize();
    return 0;
}
