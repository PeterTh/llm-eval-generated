#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize the locally owned slab (plus halo planes) directly from the global
// index formula used by the sequential version: grid[globalIdx] = (globalIdx % 19) * 1.0
// `local` holds (lnz + 2) planes; local plane 0 is the lower halo (global plane z0 - 1).
void initializeLocalGrid(std::vector<Real>& local, const size_t nx, const size_t ny, const size_t nz,
                         const size_t z0, const size_t lnz) {
    const size_t planeSize = nx * ny;
    for (size_t lz = 0; lz < lnz + 2; ++lz) {
        // global plane index (may be out of range for halos at the domain boundary)
        const long long gz = static_cast<long long>(z0) + static_cast<long long>(lz) - 1;
        Real* dst = local.data() + lz * planeSize;
        if (gz < 0 || gz >= static_cast<long long>(nz)) {
            std::fill(dst, dst + planeSize, Real(0));
            continue;
        }
        const size_t base = static_cast<size_t>(gz) * planeSize;
        for (size_t i = 0; i < planeSize; ++i) {
            dst[i] = static_cast<Real>((base + i) % 19);
        }
    }
}

// 7-point stencil for a single owned plane `lz` (local coordinates), where the
// corresponding global plane index is gz. Interior planes are updated with the
// stencil, global boundary planes are copied verbatim.
static inline void stencilPlane(const Real* __restrict in, Real* __restrict out,
                                const size_t nx, const size_t ny, const size_t nz,
                                const size_t lz, const size_t gz) {
    const size_t planeSize = nx * ny;
    const Real* inP = in + lz * planeSize;
    Real* outP = out + lz * planeSize;

    if (gz == 0 || gz + 1 >= nz) {
        // Global boundary plane: pure copy
        std::memcpy(outP, inP, planeSize * sizeof(Real));
        return;
    }

    const Real* inBelow = inP - planeSize;
    const Real* inAbove = inP + planeSize;

    // y = 0 boundary row: copy
    std::memcpy(outP, inP, nx * sizeof(Real));

    for (size_t y = 1; y + 1 < ny; ++y) {
        const Real* __restrict c = inP + y * nx;
        const Real* __restrict f = c - nx;
        const Real* __restrict b = c + nx;
        const Real* __restrict d = inBelow + y * nx;
        const Real* __restrict u = inAbove + y * nx;
        Real* __restrict o = outP + y * nx;

        o[0] = c[0];  // x = 0 boundary
        for (size_t x = 1; x + 1 < nx; ++x) {
            o[x] = (c[x] + c[x - 1] + c[x + 1] + f[x] + b[x] + d[x] + u[x]) / 7.0;
        }
        o[nx - 1] = c[nx - 1];  // x = nx-1 boundary
    }

    // y = ny-1 boundary row: copy
    if (ny > 1) {
        std::memcpy(outP + (ny - 1) * nx, inP + (ny - 1) * nx, nx * sizeof(Real));
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
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

    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (worldRank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", worldSize);
    }

    const size_t gridSize = nx * ny * nz;
    const size_t planeSize = nx * ny;

    // 1D domain decomposition along Z: each rank owns a contiguous slab of planes,
    // so that halo planes are contiguous in memory and can be exchanged directly.
    const size_t base = nz / static_cast<size_t>(worldSize);
    const size_t rem = nz % static_cast<size_t>(worldSize);
    const size_t r = static_cast<size_t>(worldRank);
    const size_t lnz = base + (r < rem ? 1 : 0);
    const size_t z0 = r * base + std::min(r, rem);

    // Ranks without any planes (more ranks than planes) stay out of the compute
    // communicator but still participate in the final gather.
    const int active = (lnz > 0) ? 1 : 0;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active, worldRank, &comm);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    // Local grids, double buffered, with one halo plane on each side
    const size_t localCells = (lnz + 2) * planeSize;
    std::vector<Real> grid1(active ? localCells : 0);
    std::vector<Real> grid2(active ? localCells : 0);

    if (worldRank == 0) printf("Initializing grid...\n");
    if (active) {
        initializeLocalGrid(grid1, nx, ny, nz, z0, lnz);
        initializeLocalGrid(grid2, nx, ny, nz, z0, lnz);
    }

    if (worldRank == 0) printf("Running stencil computation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (active) {
        const int below = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        const int above = (rank + 1 < nranks) ? rank + 1 : MPI_PROC_NULL;
        const int haloCount = static_cast<int>(planeSize);

        for (int iter = 0; iter < iterations; ++iter) {
            Real* in = ((iter % 2) == 0) ? grid1.data() : grid2.data();
            Real* out = ((iter % 2) == 0) ? grid2.data() : grid1.data();

            // Exchange halo planes of the input grid
            MPI_Request reqs[4];
            MPI_Irecv(in, haloCount, MPI_DOUBLE, below, 0, comm, &reqs[0]);
            MPI_Irecv(in + (lnz + 1) * planeSize, haloCount, MPI_DOUBLE, above, 1, comm, &reqs[1]);
            MPI_Isend(in + planeSize, haloCount, MPI_DOUBLE, below, 1, comm, &reqs[2]);
            MPI_Isend(in + lnz * planeSize, haloCount, MPI_DOUBLE, above, 0, comm, &reqs[3]);

            // Compute planes that do not depend on the halos while communication is in flight
            for (size_t lz = 2; lz + 1 <= lnz; ++lz) {
                stencilPlane(in, out, nx, ny, nz, lz, z0 + lz - 1);
            }

            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

            // Now the halo-dependent planes (first and last owned plane)
            stencilPlane(in, out, nx, ny, nz, 1, z0);
            if (lnz > 1) {
                stencilPlane(in, out, nx, ny, nz, lnz, z0 + lnz - 1);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long elapsedMs = duration.count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsedMs, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (worldRank == 0) {
        printf("Computation time: %ld ms\n", elapsedMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (elapsedMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final grid on rank 0 only if it is actually needed
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        // Use a plane-sized datatype so element counts stay well within int range
        MPI_Datatype planeType;
        MPI_Type_contiguous(static_cast<int>(planeSize), MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);

        std::vector<int> counts, displs;
        if (worldRank == 0) {
            counts.resize(worldSize);
            displs.resize(worldSize);
        }
        const int sendCount = static_cast<int>(lnz);
        MPI_Gather(&sendCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (worldRank == 0) {
            int off = 0;
            for (int i = 0; i < worldSize; ++i) {
                displs[i] = off;
                off += counts[i];
            }
            finalGrid.resize(gridSize);
        }

        const Real* sendBuf = active ? (((iterations % 2) == 0 ? grid1.data() : grid2.data()) + planeSize) : nullptr;
        MPI_Gatherv(sendBuf, sendCount, planeType,
                    finalGrid.data(), counts.data(), displs.data(), planeType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&planeType);
    }

    int ret = 0;
    if (worldRank == 0) {
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
                ret = 1;
            }
        }
    }
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Comm_free(&comm);
    MPI_Finalize();
    return ret;
}
