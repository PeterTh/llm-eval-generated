#include <algorithm>
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

// Initialize the local slab (planes gz0 .. gz0+lnz-1 of the global grid) plus
// one ghost plane on each side. Values match the serial initialization based
// on the global linear index.
void initializeLocalGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                         const size_t lnz, const size_t gz0, const size_t gnz) {
    const size_t plane = nx * ny;
    for (size_t lz = 0; lz < lnz + 2; ++lz) {
        // Local plane lz corresponds to global plane gz0 + lz - 1 (lz==0 and
        // lz==lnz+1 are ghost planes). Skip ghosts outside the global grid.
        if (gz0 + lz == 0 || gz0 + lz - 1 >= gnz) continue;
        const size_t gz = gz0 + lz - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = idx3(x, y, gz, nx, ny);
                grid[lz * plane + y * nx + x] = (gidx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil update for a single local plane lz (global plane gz0+lz-1).
// Only x/y interior points are updated; x/y boundary values are never touched
// after initialization, matching the serial boundary-copy semantics.
inline void stencilPlane(const Real* __restrict__ input, Real* __restrict__ output,
                         const size_t nx, const size_t ny, const size_t lz) {
    const size_t plane = nx * ny;
    const Real* in0 = input + (lz - 1) * plane;
    const Real* in1 = input + lz * plane;
    const Real* in2 = input + (lz + 1) * plane;
    Real* out = output + lz * plane;
    for (size_t y = 1; y < ny - 1; ++y) {
        const size_t row = y * nx;
        for (size_t x = 1; x < nx - 1; ++x) {
            const size_t i = row + x;
            const Real center = in1[i];
            const Real left = in1[i - 1];
            const Real right = in1[i + 1];
            const Real front = in1[i - nx];
            const Real back = in1[i + nx];
            const Real bottom = in0[i];
            const Real top = in2[i];
            out[i] = (center + left + right + front + back + bottom + top) / 7.0;
        }
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

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        printf("MPI ranks: %d\n", nprocs);
    }

    // 1D block decomposition of the z dimension: rank r owns lnz contiguous
    // z-planes starting at gz0. Ranks beyond nz planes own nothing.
    const size_t base = nz / nprocs;
    const size_t rem = nz % nprocs;
    const size_t r = (size_t)rank;
    const size_t lnz = base + (r < rem ? 1 : 0);
    const size_t gz0 = r * base + std::min(r, rem);

    const size_t plane = nx * ny;
    const size_t localSize = (lnz + 2) * plane;  // owned planes + 2 ghost planes

    // Neighbor ranks in z (MPI_PROC_NULL at the chain ends and for/around
    // empty ranks; empty ranks can only occur at the tail of the rank order).
    int prev = MPI_PROC_NULL, next = MPI_PROC_NULL;
    if (lnz > 0) {
        if (gz0 > 0) prev = rank - 1;
        if (gz0 + lnz < nz) next = rank + 1;
    }

    // Allocate grids (double buffering). Both buffers are fully initialized:
    // boundary values are never overwritten (output = input at boundaries in
    // the serial code, so they keep their initial values forever).
    std::vector<Real> grid1(localSize, 0.0);
    std::vector<Real> grid2(localSize, 0.0);

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeLocalGrid(grid1, nx, ny, lnz, gz0, nz);
    initializeLocalGrid(grid2, nx, ny, lnz, gz0, nz);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    Real* src = grid1.data();
    Real* dst = grid2.data();

    for (int iter = 0; iter < iterations; ++iter) {
        // Exchange ghost planes of the input buffer with z-neighbors.
        MPI_Request reqs[4];
        MPI_Irecv(src, plane, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(src + (lnz + 1) * plane, plane, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(src + plane, plane, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(src + lnz * plane, plane, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[3]);

        // Overlap: update planes that do not depend on ghost data while the
        // halo exchange is in flight. Local plane lz maps to global plane
        // gz0 + lz - 1; only global planes 1 .. nz-2 are updated.
        for (size_t lz = 2; lz + 1 <= lnz; ++lz) {
            const size_t gz = gz0 + lz - 1;
            if (gz >= 1 && gz <= nz - 2)
                stencilPlane(src, dst, nx, ny, lz);
        }

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // Update the boundary planes of the local slab (need ghost data).
        if (lnz >= 1) {
            const size_t gzFirst = gz0;  // global plane of lz = 1
            if (gzFirst >= 1 && gzFirst <= nz - 2)
                stencilPlane(src, dst, nx, ny, 1);
        }
        if (lnz >= 2) {
            const size_t gzLast = gz0 + lnz - 1;  // global plane of lz = lnz
            if (gzLast >= 1 && gzLast <= nz - 2)
                stencilPlane(src, dst, nx, ny, lnz);
        }

        std::swap(src, dst);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - tStart;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long durationMs = (long)(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full grid on rank 0 for result printing / validation.
    if (printResults || validate) {
        std::vector<Real> fullGrid;
        std::vector<int> counts(nprocs), displs(nprocs);
        for (int p = 0; p < nprocs; ++p) {
            const size_t pr = (size_t)p;
            const size_t plnz = base + (pr < rem ? 1 : 0);
            const size_t pz0 = pr * base + std::min(pr, rem);
            counts[p] = (int)(plnz * plane);
            displs[p] = (int)(pz0 * plane);
        }
        if (rank == 0) fullGrid.resize(nx * ny * nz);
        MPI_Gatherv(src + plane, (int)(lnz * plane), MPI_DOUBLE,
                    fullGrid.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0 && printResults) {
            print_results(fullGrid, "Grid");
        }

        if (validate) {
            int valid = 1;
            if (rank == 0) {
                printf("Validating result...\n");
                valid = validateResult(fullGrid, nx, ny, nz) ? 1 : 0;
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
            MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
