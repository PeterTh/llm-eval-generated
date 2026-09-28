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

// Initialize the locally owned z-slab [z0, z0+lnz) of the global grid.
// Local plane k is stored at plane k+1 (planes 0 and lnz+1 are halos).
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t z0, const size_t lnz) {
    for (size_t k = 0; k < lnz; ++k) {
        const size_t z = z0 + k;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = idx3(x, y, z, nx, ny);
                grid[idx3(x, y, k + 1, nx, ny)] = (gidx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil on one local plane (interior x/y points only).
// Boundary points are never touched: they were copied once at startup and
// stay at their initial values, exactly as in the serial code where
// output boundary = input boundary every iteration.
inline void stencilPlane(const Real* __restrict__ input, Real* __restrict__ output,
                         const size_t nx, const size_t ny, const size_t plane) {
    const size_t planeSize = nx * ny;
    const Real* in = input + plane * planeSize;
    Real* out = output + plane * planeSize;
    for (size_t y = 1; y < ny - 1; ++y) {
        const size_t row = y * nx;
        for (size_t x = 1; x < nx - 1; ++x) {
            const size_t idx = row + x;
            const Real center = in[idx];
            const Real left = in[idx - 1];
            const Real right = in[idx + 1];
            const Real front = in[idx - nx];
            const Real back = in[idx + nx];
            const Real bottom = in[idx - planeSize];
            const Real top = in[idx + planeSize];
            out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
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

    int rank, nprocs;
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

    const size_t planeSize = nx * ny;

    // 1D slab decomposition along z: rank r owns global planes [z0, z1)
    const size_t z0 = (nz * (size_t)rank) / (size_t)nprocs;
    const size_t z1 = (nz * (size_t)(rank + 1)) / (size_t)nprocs;
    const size_t lnz = z1 - z0;

    // Communicator of ranks that actually own planes (handles nprocs > nz)
    MPI_Comm activeComm;
    MPI_Comm_split(MPI_COMM_WORLD, lnz > 0 ? 0 : MPI_UNDEFINED, rank, &activeComm);

    int prev = MPI_PROC_NULL, next = MPI_PROC_NULL;
    if (lnz > 0) {
        int arank, asize;
        MPI_Comm_rank(activeComm, &arank);
        MPI_Comm_size(activeComm, &asize);
        if (arank > 0) prev = arank - 1;
        if (arank < asize - 1) next = arank + 1;
    }

    // Local grids with one halo plane on each side (double buffering)
    const size_t localSize = (lnz + 2) * planeSize;
    std::vector<Real> grid1(localSize, 0.0);
    std::vector<Real> grid2(localSize);

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, z0, lnz);

    // Boundary values never change (output boundary = input boundary each
    // iteration in the original code), so seed grid2 with them once instead
    // of copying every iteration. Interior points are overwritten anyway.
    grid2 = grid1;

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    // A local plane k (stored at index k+1) is computed iff its global plane
    // z0+k is interior, i.e. 1 <= z0+k <= nz-2.
    const size_t firstK = (z0 == 0) ? 1 : 0;
    const size_t lastK1 = (z1 == nz) ? (lnz > 0 ? lnz - 1 : 0) : lnz;  // exclusive

    Real* in = grid1.data();
    Real* out = grid2.data();

    for (int iter = 0; iter < iterations; ++iter) {
        if (lnz > 0) {
            // Start halo exchange of the current input's edge planes
            MPI_Request reqs[4];
            MPI_Irecv(in, (int)planeSize, MPI_DOUBLE, prev, 0, activeComm, &reqs[0]);
            MPI_Irecv(in + (lnz + 1) * planeSize, (int)planeSize, MPI_DOUBLE, next, 1, activeComm, &reqs[1]);
            MPI_Isend(in + planeSize, (int)planeSize, MPI_DOUBLE, prev, 1, activeComm, &reqs[2]);
            MPI_Isend(in + lnz * planeSize, (int)planeSize, MPI_DOUBLE, next, 0, activeComm, &reqs[3]);

            // Overlap: compute planes that do not touch the halos,
            // i.e. local k in [1, lnz-2] intersected with [firstK, lastK1)
            const size_t innerLo = std::max(firstK, (size_t)1);
            const size_t innerHi = std::min(lastK1, lnz - 1);
            for (size_t k = innerLo; k < innerHi; ++k) {
                stencilPlane(in, out, nx, ny, k + 1);
            }

            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

            // Remaining edge planes (need fresh halo data)
            if (lastK1 > firstK) {
                const size_t kLo = firstK;
                const size_t kHi = lastK1 - 1;
                if (kLo < innerLo || kLo >= innerHi) {
                    stencilPlane(in, out, nx, ny, kLo + 1);
                }
                if (kHi != kLo && (kHi < innerLo || kHi >= innerHi)) {
                    stencilPlane(in, out, nx, ny, kHi + 1);
                }
            }
        }
        std::swap(in, out);
    }

    const double tEnd = MPI_Wtime();
    double localTime = tEnd - tStart;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const long durationMs = (long)(maxTime * 1000.0);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / maxTime / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final grid to rank 0 for result output / validation
    const Real* finalLocal = (iterations % 2 == 0) ? grid1.data() : grid2.data();
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        std::vector<int> counts(nprocs), displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            const size_t rz0 = (nz * (size_t)r) / (size_t)nprocs;
            const size_t rz1 = (nz * (size_t)(r + 1)) / (size_t)nprocs;
            counts[r] = (int)((rz1 - rz0) * planeSize);
            displs[r] = (int)(rz0 * planeSize);
        }
        if (rank == 0) finalGrid.resize(nx * ny * nz);
        MPI_Gatherv(finalLocal + planeSize, (int)(lnz * planeSize), MPI_DOUBLE,
                    finalGrid.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0) {
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

    if (activeComm != MPI_COMM_NULL) MPI_Comm_free(&activeComm);
    MPI_Finalize();
    return exitCode;
}
