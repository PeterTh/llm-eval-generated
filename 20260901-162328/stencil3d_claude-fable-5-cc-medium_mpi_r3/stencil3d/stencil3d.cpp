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

// Initialize the local slab (planes zStart .. zStart+localNz-1 of the global grid).
// Local storage has one ghost plane on each side: local plane k holds global plane zStart+k-1.
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t localNz, const size_t zStart) {
    const size_t plane = nx * ny;
    for (size_t k = 0; k < localNz; ++k) {
        const size_t gz = zStart + k;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = idx3(x, y, gz, nx, ny);
                grid[(k + 1) * plane + y * nx + x] = (gidx % 19) * 1.0;
            }
        }
    }
}

// Apply the 7-point stencil to one local z-plane k (1-based, ghosts at 0 and localNz+1).
// gz is the corresponding global z index. Global boundary planes are copied verbatim;
// interior planes get the stencil in the interior and copied x/y boundaries.
static void computePlane(const Real* __restrict input, Real* __restrict output,
                         const size_t nx, const size_t ny, const size_t nz,
                         const size_t k, const size_t gz) {
    const size_t plane = nx * ny;
    const Real* in = input + k * plane;
    Real* out = output + k * plane;

    if (gz == 0 || gz == nz - 1) {
        // Global z-boundary plane: copy unchanged
        memcpy(out, in, plane * sizeof(Real));
        return;
    }

    const Real* inBelow = in - plane;
    const Real* inAbove = in + plane;

    // y = 0 boundary row: copy
    memcpy(out, in, nx * sizeof(Real));
    for (size_t y = 1; y < ny - 1; ++y) {
        const size_t row = y * nx;
        // x boundaries: copy
        out[row] = in[row];
        out[row + nx - 1] = in[row + nx - 1];
        for (size_t x = 1; x < nx - 1; ++x) {
            const size_t i = row + x;
            const Real center = in[i];
            const Real left = in[i - 1];
            const Real right = in[i + 1];
            const Real front = in[i - nx];
            const Real back = in[i + nx];
            const Real bottom = inBelow[i];
            const Real top = inAbove[i];
            out[i] = (center + left + right + front + back + bottom + top) / 7.0;
        }
    }
    // y = ny-1 boundary row: copy
    memcpy(out + (ny - 1) * nx, in + (ny - 1) * nx, nx * sizeof(Real));
}

// One stencil iteration over the local slab, overlapping the halo exchange
// with computation of planes that do not need ghost data.
void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz,
                      const size_t localNz, const size_t zStart,
                      const int prevRank, const int nextRank) {
    if (localNz == 0) return;

    const size_t plane = nx * ny;
    Real* in = input.data();
    Real* out = output.data();

    MPI_Request reqs[4];
    int nreq = 0;
    if (prevRank != MPI_PROC_NULL) {
        MPI_Irecv(in, (int)plane, MPI_DOUBLE, prevRank, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Isend(in + plane, (int)plane, MPI_DOUBLE, prevRank, 1, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    if (nextRank != MPI_PROC_NULL) {
        MPI_Irecv(in + (localNz + 1) * plane, (int)plane, MPI_DOUBLE, nextRank, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Isend(in + localNz * plane, (int)plane, MPI_DOUBLE, nextRank, 0, MPI_COMM_WORLD, &reqs[nreq++]);
    }

    // Compute planes that only need locally-owned data while halos are in flight
    for (size_t k = 2; k + 1 <= localNz; ++k) {
        computePlane(in, out, nx, ny, nz, k, zStart + k - 1);
    }

    if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // Boundary planes of the slab (need ghost data)
    computePlane(in, out, nx, ny, nz, 1, zStart);
    if (localNz > 1) {
        computePlane(in, out, nx, ny, nz, localNz, zStart + localNz - 1);
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
        printf("MPI ranks: %d\n", size);
    }

    // 1D domain decomposition along z: each rank owns a contiguous slab of planes
    const size_t base = nz / (size_t)size;
    const size_t rem = nz % (size_t)size;
    const size_t localNz = base + ((size_t)rank < rem ? 1 : 0);
    const size_t zStart = (size_t)rank * base + std::min((size_t)rank, rem);

    // Neighbors: ranks with zero planes sit at the tail, so active ranks are contiguous
    const int activeRanks = (int)std::min((size_t)size, nz);
    const int prevRank = (localNz > 0 && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int nextRank = (localNz > 0 && rank + 1 < activeRanks) ? rank + 1 : MPI_PROC_NULL;

    const size_t plane = nx * ny;
    const size_t localSize = (localNz + 2) * plane;  // slab + one ghost plane per side

    // Allocate grids (double buffering)
    std::vector<Real> grid1(localSize);
    std::vector<Real> grid2(localSize);

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, localNz, zStart);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, nz, localNz, zStart, prevRank, nextRank);
        } else {
            stencilIteration(grid2, grid1, nx, ny, nz, localNz, zStart, prevRank, nextRank);
        }
    }

    const double tEnd = MPI_Wtime();
    const double localDuration = tEnd - tStart;
    double maxDuration = 0.0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long durationMs = (long)(maxDuration * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;

    // Gather the full grid on rank 0 only when needed for output/validation
    if (printResults || validate) {
        std::vector<Real> fullGrid;
        std::vector<int> counts, displs;
        if (rank == 0) {
            fullGrid.resize(nx * ny * nz);
            counts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                const size_t rNz = base + ((size_t)r < rem ? 1 : 0);
                const size_t rStart = (size_t)r * base + std::min((size_t)r, rem);
                counts[r] = (int)(rNz * plane);
                displs[r] = (int)(rStart * plane);
            }
        }
        MPI_Gatherv(finalLocal.data() + plane, (int)(localNz * plane), MPI_DOUBLE,
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
