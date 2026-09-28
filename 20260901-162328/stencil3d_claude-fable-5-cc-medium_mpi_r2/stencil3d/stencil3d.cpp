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

// Initialize the local slab (z-slices [z0, z0+lnz) of the global grid) using
// the same formula as the original serial code, based on the global index.
// The local grid has one ghost layer on each side in z (offset +1).
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t z0, const size_t lnz) {
    for (size_t z = 0; z < lnz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = idx3(x, y, z0 + z, nx, ny);
                grid[idx3(x, y, z + 1, nx, ny)] = (gidx % 19) * 1.0;
            }
        }
    }
}

// Process one local z-slice (local index lz in [1, lnz], ghost offset layout).
// gz is the corresponding global z coordinate.
static void processSlice(const Real* __restrict input, Real* __restrict output,
                         const size_t nx, const size_t ny, const size_t nz,
                         const size_t lz, const size_t gz) {
    const size_t plane = nx * ny;
    const Real* in = input + lz * plane;
    Real* out = output + lz * plane;

    if (gz == 0 || gz == nz - 1) {
        // Global z boundary: copy the whole slice
        memcpy(out, in, plane * sizeof(Real));
        return;
    }

    // y boundaries: copy rows y=0 and y=ny-1
    memcpy(out, in, nx * sizeof(Real));
    memcpy(out + (ny - 1) * nx, in + (ny - 1) * nx, nx * sizeof(Real));

    for (size_t y = 1; y + 1 < ny; ++y) {
        const Real* row = in + y * nx;
        const Real* rowN = row - nx;         // y-1
        const Real* rowS = row + nx;         // y+1
        const Real* rowB = row - plane;      // z-1
        const Real* rowT = row + plane;      // z+1
        Real* orow = out + y * nx;

        // x boundaries: copy
        orow[0] = row[0];
        orow[nx - 1] = row[nx - 1];

        for (size_t x = 1; x + 1 < nx; ++x) {
            orow[x] = (row[x] + row[x - 1] + row[x + 1] + rowN[x] + rowS[x] +
                       rowB[x] + rowT[x]) / 7.0;
        }
    }
}

// 7-point stencil iteration over the local slab, with non-blocking halo
// exchange overlapped with the computation of interior slices.
void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz,
                      const size_t z0, const size_t lnz,
                      const int prev, const int next) {
    const size_t plane = nx * ny;
    Real* in = input.data();
    Real* out = output.data();

    // Exchange halos of the input grid: first/last owned slices go to the
    // neighbors' ghost layers.
    MPI_Request reqs[4];
    MPI_Irecv(in, (int)plane, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(in + (lnz + 1) * plane, (int)plane, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &reqs[1]);
    MPI_Isend(in + plane, (int)plane, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(in + lnz * plane, (int)plane, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[3]);

    // Interior slices (do not need ghost data)
    for (size_t lz = 2; lz + 1 <= lnz; ++lz) {
        processSlice(in, out, nx, ny, nz, lz, z0 + lz - 1);
    }

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    // Boundary slices (need ghost data)
    if (lnz >= 1) {
        processSlice(in, out, nx, ny, nz, 1, z0);
        if (lnz >= 2) {
            processSlice(in, out, nx, ny, nz, lnz, z0 + lnz - 1);
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

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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
        printf("MPI ranks: %d\n", nranks);
    }

    // 1D slab decomposition along z: rank owns global slices [z0, z0+lnz)
    const size_t base = nz / (size_t)nranks;
    const size_t rem = nz % (size_t)nranks;
    const size_t lnz = base + ((size_t)rank < rem ? 1 : 0);
    const size_t z0 = (size_t)rank * base + std::min((size_t)rank, rem);

    // Neighbors in z (empty ranks, if any, are all at the top end)
    const int prev = (z0 > 0 && lnz > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next = (lnz > 0 && z0 + lnz < nz) ? rank + 1 : MPI_PROC_NULL;

    const size_t plane = nx * ny;
    const size_t localSize = (lnz + 2) * plane;  // includes 2 ghost planes

    // Allocate local grids (double buffering)
    std::vector<Real> grid1(localSize);
    std::vector<Real> grid2(localSize);

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, z0, lnz);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, nz, z0, lnz, prev, next);
        } else {
            stencilIteration(grid2, grid1, nx, ny, nz, z0, lnz, prev, next);
        }
    }

    double elapsed = MPI_Wtime() - tStart;
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const long durationMs = (long)(elapsed * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / elapsed / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;

    // Gather the full grid on rank 0 for result printing / validation
    if (printResults || validate) {
        std::vector<Real> fullGrid;
        std::vector<int> counts(nranks), displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            const size_t rlnz = base + ((size_t)r < rem ? 1 : 0);
            const size_t rz0 = (size_t)r * base + std::min((size_t)r, rem);
            counts[r] = (int)(rlnz * plane);
            displs[r] = (int)(rz0 * plane);
        }
        if (rank == 0) fullGrid.resize(nx * ny * nz);
        MPI_Gatherv(finalLocal.data() + plane, (int)(lnz * plane), MPI_DOUBLE,
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
