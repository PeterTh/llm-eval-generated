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

// 3D index calculation (global indexing)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local 3D index calculation for a rank's slab, which is padded with one
// ghost plane below (local z = 0) and one ghost plane above (local z =
// nzLocal + 1). Local z = 1 .. nzLocal corresponds to global z = zStart ..
// zStart + nzLocal - 1.
inline constexpr size_t lidx3(const size_t x, const size_t y, const size_t lz, const size_t nx, const size_t ny) noexcept {
    return lz * (nx * ny) + y * nx + x;
}

// Initialize the local slab using the same value formula as the original
// single-process implementation, keyed on the *global* index.
void initializeLocalGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                          const size_t nzLocal, const size_t zStart) {
    for (size_t lz = 1; lz <= nzLocal; ++lz) {
        const size_t z = zStart + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = idx3(x, y, z, nx, ny);
                grid[lidx3(x, y, lz, nx, ny)] = (gidx % 19) * 1.0;
            }
        }
    }
}

// Exchange the ghost planes needed to compute the stencil at the slab
// boundaries with the immediate neighbor ranks in the Z decomposition.
void exchangeHalos(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nzLocal,
                    const int rankBelow, const int rankAbove, MPI_Comm comm) {
    const size_t planeElems = nx * ny;
    Real* lowGhost = grid.data() + lidx3(0, 0, 0, nx, ny);
    Real* lowOwned = grid.data() + lidx3(0, 0, 1, nx, ny);
    Real* highOwned = grid.data() + lidx3(0, 0, nzLocal, nx, ny);
    Real* highGhost = grid.data() + lidx3(0, 0, nzLocal + 1, nx, ny);

    MPI_Sendrecv(lowOwned, static_cast<int>(planeElems), MPI_DOUBLE, rankBelow, 0,
                 highGhost, static_cast<int>(planeElems), MPI_DOUBLE, rankAbove, 0,
                 comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(highOwned, static_cast<int>(planeElems), MPI_DOUBLE, rankAbove, 1,
                 lowGhost, static_cast<int>(planeElems), MPI_DOUBLE, rankBelow, 1,
                 comm, MPI_STATUS_IGNORE);
}

// 7-point stencil computation on a rank's local slab (with ghost planes).
// zStart/nz identify where this slab sits within the global domain so that
// the global boundary planes (z == 0 and z == nz-1) are copied rather than
// computed, exactly like the original single-process code.
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz,
                      const size_t nzLocal, const size_t zStart) {
    for (size_t lz = 1; lz <= nzLocal; ++lz) {
        const size_t z = zStart + (lz - 1);
        const bool zBoundary = (z == 0 || z == nz - 1);

        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t lidx = lidx3(x, y, lz, nx, ny);
                if (zBoundary || x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
                    output[lidx] = input[lidx];
                    continue;
                }

                const Real center = input[lidx];
                const Real left = input[lidx3(x - 1, y, lz, nx, ny)];
                const Real right = input[lidx3(x + 1, y, lz, nx, ny)];
                const Real front = input[lidx3(x, y - 1, lz, nx, ny)];
                const Real back = input[lidx3(x, y + 1, lz, nx, ny)];
                const Real bottom = input[lidx3(x, y, lz - 1, nx, ny)];
                const Real top = input[lidx3(x, y, lz + 1, nx, ny)];

                // Simple averaging stencil
                output[lidx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
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

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    const bool isRoot = (rank == 0);

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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (static_cast<size_t>(numRanks) > nz) {
        if (isRoot) {
            printf("Error: number of MPI ranks (%d) exceeds grid size in Z dimension (%zu)\n", numRanks, nz);
        }
        MPI_Finalize();
        return 1;
    }

    if (isRoot) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
    }

    // Block decomposition of the Z dimension across ranks.
    const size_t baseZ = nz / static_cast<size_t>(numRanks);
    const size_t remZ = nz % static_cast<size_t>(numRanks);
    const size_t nzLocal = baseZ + (static_cast<size_t>(rank) < remZ ? 1 : 0);
    const size_t zStart = static_cast<size_t>(rank) * baseZ + std::min(static_cast<size_t>(rank), remZ);

    const int rankBelow = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int rankAbove = (rank < numRanks - 1) ? rank + 1 : MPI_PROC_NULL;

    const size_t planeElems = nx * ny;
    const size_t localSize = (nzLocal + 2) * planeElems;

    // Allocate local slabs (double buffering), padded with ghost planes.
    std::vector<Real> local1(localSize);
    std::vector<Real> local2(localSize);

    // Initialize
    if (isRoot) printf("Initializing grid...\n");
    initializeLocalGrid(local1, nx, ny, nzLocal, zStart);

    // Run stencil iterations
    if (isRoot) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& in = (iter % 2 == 0) ? local1 : local2;
        std::vector<Real>& out = (iter % 2 == 0) ? local2 : local1;

        exchangeHalos(in, nx, ny, nzLocal, rankBelow, rankAbove, MPI_COMM_WORLD);
        stencilIteration(in, out, nx, ny, nz, nzLocal, zStart);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (isRoot) printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    if (isRoot) {
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final grid onto rank 0 for result printing/validation, in
    // the same global layout the single-process version used.
    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? local1 : local2;

    std::vector<Real> finalGrid;
    std::vector<int> recvCounts;
    std::vector<int> displs;
    if (isRoot) {
        finalGrid.resize(nx * ny * nz);
        recvCounts.resize(numRanks);
        displs.resize(numRanks);
        size_t z = 0;
        for (int r = 0; r < numRanks; ++r) {
            const size_t cnt = baseZ + (static_cast<size_t>(r) < remZ ? 1 : 0);
            recvCounts[r] = static_cast<int>(cnt * planeElems);
            displs[r] = static_cast<int>(z * planeElems);
            z += cnt;
        }
    }

    MPI_Gatherv(finalLocal.data() + lidx3(0, 0, 1, nx, ny), static_cast<int>(nzLocal * planeElems), MPI_DOUBLE,
                isRoot ? finalGrid.data() : nullptr, isRoot ? recvCounts.data() : nullptr,
                isRoot ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print results for external validation
    if (isRoot && printResults) {
        print_results(finalGrid, "Grid");
    }

    // Validation
    int validExitCode = 0;
    if (validate) {
        if (isRoot) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
                validExitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                validExitCode = 1;
            }
        }
        MPI_Bcast(&validExitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return validExitCode;
    }

    MPI_Finalize();
    return 0;
}
