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

// 3D index calculation (global indexing convention: z is the slowest-varying dimension)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local grid layout: nx * ny * (localNz + 2) with one ghost plane below and above the
// owned slab. Local z-index 0 is the "below" ghost, 1..localNz are owned planes, and
// localNz+1 is the "above" ghost.
inline constexpr size_t localIdx(const size_t x, const size_t y, const size_t lz, const size_t nx, const size_t ny) noexcept {
    return lz * (nx * ny) + y * nx + x;
}

// Initialize this rank's owned planes using the same value formula as the serial
// version, but keyed off the global index so results are identical regardless of
// decomposition.
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                     const size_t localNz, const size_t zOffset) {
    for (size_t lz = 1; lz <= localNz; ++lz) {
        const size_t zGlobal = zOffset + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIdx = idx3(x, y, zGlobal, nx, ny);
                grid[localIdx(x, y, lz, nx, ny)] = (globalIdx % 19) * 1.0;
            }
        }
    }
}

// Exchange one ghost plane (nx*ny doubles) with each neighbor along z, non-blocking so
// the exchange can overlap with computation of the interior planes that don't depend on
// ghost data.
void exchangeHalos(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t localNz, const int rank, const int numRanks,
                    MPI_Comm comm, MPI_Request reqs[4], int& numReqs) {
    numReqs = 0;
    const size_t planeSize = nx * ny;
    const int downRank = rank - 1;
    const int upRank = rank + 1;

    if (downRank >= 0) {
        MPI_Irecv(&grid[localIdx(0, 0, 0, nx, ny)], planeSize, MPI_DOUBLE, downRank, 0, comm, &reqs[numReqs++]);
        MPI_Isend(&grid[localIdx(0, 0, 1, nx, ny)], planeSize, MPI_DOUBLE, downRank, 1, comm, &reqs[numReqs++]);
    }
    if (upRank < numRanks) {
        MPI_Irecv(&grid[localIdx(0, 0, localNz + 1, nx, ny)], planeSize, MPI_DOUBLE, upRank, 1, comm, &reqs[numReqs++]);
        MPI_Isend(&grid[localIdx(0, 0, localNz, nx, ny)], planeSize, MPI_DOUBLE, upRank, 0, comm, &reqs[numReqs++]);
    }
}

// Compute (or boundary-copy) a single owned z-plane. zInterior indicates whether the
// global z coordinate of this plane is strictly interior (i.e. not a global boundary).
inline void computePlane(const std::vector<Real>& input, std::vector<Real>& output,
                          const size_t nx, const size_t ny, const size_t lz, const bool zInterior) {
    for (size_t y = 0; y < ny; ++y) {
        const bool yInterior = (y >= 1 && y + 1 < ny);
        for (size_t x = 0; x < nx; ++x) {
            const size_t outIdx = localIdx(x, y, lz, nx, ny);
            const bool xInterior = (x >= 1 && x + 1 < nx);
            if (zInterior && yInterior && xInterior) {
                const Real center = input[outIdx];
                const Real left = input[localIdx(x - 1, y, lz, nx, ny)];
                const Real right = input[localIdx(x + 1, y, lz, nx, ny)];
                const Real front = input[localIdx(x, y - 1, lz, nx, ny)];
                const Real back = input[localIdx(x, y + 1, lz, nx, ny)];
                const Real bottom = input[localIdx(x, y, lz - 1, nx, ny)];
                const Real top = input[localIdx(x, y, lz + 1, nx, ny)];
                output[outIdx] = (center + left + right + front + back + bottom + top) / 7.0;
            } else {
                output[outIdx] = input[outIdx];
            }
        }
    }
}

// 7-point stencil computation for this rank's owned slab. Ghost planes in `input` must
// already hold up-to-date neighbor data (or be irrelevant, at a global z boundary)
// before this is called.
void stencilIteration(const std::vector<Real>& input, std::vector<Real>& output,
                       const size_t nx, const size_t ny, const size_t localNz,
                       const size_t nzGlobal, const size_t zOffset) {
    for (size_t lz = 1; lz <= localNz; ++lz) {
        const size_t zGlobal = zOffset + (lz - 1);
        const bool zInterior = (zGlobal >= 1 && zGlobal + 1 < nzGlobal);
        computePlane(input, output, nx, ny, lz, zInterior);
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

    if (nz < static_cast<size_t>(numRanks)) {
        if (rank == 0) {
            printf("Error: grid Z dimension (%zu) must be >= number of MPI ranks (%d)\n", nz, numRanks);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI, %d ranks)\n", numRanks);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Block decomposition of the Z dimension across ranks: the first (nz % numRanks)
    // ranks get one extra plane so the split is as even as possible.
    const size_t baseNz = nz / static_cast<size_t>(numRanks);
    const size_t remainder = nz % static_cast<size_t>(numRanks);
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t zOffset = static_cast<size_t>(rank) * baseNz + std::min(static_cast<size_t>(rank), remainder);

    const size_t localGridSize = nx * ny * (localNz + 2);

    // Allocate local grids (double buffering), including ghost planes
    std::vector<Real> grid1(localGridSize, 0.0);
    std::vector<Real> grid2(localGridSize, 0.0);

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, localNz, zOffset);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& in = (iter % 2 == 0) ? grid1 : grid2;
        std::vector<Real>& out = (iter % 2 == 0) ? grid2 : grid1;

        MPI_Request reqs[4];
        int numReqs = 0;
        exchangeHalos(in, nx, ny, localNz, rank, numRanks, MPI_COMM_WORLD, reqs, numReqs);

        // Compute interior planes that don't touch ghost data while the halo exchange
        // is in flight.
        for (size_t lz = 2; lz + 1 <= localNz; ++lz) {
            const size_t zGlobal = zOffset + (lz - 1);
            const bool zInterior = (zGlobal >= 1 && zGlobal + 1 < nz);
            computePlane(in, out, nx, ny, lz, zInterior);
        }

        if (numReqs > 0) {
            MPI_Waitall(numReqs, reqs, MPI_STATUSES_IGNORE);
        }

        // Compute (or boundary-copy) the first and last owned planes, which may depend
        // on freshly-received ghost data.
        {
            const size_t lz = 1;
            const size_t zGlobal = zOffset + (lz - 1);
            const bool zInterior = (zGlobal >= 1 && zGlobal + 1 < nz);
            computePlane(in, out, nx, ny, lz, zInterior);
        }
        if (localNz > 1) {
            const size_t lz = localNz;
            const size_t zGlobal = zOffset + (lz - 1);
            const bool zInterior = (zGlobal >= 1 && zGlobal + 1 < nz);
            computePlane(in, out, nx, ny, lz, zInterior);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double localSeconds = std::chrono::duration<double>(end - start).count();
    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long durationMs = static_cast<long>(maxSeconds * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / maxSeconds / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final grid onto rank 0 (in global z order) for result printing and
    // validation, matching the semantics of the original single-process program.
    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;

    std::vector<Real> globalGrid;
    std::vector<int> recvCounts;
    std::vector<int> displs;
    if (rank == 0) {
        globalGrid.resize(nx * ny * nz);
        recvCounts.resize(numRanks);
        displs.resize(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            const size_t rNz = baseNz + (static_cast<size_t>(r) < remainder ? 1 : 0);
            const size_t rOffset = static_cast<size_t>(r) * baseNz + std::min(static_cast<size_t>(r), remainder);
            recvCounts[r] = static_cast<int>(nx * ny * rNz);
            displs[r] = static_cast<int>(nx * ny * rOffset);
        }
    }

    MPI_Gatherv(&finalLocal[localIdx(0, 0, 1, nx, ny)], static_cast<int>(nx * ny * localNz), MPI_DOUBLE,
                rank == 0 ? globalGrid.data() : nullptr,
                rank == 0 ? recvCounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;

    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(globalGrid, "Grid");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(globalGrid, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
