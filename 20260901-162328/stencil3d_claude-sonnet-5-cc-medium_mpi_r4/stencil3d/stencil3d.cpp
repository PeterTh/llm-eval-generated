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

// Fills the locally-owned planes [zStart, zStart+localNz) of the global grid.
// The local buffer has one ghost plane on each side (index 0 and localNz+1).
void initializeGridLocal(std::vector<Real>& localGrid, const size_t nx, const size_t ny,
                          const size_t zStart, const size_t localNz) {
    for (size_t zl = 0; zl < localNz; ++zl) {
        const size_t z = zStart + zl;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIdx = idx3(x, y, z, nx, ny);
                const size_t localIdx = idx3(x, y, zl + 1, nx, ny);
                localGrid[localIdx] = (globalIdx % 19) * 1.0;
            }
        }
    }
}

// Exchanges the single-plane halos with the up/down neighbors in the z decomposition.
void haloExchange(std::vector<Real>& localGrid, const size_t nx, const size_t ny, const size_t localNz,
                   const int up, const int down, MPI_Comm comm) {
    const size_t planeElems = nx * ny;

    // Send our last owned plane down, receive the neighbor-above's plane into our top ghost.
    MPI_Sendrecv(&localGrid[idx3(0, 0, localNz, nx, ny)], (int)planeElems, MPI_DOUBLE, down, 0,
                 &localGrid[idx3(0, 0, 0, nx, ny)], (int)planeElems, MPI_DOUBLE, up, 0,
                 comm, MPI_STATUS_IGNORE);

    // Send our first owned plane up, receive the neighbor-below's plane into our bottom ghost.
    MPI_Sendrecv(&localGrid[idx3(0, 0, 1, nx, ny)], (int)planeElems, MPI_DOUBLE, up, 1,
                 &localGrid[idx3(0, 0, localNz + 1, nx, ny)], (int)planeElems, MPI_DOUBLE, down, 1,
                 comm, MPI_STATUS_IGNORE);
}

// 7-point stencil computation over the locally-owned planes.
// zStart/nzGlobal identify this rank's slice within the global domain so that
// global boundary planes (z == 0 or z == nzGlobal-1) are copied rather than computed,
// matching the semantics of the original single-process implementation.
void stencilIterationLocal(const std::vector<Real>& input,
                            std::vector<Real>& output,
                            const size_t nx, const size_t ny,
                            const size_t zStart, const size_t localNz, const size_t nzGlobal) {
    for (size_t zl = 1; zl <= localNz; ++zl) {
        const size_t z = zStart + zl - 1;
        const bool zBoundary = (z == 0 || z == nzGlobal - 1);

        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, zl, nx, ny);

                if (!zBoundary && x != 0 && x != nx - 1 && y != 0 && y != ny - 1) {
                    const Real center = input[idx];
                    const Real left = input[idx3(x - 1, y, zl, nx, ny)];
                    const Real right = input[idx3(x + 1, y, zl, nx, ny)];
                    const Real front = input[idx3(x, y - 1, zl, nx, ny)];
                    const Real back = input[idx3(x, y + 1, zl, nx, ny)];
                    const Real bottom = input[idx3(x, y, zl - 1, nx, ny)];
                    const Real top = input[idx3(x, y, zl + 1, nx, ny)];

                    // Simple averaging stencil
                    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                } else {
                    // Boundary point (global domain edge, in x, y, or z): copy through.
                    output[idx] = input[idx];
                }
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

    int rank = 0, numRanks = 1;
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

    if (nz < (size_t)numRanks) {
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

    // 1D domain decomposition along Z: split nz as evenly as possible across ranks.
    const size_t baseNz = nz / (size_t)numRanks;
    const size_t remNz = nz % (size_t)numRanks;
    const size_t localNz = baseNz + ((size_t)rank < remNz ? 1 : 0);
    const size_t zStart = (size_t)rank * baseNz + std::min((size_t)rank, remNz);

    const int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int down = (rank == numRanks - 1) ? MPI_PROC_NULL : rank + 1;

    const size_t localPlanes = localNz + 2;  // +2 ghost planes
    const size_t localSize = nx * ny * localPlanes;

    // Allocate local grids (double buffering), each with ghost planes for halo exchange.
    std::vector<Real> grid1(localSize);
    std::vector<Real> grid2(localSize);

    // Initialize (only the locally-owned planes; ghosts are filled by the first halo exchange)
    if (rank == 0) printf("Initializing grid...\n");
    initializeGridLocal(grid1, nx, ny, zStart, localNz);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& input = (iter % 2 == 0) ? grid1 : grid2;
        std::vector<Real>& output = (iter % 2 == 0) ? grid2 : grid1;

        haloExchange(input, nx, ny, localNz, up, down, MPI_COMM_WORLD);
        stencilIterationLocal(input, output, nx, ny, zStart, localNz, nz);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localMs = duration.count();
    long globalMs = 0;
    MPI_Reduce(&localMs, &globalMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", globalMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (globalMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalLocalGrid = (iterations % 2 == 0) ? grid1 : grid2;

    int exitCode = 0;

    if (printResults || validate) {
        // Gather the full grid on rank 0 for external validation / result printing,
        // matching the semantics of the original single-process implementation.
        std::vector<int> recvCounts;
        std::vector<int> displs;
        if (rank == 0) {
            recvCounts.resize(numRanks);
            displs.resize(numRanks);
            for (int r = 0; r < numRanks; ++r) {
                const size_t rLocalNz = baseNz + ((size_t)r < remNz ? 1 : 0);
                const size_t rZStart = (size_t)r * baseNz + std::min((size_t)r, remNz);
                recvCounts[r] = (int)(rLocalNz * nx * ny);
                displs[r] = (int)(rZStart * nx * ny);
            }
        }

        std::vector<Real> fullGrid;
        if (rank == 0) fullGrid.resize(nx * ny * nz);

        MPI_Gatherv(&finalLocalGrid[idx3(0, 0, 1, nx, ny)], (int)(localNz * nx * ny), MPI_DOUBLE,
                    rank == 0 ? fullGrid.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(fullGrid, "Grid");
            }

            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(fullGrid, nx, ny, nz);

                if (valid) {
                    printf("Validation: PASSED\n");
                    exitCode = 0;
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exitCode;
}
