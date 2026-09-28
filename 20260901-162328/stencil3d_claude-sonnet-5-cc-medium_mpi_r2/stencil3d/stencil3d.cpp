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

// Computes the [start, start+count) range of global z-planes owned by a given rank
// under a balanced 1D block decomposition of nz planes across nranks ranks.
void decomposeRange(const size_t nz, const int nranks, const int rank, size_t& zStart, size_t& zCount) {
    const size_t base = nz / static_cast<size_t>(nranks);
    const size_t rem = nz % static_cast<size_t>(nranks);
    const size_t r = static_cast<size_t>(rank);
    zCount = base + (r < rem ? 1 : 0);
    zStart = r * base + std::min(r, rem);
}

// Initializes the local slab (with one ghost plane on each side) so that each
// element matches the value the original single-process code would have
// produced at the corresponding global index.
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                     const size_t zStart, const size_t nzLocal) {
    for (size_t li = 1; li <= nzLocal; ++li) {
        const size_t zGlobal = zStart + li - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t localIdx = idx3(x, y, li, nx, ny);
                const size_t globalIdx = idx3(x, y, zGlobal, nx, ny);
                grid[localIdx] = (globalIdx % 19) * 1.0;
            }
        }
    }
}

// Exchanges the single-plane halos needed for the 7-point stencil with the
// rank below (z-1) and the rank above (z+1) in the 1D z decomposition.
void exchangeHalos(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nzLocal,
                    const int rankBelow, const int rankAbove, MPI_Comm comm) {
    const size_t planeElems = nx * ny;
    Real* sendDown = &grid[idx3(0, 0, 1, nx, ny)];
    Real* recvDown = &grid[idx3(0, 0, 0, nx, ny)];
    Real* sendUp = &grid[idx3(0, 0, nzLocal, nx, ny)];
    Real* recvUp = &grid[idx3(0, 0, nzLocal + 1, nx, ny)];

    MPI_Sendrecv(sendDown, static_cast<int>(planeElems), MPI_DOUBLE, rankBelow, 0,
                 recvUp, static_cast<int>(planeElems), MPI_DOUBLE, rankAbove, 0,
                 comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(sendUp, static_cast<int>(planeElems), MPI_DOUBLE, rankAbove, 1,
                 recvDown, static_cast<int>(planeElems), MPI_DOUBLE, rankBelow, 1,
                 comm, MPI_STATUS_IGNORE);
}

// 7-point stencil computation over the local slab (local z indices 1..nzLocal,
// with ghost planes at 0 and nzLocal+1 already populated via halo exchange).
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny,
                      const size_t nzLocal, const size_t zStart, const size_t nzGlobal) {
    for (size_t li = 1; li <= nzLocal; ++li) {
        const size_t zGlobal = zStart + li - 1;
        const bool zBoundary = (zGlobal == 0 || zGlobal == nzGlobal - 1);
        for (size_t y = 0; y < ny; ++y) {
            const bool yBoundary = (y == 0 || y == ny - 1);
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, li, nx, ny);
                if (zBoundary || yBoundary || x == 0 || x == nx - 1) {
                    output[idx] = input[idx];
                    continue;
                }

                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, li, nx, ny)];
                const Real right = input[idx3(x+1, y, li, nx, ny)];
                const Real front = input[idx3(x, y-1, li, nx, ny)];
                const Real back = input[idx3(x, y+1, li, nx, ny)];
                const Real bottom = input[idx3(x, y, li-1, nx, ny)];
                const Real top = input[idx3(x, y, li+1, nx, ny)];

                // Simple averaging stencil
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
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
    int nranks = 1;
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

    if (static_cast<size_t>(nranks) > nz) {
        if (rank == 0) {
            printf("Error: number of MPI ranks (%d) exceeds grid size in Z dimension (%zu)\n", nranks, nz);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nranks);
    }

    // Decompose the Z dimension across ranks (1D slab decomposition)
    size_t zStart = 0;
    size_t nzLocal = 0;
    decomposeRange(nz, nranks, rank, zStart, nzLocal);

    const int rankBelow = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int rankAbove = (rank == nranks - 1) ? MPI_PROC_NULL : rank + 1;

    const size_t localBufSize = nx * ny * (nzLocal + 2);

    // Allocate local slabs (double buffering), each with one ghost plane per side
    std::vector<Real> grid1(localBufSize);
    std::vector<Real> grid2(localBufSize);

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, zStart, nzLocal);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& input = (iter % 2 == 0) ? grid1 : grid2;
        std::vector<Real>& output = (iter % 2 == 0) ? grid2 : grid1;

        exchangeHalos(input, nx, ny, nzLocal, rankBelow, rankAbove, MPI_COMM_WORLD);
        stencilIteration(input, output, nx, ny, nzLocal, zStart, nz);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDurationMs = duration.count();
    long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDurationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (maxDurationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;

    // Gather the full grid on rank 0 for external validation / printing
    if (printResults || validate) {
        std::vector<int> recvCounts;
        std::vector<int> displs;
        std::vector<Real> fullGrid;

        if (rank == 0) {
            recvCounts.resize(nranks);
            displs.resize(nranks);
            for (int r = 0; r < nranks; ++r) {
                size_t rzStart, rzCount;
                decomposeRange(nz, nranks, r, rzStart, rzCount);
                recvCounts[r] = static_cast<int>(rzCount * nx * ny);
                displs[r] = static_cast<int>(rzStart * nx * ny);
            }
            fullGrid.resize(nx * ny * nz);
        }

        MPI_Gatherv(&finalLocal[idx3(0, 0, 1, nx, ny)], static_cast<int>(nzLocal * nx * ny), MPI_DOUBLE,
                    rank == 0 ? fullGrid.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(fullGrid, "Grid");
            }

            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(fullGrid, nx, ny, nz);

                int result = 0;
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    result = 1;
                }
                MPI_Finalize();
                return result;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
