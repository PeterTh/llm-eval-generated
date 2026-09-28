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

// 3D index calculation (global grid)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local padded array layout: local plane index 0 and (nzLocal+1) are ghost
// planes (halo received from neighboring ranks), local planes 1..nzLocal
// hold this rank's own z-slab (global z = zStart + li - 1).
inline constexpr size_t idxLocal(const size_t x, const size_t y, const size_t li, const size_t nx, const size_t ny) noexcept {
    return li * (nx * ny) + y * nx + x;
}

// Compute this rank's slab of z-planes: [zStart, zStart + nzLocal)
void computeDecomposition(const size_t nz, const int size, const int rank, size_t& zStart, size_t& nzLocal) {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    nzLocal = base + (r < rem ? 1 : 0);
    zStart = r * base + std::min(r, rem);
}

void initializeGridLocal(std::vector<Real>& grid, const size_t nx, const size_t ny,
                          const size_t zStart, const size_t nzLocal) {
    for (size_t li = 1; li <= nzLocal; ++li) {
        const size_t z = zStart + li - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIdx = idx3(x, y, z, nx, ny);
                grid[idxLocal(x, y, li, nx, ny)] = (globalIdx % 19) * 1.0;
            }
        }
    }
}

// Exchange halo planes with neighboring ranks (z-decomposition)
void exchangeHalos(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nzLocal,
                    const int prevRank, const int nextRank, MPI_Comm comm) {
    const size_t planeElems = nx * ny;

    // Exchange with "next" neighbor: send our top real plane, receive into our top ghost
    MPI_Sendrecv(&grid[idxLocal(0, 0, nzLocal, nx, ny)], static_cast<int>(planeElems), MPI_DOUBLE, nextRank, 0,
                 &grid[idxLocal(0, 0, nzLocal + 1, nx, ny)], static_cast<int>(planeElems), MPI_DOUBLE, nextRank, 1,
                 comm, MPI_STATUS_IGNORE);

    // Exchange with "prev" neighbor: send our bottom real plane, receive into our bottom ghost
    MPI_Sendrecv(&grid[idxLocal(0, 0, 1, nx, ny)], static_cast<int>(planeElems), MPI_DOUBLE, prevRank, 1,
                 &grid[idxLocal(0, 0, 0, nx, ny)], static_cast<int>(planeElems), MPI_DOUBLE, prevRank, 0,
                 comm, MPI_STATUS_IGNORE);
}

// 7-point stencil computation on this rank's local slab (with halo planes already exchanged)
void stencilIterationLocal(const std::vector<Real>& input,
                            std::vector<Real>& output,
                            const size_t nx, const size_t ny, const size_t nz,
                            const size_t zStart, const size_t nzLocal) {
    for (size_t li = 1; li <= nzLocal; ++li) {
        const size_t z = zStart + li - 1;

        if (z == 0 || z == nz - 1) {
            // Global z-boundary plane: copy entire plane
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t li_idx = idxLocal(x, y, li, nx, ny);
                    output[li_idx] = input[li_idx];
                }
            }
            continue;
        }

        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t li_idx = idxLocal(x, y, li, nx, ny);

                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
                    // Boundary in x or y: copy
                    output[li_idx] = input[li_idx];
                    continue;
                }

                const Real center = input[li_idx];
                const Real left = input[idxLocal(x - 1, y, li, nx, ny)];
                const Real right = input[idxLocal(x + 1, y, li, nx, ny)];
                const Real front = input[idxLocal(x, y - 1, li, nx, ny)];
                const Real back = input[idxLocal(x, y + 1, li, nx, ny)];
                const Real bottom = input[idxLocal(x, y, li - 1, nx, ny)];
                const Real top = input[idxLocal(x, y, li + 1, nx, ny)];

                // Simple averaging stencil
                output[li_idx] = (center + left + right + front + back + bottom + top) / 7.0;
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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
        printf("3D Stencil Benchmark (MPI, %d ranks)\n", worldSize);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t gridSize = nx * ny * nz;

    // Decompose the grid along Z into contiguous slabs, one per rank
    size_t zStart = 0;
    size_t nzLocal = 0;
    computeDecomposition(nz, worldSize, rank, zStart, nzLocal);

    const int prevRank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int nextRank = (rank < worldSize - 1) ? rank + 1 : MPI_PROC_NULL;

    const size_t localPaddedSize = nx * ny * (nzLocal + 2);

    // Allocate local grids (double buffering), including halo planes
    std::vector<Real> grid1(localPaddedSize);
    std::vector<Real> grid2(localPaddedSize);

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGridLocal(grid1, nx, ny, zStart, nzLocal);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& in = (iter % 2 == 0) ? grid1 : grid2;
        std::vector<Real>& out = (iter % 2 == 0) ? grid2 : grid1;

        exchangeHalos(in, nx, ny, nzLocal, prevRank, nextRank, MPI_COMM_WORLD);
        stencilIterationLocal(in, out, nx, ny, nz, zStart, nzLocal);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto localDurationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    long long durationMsLL = static_cast<long long>(localDurationMs);
    long long maxDurationMsLL = 0;
    MPI_Reduce(&durationMsLL, &maxDurationMsLL, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", maxDurationMsLL);

        // Calculate performance metrics
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (maxDurationMsLL / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final grid to rank 0 for result printing / validation
    const std::vector<Real>& finalLocalGrid = (iterations % 2 == 0) ? grid1 : grid2;

    std::vector<Real> globalGrid;
    if (printResults || validate) {
        std::vector<int> recvCounts;
        std::vector<int> displs;
        if (rank == 0) {
            globalGrid.resize(gridSize);
            recvCounts.resize(worldSize);
            displs.resize(worldSize);
            for (int r = 0; r < worldSize; ++r) {
                size_t rZStart, rNzLocal;
                computeDecomposition(nz, worldSize, r, rZStart, rNzLocal);
                recvCounts[r] = static_cast<int>(rNzLocal * nx * ny);
                displs[r] = static_cast<int>(rZStart * nx * ny);
            }
        }

        MPI_Gatherv(&finalLocalGrid[idxLocal(0, 0, 1, nx, ny)], static_cast<int>(nzLocal * nx * ny), MPI_DOUBLE,
                    rank == 0 ? globalGrid.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    int result = 0;

    if (rank == 0) {
        if (printResults) {
            print_results(globalGrid, "Grid");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(globalGrid, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
                result = 0;
            } else {
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }

    if (validate) {
        MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return result;
}
