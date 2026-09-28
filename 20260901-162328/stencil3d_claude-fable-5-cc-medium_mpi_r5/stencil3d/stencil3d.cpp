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

// Initialize the local slab (planes gzStart..gzStart+lnz-1 of the global grid)
// using the same values as the original serial initialization.
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t gzStart, const size_t lnz) {
    for (size_t lz = 0; lz < lnz; ++lz) {
        const size_t gz = gzStart + lz;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = idx3(x, y, gz, nx, ny);
                grid[idx3(x, y, lz + 1, nx, ny)] = (gidx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil update of the interior (1..nx-2, 1..ny-2) of one local plane.
// Local plane lz corresponds to a global interior plane; its z-neighbors are
// the adjacent local planes (owned or halo).
void stencilPlane(const Real* __restrict input, Real* __restrict output,
                  const size_t lz, const size_t nx, const size_t ny) {
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
            out[i] =(center + left + right + front + back + bottom + top) / 7.0;
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

    const size_t plane = nx * ny;

    // 1D block decomposition of the nz planes along Z.
    // Rank r owns global planes [gzStart, gzStart + lnz).
    const size_t base = nz / (size_t)nprocs;
    const size_t rem = nz % (size_t)nprocs;
    const size_t ur = (size_t)rank;
    const size_t lnz = base + (ur < rem ? 1 : 0);
    const size_t gzStart = ur * base + std::min(ur, rem);
    const size_t gzEnd = gzStart + lnz;  // exclusive

    // Neighbors in Z (ranks owning zero planes are trailing and have none)
    const int prevRank = (lnz > 0 && gzStart > 0) ? rank - 1 : MPI_PROC_NULL;
    const int nextRank = (lnz > 0 && gzEnd < nz) ? rank + 1 : MPI_PROC_NULL;

    // Allocate local slabs with one halo plane on each side (double buffering)
    const size_t localSize = (lnz + 2) * plane;
    std::vector<Real> grid1(localSize, 0.0);
    std::vector<Real> grid2(localSize, 0.0);

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, gzStart, lnz);
    // The global boundary values are invariant: each iteration copies them
    // from input to output, so they keep their initial values throughout.
    // Initialize them once in both buffers instead of recopying every step.
    for (size_t lz = 1; lz <= lnz; ++lz) {
        const size_t gz = gzStart + lz - 1;
        Real* g1 = grid1.data() + lz * plane;
        Real* g2 = grid2.data() + lz * plane;
        if (gz == 0 || gz == nz - 1) {
            memcpy(g2, g1, plane * sizeof(Real));
        } else {
            for (size_t y = 0; y < ny; ++y) {
                if (y == 0 || y == ny - 1) {
                    memcpy(g2 + y * nx, g1 + y * nx, nx * sizeof(Real));
                } else {
                    g2[y * nx] = g1[y * nx];
                    g2[y * nx + nx - 1] = g1[y * nx + nx - 1];
                }
            }
        }
    }

    // Range of local planes that are updated by the stencil
    // (owned planes whose global index is in [1, nz-2])
    const size_t lzFirst = (gzStart == 0) ? 2 : 1;
    const size_t lzLast = (gzEnd == nz) ? lnz - 1 : lnz;  // inclusive; underflows only if lnz==0
    const bool hasWork = (lnz > 0) && (lzFirst <= lzLast) && (lzLast <= lnz);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    Real* in = grid1.data();
    Real* out = grid2.data();

    for (int iter = 0; iter < iterations; ++iter) {
        // Exchange halo planes (non-blocking, overlapped with interior work)
        MPI_Request reqs[4];
        MPI_Irecv(in, (int)plane, MPI_DOUBLE, prevRank, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(in + (lnz + 1) * plane, (int)plane, MPI_DOUBLE, nextRank, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(in + plane, (int)plane, MPI_DOUBLE, prevRank, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(in + lnz * plane, (int)plane, MPI_DOUBLE, nextRank, 0, MPI_COMM_WORLD, &reqs[3]);

        if (hasWork && lnz >= 2) {
            // Planes that only depend on owned data, computed while halos travel
            const size_t innerFirst = std::max(lzFirst, (size_t)2);
            const size_t innerLast = std::min(lzLast, lnz - 1);
            for (size_t lz = innerFirst; lz <= innerLast; ++lz) {
                stencilPlane(in, out, lz, nx, ny);
            }
        }

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        if (hasWork) {
            // Halo-adjacent planes
            if (lzFirst == 1) {
                stencilPlane(in, out, 1, nx, ny);
            }
            if (lzLast == lnz && !(lnz == 1 && lzFirst == 1)) {
                stencilPlane(in, out, lnz, nx, ny);
            }
        }

        std::swap(in, out);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double tEnd = MPI_Wtime();
    const long durationMs = (long)((tEnd - tStart) * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full grid on rank 0 for result output / validation
    if (printResults || validate) {
        std::vector<Real> fullGrid;
        std::vector<int> counts, displs;
        if (rank == 0) {
            fullGrid.resize(nx * ny * nz);
            counts.resize(nprocs);
            displs.resize(nprocs);
            for (int r = 0; r < nprocs; ++r) {
                const size_t sr = (size_t)r;
                const size_t rlnz = base + (sr < rem ? 1 : 0);
                const size_t rgzStart = sr * base + std::min(sr, rem);
                counts[r] = (int)(rlnz * plane);
                displs[r] = (int)(rgzStart * plane);
            }
        }
        // "in" holds the final grid after the last swap
        MPI_Gatherv(in + plane, (int)(lnz * plane), MPI_DOUBLE,
                    fullGrid.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        int exitCode = 0;
        if (rank == 0) {
            if (printResults) {
                print_results(fullGrid, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(fullGrid, nx, ny, nz);

                if (valid) {
                    printf("Validation: PASSED\n");
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

    MPI_Finalize();
    return 0;
}
