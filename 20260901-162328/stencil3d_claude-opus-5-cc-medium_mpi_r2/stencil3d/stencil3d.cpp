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

// The grid is distributed over the ranks as slabs of contiguous XY planes along Z.
// Each rank stores its own planes plus one halo plane on each side, so a local
// buffer holds (localNz + 2) planes; local plane p corresponds to global plane
// zOffset + p - 1 for p in [1, localNz].
struct Decomposition {
    size_t zOffset;   // global z of first owned plane
    size_t localNz;   // number of owned planes
    int lowerRank;    // neighbor owning plane zOffset-1, or MPI_PROC_NULL
    int upperRank;    // neighbor owning plane zOffset+localNz, or MPI_PROC_NULL
};

static Decomposition decompose(const size_t nz, const int rank, const int numRanks) {
    const size_t base = nz / (size_t)numRanks;
    const size_t rem = nz % (size_t)numRanks;
    const size_t r = (size_t)rank;

    Decomposition d;
    d.localNz = base + (r < rem ? 1 : 0);
    d.zOffset = r * base + std::min(r, rem);

    // Ranks without any planes are skipped when determining the neighbors, so that
    // halo data always travels between the ranks that actually own adjacent planes.
    d.lowerRank = MPI_PROC_NULL;
    d.upperRank = MPI_PROC_NULL;
    if (d.localNz > 0) {
        for (int i = rank - 1; i >= 0; --i) {
            if (base + ((size_t)i < rem ? 1 : 0) > 0) { d.lowerRank = i; break; }
        }
        for (int i = rank + 1; i < numRanks; ++i) {
            if (base + ((size_t)i < rem ? 1 : 0) > 0) { d.upperRank = i; break; }
        }
    }
    return d;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const Decomposition& dec) {
    const size_t plane = nx * ny;
    for (size_t p = 1; p <= dec.localNz; ++p) {
        const size_t globalBase = (dec.zOffset + p - 1) * plane;
        Real* out = grid.data() + p * plane;
        for (size_t i = 0; i < plane; ++i) {
            out[i] = (Real)((globalBase + i) % 19);
        }
    }
}

// 7-point stencil for a single local plane; boundary cells are copied, matching
// the serial version's interior/boundary split.
static void stencilPlane(const Real* __restrict input, Real* __restrict output, const size_t p,
                         const size_t nx, const size_t ny, const size_t nz, const size_t globalZ) {
    const size_t plane = nx * ny;
    const Real* c = input + p * plane;
    Real* o = output + p * plane;

    if (globalZ == 0 || globalZ == nz - 1) {
        std::memcpy(o, c, plane * sizeof(Real));
        return;
    }

    const Real* down = c - plane;
    const Real* up = c + plane;

    std::memcpy(o, c, nx * sizeof(Real));  // y == 0
    for (size_t y = 1; y < ny - 1; ++y) {
        const size_t row = y * nx;
        o[row] = c[row];
        for (size_t x = 1; x < nx - 1; ++x) {
            const size_t i = row + x;
            o[i] = (c[i] + c[i - 1] + c[i + 1] + c[i - nx] + c[i + nx] + down[i] + up[i]) / 7.0;
        }
        o[row + nx - 1] = c[row + nx - 1];
    }
    std::memcpy(o + (ny - 1) * nx, c + (ny - 1) * nx, nx * sizeof(Real));  // y == ny-1
}

// One full iteration on the local slab: halo exchange overlapped with the planes
// that do not depend on halo data.
static void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                             const size_t nx, const size_t ny, const size_t nz,
                             const Decomposition& dec) {
    const size_t plane = nx * ny;
    if (dec.localNz == 0) return;

    Real* in = input.data();

    MPI_Request requests[4];
    // Receive halos into local planes 0 and localNz+1, send own planes 1 and localNz.
    MPI_Irecv(in, (int)plane, MPI_DOUBLE, dec.lowerRank, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(in + (dec.localNz + 1) * plane, (int)plane, MPI_DOUBLE, dec.upperRank, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(in + plane, (int)plane, MPI_DOUBLE, dec.lowerRank, 1, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(in + dec.localNz * plane, (int)plane, MPI_DOUBLE, dec.upperRank, 0, MPI_COMM_WORLD, &requests[3]);

    // Planes 2 .. localNz-1 only read owned data.
    for (size_t p = 2; p + 1 <= dec.localNz; ++p) {
        stencilPlane(in, output.data(), p, nx, ny, nz, dec.zOffset + p - 1);
    }

    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    stencilPlane(in, output.data(), 1, nx, ny, nz, dec.zOffset);
    if (dec.localNz > 1) {
        stencilPlane(in, output.data(), dec.localNz, nx, ny, nz, dec.zOffset + dec.localNz - 1);
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

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
    }

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;
    const Decomposition dec = decompose(nz, rank, numRanks);
    const size_t localSize = plane * (dec.localNz + 2);

    // Allocate local slabs including halo planes (double buffering)
    std::vector<Real> grid1(localSize);
    std::vector<Real> grid2(localSize);

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, dec);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, nz, dec);
        } else {
            stencilIteration(grid2, grid1, nx, ny, nz, dec);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localMs = (long)duration.count();
    long elapsedMs = localMs;
    MPI_Reduce(&localMs, &elapsedMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", elapsedMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (elapsedMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults || validate) {
        // Gather the distributed slabs into the full grid on rank 0 for output/validation
        const std::vector<Real>& localFinal = (iterations % 2 == 0) ? grid1 : grid2;
        std::vector<Real> finalGrid(rank == 0 ? gridSize : 0);
        std::vector<int> counts, displs;
        if (rank == 0) {
            counts.resize(numRanks);
            displs.resize(numRanks);
            for (int r = 0; r < numRanks; ++r) {
                const Decomposition rd = decompose(nz, r, numRanks);
                counts[r] = (int)(rd.localNz * plane);
                displs[r] = (int)(rd.zOffset * plane);
            }
        }
        MPI_Gatherv(localFinal.data() + plane, (int)(dec.localNz * plane), MPI_DOUBLE,
                    finalGrid.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        int exitCode = 0;
        if (rank == 0) {
            if (printResults) {
                print_results(finalGrid, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                if (validateResult(finalGrid, nx, ny, nz)) {
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
