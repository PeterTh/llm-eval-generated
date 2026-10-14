#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Contiguous range of global z-planes owned by one rank
struct Slab {
    size_t z0;     // first owned global z-plane
    size_t count;  // number of owned z-planes
};

// Block-partition nz z-planes over the first `active` ranks; the rest own nothing
Slab slabFor(const int rank, const int active, const size_t nz) {
    if (rank >= active) return {0, 0};
    const size_t r = rank;
    const size_t p = active;
    const size_t z0 = r * nz / p;
    return {z0, (r + 1) * nz / p - z0};
}

// Fill owned planes of the local buffer (local plane 0 and count+1 are ghosts)
void initializeSlab(Real* local, const size_t nx, const size_t ny, const Slab& slab) {
    const size_t plane = nx * ny;
    for (size_t l = 0; l < slab.count; ++l) {
        for (size_t i = 0; i < plane; ++i) {
            const size_t idx = (slab.z0 + l) * plane + i;
            local[(l + 1) * plane + i] = (idx % 19) * 1.0;
        }
    }
}

// 7-point stencil for one z-plane. below/above are the neighbouring planes.
// Faces of the global cube are copied unchanged, matching the serial kernel.
void stencilPlane(const Real* __restrict below, const Real* __restrict cur, const Real* __restrict above,
                  Real* __restrict out, const size_t nx, const size_t ny, const bool zBoundary) {
    const size_t plane = nx * ny;
    if (zBoundary || nx < 3 || ny < 3) {
        std::copy(cur, cur + plane, out);
        return;
    }

    // First and last rows are on the y-boundary
    std::copy(cur, cur + nx, out);
    std::copy(cur + (ny - 1) * nx, cur + plane, out + (ny - 1) * nx);

    for (size_t y = 1; y < ny - 1; ++y) {
        const size_t row = y * nx;
        out[row] = cur[row];
        out[row + nx - 1] = cur[row + nx - 1];
        for (size_t x = 1; x < nx - 1; ++x) {
            const size_t i = row + x;
            // Simple averaging stencil
            out[i] = (cur[i] + cur[i - 1] + cur[i + 1] + cur[i - nx] + cur[i + nx] + below[i] + above[i]) / 7.0;
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

int runBenchmark(int argc, char** argv, const int rank, const int size) {
    const bool root = rank == 0;

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
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (nx == 0 || ny == 0 || nz == 0) {
        if (root) printf("Grid dimensions must be positive\n");
        return 1;
    }

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;

    // Ranks beyond nz own no planes and only take part in the collectives
    const int active = static_cast<int>(std::min<size_t>(static_cast<size_t>(size), nz));
    const Slab slab = slabFor(rank, active, nz);
    const int lower = (rank > 0 && rank < active) ? rank - 1 : MPI_PROC_NULL;
    const int upper = (rank + 1 < active) ? rank + 1 : MPI_PROC_NULL;

    // Contiguous block of z-planes as an MPI datatype, so counts stay small for huge grids
    MPI_Datatype planeType;
    MPI_Type_contiguous(static_cast<int>(plane), MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    // Double buffering on local slabs, each with one ghost plane on either side
    std::vector<Real> bufA((slab.count + 2) * plane, 0.0);
    std::vector<Real> bufB((slab.count + 2) * plane, 0.0);
    Real* in = bufA.data();
    Real* out = bufB.data();

    // Initialize
    if (root) printf("Initializing grid...\n");
    initializeSlab(in, nx, ny, slab);

    // Run stencil iterations
    if (root) printf("Running stencil computation...\n");

    const auto updatePlane = [&](const Real* src, Real* dst, const size_t l) {
        const size_t z = slab.z0 + l - 1;
        stencilPlane(src + (l - 1) * plane, src + l * plane, src + (l + 1) * plane, dst + l * plane,
                     nx, ny, z == 0 || z == nz - 1);
    };

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (slab.count > 0) {
            // Exchange one-plane halos of the input with neighbours
            MPI_Request reqs[4];
            MPI_Irecv(in, 1, planeType, lower, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(in + (slab.count + 1) * plane, 1, planeType, upper, 1, MPI_COMM_WORLD, &reqs[1]);
            MPI_Isend(in + plane, 1, planeType, lower, 1, MPI_COMM_WORLD, &reqs[2]);
            MPI_Isend(in + slab.count * plane, 1, planeType, upper, 0, MPI_COMM_WORLD, &reqs[3]);

            // Interior planes need no halo data, so compute them while messages are in flight
            for (size_t l = 2; l < slab.count; ++l) {
                updatePlane(in, out, l);
            }

            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

            // Edge planes depend on the received halos
            updatePlane(in, out, 1);
            if (slab.count > 1) {
                updatePlane(in, out, slab.count);
            }
        }
        std::swap(in, out);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // The slowest rank determines the run time
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(maxSeconds));
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full final grid on root (same buffer choice as the serial version)
    const Real* localFinal = (iterations % 2 == 0) ? bufA.data() : bufB.data();
    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    for (int r = 0; r < size; ++r) {
        const Slab s = slabFor(r, active, nz);
        counts[r] = static_cast<int>(s.count);
        displs[r] = static_cast<int>(s.z0);
    }
    std::vector<Real> finalGrid(root ? gridSize : 0);
    MPI_Gatherv(localFinal + plane, static_cast<int>(slab.count), planeType,
                finalGrid.data(), counts.data(), displs.data(), planeType, 0, MPI_COMM_WORLD);

    int status = 0;

    // Print results for external validation
    if (root && printResults) {
        print_results(finalGrid, "Grid");
    }

    // Validation
    if (root && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&planeType);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int status = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return status;
}
