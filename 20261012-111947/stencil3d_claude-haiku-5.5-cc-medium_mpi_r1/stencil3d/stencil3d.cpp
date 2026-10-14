#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// Message tags for halo exchange
constexpr int kTagToUpper = 0;  // message travelling to rank + 1
constexpr int kTagToLower = 1;  // message travelling to rank - 1

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Contiguous block of global z-planes owned by one rank
struct Slab {
    size_t z0;      // first owned global z-plane
    size_t nLocal;  // number of owned z-planes (zero for surplus ranks)
    size_t active;  // number of ranks that own at least one plane
};

Slab makeSlab(const size_t rank, const size_t size, const size_t nz) {
    const size_t active = std::max<size_t>(1, std::min(size, nz));
    if (rank >= active) {
        return {nz, 0, active};
    }
    const size_t base = nz / active;
    const size_t rem = nz % active;
    return {rank * base + std::min(rank, rem), base + (rank < rem ? 1 : 0), active};
}

// Local storage holds the owned planes at local indices [1, nLocal],
// with one halo plane below (index 0) and above (index nLocal + 1).
void initializeSlab(std::vector<Real>& grid, const size_t nx, const size_t ny, const Slab& slab) {
    for (size_t lz = 1; lz <= slab.nLocal; ++lz) {
        const size_t z = slab.z0 + lz - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx3(x, y, lz, nx, ny)] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil for one global z-plane. `in` and `out` point at the first
// element of that plane; the planes at z - 1 and z + 1 must be addressable
// (halo planes supply them across rank boundaries).
void updatePlane(const Real* in, Real* out, const size_t nx, const size_t ny, const size_t z, const size_t nz) {
    const size_t plane = nx * ny;
    if (z == 0 || z == nz - 1) {
        // Global boundary plane: copied unchanged
        std::copy(in, in + plane, out);
        return;
    }

    // Boundary rows and columns of an interior plane are copied unchanged
    std::copy(in, in + nx, out);
    std::copy(in + (ny - 1) * nx, in + plane, out + (ny - 1) * nx);
    for (size_t y = 1; y + 1 < ny; ++y) {
        out[y * nx] = in[y * nx];
        out[y * nx + nx - 1] = in[y * nx + nx - 1];
    }

    // Interior points (same summation order as the original kernel)
    for (size_t y = 1; y + 1 < ny; ++y) {
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t idx = y * nx + x;

            const Real center = in[idx];
            const Real left = in[idx - 1];
            const Real right = in[idx + 1];
            const Real front = in[idx - nx];
            const Real back = in[idx + nx];
            const Real bottom = in[idx - plane];
            const Real top = in[idx + plane];

            // Simple averaging stencil
            out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
        }
    }
}

// One sweep over this rank's slab. Halo planes of `in` are refreshed from the
// neighbouring ranks first; interior planes are computed while those messages
// are in flight.
void stencilIteration(Real* in, Real* out, const size_t nx, const size_t ny, const size_t nz,
                      const Slab& slab, const int rank) {
    if (slab.nLocal == 0) {
        return;
    }

    const size_t plane = nx * ny;
    const int planeCount = static_cast<int>(plane);
    const size_t last = slab.nLocal;  // local index of the last owned plane
    const bool hasLower = rank > 0;
    const bool hasUpper = static_cast<size_t>(rank) + 1 < slab.active;

    MPI_Request reqs[4];
    int nReq = 0;
    if (hasLower) {
        MPI_Irecv(in, planeCount, MPI_DOUBLE, rank - 1, kTagToUpper, MPI_COMM_WORLD, &reqs[nReq++]);
        MPI_Isend(in + plane, planeCount, MPI_DOUBLE, rank - 1, kTagToLower, MPI_COMM_WORLD, &reqs[nReq++]);
    }
    if (hasUpper) {
        MPI_Irecv(in + (last + 1) * plane, planeCount, MPI_DOUBLE, rank + 1, kTagToLower, MPI_COMM_WORLD, &reqs[nReq++]);
        MPI_Isend(in + last * plane, planeCount, MPI_DOUBLE, rank + 1, kTagToUpper, MPI_COMM_WORLD, &reqs[nReq++]);
    }

    // Planes 2 .. last-1 only read owned data
    for (size_t lz = 2; lz < last; ++lz) {
        updatePlane(in + lz * plane, out + lz * plane, nx, ny, slab.z0 + lz - 1, nz);
    }

    MPI_Waitall(nReq, reqs, MPI_STATUSES_IGNORE);

    // Planes that need the halos
    updatePlane(in + plane, out + plane, nx, ny, slab.z0, nz);
    if (last > 1) {
        updatePlane(in + last * plane, out + last * plane, nx, ny, slab.z0 + last - 1, nz);
    }
}

// Global min/max and NaN/Inf check over all ranks' owned values
bool validateResult(const Real* owned, const size_t count, const bool root) {
    // 1. No NaN or Inf values
    long long bad = 0;
    Real minVal = std::numeric_limits<Real>::infinity();
    Real maxVal = -std::numeric_limits<Real>::infinity();
    for (size_t i = 0; i < count; ++i) {
        const Real val = owned[i];
        if (!std::isfinite(val)) {
            ++bad;
        } else {
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
    }

    long long badTotal = 0;
    MPI_Allreduce(&bad, &badTotal, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (badTotal > 0) {
        if (root) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    // 2. Values should be reasonable (bounded)
    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (root) {
        printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    }

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        if (root) {
            printf("Validation failed: values out of expected range\n");
        }
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

int run(int argc, char** argv, const int rank, const int size) {
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
            if (root) {
                printUsage(argv[0]);
            }
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

    if (nx == 0) {
        if (root) {
            printf("Grid size must be greater than zero\n");
        }
        return 1;
    }

    const size_t plane = nx * ny;
    if (plane > INT_MAX || nz > INT_MAX) {
        if (root) {
            printf("Grid plane (nx * ny) and nz must each be at most %d\n", INT_MAX);
        }
        return 1;
    }

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t gridSize = nx * ny * nz;
    const Slab slab = makeSlab(static_cast<size_t>(rank), static_cast<size_t>(size), nz);

    // Local blocks (double buffering), including halo planes
    std::vector<Real> grid1((slab.nLocal + 2) * plane);
    std::vector<Real> grid2((slab.nLocal + 2) * plane);

    // Initialize
    if (root) {
        printf("Initializing grid...\n");
    }
    initializeSlab(grid1, nx, ny, slab);

    // Run stencil iterations
    if (root) {
        printf("Running stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1.data(), grid2.data(), nx, ny, nz, slab, rank);
        } else {
            stencilIteration(grid2.data(), grid1.data(), nx, ny, nz, slab, rank);
        }
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;  // Slowest rank determines the wall time
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long durationMs = static_cast<long>(elapsed * 1000.0);

    if (root) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Owned planes of the final grid (skipping the lower halo plane)
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    const Real* finalOwned = finalGrid.data() + plane;
    const size_t ownedCount = slab.nLocal * plane;

    // Print results for external validation
    if (printResults) {
        // Gather the full grid in global z-order on rank 0
        MPI_Datatype planeType;
        MPI_Type_contiguous(static_cast<int>(plane), MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);

        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            const Slab s = makeSlab(static_cast<size_t>(r), static_cast<size_t>(size), nz);
            counts[r] = static_cast<int>(s.nLocal);
            displs[r] = static_cast<int>(s.z0);
        }

        std::vector<Real> fullGrid(root ? gridSize : 0);
        MPI_Gatherv(finalOwned, static_cast<int>(slab.nLocal), planeType,
                    fullGrid.data(), counts.data(), displs.data(), planeType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&planeType);

        if (root) {
            print_results(fullGrid, "Grid");
        }
    }

    // Validation
    if (validate) {
        if (root) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(finalOwned, ownedCount, root);

        if (root) {
            printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int status = run(argc, argv, rank, size);

    MPI_Finalize();
    return status;
}
