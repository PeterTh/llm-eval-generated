#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// The z-planes are decomposed into contiguous slabs, one per MPI rank.
// Each local array is padded with one ghost plane on either side:
// planes 1..nzl are owned, planes 0 and nzl+1 hold copies of the neighbouring ranks' planes.
// At the global domain edges the ghost plane is a copy of the adjacent owned plane,
// which reproduces the clamped boundary condition of the serial code.

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// z-slab owned by one rank
struct Slab {
    size_t z0;   // global index of the first owned z-plane
    size_t nzl;  // number of owned z-planes
    int down;    // rank holding the planes below, or MPI_PROC_NULL at the domain edge
    int up;      // rank holding the planes above, or MPI_PROC_NULL at the domain edge
};

Slab decompose(const size_t nz, const int rank, const int size) {
    const size_t base = nz / size;
    const size_t rem = nz % size;
    Slab s;
    s.nzl = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    s.z0 = base * rank + std::min(static_cast<size_t>(rank), rem);
    s.down = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    s.up = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;
    return s;
}

// Compute Laplacian with clamped boundary conditions in x and y.
// z neighbours are always present in the padded array (ghost planes).
double computeLaplacian(const double* f, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double cxx = (f[idx3(xp, y, z, nx, ny)] + f[idx3(xn, y, z, nx, ny)] -
                  2.0 * f[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (f[idx3(x, yp, z, nx, ny)] + f[idx3(x, yn, z, nx, ny)] -
                  2.0 * f[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (f[idx3(x, y, z + 1, nx, ny)] + f[idx3(x, y, z - 1, nx, ny)] -
                  2.0 * f[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential on padded z-planes [zBegin, zEnd)
void computeChemicalPotential(const double* c, double* mu,
                              const size_t nx, const size_t ny,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const size_t zBegin, const size_t zEnd) {
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step on padded z-planes [zBegin, zEnd)
void cahnHilliardUpdate(double* cnew, const double* cold, const double* mu,
                        const size_t nx, const size_t ny,
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        const size_t zBegin, const size_t zEnd) {
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Copy the adjacent owned plane into the ghost planes that lie outside the global domain
void fillDomainEdgeGhosts(double* f, const size_t plane, const Slab& s) {
    if (s.down == MPI_PROC_NULL) {
        std::copy_n(f + plane, plane, f);
    }
    if (s.up == MPI_PROC_NULL) {
        std::copy_n(f + s.nzl * plane, plane, f + (s.nzl + 1) * plane);
    }
}

// Start non-blocking exchange of the owned boundary planes with the z-neighbours.
// Requests towards MPI_PROC_NULL complete immediately.
void startHaloExchange(double* f, const size_t plane, const Slab& s, const MPI_Datatype planeType,
                       const int tag, MPI_Request* req) {
    MPI_Irecv(f, 1, planeType, s.down, tag, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(f + (s.nzl + 1) * plane, 1, planeType, s.up, tag, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(f + plane, 1, planeType, s.down, tag, MPI_COMM_WORLD, &req[2]);
    MPI_Isend(f + s.nzl * plane, 1, planeType, s.up, tag, MPI_COMM_WORLD, &req[3]);
}

// Initialize concentration field (owned planes only)
void initializeConcentration(double* c, const size_t nx, const size_t ny, const size_t nz, const Slab& s) {
    const size_t vol = nx * ny * nz;

    for (size_t zl = 1; zl <= s.nzl; ++zl) {
        const size_t z = s.z0 + zl - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, zl, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const double* c, const size_t plane, const Slab& s, const int rank) {
    // Check for NaN or Inf over the owned planes
    int bad = 0;
    double minVal = INFINITY;
    double maxVal = -INFINITY;
    for (size_t i = plane; i < (s.nzl + 1) * plane; ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            bad = 1;
        }
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }

    int anyBad = 0;
    MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anyBad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double globalMin = 0.0;
    double globalMax = 0.0;
    MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);

    // Values should generally stay within reasonable bounds
    if (globalMax > 10.0 || globalMin < -10.0) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Every rank must own at least one z-plane
    if (nz < static_cast<size_t>(size)) {
        if (rank == 0) printf("Grid depth (%zu) must be at least the number of MPI ranks (%d)\n", nz, size);
        return 1;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;
    const Slab slab = decompose(nz, rank, size);

    // Datatype for one z-plane; the slab is contiguous in memory
    MPI_Datatype planeType;
    MPI_Type_contiguous(static_cast<int>(plane), MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    // Allocate padded arrays (owned planes + 2 ghost planes)
    const size_t paddedSize = (slab.nzl + 2) * plane;
    std::vector<double> cold(paddedSize);
    std::vector<double> cnew(paddedSize);
    std::vector<double> mu(paddedSize);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold.data(), nx, ny, nz, slab);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Request reqC[4];
    MPI_Request reqMu[4];
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Exchange concentration halos; interior chemical potential planes need no ghosts
        startHaloExchange(cold.data(), plane, slab, planeType, 0, reqC);
        fillDomainEdgeGhosts(cold.data(), plane, slab);
        computeChemicalPotential(cold.data(), mu.data(), nx, ny, dx, dy, dz,
                                 gamma, e_AA, e_BB, e_AB, 2, slab.nzl);
        MPI_Waitall(4, reqC, MPI_STATUSES_IGNORE);

        // Boundary planes of chemical potential depend on the concentration halos
        computeChemicalPotential(cold.data(), mu.data(), nx, ny, dx, dy, dz,
                                 gamma, e_AA, e_BB, e_AB, 1, 2);
        if (slab.nzl > 1) {
            computeChemicalPotential(cold.data(), mu.data(), nx, ny, dx, dy, dz,
                                     gamma, e_AA, e_BB, e_AB, slab.nzl, slab.nzl + 1);
        }

        // Exchange chemical potential halos; interior concentration planes need no ghosts
        fillDomainEdgeGhosts(mu.data(), plane, slab);
        startHaloExchange(mu.data(), plane, slab, planeType, 1, reqMu);
        cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), nx, ny, D, dt, dx, dy, dz, 2, slab.nzl);
        MPI_Waitall(4, reqMu, MPI_STATUSES_IGNORE);

        // Boundary planes of the update depend on the chemical potential halos
        cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), nx, ny, D, dt, dx, dy, dz, 1, 2);
        if (slab.nzl > 1) {
            cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), nx, ny, D, dt, dx, dy, dz,
                               slab.nzl, slab.nzl + 1);
        }

        // Swap buffers
        std::swap(cold, cnew);
    }

    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long durationMs = static_cast<long>(maxElapsed * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        const double cellUpdates = (double)gridSize * iterations;
        const double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full field on rank 0 for external validation
    if (printResults) {
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            const Slab sr = decompose(nz, r, size);
            counts[r] = static_cast<int>(sr.nzl);
            displs[r] = static_cast<int>(sr.z0);
        }
        std::vector<double> full(rank == 0 ? gridSize : 0);
        MPI_Gatherv(cold.data() + plane, static_cast<int>(slab.nzl), planeType,
                    full.data(), counts.data(), displs.data(), planeType, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(full, "Concentration");
        }
    }

    // Validation
    int rc = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(cold.data(), plane, slab, rank);

        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            rc = 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    MPI_Type_free(&planeType);
    return rc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = runBenchmark(argc, argv);
    MPI_Finalize();
    return rc;
}
