#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Z-slab decomposition: rank r owns the contiguous z-planes [z0, z0 + nzl).
// Local fields hold the owned planes at padded z-indices 1..nzl, with one ghost plane
// below (index 0) and one above (index nzl + 1). Ghosts at the global boundaries mirror
// the adjacent owned plane, which reproduces the clamped boundary conditions exactly.
struct Slab {
    size_t nx = 0;
    size_t ny = 0;
    size_t plane = 0;   // nx * ny, cells per z-plane
    size_t z0 = 0;      // global index of the first owned plane
    size_t nzl = 0;     // number of owned planes (0 for ranks beyond nz)
    int lower = -1;     // neighbour rank holding the planes below, -1 at the global bottom
    int upper = -1;     // neighbour rank holding the planes above, -1 at the global top
    MPI_Datatype planeType = MPI_DATATYPE_NULL;
};

// Owned z-range of rank r. Ranks beyond nz own nothing and stay idle.
void slabRange(const size_t nz, const int size, const int r, size_t& z0, size_t& nzl) {
    const size_t active = std::min(static_cast<size_t>(size), nz);
    const size_t ur = static_cast<size_t>(r);
    if (ur >= active) {
        z0 = 0;
        nzl = 0;
        return;
    }
    const size_t base = nz / active;
    const size_t rem = nz % active;
    nzl = base + (ur < rem ? 1 : 0);
    z0 = ur * base + std::min(ur, rem);
}

Slab makeSlab(const size_t nx, const size_t ny, const size_t nz, const int rank, const int size) {
    Slab s;
    s.nx = nx;
    s.ny = ny;
    s.plane = nx * ny;
    slabRange(nz, size, rank, s.z0, s.nzl);
    if (s.nzl > 0) {
        s.lower = rank > 0 ? rank - 1 : -1;
        s.upper = (s.z0 + s.nzl < nz) ? rank + 1 : -1;
    }
    MPI_Type_contiguous(static_cast<int>(s.plane), MPI_DOUBLE, &s.planeType);
    MPI_Type_commit(&s.planeType);
    return s;
}

// Compute Laplacian with clamped boundary conditions.
// Ghost planes supply the z-neighbours, so no z-clamping is needed here.
double computeLaplacian(const double* c, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double center = c[idx3(x, y, z, nx, ny)];
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * center) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * center) / (dy * dy);
    const double czz = (c[idx3(x, y, z + 1, nx, ny)] + c[idx3(x, y, z - 1, nx, ny)] - 2.0 * center) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential on padded planes zlo..zhi
void computeChemicalPotential(const double* c, double* mu, const Slab& s, const size_t zlo, const size_t zhi,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = zlo; z <= zhi; ++z) {
        for (size_t y = 0; y < s.ny; ++y) {
            for (size_t x = 0; x < s.nx; ++x) {
                const size_t idx = idx3(x, y, z, s.nx, s.ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, s.nx, s.ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update on padded planes zlo..zhi
void cahnHilliardUpdate(double* cnew, const double* cold, const double* mu, const Slab& s,
                        const size_t zlo, const size_t zhi,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = zlo; z <= zhi; ++z) {
        for (size_t y = 0; y < s.ny; ++y) {
            for (size_t x = 0; x < s.nx; ++x) {
                const size_t idx = idx3(x, y, z, s.nx, s.ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, s.nx, s.ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field on the owned planes
void initializeConcentration(double* c, const Slab& s, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    for (size_t z = 1; z <= s.nzl; ++z) {
        const size_t zg = s.z0 + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1] from the global linear index
                const size_t linear_id = zg * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Ghost planes that lie on the global boundary mirror the adjacent owned plane (clamped BC)
void clampPhysicalGhosts(double* f, const Slab& s) {
    if (s.nzl == 0) return;
    if (s.lower < 0) {
        std::copy(f + s.plane, f + 2 * s.plane, f);
    }
    if (s.upper < 0) {
        std::copy(f + s.nzl * s.plane, f + (s.nzl + 1) * s.plane, f + (s.nzl + 1) * s.plane);
    }
}

// Non-blocking exchange of the boundary planes of a field with its z-neighbours.
// Owned boundary planes are sent, neighbouring planes are received into the ghosts.
// stage selects a distinct tag pair so that concurrent exchanges never mix.
struct Halo {
    MPI_Request req[4];
    int count = 0;
};

void startHalo(double* f, const Slab& s, const int stage, Halo& h) {
    const int toUp = 2 * stage;
    const int toDown = 2 * stage + 1;
    h.count = 0;
    if (s.lower >= 0) {
        MPI_Irecv(f, 1, s.planeType, s.lower, toUp, MPI_COMM_WORLD, &h.req[h.count++]);
        MPI_Isend(f + s.plane, 1, s.planeType, s.lower, toDown, MPI_COMM_WORLD, &h.req[h.count++]);
    }
    if (s.upper >= 0) {
        MPI_Irecv(f + (s.nzl + 1) * s.plane, 1, s.planeType, s.upper, toDown, MPI_COMM_WORLD, &h.req[h.count++]);
        MPI_Isend(f + s.nzl * s.plane, 1, s.planeType, s.upper, toUp, MPI_COMM_WORLD, &h.req[h.count++]);
    }
}

void finishHalo(Halo& h) {
    MPI_Waitall(h.count, h.req, MPI_STATUSES_IGNORE);
}

// Global min/max and NaN/Inf check over the distributed field
bool validateResult(const double* c, const Slab& s, const int rank) {
    int localBad = 0;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    for (size_t z = 1; z <= s.nzl; ++z) {
        for (size_t i = 0; i < s.plane; ++i) {
            const double val = c[z * s.plane + i];
            if (std::isnan(val) || std::isinf(val)) {
                localBad = 1;
            }
            localMin = std::min(localMin, val);
            localMax = std::max(localMax, val);
        }
    }

    // Check for NaN or Inf
    int anyBad = 0;
    MPI_Allreduce(&localBad, &anyBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anyBad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double minVal = 0.0;
    double maxVal = 0.0;
    MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
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

int runBenchmark(int argc, char** argv, const int rank, const int size) {
    const bool root = rank == 0;

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

    if (root) {
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

    Slab s = makeSlab(nx, ny, nz, rank, size);

    // Allocate arrays: owned planes plus two ghost planes
    const size_t paddedSize = (s.nzl + 2) * s.plane;
    std::vector<double> cold(paddedSize);
    std::vector<double> cnew(paddedSize);
    std::vector<double> mu(paddedSize);

    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    initializeConcentration(cold.data(), s, nx, ny, nz);

    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    if (s.nzl > 0) {
        Halo halo;
        const size_t last = s.nzl;
        for (int t = 0; t < iterations; ++t) {
            // Chemical potential: interior planes need no ghost data, so overlap with the c exchange
            clampPhysicalGhosts(cold.data(), s);
            startHalo(cold.data(), s, 0, halo);
            computeChemicalPotential(cold.data(), mu.data(), s, 2, last - 1, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            finishHalo(halo);
            computeChemicalPotential(cold.data(), mu.data(), s, 1, 1, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            if (last > 1) {
                computeChemicalPotential(cold.data(), mu.data(), s, last, last, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }

            // Update concentration: same overlap pattern for the mu exchange
            clampPhysicalGhosts(mu.data(), s);
            startHalo(mu.data(), s, 1, halo);
            cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), s, 2, last - 1, D, dt, dx, dy, dz);
            finishHalo(halo);
            cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), s, 1, 1, D, dt, dx, dy, dz);
            if (last > 1) {
                cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), s, last, last, D, dt, dx, dy, dz);
            }

            // Swap buffers
            std::swap(cold, cnew);
        }
    }

    const double local = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        const long durationMs = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        const double cellUpdates = (double)gridSize * iterations;
        const double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation (gathered in global z-order on the root)
    if (printResults) {
        std::vector<int> counts(size, 0);
        std::vector<int> displs(size, 0);
        for (int r = 0; r < size; ++r) {
            size_t z0 = 0;
            size_t nzl = 0;
            slabRange(nz, size, r, z0, nzl);
            counts[r] = static_cast<int>(nzl);
            displs[r] = static_cast<int>(z0);
        }
        std::vector<double> global(root ? gridSize : 0);
        MPI_Gatherv(cold.data() + s.plane, static_cast<int>(s.nzl), s.planeType,
                    global.data(), counts.data(), displs.data(), s.planeType, 0, MPI_COMM_WORLD);
        if (root) print_results(global, "Concentration");
    }

    int status = 0;

    // Validation
    if (validate) {
        if (root) printf("Validating result...\n");
        const bool valid = validateResult(cold.data(), s, rank);

        if (valid) {
            if (root) printf("Validation: PASSED\n");
        } else {
            if (root) printf("Validation: FAILED\n");
            status = 1;
        }
    }

    MPI_Type_free(&s.planeType);
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
