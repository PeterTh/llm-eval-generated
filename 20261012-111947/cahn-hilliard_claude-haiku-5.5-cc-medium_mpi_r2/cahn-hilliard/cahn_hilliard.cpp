#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Z-slab domain decomposition. Rank r owns the global planes [z0, z0 + nzLocal).
// Local storage holds the owned planes at local indices 1..nzLocal, with one ghost
// plane below (index 0) and one above (index nzLocal + 1).
struct Slab {
    size_t nx;
    size_t ny;
    size_t nz;
    size_t planeSize;   // nx * ny
    size_t z0;          // first owned global plane
    size_t nzLocal;     // number of owned planes
    int lower;          // rank owning the planes below, or MPI_PROC_NULL
    int upper;          // rank owning the planes above, or MPI_PROC_NULL
};

// Block distribution of nz planes over size ranks: the first (nz % size) ranks get one extra plane
size_t planesOwned(const size_t nz, const int size, const int r) {
    const size_t q = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    return q + (static_cast<size_t>(r) < rem ? 1 : 0);
}

size_t firstPlane(const size_t nz, const int size, const int r) {
    const size_t q = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t rr = static_cast<size_t>(r);
    return rr * q + std::min(rr, rem);
}

Slab makeSlab(const size_t nx, const size_t ny, const size_t nz, const int rank, const int size) {
    Slab s{};
    s.nx = nx;
    s.ny = ny;
    s.nz = nz;
    s.planeSize = nx * ny;
    s.z0 = firstPlane(nz, size, rank);
    s.nzLocal = planesOwned(nz, size, rank);
    // Only a trailing run of ranks can be empty, so a neighbour exists only if it owns planes too
    s.lower = (s.nzLocal > 0 && rank > 0 && planesOwned(nz, size, rank - 1) > 0) ? rank - 1 : MPI_PROC_NULL;
    s.upper = (s.nzLocal > 0 && rank + 1 < size && planesOwned(nz, size, rank + 1) > 0) ? rank + 1 : MPI_PROC_NULL;
    return s;
}

// Posts the ghost-plane exchange for a field. Ghost planes on the global z-boundary
// replicate the edge plane, which is exactly the clamped boundary condition.
// Completion is signalled through req[0..3]; wait on them before reading ghosts.
void startHaloExchange(std::vector<double>& f, const Slab& s, MPI_Datatype planeType, MPI_Comm comm, MPI_Request req[4]) {
    for (int i = 0; i < 4; ++i) req[i] = MPI_REQUEST_NULL;
    if (s.nzLocal == 0) return;

    double* base = f.data();
    const size_t top = s.nzLocal;
    if (s.lower == MPI_PROC_NULL) {
        std::memcpy(base, base + s.planeSize, s.planeSize * sizeof(double));
    }
    if (s.upper == MPI_PROC_NULL) {
        std::memcpy(base + (top + 1) * s.planeSize, base + top * s.planeSize, s.planeSize * sizeof(double));
    }

    // Tag 0 carries data sent upwards, tag 1 data sent downwards
    MPI_Irecv(base, 1, planeType, s.lower, 0, comm, &req[0]);
    MPI_Irecv(base + (top + 1) * s.planeSize, 1, planeType, s.upper, 1, comm, &req[1]);
    MPI_Isend(base + s.planeSize, 1, planeType, s.lower, 1, comm, &req[2]);
    MPI_Isend(base + top * s.planeSize, 1, planeType, s.upper, 0, comm, &req[3]);
}

// Laplacian with clamped boundary conditions in x and y. The z neighbours are always
// present in local storage (ghost planes), so z is a local plane index.
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, z + 1, nx, ny)] + c[idx3(x, y, z - 1, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential on local planes [zBegin, zEnd)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu, const Slab& s,
                              const size_t zBegin, const size_t zEnd,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = zBegin; z < zEnd; ++z) {
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

// Cahn-Hilliard update step on local planes [zBegin, zEnd)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu, const Slab& s,
                        const size_t zBegin, const size_t zEnd,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = zBegin; z < zEnd; ++z) {
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
void initializeConcentration(std::vector<double>& c, const Slab& s) {
    const size_t vol = s.nx * s.ny * s.nz;

    for (size_t zl = 1; zl <= s.nzLocal; ++zl) {
        const size_t zg = s.z0 + zl - 1;
        for (size_t y = 0; y < s.ny; ++y) {
            for (size_t x = 0; x < s.nx; ++x) {
                const size_t idx = idx3(x, y, zl, s.nx, s.ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = zg * (s.nx * s.ny) + y * s.nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Gathers the owned planes of every rank into the full field on rank 0 (empty elsewhere)
std::vector<double> gatherField(const std::vector<double>& f, const Slab& s, int rank, int size,
                                MPI_Datatype planeType, MPI_Comm comm) {
    std::vector<double> full;
    std::vector<int> counts;
    std::vector<int> displs;
    if (rank == 0) {
        full.resize(s.nx * s.ny * s.nz);
        counts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = static_cast<int>(planesOwned(s.nz, size, r));
            displs[r] = static_cast<int>(firstPlane(s.nz, size, r));
        }
    }
    MPI_Gatherv(f.data() + s.planeSize, static_cast<int>(s.nzLocal), planeType,
                full.data(), counts.data(), displs.data(), planeType, 0, comm);
    return full;
}

bool validateResult(const std::vector<double>& c, const Slab& s, int rank, MPI_Comm comm) {
    // Check for NaN or Inf and find the local range over the owned planes
    long localBad = 0;
    double localMin = INFINITY;
    double localMax = -INFINITY;
    for (size_t i = s.planeSize; i < (s.nzLocal + 1) * s.planeSize; ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) {
            ++localBad;
        }
        localMin = std::min(localMin, val);
        localMax = std::max(localMax, val);
    }

    long totalBad = 0;
    MPI_Allreduce(&localBad, &totalBad, 1, MPI_LONG, MPI_SUM, comm);
    if (totalBad > 0) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double minVal = 0.0;
    double maxVal = 0.0;
    MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);

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

int runBenchmark(int argc, char** argv, int rank, int size) {
    const bool root = rank == 0;

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks take the same path)
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

    const Slab s = makeSlab(nx, ny, nz, rank, size);
    const size_t gridSize = nx * ny * nz;

    MPI_Datatype planeType;
    MPI_Type_contiguous(static_cast<int>(s.planeSize), MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    // Allocate arrays: owned planes plus one ghost plane on each side
    const size_t localSize = (s.nzLocal + 2) * s.planeSize;
    std::vector<double> cold(localSize);
    std::vector<double> cnew(localSize);
    std::vector<double> mu(localSize);

    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    initializeConcentration(cold, s);

    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Request req[4];
    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential; it needs ghost planes of cold.
        // Interior planes are computed while the halo messages are in flight.
        startHaloExchange(cold, s, planeType, comm, req);
        if (s.nzLocal > 0) {
            computeChemicalPotential(cold, mu, s, 2, s.nzLocal, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        }
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        if (s.nzLocal > 0) {
            computeChemicalPotential(cold, mu, s, 1, 2, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            if (s.nzLocal > 1) {
                computeChemicalPotential(cold, mu, s, s.nzLocal, s.nzLocal + 1, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }
        }

        // Update concentration; it needs ghost planes of mu
        startHaloExchange(mu, s, planeType, comm, req);
        if (s.nzLocal > 0) {
            cahnHilliardUpdate(cnew, cold, mu, s, 2, s.nzLocal, D, dt, dx, dy, dz);
        }
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        if (s.nzLocal > 0) {
            cahnHilliardUpdate(cnew, cold, mu, s, 1, 2, D, dt, dx, dy, dz);
            if (s.nzLocal > 1) {
                cahnHilliardUpdate(cnew, cold, mu, s, s.nzLocal, s.nzLocal + 1, D, dt, dx, dy, dz);
            }
        }

        // Swap buffers
        std::swap(cold, cnew);
    }

    // The slowest rank determines the run time
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (root) {
        const long durationMs = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    if (printResults) {
        const std::vector<double> full = gatherField(cold, s, rank, size, planeType, comm);
        if (root) print_results(full, "Concentration");
    }

    int exitCode = 0;
    // Validation
    if (validate) {
        if (root) printf("Validating result...\n");
        const bool valid = validateResult(cold, s, rank, comm);

        if (valid) {
            if (root) printf("Validation: PASSED\n");
        } else {
            if (root) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Type_free(&planeType);
    return exitCode;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int exitCode = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return exitCode;
}
