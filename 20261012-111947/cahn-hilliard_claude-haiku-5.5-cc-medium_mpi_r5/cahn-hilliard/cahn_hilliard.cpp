#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Physical and discretisation constants plus the in-plane geometry
struct Model {
    size_t nx, ny;
    double dx2, dy2, dz2;
    double gamma, e_AA, e_BB, e_AB;
    double D, dt;
};

// Message tags: data travelling to a lower rank, and data travelling to a higher rank
constexpr int TAG_DOWN = 0;
constexpr int TAG_UP = 1;

// Global z-plane range [start, start + count) owned by rank r out of P ranks
inline size_t planeStart(const int r, const int P, const size_t nz) {
    const size_t base = nz / P;
    const size_t rem = nz % P;
    return r * base + std::min(static_cast<size_t>(r), rem);
}

inline size_t planeCount(const int r, const int P, const size_t nz) {
    return nz / P + (static_cast<size_t>(r) < nz % P ? 1 : 0);
}

// Laplacian of one row of cells. The x neighbours are clamped at the grid edge as in the
// serial code; the y and z neighbour rows are supplied already clamped (z ghost planes are
// copies of the boundary plane at the global domain edge).
void laplacianRow(const double* rc, const double* rym, const double* ryp,
                  const double* rzm, const double* rzp, double* lap, const size_t nx,
                  const double dx2, const double dy2, const double dz2) {
    auto cell = [&](const size_t x, const size_t xp, const size_t xn) {
        const double cxx = (rc[xp] + rc[xn] - 2.0 * rc[x]) / dx2;
        const double cyy = (ryp[x] + rym[x] - 2.0 * rc[x]) / dy2;
        const double czz = (rzp[x] + rzm[x] - 2.0 * rc[x]) / dz2;
        return cxx + cyy + czz;
    };

    lap[0] = cell(0, nx > 1 ? 1 : 0, 0);
    // Branch-free interior so the compiler can vectorise it
    for (size_t x = 1; x + 1 < nx; ++x) {
        lap[x] = cell(x, x + 1, x - 1);
    }
    if (nx > 1) {
        lap[nx - 1] = cell(nx - 1, nx - 1, nx - 2);
    }
}

// Chemical potential on local plane p (ghost planes p-1 and p+1 of c must be valid)
void computeChemicalPotentialPlane(const Model& m, const double* c, double* mu, double* lap, const size_t p) {
    const size_t S = m.nx * m.ny;
    const double* cp = c + p * S;
    double* mup = mu + p * S;

    for (size_t y = 0; y < m.ny; ++y) {
        const size_t yn = (y > 0) ? y - 1 : 0;
        const size_t yp = (y < m.ny - 1) ? y + 1 : y;
        const double* rc = cp + y * m.nx;
        double* rmu = mup + y * m.nx;

        laplacianRow(rc, cp + yn * m.nx, cp + yp * m.nx, rc - S, rc + S, lap, m.nx, m.dx2, m.dy2, m.dz2);

        for (size_t x = 0; x < m.nx; ++x) {
            const double cv = rc[x];
            rmu[x] = 4.5 * ((cv + 1.0) * m.e_AA + (cv - 1.0) * m.e_BB - 2.0 * cv * m.e_AB)
                     + 3.0 * cv + cv * cv * cv
                     - m.gamma * lap[x];
        }
    }
}

// Cahn-Hilliard update on local plane p (ghost planes p-1 and p+1 of mu must be valid)
void cahnHilliardUpdatePlane(const Model& m, const double* cold, double* cnew, const double* mu,
                             double* lap, const size_t p) {
    const size_t S = m.nx * m.ny;
    const double* mup = mu + p * S;
    const double* oldp = cold + p * S;
    double* newp = cnew + p * S;

    for (size_t y = 0; y < m.ny; ++y) {
        const size_t yn = (y > 0) ? y - 1 : 0;
        const size_t yp = (y < m.ny - 1) ? y + 1 : y;
        const double* rm = mup + y * m.nx;

        laplacianRow(rm, mup + yn * m.nx, mup + yp * m.nx, rm - S, rm + S, lap, m.nx, m.dx2, m.dy2, m.dz2);

        const double* ro = oldp + y * m.nx;
        double* rn = newp + y * m.nx;
        for (size_t x = 0; x < m.nx; ++x) {
            rn[x] = ro[x] + m.dt * m.D * lap[x];
        }
    }
}

// Initialize local planes 1..nzl (global planes z0..z0+nzl-1); ghost planes are left untouched
void initializeConcentration(double* c, const size_t z0, const size_t nzl,
                             const size_t nx, const size_t ny, const size_t nz) {
    const size_t S = nx * ny;
    const size_t vol = nx * ny * nz;

    for (size_t zl = 0; zl < nzl; ++zl) {
        const size_t z = z0 + zl;
        double* plane = c + (zl + 1) * S;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                plane[y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Post non-blocking exchange of the boundary planes of a field with the z-neighbours.
// Local planes 1 and nzl are sent; ghost planes 0 and nzl+1 are received.
// MPI_PROC_NULL neighbours make the corresponding transfers no-ops.
void startHaloExchange(double* f, const size_t S, const size_t nzl, MPI_Datatype planeType,
                       const int down, const int up, MPI_Comm comm, MPI_Request req[4]) {
    MPI_Irecv(f, 1, planeType, down, TAG_UP, comm, &req[0]);
    MPI_Irecv(f + (nzl + 1) * S, 1, planeType, up, TAG_DOWN, comm, &req[1]);
    MPI_Isend(f + S, 1, planeType, down, TAG_DOWN, comm, &req[2]);
    MPI_Isend(f + nzl * S, 1, planeType, up, TAG_UP, comm, &req[3]);
}

// At the global domain edges there is no neighbour; clamp by copying the boundary plane
// into the ghost plane, which reproduces the serial code's clamped z-neighbour
void fillPhysicalGhosts(double* f, const size_t S, const size_t nzl, const bool hasDown, const bool hasUp) {
    if (!hasDown) {
        std::copy_n(f + S, S, f);
    }
    if (!hasUp) {
        std::copy_n(f + nzl * S, S, f + (nzl + 1) * S);
    }
}

// Validate the distributed field; all ranks of comm must call this
bool validateResult(const double* c, const size_t n, MPI_Comm comm, const bool root) {
    // Check for NaN or Inf
    int localBad = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            localBad = 1;
            break;
        }
    }
    int anyBad = 0;
    MPI_Allreduce(&localBad, &anyBad, 1, MPI_INT, MPI_MAX, comm);
    if (anyBad) {
        if (root) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double minVal = c[0];
    double maxVal = c[0];
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }
    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (root) printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        if (root) printf("Validation failed: values out of expected range\n");
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
    const bool root = (rank == 0);

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

    if (nx == 0 || ny == 0 || nz == 0) {
        if (root) printf("Grid dimensions must be positive\n");
        return 1;
    }

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

    const Model m{nx, ny, dx * dx, dy * dy, dz * dz, gamma, e_AA, e_BB, e_AB, D, dt};
    const size_t S = nx * ny;             // cells per z-plane
    const size_t gridSize = S * nz;

    // Each rank needs at least one z-plane; surplus ranks sit out the computation
    const int active = static_cast<int>(std::min(static_cast<size_t>(size), nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    if (comm == MPI_COMM_NULL) {
        return 0;
    }

    int crank = 0;
    int csize = 1;
    MPI_Comm_rank(comm, &crank);
    MPI_Comm_size(comm, &csize);
    const bool isCommRoot = (crank == 0);

    // Slab decomposition along z: this rank owns planes z0 .. z0+nzl-1
    const size_t z0 = planeStart(crank, csize, nz);
    const size_t nzl = planeCount(crank, csize, nz);
    const int down = (crank > 0) ? crank - 1 : MPI_PROC_NULL;
    const int up = (crank < csize - 1) ? crank + 1 : MPI_PROC_NULL;

    // One contiguous z-plane as a single MPI element
    MPI_Datatype planeType;
    MPI_Type_contiguous(static_cast<int>(S), MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    // Local storage: planes 0 and nzl+1 are ghosts
    std::vector<double> cold((nzl + 2) * S);
    std::vector<double> cnew((nzl + 2) * S);
    std::vector<double> mu((nzl + 2) * S);
    std::vector<double> lap(nx);

    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    initializeConcentration(cold.data(), z0, nzl, nx, ny, nz);

    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double tStart = MPI_Wtime();

    const bool hasDown = (down != MPI_PROC_NULL);
    const bool hasUp = (up != MPI_PROC_NULL);
    MPI_Request req[4];

    for (int t = 0; t < iterations; ++t) {
        // Chemical potential: exchange concentration halos while computing interior planes
        startHaloExchange(cold.data(), S, nzl, planeType, down, up, comm, req);
        for (size_t p = 2; p < nzl; ++p) {
            computeChemicalPotentialPlane(m, cold.data(), mu.data(), lap.data(), p);
        }
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        fillPhysicalGhosts(cold.data(), S, nzl, hasDown, hasUp);
        computeChemicalPotentialPlane(m, cold.data(), mu.data(), lap.data(), 1);
        if (nzl > 1) {
            computeChemicalPotentialPlane(m, cold.data(), mu.data(), lap.data(), nzl);
        }

        // Update concentration: exchange mu halos while computing interior planes
        startHaloExchange(mu.data(), S, nzl, planeType, down, up, comm, req);
        for (size_t p = 2; p < nzl; ++p) {
            cahnHilliardUpdatePlane(m, cold.data(), cnew.data(), mu.data(), lap.data(), p);
        }
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        fillPhysicalGhosts(mu.data(), S, nzl, hasDown, hasUp);
        cahnHilliardUpdatePlane(m, cold.data(), cnew.data(), mu.data(), lap.data(), 1);
        if (nzl > 1) {
            cahnHilliardUpdatePlane(m, cold.data(), cnew.data(), mu.data(), lap.data(), nzl);
        }

        // Swap buffers
        std::swap(cold, cnew);
    }

    const double elapsed = MPI_Wtime() - tStart;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (root) {
        // Milliseconds, truncated as before
        const long long ms = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", ms);

        // Calculate performance
        const double cellUpdates = (double)gridSize * iterations;
        const double mcups = cellUpdates / (ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full field for printing
    std::vector<double> full;
    if (printResults) {
        std::vector<int> counts(csize), displs(csize);
        if (isCommRoot) {
            full.resize(gridSize);
            for (int r = 0; r < csize; ++r) {
                counts[r] = static_cast<int>(planeCount(r, csize, nz));
                displs[r] = static_cast<int>(planeStart(r, csize, nz));
            }
        }
        MPI_Gatherv(cold.data() + S, static_cast<int>(nzl), planeType,
                    full.data(), counts.data(), displs.data(), planeType, 0, comm);

        if (isCommRoot) {
            print_results(full, "Concentration");
        }
    }

    // Validation
    int rc = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        const bool valid = validateResult(cold.data() + S, nzl * S, comm, isCommRoot);

        if (valid) {
            if (root) printf("Validation: PASSED\n");
        } else {
            if (root) printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    MPI_Type_free(&planeType);
    MPI_Comm_free(&comm);
    return rc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rc = runBenchmark(argc, argv, rank, size);
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return rc;
}
