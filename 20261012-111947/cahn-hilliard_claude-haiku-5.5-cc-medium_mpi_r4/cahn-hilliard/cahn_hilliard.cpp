#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

// Halo message tags
constexpr int TAG_UP = 0;    // plane sent to the rank above
constexpr int TAG_DOWN = 1;  // plane sent to the rank below

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Z-slab owned by one rank. Local fields hold nzLocal + 2 planes: planes 1..nzLocal
// are owned (global planes z0..z0+nzLocal-1), planes 0 and nzLocal+1 are ghosts.
struct Slab {
    size_t z0 = 0;
    size_t nzLocal = 0;
    int down = MPI_PROC_NULL;  // rank owning the plane below, MPI_PROC_NULL at the domain edge
    int up = MPI_PROC_NULL;    // rank owning the plane above, MPI_PROC_NULL at the domain edge
};

// Block distribution of nz planes over size ranks (the first nz % size ranks get one extra plane)
Slab makeSlab(const size_t nz, const int rank, const int size) {
    const size_t P = static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = nz / P;
    const size_t rem = nz % P;
    auto count = [&](const size_t q) { return base + (q < rem ? 1 : 0); };

    Slab s;
    s.nzLocal = count(r);
    s.z0 = r * base + std::min(r, rem);
    if (s.nzLocal > 0) {
        if (r > 0) s.down = rank - 1;
        if (r + 1 < P && count(r + 1) > 0) s.up = rank + 1;
    }
    return s;
}

// Starts the ghost-plane exchange of field f. Ghosts at the domain edges are copies of the
// boundary plane, which reproduces the clamped boundary conditions of the serial code.
void startHalo(std::vector<double>& f, const Slab& s, const size_t nxy, MPI_Datatype plane,
               MPI_Comm comm, MPI_Request (&req)[4]) {
    for (auto& r : req) r = MPI_REQUEST_NULL;
    if (s.nzLocal == 0) return;

    double* ghostLow = f.data();
    double* ownLow = f.data() + nxy;
    double* ownHigh = f.data() + s.nzLocal * nxy;
    double* ghostHigh = f.data() + (s.nzLocal + 1) * nxy;

    if (s.down == MPI_PROC_NULL) std::copy_n(ownLow, nxy, ghostLow);
    if (s.up == MPI_PROC_NULL) std::copy_n(ownHigh, nxy, ghostHigh);

    MPI_Irecv(ghostLow, 1, plane, s.down, TAG_UP, comm, &req[0]);
    MPI_Irecv(ghostHigh, 1, plane, s.up, TAG_DOWN, comm, &req[1]);
    MPI_Isend(ownLow, 1, plane, s.down, TAG_DOWN, comm, &req[2]);
    MPI_Isend(ownHigh, 1, plane, s.up, TAG_UP, comm, &req[3]);
}

// Compute Laplacian with clamped boundary conditions. Local plane k is the z-index;
// z-neighbours come from the ghost planes.
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t k) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const size_t center = idx3(x, y, k, nx, ny);
    const double cxx = (c[idx3(xp, y, k, nx, ny)] + c[idx3(xn, y, k, nx, ny)] -
                  2.0 * c[center]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, k, nx, ny)] + c[idx3(x, yn, k, nx, ny)] -
                  2.0 * c[center]) / (dy * dy);
    const double czz = (c[center + nx * ny] + c[center - nx * ny] -
                  2.0 * c[center]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential on local planes [kBegin, kEnd)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const size_t kBegin, const size_t kEnd) {
    for (size_t k = kBegin; k < kEnd; ++k) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, k, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, k);
            }
        }
    }
}

// Cahn-Hilliard update step on local planes [kBegin, kEnd)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny,
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        const size_t kBegin, const size_t kEnd) {
    for (size_t k = kBegin; k < kEnd; ++k) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, k, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, k);
            }
        }
    }
}

// Initialize the owned planes of concentration field
void initializeConcentration(std::vector<double>& c, const Slab& s, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    for (size_t k = 1; k <= s.nzLocal; ++k) {
        const size_t z = s.z0 + k - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, k, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Collect the full field on rank 0 in global z-major order
std::vector<double> gatherField(const std::vector<double>& f, const Slab& s, const size_t nz, const size_t nxy,
                                MPI_Datatype plane, const int rank, const int size, MPI_Comm comm) {
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        const Slab o = makeSlab(nz, r, size);
        counts[r] = static_cast<int>(o.nzLocal);
        displs[r] = static_cast<int>(o.z0);
    }

    std::vector<double> full(rank == 0 ? nxy * nz : 0);
    MPI_Gatherv(f.data() + nxy, static_cast<int>(s.nzLocal), plane,
                full.data(), counts.data(), displs.data(), plane, 0, comm);
    return full;
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
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
    const bool root = (rank == 0);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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

    // A grid plane and the plane count must fit in MPI's int counts
    if (nx * ny > static_cast<size_t>(INT_MAX) || nz > static_cast<size_t>(INT_MAX)) {
        if (root) printf("Grid too large: nx*ny and nz must be at most %d\n", INT_MAX);
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

    const size_t nxy = nx * ny;
    const size_t gridSize = nxy * nz;
    const Slab slab = makeSlab(nz, rank, size);
    const size_t L = slab.nzLocal;

    // One xy-plane as a single MPI element
    MPI_Datatype planeType;
    MPI_Type_contiguous(static_cast<int>(nxy), MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    // Allocate arrays (owned planes plus two ghost planes)
    std::vector<double> cold((L + 2) * nxy);
    std::vector<double> cnew((L + 2) * nxy);
    std::vector<double> mu((L + 2) * nxy);

    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    initializeConcentration(cold, slab, nx, ny, nz);

    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Ranks without planes have nothing to compute
        if (L == 0) continue;

        // Chemical potential: interior planes need no ghosts, so they overlap the halo exchange
        MPI_Request req[4];
        startHalo(cold, slab, nxy, planeType, MPI_COMM_WORLD, req);
        computeChemicalPotential(cold, mu, nx, ny, dx, dy, dz, gamma, e_AA, e_BB, e_AB, 2, L);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        computeChemicalPotential(cold, mu, nx, ny, dx, dy, dz, gamma, e_AA, e_BB, e_AB, 1, 2);
        if (L > 1) computeChemicalPotential(cold, mu, nx, ny, dx, dy, dz, gamma, e_AA, e_BB, e_AB, L, L + 1);

        // Update concentration: same overlap for the halo exchange of mu
        startHalo(mu, slab, nxy, planeType, MPI_COMM_WORLD, req);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, D, dt, dx, dy, dz, 2, L);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, D, dt, dx, dy, dz, 1, 2);
        if (L > 1) cahnHilliardUpdate(cnew, cold, mu, nx, ny, D, dt, dx, dy, dz, L, L + 1);

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full field on rank 0 for output and validation
    if (printResults || validate) {
        const std::vector<double> full = gatherField(cold, slab, nz, nxy, planeType, rank, size, MPI_COMM_WORLD);

        // Print results for external validation
        if (printResults && root) {
            print_results(full, "Concentration");
        }

        // Validation
        if (validate) {
            if (root) printf("Validating result...\n");
            int valid = (root && validateResult(full, nx, ny, nz)) ? 1 : 0;
            MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (root) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
            MPI_Type_free(&planeType);
            return valid ? 0 : 1;
        }
    }

    MPI_Type_free(&planeType);
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = runBenchmark(argc, argv);
    MPI_Finalize();
    return rc;
}
