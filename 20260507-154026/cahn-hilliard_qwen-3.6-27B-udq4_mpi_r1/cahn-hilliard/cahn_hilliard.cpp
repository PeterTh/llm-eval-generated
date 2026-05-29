#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// ============================================================================
// MPI parallelization: domain decomposition along Z-axis with ghost cells
// ============================================================================

struct DomainInfo {
    int rank;
    int num_ranks;
    size_t nx, ny, nz;

    // Global Z range owned by this rank (real cells, exclusive of ghosts)
    size_t z_start;    // first real Z (inclusive)
    size_t z_end;      // last real Z (exclusive)
    size_t local_nz;   // number of real Z cells

    // Ghost cells: 1 plane below (local index 0) and 1 plane above (local index local_nz+1)
    // Real cells: local indices 1 .. local_nz  (offset = 1)
    static constexpr size_t ghost_offset = 1;

    size_t local_size; // nx * ny * (local_nz + 2)
};

static inline DomainInfo computeDomain(size_t nz_global, size_t nx, size_t ny, int rank, int num_ranks) {
    DomainInfo info;
    info.rank = rank;
    info.num_ranks = num_ranks;
    info.nx = nx;
    info.ny = ny;
    info.nz = nz_global;

    // Distribute Z planes as evenly as possible
    size_t base = nz_global / num_ranks;
    size_t remainder = nz_global % num_ranks;

    size_t start = 0;
    for (int r = 0; r < rank; ++r) {
        start += base + (r < static_cast<int>(remainder) ? 1 : 0);
    }
    info.z_start = start;
    info.local_nz = base + (rank < static_cast<int>(remainder) ? 1 : 0);
    info.z_end = start + info.local_nz;

    info.local_size = nx * ny * (info.local_nz + 2 * DomainInfo::ghost_offset);
    return info;
}

// Local index: lz=0 is ghost below, lz=local_nz+1 is ghost above
static inline size_t lidx(const DomainInfo& dom, size_t x, size_t y, size_t lz) noexcept {
    return lz * dom.nx * dom.ny + y * dom.nx + x;
}

// Compute Laplacian using local indices (ghost cells hold correct boundary values)
static inline double computeLaplacianLocal(const std::vector<double>& c,
    const DomainInfo& dom, const double dx, const double dy, const double dz,
    size_t x, size_t y, size_t lz) noexcept {
    const size_t nx = dom.nx;
    const size_t ny = dom.ny;

    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : y;

    const size_t idx = lidx(dom, x, y, lz);
    const double cxx = (c[lidx(dom, xp, y, lz)] + c[lidx(dom, xn, y, lz)] - 2.0 * c[idx]) / (dx * dx);
    const double cyy = (c[lidx(dom, x, yp, lz)] + c[lidx(dom, x, yn, lz)] - 2.0 * c[idx]) / (dy * dy);
    const double czz = (c[lidx(dom, x, y, lz + 1)] + c[lidx(dom, x, y, lz - 1)] - 2.0 * c[idx]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential on local domain (real cells only)
static void computeChemicalPotentialLocal(const std::vector<double>& c, std::vector<double>& mu,
    const DomainInfo& dom,
    const double dx, const double dy, const double dz,
    const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t nx = dom.nx;
    const size_t ny = dom.ny;
    const size_t go = DomainInfo::ghost_offset;

    for (size_t lz = go; lz < go + dom.local_nz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = lidx(dom, x, y, lz);
                const double cv = c[idx];
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacianLocal(c, dom, dx, dy, dz, x, y, lz);
            }
        }
    }
}

// Cahn-Hilliard update on local domain (real cells only)
static void cahnHilliardUpdateLocal(std::vector<double>& cnew, const std::vector<double>& cold,
    const std::vector<double>& mu,
    const DomainInfo& dom,
    const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t nx = dom.nx;
    const size_t ny = dom.ny;
    const size_t go = DomainInfo::ghost_offset;

    for (size_t lz = go; lz < go + dom.local_nz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = lidx(dom, x, y, lz);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacianLocal(mu, dom, dx, dy, dz, x, y, lz);
            }
        }
    }
}

// Initialize concentration field on local domain
static void initializeConcentrationLocal(std::vector<double>& c, const DomainInfo& dom) {
    const size_t vol = dom.nx * dom.ny * dom.nz;
    const size_t nx = dom.nx;
    const size_t ny = dom.ny;
    const size_t go = DomainInfo::ghost_offset;

    for (size_t lz = go; lz < go + dom.local_nz; ++lz) {
        const size_t z = dom.z_start + (lz - go);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = lidx(dom, x, y, lz);
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Exchange ghost cells between neighboring ranks
static void exchangeGhostCells(std::vector<double>& c, const DomainInfo& dom) {
    const size_t plane_size = dom.nx * dom.ny;

    MPI_Request sends[2], recvs[2];
    int count = 0;

    // Send top real boundary to rank+1; receive from rank+1 into top ghost plane
    if (dom.rank < dom.num_ranks - 1) {
        const size_t top_real = DomainInfo::ghost_offset + dom.local_nz - 1;
        MPI_Isend(&c[lidx(dom, 0, 0, top_real)], static_cast<int>(plane_size), MPI_DOUBLE,
                  dom.rank + 1, 0, MPI_COMM_WORLD, &sends[count]);
        MPI_Irecv(&c[lidx(dom, 0, 0, DomainInfo::ghost_offset + dom.local_nz)],
                  static_cast<int>(plane_size), MPI_DOUBLE,
                  dom.rank + 1, 1, MPI_COMM_WORLD, &recvs[count]);
        count++;
    }

    // Send bottom real boundary to rank-1; receive from rank-1 into bottom ghost plane
    if (dom.rank > 0) {
        const size_t bottom_real = DomainInfo::ghost_offset;
        MPI_Isend(&c[lidx(dom, 0, 0, bottom_real)], static_cast<int>(plane_size), MPI_DOUBLE,
                  dom.rank - 1, 1, MPI_COMM_WORLD, &sends[count]);
        MPI_Irecv(&c[lidx(dom, 0, 0, 0)], static_cast<int>(plane_size), MPI_DOUBLE,
                  dom.rank - 1, 0, MPI_COMM_WORLD, &recvs[count]);
        count++;
    }

    if (count > 0) {
        MPI_Waitall(count, sends, MPI_STATUSES_IGNORE);
        MPI_Waitall(count, recvs, MPI_STATUSES_IGNORE);
    }

    // Clamp ghost cells for boundary ranks (matching original clamped BC)
    if (dom.rank == 0) {
        const size_t bottom_real = DomainInfo::ghost_offset;
        std::memcpy(&c[lidx(dom, 0, 0, 0)], &c[lidx(dom, 0, 0, bottom_real)],
                    plane_size * sizeof(double));
    }
    if (dom.rank == dom.num_ranks - 1) {
        const size_t top_real = DomainInfo::ghost_offset + dom.local_nz - 1;
        const size_t top_ghost = DomainInfo::ghost_offset + dom.local_nz;
        std::memcpy(&c[lidx(dom, 0, 0, top_ghost)], &c[lidx(dom, 0, 0, top_real)],
                    plane_size * sizeof(double));
    }
}

// Gather real cells from all ranks to rank 0 in global order
static std::vector<double> gatherToRank0(const std::vector<double>& local_c,
                                         const DomainInfo& dom, size_t global_size) {
    std::vector<double> global_c;
    if (dom.rank == 0) global_c.resize(global_size);

    // Compute per-rank counts and displacements
    std::vector<int> counts(dom.num_ranks);
    std::vector<int> displs(dom.num_ranks);
    size_t base = dom.nz / dom.num_ranks;
    size_t remainder = dom.nz % dom.num_ranks;
    size_t offset = 0;
    for (int r = 0; r < dom.num_ranks; ++r) {
        size_t r_nz = base + (r < static_cast<int>(remainder) ? 1 : 0);
        counts[r] = static_cast<int>(dom.nx * dom.ny * r_nz);
        displs[r] = static_cast<int>(offset);
        offset += r_nz * dom.nx * dom.ny;
    }

    // Extract real cells into contiguous buffer
    size_t local_real_count = dom.nx * dom.ny * dom.local_nz;
    std::vector<double> local_real(local_real_count);
    const size_t go = DomainInfo::ghost_offset;
    for (size_t lz = go; lz < go + dom.local_nz; ++lz) {
        std::memcpy(&local_real[(lz - go) * dom.nx * dom.ny],
                    &local_c[lidx(dom, 0, 0, lz)],
                    dom.nx * dom.ny * sizeof(double));
    }

    MPI_Gatherv(local_real.data(), static_cast<int>(local_real_count), MPI_DOUBLE,
                global_c.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    return global_c;
}

// MPI-aware validation: check NaN/Inf and range across all ranks
static bool validateResultLocal(const std::vector<double>& c, const DomainInfo& dom) {
    bool local_valid = true;
    double local_min = 0.0, local_max = 0.0;

    const size_t nx = dom.nx;
    const size_t ny = dom.ny;
    const size_t go = DomainInfo::ghost_offset;

    for (size_t lz = go; lz < go + dom.local_nz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const double val = c[lidx(dom, x, y, lz)];
                if (std::isnan(val) || std::isinf(val)) local_valid = false;
                if (lz == go && y == 0 && x == 0) {
                    local_min = val;
                    local_max = val;
                } else {
                    local_min = std::min(local_min, val);
                    local_max = std::max(local_max, val);
                }
            }
        }
    }

    bool global_valid;
    double global_min, global_max;
    MPI_Allreduce(&local_valid, &global_valid, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (dom.rank == 0) {
        if (!global_valid) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
        if (global_max > 10.0 || global_min < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }
    return global_valid;
}

void printUsage(const char* progName) {
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d\n", num_ranks);
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

    size_t gridSize = nx * ny * nz;

    // Domain decomposition
    DomainInfo dom = computeDomain(nz, nx, ny, rank, num_ranks);

    // Allocate local arrays (with ghost cells)
    std::vector<double> cold(dom.local_size);
    std::vector<double> cnew(dom.local_size);
    std::vector<double> mu(dom.local_size);

    // Initialize
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentrationLocal(cold, dom);
    initializeConcentrationLocal(cnew, dom);
    exchangeGhostCells(cold, dom);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotentialLocal(cold, mu, dom, dx, dy, dz,
                                      gamma, e_AA, e_BB, e_AB);
        exchangeGhostCells(mu, dom);
        cahnHilliardUpdateLocal(cnew, cold, mu, dom, D, dt, dx, dy, dz);
        std::swap(cold, cnew);
        exchangeGhostCells(cold, dom);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Synchronize timing across ranks
    double local_duration_ms = static_cast<double>(duration.count());
    double max_duration_ms;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", max_duration_ms);
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (max_duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> global_c = gatherToRank0(cold, dom, gridSize);
        if (rank == 0) print_results(global_c, "Concentration");
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResultLocal(cold, dom);
        if (rank == 0) {
            printf("%s\n", valid ? "Validation: PASSED" : "Validation: FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
