#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation (z-major)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
static inline double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                                     const double dx, const double dy, const double dz, const size_t x, const size_t y,
                                     const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * c[idx3(x, y, z, nx, ny)]) /
                       (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * c[idx3(x, y, z, nx, ny)]) /
                       (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 2.0 * c[idx3(x, y, z, nx, ny)]) /
                       (dz * dz);

    return cxx + cyy + czz;
}

static inline void halo_exchange_z(std::vector<double>& field, const size_t nx, const size_t ny, const size_t local_nz,
                                  MPI_Comm comm, const int prev, const int next) {
    const size_t plane_elems = nx * ny;
    double* base = field.data();
    double* const ghost_lo = base;
    double* const interior_lo = base + plane_elems;
    double* const interior_hi = base + plane_elems * local_nz;
    double* const ghost_hi = base + plane_elems * (local_nz + 1);

    if (prev == MPI_PROC_NULL) {
        std::memcpy(ghost_lo, interior_lo, plane_elems * sizeof(double));
    } else {
        MPI_Sendrecv(interior_lo, static_cast<int>(plane_elems), MPI_DOUBLE, prev, 0, ghost_lo,
                     static_cast<int>(plane_elems), MPI_DOUBLE, prev, 1, comm, MPI_STATUS_IGNORE);
    }

    if (next == MPI_PROC_NULL) {
        std::memcpy(ghost_hi, interior_hi, plane_elems * sizeof(double));
    } else {
        MPI_Sendrecv(interior_hi, static_cast<int>(plane_elems), MPI_DOUBLE, next, 1, ghost_hi,
                     static_cast<int>(plane_elems), MPI_DOUBLE, next, 0, comm, MPI_STATUS_IGNORE);
    }
}

// Compute chemical potential on a z-range [z_begin, z_end) of a ghosted domain
static inline void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                                            const size_t nx, const size_t ny, const size_t nz,
                                            const double dx, const double dy, const double dz,
                                            const double gamma, const double e_AA, const double e_BB, const double e_AB,
                                            const size_t z_begin, const size_t z_end) {
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv -
                          gamma * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step on a z-range [z_begin, z_end) of a ghosted domain
static inline void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold, const std::vector<double>& mu,
                                      const size_t nx, const size_t ny, const size_t nz,
                                      const double D, const double dt, const double dx, const double dy, const double dz,
                                      const size_t z_begin, const size_t z_end) {
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field for a local z-slab; stores into ghosted array with interior at z=1..local_nz
static inline void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz_global,
                                           const size_t z0, const size_t local_nz) {
    const size_t vol = nx * ny * nz_global;

    for (size_t lz = 0; lz < local_nz; ++lz) {
        const size_t gz = z0 + lz;
        const size_t z = lz + 1;  // shift for ghost plane
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

static inline bool validateResultMPI(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz,
                                     MPI_Comm comm, const int rank) {
    const size_t plane_elems = nx * ny;

    int local_bad = 0;
    double local_min = 0.0;
    double local_max = 0.0;

    bool first = true;
    for (size_t z = 1; z <= local_nz; ++z) {
        const size_t base = z * plane_elems;
        for (size_t i = 0; i < plane_elems; ++i) {
            const double v = c[base + i];
            if (std::isnan(v) || std::isinf(v)) {
                local_bad = 1;
            }
            if (first) {
                local_min = v;
                local_max = v;
                first = false;
            } else {
                local_min = std::min(local_min, v);
                local_max = std::max(local_max, v);
            }
        }
    }

    int any_bad = 0;
    MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_MAX, comm);

    double global_min = 0.0;
    double global_max = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        if (any_bad) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }

        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);

        if (global_max > 10.0 || global_min < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }

    int root_ok = 1;
    if (rank == 0) {
        root_ok = (!any_bad && !(global_max > 10.0 || global_min < -10.0)) ? 1 : 0;
    }
    MPI_Bcast(&root_ok, 1, MPI_INT, 0, comm);
    return root_ok != 0;
}

static inline void printUsage(const char* progName) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Use at most nz ranks (slab decomposition in Z)
    const int active_size = std::min<int>(world_size, static_cast<int>(nz));
    const int color = (world_rank < active_size) ? 0 : MPI_UNDEFINED;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &comm);

    if (color == MPI_UNDEFINED) {
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        if (active_size != world_size) {
            printf("Note: using %d/%d MPI ranks (nz=%zu)\n", active_size, world_size, nz);
        }
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

    // Z-slab decomposition
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t local_nz = base + ((static_cast<size_t>(rank) < rem) ? 1 : 0);
    const size_t z0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);

    const int prev = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
    const int next = (rank + 1 < size) ? (rank + 1) : MPI_PROC_NULL;

    const size_t nz_ghosted = local_nz + 2;
    const size_t local_size = nx * ny * nz_ghosted;

    std::vector<double> cold(local_size);
    std::vector<double> cnew(local_size);
    std::vector<double> mu(local_size);

    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, z0, local_nz);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const auto start = std::chrono::high_resolution_clock::now();

    const size_t z_begin = 1;
    const size_t z_end = local_nz + 1;

    for (int t = 0; t < iterations; ++t) {
        halo_exchange_z(cold, nx, ny, local_nz, comm, prev, next);
        computeChemicalPotential(cold, mu, nx, ny, nz_ghosted, dx, dy, dz, gamma, e_AA, e_BB, e_AB, z_begin, z_end);

        halo_exchange_z(mu, nx, ny, local_nz, comm, prev, next);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz_ghosted, D, dt, dx, dy, dz, z_begin, z_end);

        std::swap(cold, cnew);
    }

    MPI_Barrier(comm);
    const auto end = std::chrono::high_resolution_clock::now();
    const long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    long long max_duration_ms = 0;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);

    if (rank == 0) {
        printf("Computation time (max rank): %lld ms\n", max_duration_ms);
        const double cellUpdates = static_cast<double>(gridSize) * static_cast<double>(iterations);
        const double mcups = cellUpdates / (static_cast<double>(max_duration_ms) / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    if (printResults) {
        const size_t plane = nx * ny;
        std::vector<double> gathered;
        std::vector<int> recvcounts;
        std::vector<int> displs;
        if (rank == 0) {
            gathered.resize(gridSize);
            recvcounts.resize(static_cast<size_t>(size));
            displs.resize(static_cast<size_t>(size));
            for (int r = 0; r < size; ++r) {
                const size_t lnz = base + ((static_cast<size_t>(r) < rem) ? 1 : 0);
                const size_t rz0 = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
                recvcounts[static_cast<size_t>(r)] = static_cast<int>(lnz * plane);
                displs[static_cast<size_t>(r)] = static_cast<int>(rz0 * plane);
            }
        }

        MPI_Gatherv(cold.data() + plane, static_cast<int>(local_nz * plane), MPI_DOUBLE,
                    (rank == 0) ? gathered.data() : nullptr, (rank == 0) ? recvcounts.data() : nullptr,
                    (rank == 0) ? displs.data() : nullptr, MPI_DOUBLE, 0, comm);

        if (rank == 0) {
            print_results(gathered, "Concentration");
        }
    }

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool ok = validateResultMPI(cold, nx, ny, local_nz, comm, rank);
        if (rank == 0) {
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
        }
        MPI_Comm_free(&comm);
        MPI_Finalize();
        return ok ? 0 : 1;
    }

    MPI_Comm_free(&comm);
    MPI_Finalize();
    return 0;
}
