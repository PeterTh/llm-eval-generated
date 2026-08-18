#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Domain {
    size_t nx;
    size_t ny;
    size_t nz;
    size_t local_nz;
    size_t z_begin;
    size_t plane;
    int rank;
    int size;
    int prev;
    int next;
    MPI_Comm comm;
};

Domain makeDomain(const size_t nx, const size_t ny, const size_t nz,
                  const int rank, const int size, MPI_Comm comm) {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t remainder = nz % static_cast<size_t>(size);
    const size_t local_nz = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t z_begin = static_cast<size_t>(rank) * base +
                           std::min(static_cast<size_t>(rank), remainder);
    return {nx, ny, nz, local_nz, z_begin, nx * ny, rank, size,
            rank == 0 ? MPI_PROC_NULL : rank - 1,
            rank + 1 == size ? MPI_PROC_NULL : rank + 1, comm};
}

void beginHaloExchange(std::vector<double>& field, const Domain& d,
                       MPI_Request requests[4]) {
    const int count = static_cast<int>(d.plane);
    MPI_Irecv(field.data(), count, MPI_DOUBLE, d.prev, 10, d.comm, &requests[0]);
    MPI_Irecv(field.data() + (d.local_nz + 1) * d.plane, count, MPI_DOUBLE,
              d.next, 11, d.comm, &requests[1]);
    MPI_Isend(field.data() + d.plane, count, MPI_DOUBLE, d.prev, 11, d.comm,
              &requests[2]);
    MPI_Isend(field.data() + d.local_nz * d.plane, count, MPI_DOUBLE, d.next,
              10, d.comm, &requests[3]);
}

inline double laplacian(const std::vector<double>& field, const Domain& d,
                        const size_t x, const size_t y, const size_t local_z,
                        const double dx, const double dy, const double dz) noexcept {
    const size_t xp = x + (x + 1 < d.nx);
    const size_t xn = x - (x > 0);
    const size_t yp = y + (y + 1 < d.ny);
    const size_t yn = y - (y > 0);
    const size_t global_z = d.z_begin + local_z - 1;
    const size_t zp = local_z + (global_z + 1 < d.nz);
    const size_t zn = local_z - (global_z > 0);
    const size_t center = idx3(x, y, local_z, d.nx, d.ny);
    const double value = field[center];
    const double cxx = (field[idx3(xp, y, local_z, d.nx, d.ny)] +
                        field[idx3(xn, y, local_z, d.nx, d.ny)] - 2.0 * value) /
                       (dx * dx);
    const double cyy = (field[idx3(x, yp, local_z, d.nx, d.ny)] +
                        field[idx3(x, yn, local_z, d.nx, d.ny)] - 2.0 * value) /
                       (dy * dy);
    const double czz = (field[idx3(x, y, zp, d.nx, d.ny)] +
                        field[idx3(x, y, zn, d.nx, d.ny)] - 2.0 * value) /
                       (dz * dz);
    return cxx + cyy + czz;
}

void computeChemicalRange(const std::vector<double>& c, std::vector<double>& mu,
                          const Domain& d, const size_t first_z, const size_t last_z,
                          const double dx, const double dy, const double dz,
                          const double gamma, const double e_AA, const double e_BB,
                          const double e_AB) {
    for (size_t z = first_z; z < last_z; ++z) {
        for (size_t y = 0; y < d.ny; ++y) {
            for (size_t x = 0; x < d.nx; ++x) {
                const size_t i = idx3(x, y, z, d.nx, d.ny);
                const double cv = c[i];
                mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB -
                               2.0 * cv * e_AB) +
                        3.0 * cv + cv * cv * cv -
                        gamma * laplacian(c, d, x, y, z, dx, dy, dz);
            }
        }
    }
}

void computeChemicalPotential(std::vector<double>& c, std::vector<double>& mu,
                              const Domain& d, const double dx, const double dy,
                              const double dz, const double gamma, const double e_AA,
                              const double e_BB, const double e_AB) {
    MPI_Request requests[4];
    beginHaloExchange(c, d, requests);
    if (d.local_nz > 2) {
        computeChemicalRange(c, mu, d, 2, d.local_nz, dx, dy, dz, gamma,
                             e_AA, e_BB, e_AB);
    }
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    computeChemicalRange(c, mu, d, 1, 2, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
    if (d.local_nz > 1) {
        computeChemicalRange(c, mu, d, d.local_nz, d.local_nz + 1, dx, dy, dz,
                             gamma, e_AA, e_BB, e_AB);
    }
}

void updateRange(std::vector<double>& cnew, const std::vector<double>& cold,
                 const std::vector<double>& mu, const Domain& d,
                 const size_t first_z, const size_t last_z, const double D,
                 const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = first_z; z < last_z; ++z) {
        for (size_t y = 0; y < d.ny; ++y) {
            for (size_t x = 0; x < d.nx; ++x) {
                const size_t i = idx3(x, y, z, d.nx, d.ny);
                cnew[i] = cold[i] + dt * D * laplacian(mu, d, x, y, z, dx, dy, dz);
            }
        }
    }
}

void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        std::vector<double>& mu, const Domain& d, const double D,
                        const double dt, const double dx, const double dy,
                        const double dz) {
    MPI_Request requests[4];
    beginHaloExchange(mu, d, requests);
    if (d.local_nz > 2) {
        updateRange(cnew, cold, mu, d, 2, d.local_nz, D, dt, dx, dy, dz);
    }
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    updateRange(cnew, cold, mu, d, 1, 2, D, dt, dx, dy, dz);
    if (d.local_nz > 1) {
        updateRange(cnew, cold, mu, d, d.local_nz, d.local_nz + 1, D, dt, dx,
                    dy, dz);
    }
}

void initializeConcentration(std::vector<double>& c, const Domain& d) {
    const size_t volume = d.nx * d.ny * d.nz;
    for (size_t z = 1; z <= d.local_nz; ++z) {
        const size_t global_z = d.z_begin + z - 1;
        for (size_t y = 0; y < d.ny; ++y) {
            for (size_t x = 0; x < d.nx; ++x) {
                const size_t linear_id = global_z * d.plane + y * d.nx + x;
                const double pseudo = (((linear_id + 1) * 1299709) % volume) /
                                      static_cast<double>(volume);
                c[idx3(x, y, z, d.nx, d.ny)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, const Domain& d) {
    bool local_finite = true;
    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();
    for (size_t i = d.plane; i < (d.local_nz + 1) * d.plane; ++i) {
        local_finite = local_finite && std::isfinite(c[i]);
        local_min = std::min(local_min, c[i]);
        local_max = std::max(local_max, c[i]);
    }
    int finite = local_finite ? 1 : 0;
    int all_finite = 0;
    double global_min = 0.0;
    double global_max = 0.0;
    MPI_Allreduce(&finite, &all_finite, 1, MPI_INT, MPI_LAND, d.comm);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, d.comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, d.comm);
    if (d.rank == 0) {
        if (!all_finite) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        std::printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
        if (global_max > 10.0 || global_min < -10.0) {
            std::printf("Validation failed: values out of expected range\n");
        }
    }
    return all_finite && global_max <= 10.0 && global_min >= -10.0;
}

std::vector<double> gatherResult(const std::vector<double>& local, const Domain& d) {
    std::vector<int> counts(d.size);
    std::vector<int> displacements(d.size);
    for (int r = 0; r < d.size; ++r) {
        const size_t base = d.nz / static_cast<size_t>(d.size);
        const size_t remainder = d.nz % static_cast<size_t>(d.size);
        const size_t slabs = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
        const size_t begin = static_cast<size_t>(r) * base +
                             std::min(static_cast<size_t>(r), remainder);
        counts[r] = static_cast<int>(slabs * d.plane);
        displacements[r] = static_cast<int>(begin * d.plane);
    }
    std::vector<double> result;
    if (d.rank == 0) result.resize(d.nx * d.ny * d.nz);
    MPI_Gatherv(local.data() + d.plane, static_cast<int>(d.local_nz * d.plane),
                MPI_DOUBLE, result.data(), counts.data(), displacements.data(),
                MPI_DOUBLE, 0, d.comm);
    return result;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

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
    bool help = false;
    bool arguments_ok = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (world_rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            arguments_ok = false;
            break;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (help || !arguments_ok) {
        if (world_rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return help ? 0 : 1;
    }

    const bool dimensions_ok = nx > 0 && ny > 0 && nz > 0 && iterations >= 0 &&
        nx <= std::numeric_limits<size_t>::max() / ny &&
        nx * ny <= std::numeric_limits<size_t>::max() / nz &&
        nx * ny <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
        nx * ny * nz <= static_cast<size_t>(std::numeric_limits<int>::max());
    if (!dimensions_ok) {
        if (world_rank == 0) {
            std::fprintf(stderr, "Grid dimensions or iteration count are invalid or too large for MPI counts\n");
        }
        MPI_Finalize();
        return 1;
    }

    // More ranks than z planes cannot contribute to a slab decomposition.
    const int active_size = std::min(world_size, static_cast<int>(nz));
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_size ? 0 : MPI_UNDEFINED,
                   world_rank, &active_comm);
    if (world_rank >= active_size) {
        MPI_Finalize();
        return 0;
    }

    const Domain domain = makeDomain(nx, ny, nz, world_rank, active_size, active_comm);
    if (domain.rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("MPI ranks: %d\n", active_size);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    constexpr double dx = 1.0;
    constexpr double dy = 1.0;
    constexpr double dz = 1.0;
    constexpr double dt = 0.01;
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = (2.0 / 9.0);
    constexpr double gamma = 0.5;
    constexpr double D = 1.0;

    const size_t local_size = (domain.local_nz + 2) * domain.plane;
    std::vector<double> cold(local_size);
    std::vector<double> cnew(local_size);
    std::vector<double> mu(local_size);
    if (domain.rank == 0) std::printf("Initializing concentration field...\n");
    initializeConcentration(cold, domain);

    if (domain.rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(active_comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotential(cold, mu, domain, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        cahnHilliardUpdate(cnew, cold, mu, domain, D, dt, dx, dy, dz);
        std::swap(cold, cnew);
    }
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm);

    if (domain.rank == 0) {
        const double cell_updates = static_cast<double>(nx * ny * nz) * iterations;
        const double mcups = elapsed > 0.0 ? cell_updates / elapsed / 1.0e6 : 0.0;
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        std::vector<double> result = gatherResult(cold, domain);
        if (domain.rank == 0) print_results(result, "Concentration");
    }

    bool valid = true;
    if (validate) {
        if (domain.rank == 0) std::printf("Validating result...\n");
        valid = validateResult(cold, domain);
        if (domain.rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    MPI_Comm_free(&active_comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
