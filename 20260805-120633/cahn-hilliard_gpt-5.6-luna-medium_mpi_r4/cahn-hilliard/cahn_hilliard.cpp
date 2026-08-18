#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// The arrays contain two halo planes: z=0 and z=local_nz+1. Owned planes are
// z=1..local_nz. The halo values implement the original clamped boundaries.
void exchangeHalos(std::vector<double>& field, const size_t nx, const size_t ny,
                   const size_t local_nz, const int rank, const int ranks,
                   MPI_Comm comm) {
    const int plane = static_cast<int>(nx * ny);
    if (ranks == 1) {
        std::copy_n(field.data() + idx3(0, 0, 1, nx, ny), plane,
                    field.data() + idx3(0, 0, 0, nx, ny));
        std::copy_n(field.data() + idx3(0, 0, local_nz, nx, ny), plane,
                    field.data() + idx3(0, 0, local_nz + 1, nx, ny));
        return;
    }

    const int lower = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int upper = (rank == ranks - 1) ? MPI_PROC_NULL : rank + 1;
    MPI_Sendrecv(field.data() + idx3(0, 0, 1, nx, ny), plane, MPI_DOUBLE,
                 lower, 0, field.data() + idx3(0, 0, local_nz + 1, nx, ny),
                 plane, MPI_DOUBLE, upper, 0, comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(field.data() + idx3(0, 0, local_nz, nx, ny), plane, MPI_DOUBLE,
                 upper, 1, field.data() + idx3(0, 0, 0, nx, ny), plane,
                 MPI_DOUBLE, lower, 1, comm, MPI_STATUS_IGNORE);

    // MPI_PROC_NULL leaves the physical boundary halo untouched; explicitly
    // set it to the adjacent owned plane for the clamped boundary condition.
    if (rank == 0) {
        std::copy_n(field.data() + idx3(0, 0, 1, nx, ny), plane,
                    field.data() + idx3(0, 0, 0, nx, ny));
    }
    if (rank == ranks - 1) {
        std::copy_n(field.data() + idx3(0, 0, local_nz, nx, ny), plane,
                    field.data() + idx3(0, 0, local_nz + 1, nx, ny));
    }
}

inline double laplacian(const std::vector<double>& field, const size_t x,
                        const size_t y, const size_t z, const size_t nx,
                        const size_t ny, const double dx2, const double dy2,
                        const double dz2) noexcept {
    const size_t center = idx3(x, y, z, nx, ny);
    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : x;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : y;
    return (field[idx3(xp, y, z, nx, ny)] + field[idx3(xn, y, z, nx, ny)] -
            2.0 * field[center]) / dx2 +
           (field[idx3(x, yp, z, nx, ny)] + field[idx3(x, yn, z, nx, ny)] -
            2.0 * field[center]) / dy2 +
           (field[idx3(x, y, z + 1, nx, ny)] + field[idx3(x, y, z - 1, nx, ny)] -
            2.0 * field[center]) / dz2;
}

void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t local_nz,
                              const double gamma, const double e_AA, const double e_BB,
                              const double e_AB) {
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t pos = idx3(x, y, z, nx, ny);
                const double cv = c[pos];
                mu[pos] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv
                        - gamma * laplacian(c, x, y, z, nx, ny, 1.0, 1.0, 1.0);
            }
        }
    }
}

void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu, const size_t nx, const size_t ny,
                        const size_t local_nz, const double D, const double dt) {
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t pos = idx3(x, y, z, nx, ny);
                cnew[pos] = cold[pos] + dt * D * laplacian(mu, x, y, z, nx, ny, 1.0, 1.0, 1.0);
            }
        }
    }
}

void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t local_nz, const size_t z_start, const size_t global_volume) {
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_id = (z_start + z - 1) * (nx * ny) + y * nx + x;
                const double pseudo = (((global_id + 1) * 1299709) % global_volume) /
                                      static_cast<double>(global_volume);
                c[idx3(x, y, z, nx, ny)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\nOptions:\n", progName);
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || static_cast<size_t>(ranks) > nz) {
        if (rank == 0) printf("Invalid grid or iteration count (requires MPI ranks <= Z size)\n");
        MPI_Finalize(); return 1;
    }

    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t extra = nz % static_cast<size_t>(ranks);
    const size_t local_nz = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t z_start = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const size_t plane_size = nx * ny;
    const size_t global_size = plane_size * nz;
    const size_t local_size = plane_size * (local_nz + 2);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\nInitializing concentration field...\n", ranks);
    }
    std::vector<double> cold(local_size), cnew(local_size), mu(local_size);
    initializeConcentration(cold, nx, ny, local_nz, z_start, global_size);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, nx, ny, local_nz, rank, ranks, MPI_COMM_WORLD);
        computeChemicalPotential(cold, mu, nx, ny, local_nz, 0.5, -(2.0 / 9.0), -(2.0 / 9.0), 2.0 / 9.0);
        exchangeHalos(mu, nx, ny, local_nz, rank, ranks, MPI_COMM_WORLD);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, 1.0, 0.01);
        std::swap(cold, cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", global_size * static_cast<double>(iterations) / max_elapsed / 1e6);
    }

    std::vector<double> global;
    std::vector<int> counts, displacements;
    if (printResults && rank == 0) { global.resize(global_size); counts.resize(ranks); displacements.resize(ranks); }
    const int local_count = static_cast<int>(local_nz * plane_size);
    if (printResults) {
        int count = local_count;
        MPI_Gather(&count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) { for (int r = 1; r < ranks; ++r) displacements[r] = displacements[r - 1] + counts[r - 1]; }
        MPI_Gatherv(cold.data() + plane_size, local_count, MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(global, "Concentration");
    }

    if (validate) {
        double local_min = cold[plane_size], local_max = local_min;
        bool local_valid = true;
        for (size_t i = 0; i < local_nz * plane_size; ++i) {
            const double v = cold[plane_size + i];
            local_valid = local_valid && std::isfinite(v);
            local_min = std::min(local_min, v); local_max = std::max(local_max, v);
        }
        double global_min = 0.0, global_max = 0.0;
        MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        int valid_flag = local_valid && local_min >= -10.0 && local_max <= 10.0;
        int all_valid = 0;
        MPI_Reduce(&valid_flag, &all_valid, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0) { printf("Validating result...\nConcentration range: [%.6f, %.6f]\nValidation: %s\n", global_min, global_max, all_valid ? "PASSED" : "FAILED"); }
        const int exit_code = (rank == 0 && !all_valid) ? 1 : 0;
        MPI_Finalize(); return exit_code;
    }
    MPI_Finalize();
    return 0;
}
