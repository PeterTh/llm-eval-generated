#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Slab {
    size_t first_z;
    size_t nz;
};

Slab slab_for_rank(const size_t global_nz, const int rank, const int ranks) {
    const size_t base = global_nz / static_cast<size_t>(ranks);
    const size_t remainder = global_nz % static_cast<size_t>(ranks);
    const size_t r = static_cast<size_t>(rank);
    return {r * base + std::min(r, remainder), base + (r < remainder ? 1 : 0)};
}

inline double laplacian(const std::vector<double>& field, const size_t nx,
                        const size_t ny,
                        const double dx, const double dy, const double dz,
                        const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    // z=1 and z=local_nz deliberately address the halo planes; those
    // planes contain either a neighbor's data or a clamped boundary copy.
    const size_t zp = z + 1;
    const size_t zn = z - 1;
    const double center = field[idx3(x, y, z, nx, ny)];
    return (field[idx3(xp, y, z, nx, ny)] + field[idx3(xn, y, z, nx, ny)] - 2.0 * center) / (dx * dx)
         + (field[idx3(x, yp, z, nx, ny)] + field[idx3(x, yn, z, nx, ny)] - 2.0 * center) / (dy * dy)
         + (field[idx3(x, y, zp, nx, ny)] + field[idx3(x, y, zn, nx, ny)] - 2.0 * center) / (dz * dz);
}

void exchange_halos(std::vector<double>& field, const size_t plane,
                    const size_t local_nz, const int rank, const int ranks,
                    MPI_Comm comm) {
    // The extra planes are physical clamped-boundary values at the ends.
    if (rank == 0) {
        std::copy_n(field.data() + plane, plane, field.data());
    }
    if (rank == ranks - 1) {
        std::copy_n(field.data() + local_nz * plane, plane,
                    field.data() + (local_nz + 1) * plane);
    }

    MPI_Request requests[4];
    int request_count = 0;
    if (rank > 0) {
        MPI_Irecv(field.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1,
                  0, comm, &requests[request_count++]);
        MPI_Isend(field.data() + plane, static_cast<int>(plane), MPI_DOUBLE,
                  rank - 1, 1, comm, &requests[request_count++]);
    }
    if (rank + 1 < ranks) {
        MPI_Irecv(field.data() + (local_nz + 1) * plane,
                  static_cast<int>(plane), MPI_DOUBLE, rank + 1, 1, comm,
                  &requests[request_count++]);
        MPI_Isend(field.data() + local_nz * plane, static_cast<int>(plane),
                  MPI_DOUBLE, rank + 1, 0, comm, &requests[request_count++]);
    }
    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
}

void initialize_concentration(std::vector<double>& c, const size_t nx,
                              const size_t ny, const size_t local_nz,
                              const size_t first_z, const size_t global_nz) {
    const size_t plane = nx * ny;
    const size_t volume = plane * global_nz;
    for (size_t z = 1; z <= local_nz; ++z) {
        const size_t global_z = first_z + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_id = global_z * plane + y * nx + x;
                const double pseudo = (((global_id + 1) * 1299709) % volume)
                                    / static_cast<double>(volume);
                c[idx3(x, y, z, nx, ny)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

void compute_chemical_potential(const std::vector<double>& c,
                                std::vector<double>& mu, const size_t nx,
                                const size_t ny, const size_t local_nz,
                                const double dx, const double dy,
                                const double dz, const double gamma,
                                const double e_AA, const double e_BB,
                                const double e_AB) {
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t id = idx3(x, y, z, nx, ny);
                const double cv = c[id];
                mu[id] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                       + 3.0 * cv + cv * cv * cv
                       - gamma * laplacian(c, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

void update(std::vector<double>& cnew, const std::vector<double>& cold,
            const std::vector<double>& mu, const size_t nx, const size_t ny,
            const size_t local_nz, const double D, const double dt,
            const double dx, const double dy, const double dz) {
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t id = idx3(x, y, z, nx, ny);
                cnew[id] = cold[id] + dt * D * laplacian(mu, nx, ny,
                                                           dx, dy, dz, x, y, z);
            }
        }
    }
}

void print_usage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -x <num> Grid size in X (default: 64)\n");
    printf("  -y <num> Grid size in Y (default: X)\n");
    printf("  -z <num> Grid size in Z (default: X)\n");
    printf("  -i <num> Number of time steps (default: 20)\n");
    printf("  -v       Enable validation\n  -r       Print results\n  -h       Show this help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) print_results_flag = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) print_usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); print_usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || ranks > static_cast<int>(nz)) {
        if (rank == 0) printf("Invalid dimensions/iterations (MPI ranks must not exceed z dimension)\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Grid plane is too large for MPI\n");
        MPI_Finalize(); return 1;
    }
    const Slab slab = slab_for_rank(nz, rank, ranks);
    const size_t local_size = (slab.nz + 2) * plane;
    std::vector<double> cold(local_size), cnew(local_size), mu(local_size);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\nInitializing concentration field...\nRunning Cahn-Hilliard simulation...\n", ranks);
    }
    initialize_concentration(cold, nx, ny, slab.nz, slab.first_z, nz);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        exchange_halos(cold, plane, slab.nz, rank, ranks, MPI_COMM_WORLD);
        compute_chemical_potential(cold, mu, nx, ny, slab.nz, 1.0, 1.0, 1.0,
                                   0.5, -(2.0 / 9.0), -(2.0 / 9.0), 2.0 / 9.0);
        exchange_halos(mu, plane, slab.nz, rank, ranks, MPI_COMM_WORLD);
        update(cnew, cold, mu, nx, ny, slab.nz, 1.0, 0.01, 1.0, 1.0, 1.0);
        cold.swap(cnew);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n",
               max_elapsed * 1000.0,
               static_cast<double>(nx * ny * nz) * iterations / max_elapsed / 1.0e6);
    }

    const int local_count = static_cast<int>(slab.nz * plane);
    std::vector<double> global;
    std::vector<int> counts, displacements;
    if (rank == 0 && (print_results_flag || validate)) {
        global.resize(nx * ny * nz);
        counts.resize(ranks); displacements.resize(ranks);
        for (int r = 0; r < ranks; ++r) {
            const Slab s = slab_for_rank(nz, r, ranks);
            counts[r] = static_cast<int>(s.nz * plane);
            displacements[r] = static_cast<int>(s.first_z * plane);
        }
    }
    int result = 0;
    if (print_results_flag || validate) {
        MPI_Gatherv(cold.data() + plane, local_count, MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0 && print_results_flag) print_results(global, "Concentration");
    }
    if (validate) {
        int local_bad = 0;
        double local_min = std::numeric_limits<double>::infinity();
        double local_max = -std::numeric_limits<double>::infinity();
        for (size_t i = plane; i < (slab.nz + 1) * plane; ++i) {
            if (!std::isfinite(cold[i])) local_bad = 1;
            local_min = std::min(local_min, cold[i]);
            local_max = std::max(local_max, cold[i]);
        }
        int bad = 0; double min_value = 0.0, max_value = 0.0;
        MPI_Reduce(&local_bad, &bad, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_min, &min_value, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &max_value, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\nConcentration range: [%.6f, %.6f]\n", min_value, max_value);
            result = bad || max_value > 10.0 || min_value < -10.0;
            printf("Validation: %s\n", result ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
