#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Local arrays contain one halo plane on either side of the owned Z slab.
inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

inline double laplacian(const std::vector<double>& a, size_t nx, size_t ny,
                        double dx, double dy, double dz,
                        size_t x, size_t y, size_t z) noexcept {
    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = z + 1;
    const size_t zn = z - 1;
    const size_t center = idx3(x, y, z, nx, ny);
    return (a[idx3(xp, y, z, nx, ny)] + a[idx3(xn, y, z, nx, ny)] - 2.0 * a[center]) / (dx * dx)
         + (a[idx3(x, yp, z, nx, ny)] + a[idx3(x, yn, z, nx, ny)] - 2.0 * a[center]) / (dy * dy)
         + (a[idx3(x, y, zp, nx, ny)] + a[idx3(x, y, zn, nx, ny)] - 2.0 * a[center]) / (dz * dz);
}

void exchangeHalos(std::vector<double>& a, size_t nx, size_t ny, size_t local_nz,
                   int rank, int nranks, MPI_Comm comm) {
    const int plane = static_cast<int>(nx * ny);
    const size_t first = idx3(0, 0, 1, nx, ny);
    const size_t last = idx3(0, 0, local_nz, nx, ny);
    const size_t lower_halo = idx3(0, 0, 0, nx, ny);
    const size_t upper_halo = idx3(0, 0, local_nz + 1, nx, ny);

    if (rank == 0)
        std::copy_n(a.begin() + static_cast<std::ptrdiff_t>(first), plane, a.begin() + static_cast<std::ptrdiff_t>(lower_halo));
    if (rank == nranks - 1)
        std::copy_n(a.begin() + static_cast<std::ptrdiff_t>(last), plane, a.begin() + static_cast<std::ptrdiff_t>(upper_halo));

    if (rank > 0)
        MPI_Sendrecv(a.data() + first, plane, MPI_DOUBLE, rank - 1, 0,
                     a.data() + lower_halo, plane, MPI_DOUBLE, rank - 1, 1, comm, MPI_STATUS_IGNORE);
    if (rank + 1 < nranks)
        MPI_Sendrecv(a.data() + last, plane, MPI_DOUBLE, rank + 1, 1,
                     a.data() + upper_halo, plane, MPI_DOUBLE, rank + 1, 0, comm, MPI_STATUS_IGNORE);
}

void initializeConcentration(std::vector<double>& c, size_t nx, size_t ny,
                             size_t local_nz, size_t z_start, size_t global_volume) {
    for (size_t z = 1; z <= local_nz; ++z) {
        const size_t global_z = z_start + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_id = global_z * nx * ny + y * nx + x;
                const size_t pseudo_id = ((global_id + 1) * 1299709) % global_volume;
                c[idx3(x, y, z, nx, ny)] = -1.0 + 2.0 * pseudo_id / static_cast<double>(global_volume);
            }
        }
    }
}

void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              size_t nx, size_t ny, size_t local_nz,
                              double dx, double dy, double dz, double gamma,
                              double e_AA, double e_BB, double e_AB) {
    for (size_t z = 1; z <= local_nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t p = idx3(x, y, z, nx, ny);
                const double cv = c[p];
                mu[p] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                      + 3.0 * cv + cv * cv * cv
                      - gamma * laplacian(c, nx, ny, dx, dy, dz, x, y, z);
            }
}

void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu, size_t nx, size_t ny,
                        size_t local_nz, double D, double dt, double dx, double dy, double dz) {
    for (size_t z = 1; z <= local_nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t p = idx3(x, y, z, nx, ny);
                cnew[p] = cold[p] + dt * D * laplacian(mu, nx, ny, dx, dy, dz, x, y, z);
            }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\nOptions:\n", progName);
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || nranks > static_cast<int>(nz)) {
        if (rank == 0) printf("Invalid dimensions/iterations (requires 1 <= MPI ranks <= Z dimension)\n");
        MPI_Finalize(); return 1;
    }
    const size_t base = nz / static_cast<size_t>(nranks), remainder = nz % static_cast<size_t>(nranks);
    const size_t local_nz = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t z_start = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    const size_t gridSize = nx * ny * nz, local_size = nx * ny * (local_nz + 2);
    if (nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Grid plane is too large for MPI count limits\n");
        MPI_Finalize(); return 1;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\nInitializing concentration field...\n", nranks);
    }
    const double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0, gamma = 0.5, D = 1.0;
    std::vector<double> cold(local_size), cnew(local_size), mu(local_size);
    initializeConcentration(cold, nx, ny, local_nz, z_start, gridSize);
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, nx, ny, local_nz, rank, nranks, MPI_COMM_WORLD);
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        exchangeHalos(mu, nx, ny, local_nz, rank, nranks, MPI_COMM_WORLD);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz);
        std::swap(cold, cnew);
    }
    const double local_time = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&local_time, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double safe_duration = std::max(duration, std::numeric_limits<double>::min());
        printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", duration * 1000.0,
               static_cast<double>(gridSize) * iterations / safe_duration / 1.0e6);
    }

    const size_t owned_size = nx * ny * local_nz;
    std::vector<double> global;
    std::vector<int> counts, displs;
    if (printResults || rank == 0) {
        if (rank == 0) { global.resize(gridSize); counts.resize(nranks); displs.resize(nranks); }
    }
    if (printResults) {
        int send_count = static_cast<int>(owned_size);
        if (rank == 0) for (int r = 0; r < nranks; ++r) {
            const size_t rnz = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
            counts[r] = static_cast<int>(nx * ny * rnz);
            displs[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder)) * nx * ny);
        }
        MPI_Gatherv(cold.data() + nx * ny, send_count, MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(global, "Concentration");
    }
    if (validate) {
        int local_bad = 0;
        double local_min = cold[nx * ny], local_max = local_min;
        for (size_t i = nx * ny; i < nx * ny * (local_nz + 1); ++i) {
            local_bad |= std::isfinite(cold[i]) ? 0 : 1;
            local_min = std::min(local_min, cold[i]); local_max = std::max(local_max, cold[i]);
        }
        int bad = 0; double minVal = 0.0, maxVal = 0.0;
        MPI_Reduce(&local_bad, &bad, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_min, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\nConcentration range: [%.6f, %.6f]\n", minVal, maxVal);
            const bool valid = !bad && maxVal <= 10.0 && minVal >= -10.0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize(); return valid ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
