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

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

struct Slab {
    size_t first_z;
    size_t nz;
};

Slab slab_for(int rank, int size, size_t nz) {
    const size_t q = nz / static_cast<size_t>(size);
    const size_t r = nz % static_cast<size_t>(size);
    return {static_cast<size_t>(rank) * q + std::min(static_cast<size_t>(rank), r),
            q + (static_cast<size_t>(rank) < r)};
}

// Exchange the two contiguous Z faces. Computation of planes that do not use a
// received halo is done while these requests are in flight.
void begin_halo(std::vector<double>& a, size_t plane, size_t local_nz,
                int rank, int nranks, MPI_Request req[4]) {
    int n = 0;
    if (rank > 0) {
        MPI_Irecv(a.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1,
                  MPI_COMM_WORLD, &req[n++]);
        MPI_Isend(a.data() + plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0,
                  MPI_COMM_WORLD, &req[n++]);
    } else {
        std::copy_n(a.data() + plane, plane, a.data()); // clamped global face
    }
    if (rank + 1 < nranks) {
        MPI_Irecv(a.data() + (local_nz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE,
                  rank + 1, 0, MPI_COMM_WORLD, &req[n++]);
        MPI_Isend(a.data() + local_nz * plane, static_cast<int>(plane), MPI_DOUBLE,
                  rank + 1, 1, MPI_COMM_WORLD, &req[n++]);
    } else {
        std::copy_n(a.data() + local_nz * plane, plane,
                    a.data() + (local_nz + 1) * plane); // clamped global face
    }
    for (; n < 4; ++n) req[n] = MPI_REQUEST_NULL;
}

inline double laplacian(const double* a, size_t x, size_t y, size_t z,
                        size_t nx, size_t ny, size_t plane) noexcept {
    const size_t i = z * plane + y * nx + x;
    const size_t xm = x ? i - 1 : i;
    const size_t xp = x + 1 < nx ? i + 1 : i;
    const size_t ym = y ? i - nx : i;
    const size_t yp = y + 1 < ny ? i + nx : i;
    return (a[xp] + a[xm] - 2.0 * a[i]) +
           (a[yp] + a[ym] - 2.0 * a[i]) +
           (a[i + plane] + a[i - plane] - 2.0 * a[i]);
}

void chemical_planes(const std::vector<double>& c, std::vector<double>& mu,
                     size_t nx, size_t ny, size_t plane, size_t z0, size_t z1) {
    const double* cp = c.data();
    double* mp = mu.data();
    for (size_t z = z0; z < z1; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                const double v = cp[i];
                // Algebra is deliberately kept in the same form as the serial code.
                mp[i] = 4.5 * ((v + 1.0) * (-2.0 / 9.0) +
                               (v - 1.0) * (-2.0 / 9.0) -
                               2.0 * v * (2.0 / 9.0)) +
                        3.0 * v + v * v * v - 0.5 * laplacian(cp, x, y, z, nx, ny, plane);
            }
}

void update_planes(std::vector<double>& out, const std::vector<double>& in,
                   const std::vector<double>& mu, size_t nx, size_t ny, size_t plane,
                   size_t z0, size_t z1) {
    double* op = out.data();
    const double* ip = in.data();
    const double* mp = mu.data();
    for (size_t z = z0; z < z1; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                op[i] = ip[i] + 0.01 * laplacian(mp, x, y, z, nx, ny, plane);
            }
}

void print_usage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -x <num>  Grid size in X (default: 64)\n"
                "  -y <num>  Grid size in Y (default: X)\n"
                "  -z <num>  Grid size in Z (default: X)\n"
                "  -i <num>  Time steps (default: 20)\n"
                "  -v        Enable validation\n"
                "  -r        Print results for external validation\n"
                "  -h        Show this help\n");
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, print_results_flag = false, help = false;
    int parse_error = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else parse_error = 1;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (help || parse_error || !nx || !ny || !nz || iterations < 0 ||
        static_cast<size_t>(nranks) > nz || nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            if (!help) std::fprintf(stderr, "Invalid arguments (MPI ranks must not exceed Z planes).\n");
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return help ? 0 : 1;
    }

    const Slab slab = slab_for(rank, nranks, nz);
    const size_t plane = nx * ny;
    const size_t local_count = slab.nz * plane;
    std::vector<double> cold((slab.nz + 2) * plane);
    std::vector<double> cnew((slab.nz + 2) * plane);
    std::vector<double> mu((slab.nz + 2) * plane);

    const size_t volume = nx * ny * nz;
    for (size_t z = 1; z <= slab.nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear = (slab.first_z + z - 1) * plane + y * nx + x;
                const double pseudo = (((linear + 1) * static_cast<size_t>(1299709)) % volume) /
                                      static_cast<double>(volume);
                cold[idx3(x, y, z, nx, ny)] = -1.0 + 2.0 * pseudo;
            }

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n"
                    "Grid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n"
                    "MPI processes: %d\nInitializing concentration field...\n"
                    "Running Cahn-Hilliard simulation...\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled", nranks);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    MPI_Request req[4];
    for (int t = 0; t < iterations; ++t) {
        begin_halo(cold, plane, slab.nz, rank, nranks, req);
        if (slab.nz > 2) chemical_planes(cold, mu, nx, ny, plane, 2, slab.nz);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        chemical_planes(cold, mu, nx, ny, plane, 1, 2);
        if (slab.nz > 1) chemical_planes(cold, mu, nx, ny, plane, slab.nz, slab.nz + 1);

        begin_halo(mu, plane, slab.nz, rank, nranks, req);
        if (slab.nz > 2) update_planes(cnew, cold, mu, nx, ny, plane, 2, slab.nz);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        update_planes(cnew, cold, mu, nx, ny, plane, 1, 2);
        if (slab.nz > 1) update_planes(cnew, cold, mu, nx, ny, plane, slab.nz, slab.nz + 1);
        cold.swap(cnew);
    }
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int result = 0;
    if (rank == 0) {
        std::printf("Computation time: %.0f ms\n", elapsed * 1000.0);
        const double mcups = elapsed > 0.0 ? static_cast<double>(volume) * iterations / elapsed / 1e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (print_results_flag) {
        std::vector<int> counts(rank == 0 ? nranks : 0), displs(rank == 0 ? nranks : 0);
        if (rank == 0) for (int r = 0; r < nranks; ++r) {
            const Slab s = slab_for(r, nranks, nz);
            counts[r] = static_cast<int>(s.nz * plane);
            displs[r] = static_cast<int>(s.first_z * plane);
        }
        std::vector<double> global(rank == 0 ? volume : 0);
        MPI_Gatherv(cold.data() + plane, static_cast<int>(local_count), MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        if (rank == 0) print_results(global, "Concentration");
    }

    if (validate) {
        int local_ok = 1;
        double local_min = std::numeric_limits<double>::infinity();
        double local_max = -std::numeric_limits<double>::infinity();
        for (size_t i = plane; i < (slab.nz + 1) * plane; ++i) {
            local_ok &= std::isfinite(cold[i]);
            local_min = std::min(local_min, cold[i]);
            local_max = std::max(local_max, cold[i]);
        }
        int global_ok = 0;
        double global_min = 0.0, global_max = 0.0;
        MPI_Reduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::printf("Validating result...\nConcentration range: [%.6f, %.6f]\n", global_min, global_max);
            global_ok = global_ok && global_max <= 10.0 && global_min >= -10.0;
            std::printf("Validation: %s\n", global_ok ? "PASSED" : "FAILED");
            result = global_ok ? 0 : 1;
        }
        MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return result;
}
