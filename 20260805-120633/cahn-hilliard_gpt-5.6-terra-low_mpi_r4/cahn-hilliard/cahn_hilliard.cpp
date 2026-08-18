#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

static inline double laplacian(const std::vector<double>& a, size_t x, size_t y, size_t z,
                               size_t nx, size_t ny) noexcept {
    const size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
    const size_t i = idx3(x, y, z, nx, ny);
    return a[idx3(xp, y, z, nx, ny)] + a[idx3(xn, y, z, nx, ny)]
         + a[idx3(x, yp, z, nx, ny)] + a[idx3(x, yn, z, nx, ny)]
         + a[idx3(x, y, z + 1, nx, ny)] + a[idx3(x, y, z - 1, nx, ny)] - 6.0 * a[i];
}

// Exchange the two z faces.  MPI_PROC_NULL faces are filled by copying the
// adjacent physical plane, exactly implementing the original clamped boundary.
struct HaloExchange { std::array<MPI_Request, 4> requests; int count = 0; };

static HaloExchange begin_halo_exchange(std::vector<double>& a, size_t plane, size_t local_nz,
                                        int below, int above, MPI_Comm comm) {
    HaloExchange exchange{};
    if (below == MPI_PROC_NULL) {
        std::copy_n(a.data() + plane, plane, a.data());
    } else {
        MPI_Irecv(a.data(), static_cast<int>(plane), MPI_DOUBLE, below, 1, comm, &exchange.requests[exchange.count++]);
        MPI_Isend(a.data() + plane, static_cast<int>(plane), MPI_DOUBLE, below, 0, comm, &exchange.requests[exchange.count++]);
    }
    if (above == MPI_PROC_NULL) {
        std::copy_n(a.data() + local_nz * plane, plane, a.data() + (local_nz + 1) * plane);
    } else {
        MPI_Irecv(a.data() + (local_nz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, above, 0, comm, &exchange.requests[exchange.count++]);
        MPI_Isend(a.data() + local_nz * plane, static_cast<int>(plane), MPI_DOUBLE, above, 1, comm, &exchange.requests[exchange.count++]);
    }
    return exchange;
}

static void finish_halo_exchange(HaloExchange& exchange) {
    MPI_Waitall(exchange.count, exchange.requests.data(), MPI_STATUSES_IGNORE);
}

static void chemical_range(const std::vector<double>& c, std::vector<double>& mu, size_t first, size_t last,
                           size_t nx, size_t ny, double gamma, double e_AA, double e_BB, double e_AB) {
    for (size_t z = first; z <= last; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                const double v = c[i];
                mu[i] = 4.5 * ((v + 1.0) * e_AA + (v - 1.0) * e_BB - 2.0 * v * e_AB)
                      + 3.0 * v + v * v * v - gamma * laplacian(c, x, y, z, nx, ny);
            }
}

static void update_range(std::vector<double>& dst, const std::vector<double>& src, const std::vector<double>& mu,
                         size_t first, size_t last, size_t nx, size_t ny, double dt) {
    for (size_t z = first; z <= last; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                dst[i] = src[i] + dt * laplacian(mu, x, y, z, nx, ny);
            }
}

static void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -x <num>  X grid (default 64)\n  -y <num>  Y grid (default X)\n"
                "  -z <num>  Z grid (default X)\n  -i <num>  time steps (default 20)\n"
                "  -v        enable validation\n  -r        print results\n  -h        show help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    int argument_error = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else argument_error = 1;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (argument_error || !nx || !ny || !nz || iterations < 0 || static_cast<size_t>(ranks) > nz) {
        if (!rank) { if (argument_error) printUsage(argv[0]); else std::fprintf(stderr, "Invalid grid/iteration count or more MPI ranks than z planes.\n"); }
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (!rank) std::fprintf(stderr, "A z plane is too large for MPI count arguments.\n");
        MPI_Finalize(); return 1;
    }
    const size_t base = nz / ranks, extra = nz % ranks;
    const size_t local_nz = base + (static_cast<size_t>(rank) < extra);
    const size_t global_z0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const int below = rank ? rank - 1 : MPI_PROC_NULL;
    const int above = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    const size_t local_size = local_nz * plane;
    std::vector<double> cold((local_nz + 2) * plane), cnew((local_nz + 2) * plane), mu((local_nz + 2) * plane);
    for (size_t z = 1; z <= local_nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_id = (global_z0 + z - 1) * plane + y * nx + x;
                cold[idx3(x, y, z, nx, ny)] = -1.0 + 2.0 * (((global_id + 1) * 1299709 % (nx * ny * nz)) / static_cast<double>(nx * ny * nz));
            }
    if (!rank) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\nInitializing concentration field...\nRunning Cahn-Hilliard simulation...\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }
    constexpr double gamma = 0.5, e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0, dt = 0.01;
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (int t = 0; t < iterations; ++t) {
        auto c_exchange = begin_halo_exchange(cold, plane, local_nz, below, above, MPI_COMM_WORLD);
        // Interior planes use no remote data and hide most halo latency.
        if (local_nz > 2) chemical_range(cold, mu, 2, local_nz - 1, nx, ny, gamma, e_AA, e_BB, e_AB);
        finish_halo_exchange(c_exchange);
        chemical_range(cold, mu, 1, 1, nx, ny, gamma, e_AA, e_BB, e_AB);
        if (local_nz > 1) chemical_range(cold, mu, local_nz, local_nz, nx, ny, gamma, e_AA, e_BB, e_AB);

        auto mu_exchange = begin_halo_exchange(mu, plane, local_nz, below, above, MPI_COMM_WORLD);
        if (local_nz > 2) update_range(cnew, cold, mu, 2, local_nz - 1, nx, ny, dt);
        finish_halo_exchange(mu_exchange);
        update_range(cnew, cold, mu, 1, 1, nx, ny, dt);
        if (local_nz > 1) update_range(cnew, cold, mu, local_nz, local_nz, nx, ny, dt);
        std::swap(cold, cnew);
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<int> counts, displs;
    std::vector<double> global;
    const int local_count = static_cast<int>(local_size);
    if (!rank) { counts.resize(ranks); displs.resize(ranks); }
    MPI_Gather(&local_count, 1, MPI_INT, rank ? nullptr : counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!rank) { int offset = 0; for (int r = 0; r < ranks; ++r) { displs[r] = offset; offset += counts[r]; } global.resize(offset); }
    MPI_Gatherv(cold.data() + plane, local_count, MPI_DOUBLE, rank ? nullptr : global.data(), rank ? nullptr : counts.data(), rank ? nullptr : displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int result = 0;
    if (!rank) {
        const auto ms = static_cast<long>(max_elapsed * 1000.0);
        std::printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n", ms,
                    max_elapsed > 0.0 ? (static_cast<double>(nx * ny * nz) * iterations / max_elapsed / 1e6) : 0.0);
        if (printResults) print_results(global, "Concentration");
        if (validate) {
            const auto [lo, hi] = std::minmax_element(global.begin(), global.end());
            const bool finite = std::all_of(global.begin(), global.end(), [](double v) { return std::isfinite(v); });
            std::printf("Validating result...\nConcentration range: [%.6f, %.6f]\n", *lo, *hi);
            result = !finite || *hi > 10.0 || *lo < -10.0;
            std::printf("Validation: %s\n", result ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
