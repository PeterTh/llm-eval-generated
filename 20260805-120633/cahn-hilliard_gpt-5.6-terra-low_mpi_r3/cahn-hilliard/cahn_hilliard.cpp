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

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

/* The two ghost planes hold clamped values at physical boundaries and values
 * received from adjacent Z slabs everywhere else. */
static void exchange_halos(std::vector<double>& a, size_t plane, int lower, int upper, MPI_Comm comm) {
    MPI_Request requests[4];
    int nreq = 0;
    if (lower == MPI_PROC_NULL)
        std::copy_n(a.data() + plane, plane, a.data());
    else {
        MPI_Irecv(a.data(), static_cast<int>(plane), MPI_DOUBLE, lower, 1, comm, &requests[nreq++]);
        MPI_Isend(a.data() + plane, static_cast<int>(plane), MPI_DOUBLE, lower, 0, comm, &requests[nreq++]);
    }
    if (upper == MPI_PROC_NULL)
        std::copy_n(a.data() + plane * (a.size() / plane - 2), plane, a.data() + plane * (a.size() / plane - 1));
    else {
        const size_t last = a.size() - 2 * plane;
        MPI_Irecv(a.data() + a.size() - plane, static_cast<int>(plane), MPI_DOUBLE, upper, 0, comm, &requests[nreq++]);
        MPI_Isend(a.data() + last, static_cast<int>(plane), MPI_DOUBLE, upper, 1, comm, &requests[nreq++]);
    }
    MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
}

static void chemical_potential(const std::vector<double>& c, std::vector<double>& mu,
                               size_t nx, size_t ny, size_t local_nz,
                               double gamma, double e_AA, double e_BB, double e_AB) {
    const size_t plane = nx * ny;
    for (size_t z = 1; z <= local_nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                const size_t xp = x + (x + 1 < nx);
                const size_t xn = x - (x > 0);
                const size_t yp = y + (y + 1 < ny);
                const size_t yn = y - (y > 0);
                const double cv = c[i];
                const double cxx = c[idx3(xp,y,z,nx,ny)] + c[idx3(xn,y,z,nx,ny)] - 2.0 * cv;
                const double cyy = c[idx3(x,yp,z,nx,ny)] + c[idx3(x,yn,z,nx,ny)] - 2.0 * cv;
                const double czz = c[i + plane] + c[i - plane] - 2.0 * cv;
                const double lap = cxx + cyy + czz;
                mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                      + 3.0 * cv + cv * cv * cv - gamma * lap;
            }
}

static void update(std::vector<double>& next, const std::vector<double>& c, const std::vector<double>& mu,
                   size_t nx, size_t ny, size_t local_nz, double dt) {
    const size_t plane = nx * ny;
    for (size_t z = 1; z <= local_nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                const size_t xp = x + (x + 1 < nx);
                const size_t xn = x - (x > 0);
                const size_t yp = y + (y + 1 < ny);
                const size_t yn = y - (y > 0);
                const double cxx = mu[idx3(xp,y,z,nx,ny)] + mu[idx3(xn,y,z,nx,ny)] - 2.0 * mu[i];
                const double cyy = mu[idx3(x,yp,z,nx,ny)] + mu[idx3(x,yn,z,nx,ny)] - 2.0 * mu[i];
                const double czz = mu[i + plane] + mu[i - plane] - 2.0 * mu[i];
                const double lap = cxx + cyy + czz;
                next[i] = c[i] + dt * lap;
            }
}

static void usage(const char* p) {
    std::printf("Usage: %s [options]\n  -x <num>  X grid size (default: 64)\n  -y <num>  Y grid size (default: X)\n  -z <num>  Z grid size (default: X)\n  -i <num>  Time steps (default: 20)\n  -v        Enable validation\n  -r        Print results for external validation\n  -h        Show this help message\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20; bool validate = false, print_results_flag = false;
    bool bad_args = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!world_rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else bad_args = true;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || iterations < 0 || nx > std::numeric_limits<int>::max() / ny) bad_args = true;
    int any_bad = 0, local_bad = bad_args;
    MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if (any_bad) { if (!world_rank) { std::printf("Invalid command line arguments\n"); usage(argv[0]); } MPI_Finalize(); return 1; }

    const size_t plane = nx * ny;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) { if (!world_rank) std::printf("Grid plane is too large for MPI counts\n"); MPI_Finalize(); return 1; }
    const size_t base = nz / static_cast<size_t>(world_size), rem = nz % static_cast<size_t>(world_size);
    const size_t local_nz = base + (static_cast<size_t>(world_rank) < rem);
    const size_t z0 = static_cast<size_t>(world_rank) * base + std::min(static_cast<size_t>(world_rank), rem);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, local_nz ? 1 : MPI_UNDEFINED, world_rank, &comm);
    if (!local_nz) { MPI_Finalize(); return 0; }
    int rank, size; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
    const int lower = rank ? rank - 1 : MPI_PROC_NULL, upper = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;

    if (!world_rank) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\nInitializing concentration field...\nRunning Cahn-Hilliard simulation...\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }
    std::vector<double> cold((local_nz + 2) * plane), cnew((local_nz + 2) * plane), mu((local_nz + 2) * plane);
    const size_t volume = nx * ny * nz;
    for (size_t z = 0; z < local_nz; ++z) for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
        const size_t id = (z0 + z) * plane + y * nx + x;
        cold[idx3(x,y,z + 1,nx,ny)] = -1.0 + 2.0 * (((id + 1) * 1299709 % volume) / static_cast<double>(volume));
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        exchange_halos(cold, plane, lower, upper, comm);
        chemical_potential(cold, mu, nx, ny, local_nz, 0.5, -(2.0/9.0), -(2.0/9.0), 2.0/9.0);
        exchange_halos(mu, plane, lower, upper, comm);
        update(cnew, cold, mu, nx, ny, local_nz, 0.01);
        std::swap(cold, cnew);
    }
    double elapsed = MPI_Wtime() - start, max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!world_rank) {
        std::printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n", static_cast<long>(max_elapsed * 1000.0),
                    static_cast<double>(nx * ny * nz) * iterations / max_elapsed / 1e6);
    }

    if (print_results_flag) {
        std::vector<int> counts, offsets; std::vector<double> global;
        if (!rank) { counts.resize(size); offsets.resize(size); for (int r = 0; r < size; ++r) { const size_t n = base + (static_cast<size_t>(r) < rem); counts[r] = static_cast<int>(n * plane); offsets[r] = r ? offsets[r-1] + counts[r-1] : 0; } global.resize(volume); }
        MPI_Gatherv(cold.data() + plane, static_cast<int>(local_nz * plane), MPI_DOUBLE, global.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, comm);
        if (!rank) print_results(global, "Concentration");
    }
    int local_ok = 1; double local_min = cold[plane], local_max = cold[plane];
    for (size_t i = plane; i < (local_nz + 1) * plane; ++i) { if (!std::isfinite(cold[i])) local_ok = 0; local_min = std::min(local_min, cold[i]); local_max = std::max(local_max, cold[i]); }
    int ok; double minv, maxv;
    MPI_Reduce(&local_ok, &ok, 1, MPI_INT, MPI_LAND, 0, comm); MPI_Reduce(&local_min, &minv, 1, MPI_DOUBLE, MPI_MIN, 0, comm); MPI_Reduce(&local_max, &maxv, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    int result = 0;
    if (!rank && validate) { std::printf("Validating result...\nConcentration range: [%.6f, %.6f]\n", minv, maxv); result = ok && maxv <= 10.0 && minv >= -10.0; std::printf("Validation: %s\n", result ? "PASSED" : "FAILED"); }
    MPI_Bcast(&result, 1, MPI_INT, 0, comm);
    MPI_Comm_free(&comm); MPI_Finalize();
    return validate && !result;
}
