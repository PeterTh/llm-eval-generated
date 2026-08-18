#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static void usage(const char* p) {
    std::printf("Usage: %s [options]\n  -x <num>  Grid X (default 64)\n  -y <num>  Grid Y\n"
                "  -z <num>  Grid Z\n  -i <num>  Time steps (default 20)\n"
                "  -v        Validate\n  -r        Print results\n  -h        Help\n", p);
}

struct Slab {
    size_t nx, ny, plane, local_z, z0;
    int rank, nranks, lower, upper;
};

// Exchange the two Z faces. Physical boundary halos copy the adjacent plane,
// exactly reproducing the original clamped (zero normal derivative) boundary.
static void begin_halo(std::vector<double>& a, const Slab& s, MPI_Request req[4]) {
    const int n = static_cast<int>(s.plane);
    int q = 0;
    if (s.lower != MPI_PROC_NULL) {
        MPI_Irecv(a.data(), n, MPI_DOUBLE, s.lower, 17, MPI_COMM_WORLD, &req[q++]);
        MPI_Isend(a.data() + s.plane, n, MPI_DOUBLE, s.lower, 18, MPI_COMM_WORLD, &req[q++]);
    } else {
        std::copy_n(a.data() + s.plane, s.plane, a.data());
    }
    if (s.upper != MPI_PROC_NULL) {
        MPI_Irecv(a.data() + (s.local_z + 1) * s.plane, n, MPI_DOUBLE,
                  s.upper, 18, MPI_COMM_WORLD, &req[q++]);
        MPI_Isend(a.data() + s.local_z * s.plane, n, MPI_DOUBLE,
                  s.upper, 17, MPI_COMM_WORLD, &req[q++]);
    } else {
        std::copy_n(a.data() + s.local_z * s.plane, s.plane,
                    a.data() + (s.local_z + 1) * s.plane);
    }
    while (q < 4) req[q++] = MPI_REQUEST_NULL;
}

static inline void wait_halo(MPI_Request req[4]) { MPI_Waitall(4, req, MPI_STATUSES_IGNORE); }

template<bool Chemical>
static void stencil_range(const std::vector<double>& in, std::vector<double>& out,
                          const Slab& s, size_t zb, size_t ze,
                          const std::vector<double>* baseline = nullptr) {
    constexpr double gamma = 0.5, dtD = 0.01;
    const size_t nx = s.nx, ny = s.ny, p = s.plane;
    for (size_t z = zb; z < ze; ++z) {
        const size_t zo = z * p;
        for (size_t y = 0; y < ny; ++y) {
            const size_t ym = y ? y - 1 : y, yp = y + 1 < ny ? y + 1 : y;
            const size_t row = zo + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xm = x ? x - 1 : x, xp = x + 1 < nx ? x + 1 : x;
                const size_t i = row + x;
                const double v = in[i];
                const double lap = in[row + xm] + in[row + xp]
                                 + in[zo + ym * nx + x] + in[zo + yp * nx + x]
                                 + in[i - p] + in[i + p] - 6.0 * v;
                if constexpr (Chemical) {
                    // Algebraically identical to the original free-energy derivative.
                    out[i] = 4.5 * ((v + 1.0) * (-2.0 / 9.0)
                           + (v - 1.0) * (-2.0 / 9.0) - 2.0 * v * (2.0 / 9.0))
                           + 3.0 * v + v * v * v - gamma * lap;
                } else {
                    out[i] = (*baseline)[i] + dtD * lap;
                }
            }
        }
    }
}

template<bool Chemical>
static void distributed_stencil(std::vector<double>& in, std::vector<double>& out,
                                const Slab& s, const std::vector<double>* baseline = nullptr) {
    MPI_Request req[4];
    begin_halo(in, s, req);
    if (s.local_z > 2) stencil_range<Chemical>(in, out, s, 2, s.local_z, baseline);
    wait_halo(req);
    stencil_range<Chemical>(in, out, s, 1, std::min<size_t>(2, s.local_z + 1), baseline);
    if (s.local_z > 1) stencil_range<Chemical>(in, out, s, s.local_z, s.local_z + 1, baseline);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, print = false, ok = true, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) print = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else { if (!rank) std::printf("Unknown option: %s\n", argv[i]); ok = false; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (help || !ok) {
        if (!rank) usage(argv[0]);
        MPI_Finalize();
        return ok ? 0 : 1;
    }
    if (!nx || !ny || !nz || iterations < 0 || static_cast<size_t>(nranks) > nz ||
        nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (!rank) std::fprintf(stderr, "Invalid grid/iterations, or MPI ranks exceed Z planes/MPI count limit\n");
        MPI_Finalize();
        return 1;
    }

    const size_t base = nz / nranks, rem = nz % nranks;
    const size_t lz = base + (static_cast<size_t>(rank) < rem);
    const size_t z0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    Slab s{nx, ny, nx * ny, lz, z0, rank, nranks,
           rank ? rank - 1 : MPI_PROC_NULL, rank + 1 < nranks ? rank + 1 : MPI_PROC_NULL};
    std::vector<double> cold((lz + 2) * s.plane), cnew((lz + 2) * s.plane), mu((lz + 2) * s.plane);
    const size_t volume = nx * ny * nz;
    for (size_t z = 1; z <= lz; ++z)
        for (size_t i = 0; i < s.plane; ++i) {
            const size_t gid = (z0 + z - 1) * s.plane + i;
            cold[z * s.plane + i] = -1.0 + 2.0 * (((gid + 1) * size_t(1299709)) % volume) / double(volume);
        }

    if (!rank) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\nValidation: %s\nMPI processes: %d\n", iterations, validate ? "enabled" : "disabled", nranks);
        std::printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        distributed_stencil<true>(cold, mu, s);
        distributed_stencil<false>(mu, cnew, s, &cold);
        cold.swap(cnew);
    }
    const double local_time = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!rank) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        const double mcups = elapsed > 0 ? double(volume) * iterations / elapsed / 1e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    if (validate) {
        double local_min = std::numeric_limits<double>::infinity();
        double local_max = -local_min;
        int local_valid = 1;
        for (size_t i = s.plane; i < (lz + 1) * s.plane; ++i) {
            local_min = std::min(local_min, cold[i]); local_max = std::max(local_max, cold[i]);
            if (!std::isfinite(cold[i])) local_valid = 0;
        }
        double global_min, global_max; int global_valid;
        MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        if (!rank) {
            std::printf("Validating result...\nConcentration range: [%.6f, %.6f]\n", global_min, global_max);
            global_valid = global_valid && global_max <= 10.0 && global_min >= -10.0;
            std::printf("Validation: %s\n", global_valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&global_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        ok = global_valid;
    }
    if (print) {
        std::vector<int> counts(nranks), displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            size_t rz = base + (static_cast<size_t>(r) < rem);
            size_t rz0 = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
            counts[r] = static_cast<int>(rz * s.plane); displs[r] = static_cast<int>(rz0 * s.plane);
        }
        std::vector<double> all(rank == 0 ? volume : 0);
        MPI_Gatherv(cold.data() + s.plane, static_cast<int>(lz * s.plane), MPI_DOUBLE,
                    rank == 0 ? all.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(all, "Concentration");
    }
    MPI_Finalize();
    return ok ? 0 : 1;
}
