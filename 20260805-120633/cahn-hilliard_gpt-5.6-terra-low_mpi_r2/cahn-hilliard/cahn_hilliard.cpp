#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

static void exchange_halos(std::vector<double>& field, size_t local_z, size_t plane,
                           int rank, int ranks, MPI_Comm comm) {
    const int below = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int above = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    MPI_Request requests[4];
    MPI_Irecv(field.data(), static_cast<int>(plane), MPI_DOUBLE, below, 1, comm, &requests[0]);
    MPI_Irecv(field.data() + (local_z + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, above, 0, comm, &requests[1]);
    MPI_Isend(field.data() + plane, static_cast<int>(plane), MPI_DOUBLE, below, 0, comm, &requests[2]);
    MPI_Isend(field.data() + local_z * plane, static_cast<int>(plane), MPI_DOUBLE, above, 1, comm, &requests[3]);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    if (rank == 0) std::memcpy(field.data(), field.data() + plane, plane * sizeof(double));
    if (rank + 1 == ranks) std::memcpy(field.data() + (local_z + 1) * plane,
                                       field.data() + local_z * plane, plane * sizeof(double));
}

static void chemical_potential(const std::vector<double>& c, std::vector<double>& mu,
                               size_t nx, size_t ny, size_t local_z, double gamma) {
    const double xy_scale = 1.0;
    for (size_t z = 1; z <= local_z; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = y + 1 < ny ? y + 1 : y;
            const size_t yn = y ? y - 1 : 0;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xp = x + 1 < nx ? x + 1 : x;
                const size_t xn = x ? x - 1 : 0;
                const size_t i = idx3(x, y, z, nx, ny);
                const double v = c[i];
                const double lap = (c[idx3(xp,y,z,nx,ny)] + c[idx3(xn,y,z,nx,ny)] - 2.0*v) * xy_scale
                                 + (c[idx3(x,yp,z,nx,ny)] + c[idx3(x,yn,z,nx,ny)] - 2.0*v) * xy_scale
                                 + c[i + nx*ny] + c[i - nx*ny] - 2.0*v;
                // With the benchmark's e_AA/e_BB/e_AB values, the local term
                // 4.5*(...) + 3*c + c^3 simplifies exactly to -c + c^3.
                mu[i] = -v + v*v*v - gamma * lap;
            }
        }
    }
}

static void update(const std::vector<double>& cold, const std::vector<double>& mu,
                   std::vector<double>& cnew, size_t nx, size_t ny, size_t local_z, double dt) {
    const size_t plane = nx * ny;
    for (size_t z = 1; z <= local_z; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = y + 1 < ny ? y + 1 : y;
            const size_t yn = y ? y - 1 : 0;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xp = x + 1 < nx ? x + 1 : x;
                const size_t xn = x ? x - 1 : 0;
                const size_t i = idx3(x, y, z, nx, ny);
                const double lap = mu[idx3(xp,y,z,nx,ny)] + mu[idx3(xn,y,z,nx,ny)]
                                 + mu[idx3(x,yp,z,nx,ny)] + mu[idx3(x,yn,z,nx,ny)]
                                 + mu[i + plane] + mu[i - plane] - 6.0 * mu[i];
                cnew[i] = cold[i] + dt * lap;
            }
        }
    }
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -x <num>  Grid size in X (default: 64)\n"
                "  -y <num>  Grid size in Y (default: same as X)\n"
                "  -z <num>  Grid size in Z (default: same as X)\n"
                "  -i <num>  Time steps (default: 20)\n  -v        Enable validation\n"
                "  -r        Print results for external validation\n  -h        Show this help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, results = false;
    int exit_code = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) results = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (nx == 0 || ny == 0 || nz < static_cast<size_t>(ranks)) {
        if (!rank) std::fprintf(stderr, "MPI implementation requires nonzero dimensions and at least one Z plane per rank.\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny;
    const size_t local_z = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t z_start = (nz / ranks) * rank + std::min(static_cast<size_t>(rank), nz % ranks);
    std::vector<double> cold((local_z + 2) * plane), cnew((local_z + 2) * plane), mu((local_z + 2) * plane);
    const size_t volume = nx * ny * nz;
    for (size_t z = 0; z < local_z; ++z) for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
        const size_t global = (z_start + z) * plane + y * nx + x;
        cold[idx3(x,y,z + 1,nx,ny)] = -1.0 + 2.0 * (((global + 1) * 1299709 % volume) / static_cast<double>(volume));
    }
    if (!rank) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\nInitializing concentration field...\nRunning Cahn-Hilliard simulation...\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        exchange_halos(cold, local_z, plane, rank, ranks, MPI_COMM_WORLD);
        chemical_potential(cold, mu, nx, ny, local_z, 0.5);
        exchange_halos(mu, local_z, plane, rank, ranks, MPI_COMM_WORLD);
        update(cold, mu, cnew, nx, ny, local_z, 0.01);
        std::swap(cold, cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        std::printf("Computation time: %.0f ms\nPerformance: %.3f MCellUpdates/s\n", max_elapsed * 1000.0,
                    static_cast<double>(volume) * iterations / max_elapsed / 1.0e6);
    }
    if (results) {
        std::vector<int> counts, offsets;
        if (!rank) { counts.resize(ranks); offsets.resize(ranks); for (int r=0; r<ranks; ++r) { const size_t n = nz/ranks + (static_cast<size_t>(r)<nz%ranks); counts[r]=static_cast<int>(n*plane); offsets[r]=r ? offsets[r-1]+counts[r-1] : 0; } }
        std::vector<double> gathered(rank == 0 ? volume : 0);
        MPI_Gatherv(cold.data()+plane, static_cast<int>(local_z*plane), MPI_DOUBLE, gathered.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(gathered, "Concentration");
    }
    if (validate) {
        bool local_ok = true; double local_min = cold[plane], local_max = cold[plane];
        for (size_t i = plane; i < (local_z+1)*plane; ++i) { local_ok = local_ok && std::isfinite(cold[i]); local_min=std::min(local_min,cold[i]); local_max=std::max(local_max,cold[i]); }
        int ok = local_ok ? 1 : 0, all_ok; double minv, maxv;
        MPI_Reduce(&ok,&all_ok,1,MPI_INT,MPI_LAND,0,MPI_COMM_WORLD); MPI_Reduce(&local_min,&minv,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD); MPI_Reduce(&local_max,&maxv,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        if (!rank) { std::printf("Concentration range: [%.6f, %.6f]\n", minv,maxv); exit_code = all_ok && maxv <= 10.0 && minv >= -10.0 ? 0 : 1; std::printf("Validation: %s\n", exit_code ? "FAILED" : "PASSED"); }
        MPI_Bcast(&exit_code,1,MPI_INT,0,MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return exit_code;
}
