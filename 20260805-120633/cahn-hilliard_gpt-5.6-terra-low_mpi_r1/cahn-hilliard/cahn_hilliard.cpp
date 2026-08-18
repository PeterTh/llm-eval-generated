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

static inline double laplacian(const double* a, size_t x, size_t y, size_t z,
                               size_t nx, size_t ny) noexcept {
    const size_t xp = x + (x + 1 < nx);
    const size_t xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t yn = y - (y > 0);
    // z=0 and z=localNz+1 are halo planes.  At physical boundaries the
    // exchange routine fills them by copying the boundary plane, giving the
    // same clamped stencil as the serial implementation.
    const size_t zp = z + 1;
    const size_t zn = z - 1;
    const size_t i = idx3(x, y, z, nx, ny);
    return a[idx3(xp, y, z, nx, ny)] + a[idx3(xn, y, z, nx, ny)] - 2.0 * a[i]
         + a[idx3(x, yp, z, nx, ny)] + a[idx3(x, yn, z, nx, ny)] - 2.0 * a[i]
         + a[idx3(x, y, zp, nx, ny)] + a[idx3(x, y, zn, nx, ny)] - 2.0 * a[i];
}

// Exchange z faces. Physical boundaries use clamped values by copying the owned face.
static void exchangeHalos(std::vector<double>& a, size_t nx, size_t ny, size_t localNz,
                          int rank, int activeRanks, MPI_Comm comm) {
    const int face = static_cast<int>(nx * ny);
    if (rank == 0)
        std::memcpy(a.data(), a.data() + face, static_cast<size_t>(face) * sizeof(double));
    if (rank == activeRanks - 1)
        std::memcpy(a.data() + static_cast<size_t>(localNz + 1) * face,
                    a.data() + static_cast<size_t>(localNz) * face,
                    static_cast<size_t>(face) * sizeof(double));

    MPI_Request requests[4];
    int count = 0;
    if (rank > 0) {
        MPI_Irecv(a.data(), face, MPI_DOUBLE, rank - 1, 17, comm, &requests[count++]);
        MPI_Isend(a.data() + face, face, MPI_DOUBLE, rank - 1, 18, comm, &requests[count++]);
    }
    if (rank + 1 < activeRanks) {
        MPI_Irecv(a.data() + static_cast<size_t>(localNz + 1) * face, face, MPI_DOUBLE,
                  rank + 1, 18, comm, &requests[count++]);
        MPI_Isend(a.data() + static_cast<size_t>(localNz) * face, face, MPI_DOUBLE,
                  rank + 1, 17, comm, &requests[count++]);
    }
    if (count) MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
}

static void chemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              size_t nx, size_t ny, size_t localNz) {
    constexpr double gamma = 0.5;
    for (size_t z = 1; z <= localNz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                const double v = c[i];
                // e_AA=e_BB=-2/9 and e_AB=2/9, algebraically simplified.
                mu[i] = 4.5 * (-8.0 / 9.0 * v) + 3.0 * v + v * v * v
                      - gamma * laplacian(c.data(), x, y, z, nx, ny);
            }
}

static void update(std::vector<double>& out, const std::vector<double>& in, const std::vector<double>& mu,
                   size_t nx, size_t ny, size_t localNz) {
    constexpr double dtD = 0.01;
    for (size_t z = 1; z <= localNz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                out[i] = in[i] + dtD * laplacian(mu.data(), x, y, z, nx, ny);
            }
}

static void printUsage(const char* prog) {
    std::printf("Usage: %s [options]\n  -x <num>  Grid X (default 64)\n  -y <num>  Grid Y\n"
                "  -z <num>  Grid Z\n  -i <num>  Time steps (default 20)\n  -v        Validate\n"
                "  -r        Print results\n  -h        Help\n", prog);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    int badArgs = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else badArgs = 1;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || iterations < 0 || nx > static_cast<size_t>(std::numeric_limits<int>::max() / ny)) badArgs = 1;
    int anyBad = 0;
    MPI_Allreduce(&badArgs, &anyBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anyBad) { if (!rank) { std::printf("Invalid option or grid dimensions\n"); printUsage(argv[0]); } MPI_Finalize(); return 1; }

    const int activeRanks = std::min<int>(worldSize, static_cast<int>(nz));
    const bool active = rank < activeRanks;
    const size_t base = nz / activeRanks, extra = nz % activeRanks;
    const size_t localNz = active ? base + (static_cast<size_t>(rank) < extra) : 0;
    const size_t zStart = active ? static_cast<size_t>(rank) * base + std::min<size_t>(rank, extra) : 0;
    const size_t plane = nx * ny;
    std::vector<double> cold(active ? (localNz + 2) * plane : 0), cnew(cold.size()), mu(cold.size());
    if (active) {
        const size_t volume = nx * ny * nz;
        for (size_t z = 1; z <= localNz; ++z)
            for (size_t y = 0; y < ny; ++y)
                for (size_t x = 0; x < nx; ++x) {
                    const size_t global = (zStart + z - 1) * plane + y * nx + x;
                    cold[idx3(x, y, z, nx, ny)] = -1.0 + 2.0 * (((global + 1) * 1299709 % volume) / static_cast<double>(volume));
                }
    }
    if (!rank) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d (%d active)\nInitializing concentration field...\nRunning Cahn-Hilliard simulation...\n", worldSize, activeRanks);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (active) for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, nx, ny, localNz, rank, activeRanks, MPI_COMM_WORLD);
        chemicalPotential(cold, mu, nx, ny, localNz);
        exchangeHalos(mu, nx, ny, localNz, rank, activeRanks, MPI_COMM_WORLD);
        update(cnew, cold, mu, nx, ny, localNz);
        cold.swap(cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts, displs;
    std::vector<double> global;
    const int sendCount = active ? static_cast<int>(localNz * plane) : 0;
    if (!rank) { counts.resize(worldSize); displs.resize(worldSize); }
    MPI_Gather(&sendCount, 1, MPI_INT, rank ? nullptr : counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!rank) {
        int displacement = 0;
        for (int r = 0; r < worldSize; ++r) { displs[r] = displacement; displacement += counts[r]; }
        global.resize(nx * ny * nz);
        std::printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", maxElapsed * 1000.0,
                    (static_cast<double>(nx) * ny * nz * iterations) / maxElapsed / 1e6);
    }
    MPI_Gatherv(active ? cold.data() + plane : nullptr, sendCount, MPI_DOUBLE,
                rank ? nullptr : global.data(), rank ? nullptr : counts.data(), rank ? nullptr : displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int status = 0;
    if (!rank && printResults) print_results(global, "Concentration");
    if (!rank && validate) {
        bool valid = true;
        double lo = global.empty() ? 0.0 : global[0], hi = lo;
        for (double v : global) { valid = valid && std::isfinite(v); lo = std::min(lo, v); hi = std::max(hi, v); }
        std::printf("Concentration range: [%.6f, %.6f]\nValidation: %s\n", lo, hi,
                    valid && hi <= 10.0 && lo >= -10.0 ? "PASSED" : "FAILED");
        status = valid && hi <= 10.0 && lo >= -10.0 ? 0 : 1;
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
