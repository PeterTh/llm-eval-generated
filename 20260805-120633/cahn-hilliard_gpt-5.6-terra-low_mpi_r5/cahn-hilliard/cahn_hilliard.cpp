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

static void printUsage(const char* prog) {
    std::printf("Usage: %s [options]\n", prog);
    std::printf("Options:\n  -x <num>     Grid X (default: 64)\n  -y <num>     Grid Y (default: X)\n"
                "  -z <num>     Grid Z (default: X)\n  -i <num>     Time steps (default: 20)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

// Exchange the first/last owned Z plane.  Ghost planes are at z=0 and z=local_z+1.
static void exchangeHalos(std::vector<double>& a, size_t plane, size_t local_z, int below, int above,
                          MPI_Comm comm) {
    MPI_Request req[4];
    MPI_Irecv(a.data(), static_cast<int>(plane), MPI_DOUBLE, below, 17, comm, &req[0]);
    MPI_Irecv(a.data() + (local_z + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, above, 19, comm, &req[1]);
    MPI_Isend(a.data() + plane, static_cast<int>(plane), MPI_DOUBLE, below, 19, comm, &req[2]);
    MPI_Isend(a.data() + local_z * plane, static_cast<int>(plane), MPI_DOUBLE, above, 17, comm, &req[3]);
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
}

static void chemicalRange(const std::vector<double>& c, std::vector<double>& mu, size_t nx, size_t ny,
                          size_t first_z, size_t last_z, double gamma) {
    const size_t plane = nx * ny;
    for (size_t z = first_z; z <= last_z; ++z) {
        const size_t base = z * plane;
        const size_t zm = base - plane, zp = base + plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = base + y * nx;
            const size_t ym = base + (y ? y - 1 : y) * nx;
            const size_t yp = base + (y + 1 < ny ? y + 1 : y) * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = row + x;
                const size_t xm = x ? x - 1 : x, xp = x + 1 < nx ? x + 1 : x;
                const double v = c[i];
                const double lap = c[row + xm] + c[row + xp] + c[ym + x] + c[yp + x] + c[zm + y * nx + x]
                                 + c[zp + y * nx + x] - 6.0 * v;
                mu[i] = 4.5 * ((v + 1.0) * (-2.0 / 9.0) + (v - 1.0) * (-2.0 / 9.0)
                              - 2.0 * v * (2.0 / 9.0)) + 3.0 * v + v * v * v - gamma * lap;
            }
        }
    }
}

static void updateRange(const std::vector<double>& cold, const std::vector<double>& mu, std::vector<double>& out,
                        size_t nx, size_t ny, size_t first_z, size_t last_z, double dt) {
    const size_t plane = nx * ny;
    for (size_t z = first_z; z <= last_z; ++z) {
        const size_t base = z * plane;
        const size_t zm = base - plane, zp = base + plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = base + y * nx;
            const size_t ym = base + (y ? y - 1 : y) * nx;
            const size_t yp = base + (y + 1 < ny ? y + 1 : y) * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = row + x;
                const size_t xm = x ? x - 1 : x, xp = x + 1 < nx ? x + 1 : x;
                const double lap = mu[row + xm] + mu[row + xp] + mu[ym + x] + mu[yp + x] + mu[zm + y * nx + x]
                                 + mu[zp + y * nx + x] - 6.0 * mu[i];
                out[i] = cold[i] + dt * lap;
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    size_t nx = 64, ny = 0, nz = 0; int iterations = 20; bool validate = false, printResults = false;
    int parse_ok = 1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!world_rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!world_rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } parse_ok = 0; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || iterations < 0 || nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) parse_ok = 0;
    int all_ok; MPI_Allreduce(&parse_ok, &all_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!all_ok) { if (!world_rank) std::printf("Invalid grid dimensions or iteration count\n"); MPI_Finalize(); return 1; }

    const int active_size = std::min<size_t>(nz, world_size);
    MPI_Comm comm; MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_size ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active_size) { MPI_Finalize(); return 0; }
    int rank, size; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
    const size_t base_z = nz / size, remainder = nz % size;
    const size_t local_z = base_z + (static_cast<size_t>(rank) < remainder);
    const size_t global_z = static_cast<size_t>(rank) * base_z + std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny, local_count = local_z * plane;
    const int below = rank ? rank - 1 : MPI_PROC_NULL, above = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;
    std::vector<double> cold((local_z + 2) * plane), cnew((local_z + 2) * plane), mu((local_z + 2) * plane);
    for (size_t z = 0; z < local_z; ++z) for (size_t i = 0; i < plane; ++i) {
        const size_t id = (global_z + z) * plane + i;
        cold[(z + 1) * plane + i] = -1.0 + 2.0 * (((id + 1) * 1299709 % (nx * ny * nz)) / static_cast<double>(nx * ny * nz));
    }
    if (!world_rank) { std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\nInitializing concentration field...\nRunning Cahn-Hilliard simulation...\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled"); }
    MPI_Barrier(comm); const auto start = std::chrono::steady_clock::now();
    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, plane, local_z, below, above, comm);
        if (rank == 0) std::copy_n(cold.data() + plane, plane, cold.data());
        if (rank == size - 1) std::copy_n(cold.data() + local_z * plane, plane, cold.data() + (local_z + 1) * plane);
        chemicalRange(cold, mu, nx, ny, 1, local_z, 0.5);
        exchangeHalos(mu, plane, local_z, below, above, comm);
        if (rank == 0) std::copy_n(mu.data() + plane, plane, mu.data());
        if (rank == size - 1) std::copy_n(mu.data() + local_z * plane, plane, mu.data() + (local_z + 1) * plane);
        updateRange(cold, mu, cnew, nx, ny, 1, local_z, 0.01);
        std::swap(cold, cnew);
    }
    MPI_Barrier(comm); const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double max_seconds; MPI_Reduce(&seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!rank) { std::printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n", static_cast<long>(max_seconds * 1000), (nx * ny * nz * static_cast<double>(iterations)) / max_seconds / 1e6); }
    std::vector<double> full;
    if (printResults || validate) {
        std::vector<int> counts, offsets;
        if (!rank) { counts.resize(size); offsets.resize(size); for (int r = 0; r < size; ++r) { const size_t lz = base_z + (static_cast<size_t>(r) < remainder); counts[r] = static_cast<int>(lz * plane); offsets[r] = static_cast<int>((static_cast<size_t>(r) * base_z + std::min(static_cast<size_t>(r), remainder)) * plane); } full.resize(nx * ny * nz); }
        MPI_Gatherv(cold.data() + plane, static_cast<int>(local_count), MPI_DOUBLE, rank ? nullptr : full.data(), rank ? nullptr : counts.data(), rank ? nullptr : offsets.data(), MPI_DOUBLE, 0, comm);
    }
    int result = 0;
    if (!rank && printResults) print_results(full, "Concentration");
    if (!rank && validate) { double lo = *std::min_element(full.begin(), full.end()), hi = *std::max_element(full.begin(), full.end()); const bool ok = std::isfinite(lo) && std::isfinite(hi) && lo >= -10.0 && hi <= 10.0; std::printf("Validating result...\nConcentration range: [%.6f, %.6f]\nValidation: %s\n", lo, hi, ok ? "PASSED" : "FAILED"); result = ok ? 0 : 1; }
    MPI_Bcast(&result, 1, MPI_INT, 0, comm); MPI_Comm_free(&comm); MPI_Finalize(); return result;
}
