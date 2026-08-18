#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;
static MPI_Comm simulation_comm = MPI_COMM_WORLD;

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// A block of complete grid rows.  Rows 0 and local_rows+1 are ghost rows.
struct Domain {
    int n = 0;
    int first_row = 0;
    int local_rows = 0;
    int rank = 0;
    int size = 1;
    std::vector<ElementDynamic> state;
    std::vector<ElementDynamic> next;
};

static Domain buildDomain(int n, int rank, int size) {
    Domain d;
    d.n = n; d.rank = rank; d.size = size;
    const int base = n / size, extra = n % size;
    d.local_rows = base + (rank < extra);
    d.first_row = rank * base + std::min(rank, extra);
    d.state.resize(static_cast<size_t>(d.local_rows + 2) * n, {0.0, 0.0});
    d.next.resize(d.state.size(), {0.0, 0.0});
    return d;
}

static void exchangeHalos(Domain& d, MPI_Datatype energy_row) {
    const int n = d.n;
    MPI_Request req[4];
    int nr = 0;
    if (d.rank > 0) {
        MPI_Irecv(&d.state[0].current_energy, 1, energy_row, d.rank - 1, 1,
                  simulation_comm, &req[nr++]);
        MPI_Isend(&d.state[n].current_energy, 1, energy_row, d.rank - 1, 0,
                  simulation_comm, &req[nr++]);
    }
    if (d.rank + 1 < d.size) {
        MPI_Irecv(&d.state[static_cast<size_t>(d.local_rows + 1) * n].current_energy,
                  1, energy_row, d.rank + 1, 0, simulation_comm, &req[nr++]);
        MPI_Isend(&d.state[static_cast<size_t>(d.local_rows) * n].current_energy,
                  1, energy_row, d.rank + 1, 1, simulation_comm, &req[nr++]);
    }
    if (nr) MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
}

static void runSimulation(Domain& d, int iterations) {
    const int n = d.n;
    // The original coefficients reduce to 0.8 * 1.0 * 0.25.
    constexpr val_t coefficient = 0.2;
    // Only energy is communicated. Reuse the datatype across all iterations.
    MPI_Datatype energy_row;
    MPI_Type_vector(n, 1, 2, MPI_DOUBLE, &energy_row);
    MPI_Type_commit(&energy_row);
    for (int iter = 0; iter < iterations; ++iter) {
        exchangeHalos(d, energy_row);
        for (int lx = 1; lx <= d.local_rows; ++lx) {
            const int gx = d.first_row + lx - 1;
            for (int y = 0; y < n; ++y) {
                const size_t p = static_cast<size_t>(lx) * n + y;
                const val_t e = d.state[p].current_energy;
                val_t flux = ((gx == 0 || gx == n - 1) && (y == 0 || y == n - 1))
                                 ? ((gx == y) ? 0.5 : -0.5) : 0.0;
                // Preserve the reference order: +x, -x, +y, -y.
                if (gx + 1 < n) flux += (d.state[p + n].current_energy - e) * coefficient;
                if (gx > 0)     flux += (d.state[p - n].current_energy - e) * coefficient;
                if (y + 1 < n)  flux += (d.state[p + 1].current_energy - e) * coefficient;
                if (y > 0)      flux += (d.state[p - 1].current_energy - e) * coefficient;
                d.next[p] = {e + flux, d.state[p].total_flux + std::abs(flux)};
            }
        }
        d.state.swap(d.next);
    }
    MPI_Type_free(&energy_row);
}

static uint64_t localHash(const Domain& d) {
    uint64_t hash = 0;
    for (int lx = 1; lx <= d.local_rows; ++lx) {
        for (int y = 0; y < d.n; ++y) {
            const size_t p = static_cast<size_t>(lx) * d.n + y;
            const uint64_t i = static_cast<uint64_t>(d.first_row + lx - 1) * d.n + y;
            uint64_t e, f;
            std::memcpy(&e, &d.state[p].current_energy, sizeof(e));
            std::memcpy(&f, &d.state[p].total_flux, sizeof(f));
            hash ^= (e + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

static bool validateResults(const Domain& d) {
    val_t local[4] = {0.0, 0.0, std::numeric_limits<val_t>::lowest(),
                     std::numeric_limits<val_t>::max()};
    for (int lx = 1; lx <= d.local_rows; ++lx)
        for (int y = 0; y < d.n; ++y) {
            const auto& e = d.state[static_cast<size_t>(lx) * d.n + y];
            local[0] += e.current_energy; local[1] += e.total_flux;
            local[2] = std::max(local[2], e.current_energy);
            local[3] = std::min(local[3], e.current_energy);
        }
    val_t sums[2], maximum, minimum;
    MPI_Reduce(local, sums, 2, MPI_DOUBLE, MPI_SUM, 0, simulation_comm);
    MPI_Reduce(local + 2, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, simulation_comm);
    MPI_Reduce(local + 3, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, simulation_comm);
    int valid = 1;
    if (d.rank == 0) {
        std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n"
                    "  Energy range: [%.6f, %.6f]\n", sums[0], sums[1], minimum, maximum);
        if (!std::isfinite(sums[0]) || !std::isfinite(sums[1]) ||
            !std::isfinite(maximum) || !std::isfinite(minimum)) valid = 0;
        if (std::abs(sums[0]) > 1e-8)
            std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        std::printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, simulation_comm);
    return valid != 0;
}

static void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num> Grid size (default: 512)\n"
                "  -i <num> Iterations (default: 10)\n  -v Validate\n"
                "  -r Print results\n  -h Show help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int n = 512, iterations = 10; bool validate = false, printResults = false;
    int parse_ok = 1, help = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = 1;
        else parse_ok = 0;
    }
    if (help || !parse_ok || n <= 0 || iterations < 0) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize(); return parse_ok && n > 0 && iterations >= 0 ? 0 : 1;
    }
    // More ranks than rows cannot contribute; exclude them cleanly.
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < n ? 0 : MPI_UNDEFINED, rank, &active);
    if (rank >= n) { MPI_Finalize(); return 0; }
    simulation_comm = active;
    MPI_Comm_set_errhandler(simulation_comm, MPI_ERRORS_ARE_FATAL);
    const int size = std::min(world_size, n);
    Domain d = buildDomain(n, rank, size);
    const uint64_t elems = static_cast<uint64_t>(n) * n;
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n"
                    "Grid size: %d x %d = %llu elements\nIterations: %d\nValidation: %s\n"
                    "MPI ranks: %d\n\nBuilding unstructured mesh...\n", n, n,
                    static_cast<unsigned long long>(elems), iterations,
                    validate ? "enabled" : "disabled", size);
        const double dynamic_mb = elems * sizeof(ElementDynamic) * 2 / 1048576.0;
        std::printf("Memory usage: %.2f MB (distributed dynamic state)\n\nRunning simulation...\n", dynamic_mb);
    }
    MPI_Barrier(simulation_comm);
    const double start = MPI_Wtime();
    runSimulation(d, iterations);
    const double local_time = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, simulation_comm);
    uint64_t lh = localHash(d), hash = 0;
    MPI_Reduce(&lh, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, simulation_comm);
    if (rank == 0) {
        const double ms = elapsed * 1000.0;
        const int measured = std::max(iterations - 1, 1);
        const double ges = elapsed > 0 ? measured * static_cast<double>(elems) / elapsed / 1e9 : 0;
        std::printf("Computation time: %.0f ms\nPerformance:\n  Time per iteration: %.4f ms\n"
                    "  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n"
                    "  Result hash: %016llX\n\n", ms, ms / measured, ges, ges * 22.0,
                    static_cast<unsigned long long>(hash));
    }
    if (printResults) {
        std::vector<double> local_energy(static_cast<size_t>(d.local_rows) * n);
        for (int x = 0; x < d.local_rows; ++x)
            for (int y = 0; y < n; ++y)
                local_energy[static_cast<size_t>(x) * n + y] = d.state[static_cast<size_t>(x + 1) * n + y].current_energy;
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            const int rows = n / size + (r < n % size);
            counts[r] = rows * n; displs[r] = (r * (n / size) + std::min(r, n % size)) * n;
        }
        std::vector<double> all(rank == 0 ? static_cast<size_t>(elems) : 0);
        MPI_Gatherv(local_energy.data(), counts[rank], MPI_DOUBLE, all.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, simulation_comm);
        if (rank == 0) print_results(all, "ElementEnergy");
    }
    const bool valid = !validate || validateResults(d);
    MPI_Comm_free(&active);
    MPI_Finalize();
    return valid ? 0 : 1;
}
