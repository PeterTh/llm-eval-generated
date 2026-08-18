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

using val_t = double;

struct ElementDynamic {
    val_t current_energy = 0.0;
    val_t total_flux = 0.0;
};

struct Decomposition {
    int first_row;
    int rows;
};

static Decomposition decomposition(int n, int ranks, int rank) {
    const int base = n / ranks;
    const int extra = n % ranks;
    return {rank * base + std::min(rank, extra), base + (rank < extra ? 1 : 0)};
}

static inline val_t flux(val_t self, val_t other) {
    return (other - self) * 0.8 * 1.0 * 0.25;
}

// The original mesh is a square grid stored in row-major order.  Each rank
// owns a contiguous band and retains only two halo rows from neighboring ranks.
static void runSimulation(std::vector<ElementDynamic>& state,
                          std::vector<ElementDynamic>& next, int n,
                          const Decomposition& part, int rank, int ranks,
                          int n_iters) {
    if (part.rows == 0)
        return;

    std::vector<val_t> upper(n), lower(n), send_upper(n), send_lower(n);
    const int active_ranks = std::min(n, ranks);
    const int above = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int below = rank + 1 < active_ranks ? rank + 1 : MPI_PROC_NULL;

    for (int iter = 0; iter < n_iters; ++iter) {
        for (int y = 0; y < n; ++y) {
            send_upper[y] = state[y].current_energy;
            send_lower[y] = state[(part.rows - 1) * n + y].current_energy;
        }

        MPI_Request req[4];
        int nr = 0;
        if (above != MPI_PROC_NULL) {
            MPI_Irecv(upper.data(), n, MPI_DOUBLE, above, 1, MPI_COMM_WORLD, &req[nr++]);
            MPI_Isend(send_upper.data(), n, MPI_DOUBLE, above, 2, MPI_COMM_WORLD, &req[nr++]);
        }
        if (below != MPI_PROC_NULL) {
            MPI_Irecv(lower.data(), n, MPI_DOUBLE, below, 2, MPI_COMM_WORLD, &req[nr++]);
            MPI_Isend(send_lower.data(), n, MPI_DOUBLE, below, 1, MPI_COMM_WORLD, &req[nr++]);
        }

        // Interior rows do not depend on communication.
        for (int lx = 1; lx + 1 < part.rows; ++lx) {
            const int gx = part.first_row + lx;
            for (int y = 0; y < n; ++y) {
                const size_t i = static_cast<size_t>(lx) * n + y;
                const val_t self = state[i].current_energy;
                val_t total = 0.0;
                total += flux(self, state[i + n].current_energy); // +x
                total += flux(self, state[i - n].current_energy); // -x
                if (y + 1 < n) total += flux(self, state[i + 1].current_energy);
                if (y > 0) total += flux(self, state[i - 1].current_energy);
                next[i] = {self + total, state[i].total_flux + std::abs(total)};
            }
            (void)gx;
        }
        if (nr) MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);

        // Boundary rows (both may refer to the same row for a one-row band).
        for (int lx = 0; lx < part.rows; ++lx) {
            if (lx != 0 && lx != part.rows - 1) continue;
            const int gx = part.first_row + lx;
            for (int y = 0; y < n; ++y) {
                const size_t i = static_cast<size_t>(lx) * n + y;
                const val_t self = state[i].current_energy;
                val_t total = ((gx == 0 || gx == n - 1) && (y == 0 || y == n - 1))
                                  ? ((gx == y) ? 0.5 : -0.5) : 0.0;
                if (gx + 1 < n)
                    total += flux(self, lx + 1 < part.rows ? state[i + n].current_energy : lower[y]);
                if (gx > 0)
                    total += flux(self, lx > 0 ? state[i - n].current_energy : upper[y]);
                if (y + 1 < n) total += flux(self, state[i + 1].current_energy);
                if (y > 0) total += flux(self, state[i - 1].current_energy);
                next[i] = {self + total, state[i].total_flux + std::abs(total)};
            }
        }
        state.swap(next);
    }
}

static uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t e, f;
        std::memcpy(&e, &elements[i].current_energy, sizeof(e));
        std::memcpy(&f, &elements[i].total_flux, sizeof(f));
        hash ^= (e + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static bool validateResults(const std::vector<ElementDynamic>& state) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (const auto& e : state) {
        energy_sum += e.current_energy;
        flux_sum += e.total_flux;
        energy_max = std::max(energy_max, e.current_energy);
        energy_min = std::min(energy_min, e.current_energy);
    }
    std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n"
                "  Energy range: [%.6f, %.6f]\n", energy_sum, flux_sum,
                energy_min, energy_max);
    const bool valid = std::isfinite(energy_sum) && std::isfinite(flux_sum) &&
                       std::isfinite(energy_min) && std::isfinite(energy_max);
    if (!valid) std::printf("  ERROR: simulation produced non-finite values\n");
    if (std::abs(energy_sum) > 1e-8)
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (valid) std::printf("  Validation: PASSED\n");
    return valid;
}

static void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num>  Grid size (default: 512)\n"
                "  -i <num>  Iterations (default: 10)\n  -v  Validate\n"
                "  -r  Print external-validation results\n  -h  Help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int n = 512, iters = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else bad = true;
    }
    if (help || bad || n <= 0 || iters < 0 || static_cast<uint64_t>(n) * n > INT32_MAX / 2) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return (help && !bad) ? 0 : 1;
    }

    const Decomposition part = decomposition(n, ranks, rank);
    const size_t local_count = static_cast<size_t>(part.rows) * n;
    std::vector<ElementDynamic> state(local_count), next(local_count);
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n"
                    "============================================\n"
                    "Grid size: %d x %d = %d elements\nIterations: %d\n"
                    "MPI ranks: %d\nValidation: %s\n\nBuilding distributed mesh...\n",
                    n, n, n * n, iters, ranks, validate ? "enabled" : "disabled");
        const size_t total_mem = static_cast<size_t>(n) * n * sizeof(ElementDynamic) * 2;
        std::printf("Aggregate dynamic memory: %.2f MB\n\nRunning simulation...\n",
                    total_mem / (1024.0 * 1024.0));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(state, next, n, part, rank, ranks, iters);
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        const auto p = decomposition(n, ranks, r);
        counts[r] = 2 * p.rows * n;
        displs[r] = 2 * p.first_row * n;
    }
    std::vector<ElementDynamic> global;
    if (rank == 0) global.resize(static_cast<size_t>(n) * n);
    MPI_Gatherv(state.data(), static_cast<int>(2 * local_count), MPI_DOUBLE,
                rank == 0 ? reinterpret_cast<double*>(global.data()) : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int result = 0;
    if (rank == 0) {
        const double ms = elapsed * 1000.0;
        const int measured = std::max(iters - 1, 1);
        const double ges = elapsed > 0.0 ? measured * static_cast<double>(n) * n / elapsed / 1e9 : 0.0;
        std::printf("Computation time: %.3f ms\nPerformance:\n"
                    "  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n"
                    "  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n",
                    ms, ms / measured, ges, ges * 22.0,
                    static_cast<unsigned long long>(computeHash(global)));
        if (printResults) {
            std::vector<double> energy;
            energy.reserve(global.size());
            for (const auto& e : global) energy.push_back(e.current_energy);
            print_results(energy, "ElementEnergy");
        }
        if (validate && !validateResults(global)) result = 1;
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
