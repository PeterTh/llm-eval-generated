#include <mpi.h>

#include <algorithm>
#include <chrono>
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

struct Partition {
    int n = 0;
    int first_row = 0;
    int rows = 0;
    int rank = 0;
    int ranks = 1;
    std::vector<ElementDynamic> state;
    std::vector<ElementDynamic> next;
    std::vector<val_t> upper_halo;
    std::vector<val_t> lower_halo;
    std::vector<val_t> upper_send;
    std::vector<val_t> lower_send;
};

static Partition buildPartition(int n, int rank, int ranks) {
    Partition p;
    p.n = n;
    p.rank = rank;
    p.ranks = ranks;
    const int base = n / ranks;
    const int extra = n % ranks;
    p.rows = base + (rank < extra ? 1 : 0);
    p.first_row = rank * base + std::min(rank, extra);
    const size_t count = static_cast<size_t>(p.rows) * static_cast<size_t>(n);
    p.state.resize(count);
    p.next.resize(count);
    p.upper_halo.resize(static_cast<size_t>(n));
    p.lower_halo.resize(static_cast<size_t>(n));
    p.upper_send.resize(static_cast<size_t>(n));
    p.lower_send.resize(static_cast<size_t>(n));
    return p;
}

// The arithmetic and neighbor order deliberately match the original connectivity
// order: x+1, x-1, y+1, y-1.
static inline void updateRow(Partition& p, int local_row) {
    const int n = p.n;
    const int global_row = p.first_row + local_row;
    const size_t row_offset = static_cast<size_t>(local_row) * n;

    for (int y = 0; y < n; ++y) {
        const size_t i = row_offset + static_cast<size_t>(y);
        const val_t energy = p.state[i].current_energy;
        const bool inflow = (global_row == 0 || global_row == n - 1) &&
                            (y == 0 || y == n - 1) && (global_row == y);
        const bool outflow = (global_row == 0 || global_row == n - 1) &&
                             (y == 0 || y == n - 1) && (global_row != y);
        val_t flux = inflow ? 0.5 : (outflow ? -0.5 : 0.0);

        if (global_row + 1 < n) {
            const val_t neighbor = local_row + 1 < p.rows
                ? p.state[i + static_cast<size_t>(n)].current_energy
                : p.lower_halo[static_cast<size_t>(y)];
            flux += (neighbor - energy) * 0.8 * 1.0 * 0.25;
        }
        if (global_row > 0) {
            const val_t neighbor = local_row > 0
                ? p.state[i - static_cast<size_t>(n)].current_energy
                : p.upper_halo[static_cast<size_t>(y)];
            flux += (neighbor - energy) * 0.8 * 1.0 * 0.25;
        }
        if (y + 1 < n)
            flux += (p.state[i + 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
        if (y > 0)
            flux += (p.state[i - 1].current_energy - energy) * 0.8 * 1.0 * 0.25;

        p.next[i].current_energy = energy + flux;
        p.next[i].total_flux = p.state[i].total_flux + std::abs(flux);
    }
}

static void runSimulation(Partition& p, int iterations, MPI_Comm comm) {
    constexpr int TO_UP = 100;
    constexpr int TO_DOWN = 101;
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[4];
        int request_count = 0;
        const int up = p.rank > 0 ? p.rank - 1 : MPI_PROC_NULL;
        const int down = p.rank + 1 < p.ranks ? p.rank + 1 : MPI_PROC_NULL;

        if (up != MPI_PROC_NULL) {
            for (int y = 0; y < p.n; ++y)
                p.upper_send[static_cast<size_t>(y)] = p.state[static_cast<size_t>(y)].current_energy;
            MPI_Irecv(p.upper_halo.data(), p.n, MPI_DOUBLE, up, TO_DOWN, comm,
                      &requests[request_count++]);
            MPI_Isend(p.upper_send.data(), p.n, MPI_DOUBLE, up, TO_UP,
                      comm, &requests[request_count++]);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Irecv(p.lower_halo.data(), p.n, MPI_DOUBLE, down, TO_UP, comm,
                      &requests[request_count++]);
            // ElementDynamic is not contiguous in energy, so pack the boundary.
            for (int y = 0; y < p.n; ++y)
                p.lower_send[static_cast<size_t>(y)] =
                    p.state[(static_cast<size_t>(p.rows - 1) * p.n) + y].current_energy;
            MPI_Isend(p.lower_send.data(), p.n, MPI_DOUBLE, down, TO_DOWN, comm,
                      &requests[request_count++]);
        }

        for (int row = 1; row + 1 < p.rows; ++row) updateRow(p, row);
        if (request_count) MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
        updateRow(p, 0);
        if (p.rows > 1) updateRow(p, p.rows - 1);
        p.state.swap(p.next);
    }
}

static uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t result = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        result ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        result ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return result;
}

static bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(energy_max, elem.current_energy);
        energy_min = std::min(energy_min, elem.current_energy);
    }
    std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n"
                "  Energy range: [%.6f, %.6f]\n", energy_sum, flux_sum,
                energy_min, energy_max);
    const bool valid = std::isfinite(energy_sum) && std::isfinite(flux_sum) &&
                       std::isfinite(energy_max) && std::isfinite(energy_min);
    if (!valid) std::printf("  ERROR: Simulation produced non-finite values\n");
    if (valid && std::abs(energy_sum) > 1e-8)
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (valid) std::printf("  Validation: PASSED\n");
    return valid;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>  Grid size (default: 512)\n"
                "  -i <num>  Iterations (default: 10)\n  -v        Validate\n"
                "  -r        Print external-validation results\n  -h        Help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = 512, iterations = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else bad = true;
    }
    if (help || bad || n <= 0 || iterations < 0 ||
        static_cast<uint64_t>(n) * n > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        if (world_rank == 0) {
            if (!help) std::fprintf(stderr, "Invalid command line or problem size\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return help ? 0 : 1;
    }

    // At most one rank per mesh row; surplus ranks remain in MPI_COMM_WORLD but
    // do not allocate mesh data or participate in the active communicator.
    const int active_ranks = std::min(world_size, n);
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 0 : MPI_UNDEFINED,
                   world_rank, &active_comm);
    if (world_rank >= active_ranks) {
        MPI_Finalize();
        return 0;
    }

    Partition part = buildPartition(n, world_rank, active_ranks);
    const int total_elements = n * n;
    if (world_rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n"
                    "============================================\n"
                    "Grid size: %d x %d = %d elements\nIterations: %d\n"
                    "MPI ranks: %d\nValidation: %s\n\nBuilding unstructured mesh...\n",
                    n, n, total_elements, iterations, active_ranks,
                    validate ? "enabled" : "disabled");
        const size_t original_static = static_cast<size_t>(total_elements) * (2 * sizeof(uint64_t) + 8 * (sizeof(uint64_t) + sizeof(double)));
        const size_t dynamic = static_cast<size_t>(total_elements) * sizeof(ElementDynamic) * 2;
        std::printf("Global memory footprint: %.2f MB; rank 0 dynamic partition: %.2f MB\n\nRunning simulation...\n",
                    (original_static + dynamic) / (1024.0 * 1024.0),
                    part.state.size() * sizeof(ElementDynamic) * 2 / (1024.0 * 1024.0));
    }

    MPI_Barrier(active_comm);
    const double start = MPI_Wtime();
    runSimulation(part, iterations, active_comm);
    const double local_time = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm);

    std::vector<int> counts, displacements;
    std::vector<double> local_energy(part.state.size()), local_flux(part.state.size());
    for (size_t i = 0; i < part.state.size(); ++i) {
        local_energy[i] = part.state[i].current_energy;
        local_flux[i] = part.state[i].total_flux;
    }
    std::vector<double> all_energy, all_flux;
    if (world_rank == 0) {
        counts.resize(active_ranks); displacements.resize(active_ranks);
        for (int r = 0; r < active_ranks; ++r) {
            const int rows = n / active_ranks + (r < n % active_ranks ? 1 : 0);
            counts[r] = rows * n;
            displacements[r] = r * (n / active_ranks) * n + std::min(r, n % active_ranks) * n;
        }
        all_energy.resize(total_elements); all_flux.resize(total_elements);
    }
    MPI_Gatherv(local_energy.data(), static_cast<int>(local_energy.size()), MPI_DOUBLE,
                all_energy.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, active_comm);
    MPI_Gatherv(local_flux.data(), static_cast<int>(local_flux.size()), MPI_DOUBLE,
                all_flux.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, active_comm);

    int status = 0;
    if (world_rank == 0) {
        std::vector<ElementDynamic> global(static_cast<size_t>(total_elements));
        for (int i = 0; i < total_elements; ++i) global[i] = {all_energy[i], all_flux[i]};
        const double measured_iters = std::max(iterations - 1, 1);
        const double ms = elapsed * 1000.0;
        const double geps = elapsed > 0.0 ? measured_iters * total_elements / elapsed / 1e9 : 0.0;
        std::printf("Computation time: %.3f ms\nPerformance:\n"
                    "  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n"
                    "  Performance: %.4f GFLOPS\n  Result hash: %016lX\n\n",
                    ms, ms / measured_iters, geps, geps * 22.0,
                    static_cast<unsigned long>(computeHash(global)));
        if (printResults) print_results(all_energy, "ElementEnergy");
        if (validate && !validateResults(global)) status = 1;
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, active_comm);
    MPI_Comm_free(&active_comm);
    MPI_Finalize();
    return status;
}
