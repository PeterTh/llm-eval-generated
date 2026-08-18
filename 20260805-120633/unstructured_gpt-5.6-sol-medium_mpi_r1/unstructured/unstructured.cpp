#include <mpi.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using idx_t = std::uint64_t;
using val_t = double;

// A row-block decomposition gives every rank at most two neighboring ranks.
// Only current energy crosses a partition boundary; accumulated flux is local.
struct World {
    int width = 0;
    int first_row = 0;
    int local_rows = 0;
    val_t transfer_coeff = 0.8;
    int previous_rank = MPI_PROC_NULL;
    int next_rank = MPI_PROC_NULL;
    std::vector<val_t> energy;
    std::vector<val_t> energy_swap;
    std::vector<val_t> accumulated_flux;

    std::size_t owned_offset() const {
        return local_rows == 0 ? 0 : static_cast<std::size_t>(width);
    }
    std::size_t owned_elements() const {
        return static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(width);
    }
};

static World buildSquare2D(int n, int rank, int ranks) {
    World world;
    world.width = n;

    const int base = n / ranks;
    const int remainder = n % ranks;
    world.local_rows = base + (rank < remainder ? 1 : 0);
    world.first_row = rank * base + std::min(rank, remainder);

    // Nonempty row blocks are assigned to a contiguous prefix of ranks.
    if (world.local_rows != 0) {
        world.previous_rank = world.first_row == 0 ? MPI_PROC_NULL : rank - 1;
        world.next_rank = world.first_row + world.local_rows == n ? MPI_PROC_NULL : rank + 1;
    }

    const std::size_t halo_width = static_cast<std::size_t>(n);
    const std::size_t allocation = world.local_rows == 0
        ? 0
        : world.owned_elements() + 2 * halo_width;
    world.energy.assign(allocation, 0.0);
    world.energy_swap.assign(allocation, 0.0);
    world.accumulated_flux.assign(world.owned_elements(), 0.0);
    return world;
}

inline void updateRow(World& world, int local_x) {
    const int n = world.width;
    const int global_x = world.first_row + local_x;
    const std::size_t row = world.owned_offset() +
                            static_cast<std::size_t>(local_x) * static_cast<std::size_t>(n);
    const std::size_t flux_row = static_cast<std::size_t>(local_x) * static_cast<std::size_t>(n);

    for (int y = 0; y < n; ++y) {
        const std::size_t i = row + static_cast<std::size_t>(y);
        const val_t current = world.energy[i];
        const bool corner = (global_x == 0 || global_x == n - 1) && (y == 0 || y == n - 1);
        // The external source is the initial accumulator in the serial code.
        val_t flux = corner ? ((global_x == y) ? 0.5 : -0.5) : 0.0;

        // Match the original connection order: +x, -x, +y, -y.
        if (global_x + 1 < n)
            flux += (world.energy[i + n] - current) * world.transfer_coeff * 1.0 * 0.25;
        if (global_x > 0)
            flux += (world.energy[i - n] - current) * world.transfer_coeff * 1.0 * 0.25;
        if (y + 1 < n)
            flux += (world.energy[i + 1] - current) * world.transfer_coeff * 1.0 * 0.25;
        if (y > 0)
            flux += (world.energy[i - 1] - current) * world.transfer_coeff * 1.0 * 0.25;

        world.energy_swap[i] = current + flux;
        world.accumulated_flux[flux_row + static_cast<std::size_t>(y)] += std::abs(flux);
    }
}

static void runSimulation(World& world, int n_iters, MPI_Comm comm) {
    if (world.local_rows == 0) return;

    const int n = world.width;
    const std::size_t first = world.owned_offset();
    const std::size_t last = first +
        static_cast<std::size_t>(world.local_rows - 1) * static_cast<std::size_t>(n);
    const std::size_t upper_ghost = first + world.owned_elements();

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        int request_count = 0;

        if (world.previous_rank != MPI_PROC_NULL) {
            MPI_Irecv(world.energy.data(), n, MPI_DOUBLE, world.previous_rank, 1,
                      comm, &requests[request_count++]);
            MPI_Isend(world.energy.data() + first, n, MPI_DOUBLE, world.previous_rank, 0,
                      comm, &requests[request_count++]);
        }
        if (world.next_rank != MPI_PROC_NULL) {
            MPI_Irecv(world.energy.data() + upper_ghost, n, MPI_DOUBLE, world.next_rank, 0,
                      comm, &requests[request_count++]);
            MPI_Isend(world.energy.data() + last, n, MPI_DOUBLE, world.next_rank, 1,
                      comm, &requests[request_count++]);
        }

        // These rows have no remote dependencies, so compute them in flight.
        for (int x = 1; x + 1 < world.local_rows; ++x) updateRow(world, x);

        if (request_count != 0) MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

        updateRow(world, 0);
        if (world.local_rows > 1) updateRow(world, world.local_rows - 1);
        std::swap(world.energy, world.energy_swap);
    }
}

static std::uint64_t computeHash(const World& world, MPI_Comm comm) {
    std::uint64_t local_hash = 0;
    const std::size_t offset = world.owned_offset();
    const idx_t global_begin = static_cast<idx_t>(world.first_row) *
                               static_cast<idx_t>(world.width);
    for (std::size_t i = 0; i < world.owned_elements(); ++i) {
        const idx_t global_i = global_begin + static_cast<idx_t>(i);
        const std::uint64_t energy_bits = std::bit_cast<std::uint64_t>(world.energy[offset + i]);
        const std::uint64_t flux_bits = std::bit_cast<std::uint64_t>(world.accumulated_flux[i]);
        local_hash ^= (energy_bits + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (flux_bits + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    std::uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    return global_hash;
}

static bool validateResults(const World& world, int rank, MPI_Comm comm) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_max = std::numeric_limits<val_t>::lowest();
    val_t local_min = std::numeric_limits<val_t>::max();
    const std::size_t offset = world.owned_offset();
    for (std::size_t i = 0; i < world.owned_elements(); ++i) {
        local_energy_sum += world.energy[offset + i];
        local_flux_sum += world.accumulated_flux[i];
        local_max = std::max(local_max, world.energy[offset + i]);
        local_min = std::min(local_min, world.energy[offset + i]);
    }

    val_t energy_sum = 0.0, flux_sum = 0.0, energy_max = 0.0, energy_min = 0.0;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);

    int valid = 1;
    if (rank == 0) {
        std::printf("Validation results:\n");
        std::printf("  Energy sum: %.12f\n", energy_sum);
        std::printf("  Flux sum: %.2f\n", flux_sum);
        std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
        if (!std::isfinite(energy_sum)) {
            std::printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        } else if (std::abs(energy_sum) > 1e-8) {
            std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(flux_sum)) {
            std::printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            std::printf("  ERROR: Energy extrema are not finite\n");
            valid = 0;
        }
        if (valid) std::printf("  Validation: PASSED\n");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    return valid != 0;
}

static void printDistributedResults(const World& world, int rank, int ranks, MPI_Comm comm) {
    const int local_count = static_cast<int>(world.owned_elements());
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<val_t> all_energy;
    if (rank == 0) {
        counts.resize(ranks);
        displacements.resize(ranks);
    }
    MPI_Gather(&local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr,
               1, MPI_INT, 0, comm);
    if (rank == 0) {
        int displacement = 0;
        for (int r = 0; r < ranks; ++r) {
            displacements[r] = displacement;
            displacement += counts[r];
        }
        all_energy.resize(static_cast<std::size_t>(displacement));
    }
    const val_t* send_buffer = local_count == 0
        ? nullptr
        : world.energy.data() + world.owned_offset();
    MPI_Gatherv(send_buffer, local_count, MPI_DOUBLE,
                rank == 0 ? all_energy.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, comm);
    if (rank == 0) print_results(all_energy, "ElementEnergy");
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);

    int n = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_results_requested = false;
    int parse_status = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parse_status = 1;
            break;
        }
    }
    if (n <= 0 || n_iters < 0) {
        if (rank == 0) std::fprintf(stderr, "Grid size must be positive and iterations nonnegative.\n");
        parse_status = 1;
    }
    int any_parse_error = 0;
    MPI_Allreduce(&parse_status, &any_parse_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_parse_error) {
        MPI_Finalize();
        return 1;
    }

    const idx_t global_elements = static_cast<idx_t>(n) * static_cast<idx_t>(n);
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %llu elements\n", n, n,
                    static_cast<unsigned long long>(global_elements));
        std::printf("Iterations: %d\n", n_iters);
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        std::printf("Building distributed unstructured mesh...\n");
    }

    World world = buildSquare2D(n, rank, ranks);
    const unsigned long long local_memory =
        static_cast<unsigned long long>((world.energy.size() + world.energy_swap.size() +
                                         world.accumulated_flux.size()) * sizeof(val_t));
    unsigned long long total_memory = 0, max_memory = 0;
    MPI_Reduce(&local_memory, &total_memory, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_memory, &max_memory, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, comm);
    if (rank == 0) {
        std::printf("Memory usage: %.2f MB aggregate, %.2f MB maximum per rank\n\n",
                    total_memory / (1024.0 * 1024.0), max_memory / (1024.0 * 1024.0));
        std::printf("Running simulation...\n");
    }

    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters, comm);
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    const std::uint64_t result_hash = computeHash(world, comm);
    if (rank == 0) {
        const int measured_iters = std::max(n_iters - 1, 1);
        const double duration_ms = seconds * 1000.0;
        const double time_per_iter = duration_ms / measured_iters;
        const double giga_elements_per_second = seconds > 0.0
            ? (static_cast<double>(measured_iters) * static_cast<double>(global_elements)) /
              seconds / 1e9
            : 0.0;
        std::printf("Computation time: %.3f ms\n", duration_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iter);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elements_per_second);
        std::printf("  Performance: %.4f GFLOPS\n", giga_elements_per_second * 22.0);
        std::printf("  Result hash: %016llX\n\n",
                    static_cast<unsigned long long>(result_hash));
    }

    if (print_results_requested) printDistributedResults(world, rank, ranks, comm);
    const bool valid = !validate || validateResults(world, rank, comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
