#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Each rank owns a contiguous block of rows.  Dynamic arrays contain owned
// elements only; the two adjacent rows are exchanged as compact energy halos.
struct World {
    std::vector<Material> materials;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    std::vector<val_t> top_halo;
    std::vector<val_t> bottom_halo;
    std::vector<val_t> top_send;
    std::vector<val_t> bottom_send;
    int grid_width = 0;
    int first_row = 0;
    int local_rows = 0;
    int previous_rank = MPI_PROC_NULL;
    int next_rank = MPI_PROC_NULL;
};

inline val_t computeFlux(const Material& mat, const val_t this_energy,
                         const val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * 1.0 * 0.25;
}

void buildSquare2D(World& world, const int n_elems_root, const int rank,
                   const int n_ranks) {
    world.grid_width = n_elems_root;
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};

    const int active_ranks = std::min(n_ranks, n_elems_root);
    const int base_rows = n_elems_root / active_ranks;
    const int extra_rows = n_elems_root % active_ranks;
    world.local_rows = rank < active_ranks ? base_rows + (rank < extra_rows) : 0;
    world.first_row = rank < active_ranks
        ? rank * base_rows + std::min(rank, extra_rows) : n_elems_root;

    if (world.local_rows != 0) {
        world.previous_rank = rank == 0 ? MPI_PROC_NULL : rank - 1;
        world.next_rank = rank + 1 == active_ranks ? MPI_PROC_NULL : rank + 1;
    }

    const size_t local_elements = static_cast<size_t>(world.local_rows) * n_elems_root;
    world.elements_dynamic.assign(local_elements, {0.0, 0.0});
    world.elements_dynamic_swap.resize(local_elements);
    world.top_halo.resize(n_elems_root);
    world.bottom_halo.resize(n_elems_root);
    world.top_send.resize(n_elems_root);
    world.bottom_send.resize(n_elems_root);
}

inline idx_t materialIndex(const World& world, const int local_row, const int y) {
    const int x = world.first_row + local_row;
    const int last = world.grid_width - 1;
    if ((x == 0 || x == last) && (y == 0 || y == last)) {
        return (x == y) ? INFLOW_MAT_ID : OUTFLOW_MAT_ID;
    }
    return DEFAULT_MAT_ID;
}

inline void updateRow(World& world, const int local_row) {
    const int n = world.grid_width;
    const int global_row = world.first_row + local_row;
    const ElementDynamic* const read = world.elements_dynamic.data();
    ElementDynamic* const write = world.elements_dynamic_swap.data();
    const size_t row_begin = static_cast<size_t>(local_row) * n;

    for (int y = 0; y < n; ++y) {
        const size_t i = row_begin + y;
        const ElementDynamic& elem = read[i];
        const Material& mat = world.materials[materialIndex(world, local_row, y)];
        val_t total_flux = mat.external_flow;

        // Preserve the original connection order: down, up, right, left.
        if (global_row + 1 < n) {
            const val_t neighbor = local_row + 1 < world.local_rows
                ? read[i + n].current_energy : world.bottom_halo[y];
            total_flux += computeFlux(mat, elem.current_energy, neighbor);
        }
        if (global_row > 0) {
            const val_t neighbor = local_row > 0
                ? read[i - n].current_energy : world.top_halo[y];
            total_flux += computeFlux(mat, elem.current_energy, neighbor);
        }
        if (y + 1 < n) {
            total_flux += computeFlux(mat, elem.current_energy, read[i + 1].current_energy);
        }
        if (y > 0) {
            total_flux += computeFlux(mat, elem.current_energy, read[i - 1].current_energy);
        }

        write[i] = {elem.current_energy + total_flux, elem.total_flux + std::abs(total_flux)};
    }
}

void runSimulation(World& world, const int n_iters) {
    const int n = world.grid_width;
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        int request_count = 0;
        if (world.local_rows != 0) {
            // ElementDynamic is an array-of-structures, so pack just the
            // energy field rather than communicating flux history as well.
            const size_t last_row = static_cast<size_t>(world.local_rows - 1) * n;
            for (int y = 0; y < n; ++y) {
                world.top_send[y] = world.elements_dynamic[y].current_energy;
                world.bottom_send[y] = world.elements_dynamic[last_row + y].current_energy;
            }
            if (world.previous_rank != MPI_PROC_NULL) {
                MPI_Irecv(world.top_halo.data(), n, MPI_DOUBLE, world.previous_rank, 0,
                          MPI_COMM_WORLD, &requests[request_count++]);
                MPI_Isend(world.top_send.data(), n, MPI_DOUBLE,
                          world.previous_rank, 1, MPI_COMM_WORLD, &requests[request_count++]);
            }
            if (world.next_rank != MPI_PROC_NULL) {
                MPI_Irecv(world.bottom_halo.data(), n, MPI_DOUBLE, world.next_rank, 1,
                          MPI_COMM_WORLD, &requests[request_count++]);
                MPI_Isend(world.bottom_send.data(), n, MPI_DOUBLE, world.next_rank, 0, MPI_COMM_WORLD,
                          &requests[request_count++]);
            }

            // Hide halo latency behind the independent interior rows.
            for (int row = 1; row + 1 < world.local_rows; ++row) {
                updateRow(world, row);
            }
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

            updateRow(world, 0);
            if (world.local_rows > 1) {
                updateRow(world, world.local_rows - 1);
            }
        }
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

bool validateResults(const World& world, const int rank) {
    val_t local_energy_sum = 0.0, local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    for (const auto& elem : world.elements_dynamic) {
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(elem.current_energy, local_energy_max);
        local_energy_min = std::min(elem.current_energy, local_energy_min);
    }

    val_t energy_sum, flux_sum, energy_max, energy_min;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank != 0) return true;
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    if (!std::isfinite(energy_sum) || !std::isfinite(flux_sum) ||
        !std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Validation values are not finite\n");
        return false;
    }
    if (std::abs(energy_sum) > 1e-8) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    const size_t first_element = static_cast<size_t>(world.first_row) * world.grid_width;
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &world.elements_dynamic[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &world.elements_dynamic[i].total_flux, sizeof(flux_bits));
        const uint64_t global_i = first_element + i;
        hash ^= (energy_bits + global_i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -n <num>     Grid size (NxN elements) (default: 512)\n"
           "  -i <num>     Number of simulation iterations (default: 10)\n"
           "  -v           Enable validation\n  -r           Print results for external validation\n"
           "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    int n_elems_root = 512, n_iters = 10, validate = 0, print_results_flag = 0;
    int parse_ok = 1, show_help = 0;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_elems_root = atoi(argv[++i]);
            else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
            else if (strcmp(argv[i], "-v") == 0) validate = 1;
            else if (strcmp(argv[i], "-r") == 0) print_results_flag = 1;
            else if (strcmp(argv[i], "-h") == 0) show_help = 1;
            else { printf("Unknown option: %s\n", argv[i]); parse_ok = 0; }
        }
        if (n_elems_root <= 0 || n_iters < 0) parse_ok = 0;
    }
    int config[] = {n_elems_root, n_iters, validate, print_results_flag, parse_ok, show_help};
    MPI_Bcast(config, 6, MPI_INT, 0, MPI_COMM_WORLD);
    n_elems_root = config[0]; n_iters = config[1]; validate = config[2];
    print_results_flag = config[3]; parse_ok = config[4]; show_help = config[5];
    if (show_help || !parse_ok) {
        if (rank == 0) { if (!parse_ok) printf("Invalid arguments\n"); printUsage(argv[0]); }
        MPI_Finalize();
        return parse_ok ? 0 : 1;
    }

    const uint64_t n_elems = static_cast<uint64_t>(n_elems_root) * n_elems_root;
    World world;
    buildSquare2D(world, n_elems_root, rank, n_ranks);
    const size_t local_memory = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2 +
                                world.top_halo.size() * sizeof(val_t) * 4;
    size_t total_memory = 0;
    MPI_Reduce(&local_memory, &total_memory, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %llu elements\nIterations: %d\nValidation: %s\nMPI ranks: %d\n\n",
               n_elems_root, n_elems_root, static_cast<unsigned long long>(n_elems), n_iters,
               validate ? "enabled" : "disabled", n_ranks);
        printf("Building distributed unstructured mesh...\n");
        printf("Distributed dynamic memory: %.2f MB\n\n", total_memory / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    runSimulation(world, n_iters);
    const auto end = std::chrono::high_resolution_clock::now();
    const double local_seconds = std::chrono::duration<double>(end - start).count();
    double elapsed_seconds = 0.0;
    MPI_Reduce(&local_seconds, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const uint64_t local_hash = computeHash(world);
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const int measured_iters = std::max(n_iters - 1, 1);
        const double milliseconds = elapsed_seconds * 1000.0;
        const double giga_elems_per_sec = elapsed_seconds > 0.0
            ? (measured_iters * static_cast<double>(n_elems)) / elapsed_seconds / 1e9 : 0.0;
        printf("Computation time: %.3f ms\nPerformance:\n  Time per iteration: %.4f ms\n"
               "  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n"
               "  Result hash: %016llX\n\n", milliseconds, milliseconds / measured_iters,
               giga_elems_per_sec, giga_elems_per_sec * 22.0,
               static_cast<unsigned long long>(global_hash));
    }

    if (print_results_flag) {
        const int local_count = static_cast<int>(world.elements_dynamic.size());
        std::vector<int> counts, displacements;
        if (rank == 0) { counts.resize(n_ranks); displacements.resize(n_ranks); }
        MPI_Gather(&local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<double> energy_data;
        if (rank == 0) {
            int offset = 0;
            for (int r = 0; r < n_ranks; ++r) { displacements[r] = offset; offset += counts[r]; }
            energy_data.resize(offset);
        }
        std::vector<double> local_energy(local_count);
        for (int i = 0; i < local_count; ++i) local_energy[i] = world.elements_dynamic[i].current_energy;
        MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE, rank == 0 ? energy_data.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(energy_data, "ElementEnergy");
    }

    int valid = validate ? validateResults(world, rank) : 1;
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
