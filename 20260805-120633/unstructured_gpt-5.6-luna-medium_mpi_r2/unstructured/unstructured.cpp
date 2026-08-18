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

constexpr int MAX_CONNECTIONS = 8;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

struct World {
    int n_elems_root = 0;
    int row_start = 0;
    int local_rows = 0;
    int rank = 0;
    int size = 1;
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    // Rows 0 and local_rows + 1 are the top and bottom energy halos.
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    MPI_Datatype halo_type = MPI_DATATYPE_NULL;
};

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

void buildSquare2D(World& world, int n, int rank, int size) {
    world.n_elems_root = n;
    world.rank = rank;
    world.size = size;

    const int active_ranks = std::min(size, n);
    const int base = n / active_ranks;
    const int remainder = n % active_ranks;
    world.local_rows = rank < active_ranks ? base + (rank < remainder ? 1 : 0) : 0;
    world.row_start = rank < active_ranks
        ? rank * base + std::min(rank, remainder) : n;

    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    const size_t local_elements = static_cast<size_t>(world.local_rows) * n;
    world.elements_static.resize(local_elements);
    world.elements_dynamic.assign(static_cast<size_t>(world.local_rows + 2) * n, {});
    world.elements_dynamic_swap.assign(static_cast<size_t>(world.local_rows + 2) * n, {});

    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int lr = 0; lr < world.local_rows; ++lr) {
        const int x = world.row_start + lr;
        for (int y = 0; y < n; ++y) {
            const size_t local_idx = static_cast<size_t>(lr) * n + y;
            ElementStatic& elem = world.elements_static[local_idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            for (const auto& offset : offsets) {
                const int nx = x + offset[0];
                const int ny = y + offset[1];
                if (nx < 0 || nx >= n || ny < 0 || ny >= n) continue;
                // Local row 0 and local_rows+1 are the two halo rows.
                const int dynamic_row = nx - world.row_start + 1;
                elem.connected_idx[elem.num_connections++] =
                    static_cast<idx_t>(dynamic_row * n + ny);
                elem.connected_flux[elem.num_connections - 1] = 1.0;
            }
            if ((x == 0 || x == n - 1) && (y == 0 || y == n - 1)) {
                elem.material_idx = (x == 0 && y == 0) || (x == n - 1 && y == n - 1)
                    ? INFLOW_MAT_ID : OUTFLOW_MAT_ID;
            }
        }
    }

    // A vector datatype addresses only current_energy in an interleaved row.
    MPI_Type_vector(n, 1, 2, MPI_DOUBLE, &world.halo_type);
    MPI_Type_commit(&world.halo_type);
}

inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                         val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

void exchangeHalos(World& world) {
    if (world.local_rows == 0) return;
    const int up = world.rank > 0 ? world.rank - 1 : MPI_PROC_NULL;
    const int down = world.rank + 1 < std::min(world.size, world.n_elems_root)
        ? world.rank + 1 : MPI_PROC_NULL;
    auto& dyn = world.elements_dynamic;
    const int n = world.n_elems_root;

    MPI_Sendrecv(&dyn[static_cast<size_t>(world.local_rows) * n + 0].current_energy, 1,
                 world.halo_type, down, 0,
                 &dyn[0].current_energy, 1, world.halo_type, up, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&dyn[static_cast<size_t>(n) + 0].current_energy, 1,
                 world.halo_type, up, 1,
                 &dyn[static_cast<size_t>(world.local_rows + 1) * n].current_energy, 1,
                 world.halo_type, down, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void runSimulation(World& world, int n_iters) {
    const int n = world.n_elems_root;
    for (int iter = 0; iter < n_iters; ++iter) {
        exchangeHalos(world);
        const auto& input = world.elements_dynamic;
        auto& output = world.elements_dynamic_swap;
        for (int lr = 0; lr < world.local_rows; ++lr) {
            const size_t static_base = static_cast<size_t>(lr) * n;
            const size_t dynamic_base = static_cast<size_t>(lr + 1) * n;
            for (int y = 0; y < n; ++y) {
                const ElementStatic& elem_static = world.elements_static[static_base + y];
                const ElementDynamic& elem_dyn = input[dynamic_base + y];
                const Material& mat = world.materials[elem_static.material_idx];
                val_t total_flux = mat.external_flow;
                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j],
                        input[elem_static.connected_idx[j]]);
                }
                ElementDynamic& elem_write = output[dynamic_base + y];
                elem_write.current_energy = elem_dyn.current_energy + total_flux;
                elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }
        }
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

void localStatistics(const World& world, val_t& energy_sum, val_t& flux_sum,
                     val_t& energy_min, val_t& energy_max) {
    energy_sum = flux_sum = 0.0;
    energy_min = std::numeric_limits<val_t>::max();
    energy_max = std::numeric_limits<val_t>::lowest();
    const size_t n_local = static_cast<size_t>(world.local_rows) * world.n_elems_root;
    for (size_t i = 0; i < n_local; ++i) {
        const auto& elem = world.elements_dynamic[world.n_elems_root + i];
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_min = std::min(energy_min, elem.current_energy);
        energy_max = std::max(energy_max, elem.current_energy);
    }
}

uint64_t localHash(const World& world) {
    uint64_t result = 0;
    const size_t n_local = static_cast<size_t>(world.local_rows) * world.n_elems_root;
    for (size_t i = 0; i < n_local; ++i) {
        const size_t global_i = static_cast<size_t>(world.row_start) * world.n_elems_root + i;
        const auto& elem = world.elements_dynamic[world.n_elems_root + i];
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elem.current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elem.total_flux);
        result ^= (*e_ptr + global_i) * 0x9e3779b97f4a7c15ULL;
        result ^= (*f_ptr + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    return result;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\nOptions:\n", progName);
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_elems_root = 512, n_iters = 10;
    bool validate = false, printResults = false;
    int parse_error = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_elems_root = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { parse_error = 1; }
    }
    int any_error = 0;
    MPI_Allreduce(&parse_error, &any_error, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any_error || n_elems_root <= 0 || n_iters < 0) {
        if (rank == 0) { printf("Invalid command line arguments.\n"); printUsage(argv[0]); }
        MPI_Finalize(); return 1;
    }

    const long long n_elems = static_cast<long long>(n_elems_root) * n_elems_root;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %lld elements\nIterations: %d\nValidation: %s\n\n",
               n_elems_root, n_elems_root, n_elems, n_iters, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, rank, size);
    const size_t local_static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t local_dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    if (rank == 0) {
        printf("Maximum local memory: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (local_static_mem + local_dynamic_mem) / (1024.0 * 1024.0),
               local_static_mem / (1024.0 * 1024.0), local_dynamic_mem / (1024.0 * 1024.0));
        printf("Running simulation on %d MPI ranks...\n", size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters);
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long long duration_ms = static_cast<long long>(seconds * 1000.0);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);
        const int measured_iters = std::max(n_iters - 1, 1);
        const double measured_seconds = std::max(seconds, 1.0e-12);
        const double time_per_iter = measured_seconds * 1000.0 / measured_iters;
        const double giga_elems_per_sec = measured_iters * static_cast<double>(n_elems) / measured_seconds / 1e9;
        printf("Performance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n",
               time_per_iter, giga_elems_per_sec, giga_elems_per_sec * 22.0);
    }

    const uint64_t local_hash = localHash(world), hash = [&] {
        uint64_t value = 0;
        MPI_Reduce(&local_hash, &value, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, MPI_COMM_WORLD);
        return value;
    }();
    if (rank == 0) printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(hash));

    if (printResults) {
        const int local_count = world.local_rows * n_elems_root;
        std::vector<int> counts, displacements;
        std::vector<double> energies;
        if (rank == 0) { counts.resize(size); displacements.resize(size); energies.resize(n_elems); }
        MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) { for (int r = 1; r < size; ++r) displacements[r] = displacements[r - 1] + counts[r - 1]; }
        std::vector<double> local_energy(local_count);
        for (int i = 0; i < local_count; ++i) local_energy[i] = world.elements_dynamic[n_elems_root + i].current_energy;
        MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE, energies.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(energies, "ElementEnergy");
    }

    if (validate) {
        val_t local_energy_sum, local_flux_sum, local_min, local_max;
        localStatistics(world, local_energy_sum, local_flux_sum, local_min, local_max);
        val_t energy_sum = 0.0, flux_sum = 0.0, energy_min, energy_max;
        MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        int valid = 1;
        if (rank == 0) {
            printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", energy_sum, flux_sum, energy_min, energy_max);
            if (!std::isfinite(energy_sum) || !std::isfinite(flux_sum) || !std::isfinite(energy_min) || !std::isfinite(energy_max)) valid = 0;
            if (std::abs(energy_sum) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            printf("  %s\n", valid ? "Validation: PASSED" : "ERROR: Validation failed");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!valid) { MPI_Type_free(&world.halo_type); MPI_Finalize(); return 1; }
    }
    MPI_Type_free(&world.halo_type);
    MPI_Finalize();
    return 0;
}
