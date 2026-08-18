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
constexpr int MAX_CONNECTIONS = 8;

struct Material { val_t transfer_coeff; val_t external_flow; };
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};
struct ElementDynamic { val_t current_energy; val_t total_flux; };

struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static; // Owned elements only.
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    int n_root = 0;
    int row_start = 0;
    int local_rows = 0;
};

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

void buildSquare2D(World& world, int n_root, int row_start, int local_rows) {
    world.n_root = n_root;
    world.row_start = row_start;
    world.local_rows = local_rows;
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};

    const size_t local_count = static_cast<size_t>(local_rows) * n_root;
    world.elements_static.resize(local_count);
    world.elements_dynamic.assign(local_count, {0.0, 0.0});
    world.elements_dynamic_swap.resize(local_count);

    for (int x = row_start; x < row_start + local_rows; ++x) {
        for (int y = 0; y < n_root; ++y) {
            const size_t local_idx = static_cast<size_t>(x - row_start) * n_root + y;
            auto& elem = world.elements_static[local_idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            constexpr int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (const auto& offset : offsets) {
                const int nx = x + offset[0], ny = y + offset[1];
                if (nx >= 0 && nx < n_root && ny >= 0 && ny < n_root) {
                    const int j = static_cast<int>(elem.num_connections++);
                    elem.connected_idx[j] = static_cast<idx_t>(nx) * n_root + ny;
                    elem.connected_flux[j] = 1.0;
                }
            }
        }
    }

    const int last = n_root - 1;
    const auto set_material = [&](int x, int y, idx_t material) {
        if (x >= row_start && x < row_start + local_rows)
            world.elements_static[static_cast<size_t>(x - row_start) * n_root + y].material_idx = material;
    };
    set_material(0, 0, INFLOW_MAT_ID);
    set_material(0, last, OUTFLOW_MAT_ID);
    set_material(last, 0, OUTFLOW_MAT_ID);
    set_material(last, last, INFLOW_MAT_ID);
}

inline val_t computeFlux(const Material& mat, val_t this_energy, val_t connection_flux,
                         val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

void runSimulation(World& world, int n_iters, int rank, int active_ranks) {
    const int n_root = world.n_root;
    const int local_rows = world.local_rows;
    const size_t local_count = world.elements_static.size();
    std::vector<val_t> top_halo(n_root), bottom_halo(n_root);
    std::vector<val_t> first_row(n_root), last_row(n_root);
    const int previous = (rank > 0 && rank < active_ranks) ? rank - 1 : MPI_PROC_NULL;
    const int next = (rank + 1 < active_ranks) ? rank + 1 : MPI_PROC_NULL;

    for (int iter = 0; iter < n_iters; ++iter) {
        if (local_rows > 0) {
            for (int y = 0; y < n_root; ++y) {
                first_row[y] = world.elements_dynamic[y].current_energy;
                last_row[y] = world.elements_dynamic[(local_rows - 1) * n_root + y].current_energy;
            }
            MPI_Sendrecv(first_row.data(), n_root, MPI_DOUBLE, previous, 0,
                         bottom_halo.data(), n_root, MPI_DOUBLE, next, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            MPI_Sendrecv(last_row.data(), n_root, MPI_DOUBLE, next, 1,
                         top_halo.data(), n_root, MPI_DOUBLE, previous, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (size_t i = 0; i < local_count; ++i) {
            const auto& stat = world.elements_static[i];
            const val_t this_energy = world.elements_dynamic[i].current_energy;
            const Material& mat = world.materials[stat.material_idx];
            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < stat.num_connections; ++j) {
                const idx_t global_neighbor = stat.connected_idx[j];
                const int neighbor_row = static_cast<int>(global_neighbor / n_root);
                const int neighbor_col = static_cast<int>(global_neighbor % n_root);
                val_t neighbor_energy;
                if (neighbor_row < world.row_start)
                    neighbor_energy = top_halo[neighbor_col];
                else if (neighbor_row >= world.row_start + local_rows)
                    neighbor_energy = bottom_halo[neighbor_col];
                else
                    neighbor_energy = world.elements_dynamic[static_cast<size_t>(neighbor_row - world.row_start) * n_root + neighbor_col].current_energy;
                total_flux += computeFlux(mat, this_energy, stat.connected_flux[j], neighbor_energy);
            }
            auto& write = world.elements_dynamic_swap[i];
            write.current_energy = this_energy + total_flux;
            write.total_flux = world.elements_dynamic[i].total_flux + std::abs(total_flux);
        }
        world.elements_dynamic.swap(world.elements_dynamic_swap);
    }
}

bool validateResults(const World& world) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (const auto& elem : world.elements_dynamic) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n",
           energy_sum, flux_sum, energy_min, energy_max);
    if (!std::isfinite(energy_sum) || !std::isfinite(flux_sum) ||
        !std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Non-finite result\n");
        return false;
    }
    if (std::abs(energy_sum) > 1e-8)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        const auto* e = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const auto* f = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n  -n <num> Grid size (NxN) (default: 512)\n  -i <num> Iterations (default: 10)\n  -v Validation\n  -r Print results\n  -h Help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_root = 512, n_iters = 10;
    bool validate = false, print_results_flag = false, parse_ok = true;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n_root = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) printf("Unknown option: %s\n", argv[i]); parse_ok = false; }
    }
    if (n_root <= 0 || n_iters < 0) parse_ok = false;
    if (!parse_ok) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 1; }

    const int active_ranks = std::min(size, n_root);
    const int base = n_root / active_ranks, extra = n_root % active_ranks;
    const int row_start = rank < active_ranks ? rank * base + std::min(rank, extra) : n_root;
    const int local_rows = rank < active_ranks ? base + (rank < extra ? 1 : 0) : 0;
    World world;
    buildSquare2D(world, n_root, row_start, local_rows);

    if (rank == 0) {
        const size_t n = static_cast<size_t>(n_root) * n_root;
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI)\n============================================\n");
        printf("Grid size: %d x %d = %zu elements\nIterations: %d\nMPI ranks: %d\nValidation: %s\n\n",
               n_root, n_root, n, n_iters, size, validate ? "enabled" : "disabled");
        const size_t static_mem = n * sizeof(ElementStatic), dynamic_mem = n * sizeof(ElementDynamic) * 2;
        printf("Distributed memory per rank (rank 0): %.2f MB (local static: %.2f MB, dynamic: %.2f MB)\n\n",
               (world.elements_static.size() * sizeof(ElementStatic) + world.elements_dynamic.size() * sizeof(ElementDynamic) * 2) / (1024.0 * 1024.0),
               world.elements_static.size() * sizeof(ElementStatic) / (1024.0 * 1024.0), dynamic_mem / (1024.0 * 1024.0));
        (void)static_mem;
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    runSimulation(world, n_iters, rank, active_ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    const double local_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    double elapsed_ms = 0.0;
    MPI_Reduce(&local_ms, &elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const int rows = r < active_ranks ? base + (r < extra ? 1 : 0) : 0;
        const int first = r < active_ranks ? r * base + std::min(r, extra) : n_root;
        counts[r] = rows * n_root;
        displs[r] = first * n_root;
    }
    const int local_count = static_cast<int>(world.elements_dynamic.size());
    std::vector<val_t> local_energy(local_count), local_flux(local_count);
    for (int i = 0; i < local_count; ++i) {
        local_energy[i] = world.elements_dynamic[i].current_energy;
        local_flux[i] = world.elements_dynamic[i].total_flux;
    }
    std::vector<val_t> all_energy, all_flux;
    if (rank == 0) { all_energy.resize(static_cast<size_t>(n_root) * n_root); all_flux.resize(all_energy.size()); }
    MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE,
                rank == 0 ? all_energy.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_flux.data(), local_count, MPI_DOUBLE, rank == 0 ? all_flux.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int result = 0;
    if (rank == 0) {
        std::vector<ElementDynamic> global(all_energy.size());
        for (size_t i = 0; i < global.size(); ++i) { global[i].current_energy = all_energy[i]; global[i].total_flux = all_flux[i]; }
        const int measured = std::max(n_iters - 1, 1);
        const double geps = static_cast<double>(measured) * global.size() / (elapsed_ms / 1000.0) / 1e9;
        printf("Computation time: %.3f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016lX\n\n",
               elapsed_ms, elapsed_ms / measured, geps, geps * 22.0, computeHash(global));
        if (print_results_flag) { std::vector<double> energy(global.size()); for (size_t i = 0; i < global.size(); ++i) energy[i] = global[i].current_energy; print_results(energy, "ElementEnergy"); }
        if (validate) { World full; full.elements_dynamic = std::move(global); result = validateResults(full) ? 0 : 1; }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
