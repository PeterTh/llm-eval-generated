#include <algorithm>
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
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

struct Material { val_t transfer_coeff; val_t external_flow; };
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};
struct ElementDynamic { val_t current_energy; val_t total_flux; };

// Each MPI rank owns a contiguous set of grid rows.  Connection indices remain
// global so the unstructured connectivity representation is unchanged.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    std::vector<val_t> top_halo;
    std::vector<val_t> bottom_halo;
    std::vector<val_t> send_top;
    std::vector<val_t> send_bottom;
    int grid_root = 0;
    int first_row = 0;
    int local_rows = 0;
};

void buildSquare2D(World& world, int root, int rank, int ranks) {
    world.grid_root = root;
    const int active_ranks = std::min(root, ranks);
    if (rank < active_ranks) {
        const int base = root / active_ranks;
        const int extra = root % active_ranks;
        world.local_rows = base + (rank < extra ? 1 : 0);
        world.first_row = rank * base + std::min(rank, extra);
    }
    const size_t local_elements = static_cast<size_t>(world.local_rows) * root;

    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(local_elements);
    world.elements_dynamic.assign(local_elements, {0.0, 0.0});
    world.elements_dynamic_swap.resize(local_elements);
    world.top_halo.resize(root);
    world.bottom_halo.resize(root);
    world.send_top.resize(root);
    world.send_bottom.resize(root);

    for (int local_x = 0; local_x < world.local_rows; ++local_x) {
        const int x = world.first_row + local_x;
        for (int y = 0; y < root; ++y) {
            const size_t local_idx = static_cast<size_t>(local_x) * root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            constexpr int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (const auto& offset : offsets) {
                const int nx = x + offset[0], ny = y + offset[1];
                if (nx >= 0 && nx < root && ny >= 0 && ny < root) {
                    const idx_t slot = elem.num_connections++;
                    elem.connected_idx[slot] = static_cast<idx_t>(nx) * root + ny;
                    elem.connected_flux[slot] = 1.0;
                }
            }
            if ((x == 0 && y == 0) || (x == root - 1 && y == root - 1))
                elem.material_idx = INFLOW_MAT_ID;
            else if ((x == 0 && y == root - 1) || (x == root - 1 && y == 0))
                elem.material_idx = OUTFLOW_MAT_ID;
        }
    }
}

inline val_t computeFlux(const Material& mat, const ElementDynamic& self,
                         val_t connection_flux, val_t neighbor_energy) {
    return (neighbor_energy - self.current_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

inline void update_row(World& world, int local_x) {
    const int root = world.grid_root;
    const size_t row_base = static_cast<size_t>(local_x) * root;
    for (int y = 0; y < root; ++y) {
        const size_t i = row_base + y;
        const ElementStatic& stat = world.elements_static[i];
        const ElementDynamic& current = world.elements_dynamic[i];
        val_t total_flux = world.materials[stat.material_idx].external_flow;
        for (idx_t j = 0; j < stat.num_connections; ++j) {
            const idx_t global_neighbor = stat.connected_idx[j];
            const int neighbor_x = static_cast<int>(global_neighbor / root);
            const int neighbor_y = static_cast<int>(global_neighbor % root);
            val_t neighbor_energy;
            if (neighbor_x < world.first_row)
                neighbor_energy = world.top_halo[neighbor_y];
            else if (neighbor_x >= world.first_row + world.local_rows)
                neighbor_energy = world.bottom_halo[neighbor_y];
            else
                neighbor_energy = world.elements_dynamic[
                    static_cast<size_t>(neighbor_x - world.first_row) * root + neighbor_y].current_energy;
            total_flux += computeFlux(world.materials[stat.material_idx], current,
                                      stat.connected_flux[j], neighbor_energy);
        }
        ElementDynamic& next = world.elements_dynamic_swap[i];
        next.current_energy = current.current_energy + total_flux;
        next.total_flux = current.total_flux + std::abs(total_flux);
    }
}

void runSimulation(World& world, int iterations, int rank, int ranks) {
    const int active_ranks = std::min(world.grid_root, ranks);
    if (rank >= active_ranks) return;
    const int up = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int down = rank + 1 == active_ranks ? MPI_PROC_NULL : rank + 1;
    const int row_width = world.grid_root;

    for (int iter = 0; iter < iterations; ++iter) {
        // ElementDynamic contains both energy and accumulated flux, so pack the
        // energy field explicitly instead of sending a strided struct field.
        for (int y = 0; y < row_width; ++y) {
            world.send_top[y] = world.elements_dynamic[y].current_energy;
            world.send_bottom[y] = world.elements_dynamic[
                static_cast<size_t>(world.local_rows - 1) * row_width + y].current_energy;
        }
        MPI_Request requests[4];
        int request_count = 0;
        if (up != MPI_PROC_NULL)
            MPI_Irecv(world.top_halo.data(), row_width, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &requests[request_count++]);
        if (down != MPI_PROC_NULL)
            MPI_Irecv(world.bottom_halo.data(), row_width, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &requests[request_count++]);
        if (up != MPI_PROC_NULL)
            MPI_Isend(world.send_top.data(), row_width, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &requests[request_count++]);
        if (down != MPI_PROC_NULL)
            MPI_Isend(world.send_bottom.data(), row_width, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &requests[request_count++]);

        // Rows not touching a halo do not depend on communication.
        for (int row = 1; row + 1 < world.local_rows; ++row) update_row(world, row);
        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
        if (world.local_rows > 0) update_row(world, 0);
        if (world.local_rows > 1) update_row(world, world.local_rows - 1);
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

uint64_t computeHash(const std::vector<ElementDynamic>& elements, size_t global_offset) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t energy, flux;
        std::memcpy(&energy, &elements[i].current_energy, sizeof(energy));
        std::memcpy(&flux, &elements[i].total_flux, sizeof(flux));
        const uint64_t global_i = global_offset + i;
        hash ^= (energy + global_i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

bool validateResults(const World& world, int rank) {
    val_t local_energy = 0.0, local_flux = 0.0;
    val_t local_max = std::numeric_limits<val_t>::lowest();
    val_t local_min = std::numeric_limits<val_t>::max();
    for (const auto& elem : world.elements_dynamic) {
        local_energy += elem.current_energy; local_flux += elem.total_flux;
        local_max = std::max(local_max, elem.current_energy); local_min = std::min(local_min, elem.current_energy);
    }
    val_t energy, flux, maximum, minimum;
    MPI_Reduce(&local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux, &flux, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank != 0) return true;
    printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", energy, flux, minimum, maximum);
    if (!std::isfinite(energy) || !std::isfinite(flux) || !std::isfinite(maximum) || !std::isfinite(minimum)) {
        printf("  ERROR: Non-finite result\n"); return false;
    }
    if (std::abs(energy) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    printf("  Validation: PASSED\n");
    return true;
}

void printUsage(const char* program) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Grid size (NxN elements) (default: 512)\n  -i <num>     Number of simulation iterations (default: 10)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", program);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int root = 512, iterations = 10; bool validate = false, print_results_flag = false, usage = false, bad_option = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) root = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!std::strcmp(argv[i], "-h")) usage = true;
        else bad_option = true;
    }
    if (usage || bad_option || root <= 0 || iterations < 0) {
        if (rank == 0) { if (bad_option) printf("Invalid option or value\n"); printUsage(argv[0]); }
        MPI_Finalize(); return bad_option ? 1 : 0;
    }

    const int elements = root * root;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %d elements\nIterations: %d\nValidation: %s\nMPI ranks: %d\n\nBuilding unstructured mesh...\n",
               root, root, elements, iterations, validate ? "enabled" : "disabled", ranks);
    }
    World world; buildSquare2D(world, root, rank, ranks);
    const size_t local_mem = world.elements_static.size() * sizeof(ElementStatic) +
                             world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    unsigned long long local_mem_ull = local_mem, global_mem = 0;
    MPI_Reduce(&local_mem_ull, &global_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Memory usage: %.2f MB (distributed across ranks)\n\nRunning simulation...\n", global_mem / (1024.0 * 1024.0));

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime(); runSimulation(world, iterations, rank, ranks);
    double elapsed = MPI_Wtime() - start, elapsed_max = 0.0;
    MPI_Reduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const uint64_t local_hash = computeHash(world.elements_dynamic, static_cast<size_t>(world.first_row) * root);
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double duration_ms = elapsed_max * 1000.0;
        const int measured = std::max(iterations - 1, 1);
        const double giga_elements = duration_ms > 0.0 ? (measured * static_cast<double>(elements)) / (duration_ms * 1e6) : 0.0;
        printf("Computation time: %.3f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n",
               duration_ms, duration_ms / measured, giga_elements, giga_elements * 22.0,
               static_cast<unsigned long long>(global_hash));
    }
    if (print_results_flag) {
        const int local_count = static_cast<int>(world.elements_dynamic.size());
        std::vector<int> counts, displacements;
        if (rank == 0) { counts.resize(ranks); displacements.resize(ranks); }
        MPI_Gather(&local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<val_t> local_energy(local_count), global_energy;
        for (int i = 0; i < local_count; ++i) local_energy[i] = world.elements_dynamic[i].current_energy;
        if (rank == 0) { int offset = 0; for (int r = 0; r < ranks; ++r) { displacements[r] = offset; offset += counts[r]; } global_energy.resize(offset); }
        MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE, rank == 0 ? global_energy.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(global_energy, "ElementEnergy");
    }
    const bool locally_valid = !validate || validateResults(world, rank);
    int valid = locally_valid ? 1 : 0, all_valid = 0;
    MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    MPI_Finalize();
    return all_valid ? 0 : 1;
}
