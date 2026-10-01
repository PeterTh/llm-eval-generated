#include <algorithm>
#include <cstddef>
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
static_assert(offsetof(ElementDynamic, current_energy) == 0);
static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t));

struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    // Owned elements start at offset n; the two extra rows are receive halos.
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// Assign whole grid rows in global order. This also gives adjacent ranks
// adjacent rows and balances the work to within one row.
int rowBegin(int rank, int n, int ranks) {
    return static_cast<int>((static_cast<int64_t>(rank) * n) / ranks);
}

void buildSquare2D(World& world, int n, int first_row, int rows) {
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(static_cast<size_t>(rows) * n);
    world.elements_dynamic.resize(static_cast<size_t>(rows + 2) * n);
    world.elements_dynamic_swap.resize(static_cast<size_t>(rows + 2) * n);

    for (int x = 0; x < rows; ++x) {
        const int global_x = first_row + x;
        for (int y = 0; y < n; ++y) {
            const size_t local_i = static_cast<size_t>(x) * n + y;
            ElementStatic& elem = world.elements_static[local_i];
            elem.material_idx = 0;
            elem.num_connections = 0;
            if ((global_x == 0 && y == 0) || (global_x == n - 1 && y == n - 1))
                elem.material_idx = 1;
            if ((global_x == 0 && y == n - 1) || (global_x == n - 1 && y == 0))
                elem.material_idx = 2;
            // The original corner assignments overwrite one another for n=1.
            if (n == 1) elem.material_idx = 1;

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int j = 0; j < 4; ++j) {
                const int nx = global_x + offsets[j][0];
                const int ny = y + offsets[j][1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    elem.connected_idx[elem.num_connections] =
                        static_cast<idx_t>(nx - first_row + 1) * n + ny;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    ++elem.num_connections;
                }
            }
        }
    }
}

inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                         val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

void updateRange(World& world, size_t begin, size_t end, int n) {
    for (size_t i = begin; i < end; ++i) {
        const ElementStatic& elem_static = world.elements_static[i];
        const ElementDynamic& elem_dyn = world.elements_dynamic[i + n];
        const Material& mat = world.materials[elem_static.material_idx];
        val_t total_flux = mat.external_flow;
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const ElementDynamic& neighbor_dyn =
                world.elements_dynamic[elem_static.connected_idx[j]];
            total_flux += computeFlux(mat, elem_dyn,
                                      elem_static.connected_flux[j], neighbor_dyn);
        }
        ElementDynamic& elem_write = world.elements_dynamic_swap[i + n];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

void runSimulation(World& world, int n, int rows, int n_iters,
                   int rank, int ranks, MPI_Comm comm) {
    // Only current_energy is read from neighbors. A vector type exchanges that
    // field directly from the AoS storage without packing total_flux.
    MPI_Datatype energy_row;
    MPI_Type_vector(n, 1, 2, MPI_DOUBLE, &energy_row);
    MPI_Type_commit(&energy_row);
    const size_t owned = static_cast<size_t>(rows) * n;
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        int count = 0;
        if (rank > 0) {
            MPI_Irecv(&world.elements_dynamic[0].current_energy, 1, energy_row,
                      rank - 1, 0, comm, &requests[count++]);
            MPI_Isend(&world.elements_dynamic[n].current_energy, 1, energy_row,
                      rank - 1, 1, comm, &requests[count++]);
        }
        if (rank + 1 < ranks) {
            MPI_Irecv(&world.elements_dynamic[(static_cast<size_t>(rows) + 1) * n].current_energy,
                      1, energy_row, rank + 1, 1, comm, &requests[count++]);
            MPI_Isend(&world.elements_dynamic[static_cast<size_t>(rows) * n].current_energy, 1,
                      energy_row, rank + 1, 0, comm, &requests[count++]);
        }
        if (rows > 2) updateRange(world, n, owned - n, n);
        if (count) MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        updateRange(world, 0, n, n);
        if (rows > 1) updateRange(world, owned - n, owned, n);
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
    MPI_Type_free(&energy_row);
}

bool validateResults(const World& world, int n, MPI_Comm comm, int rank) {
    val_t local_energy = 0.0, local_flux = 0.0;
    val_t local_max = std::numeric_limits<val_t>::lowest();
    val_t local_min = std::numeric_limits<val_t>::max();
    for (size_t i = n; i < world.elements_dynamic.size() - n; ++i) {
        const auto& elem = world.elements_dynamic[i];
        local_energy += elem.current_energy;
        local_flux += elem.total_flux;
        local_max = std::max(elem.current_energy, local_max);
        local_min = std::min(elem.current_energy, local_min);
    }
    val_t energy_sum, flux_sum, energy_max, energy_min;
    MPI_Reduce(&local_energy, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_flux, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    if (rank != 0) return true;
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    bool valid = true;
    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        valid = false;
    }
    if (std::abs(energy_sum) > 1e-8)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        valid = false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        valid = false;
    }
    if (valid) printf("  Validation: PASSED\n");
    return valid;
}

uint64_t computeHash(const World& world, int n, int first_row) {
    uint64_t hash = 0;
    const size_t global_begin = static_cast<size_t>(first_row) * n;
    for (size_t i = 0; i < world.elements_static.size(); ++i) {
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &world.elements_dynamic[i + n].current_energy, sizeof(uint64_t));
        std::memcpy(&flux_bits, &world.elements_dynamic[i + n].total_flux, sizeof(uint64_t));
        hash ^= (energy_bits + global_begin + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + global_begin + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_ranks);
    int n = 512, n_iters = 10;
    bool validate = false, printResults = false;
    int exit_code = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (n <= 0 || static_cast<int64_t>(n) * n > std::numeric_limits<int>::max() || n_iters < 0) {
        if (world_rank == 0) fprintf(stderr, "Grid size and iteration count must be valid nonnegative values.\n");
        MPI_Finalize();
        return 1;
    }
    const int n_elems = n * n;
    const int active_ranks = std::min(n, world_ranks);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 0 : MPI_UNDEFINED,
                   world_rank, &comm);
    if (world_rank >= active_ranks) {
        MPI_Finalize();
        return 0;
    }
    const int rank = world_rank;
    const int first_row = rowBegin(rank, n, active_ranks);
    const int rows = rowBegin(rank + 1, n, active_ranks) - first_row;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n, first_row, rows);
    const unsigned long long local_mem[2] = {
        static_cast<unsigned long long>(world.elements_static.size() * sizeof(ElementStatic)),
        static_cast<unsigned long long>((world.elements_dynamic.size() +
             world.elements_dynamic_swap.size()) * sizeof(ElementDynamic))};
    unsigned long long global_mem[2] = {};
    MPI_Reduce(local_mem, global_mem, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    if (rank == 0) {
        const double static_mb = global_mem[0] / (1024.0 * 1024.0);
        const double dynamic_mb = global_mem[1] / (1024.0 * 1024.0);
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               static_mb + dynamic_mb, static_mb, dynamic_mb);
        printf("Running simulation...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    runSimulation(world, n, rows, n_iters, rank, active_ranks, comm);
    const double local_ms = (MPI_Wtime() - start) * 1000.0;
    double duration_ms = 0.0;
    MPI_Reduce(&local_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    const uint64_t local_hash = computeHash(world, n, first_row);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration_ms));
        const int measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / measured_iters;
        const double giga_elems_per_sec = (measured_iters * static_cast<double>(n_elems)) /
                                          (duration_ms / 1000.0) / 1e9;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
    }
    if (printResults) {
        std::vector<double> local_energy(static_cast<size_t>(rows) * n);
        for (size_t i = 0; i < local_energy.size(); ++i)
            local_energy[i] = world.elements_dynamic[i + n].current_energy;
        std::vector<int> counts, displacements;
        std::vector<double> energy_data;
        if (rank == 0) {
            counts.resize(active_ranks);
            displacements.resize(active_ranks);
            energy_data.resize(n_elems);
            for (int r = 0; r < active_ranks; ++r) {
                displacements[r] = rowBegin(r, n, active_ranks) * n;
                counts[r] = (rowBegin(r + 1, n, active_ranks) -
                             rowBegin(r, n, active_ranks)) * n;
            }
        }
        MPI_Gatherv(local_energy.data(), static_cast<int>(local_energy.size()), MPI_DOUBLE,
                    rank == 0 ? energy_data.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(energy_data, "ElementEnergy");
    }
    if (validate) {
        const bool valid = validateResults(world, n, comm, rank);
        if (rank == 0 && !valid) exit_code = 1;
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return exit_code;
}
