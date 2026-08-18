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

// Each rank owns complete rows [row_begin, row_end).  The two extra row
// regions in the dynamic arrays are receive buffers for the neighboring ranks.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    int n_root = 0;
    int row_begin = 0;
    int row_end = 0;
    int rank = 0;
    int ranks = 1;
    MPI_Comm communicator = MPI_COMM_WORLD;
};

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

void buildSquare2D(World& world, int n_root, int rank, int ranks) {
    world.n_root = n_root;
    world.rank = rank;
    world.ranks = ranks;

    const int base_rows = n_root / ranks;
    const int extra_rows = n_root % ranks;
    world.row_begin = rank * base_rows + std::min(rank, extra_rows);
    world.row_end = world.row_begin + base_rows + (rank < extra_rows ? 1 : 0);
    const size_t local_rows = static_cast<size_t>(world.row_end - world.row_begin);
    const size_t local_count = local_rows * static_cast<size_t>(n_root);
    const size_t ghost_count = static_cast<size_t>(2) * n_root;

    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(local_count);
    world.elements_dynamic.assign(local_count + ghost_count, {0.0, 0.0});
    world.elements_dynamic_swap.assign(local_count + ghost_count, {0.0, 0.0});

    for (int x = world.row_begin; x < world.row_end; ++x) {
        for (int y = 0; y < n_root; ++y) {
            const size_t local = static_cast<size_t>(x - world.row_begin) * n_root + y;
            ElementStatic& elem = world.elements_static[local];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (const auto& offset : offsets) {
                const int nx = x + offset[0];
                const int ny = y + offset[1];
                if (nx < 0 || nx >= n_root || ny < 0 || ny >= n_root) {
                    continue;
                }

                size_t neighbor;
                if (nx < world.row_begin) {
                    neighbor = local_count + static_cast<size_t>(ny);
                } else if (nx >= world.row_end) {
                    neighbor = local_count + static_cast<size_t>(n_root) + ny;
                } else {
                    neighbor = static_cast<size_t>(nx - world.row_begin) * n_root + ny;
                }
                elem.connected_idx[elem.num_connections] = neighbor;
                elem.connected_flux[elem.num_connections++] = 1.0;
            }
        }
    }

    const int last = n_root - 1;
    if (0 >= world.row_begin && 0 < world.row_end) {
        world.elements_static[0].material_idx = INFLOW_MAT_ID;
        world.elements_static[last].material_idx = OUTFLOW_MAT_ID;
    }
    if (last >= world.row_begin && last < world.row_end) {
        const size_t first = static_cast<size_t>(last - world.row_begin) * n_root;
        world.elements_static[first].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[first + last].material_idx = INFLOW_MAT_ID;
    }
}

inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                         val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

void exchangeGhostRows(World& world, MPI_Datatype energy_row) {
    const int n = world.n_root;
    const size_t local_count = world.elements_static.size();
    auto& current = world.elements_dynamic;
    MPI_Sendrecv(&current[0].current_energy, 1, energy_row,
                 world.rank > 0 ? world.rank - 1 : MPI_PROC_NULL, 0,
                 &current[local_count + n].current_energy, 1, energy_row,
                 world.rank + 1 < world.ranks ? world.rank + 1 : MPI_PROC_NULL, 0,
                 world.communicator, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&current[local_count - n].current_energy, 1, energy_row,
                 world.rank + 1 < world.ranks ? world.rank + 1 : MPI_PROC_NULL, 1,
                 &current[local_count].current_energy, 1, energy_row,
                 world.rank > 0 ? world.rank - 1 : MPI_PROC_NULL, 1,
                 world.communicator, MPI_STATUS_IGNORE);
}

void runSimulation(World& world, int n_iters) {
    const size_t local_count = world.elements_static.size();
    MPI_Datatype energy_row;
    MPI_Type_vector(world.n_root, 1, 2, MPI_DOUBLE, &energy_row);
    MPI_Type_commit(&energy_row);
    for (int iter = 0; iter < n_iters; ++iter) {
        exchangeGhostRows(world, energy_row);
        for (size_t i = 0; i < local_count; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j],
                                          world.elements_dynamic[elem_static.connected_idx[j]]);
            }
            ElementDynamic& write = world.elements_dynamic_swap[i];
            write.current_energy = elem_dyn.current_energy + total_flux;
            write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
    MPI_Type_free(&energy_row);
}

uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    const size_t n = world.elements_static.size();
    const size_t offset = static_cast<size_t>(world.row_begin) * world.n_root;
    for (size_t i = 0; i < n; ++i) {
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
        const uint64_t global_index = offset + i;
        hash ^= (*e_ptr + global_index) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + global_index) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

void printUsage(const char* prog) {
    printf("Usage: %s [options]\nOptions:\n", prog);
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n_root = 512, n_iters = 10;
    bool validate = false, print_results_flag = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            int value = std::atoi(argv[++i]);
            if (std::strcmp(argv[i - 1], "-n") == 0) n_root = value; else n_iters = value;
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) print_results_flag = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else bad = true;
    }
    if (help || bad || n_root <= 0 || n_iters < 0) {
        if (world_rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return bad || n_root <= 0 || n_iters < 0;
    }

    const int active_ranks = std::min(world_size, n_root);
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 0 : MPI_UNDEFINED,
                   world_rank, &active_comm);
    if (world_rank >= active_ranks) {
        MPI_Finalize();
        return 0;
    }

    int active_rank = 0;
    MPI_Comm_rank(active_comm, &active_rank);
    World world;
    world.communicator = active_comm;
    buildSquare2D(world, n_root, active_rank, active_ranks);
    const size_t global_count = static_cast<size_t>(n_root) * n_root;
    if (active_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %zu elements\nIterations: %d\nValidation: %s\n\n",
               n_root, n_root, global_count, n_iters, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", active_ranks);
        printf("Memory usage per rank: %.2f MB\n\n", (world.elements_static.size() * sizeof(ElementStatic) +
               world.elements_dynamic.size() * sizeof(ElementDynamic) * 2) / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    MPI_Barrier(active_comm);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters);
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm);

    const int measured = std::max(n_iters - 1, 1);
    if (active_rank == 0) {
        const double measured_seconds = std::max(seconds, std::numeric_limits<double>::min());
        const double ms = measured_seconds * 1000.0;
        const double elements_per_sec = measured * static_cast<double>(global_count) / measured_seconds;
        printf("Computation time: %.0f ms\nPerformance:\n  Time per iteration: %.4f ms\n", ms, ms / measured);
        printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n",
               elements_per_sec / 1e9, elements_per_sec * 22.0 / 1e9);
    }

    uint64_t local_hash = computeHash(world), global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, active_comm);
    if (active_rank == 0) printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(global_hash));

    if (print_results_flag) {
        const int local_count = static_cast<int>(world.elements_static.size());
        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<double> energies;
        if (active_rank == 0) {
            counts.resize(active_ranks);
            displacements.resize(active_ranks);
            for (int r = 0; r < active_ranks; ++r) {
                const int begin = r * (n_root / active_ranks) + std::min(r, n_root % active_ranks);
                const int rows = n_root / active_ranks + (r < n_root % active_ranks ? 1 : 0);
                counts[r] = rows * n_root;
                displacements[r] = begin * n_root;
            }
            energies.resize(global_count);
        }
        std::vector<double> local_energies(world.elements_static.size());
        for (size_t i = 0; i < local_energies.size(); ++i)
            local_energies[i] = world.elements_dynamic[i].current_energy;
        MPI_Gatherv(local_energies.data(), local_count, MPI_DOUBLE,
                    active_rank == 0 ? energies.data() : nullptr,
                    active_rank == 0 ? counts.data() : nullptr,
                    active_rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, active_comm);
        if (active_rank == 0) print_results(energies, "ElementEnergy");
    }

    if (validate) {
        val_t local_energy = 0.0, local_flux = 0.0, local_min = std::numeric_limits<val_t>::max();
        val_t local_max = std::numeric_limits<val_t>::lowest();
        for (size_t i = 0; i < world.elements_static.size(); ++i) {
            local_energy += world.elements_dynamic[i].current_energy;
            local_flux += world.elements_dynamic[i].total_flux;
            local_min = std::min(local_min, world.elements_dynamic[i].current_energy);
            local_max = std::max(local_max, world.elements_dynamic[i].current_energy);
        }
        val_t energy = 0.0, flux = 0.0, min_energy = 0.0, max_energy = 0.0;
        MPI_Reduce(&local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, active_comm);
        MPI_Reduce(&local_flux, &flux, 1, MPI_DOUBLE, MPI_SUM, 0, active_comm);
        MPI_Reduce(&local_min, &min_energy, 1, MPI_DOUBLE, MPI_MIN, 0, active_comm);
        MPI_Reduce(&local_max, &max_energy, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm);
        int all_valid = 1;
        if (active_rank == 0) {
            all_valid = std::isfinite(energy) && std::isfinite(flux) &&
                        std::isfinite(min_energy) && std::isfinite(max_energy);
        }
        MPI_Bcast(&all_valid, 1, MPI_INT, 0, active_comm);
        if (active_rank == 0) {
            printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n",
                   energy, flux, min_energy, max_energy);
            printf("  Validation: %s\n", all_valid ? "PASSED" : "FAILED");
        }
        if (!all_valid) {
            MPI_Comm_free(&active_comm);
            MPI_Finalize();
            return 1;
        }
    }
    MPI_Comm_free(&active_comm);
    MPI_Finalize();
    return 0;
}
