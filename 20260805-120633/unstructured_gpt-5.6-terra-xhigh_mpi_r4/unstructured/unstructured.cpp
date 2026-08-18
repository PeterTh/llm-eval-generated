#include <algorithm>
#include <cinttypes>
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

// The connectivity is local to an MPI tile.  References into the dynamic
// state include one ghost layer around the tile, retaining the original
// unstructured-mesh update interface in the compute kernel.
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
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    int grid_width = 0;
    int first_global_row = 0;
    int first_global_column = 0;
    int local_rows = 0;
    int local_columns = 0;
    int dynamic_columns = 0;
};

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

inline size_t localElementIndex(const World& world, const int local_row,
                                const int local_column) {
    return static_cast<size_t>(local_row) * world.local_columns + local_column;
}

// The dynamic arrays are padded with a ghost row and column at both ends.
inline size_t dynamicIndex(const World& world, const int local_row,
                           const int local_column) {
    return static_cast<size_t>(local_row + 1) * world.dynamic_columns +
           static_cast<size_t>(local_column + 1);
}

inline int blockSize(const int total, const int parts, const int coordinate) {
    return total / parts + (coordinate < total % parts ? 1 : 0);
}

inline int blockStart(const int total, const int parts, const int coordinate) {
    return coordinate * (total / parts) + std::min(coordinate, total % parts);
}

// Build one tile of the global square mesh.  The tile is represented with the
// same per-element adjacency lists as the serial mesh, while off-tile edges
// reference ghost cells populated by MPI halo exchange.
void buildSquare2D(World& world, const int grid_width, const int coordinates[2],
                   const int process_grid[2]) {
    world.grid_width = grid_width;
    world.local_rows = blockSize(grid_width, process_grid[0], coordinates[0]);
    world.local_columns = blockSize(grid_width, process_grid[1], coordinates[1]);
    world.first_global_row = blockStart(grid_width, process_grid[0], coordinates[0]);
    world.first_global_column =
        blockStart(grid_width, process_grid[1], coordinates[1]);
    world.dynamic_columns = world.local_columns + 2;

    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});

    const size_t local_elements =
        static_cast<size_t>(world.local_rows) * world.local_columns;
    const size_t dynamic_elements =
        static_cast<size_t>(world.local_rows + 2) * world.dynamic_columns;
    world.elements_static.resize(local_elements);
    world.elements_dynamic.resize(dynamic_elements);
    world.elements_dynamic_swap.resize(dynamic_elements);

    for (int local_x = 0; local_x < world.local_rows; ++local_x) {
        const int global_x = world.first_global_row + local_x;
        for (int local_y = 0; local_y < world.local_columns; ++local_y) {
            const int global_y = world.first_global_column + local_y;
            ElementStatic& elem =
                world.elements_static[localElementIndex(world, local_x, local_y)];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            // Preserve the serial traversal order: +x, -x, +y, -y.
            if (global_x + 1 < grid_width) {
                elem.connected_idx[elem.num_connections] =
                    dynamicIndex(world, local_x + 1, local_y);
                elem.connected_flux[elem.num_connections++] = 1.0;
            }
            if (global_x > 0) {
                elem.connected_idx[elem.num_connections] =
                    dynamicIndex(world, local_x - 1, local_y);
                elem.connected_flux[elem.num_connections++] = 1.0;
            }
            if (global_y + 1 < grid_width) {
                elem.connected_idx[elem.num_connections] =
                    dynamicIndex(world, local_x, local_y + 1);
                elem.connected_flux[elem.num_connections++] = 1.0;
            }
            if (global_y > 0) {
                elem.connected_idx[elem.num_connections] =
                    dynamicIndex(world, local_x, local_y - 1);
                elem.connected_flux[elem.num_connections++] = 1.0;
            }

            if ((global_x == 0 || global_x == grid_width - 1) &&
                (global_y == 0 || global_y == grid_width - 1)) {
                elem.material_idx =
                    ((global_x == 0 && global_y == 0) ||
                     (global_x == grid_width - 1 && global_y == grid_width - 1))
                        ? INFLOW_MAT_ID
                        : OUTFLOW_MAT_ID;
            }
        }
    }
}

inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                         const val_t connection_flux,
                         const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

inline void updateElement(World& world, const int local_x, const int local_y) {
    const ElementStatic& elem_static =
        world.elements_static[localElementIndex(world, local_x, local_y)];
    const size_t dynamic_idx = dynamicIndex(world, local_x, local_y);
    const ElementDynamic& elem_dyn = world.elements_dynamic[dynamic_idx];
    const Material& mat = world.materials[elem_static.material_idx];

    val_t total_flux = mat.external_flow;
    for (idx_t connection = 0; connection < elem_static.num_connections; ++connection) {
        const ElementDynamic& neighbor_dyn =
            world.elements_dynamic[elem_static.connected_idx[connection]];
        total_flux += computeFlux(mat, elem_dyn,
                                  elem_static.connected_flux[connection], neighbor_dyn);
    }

    ElementDynamic& elem_write = world.elements_dynamic_swap[dynamic_idx];
    elem_write.current_energy = elem_dyn.current_energy + total_flux;
    elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
}

inline void updateRegion(World& world, const int first_row, const int end_row,
                         const int first_column, const int end_column) {
    for (int local_x = first_row; local_x < end_row; ++local_x) {
        for (int local_y = first_column; local_y < end_column; ++local_y) {
            updateElement(world, local_x, local_y);
        }
    }
}

// Four nonblocking exchanges update just current_energy.  total_flux is never
// read from a neighboring element, so omitting it halves halo traffic.  The
// tile interior is computed while communication progresses.
void runSimulation(World& world, const int n_iters, MPI_Comm cartesian_comm,
                   const MPI_Datatype energy_row_type,
                   const MPI_Datatype energy_column_type) {
    int x_minus = MPI_PROC_NULL;
    int x_plus = MPI_PROC_NULL;
    int y_minus = MPI_PROC_NULL;
    int y_plus = MPI_PROC_NULL;
    MPI_Cart_shift(cartesian_comm, 0, 1, &x_minus, &x_plus);
    MPI_Cart_shift(cartesian_comm, 1, 1, &y_minus, &y_plus);

    for (int iteration = 0; iteration < n_iters; ++iteration) {
        MPI_Request requests[8];
        int request_count = 0;

        // x-direction row halos: receive from -x/+x into top/bottom ghosts.
        if (x_minus != MPI_PROC_NULL) {
            MPI_Irecv(&world.elements_dynamic[dynamicIndex(world, -1, 0)].current_energy,
                      1, energy_row_type, x_minus, 1, cartesian_comm,
                      &requests[request_count++]);
        }
        if (x_plus != MPI_PROC_NULL) {
            MPI_Irecv(
                &world.elements_dynamic[dynamicIndex(world, world.local_rows, 0)]
                     .current_energy,
                1, energy_row_type, x_plus, 0, cartesian_comm,
                &requests[request_count++]);
        }
        if (x_minus != MPI_PROC_NULL) {
            MPI_Isend(&world.elements_dynamic[dynamicIndex(world, 0, 0)].current_energy,
                      1, energy_row_type, x_minus, 0, cartesian_comm,
                      &requests[request_count++]);
        }
        if (x_plus != MPI_PROC_NULL) {
            MPI_Isend(&world.elements_dynamic[
                          dynamicIndex(world, world.local_rows - 1, 0)]
                          .current_energy,
                      1, energy_row_type, x_plus, 1, cartesian_comm,
                      &requests[request_count++]);
        }

        // y-direction column halos: receive from -y/+y into left/right ghosts.
        if (y_minus != MPI_PROC_NULL) {
            MPI_Irecv(&world.elements_dynamic[dynamicIndex(world, 0, -1)].current_energy,
                      1, energy_column_type, y_minus, 3, cartesian_comm,
                      &requests[request_count++]);
        }
        if (y_plus != MPI_PROC_NULL) {
            MPI_Irecv(&world.elements_dynamic[
                          dynamicIndex(world, 0, world.local_columns)]
                          .current_energy,
                      1, energy_column_type, y_plus, 2, cartesian_comm,
                      &requests[request_count++]);
        }
        if (y_minus != MPI_PROC_NULL) {
            MPI_Isend(&world.elements_dynamic[dynamicIndex(world, 0, 0)].current_energy,
                      1, energy_column_type, y_minus, 2, cartesian_comm,
                      &requests[request_count++]);
        }
        if (y_plus != MPI_PROC_NULL) {
            MPI_Isend(&world.elements_dynamic[
                          dynamicIndex(world, 0, world.local_columns - 1)]
                          .current_energy,
                      1, energy_column_type, y_plus, 3, cartesian_comm,
                      &requests[request_count++]);
        }

        if (world.local_rows > 2 && world.local_columns > 2) {
            updateRegion(world, 1, world.local_rows - 1, 1,
                         world.local_columns - 1);
        }
        if (request_count > 0) {
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
        }

        // Complete the one-cell-wide tile boundary after its ghost values are
        // available.  No interior cell is recalculated.
        for (int local_x = 0; local_x < world.local_rows; ++local_x) {
            for (int local_y = 0; local_y < world.local_columns; ++local_y) {
                if (local_x == 0 || local_x == world.local_rows - 1 ||
                    local_y == 0 || local_y == world.local_columns - 1) {
                    updateElement(world, local_x, local_y);
                }
            }
        }
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

bool validateResults(const World& world, const int rank) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    for (int local_x = 0; local_x < world.local_rows; ++local_x) {
        for (int local_y = 0; local_y < world.local_columns; ++local_y) {
            const ElementDynamic& elem =
                world.elements_dynamic[dynamicIndex(world, local_x, local_y)];
            local_energy_sum += elem.current_energy;
            local_flux_sum += elem.total_flux;
            local_energy_max = std::max(local_energy_max, elem.current_energy);
            local_energy_min = std::min(local_energy_min, elem.current_energy);
        }
    }

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0,
               MPI_COMM_WORLD);

    bool valid = true;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        constexpr val_t energy_epsilon = 1e-8;
        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = false;
        } else if (std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = false;
        }
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = false;
        }
        if (valid) {
            printf("  Validation: PASSED\n");
        }
    }

    int valid_int = valid ? 1 : 0;
    MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid_int != 0;
}

// This is the serial hash with global element indices.  Bitwise XOR reduction
// combines disjoint tiles exactly, independently of MPI process count.
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    for (int local_x = 0; local_x < world.local_rows; ++local_x) {
        const size_t global_row =
            static_cast<size_t>(world.first_global_row + local_x) * world.grid_width;
        for (int local_y = 0; local_y < world.local_columns; ++local_y) {
            const ElementDynamic& elem =
                world.elements_dynamic[dynamicIndex(world, local_x, local_y)];
            uint64_t energy_bits;
            uint64_t flux_bits;
            std::memcpy(&energy_bits, &elem.current_energy, sizeof(energy_bits));
            std::memcpy(&flux_bits, &elem.total_flux, sizeof(flux_bits));
            const uint64_t global_index = global_row + world.first_global_column + local_y;
            hash ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
            hash ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

void printUsage(const char* prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Select the largest rectangular process grid that has no empty tile.  This
// lets an arbitrary mpirun size work while avoiding communication to ranks
// that cannot own an element.
int chooseProcessGrid(const int grid_width, const int n_ranks, int process_grid[2]) {
    int best_product = 1;
    int best_delta = 0;
    process_grid[0] = 1;
    process_grid[1] = 1;
    for (int rows = 1; rows <= std::min(grid_width, n_ranks); ++rows) {
        const int columns = std::min(grid_width, n_ranks / rows);
        const int product = rows * columns;
        const int delta = std::abs(rows - columns);
        if (product > best_product ||
            (product == best_product && delta < best_delta)) {
            best_product = product;
            best_delta = delta;
            process_grid[0] = rows;
            process_grid[1] = columns;
        }
    }
    return best_product;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    int grid_width = 512;
    int n_iters = 10;
    int validate = 0;
    int print_results_requested = 0;
    // 0: run, 1: help, 2: argument error
    int action = 0;

    if (rank == 0) {
        for (int argument = 1; argument < argc; ++argument) {
            if (strcmp(argv[argument], "-n") == 0 && argument + 1 < argc) {
                grid_width = atoi(argv[++argument]);
            } else if (strcmp(argv[argument], "-i") == 0 && argument + 1 < argc) {
                n_iters = atoi(argv[++argument]);
            } else if (strcmp(argv[argument], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[argument], "-r") == 0) {
                print_results_requested = 1;
            } else if (strcmp(argv[argument], "-h") == 0) {
                printUsage(argv[0]);
                action = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[argument]);
                printUsage(argv[0]);
                action = 2;
                break;
            }
        }
        if (action == 0 && (grid_width <= 0 || n_iters < 0 || grid_width > 46340)) {
            printf("Grid size must be positive and fit in a 32-bit element count; "
                   "iterations must be non-negative.\n");
            action = 2;
        }
    }

    MPI_Bcast(&action, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (action != 0) {
        MPI_Finalize();
        return action == 1 ? 0 : 1;
    }
    int configuration[4] = {grid_width, n_iters, validate, print_results_requested};
    MPI_Bcast(configuration, 4, MPI_INT, 0, MPI_COMM_WORLD);
    grid_width = configuration[0];
    n_iters = configuration[1];
    validate = configuration[2];
    print_results_requested = configuration[3];

    const int n_elems = grid_width * grid_width;
    int process_grid[2] = {1, 1};
    const int active_ranks = chooseProcessGrid(grid_width, n_ranks, process_grid);
    const bool active = rank < active_ranks;

    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm cartesian_comm = MPI_COMM_NULL;
    int coordinates[2] = {0, 0};
    if (active) {
        MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &active_comm);
        const int periodic[2] = {0, 0};
        MPI_Cart_create(active_comm, 2, process_grid, periodic, 0, &cartesian_comm);
        int cartesian_rank = 0;
        MPI_Comm_rank(cartesian_comm, &cartesian_rank);
        MPI_Cart_coords(cartesian_comm, cartesian_rank, 2, coordinates);
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, rank, &active_comm);
    }

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", grid_width, grid_width, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate != 0 ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    World world;
    if (active) {
        buildSquare2D(world, grid_width, coordinates, process_grid);
    }

    const uint64_t local_static_mem =
        static_cast<uint64_t>(world.elements_static.size()) * sizeof(ElementStatic);
    const uint64_t local_dynamic_mem =
        static_cast<uint64_t>(world.elements_dynamic.size() +
                              world.elements_dynamic_swap.size()) *
        sizeof(ElementDynamic);
    uint64_t static_mem = 0;
    uint64_t dynamic_mem = 0;
    MPI_Reduce(&local_static_mem, &static_mem, 1, MPI_UINT64_T, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_dynamic_mem, &dynamic_mem, 1, MPI_UINT64_T, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        const uint64_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0), static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    MPI_Datatype energy_row_type = MPI_DATATYPE_NULL;
    MPI_Datatype energy_column_type = MPI_DATATYPE_NULL;
    if (active) {
        MPI_Type_create_hvector(world.local_columns, 1, sizeof(ElementDynamic), MPI_DOUBLE,
                                &energy_row_type);
        MPI_Type_create_hvector(world.local_rows, 1,
                                static_cast<MPI_Aint>(world.dynamic_columns) *
                                    sizeof(ElementDynamic),
                                MPI_DOUBLE, &energy_column_type);
        MPI_Type_commit(&energy_row_type);
        MPI_Type_commit(&energy_column_type);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();
    if (active) {
        runSimulation(world, n_iters, cartesian_comm, energy_row_type,
                      energy_column_type);
    }
    const double local_elapsed = MPI_Wtime() - start_time;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (active) {
        MPI_Type_free(&energy_row_type);
        MPI_Type_free(&energy_column_type);
        MPI_Comm_free(&cartesian_comm);
        MPI_Comm_free(&active_comm);
    }

    const uint64_t local_hash = computeHash(world);
    uint64_t result_hash = 0;
    MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const long duration_ms = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (n_measured_iters * static_cast<double>(n_elems)) /
            (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016" PRIX64 "\n", result_hash);
        printf("\n");
    }

    // Gather only when requested.  Tiles are packed locally and rank zero
    // reconstructs the original global row-major vector for print_results.
    if (print_results_requested != 0) {
        const int local_count = world.local_rows * world.local_columns;
        std::vector<double> local_energy(local_count);
        for (int local_x = 0; local_x < world.local_rows; ++local_x) {
            for (int local_y = 0; local_y < world.local_columns; ++local_y) {
                local_energy[localElementIndex(world, local_x, local_y)] =
                    world.elements_dynamic[dynamicIndex(world, local_x, local_y)]
                        .current_energy;
            }
        }
        const int local_metadata[4] = {world.first_global_row,
                                       world.first_global_column, world.local_rows,
                                       world.local_columns};
        std::vector<int> metadata;
        std::vector<int> receive_counts;
        std::vector<int> displacements;
        std::vector<double> gathered_energy;
        if (rank == 0) {
            metadata.resize(static_cast<size_t>(n_ranks) * 4);
            receive_counts.resize(n_ranks);
            displacements.resize(n_ranks);
        }
        MPI_Gather(local_metadata, 4, MPI_INT, metadata.data(), 4, MPI_INT, 0,
                   MPI_COMM_WORLD);
        if (rank == 0) {
            int total_count = 0;
            for (int peer = 0; peer < n_ranks; ++peer) {
                const int peer_rows = metadata[4 * peer + 2];
                const int peer_columns = metadata[4 * peer + 3];
                receive_counts[peer] = peer_rows * peer_columns;
                displacements[peer] = total_count;
                total_count += receive_counts[peer];
            }
            gathered_energy.resize(total_count);
        }
        MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE, gathered_energy.data(),
                    receive_counts.data(), displacements.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<double> energy_data(n_elems);
            for (int peer = 0; peer < n_ranks; ++peer) {
                const int first_row = metadata[4 * peer];
                const int first_column = metadata[4 * peer + 1];
                const int peer_rows = metadata[4 * peer + 2];
                const int peer_columns = metadata[4 * peer + 3];
                for (int local_x = 0; local_x < peer_rows; ++local_x) {
                    for (int local_y = 0; local_y < peer_columns; ++local_y) {
                        energy_data[static_cast<size_t>(first_row + local_x) * grid_width +
                                    first_column + local_y] =
                            gathered_energy[displacements[peer] +
                                            local_x * peer_columns + local_y];
                    }
                }
            }
            print_results(energy_data, "ElementEnergy");
        }
    }

    const bool valid = validate == 0 || validateResults(world, rank);
    MPI_Finalize();
    return valid ? 0 : 1;
}
