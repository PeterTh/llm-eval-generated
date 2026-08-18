#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>
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

static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t),
              "ElementDynamic must be two contiguous doubles for MPI transfers");

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

constexpr std::array<Material, 3> MATERIALS{{
    {0.8, 0.0},
    {0.8, 0.5},
    {0.8, -0.5},
}};

struct Partition {
    int global_extent;
    int row_offset;
    int column_offset;
    int local_rows;
    int local_columns;
};

struct DistributedWorld {
    Partition partition;
    int north_rank = MPI_PROC_NULL;
    int south_rank = MPI_PROC_NULL;
    int west_rank = MPI_PROC_NULL;
    int east_rank = MPI_PROC_NULL;
    MPI_Datatype row_energy_type = MPI_DATATYPE_NULL;
    MPI_Datatype column_energy_type = MPI_DATATYPE_NULL;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    std::vector<val_t> north_energy;
    std::vector<val_t> south_energy;
    std::vector<val_t> west_energy;
    std::vector<val_t> east_energy;
};

struct ProcessGrid {
    int rows;
    int columns;
    int active_ranks;
};

// Pick the largest non-empty Cartesian process grid that fits the mesh.  Ranks
// outside this grid remain in MPI_COMM_WORLD only and do no mesh work.
ProcessGrid chooseProcessGrid(const int extent, const int world_size) {
    const int rank_limit = static_cast<int>(
        std::min<int64_t>(world_size, static_cast<int64_t>(extent) * extent));

    ProcessGrid best{1, 1, 1};
    const int max_rows = std::min(extent, rank_limit);
    for (int rows = 1; rows <= max_rows; ++rows) {
        const int columns = std::min(extent, rank_limit / rows);
        const int active_ranks = rows * columns;
        const int best_aspect = std::abs(best.rows - best.columns);
        const int aspect = std::abs(rows - columns);
        if (active_ranks > best.active_ranks ||
            (active_ranks == best.active_ranks && aspect < best_aspect)) {
            best = {rows, columns, active_ranks};
        }
    }
    return best;
}

int blockOffset(const int coordinate, const int extent, const int partitions) {
    return static_cast<int>((static_cast<int64_t>(coordinate) * extent) / partitions);
}

inline idx_t materialForCoordinates(const int global_row, const int global_column,
                                    const int extent) {
    const int last = extent - 1;
    if ((global_row == 0 && global_column == 0) ||
        (global_row == last && global_column == last)) {
        return INFLOW_MAT_ID;
    }
    if ((global_row == 0 && global_column == last) ||
        (global_row == last && global_column == 0)) {
        return OUTFLOW_MAT_ID;
    }
    return DEFAULT_MAT_ID;
}

inline val_t computeFlux(const Material& material, const val_t this_energy,
                         const val_t connection_flux, const val_t other_energy) {
    return (other_energy - this_energy) * material.transfer_coeff * connection_flux * 0.25;
}

void buildSquare2D(DistributedWorld& world, const int extent, MPI_Comm cart_comm) {
    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);
    int coordinates[2];
    MPI_Cart_coords(cart_comm, cart_rank, 2, coordinates);

    int dimensions[2];
    int periods[2];
    int cart_coordinates[2];
    MPI_Cart_get(cart_comm, 2, dimensions, periods, cart_coordinates);

    const int row_offset = blockOffset(coordinates[0], extent, dimensions[0]);
    const int row_end = blockOffset(coordinates[0] + 1, extent, dimensions[0]);
    const int column_offset = blockOffset(coordinates[1], extent, dimensions[1]);
    const int column_end = blockOffset(coordinates[1] + 1, extent, dimensions[1]);
    world.partition = {extent, row_offset, column_offset, row_end - row_offset,
                       column_end - column_offset};

    MPI_Cart_shift(cart_comm, 0, 1, &world.north_rank, &world.south_rank);
    MPI_Cart_shift(cart_comm, 1, 1, &world.west_rank, &world.east_rank);

    const size_t local_elements = static_cast<size_t>(world.partition.local_rows) *
                                  world.partition.local_columns;
    world.elements_dynamic.assign(local_elements, ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.assign(local_elements, ElementDynamic{0.0, 0.0});
    world.north_energy.resize(world.partition.local_columns);
    world.south_energy.resize(world.partition.local_columns);
    world.west_energy.resize(world.partition.local_rows);
    world.east_energy.resize(world.partition.local_rows);

    // The state is an interleaved pair of doubles.  These datatypes send only
    // current_energy, avoiding communication of the locally accumulated flux.
    MPI_Type_vector(world.partition.local_columns, 1, 2, MPI_DOUBLE, &world.row_energy_type);
    MPI_Type_commit(&world.row_energy_type);
    MPI_Type_vector(world.partition.local_rows, 1, 2 * world.partition.local_columns, MPI_DOUBLE,
                    &world.column_energy_type);
    MPI_Type_commit(&world.column_energy_type);
}

inline void updateBoundaryElement(DistributedWorld& world, const int local_row,
                                  const int local_column) {
    const Partition& partition = world.partition;
    const size_t index = static_cast<size_t>(local_row) * partition.local_columns + local_column;
    const ElementDynamic& current = world.elements_dynamic[index];
    const int global_row = partition.row_offset + local_row;
    const int global_column = partition.column_offset + local_column;
    const Material& material = MATERIALS[materialForCoordinates(
        global_row, global_column, partition.global_extent)];
    const val_t this_energy = current.current_energy;
    val_t total_flux = material.external_flow;

    // Keep the original connectivity order: down, up, right, left.
    if (local_row + 1 < partition.local_rows) {
        total_flux += computeFlux(material, this_energy, 1.0,
                                  world.elements_dynamic[index + partition.local_columns].current_energy);
    } else if (world.south_rank != MPI_PROC_NULL) {
        total_flux += computeFlux(material, this_energy, 1.0, world.south_energy[local_column]);
    }
    if (local_row > 0) {
        total_flux += computeFlux(material, this_energy, 1.0,
                                  world.elements_dynamic[index - partition.local_columns].current_energy);
    } else if (world.north_rank != MPI_PROC_NULL) {
        total_flux += computeFlux(material, this_energy, 1.0, world.north_energy[local_column]);
    }
    if (local_column + 1 < partition.local_columns) {
        total_flux += computeFlux(material, this_energy, 1.0,
                                  world.elements_dynamic[index + 1].current_energy);
    } else if (world.east_rank != MPI_PROC_NULL) {
        total_flux += computeFlux(material, this_energy, 1.0, world.east_energy[local_row]);
    }
    if (local_column > 0) {
        total_flux += computeFlux(material, this_energy, 1.0,
                                  world.elements_dynamic[index - 1].current_energy);
    } else if (world.west_rank != MPI_PROC_NULL) {
        total_flux += computeFlux(material, this_energy, 1.0, world.west_energy[local_row]);
    }

    ElementDynamic& next = world.elements_dynamic_swap[index];
    next.current_energy = this_energy + total_flux;
    next.total_flux = current.total_flux + std::abs(total_flux);
}

void updateInterior(DistributedWorld& world) {
    const Partition& partition = world.partition;
    const int columns = partition.local_columns;
    const Material& material = MATERIALS[DEFAULT_MAT_ID];

    // These elements have all four neighbours locally.  Updating them while
    // halo transfers are in flight removes communication from the critical path.
    for (int local_row = 1; local_row + 1 < partition.local_rows; ++local_row) {
        const size_t row_base = static_cast<size_t>(local_row) * columns;
        for (int local_column = 1; local_column + 1 < columns; ++local_column) {
            const size_t index = row_base + local_column;
            const ElementDynamic& current = world.elements_dynamic[index];
            const val_t this_energy = current.current_energy;
            val_t total_flux = 0.0;
            total_flux += computeFlux(material, this_energy, 1.0,
                                      world.elements_dynamic[index + columns].current_energy);
            total_flux += computeFlux(material, this_energy, 1.0,
                                      world.elements_dynamic[index - columns].current_energy);
            total_flux += computeFlux(material, this_energy, 1.0,
                                      world.elements_dynamic[index + 1].current_energy);
            total_flux += computeFlux(material, this_energy, 1.0,
                                      world.elements_dynamic[index - 1].current_energy);

            ElementDynamic& next = world.elements_dynamic_swap[index];
            next.current_energy = this_energy + total_flux;
            next.total_flux = current.total_flux + std::abs(total_flux);
        }
    }
}

void runSimulation(DistributedWorld& world, const int n_iters, MPI_Comm cart_comm) {
    const Partition& partition = world.partition;
    const int columns = partition.local_columns;
    const size_t last_row = static_cast<size_t>(partition.local_rows - 1) * columns;

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];

        MPI_Irecv(world.north_energy.data(), columns, MPI_DOUBLE, world.north_rank, 101,
                  cart_comm, &requests[0]);
        MPI_Irecv(world.south_energy.data(), columns, MPI_DOUBLE, world.south_rank, 100,
                  cart_comm, &requests[1]);
        MPI_Irecv(world.west_energy.data(), partition.local_rows, MPI_DOUBLE, world.west_rank, 103,
                  cart_comm, &requests[2]);
        MPI_Irecv(world.east_energy.data(), partition.local_rows, MPI_DOUBLE, world.east_rank, 102,
                  cart_comm, &requests[3]);

        MPI_Isend(&world.elements_dynamic[0].current_energy, 1, world.row_energy_type,
                  world.north_rank, 100, cart_comm, &requests[4]);
        MPI_Isend(&world.elements_dynamic[last_row].current_energy, 1, world.row_energy_type,
                  world.south_rank, 101, cart_comm, &requests[5]);
        MPI_Isend(&world.elements_dynamic[0].current_energy, 1, world.column_energy_type,
                  world.west_rank, 102, cart_comm, &requests[6]);
        MPI_Isend(&world.elements_dynamic[columns - 1].current_energy, 1,
                  world.column_energy_type, world.east_rank, 103, cart_comm, &requests[7]);

        updateInterior(world);
        MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);

        // The local perimeter includes both MPI-facing and physical boundaries.
        // It is small relative to each subdomain and is evaluated after halos land.
        for (int local_row = 0; local_row < partition.local_rows; ++local_row) {
            for (int local_column = 0; local_column < columns; ++local_column) {
                if (local_row == 0 || local_row + 1 == partition.local_rows ||
                    local_column == 0 || local_column + 1 == columns) {
                    updateBoundaryElement(world, local_row, local_column);
                }
            }
        }

        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

void destroyWorld(DistributedWorld& world) {
    if (world.row_energy_type != MPI_DATATYPE_NULL) {
        MPI_Type_free(&world.row_energy_type);
    }
    if (world.column_energy_type != MPI_DATATYPE_NULL) {
        MPI_Type_free(&world.column_energy_type);
    }
}

bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    constexpr val_t energy_epsilon = 1e-8;
    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    if (!std::isfinite(flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const DistributedWorld& world) {
    const Partition& partition = world.partition;
    uint64_t hash = 0;
    for (int local_row = 0; local_row < partition.local_rows; ++local_row) {
        const size_t local_base = static_cast<size_t>(local_row) * partition.local_columns;
        const uint64_t global_base = static_cast<uint64_t>(partition.row_offset + local_row) *
                                     partition.global_extent + partition.column_offset;
        for (int local_column = 0; local_column < partition.local_columns; ++local_column) {
            const ElementDynamic& element = world.elements_dynamic[local_base + local_column];
            uint64_t energy_bits;
            uint64_t flux_bits;
            std::memcpy(&energy_bits, &element.current_energy, sizeof(energy_bits));
            std::memcpy(&flux_bits, &element.total_flux, sizeof(flux_bits));
            const uint64_t global_index = global_base + local_column;
            hash ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
            hash ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

// Gather optional diagnostic output in global row-major order.  This path is
// deliberately outside the timed simulation and runs only for -r or -v.
std::vector<ElementDynamic> gatherElements(const DistributedWorld& world, MPI_Comm cart_comm,
                                           const int cart_rank, const int cart_size) {
    const Partition& partition = world.partition;
    const int local_count = partition.local_rows * partition.local_columns;
    const int metadata[4] = {partition.row_offset, partition.column_offset,
                             partition.local_rows, partition.local_columns};

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<int> all_metadata;
    if (cart_rank == 0) {
        counts.resize(cart_size);
        displacements.resize(cart_size);
        all_metadata.resize(static_cast<size_t>(cart_size) * 4);
    }
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, cart_comm);
    MPI_Gather(metadata, 4, MPI_INT, all_metadata.data(), 4, MPI_INT, 0, cart_comm);

    std::vector<ElementDynamic> packed;
    if (cart_rank == 0) {
        int total_count = 0;
        for (int rank = 0; rank < cart_size; ++rank) {
            displacements[rank] = total_count;
            total_count += counts[rank];
        }
        packed.resize(total_count);
    }

    MPI_Datatype element_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &element_type);
    MPI_Type_commit(&element_type);
    MPI_Gatherv(world.elements_dynamic.data(), local_count, element_type, packed.data(), counts.data(),
                displacements.data(), element_type, 0, cart_comm);
    MPI_Type_free(&element_type);

    if (cart_rank != 0) {
        return {};
    }

    const size_t global_count = static_cast<size_t>(partition.global_extent) * partition.global_extent;
    std::vector<ElementDynamic> global_elements(global_count);
    for (int rank = 0; rank < cart_size; ++rank) {
        const int row_offset = all_metadata[4 * rank];
        const int column_offset = all_metadata[4 * rank + 1];
        const int rows = all_metadata[4 * rank + 2];
        const int columns = all_metadata[4 * rank + 3];
        const ElementDynamic* source = packed.data() + displacements[rank];
        for (int local_row = 0; local_row < rows; ++local_row) {
            ElementDynamic* destination = global_elements.data() +
                static_cast<size_t>(row_offset + local_row) * partition.global_extent + column_offset;
            std::copy_n(source + static_cast<size_t>(local_row) * columns, columns, destination);
        }
    }
    return global_elements;
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

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n_elems_root = 512;
    int n_iters = 10;
    int validate = 0;
    int print_results_enabled = 0;
    int parse_status = 0;  // 0: run, 1: help, 2: invalid input

    if (world_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                print_results_enabled = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parse_status = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parse_status = 2;
                break;
            }
        }
        if (parse_status == 0 && (n_elems_root <= 0 || n_iters < 0)) {
            printf("Grid size must be positive and iterations must be non-negative.\n");
            parse_status = 2;
        }
    }

    int configuration[5] = {n_elems_root, n_iters, validate, print_results_enabled, parse_status};
    MPI_Bcast(configuration, 5, MPI_INT, 0, MPI_COMM_WORLD);
    n_elems_root = configuration[0];
    n_iters = configuration[1];
    validate = configuration[2];
    print_results_enabled = configuration[3];
    parse_status = configuration[4];
    if (parse_status != 0) {
        MPI_Finalize();
        return parse_status == 1 ? 0 : 1;
    }

    const ProcessGrid process_grid = chooseProcessGrid(n_elems_root, world_size);
    MPI_Comm active_comm = MPI_COMM_NULL;
    const int active = world_rank < process_grid.active_ranks;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, world_rank, &active_comm);

    int exit_code = 0;
    if (active) {
        const int dimensions[2] = {process_grid.rows, process_grid.columns};
        const int periods[2] = {0, 0};
        MPI_Comm cart_comm = MPI_COMM_NULL;
        MPI_Cart_create(active_comm, 2, dimensions, periods, 0, &cart_comm);

        int cart_rank;
        int cart_size;
        MPI_Comm_rank(cart_comm, &cart_rank);
        MPI_Comm_size(cart_comm, &cart_size);

        DistributedWorld world;
        buildSquare2D(world, n_elems_root, cart_comm);

        const uint64_t n_elems = static_cast<uint64_t>(n_elems_root) * n_elems_root;
        if (world_rank == 0) {
            const size_t global_dynamic_memory = static_cast<size_t>(n_elems) *
                                                 sizeof(ElementDynamic) * 2;
            printf("Unstructured Mesh Energy Transfer Benchmark\n");
            printf("============================================\n");
            printf("Grid size: %d x %d = %" PRIu64 " elements\n",
                   n_elems_root, n_elems_root, n_elems);
            printf("Iterations: %d\n", n_iters);
            printf("MPI ranks: %d (process grid: %d x %d)\n", cart_size,
                   process_grid.rows, process_grid.columns);
            printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
            printf("Building distributed unstructured mesh...\n");
            printf("Distributed dynamic state: %.2f MB global\n\n",
                   global_dynamic_memory / (1024.0 * 1024.0));
            printf("Running simulation...\n");
        }

        MPI_Barrier(cart_comm);
        const double start = MPI_Wtime();
        runSimulation(world, n_iters, cart_comm);
        const double local_elapsed = MPI_Wtime() - start;
        double elapsed = 0.0;
        MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

        const uint64_t local_hash = computeHash(world);
        uint64_t global_hash = 0;
        MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, cart_comm);

        if (world_rank == 0) {
            const int n_measured_iters = std::max(n_iters - 1, 1);
            const double time_per_iter_ms = elapsed * 1000.0 / n_measured_iters;
            const double giga_elems_per_sec = elapsed > 0.0
                ? (static_cast<double>(n_measured_iters) * static_cast<double>(n_elems)) / elapsed / 1e9
                : 0.0;
            const double gflops = giga_elems_per_sec * 22.0;
            printf("Computation time: %.3f ms\n", elapsed * 1000.0);
            printf("Performance:\n");
            printf("  Time per iteration: %.4f ms\n", time_per_iter_ms);
            printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
            printf("  Performance: %.4f GFLOPS\n", gflops);
            printf("  Result hash: %016" PRIX64 "\n\n", global_hash);
        }

        if (validate || print_results_enabled) {
            std::vector<ElementDynamic> global_elements =
                gatherElements(world, cart_comm, cart_rank, cart_size);
            if (world_rank == 0) {
                if (print_results_enabled) {
                    std::vector<double> energy_data;
                    energy_data.reserve(global_elements.size());
                    for (const ElementDynamic& element : global_elements) {
                        energy_data.push_back(element.current_energy);
                    }
                    print_results(energy_data, "ElementEnergy");
                }
                if (validate && !validateResults(global_elements)) {
                    exit_code = 1;
                }
            }
        }

        destroyWorld(world);
        MPI_Comm_free(&cart_comm);
        MPI_Comm_free(&active_comm);
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
