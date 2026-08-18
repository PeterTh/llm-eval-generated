#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element
struct ElementStatic {
    idx_t state_idx;
    idx_t global_idx;
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state
struct World {
    MPI_Comm cart_comm = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int n_elems_root = 0;
    int dims[2] = {1, 1};
    int coords[2] = {0, 0};
    int row_begin = 0;
    int col_begin = 0;
    int local_rows = 0;
    int local_cols = 0;
    int up = MPI_PROC_NULL;
    int down = MPI_PROC_NULL;
    int left = MPI_PROC_NULL;
    int right = MPI_PROC_NULL;

    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    std::vector<size_t> interior_elements;
    std::vector<size_t> boundary_elements;
    MPI_Datatype row_energy_type = MPI_DATATYPE_NULL;
    MPI_Datatype column_energy_type = MPI_DATATYPE_NULL;
};

static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t));

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

inline int blockBegin(const int extent, const int coordinate, const int parts) {
    return static_cast<int>((static_cast<int64_t>(extent) * coordinate) / parts);
}

// Build the local block of a 2D square grid as an unstructured mesh.  The
// one-cell halo is included in the dynamic arrays, while static mesh data is
// kept only for owned elements.
void buildSquare2D(World& world, const int n_elems_root, MPI_Comm cart_comm) {
    world.cart_comm = cart_comm;
    world.n_elems_root = n_elems_root;
    MPI_Comm_rank(cart_comm, &world.rank);
    MPI_Comm_size(cart_comm, &world.size);
    int periods[2] = {0, 0};
    MPI_Cart_get(cart_comm, 2, world.dims, periods, world.coords);

    world.row_begin = blockBegin(n_elems_root, world.coords[0], world.dims[0]);
    const int row_end = blockBegin(n_elems_root, world.coords[0] + 1, world.dims[0]);
    world.col_begin = blockBegin(n_elems_root, world.coords[1], world.dims[1]);
    const int col_end = blockBegin(n_elems_root, world.coords[1] + 1, world.dims[1]);
    world.local_rows = row_end - world.row_begin;
    world.local_cols = col_end - world.col_begin;

    MPI_Cart_shift(cart_comm, 0, 1, &world.up, &world.down);
    MPI_Cart_shift(cart_comm, 1, 1, &world.left, &world.right);

    world.materials = {
        Material{0.8, 0.0},    // Default material
        Material{0.8, 0.5},    // Inflow material
        Material{0.8, -0.5}    // Outflow material
    };

    const int padded_cols = world.local_cols + 2;
    const size_t local_count = static_cast<size_t>(world.local_rows) * world.local_cols;
    const size_t padded_count = static_cast<size_t>(world.local_rows + 2) * padded_cols;
    world.elements_static.resize(local_count);
    world.elements_dynamic.assign(padded_count, ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.assign(padded_count, ElementDynamic{0.0, 0.0});

    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    const int last = n_elems_root - 1;
    for (int local_x = 0; local_x < world.local_rows; ++local_x) {
        const int global_x = world.row_begin + local_x;
        for (int local_y = 0; local_y < world.local_cols; ++local_y) {
            const int global_y = world.col_begin + local_y;
            const size_t local_idx = static_cast<size_t>(local_x) * world.local_cols + local_y;
            ElementStatic& elem = world.elements_static[local_idx];
            elem.state_idx = static_cast<idx_t>((local_x + 1) * padded_cols + local_y + 1);
            elem.global_idx = static_cast<idx_t>(global_x) * n_elems_root + global_y;
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            for (const auto& offset : offsets) {
                const int neighbor_x = global_x + offset[0];
                const int neighbor_y = global_y + offset[1];
                if (neighbor_x >= 0 && neighbor_x < n_elems_root &&
                    neighbor_y >= 0 && neighbor_y < n_elems_root) {
                    // A neighbor outside this rank's block maps into the
                    // corresponding halo row or column.
                    const int halo_x = neighbor_x - world.row_begin + 1;
                    const int halo_y = neighbor_y - world.col_begin + 1;
                    elem.connected_idx[elem.num_connections] =
                        static_cast<idx_t>(halo_x * padded_cols + halo_y);
                    elem.connected_flux[elem.num_connections] = 1.0;
                    ++elem.num_connections;
                }
            }

            if (global_x == 0 && global_y == 0) {
                elem.material_idx = INFLOW_MAT_ID;
            } else if (global_x == 0 && global_y == last) {
                elem.material_idx = OUTFLOW_MAT_ID;
            } else if (global_x == last && global_y == 0) {
                elem.material_idx = OUTFLOW_MAT_ID;
            } else if (global_x == last && global_y == last) {
                elem.material_idx = INFLOW_MAT_ID;
            }

            const bool needs_halo =
                (local_x == 0 && world.up != MPI_PROC_NULL) ||
                (local_x == world.local_rows - 1 && world.down != MPI_PROC_NULL) ||
                (local_y == 0 && world.left != MPI_PROC_NULL) ||
                (local_y == world.local_cols - 1 && world.right != MPI_PROC_NULL);
            (needs_halo ? world.boundary_elements : world.interior_elements).push_back(local_idx);
        }
    }

    // ElementDynamic is an array of two doubles, so these datatypes transfer
    // just current_energy without packing an entire face on every iteration.
    MPI_Type_vector(world.local_cols, 1, 2, MPI_DOUBLE, &world.row_energy_type);
    MPI_Type_commit(&world.row_energy_type);
    MPI_Type_vector(world.local_rows, 1, 2 * padded_cols, MPI_DOUBLE,
                    &world.column_energy_type);
    MPI_Type_commit(&world.column_energy_type);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

void startHaloExchange(World& world, std::array<MPI_Request, 8>& requests) {
    const int padded_cols = world.local_cols + 2;
    const size_t first_active = static_cast<size_t>(padded_cols + 1);
    const size_t first_active_row = first_active;
    const size_t last_active_row = static_cast<size_t>(world.local_rows) * padded_cols + 1;
    const size_t first_active_col = first_active;
    const size_t last_active_col = static_cast<size_t>(padded_cols) + world.local_cols;
    const size_t first_halo_row = 1;
    const size_t last_halo_row = static_cast<size_t>(world.local_rows + 1) * padded_cols + 1;
    const size_t first_halo_col = padded_cols;
    const size_t last_halo_col = static_cast<size_t>(padded_cols) + world.local_cols + 1;

    int request = 0;
    MPI_Irecv(&world.elements_dynamic[first_halo_row].current_energy, 1,
              world.row_energy_type, world.up, 0, world.cart_comm, &requests[request++]);
    MPI_Irecv(&world.elements_dynamic[last_halo_row].current_energy, 1,
              world.row_energy_type, world.down, 1, world.cart_comm, &requests[request++]);
    MPI_Irecv(&world.elements_dynamic[first_halo_col].current_energy, 1,
              world.column_energy_type, world.left, 2, world.cart_comm, &requests[request++]);
    MPI_Irecv(&world.elements_dynamic[last_halo_col].current_energy, 1,
              world.column_energy_type, world.right, 3, world.cart_comm, &requests[request++]);

    MPI_Isend(&world.elements_dynamic[last_active_row].current_energy, 1,
              world.row_energy_type, world.down, 0, world.cart_comm, &requests[request++]);
    MPI_Isend(&world.elements_dynamic[first_active_row].current_energy, 1,
              world.row_energy_type, world.up, 1, world.cart_comm, &requests[request++]);
    MPI_Isend(&world.elements_dynamic[last_active_col].current_energy, 1,
              world.column_energy_type, world.right, 2, world.cart_comm, &requests[request++]);
    MPI_Isend(&world.elements_dynamic[first_active_col].current_energy, 1,
              world.column_energy_type, world.left, 3, world.cart_comm, &requests[request++]);
}

inline void updateElement(World& world, const size_t local_idx) {
    const ElementStatic& elem_static = world.elements_static[local_idx];
    const ElementDynamic& elem_dyn = world.elements_dynamic[elem_static.state_idx];
    const Material& mat = world.materials[elem_static.material_idx];

    val_t total_flux = mat.external_flow;
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const ElementDynamic& neighbor_dyn =
            world.elements_dynamic[elem_static.connected_idx[j]];
        total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
    }

    ElementDynamic& elem_write = world.elements_dynamic_swap[elem_static.state_idx];
    elem_write.current_energy = elem_dyn.current_energy + total_flux;
    elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
}

// Run simulation for n_iters iterations.  Only the four one-cell faces are
// communicated; interior elements are updated while those messages progress.
void runSimulation(World& world, const int n_iters) {
    for (int iter = 0; iter < n_iters; ++iter) {
        std::array<MPI_Request, 8> requests{};
        startHaloExchange(world, requests);

        for (const size_t local_idx : world.interior_elements) {
            updateElement(world, local_idx);
        }

        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        for (const size_t local_idx : world.boundary_elements) {
            updateElement(world, local_idx);
        }

        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

struct ValidationStats {
    val_t energy_sum;
    val_t flux_sum;
    val_t energy_min;
    val_t energy_max;
};

ValidationStats localValidationStats(const World& world) {
    ValidationStats stats{0.0, 0.0, std::numeric_limits<val_t>::max(),
                          std::numeric_limits<val_t>::lowest()};
    for (const ElementStatic& elem_static : world.elements_static) {
        const ElementDynamic& elem = world.elements_dynamic[elem_static.state_idx];
        stats.energy_sum += elem.current_energy;
        stats.flux_sum += elem.total_flux;
        stats.energy_min = std::min(elem.current_energy, stats.energy_min);
        stats.energy_max = std::max(elem.current_energy, stats.energy_max);
    }
    return stats;
}

// Validate simulation results, reducing only the requested diagnostics.
bool validateResults(const World& world) {
    const ValidationStats local = localValidationStats(world);
    ValidationStats global{};
    MPI_Reduce(&local.energy_sum, &global.energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               world.cart_comm);
    MPI_Reduce(&local.flux_sum, &global.flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               world.cart_comm);
    MPI_Reduce(&local.energy_min, &global.energy_min, 1, MPI_DOUBLE, MPI_MIN, 0,
               world.cart_comm);
    MPI_Reduce(&local.energy_max, &global.energy_max, 1, MPI_DOUBLE, MPI_MAX, 0,
               world.cart_comm);

    bool valid = true;
    if (world.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global.energy_sum);
        printf("  Flux sum: %.2f\n", global.flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global.energy_min, global.energy_max);

        constexpr val_t energy_epsilon = 1e-8;
        if (!std::isfinite(global.energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = false;
        }
        if (std::abs(global.energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(global.flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = false;
        }
        if (!std::isfinite(global.energy_max) || !std::isfinite(global.energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = false;
        }
        if (valid) {
            printf("  Validation: PASSED\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, world.cart_comm);
    return valid;
}

// Compute the same index-dependent hash as the original global traversal.
uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    for (const ElementStatic& elem_static : world.elements_static) {
        const ElementDynamic& elem = world.elements_dynamic[elem_static.state_idx];
        uint64_t energy_bits;
        uint64_t flux_bits;
        std::memcpy(&energy_bits, &elem.current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elem.total_flux, sizeof(flux_bits));
        local_hash ^= (energy_bits + elem_static.global_idx) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (flux_bits + elem_static.global_idx) * 0xbf58476d1ce4e5b9ULL;
    }

    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, world.cart_comm);
    return global_hash;
}

void printGlobalEnergyResults(const World& world) {
    const size_t local_count = world.elements_static.size();
    if (local_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        MPI_Abort(world.cart_comm, 2);
    }

    std::vector<double> local_energy(local_count);
    for (size_t local_idx = 0; local_idx < local_count; ++local_idx) {
        local_energy[local_idx] =
            world.elements_dynamic[world.elements_static[local_idx].state_idx].current_energy;
    }

    std::vector<int> receive_counts;
    std::vector<int> displacements;
    std::vector<double> packed_energy;
    if (world.rank == 0) {
        receive_counts.resize(world.size);
        displacements.resize(world.size);
        int packed_count = 0;
        for (int rank = 0; rank < world.size; ++rank) {
            int coords[2];
            MPI_Cart_coords(world.cart_comm, rank, 2, coords);
            const int row_begin = blockBegin(world.n_elems_root, coords[0], world.dims[0]);
            const int row_end = blockBegin(world.n_elems_root, coords[0] + 1, world.dims[0]);
            const int col_begin = blockBegin(world.n_elems_root, coords[1], world.dims[1]);
            const int col_end = blockBegin(world.n_elems_root, coords[1] + 1, world.dims[1]);
            const int count = (row_end - row_begin) * (col_end - col_begin);
            if (count > std::numeric_limits<int>::max() - packed_count) {
                MPI_Abort(world.cart_comm, 3);
            }
            receive_counts[rank] = count;
            displacements[rank] = packed_count;
            packed_count += count;
        }
        packed_energy.resize(static_cast<size_t>(packed_count));
    }

    MPI_Gatherv(local_energy.data(), static_cast<int>(local_count), MPI_DOUBLE,
                world.rank == 0 ? packed_energy.data() : nullptr,
                world.rank == 0 ? receive_counts.data() : nullptr,
                world.rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, world.cart_comm);

    if (world.rank == 0) {
        const size_t global_count = static_cast<size_t>(world.n_elems_root) * world.n_elems_root;
        std::vector<double> global_energy(global_count);
        for (int rank = 0; rank < world.size; ++rank) {
            int coords[2];
            MPI_Cart_coords(world.cart_comm, rank, 2, coords);
            const int row_begin = blockBegin(world.n_elems_root, coords[0], world.dims[0]);
            const int row_end = blockBegin(world.n_elems_root, coords[0] + 1, world.dims[0]);
            const int col_begin = blockBegin(world.n_elems_root, coords[1], world.dims[1]);
            const int col_end = blockBegin(world.n_elems_root, coords[1] + 1, world.dims[1]);
            const int cols = col_end - col_begin;
            size_t packed_idx = static_cast<size_t>(displacements[rank]);
            for (int local_x = 0; local_x < row_end - row_begin; ++local_x) {
                for (int local_y = 0; local_y < cols; ++local_y) {
                    const size_t global_idx =
                        static_cast<size_t>(row_begin + local_x) * world.n_elems_root +
                        col_begin + local_y;
                    global_energy[global_idx] = packed_energy[packed_idx++];
                }
            }
        }
        print_results(global_energy, "ElementEnergy");
    }
}

void destroyWorld(World& world) {
    if (world.row_energy_type != MPI_DATATYPE_NULL) {
        MPI_Type_free(&world.row_energy_type);
    }
    if (world.column_energy_type != MPI_DATATYPE_NULL) {
        MPI_Type_free(&world.column_energy_type);
    }
    if (world.cart_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&world.cart_comm);
    }
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

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    bool parse_ok = true;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parse_ok = false;
        }
    }

    if (!parse_ok || n_elems_root <= 0 || n_iters < 0) {
        MPI_Finalize();
        return 1;
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    if (dims[0] > n_elems_root || dims[1] > n_elems_root) {
        if (world_rank == 0) {
            printf("Error: the process grid (%d x %d) exceeds the grid size (%d x %d).\n",
                   dims[0], dims[1], n_elems_root, n_elems_root);
        }
        MPI_Finalize();
        return 1;
    }

    int periods[2] = {0, 0};
    MPI_Comm cart_comm = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    World world;
    buildSquare2D(world, n_elems_root, cart_comm);

    const long long n_elems = static_cast<long long>(n_elems_root) * n_elems_root;
    if (world.rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %lld elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI processes: %d (%d x %d Cartesian grid)\n", world.size,
               world.dims[0], world.dims[1]);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building distributed unstructured mesh...\n");
    }

    // Calculate local memory usage.  The dynamic total includes both buffers
    // and their one-cell halos.
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (world.rank == 0) {
        printf("Memory usage per rank: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run simulation
    if (world.rank == 0) {
        printf("Running simulation...\n");
    }
    MPI_Barrier(world.cart_comm);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters);

    MPI_Barrier(world.cart_comm);
    const double local_duration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&local_duration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, world.cart_comm);

    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    if (world.rank == 0) {
        const double duration_ms = duration * 1000.0;
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * n_elems) / duration / 1e9;

        printf("Computation time: %.3f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);

    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute hash for verification
    const uint64_t hash = computeHash(world);
    if (world.rank == 0) {
        printf("  Result hash: %016llX\n", static_cast<unsigned long long>(hash));
        printf("\n");
    }

    // Print results for external validation
    if (printResults) {
        printGlobalEnergyResults(world);
    }

    // Validation
    bool valid = true;
    if (validate) {
        valid = validateResults(world);
    }

    destroyWorld(world);
    MPI_Finalize();
    return valid ? 0 : 1;
}
