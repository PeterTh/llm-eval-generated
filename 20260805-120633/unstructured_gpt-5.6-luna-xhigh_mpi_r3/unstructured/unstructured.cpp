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

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t));

// Static connectivity information for each locally-owned element. The
// connected_idx entries are local dynamic-array indices for local neighbors,
// and local-index-plus-ghost-offset for neighbors on another MPI rank.
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
    bool has_remote_connection;
};

// World state. The element arrays contain only the elements owned by this
// rank. Global element IDs are retained for deterministic result gathering
// and hashing.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    std::vector<idx_t> global_indices;
    std::vector<val_t> ghost_values;
    std::vector<size_t> remote_elements;

    int local_x0 = 0;
    int local_y0 = 0;
    int local_nx = 0;
    int local_ny = 0;

    int rank_minus_x = MPI_PROC_NULL;
    int rank_plus_x = MPI_PROC_NULL;
    int rank_minus_y = MPI_PROC_NULL;
    int rank_plus_y = MPI_PROC_NULL;

    MPI_Datatype row_energy_type = MPI_DATATYPE_NULL;
    MPI_Datatype column_energy_type = MPI_DATATYPE_NULL;
    MPI_Comm cart_comm = MPI_COMM_NULL;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Offsets into World::ghost_values. The first two faces are x faces and the
// last two faces are y faces. Face entries are ordered along the other axis.
size_t ghostBaseMinusX(const World&) { return 0; }
size_t ghostBasePlusX(const World& world) { return static_cast<size_t>(world.local_ny); }
size_t ghostBaseMinusY(const World& world) {
    return static_cast<size_t>(2) * static_cast<size_t>(world.local_ny);
}
size_t ghostBasePlusY(const World& world) {
    return ghostBaseMinusY(world) + static_cast<size_t>(world.local_nx);
}

// Compute the start and extent of one block in a dimension. Remainders are
// assigned to the lower coordinates, keeping all blocks non-empty because the
// process grid is chosen with no dimension larger than the mesh.
void blockRange(const int global_size, const int parts, const int coordinate,
                int& start, int& extent) {
    const int base = global_size / parts;
    const int remainder = global_size % parts;
    extent = base + (coordinate < remainder ? 1 : 0);
    start = coordinate * base + std::min(coordinate, remainder);
}

// Pick the largest rectangular process grid that fits both the requested MPI
// size and the mesh. Keeping each dimension <= n_elems_root avoids idle ranks
// caused by zero-sized blocks and gives every active rank useful work.
void chooseProcessGrid(const int n_elems_root, const int world_size,
                       int& active_size, int dims[2]) {
    const int max_dimension = std::min(n_elems_root, world_size);
    int best_product = 1;
    int best_x = 1;
    int best_y = 1;

    for (int x_parts = 1; x_parts <= max_dimension; ++x_parts) {
        const int y_parts = std::min(n_elems_root, world_size / x_parts);
        const int product = x_parts * y_parts;
        const int best_imbalance = std::abs(best_x - best_y);
        const int imbalance = std::abs(x_parts - y_parts);
        if (product > best_product ||
            (product == best_product && imbalance < best_imbalance)) {
            best_product = product;
            best_x = x_parts;
            best_y = y_parts;
        }
    }

    active_size = best_product;
    dims[0] = best_x;
    dims[1] = best_y;
}

// Build the locally-owned portion of the unstructured mesh. Connectivity is
// generated in the same order as the original global implementation, so each
// element's floating-point update has identical arithmetic.
void buildLocalSquare2D(World& world, const int n_elems_root,
                        const int local_x0, const int local_nx,
                        const int local_y0, const int local_ny,
                        const MPI_Comm cart_comm) {
    world.local_x0 = local_x0;
    world.local_y0 = local_y0;
    world.local_nx = local_nx;
    world.local_ny = local_ny;
    world.cart_comm = cart_comm;

    world.materials = {
        Material{0.8, 0.0},    // Default material
        Material{0.8, 0.5},    // Inflow material
        Material{0.8, -0.5},   // Outflow material
    };

    const size_t local_count = static_cast<size_t>(local_nx) *
                               static_cast<size_t>(local_ny);
    world.elements_static.resize(local_count);
    world.elements_dynamic.assign(local_count, ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.assign(local_count, ElementDynamic{0.0, 0.0});
    world.global_indices.resize(local_count);
    world.remote_elements.reserve(static_cast<size_t>(2) *
                                  (static_cast<size_t>(local_nx) +
                                   static_cast<size_t>(local_ny)));

    const size_t ghost_count = static_cast<size_t>(2) *
                               (static_cast<size_t>(local_nx) +
                                static_cast<size_t>(local_ny));
    world.ghost_values.assign(ghost_count, 0.0);

    const int last = n_elems_root - 1;
    const size_t local_count_idx = local_count;

    for (int local_x = 0; local_x < local_nx; ++local_x) {
        for (int local_y = 0; local_y < local_ny; ++local_y) {
            const int global_x = local_x0 + local_x;
            const int global_y = local_y0 + local_y;
            const size_t local_idx = static_cast<size_t>(local_x) *
                                     static_cast<size_t>(local_ny) +
                                     static_cast<size_t>(local_y);
            const idx_t global_idx = static_cast<idx_t>(global_x) *
                                     static_cast<idx_t>(n_elems_root) +
                                     static_cast<idx_t>(global_y);

            world.global_indices[local_idx] = global_idx;
            ElementStatic& element = world.elements_static[local_idx];
            element.material_idx = DEFAULT_MAT_ID;
            element.num_connections = 0;
            element.has_remote_connection = false;

            // The assignment order intentionally matches buildSquare2D.
            if (global_x == 0 && global_y == 0) {
                element.material_idx = INFLOW_MAT_ID;
            }
            if (global_x == 0 && global_y == last) {
                element.material_idx = OUTFLOW_MAT_ID;
            }
            if (global_x == last && global_y == 0) {
                element.material_idx = OUTFLOW_MAT_ID;
            }
            if (global_x == last && global_y == last) {
                element.material_idx = INFLOW_MAT_ID;
            }

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (const auto& offset : offsets) {
                const int neighbor_x = global_x + offset[0];
                const int neighbor_y = global_y + offset[1];
                if (neighbor_x < 0 || neighbor_x >= n_elems_root ||
                    neighbor_y < 0 || neighbor_y >= n_elems_root) {
                    continue;
                }

                const idx_t connection = element.num_connections++;
                element.connected_flux[connection] = 1.0;

                if (neighbor_x >= local_x0 &&
                    neighbor_x < local_x0 + local_nx &&
                    neighbor_y >= local_y0 &&
                    neighbor_y < local_y0 + local_ny) {
                    element.connected_idx[connection] =
                        static_cast<idx_t>(neighbor_x - local_x0) *
                            static_cast<idx_t>(local_ny) +
                        static_cast<idx_t>(neighbor_y - local_y0);
                } else {
                    element.has_remote_connection = true;
                    idx_t ghost_idx = local_count_idx;
                    if (neighbor_x == local_x0 - 1) {
                        ghost_idx += static_cast<idx_t>(ghostBaseMinusX(world) +
                                                        static_cast<size_t>(neighbor_y - local_y0));
                    } else if (neighbor_x == local_x0 + local_nx) {
                        ghost_idx += static_cast<idx_t>(ghostBasePlusX(world) +
                                                        static_cast<size_t>(neighbor_y - local_y0));
                    } else if (neighbor_y == local_y0 - 1) {
                        ghost_idx += static_cast<idx_t>(ghostBaseMinusY(world) +
                                                        static_cast<size_t>(neighbor_x - local_x0));
                    } else {
                        ghost_idx += static_cast<idx_t>(ghostBasePlusY(world) +
                                                        static_cast<size_t>(neighbor_x - local_x0));
                    }
                    element.connected_idx[connection] = ghost_idx;
                }
            }

            if (element.has_remote_connection) {
                world.remote_elements.push_back(local_idx);
            }
        }
    }
}

// Create MPI datatypes that select current_energy from a row or column of the
// local ElementDynamic array. This avoids packing/unpacking faces on every
// iteration while leaving the receive halos contiguous and cache-friendly.
void createEnergyTypes(World& world) {
    MPI_Type_vector(world.local_ny, 1, 2, MPI_DOUBLE,
                    &world.row_energy_type);
    MPI_Type_commit(&world.row_energy_type);

    MPI_Type_vector(world.local_nx, 1, world.local_ny * 2, MPI_DOUBLE,
                    &world.column_energy_type);
    MPI_Type_commit(&world.column_energy_type);
}

void destroyEnergyTypes(World& world) {
    if (world.row_energy_type != MPI_DATATYPE_NULL) {
        MPI_Type_free(&world.row_energy_type);
    }
    if (world.column_energy_type != MPI_DATATYPE_NULL) {
        MPI_Type_free(&world.column_energy_type);
    }
}

inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                         val_t connection_flux, val_t other_energy) {
    return (other_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

inline void updateElement(World& world, const size_t local_idx) {
    const ElementStatic& elem_static = world.elements_static[local_idx];
    const ElementDynamic& elem_dyn = world.elements_dynamic[local_idx];
    const Material& mat = world.materials[elem_static.material_idx];
    const size_t local_count = world.elements_dynamic.size();

    val_t total_flux = mat.external_flow;
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const idx_t neighbor_idx = elem_static.connected_idx[j];
        const val_t neighbor_energy =
            neighbor_idx < local_count
                ? world.elements_dynamic[static_cast<size_t>(neighbor_idx)].current_energy
                : world.ghost_values[static_cast<size_t>(neighbor_idx - local_count)];
        total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j],
                                  neighbor_energy);
    }

    ElementDynamic& elem_write = world.elements_dynamic_swap[local_idx];
    elem_write.current_energy = elem_dyn.current_energy + total_flux;
    elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
}

// Run simulation for n_iters iterations. The only inter-rank dependency is
// the previous iteration's current_energy on a neighboring block, so a single
// halo exchange per iteration is sufficient.
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();

    for (int iter = 0; iter < n_iters; ++iter) {
        const int request_count_before_compute =
            (world.rank_minus_x != MPI_PROC_NULL ? 2 : 0) +
            (world.rank_plus_x != MPI_PROC_NULL ? 2 : 0) +
            (world.rank_minus_y != MPI_PROC_NULL ? 2 : 0) +
            (world.rank_plus_y != MPI_PROC_NULL ? 2 : 0);

        if (request_count_before_compute == 0) {
            for (size_t i = 0; i < n_elems; ++i) {
                updateElement(world, i);
            }
        } else {
            // Post the exchange explicitly here so computation can overlap
            // communication; the wait below completes the face transfers.
            constexpr int TAG_X_MINUS = 101;
            constexpr int TAG_X_PLUS = 102;
            constexpr int TAG_Y_MINUS = 103;
            constexpr int TAG_Y_PLUS = 104;
            MPI_Request requests[8];
            int request_count = 0;
            val_t* current = &world.elements_dynamic[0].current_energy;

            if (world.rank_minus_x != MPI_PROC_NULL) {
                MPI_Irecv(world.ghost_values.data() + ghostBaseMinusX(world),
                          world.local_ny, MPI_DOUBLE, world.rank_minus_x,
                          TAG_X_PLUS, world.cart_comm, &requests[request_count++]);
                MPI_Isend(current, 1, world.row_energy_type, world.rank_minus_x,
                          TAG_X_MINUS, world.cart_comm, &requests[request_count++]);
            }
            if (world.rank_plus_x != MPI_PROC_NULL) {
                MPI_Irecv(world.ghost_values.data() + ghostBasePlusX(world),
                          world.local_ny, MPI_DOUBLE, world.rank_plus_x,
                          TAG_X_MINUS, world.cart_comm, &requests[request_count++]);
                val_t* last_row = &world.elements_dynamic[
                    static_cast<size_t>(world.local_nx - 1) *
                    static_cast<size_t>(world.local_ny)].current_energy;
                MPI_Isend(last_row, 1, world.row_energy_type, world.rank_plus_x,
                          TAG_X_PLUS, world.cart_comm, &requests[request_count++]);
            }
            if (world.rank_minus_y != MPI_PROC_NULL) {
                MPI_Irecv(world.ghost_values.data() + ghostBaseMinusY(world),
                          world.local_nx, MPI_DOUBLE, world.rank_minus_y,
                          TAG_Y_PLUS, world.cart_comm, &requests[request_count++]);
                MPI_Isend(current, 1, world.column_energy_type, world.rank_minus_y,
                          TAG_Y_MINUS, world.cart_comm, &requests[request_count++]);
            }
            if (world.rank_plus_y != MPI_PROC_NULL) {
                MPI_Irecv(world.ghost_values.data() + ghostBasePlusY(world),
                          world.local_nx, MPI_DOUBLE, world.rank_plus_y,
                          TAG_Y_MINUS, world.cart_comm, &requests[request_count++]);
                val_t* last_column = &world.elements_dynamic[
                    static_cast<size_t>(world.local_ny - 1)].current_energy;
                MPI_Isend(last_column, 1, world.column_energy_type, world.rank_plus_y,
                          TAG_Y_PLUS, world.cart_comm, &requests[request_count++]);
            }

            for (size_t i = 0; i < n_elems; ++i) {
                if (!world.elements_static[i].has_remote_connection) {
                    updateElement(world, i);
                }
            }
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

            for (const size_t i : world.remote_elements) {
                updateElement(world, i);
            }
        }

        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results in global element order. The caller gathers the
// distributed state before invoking this function, preserving the original
// summation and extrema semantics.
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

// Compute the same result hash as the original implementation, using global
// indices so the XOR reduction is independent of the process decomposition.
uint64_t computeLocalHash(const World& world) {
    uint64_t hash = 0;
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const uint64_t* energy_bits = reinterpret_cast<const uint64_t*>(
            &world.elements_dynamic[i].current_energy);
        const uint64_t* flux_bits = reinterpret_cast<const uint64_t*>(
            &world.elements_dynamic[i].total_flux);
        const uint64_t global_idx = world.global_indices[i];
        hash ^= (*energy_bits + global_idx) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*flux_bits + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

// Gather state only for the optional output/validation paths. The normal
// benchmark path retains distributed memory usage and performs no all-results
// gather. Results are reconstructed in global element order on rank zero.
void gatherResults(const World& world, std::vector<ElementDynamic>& global,
                   const int cart_rank, const int cart_size) {
    const int local_count = static_cast<int>(world.elements_dynamic.size());
    std::vector<int> counts(cart_rank == 0 ? static_cast<size_t>(cart_size) : 0);
    MPI_Gather(&local_count, 1, MPI_INT,
               cart_rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0,
               world.cart_comm);

    std::vector<int> displacements;
    std::vector<int> byte_counts;
    std::vector<int> byte_displacements;
    int total_count = 0;
    if (cart_rank == 0) {
        displacements.resize(static_cast<size_t>(cart_size));
        byte_counts.resize(static_cast<size_t>(cart_size));
        byte_displacements.resize(static_cast<size_t>(cart_size));
        for (int rank = 0; rank < cart_size; ++rank) {
            displacements[rank] = total_count;
            byte_displacements[rank] = total_count * static_cast<int>(sizeof(ElementDynamic));
            byte_counts[rank] = counts[rank] * static_cast<int>(sizeof(ElementDynamic));
            total_count += counts[rank];
        }
        global.assign(static_cast<size_t>(total_count), ElementDynamic{0.0, 0.0});
    }

    std::vector<idx_t> gathered_indices(
        cart_rank == 0 ? static_cast<size_t>(total_count) : 0);
    std::vector<ElementDynamic> gathered_dynamics(
        cart_rank == 0 ? static_cast<size_t>(total_count) : 0);

    MPI_Gatherv(world.global_indices.data(), local_count, MPI_UINT64_T,
                cart_rank == 0 ? gathered_indices.data() : nullptr,
                cart_rank == 0 ? counts.data() : nullptr,
                cart_rank == 0 ? displacements.data() : nullptr,
                MPI_UINT64_T, 0, world.cart_comm);

    const int local_bytes = local_count * static_cast<int>(sizeof(ElementDynamic));
    MPI_Gatherv(world.elements_dynamic.data(), local_bytes, MPI_BYTE,
                cart_rank == 0 ? gathered_dynamics.data() : nullptr,
                cart_rank == 0 ? byte_counts.data() : nullptr,
                cart_rank == 0 ? byte_displacements.data() : nullptr,
                MPI_BYTE, 0, world.cart_comm);

    if (cart_rank == 0) {
        for (int i = 0; i < total_count; ++i) {
            global[static_cast<size_t>(gathered_indices[i])] = gathered_dynamics[i];
        }
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
    bool parse_error = false;

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
            parse_error = true;
        }
    }

    if (parse_error || n_elems_root <= 0 || n_iters < 0) {
        if (world_rank == 0 && !parse_error) {
            printf("Grid size and iteration count must be non-negative, with a positive grid size.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int dims[2] = {1, 1};
    int active_size = 1;
    chooseProcessGrid(n_elems_root, world_size, active_size, dims);

    const int color = world_rank < active_size ? 1 : MPI_UNDEFINED;
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &active_comm);
    if (active_comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }

    int active_ranks = 1;
    MPI_Comm_size(active_comm, &active_ranks);

    int periods[2] = {0, 0};
    MPI_Comm cart_comm = MPI_COMM_NULL;
    MPI_Cart_create(active_comm, 2, dims, periods, 0, &cart_comm);

    int cart_rank = 0;
    int cart_size = 1;
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Comm_size(cart_comm, &cart_size);

    int coords[2] = {0, 0};
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    int local_x0 = 0;
    int local_nx = 0;
    int local_y0 = 0;
    int local_ny = 0;
    blockRange(n_elems_root, dims[0], coords[0], local_x0, local_nx);
    blockRange(n_elems_root, dims[1], coords[1], local_y0, local_ny);

    World world;
    world.cart_comm = cart_comm;
    MPI_Cart_shift(cart_comm, 0, 1, &world.rank_minus_x, &world.rank_plus_x);
    MPI_Cart_shift(cart_comm, 1, 1, &world.rank_minus_y, &world.rank_plus_y);
    buildLocalSquare2D(world, n_elems_root, local_x0, local_nx,
                       local_y0, local_ny, cart_comm);
    createEnergyTypes(world);

    if (cart_rank == 0) {
        const long long n_elems = static_cast<long long>(n_elems_root) *
                                  static_cast<long long>(n_elems_root);
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %lld elements\n", n_elems_root,
               n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d (process grid: %d x %d)\n", active_ranks,
               dims[0], dims[1]);
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    const unsigned long long local_static_mem =
        static_cast<unsigned long long>(world.elements_static.size() *
                                        sizeof(ElementStatic));
    const unsigned long long local_dynamic_mem =
        static_cast<unsigned long long>(world.elements_dynamic.size() *
                                        sizeof(ElementDynamic) * 2);
    unsigned long long static_mem = 0;
    unsigned long long dynamic_mem = 0;
    MPI_Reduce(&local_static_mem, &static_mem, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_dynamic_mem, &dynamic_mem, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, cart_comm);

    if (cart_rank == 0) {
        const unsigned long long total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
        fflush(stdout);
    }

    MPI_Barrier(cart_comm);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters);
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    uint64_t local_hash = computeLocalHash(world);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, cart_comm);

    if (cart_rank == 0) {
        const long long duration_ms = std::max<long long>(
            1, static_cast<long long>(std::llround(elapsed * 1000.0)));
        printf("Computation time: %lld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter =
            static_cast<double>(duration_ms) / n_measured_iters;
        const long long n_elems = static_cast<long long>(n_elems_root) *
                                  static_cast<long long>(n_elems_root);
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * n_elems) /
            (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", static_cast<unsigned long>(hash));
        printf("\n");
    }

    std::vector<ElementDynamic> global_results;
    if (printResults || validate) {
        gatherResults(world, global_results, cart_rank, cart_size);
        if (cart_rank == 0 && printResults) {
            std::vector<double> energy_data;
            energy_data.reserve(global_results.size());
            for (const auto& elem : global_results) {
                energy_data.push_back(elem.current_energy);
            }
            print_results(energy_data, "ElementEnergy");
        }
    }

    bool valid = true;
    if (validate && cart_rank == 0) {
        valid = validateResults(global_results);
    }
    MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, cart_comm);

    destroyEnergyTypes(world);
    MPI_Comm_free(&cart_comm);
    MPI_Comm_free(&active_comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
