#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mpi.h>
#include <vector>

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
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    // Only current_energy is needed from elements owned by adjacent ranks.
    std::vector<val_t> top_halo;
    std::vector<val_t> bottom_halo;
    std::vector<val_t> top_send;
    std::vector<val_t> bottom_send;
    size_t owned_elements = 0;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root,
                   const int first_row, const int local_rows) {
    const size_t n_elems = static_cast<size_t>(n_elems_root) * local_rows;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);
    world.top_halo.resize(n_elems_root);
    world.bottom_halo.resize(n_elems_root);
    world.top_send.resize(n_elems_root);
    world.bottom_send.resize(n_elems_root);
    world.owned_elements = n_elems;
    
    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int local_x = 0; local_x < local_rows; ++local_x) {
        const int x = first_row + local_x;
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t idx = static_cast<size_t>(local_x) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    size_t neighbor_idx;
                    if (nx < first_row) {
                        // The top ghost row follows the owned local elements.
                        neighbor_idx = n_elems + ny;
                    } else if (nx >= first_row + local_rows) {
                        // The bottom ghost row follows the top ghost row.
                        neighbor_idx = n_elems + n_elems_root + ny;
                    } else {
                        neighbor_idx = static_cast<size_t>(nx - first_row) * n_elems_root + ny;
                    }
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    if (first_row == 0) {
        world.elements_static[0].material_idx = INFLOW_MAT_ID;
        world.elements_static[last].material_idx = OUTFLOW_MAT_ID;
    }
    if (first_row + local_rows == n_elems_root) {
        const size_t last_row = static_cast<size_t>(local_rows - 1) * n_elems_root;
        world.elements_static[last_row].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[last_row + last].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

inline void updateInteriorElement(const World& world, const size_t i,
                                  ElementDynamic& elem_write) {
    const ElementStatic& elem_static = world.elements_static[i];
    const ElementDynamic& elem_dyn = world.elements_dynamic[i];
    const Material& mat = world.materials[elem_static.material_idx];

    val_t total_flux = mat.external_flow;
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const ElementDynamic& neighbor_dyn =
            world.elements_dynamic[elem_static.connected_idx[j]];
        total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
    }

    elem_write.current_energy = elem_dyn.current_energy + total_flux;
    elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
}

inline void updateBoundaryElement(const World& world, const size_t i,
                                  ElementDynamic& elem_write) {
    const ElementStatic& elem_static = world.elements_static[i];
    const ElementDynamic& elem_dyn = world.elements_dynamic[i];
    const Material& mat = world.materials[elem_static.material_idx];

    val_t total_flux = mat.external_flow;
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const size_t neighbor_idx = elem_static.connected_idx[j];
        const val_t neighbor_energy = neighbor_idx < world.owned_elements
            ? world.elements_dynamic[neighbor_idx].current_energy
            : (neighbor_idx < world.owned_elements + world.top_halo.size()
                ? world.top_halo[neighbor_idx - world.owned_elements]
                : world.bottom_halo[neighbor_idx - world.owned_elements - world.top_halo.size()]);
        total_flux += (neighbor_energy - elem_dyn.current_energy) *
                      mat.transfer_coeff * elem_static.connected_flux[j] * 0.25;
    }

    elem_write.current_energy = elem_dyn.current_energy + total_flux;
    elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
}

// Run the simulation over one contiguous block of mesh rows.  Communication
// is limited to the two rows that cross rank boundaries and is overlapped with
// the independent interior rows.
void runSimulation(World& world, const int n_elems_root, const int local_rows,
                   const int n_iters, const int active_rank,
                   const int active_size, MPI_Comm active_comm) {
    const size_t n_elems = world.owned_elements;
    const int top_rank = active_rank == 0 ? MPI_PROC_NULL : active_rank - 1;
    const int bottom_rank = active_rank + 1 == active_size ? MPI_PROC_NULL : active_rank + 1;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        for (int y = 0; y < n_elems_root; ++y) {
            world.top_send[y] = world.elements_dynamic[y].current_energy;
            world.bottom_send[y] = world.elements_dynamic[n_elems - n_elems_root + y].current_energy;
        }

        MPI_Request requests[4];
        MPI_Irecv(world.top_halo.data(), n_elems_root, MPI_DOUBLE, top_rank, 0,
                  active_comm, &requests[0]);
        MPI_Irecv(world.bottom_halo.data(), n_elems_root, MPI_DOUBLE, bottom_rank, 1,
                  active_comm, &requests[1]);
        MPI_Isend(world.top_send.data(), n_elems_root, MPI_DOUBLE, top_rank, 1,
                  active_comm, &requests[2]);
        MPI_Isend(world.bottom_send.data(), n_elems_root, MPI_DOUBLE, bottom_rank, 0,
                  active_comm, &requests[3]);

        // Interior rows depend only on locally owned elements, so their work
        // can progress while the two boundary rows are in flight.
        for (int row = 1; row + 1 < local_rows; ++row) {
            const size_t first = static_cast<size_t>(row) * n_elems_root;
            const size_t last = first + n_elems_root;
            for (size_t i = first; i < last; ++i) {
                updateInteriorElement(world, i, world.elements_dynamic_swap[i]);
            }
        }

        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

        // Finish the first and last local rows once their halo values arrive.
        const size_t first_boundary_end = n_elems_root;
        for (size_t i = 0; i < first_boundary_end; ++i) {
            updateBoundaryElement(world, i, world.elements_dynamic_swap[i]);
        }
        if (local_rows > 1) {
            const size_t last_boundary_start = n_elems - n_elems_root;
            for (size_t i = last_boundary_start; i < n_elems; ++i) {
                updateBoundaryElement(world, i, world.elements_dynamic_swap[i]);
            }
        }
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate distributed simulation results.  Reductions preserve the original
// checks while avoiding a full-state gather on the normal validation path.
bool validateResults(const World& world, const int rank) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : world.elements_dynamic) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    
    val_t global_energy_sum = 0.0;
    val_t global_flux_sum = 0.0;
    val_t global_energy_max = 0.0;
    val_t global_energy_min = 0.0;
    MPI_Reduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    int valid = 1;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_energy_sum);
        printf("  Flux sum: %.2f\n", global_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);
    
    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;
    
        if (!std::isfinite(global_energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        }
    
        if (std::abs(global_energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            // Don't fail validation as this can happen with external flows
        }
    
        if (!std::isfinite(global_flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }
    
        if (!std::isfinite(global_energy_max) || !std::isfinite(global_energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = 0;
        }
    
        if (valid) {
            printf("  Validation: PASSED\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
}

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements, const size_t global_offset) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        uint64_t energy_bits;
        uint64_t flux_bits;
        std::memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        const uint64_t global_i = global_offset + i;
        hash ^= (energy_bits + global_i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + global_i) * 0xbf58476d1ce4e5b9ULL;
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

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (n_elems_root <= 0 || n_iters < 0) {
        if (rank == 0) {
            printf("Grid size must be positive and iterations must be non-negative\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int active_size = std::min(world_size, n_elems_root);
    const bool active = rank < active_size;
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 1 : MPI_UNDEFINED, rank, &active_comm);

    int local_rows = 0;
    int first_row = 0;
    if (active) {
        const int rows_per_rank = n_elems_root / active_size;
        const int extra_rows = n_elems_root % active_size;
        local_rows = rows_per_rank + (rank < extra_rows ? 1 : 0);
        first_row = rank * rows_per_rank + std::min(rank, extra_rows);
    }
    const size_t n_elems = static_cast<size_t>(n_elems_root) * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (active: %d)\n\n", world_size, active_size);
        printf("Building unstructured mesh...\n");
    }

    World world;
    if (active) {
        buildSquare2D(world, n_elems_root, first_row, local_rows);
    }

    if (rank == 0) {
        const size_t static_mem = n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (active) {
        runSimulation(world, n_elems_root, local_rows, n_iters, rank,
                      active_size, active_comm);
    }
    const double local_duration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&local_duration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = duration * 1000.0 / n_measured_iters;
    const double giga_elems_per_sec = duration > 0.0
        ? (static_cast<double>(n_measured_iters) * n_elems) / duration / 1e9
        : 0.0;
    const double gflops = giga_elems_per_sec * 22.0;

    const size_t global_offset = static_cast<size_t>(first_row) * n_elems_root;
    const uint64_t local_hash = active ? computeHash(world.elements_dynamic, global_offset) : 0;
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016" PRIX64 "\n\n", global_hash);
    }

    if (printResults) {
        std::vector<int> receive_counts;
        std::vector<int> displacements;
        std::vector<double> energy_data;
        if (rank == 0) {
            receive_counts.resize(world_size, 0);
            displacements.resize(world_size, 0);
            energy_data.resize(n_elems);
            const int rows_per_rank = n_elems_root / active_size;
            const int extra_rows = n_elems_root % active_size;
            for (int process = 0; process < active_size; ++process) {
                const int rows = rows_per_rank + (process < extra_rows ? 1 : 0);
                const int row_start = process * rows_per_rank + std::min(process, extra_rows);
                receive_counts[process] = rows * n_elems_root;
                displacements[process] = row_start * n_elems_root;
            }
        }

        std::vector<double> local_energy(world.owned_elements);
        for (size_t i = 0; i < world.owned_elements; ++i) {
            local_energy[i] = world.elements_dynamic[i].current_energy;
        }
        MPI_Gatherv(local_energy.data(), static_cast<int>(local_energy.size()), MPI_DOUBLE,
                    energy_data.data(), receive_counts.data(), displacements.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(energy_data, "ElementEnergy");
        }
    }

    const bool valid = !validate || validateResults(world, rank);
    if (active_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&active_comm);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
