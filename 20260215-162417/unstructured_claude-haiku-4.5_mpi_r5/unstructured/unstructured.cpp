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
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);
    
    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations with MPI parallelization
void runSimulation(World& world, const int n_iters, const int n_elems_root,
                   int rank, int num_ranks) {
    // Compute local grid bounds for this rank (owns these rows)
    int rows_per_rank = n_elems_root / num_ranks;
    int extra_rows = n_elems_root % num_ranks;
    
    int local_row_start = rank * rows_per_rank + std::min(rank, extra_rows);
    int local_row_end = (rank + 1) * rows_per_rank + std::min(rank + 1, extra_rows);
    
    int local_start_idx = local_row_start * n_elems_root;
    int local_end_idx = local_row_end * n_elems_root;
    
    // Determine neighbors
    int rank_below = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int rank_above = (rank < num_ranks - 1) ? rank + 1 : MPI_PROC_NULL;
    
    // Ghostcell storage for neighbor boundaries
    std::vector<ElementDynamic> ghostcell_below(n_elems_root);
    std::vector<ElementDynamic> ghostcell_above(n_elems_root);
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Phase 1: Exchange boundary rows with neighbors
        if (num_ranks > 1) {
            std::vector<ElementDynamic> send_above(n_elems_root);
            std::vector<ElementDynamic> send_below(n_elems_root);
            
            // Pack rows for sending
            for (int y = 0; y < n_elems_root; ++y) {
                send_above[y] = world.elements_dynamic[local_end_idx - n_elems_root + y];
                send_below[y] = world.elements_dynamic[local_start_idx + y];
            }
            
            // Exchange with neighbors
            MPI_Sendrecv(send_above.data(), n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
                         rank_above, 0,
                         ghostcell_below.data(), n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
                         rank_below, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            MPI_Sendrecv(send_below.data(), n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
                         rank_below, 0,
                         ghostcell_above.data(), n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
                         rank_above, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Phase 2: Compute local elements using ghostcells
        for (int i = local_start_idx; i < local_end_idx; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            val_t total_flux = mat.external_flow;
            
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic* neighbor_dyn_ptr;
                
                if (num_ranks > 1) {
                    // Check if neighbor is in ghostcells
                    if ((int)neighbor_idx < local_start_idx && rank_below != MPI_PROC_NULL) {
                        // Neighbor is below: use ghostcell
                        int offset = (int)neighbor_idx - (local_start_idx - n_elems_root);
                        neighbor_dyn_ptr = &ghostcell_below[offset];
                    } else if ((int)neighbor_idx >= local_end_idx && rank_above != MPI_PROC_NULL) {
                        // Neighbor is above: use ghostcell
                        int offset = (int)neighbor_idx - local_end_idx;
                        neighbor_dyn_ptr = &ghostcell_above[offset];
                    } else {
                        // Neighbor is local
                        neighbor_dyn_ptr = &world.elements_dynamic[neighbor_idx];
                    }
                } else {
                    neighbor_dyn_ptr = &world.elements_dynamic[neighbor_idx];
                }
                
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], *neighbor_dyn_ptr);
            }
            
            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results
bool validateResults(const World& world) {
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
    
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    
    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;
    
    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    
    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        // Don't fail validation as this can happen with external flows
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

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    
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
        } else if (rank == 0) {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    // Only rank 0 prints output (or we print minimal info)
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI)\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI Ranks: %d\n", num_ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh (all ranks build their portion)
    if (rank == 0) {
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root);
    
    if (rank == 0) {
        const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
        const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (rank == 0) {
        printf("Running simulation...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters, n_elems_root, rank, num_ranks);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Gather all element data to rank 0
    World global_world;
    if (rank == 0) {
        global_world.materials = world.materials;
        global_world.elements_static = world.elements_static;
        global_world.elements_dynamic.resize(n_elems);
        global_world.elements_dynamic_swap.resize(n_elems);
    }
    
    // Gather dynamic element data
    int rows_per_rank = n_elems_root / num_ranks;
    int extra_rows = n_elems_root % num_ranks;
    
    std::vector<int> recv_counts(num_ranks);
    std::vector<int> recv_displacements(num_ranks);
    
    for (int r = 0; r < num_ranks; ++r) {
        int r_row_start = r * rows_per_rank + std::min(r, extra_rows);
        int r_row_end = (r + 1) * rows_per_rank + std::min(r + 1, extra_rows);
        int r_start_idx = r_row_start * n_elems_root;
        int r_end_idx = r_row_end * n_elems_root;
        recv_counts[r] = (r_end_idx - r_start_idx) * sizeof(ElementDynamic);
        recv_displacements[r] = r_start_idx * sizeof(ElementDynamic);
    }
    
    int local_start_elem = (rank * rows_per_rank + std::min(rank, extra_rows)) * n_elems_root;
    
    MPI_Gatherv(world.elements_dynamic.data() + local_start_elem,
                recv_counts[rank], MPI_BYTE,
                rank == 0 ? (char*)global_world.elements_dynamic.data() : nullptr,
                recv_counts.data(), recv_displacements.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);
    
    auto duration_ms_local = duration_ms;
    
    // Gather timing from all ranks
    long max_duration_ms = 0;
    MPI_Reduce(&duration_ms_local, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        // Use gathered global data for validation/output
        printf("Computation time: %ld ms\n", max_duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(max_duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (max_duration_ms / 1000.0) / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        
        // Compute hash for verification
        const uint64_t hash = computeHash(global_world.elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation (rank 0 collects and prints)
    if (printResults && rank == 0) {
        std::vector<double> energyData;
        energyData.reserve(n_elems);
        for (const auto& elem : global_world.elements_dynamic) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation (rank 0 validates)
    if (validate && rank == 0) {
        bool valid = validateResults(global_world);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
