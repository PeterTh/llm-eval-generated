#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <map>

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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (local indices)
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
    
    // MPI related data
    int rank;
    int size;
    idx_t local_start_x;
    idx_t local_end_x;
    idx_t n_elems_global;
    idx_t n_elems_root;
    
    // Ghost cell mapping: global_index -> local_ghost_index
    std::map<idx_t, idx_t> ghost_map;
    std::vector<idx_t> ghost_global_indices;
    
    // Communication buffers
    std::vector<val_t> send_buffer_top;
    std::vector<val_t> recv_buffer_top;
    std::vector<val_t> send_buffer_bottom;
    std::vector<val_t> recv_buffer_bottom;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Helper to get local index from global coordinates
// Returns -1 if not local and not a ghost (yet)
idx_t getLocalIndex(World& world, int x, int y) {
    idx_t global_idx = x * world.n_elems_root + y;
    
    // Check if it's a local element
    if (x >= (int)world.local_start_x && x < (int)world.local_end_x) {
        return (x - world.local_start_x) * world.n_elems_root + y;
    }
    
    // Check if it's already a ghost
    auto it = world.ghost_map.find(global_idx);
    if (it != world.ghost_map.end()) {
        return it->second;
    }
    
    // Create new ghost
    idx_t ghost_idx = world.elements_static.size();
    world.ghost_map[global_idx] = ghost_idx;
    world.ghost_global_indices.push_back(global_idx);
    
    // Resize vectors to accommodate ghost
    world.elements_static.resize(ghost_idx + 1);
    world.elements_dynamic.resize(ghost_idx + 1);
    world.elements_dynamic_swap.resize(ghost_idx + 1);
    
    // Initialize ghost state (will be overwritten by communication)
    world.elements_static[ghost_idx].num_connections = 0; // Ghosts don't compute, they just provide data
    world.elements_dynamic[ghost_idx].current_energy = 0.0;
    world.elements_dynamic[ghost_idx].total_flux = 0.0;
    
    return ghost_idx;
}

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    world.n_elems_root = n_elems_root;
    world.n_elems_global = (idx_t)n_elems_root * n_elems_root;
    
    // Decompose domain by rows (x-coordinate)
    idx_t rows_per_rank = n_elems_root / world.size;
    idx_t remainder = n_elems_root % world.size;
    
    if (world.rank < (int)remainder) {
        world.local_start_x = world.rank * (rows_per_rank + 1);
        world.local_end_x = world.local_start_x + rows_per_rank + 1;
    } else {
        world.local_start_x = world.rank * rows_per_rank + remainder;
        world.local_end_x = world.local_start_x + rows_per_rank;
    }
    
    idx_t n_local_elems = (world.local_end_x - world.local_start_x) * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate local elements
    world.elements_static.reserve(n_local_elems + n_elems_root * 2); // heuristic for ghosts
    world.elements_dynamic.reserve(n_local_elems + n_elems_root * 2);
    world.elements_dynamic_swap.reserve(n_local_elems + n_elems_root * 2);
    
    world.elements_static.resize(n_local_elems);
    world.elements_dynamic.resize(n_local_elems);
    world.elements_dynamic_swap.resize(n_local_elems);
    
    // Initialize all local elements with default material and zero energy
    for (size_t i = 0; i < n_local_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = world.local_start_x; x < (int)world.local_end_x; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            // Local index
            idx_t local_idx = (x - world.local_start_x) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    idx_t neighbor_local_idx = getLocalIndex(world, nx, ny);
                    elem.connected_idx[elem.num_connections] = neighbor_local_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    // We need to check if we own the corners
    
    // (0, 0) - Inflow
    if (world.local_start_x == 0) {
        world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    }
    
    const int last = n_elems_root - 1;
    
    // (0, last) - Outflow
    if (world.local_start_x == 0) {
        world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    }
    
    // (last, 0) - Outflow
    if (world.local_end_x == (idx_t)n_elems_root) {
        idx_t local_x = (n_elems_root - 1) - world.local_start_x;
        world.elements_static[local_x * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    }
    
    // (last, last) - Inflow
    if (world.local_end_x == (idx_t)n_elems_root) {
        idx_t local_x = (n_elems_root - 1) - world.local_start_x;
        world.elements_static[local_x * n_elems_root + last].material_idx = INFLOW_MAT_ID;
    }
    
    // Prepare communication buffers
    // In this simple 1D decomposition, we only need to exchange top/bottom rows
    world.send_buffer_top.resize(n_elems_root);
    world.recv_buffer_top.resize(n_elems_root);
    world.send_buffer_bottom.resize(n_elems_root);
    world.recv_buffer_bottom.resize(n_elems_root);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

void exchangeGhosts(World& world) {
    int top_rank = (world.rank == 0) ? MPI_PROC_NULL : world.rank - 1;
    int bottom_rank = (world.rank == world.size - 1) ? MPI_PROC_NULL : world.rank + 1;
    
    // Pack data for top neighbor (our first row)
    if (world.local_end_x > world.local_start_x) {
        for (int i = 0; i < (int)world.n_elems_root; ++i) {
            world.send_buffer_top[i] = world.elements_dynamic[i].current_energy;
        }
    }
    
    // Pack data for bottom neighbor (our last row)
    if (world.local_end_x > world.local_start_x) {
        idx_t last_row_start = (world.local_end_x - world.local_start_x - 1) * world.n_elems_root;
        for (int i = 0; i < (int)world.n_elems_root; ++i) {
            world.send_buffer_bottom[i] = world.elements_dynamic[last_row_start + i].current_energy;
        }
    }
    
    // Exchange with top
    // If we have no elements, we send nothing? No, neighbor expects a message.
    // But if we have no elements, we shouldn't be neighbor to anyone in a 1D decomposition...
    // Except maybe if we are squeezed out?
    // If I have 0 rows, my 'top_rank' (rank-1) might have rows. My 'bottom_rank' (rank+1) might not.
    // If I have 0 rows, I effectively don't exist in the grid.
    // But MPI communication pattern is fixed (rank i talks to i-1 and i+1).
    // If rank i has 0 rows, it should probably just send dummy data or not participate?
    // However, the neighbor rank i-1 expects data from rank i.
    // If rank i has 0 rows, rank i-1 should NOT expect data from rank i because rank i doesn't own the adjacent row.
    // But rank i-1 logic is: bottom_rank = rank + 1. It assumes rank + 1 owns the next row.
    
    // This simple 1D logic breaks if ranks are empty.
    // Fix: We assume n_elems_root >= size for this benchmark.
    // Or we add a check to only exchange if we have rows.
    // AND if the neighbor has rows.
    // But checking neighbor's row count requires communication.
    
    // For this benchmark, let's assume n_elems_root >= size.
    // But to be safe against segfaults, I'll add the check and fill with zeros if empty.
    
    if (world.local_end_x == world.local_start_x) {
        std::fill(world.send_buffer_top.begin(), world.send_buffer_top.end(), 0.0);
        std::fill(world.send_buffer_bottom.begin(), world.send_buffer_bottom.end(), 0.0);
    }
    
    // Exchange with top
    MPI_Sendrecv(world.send_buffer_top.data(), world.n_elems_root, MPI_DOUBLE, top_rank, 0,
                 world.recv_buffer_top.data(), world.n_elems_root, MPI_DOUBLE, top_rank, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                 
    // Exchange with bottom
    MPI_Sendrecv(world.send_buffer_bottom.data(), world.n_elems_root, MPI_DOUBLE, bottom_rank, 1,
                 world.recv_buffer_bottom.data(), world.n_elems_root, MPI_DOUBLE, bottom_rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                 
    // Unpack received data into ghosts
    // Data received from top corresponds to row (start_x - 1)
    if (top_rank != MPI_PROC_NULL) {
        int global_x = world.local_start_x - 1;
        for (int y = 0; y < (int)world.n_elems_root; ++y) {
            idx_t global_idx = global_x * world.n_elems_root + y;
            auto it = world.ghost_map.find(global_idx);
            if (it != world.ghost_map.end()) {
                world.elements_dynamic[it->second].current_energy = world.recv_buffer_top[y];
            }
        }
    }
    
    // Data received from bottom corresponds to row (end_x)
    if (bottom_rank != MPI_PROC_NULL) {
        int global_x = world.local_end_x;
        for (int y = 0; y < (int)world.n_elems_root; ++y) {
            idx_t global_idx = global_x * world.n_elems_root + y;
            auto it = world.ghost_map.find(global_idx);
            if (it != world.ghost_map.end()) {
                world.elements_dynamic[it->second].current_energy = world.recv_buffer_bottom[y];
            }
        }
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    // Only update OWNED elements
    const size_t n_local_elems = (world.local_end_x - world.local_start_x) * world.n_elems_root;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange ghost data
        exchangeGhosts(world);
        
        // Update all LOCAL elements
        for (size_t i = 0; i < n_local_elems; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements (which may include ghosts)
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }
            
            // Update element state
            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
        
        // Swap buffers (pointers only conceptually, but here we copy vector data or swap vectors)
        // Since ghosts are at the end, and we only updated local elements in swap buffer...
        // We actually need to swap the whole vectors, but ghosts in swap buffer might be stale?
        // No, ghosts are read-only during update. Next iteration exchangeGhosts will overwrite them in 'elements_dynamic'.
        // So we just need to make sure 'elements_dynamic' gets the new values.
        
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
        
        // The ghosts in the new 'elements_dynamic' are garbage now, but exchangeGhosts will fix them.
    }
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    const size_t n_local_elems = (world.local_end_x - world.local_start_x) * world.n_elems_root;
    
    for (size_t i = 0; i < n_local_elems; ++i) {
        const auto& elem = world.elements_dynamic[i];
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    
    // Reduce results
    val_t global_energy_sum, global_flux_sum, global_energy_max, global_energy_min;
    
    MPI_Reduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    if (world.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_energy_sum);
        printf("  Flux sum: %.2f\n", global_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);
        
        // Check for numerical issues
        constexpr val_t energy_epsilon = 1e-8;
        
        if (!std::isfinite(global_energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            return false;
        }
        
        if (std::abs(global_energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            // Don't fail validation as this can happen with external flows
        }
        
        if (!std::isfinite(global_flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            return false;
        }
        
        if (!std::isfinite(global_energy_max) || !std::isfinite(global_energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            return false;
        }
        
        printf("  Validation: PASSED\n");
    }
    return true;
}

// Compute a simple hash of the results for verification
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    const size_t n_local_elems = (world.local_end_x - world.local_start_x) * world.n_elems_root;
    
    for (size_t i = 0; i < n_local_elems; ++i) {
        // Global index to make hash independent of partitioning
        idx_t global_idx = (world.local_start_x + i / world.n_elems_root) * world.n_elems_root + (i % world.n_elems_root);
        
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
        
        // Use global index for hash seed
        hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }
    
    uint64_t global_hash;
    MPI_Reduce(&hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, MPI_COMM_WORLD);
    
    return global_hash;
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
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
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
            if (rank == 0) printUsage(argv[0]);
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
    
    const idx_t n_elems = (idx_t)n_elems_root * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI)\n");
        printf("==================================================\n");
        printf("Grid size: %d x %d = %ld elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI Ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        
        printf("Building unstructured mesh...\n");
    }
    
    World world;
    world.rank = rank;
    world.size = size;
    buildSquare2D(world, n_elems_root);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t local_total_mem = static_mem + dynamic_mem;
    
    size_t global_total_mem;
    MPI_Reduce(&local_total_mem, &global_total_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Total Memory usage: %.2f MB\n", global_total_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long local_duration_ms = static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (double)n_elems) / (duration_ms / 1000.0) / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    // Gather all data to rank 0 (careful with memory!)
    // For large runs, we shouldn't do this, but for this benchmark functionality it's needed
    if (printResults) {
        // Collect local energy data
        const size_t n_local_elems = (world.local_end_x - world.local_start_x) * world.n_elems_root;
        std::vector<double> local_energy;
        local_energy.reserve(n_local_elems);
        for (size_t i = 0; i < n_local_elems; ++i) {
            local_energy.push_back(world.elements_dynamic[i].current_energy);
        }
        
        // Determine counts and displacements for gather
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        int local_count = (int)n_local_elems;
        
        MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + counts[i-1];
            }
        }
        
        // Gather data
        std::vector<double> all_energy;
        if (rank == 0) {
            all_energy.resize(n_elems);
        }
        
        MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE, 
                    all_energy.data(), counts.data(), displs.data(), MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);
                    
        if (rank == 0) {
            print_results(all_energy, "ElementEnergy");
        }
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (!valid && rank == 0) {
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
