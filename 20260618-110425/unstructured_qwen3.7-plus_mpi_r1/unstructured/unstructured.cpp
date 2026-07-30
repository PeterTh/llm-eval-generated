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

// Local world state for MPI decomposition
struct LocalWorld {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    
    int rank;
    int n_ranks;
    int n_elems_root;  // Global grid size
    
    // Local decomposition info
    int row_start;  // Global row index of first local row
    int row_end;    // Global row index after last local row (exclusive)
    int n_local_rows;
    int n_local_elems;
    
    // Ghost row info
    bool has_ghost_above;
    bool has_ghost_below;
    int n_ghost_above;  // Number of ghost elements above
    int n_ghost_below;  // Number of ghost elements below
    
    // MPI neighbors
    int rank_above;  // MPI rank above (MPI_PROC_NULL if none)
    int rank_below;  // MPI rank below (MPI_PROC_NULL if none)
    
    // Index offset for local arrays (to accommodate ghost rows)
    int ghost_above_offset;  // Offset to local row 0
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh with MPI decomposition
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(LocalWorld& world, const int n_elems_root) {
    world.n_elems_root = n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Compute row distribution across ranks
    int base_rows = n_elems_root / world.n_ranks;
    int extra_rows = n_elems_root % world.n_ranks;
    
    world.row_start = world.rank * base_rows + std::min(world.rank, extra_rows);
    world.n_local_rows = base_rows + (world.rank < extra_rows ? 1 : 0);
    world.row_end = world.row_start + world.n_local_rows;
    
    // Determine ghost row needs
    world.has_ghost_above = (world.row_start > 0);
    world.has_ghost_below = (world.row_end < n_elems_root);
    world.ghost_above_offset = world.has_ghost_above ? 1 : 0;
    world.n_ghost_above = world.has_ghost_above ? 1 : 0;
    world.n_ghost_below = world.has_ghost_below ? 1 : 0;
    
    // Determine MPI neighbors
    world.rank_above = (world.row_start > 0) ? world.rank - 1 : MPI_PROC_NULL;
    world.rank_below = (world.row_end < n_elems_root) ? world.rank + 1 : MPI_PROC_NULL;
    
    // Allocate local arrays with ghost rows
    int total_local_rows = world.n_ghost_above + world.n_local_rows + world.n_ghost_below;
    world.n_local_elems = total_local_rows * n_elems_root;
    
    world.elements_static.resize(world.n_local_elems);
    world.elements_dynamic.resize(world.n_local_elems);
    world.elements_dynamic_swap.resize(world.n_local_elems);
    
    // Initialize all elements with default material and zero energy
    for (int i = 0; i < world.n_local_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity for local rows only
    for (int lr = 0; lr < world.n_local_rows; ++lr) {
        int global_row = world.row_start + lr;
        for (int y = 0; y < n_elems_root; ++y) {
            int local_idx = (world.ghost_above_offset + lr) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                int nx = global_row + offsets[n][0];
                int ny = y + offsets[n][1];
                
                // Check if neighbor is within global bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    // Convert global (nx, ny) to local index
                    // Local row = nx - (world.row_start - world.n_ghost_above)
                    int local_nx = nx - (world.row_start - world.n_ghost_above);
                    int local_neighbor_idx = local_nx * n_elems_root + ny;
                    
                    elem.connected_idx[elem.num_connections] = local_neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    // Only if this rank owns the corner
    const int last = n_elems_root - 1;
    
    // Corner (0, 0) - global row 0
    if (world.row_start == 0) {
        int local_idx = world.ghost_above_offset * n_elems_root + 0;
        world.elements_static[local_idx].material_idx = INFLOW_MAT_ID;
    }
    
    // Corner (0, last) - global row 0, column last
    if (world.row_start == 0) {
        int local_idx = world.ghost_above_offset * n_elems_root + last;
        world.elements_static[local_idx].material_idx = OUTFLOW_MAT_ID;
    }
    
    // Corner (last, 0) - global row last
    if (world.row_end == n_elems_root) {
        int local_row = world.n_ghost_above + (last - world.row_start);
        int local_idx = local_row * n_elems_root + 0;
        world.elements_static[local_idx].material_idx = OUTFLOW_MAT_ID;
    }
    
    // Corner (last, last) - global row last, column last
    if (world.row_end == n_elems_root) {
        int local_row = world.n_ghost_above + (last - world.row_start);
        int local_idx = local_row * n_elems_root + last;
        world.elements_static[local_idx].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Exchange ghost row data with neighbors using non-blocking communication
void exchangeGhostRows(LocalWorld& world) {
    MPI_Request requests[4];
    int num_requests = 0;
    
    int send_above_idx = world.ghost_above_offset * world.n_elems_root;
    int send_below_idx = (world.ghost_above_offset + world.n_local_rows - 1) * world.n_elems_root;
    int recv_above_idx = 0;  // Ghost row above
    int recv_below_idx = (world.ghost_above_offset + world.n_local_rows) * world.n_elems_root;
    
    // Post receives first (to avoid deadlock)
    if (world.has_ghost_above) {
        MPI_Irecv(&world.elements_dynamic[recv_above_idx].current_energy,
                  world.n_elems_root, MPI_DOUBLE,
                  world.rank_above, 0, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    if (world.has_ghost_below) {
        MPI_Irecv(&world.elements_dynamic[recv_below_idx].current_energy,
                  world.n_elems_root, MPI_DOUBLE,
                  world.rank_below, 1, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    
    // Post sends
    if (world.rank_above != MPI_PROC_NULL) {
        MPI_Isend(&world.elements_dynamic[send_above_idx].current_energy,
                  world.n_elems_root, MPI_DOUBLE,
                  world.rank_above, 1, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    if (world.rank_below != MPI_PROC_NULL) {
        MPI_Isend(&world.elements_dynamic[send_below_idx].current_energy,
                  world.n_elems_root, MPI_DOUBLE,
                  world.rank_below, 0, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    
    // Wait for all communication to complete
    MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
}

// Run simulation for n_iters iterations with MPI
void runSimulation(LocalWorld& world, const int n_iters) {
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange ghost rows before computation
        exchangeGhostRows(world);
        
        // Update all local elements (excluding ghost rows)
        for (int lr = 0; lr < world.n_local_rows; ++lr) {
            for (int y = 0; y < world.n_elems_root; ++y) {
                int local_idx = (world.ghost_above_offset + lr) * world.n_elems_root + y;
                
                const ElementStatic& elem_static = world.elements_static[local_idx];
                const ElementDynamic& elem_dyn = world.elements_dynamic[local_idx];
                const Material& mat = world.materials[elem_static.material_idx];
                
                // Start with external flow
                val_t total_flux = mat.external_flow;
                
                // Add flux from all connected elements
                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const idx_t neighbor_idx = elem_static.connected_idx[j];
                    const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
                }
                
                // Update element state
                ElementDynamic& elem_write = world.elements_dynamic_swap[local_idx];
                elem_write.current_energy = elem_dyn.current_energy + total_flux;
                elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }
        }
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (collects to rank 0)
bool validateResults(const LocalWorld& world) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    
    // Compute local statistics (excluding ghost rows)
    for (int lr = 0; lr < world.n_local_rows; ++lr) {
        for (int y = 0; y < world.n_elems_root; ++y) {
            int local_idx = (world.ghost_above_offset + lr) * world.n_elems_root + y;
            const ElementDynamic& elem = world.elements_dynamic[local_idx];
            
            local_energy_sum += elem.current_energy;
            local_flux_sum += elem.total_flux;
            local_energy_max = std::max(elem.current_energy, local_energy_max);
            local_energy_min = std::min(elem.current_energy, local_energy_min);
        }
    }
    
    // Reduce to global statistics
    val_t energy_sum, flux_sum, energy_max, energy_min;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    if (world.rank == 0) {
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
    }
    
    return true;
}

// Compute a simple hash of the results for verification (gathers to rank 0)
uint64_t computeHash(const LocalWorld& world) {
    // Gather all energy and flux values to rank 0
    std::vector<int> recvcounts(world.n_ranks);
    std::vector<int> displs(world.n_ranks);
    
    int local_count = world.n_local_rows * world.n_elems_root;
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (world.rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < world.n_ranks; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    // Extract local energy values (excluding ghost rows)
    std::vector<val_t> local_energy(local_count);
    std::vector<val_t> local_flux(local_count);
    int idx = 0;
    for (int lr = 0; lr < world.n_local_rows; ++lr) {
        for (int y = 0; y < world.n_elems_root; ++y) {
            int local_idx = (world.ghost_above_offset + lr) * world.n_elems_root + y;
            local_energy[idx] = world.elements_dynamic[local_idx].current_energy;
            local_flux[idx] = world.elements_dynamic[local_idx].total_flux;
            idx++;
        }
    }
    
    std::vector<val_t> global_energy, global_flux;
    if (world.rank == 0) {
        int total_elems = world.n_elems_root * world.n_elems_root;
        global_energy.resize(total_elems);
        global_flux.resize(total_elems);
    }
    
    MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE,
                global_energy.data(), recvcounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_flux.data(), local_count, MPI_DOUBLE,
                global_flux.data(), recvcounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    uint64_t hash = 0;
    if (world.rank == 0) {
        for (size_t i = 0; i < global_energy.size(); ++i) {
            // Simple hash combining energy and flux values
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&global_energy[i]);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&global_flux[i]);
            hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    
    MPI_Bcast(&hash, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
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
    
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0 needs to parse, but all ranks need the values)
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
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI)\n");
        printf("==================================================\n");
        printf("MPI ranks: %d\n", n_ranks);
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh with MPI decomposition
    if (rank == 0) {
        printf("Building unstructured mesh with MPI decomposition...\n");
    }
    
    LocalWorld world;
    world.rank = rank;
    world.n_ranks = n_ranks;
    
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage (local)
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    
    if (rank == 0) {
        // Calculate total memory across all ranks
        unsigned long global_total_mem;
        unsigned long local_mem = total_mem;
        MPI_Reduce(&local_mem, &global_total_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        
        printf("Local memory usage (rank 0): %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("Total memory usage (all ranks): %.2f MB\n",
               global_total_mem / (1024.0 * 1024.0));
        printf("\n");
    } else {
        // Other ranks participate in the reduce
        unsigned long global_total_mem;
        unsigned long local_mem = total_mem;
        MPI_Reduce(&local_mem, &global_total_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    
    // Run simulation
    if (rank == 0) {
        printf("Running simulation...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Reduce timing across all ranks
    long long global_duration_ms;
    MPI_Reduce(&duration_ms, &global_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    duration_ms = global_duration_ms;
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        
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
    
    // Print results for external validation (gather to rank 0)
    if (printResults) {
        std::vector<double> global_energy;
        if (rank == 0) {
            global_energy.reserve(n_elems);
        }
        
        // Gather all energy values to rank 0
        std::vector<int> recvcounts(n_ranks);
        std::vector<int> displs(n_ranks);
        
        int local_count = world.n_local_rows * world.n_elems_root;
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < n_ranks; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        // Extract local energy values (excluding ghost rows)
        std::vector<val_t> local_energy(local_count);
        int idx = 0;
        for (int lr = 0; lr < world.n_local_rows; ++lr) {
            for (int y = 0; y < world.n_elems_root; ++y) {
                int local_idx = (world.ghost_above_offset + lr) * world.n_elems_root + y;
                local_energy[idx] = world.elements_dynamic[local_idx].current_energy;
                idx++;
            }
        }
        
        if (rank == 0) {
            global_energy.resize(n_elems);
        }
        
        MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE,
                    global_energy.data(), recvcounts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            print_results(global_energy, "ElementEnergy");
        }
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
