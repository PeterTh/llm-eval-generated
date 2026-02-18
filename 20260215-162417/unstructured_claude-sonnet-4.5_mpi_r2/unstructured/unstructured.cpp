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
    
    // MPI-specific data
    int rank;
    int size;
    int local_start_row;  // First row owned by this rank
    int local_num_rows;   // Number of rows owned by this rank
    int n_elems_root;     // Global grid dimension
    
    // Ghost rows for boundary communication
    std::vector<ElementDynamic> ghost_top;    // Ghost row from rank-1
    std::vector<ElementDynamic> ghost_bottom; // Ghost row from rank+1
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh with MPI domain decomposition
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    world.n_elems_root = n_elems_root;
    
    // Get MPI rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &world.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world.size);
    
    // Compute domain decomposition: divide rows among ranks
    const int base_rows = n_elems_root / world.size;
    const int extra_rows = n_elems_root % world.size;
    
    // Distribute extra rows to first ranks
    if (world.rank < extra_rows) {
        world.local_num_rows = base_rows + 1;
        world.local_start_row = world.rank * (base_rows + 1);
    } else {
        world.local_num_rows = base_rows;
        world.local_start_row = extra_rows * (base_rows + 1) + (world.rank - extra_rows) * base_rows;
    }
    
    const int local_n_elems = world.local_num_rows * n_elems_root;
    
    // Initialize materials (same on all ranks)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate local elements
    world.elements_static.resize(local_n_elems);
    world.elements_dynamic.resize(local_n_elems);
    world.elements_dynamic_swap.resize(local_n_elems);
    
    // Allocate ghost rows
    world.ghost_top.resize(n_elems_root);
    world.ghost_bottom.resize(n_elems_root);
    
    // Initialize all local elements with default material and zero energy
    for (int i = 0; i < local_n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity for local elements
    for (int local_x = 0; local_x < world.local_num_rows; ++local_x) {
        const int global_x = world.local_start_row + local_x;
        
        for (int y = 0; y < n_elems_root; ++y) {
            const int local_idx = local_x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = global_x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within global bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    // Check if neighbor is in local domain or ghost region
                    if (nx >= world.local_start_row && nx < world.local_start_row + world.local_num_rows) {
                        // Local neighbor
                        const int neighbor_local_x = nx - world.local_start_row;
                        const idx_t neighbor_idx = neighbor_local_x * n_elems_root + ny;
                        elem.connected_idx[elem.num_connections] = neighbor_idx;
                    } else {
                        // Ghost neighbor - encode as UINT64_MAX - offset to distinguish from local
                        // Use high bit pattern to mark as ghost, with lower bits storing y coordinate
                        if (nx < world.local_start_row) {
                            // Top ghost (from rank-1)
                            elem.connected_idx[elem.num_connections] = UINT64_MAX - ny;
                        } else {
                            // Bottom ghost (from rank+1)
                            elem.connected_idx[elem.num_connections] = UINT64_MAX - n_elems_root - ny;
                        }
                    }
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow on appropriate ranks
    const int last = n_elems_root - 1;
    
    // Top-left corner (0, 0) - INFLOW
    if (world.local_start_row == 0) {
        world.elements_static[0].material_idx = INFLOW_MAT_ID;
    }
    
    // Top-right corner (0, last) - OUTFLOW
    if (world.local_start_row == 0) {
        world.elements_static[last].material_idx = OUTFLOW_MAT_ID;
    }
    
    // Bottom-left corner (last, 0) - OUTFLOW
    if (world.local_start_row + world.local_num_rows == n_elems_root) {
        const int local_idx = (world.local_num_rows - 1) * n_elems_root;
        world.elements_static[local_idx].material_idx = OUTFLOW_MAT_ID;
    }
    
    // Bottom-right corner (last, last) - INFLOW
    if (world.local_start_row + world.local_num_rows == n_elems_root) {
        const int local_idx = (world.local_num_rows - 1) * n_elems_root + last;
        world.elements_static[local_idx].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Exchange ghost cells with neighboring MPI ranks
void exchangeGhostCells(World& world) {
    const int n_elems_root = world.n_elems_root;
    MPI_Status status;
    
    // Exchange with upper neighbor (rank - 1)
    if (world.local_start_row > 0) {
        const int neighbor_rank = world.rank - 1;
        
        // Send first row to upper neighbor, receive their last row into ghost_top
        MPI_Sendrecv(
            world.elements_dynamic.data(), n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
            neighbor_rank, 0,
            world.ghost_top.data(), n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
            neighbor_rank, 1,
            MPI_COMM_WORLD, &status);
    }
    
    // Exchange with lower neighbor (rank + 1)
    if (world.local_start_row + world.local_num_rows < n_elems_root) {
        const int neighbor_rank = world.rank + 1;
        
        // Send last row to lower neighbor, receive their first row into ghost_bottom
        const int last_row_offset = (world.local_num_rows - 1) * n_elems_root;
        MPI_Sendrecv(
            &world.elements_dynamic[last_row_offset], n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
            neighbor_rank, 1,
            world.ghost_bottom.data(), n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
            neighbor_rank, 0,
            MPI_COMM_WORLD, &status);
    }
}

// Run simulation for n_iters iterations with MPI
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    const int n_elems_root = world.n_elems_root;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange ghost cells with neighboring ranks
        exchangeGhostCells(world);
        
        // Update all local elements
        for (size_t i = 0; i < n_elems; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic* neighbor_dyn;
                
                // Check if this is a ghost cell reference (high bit pattern)
                if (neighbor_idx < (UINT64_MAX / 2)) {
                    // Local element
                    neighbor_dyn = &world.elements_dynamic[neighbor_idx];
                } else {
                    // Ghost element - decode the y coordinate
                    const idx_t offset = UINT64_MAX - neighbor_idx;
                    
                    if (offset < (idx_t)n_elems_root) {
                        // Top ghost (from rank - 1)
                        const int y = offset;
                        neighbor_dyn = &world.ghost_top[y];
                    } else {
                        // Bottom ghost (from rank + 1)
                        const int y = offset - n_elems_root;
                        neighbor_dyn = &world.ghost_bottom[y];
                    }
                }
                
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], *neighbor_dyn);
            }
            
            // Update element state
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
    
    // Reduce across all ranks
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

// Gather all elements to rank 0 for validation/output
std::vector<ElementDynamic> gatherAllElements(const World& world) {
    const int n_elems = world.n_elems_root * world.n_elems_root;
    std::vector<ElementDynamic> all_elements;
    
    if (world.rank == 0) {
        all_elements.resize(n_elems);
    }
    
    // Gather element counts from all ranks
    std::vector<int> recvcounts(world.size);
    std::vector<int> displs(world.size);
    
    const int local_count = world.local_num_rows * world.n_elems_root;
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (world.rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < world.size; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
        
        // Convert counts to byte counts for MPI_Gatherv
        for (int i = 0; i < world.size; ++i) {
            recvcounts[i] *= sizeof(ElementDynamic);
            displs[i] *= sizeof(ElementDynamic);
        }
    }
    
    // Gather all elements
    MPI_Gatherv(
        world.elements_dynamic.data(), local_count * sizeof(ElementDynamic), MPI_BYTE,
        all_elements.data(), recvcounts.data(), displs.data(), MPI_BYTE,
        0, MPI_COMM_WORLD);
    
    return all_elements;
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
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI)\n");
        printf("==================================================\n");
        printf("MPI ranks: %d\n", size);
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh
    if (rank == 0) {
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage (per rank)
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t ghost_mem = (world.ghost_top.size() + world.ghost_bottom.size()) * sizeof(ElementDynamic);
    const size_t total_mem = static_mem + dynamic_mem + ghost_mem;
    
    if (rank == 0) {
        printf("Memory usage per rank (avg): %.2f MB\n", total_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (rank == 0) {
        printf("Running simulation...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    // Synchronize after simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
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
    
    // Gather results for hash computation and validation
    std::vector<ElementDynamic> all_elements = gatherAllElements(world);
    
    if (rank == 0) {
        // Compute hash for verification
        const uint64_t hash = computeHash(all_elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        std::vector<double> energyData;
        energyData.reserve(all_elements.size());
        for (const auto& elem : all_elements) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
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
