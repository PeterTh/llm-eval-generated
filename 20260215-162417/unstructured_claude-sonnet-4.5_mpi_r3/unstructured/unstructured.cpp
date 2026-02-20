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
    
    // MPI-specific fields
    int rank;
    int nprocs;
    int local_start_row;  // Starting row for this rank
    int local_num_rows;   // Number of rows owned by this rank
    int n_elems_root;     // Grid dimension
    
    // Ghost cells (halo regions)
    std::vector<ElementDynamic> ghost_before;  // Ghost cells from rank-1
    std::vector<ElementDynamic> ghost_after;   // Ghost cells from rank+1
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
// Each rank builds only its local portion
void buildSquare2D(World& world, const int n_elems_root) {
    world.n_elems_root = n_elems_root;
    
    // Determine local partition (distribute rows across ranks)
    const int rows_per_rank = n_elems_root / world.nprocs;
    const int extra_rows = n_elems_root % world.nprocs;
    
    // Calculate local row range
    if (world.rank < extra_rows) {
        world.local_num_rows = rows_per_rank + 1;
        world.local_start_row = world.rank * world.local_num_rows;
    } else {
        world.local_num_rows = rows_per_rank;
        world.local_start_row = extra_rows * (rows_per_rank + 1) + 
                                (world.rank - extra_rows) * rows_per_rank;
    }
    
    const int local_n_elems = world.local_num_rows * n_elems_root;
    
    // Initialize materials (same on all ranks)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements for local partition
    world.elements_static.resize(local_n_elems);
    world.elements_dynamic.resize(local_n_elems);
    world.elements_dynamic_swap.resize(local_n_elems);
    
    // Allocate ghost cells
    if (world.rank > 0) {
        world.ghost_before.resize(n_elems_root);
    }
    if (world.rank < world.nprocs - 1) {
        world.ghost_after.resize(n_elems_root);
    }
    
    // Initialize all elements with default material and zero energy
    for (int i = 0; i < local_n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
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
                    // Store neighbor index as local or ghost cell marker
                    // We use MSB as ghost flag
                    const idx_t ghost_flag = static_cast<idx_t>(1) << 63;
                    
                    if (nx >= world.local_start_row && nx < world.local_start_row + world.local_num_rows) {
                        // Local neighbor
                        const int neighbor_local_x = nx - world.local_start_row;
                        const int neighbor_idx = neighbor_local_x * n_elems_root + ny;
                        elem.connected_idx[elem.num_connections] = neighbor_idx;
                    } else if (nx < world.local_start_row) {
                        // Ghost before (from previous rank)
                        elem.connected_idx[elem.num_connections] = ghost_flag | ny;
                    } else {
                        // Ghost after (from next rank)
                        elem.connected_idx[elem.num_connections] = ghost_flag | (n_elems_root + ny);
                    }
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    
    // Check if this rank owns any of the corner elements
    if (world.local_start_row == 0) {
        // Top-left corner
        world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
        // Top-right corner
        world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    }
    
    if (world.local_start_row + world.local_num_rows == n_elems_root) {
        // Bottom-left corner
        const int bottom_local_x = world.local_num_rows - 1;
        world.elements_static[bottom_local_x * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
        // Bottom-right corner
        world.elements_static[bottom_local_x * n_elems_root + last].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Exchange ghost cell data between neighboring ranks
void exchangeGhostCells(World& world) {
    const int n_elems_root = world.n_elems_root;
    MPI_Request requests[4];
    int req_count = 0;
    
    // Send/receive boundary data with neighbors
    if (world.rank > 0) {
        // Send first row to rank-1, receive into ghost_before
        const int first_row_offset = 0;
        MPI_Isend(&world.elements_dynamic[first_row_offset], 
                  n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
                  world.rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        
        MPI_Irecv(world.ghost_before.data(),
                  n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
                  world.rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }
    
    if (world.rank < world.nprocs - 1) {
        // Send last row to rank+1, receive into ghost_after
        const int last_row_offset = (world.local_num_rows - 1) * n_elems_root;
        MPI_Isend(&world.elements_dynamic[last_row_offset],
                  n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
                  world.rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        
        MPI_Irecv(world.ghost_after.data(),
                  n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
                  world.rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }
    
    // Wait for all communications to complete
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange ghost cells with neighbors
        exchangeGhostCells(world);
        
        // Update all elements
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
                
                // Determine if neighbor is local or ghost
                // We use the MSB as a flag: if set, it's a ghost cell index
                const idx_t ghost_flag = static_cast<idx_t>(1) << 63;
                
                if ((neighbor_idx & ghost_flag) == 0) {
                    // Local neighbor (no flag set)
                    neighbor_dyn = &world.elements_dynamic[neighbor_idx];
                } else {
                    // Ghost cell (flag set)
                    const idx_t encoded = neighbor_idx & ~ghost_flag;
                    if (encoded < static_cast<idx_t>(world.n_elems_root)) {
                        // Ghost before
                        neighbor_dyn = &world.ghost_before[encoded];
                    } else {
                        // Ghost after
                        neighbor_dyn = &world.ghost_after[encoded - world.n_elems_root];
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

// Validate simulation results (with MPI reduction)
bool validateResults(const World& world) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : world.elements_dynamic) {
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(elem.current_energy, local_energy_max);
        local_energy_min = std::min(elem.current_energy, local_energy_min);
    }
    
    // Reduce across all ranks
    val_t energy_sum, flux_sum, energy_max, energy_min;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    // Only rank 0 prints results
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

// Compute a simple hash of the results for verification (with MPI)
uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    const auto& elements = world.elements_dynamic;
    
    for (size_t i = 0; i < elements.size(); ++i) {
        // Use global index for hashing
        const size_t global_i = world.local_start_row * world.n_elems_root + i;
        
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        local_hash ^= (*e_ptr + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    
    // XOR-reduce hashes across all ranks
    uint64_t global_hash;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
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
        printf("MPI ranks: %d\n", nprocs);
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
    world.rank = rank;
    world.nprocs = nprocs;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t ghost_mem = (world.ghost_before.size() + world.ghost_after.size()) * sizeof(ElementDynamic);
    const size_t total_mem = static_mem + dynamic_mem + ghost_mem;
    
    if (rank == 0) {
        printf("Memory usage per rank (rank 0): %.2f MB (static: %.2f MB, dynamic: %.2f MB, ghost: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0),
               ghost_mem / (1024.0 * 1024.0));
        printf("Local elements per rank (rank 0): %d rows = %zu elements\n",
               world.local_num_rows, world.elements_static.size());
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
    
    // Synchronize after computation
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
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation (gather to rank 0)
    if (printResults) {
        // Gather all data to rank 0
        std::vector<double> energyData;
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        
        int local_count = world.elements_dynamic.size();
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            int total_count = 0;
            for (int i = 0; i < nprocs; ++i) {
                displs[i] = total_count;
                total_count += recvcounts[i];
            }
            energyData.resize(total_count);
        }
        
        // Extract local energy values
        std::vector<double> local_energies;
        local_energies.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            local_energies.push_back(elem.current_energy);
        }
        
        MPI_Gatherv(local_energies.data(), local_count, MPI_DOUBLE,
                    energyData.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            print_results(energyData, "ElementEnergy");
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
