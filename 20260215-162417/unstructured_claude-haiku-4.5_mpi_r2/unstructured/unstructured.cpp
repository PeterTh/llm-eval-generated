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

// MPI parallelization state - extended
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    
    // Ghost cells for boundary rows from neighboring processes
    std::vector<ElementDynamic> ghost_cells_top;
    std::vector<ElementDynamic> ghost_cells_bottom;
    
    // MPI parallelization state
    int mpi_rank = 0;
    int mpi_size = 1;
    int local_start_row = 0;    // First row this rank owns
    int local_end_row = 0;      // Last row this rank owns (exclusive)
    int n_elems_root = 0;       // Global grid dimension
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    world.n_elems_root = n_elems_root;
    
    // Initialize materials (on all ranks)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Compute local domain partition: 1D decomposition by rows
    int rows_per_rank = (n_elems_root + world.mpi_size - 1) / world.mpi_size;
    world.local_start_row = world.mpi_rank * rows_per_rank;
    world.local_end_row = std::min((world.mpi_rank + 1) * rows_per_rank, n_elems_root);
    
    // Only create local elements within this rank's domain
    const int local_rows = world.local_end_row - world.local_start_row;
    const int local_n_elems = local_rows * n_elems_root;
    
    world.elements_static.resize(local_n_elems);
    world.elements_dynamic.resize(local_n_elems);
    world.elements_dynamic_swap.resize(local_n_elems);
    
    // Initialize ghost cells for boundary rows
    world.ghost_cells_top.resize(n_elems_root);
    world.ghost_cells_bottom.resize(n_elems_root);
    
    // Initialize local elements with default material and zero energy
    for (int i = 0; i < local_n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Initialize ghost cells
    for (int y = 0; y < n_elems_root; ++y) {
        world.ghost_cells_top[y].current_energy = 0.0;
        world.ghost_cells_top[y].total_flux = 0.0;
        world.ghost_cells_bottom[y].current_energy = 0.0;
        world.ghost_cells_bottom[y].total_flux = 0.0;
    }
    
    // Build connectivity for local elements
    for (int x = world.local_start_row; x < world.local_end_row; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            // Convert global indices to local storage indices
            const int local_x = x - world.local_start_row;
            const int local_idx = local_x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within global bounds
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
    
    // Check if corners are in this rank's domain
    if (world.local_start_row <= 0 && 0 < world.local_end_row) {
        world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
        world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    }
    if (world.local_start_row <= last && last < world.local_end_row) {
        const int local_x = last - world.local_start_row;
        world.elements_static[local_x * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[local_x * n_elems_root + last].material_idx = INFLOW_MAT_ID;
    }
}

// Helper function to get local index from global element index
inline int getLocalIndex(const World& world, idx_t global_idx) {
    const int global_x = global_idx / world.n_elems_root;
    const int global_y = global_idx % world.n_elems_root;
    
    // Check if element is in local domain
    if (global_x >= world.local_start_row && global_x < world.local_end_row) {
        const int local_x = global_x - world.local_start_row;
        return local_x * world.n_elems_root + global_y;
    }
    return -1;  // Not in local domain
}

// Exchange ghost cells between processes for a row
void exchangeGhostCells(World& world) {
    if (world.mpi_size == 1) return;  // No communication needed for single process
    
    const int n_elems_root = world.n_elems_root;
    const int local_rows = world.local_end_row - world.local_start_row;
    
    // Create buffers for boundary exchange (one row = n_elems_root elements)
    std::vector<double> top_send, top_recv(n_elems_root);
    std::vector<double> bottom_send, bottom_recv(n_elems_root);
    
    if (local_rows > 0) {
        // Pack top boundary row for sending upward (current_energy values)
        top_send.resize(n_elems_root);
        for (int y = 0; y < n_elems_root; ++y) {
            top_send[y] = world.elements_dynamic[0 * n_elems_root + y].current_energy;
        }
        
        // Pack bottom boundary row for sending downward (current_energy values)
        bottom_send.resize(n_elems_root);
        for (int y = 0; y < n_elems_root; ++y) {
            bottom_send[y] = world.elements_dynamic[(local_rows - 1) * n_elems_root + y].current_energy;
        }
    } else {
        top_send.resize(n_elems_root, 0.0);
        bottom_send.resize(n_elems_root, 0.0);
    }
    
    // Exchange with neighbors
    int prev_rank = (world.mpi_rank - 1 + world.mpi_size) % world.mpi_size;
    int next_rank = (world.mpi_rank + 1) % world.mpi_size;
    
    // Use MPI_Sendrecv for bidirectional communication
    MPI_Sendrecv(bottom_send.data(), n_elems_root, MPI_DOUBLE, 
                 next_rank, 0, 
                 top_recv.data(), n_elems_root, MPI_DOUBLE,
                 prev_rank, 0, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    MPI_Sendrecv(top_send.data(), n_elems_root, MPI_DOUBLE,
                 prev_rank, 1,
                 bottom_recv.data(), n_elems_root, MPI_DOUBLE,
                 next_rank, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Unpack received ghost cells
    for (int y = 0; y < n_elems_root; ++y) {
        world.ghost_cells_top[y].current_energy = top_recv[y];
        world.ghost_cells_bottom[y].current_energy = bottom_recv[y];
    }
}

// Compute flux handling ghost cells and remote elements
inline val_t computeFluxMPI(const World& world, const Material& mat, 
                           const ElementDynamic& this_elem,
                           val_t connection_flux, idx_t neighbor_idx) {
    int local_neighbor = getLocalIndex(world, neighbor_idx);
    
    if (local_neighbor >= 0) {
        // Neighbor is in local domain
        const ElementDynamic& other_elem = world.elements_dynamic[local_neighbor];
        return (other_elem.current_energy - this_elem.current_energy) * 
               mat.transfer_coeff * connection_flux * 0.25;
    }
    
    // Neighbor is in ghost cells
    const int global_x = neighbor_idx / world.n_elems_root;
    const int global_y = neighbor_idx % world.n_elems_root;
    
    if (global_x < world.local_start_row) {
        // Neighbor is above (in ghost_cells_top)
        return (world.ghost_cells_top[global_y].current_energy - this_elem.current_energy) * 
               mat.transfer_coeff * connection_flux * 0.25;
    } else if (global_x >= world.local_end_row) {
        // Neighbor is below (in ghost_cells_bottom)
        return (world.ghost_cells_bottom[global_y].current_energy - this_elem.current_energy) * 
               mat.transfer_coeff * connection_flux * 0.25;
    }
    
    // Should not reach here
    return 0.0;
}

// Run simulation for n_iters iterations with MPI parallelization
void runSimulation(World& world, const int n_iters) {
    const size_t local_n_elems = world.elements_dynamic.size();
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange ghost cells before each iteration
        exchangeGhostCells(world);
        
        // Update all local elements
        for (size_t i = 0; i < local_n_elems; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                total_flux += computeFluxMPI(world, mat, elem_dyn, 
                                            elem_static.connected_flux[j], neighbor_idx);
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

// Validate simulation results (aggregated across all MPI processes)
bool validateResults(const World& world) {
    const auto& local_elements = world.elements_dynamic;
    
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : local_elements) {
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(elem.current_energy, local_energy_max);
        local_energy_min = std::min(elem.current_energy, local_energy_min);
    }
    
    // Aggregate across all processes
    val_t global_energy_sum, global_flux_sum, global_energy_max, global_energy_min;
    
    MPI_Reduce(&local_energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    // Only rank 0 prints validation
    if (world.mpi_rank != 0) return true;
    
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
    return true;
}

// Compute a simple hash of the results for verification (aggregated across processes)
uint64_t computeHash(const std::vector<ElementDynamic>& elements, const World& world) {
    uint64_t local_hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        local_hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    
    // Aggregate hash across processes using XOR
    uint64_t global_hash;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    
    if (world.mpi_rank == 0) {
        return global_hash;
    }
    return 0;  // Non-zero ranks don't care about the hash
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
    
    // Parse command line arguments (on all ranks for consistency)
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
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    // Only rank 0 prints output
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    world.mpi_rank = rank;
    world.mpi_size = size;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage (on rank 0)
    if (rank == 0) {
        const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic) * size;
        const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2 * size;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    // Synchronize after computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Only rank 0 prints performance metrics
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world.elements_dynamic, world);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Gather all element data on rank 0 for results printing
    if (printResults && rank == 0) {
        std::vector<double> energyData;
        energyData.reserve(n_elems);
        
        // Receive data from all ranks
        for (int r = 0; r < size; ++r) {
            if (r == 0) {
                // Local data
                for (const auto& elem : world.elements_dynamic) {
                    energyData.push_back(elem.current_energy);
                }
            } else {
                // Receive from other ranks
                int rows_per_rank = (n_elems_root + size - 1) / size;
                int start_row = r * rows_per_rank;
                int end_row = std::min((r + 1) * rows_per_rank, n_elems_root);
                int local_rows = end_row - start_row;
                int n_recv = local_rows * n_elems_root;
                
                std::vector<double> recv_data(n_recv);
                MPI_Recv(recv_data.data(), n_recv, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                for (const auto& val : recv_data) {
                    energyData.push_back(val);
                }
            }
        }
        
        print_results(energyData, "ElementEnergy");
    } else if (printResults && rank != 0) {
        // Send local data to rank 0
        std::vector<double> local_energies;
        for (const auto& elem : world.elements_dynamic) {
            local_energies.push_back(elem.current_energy);
        }
        if (!local_energies.empty()) {
            MPI_Send(local_energies.data(), local_energies.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
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
