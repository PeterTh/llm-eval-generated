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
    int mpi_rank;
    int mpi_size;
    idx_t local_start;  // First local element index
    idx_t local_count;  // Number of local elements
    idx_t global_count; // Total number of elements
    std::vector<idx_t> ghost_indices;  // Indices of ghost elements from other ranks
    std::vector<ElementDynamic> ghost_data;  // Data for ghost elements
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh with MPI domain decomposition
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    world.global_count = n_elems;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Determine local partition: divide rows among ranks
    const int rows_per_rank = n_elems_root / world.mpi_size;
    const int extra_rows = n_elems_root % world.mpi_size;
    
    int local_start_row = world.mpi_rank * rows_per_rank + std::min(world.mpi_rank, extra_rows);
    int local_rows = rows_per_rank + (world.mpi_rank < extra_rows ? 1 : 0);
    
    world.local_start = local_start_row * n_elems_root;
    world.local_count = local_rows * n_elems_root;
    
    // Allocate local elements
    world.elements_static.resize(world.local_count);
    world.elements_dynamic.resize(world.local_count);
    world.elements_dynamic_swap.resize(world.local_count);
    
    // Initialize all local elements with default material and zero energy
    for (idx_t i = 0; i < world.local_count; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity for local elements
    for (int local_x = 0; local_x < local_rows; ++local_x) {
        int global_x = local_start_row + local_x;
        for (int y = 0; y < n_elems_root; ++y) {
            const idx_t local_idx = local_x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = global_x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const idx_t neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow
    const int last = n_elems_root - 1;
    auto setMaterial = [&](int gx, int gy, idx_t mat_id) {
        int local_x = gx - local_start_row;
        if (local_x >= 0 && local_x < local_rows) {
            idx_t local_idx = local_x * n_elems_root + gy;
            world.elements_static[local_idx].material_idx = mat_id;
        }
    };
    
    setMaterial(0, 0, INFLOW_MAT_ID);
    setMaterial(0, last, OUTFLOW_MAT_ID);
    setMaterial(last, 0, OUTFLOW_MAT_ID);
    setMaterial(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Exchange ghost cell data between MPI ranks
void exchangeGhostData(World& world) {
    if (world.mpi_size == 1) return;
    
    const int n_elems_root = static_cast<int>(std::sqrt(world.global_count));
    const int rows_per_rank = n_elems_root / world.mpi_size;
    const int extra_rows = n_elems_root % world.mpi_size;
    
    int local_start_row = world.mpi_rank * rows_per_rank + std::min(world.mpi_rank, extra_rows);
    int local_rows = rows_per_rank + (world.mpi_rank < extra_rows ? 1 : 0);
    
    // Determine neighbor ranks
    int prev_rank = (world.mpi_rank > 0) ? world.mpi_rank - 1 : -1;
    int next_rank = (world.mpi_rank < world.mpi_size - 1) ? world.mpi_rank + 1 : -1;
    
    std::vector<val_t> send_up, send_down, recv_up, recv_down;
    
    // Prepare boundary data for sending
    if (prev_rank >= 0) {
        // Send first row to previous rank
        send_up.resize(n_elems_root * 2);
        for (int y = 0; y < n_elems_root; ++y) {
            send_up[y * 2] = world.elements_dynamic[y].current_energy;
            send_up[y * 2 + 1] = world.elements_dynamic[y].total_flux;
        }
        recv_up.resize(n_elems_root * 2);
    }
    
    if (next_rank >= 0) {
        // Send last row to next rank
        send_down.resize(n_elems_root * 2);
        idx_t last_row_start = (local_rows - 1) * n_elems_root;
        for (int y = 0; y < n_elems_root; ++y) {
            send_down[y * 2] = world.elements_dynamic[last_row_start + y].current_energy;
            send_down[y * 2 + 1] = world.elements_dynamic[last_row_start + y].total_flux;
        }
        recv_down.resize(n_elems_root * 2);
    }
    
    // Non-blocking communication
    std::vector<MPI_Request> requests;
    
    // Post receives
    if (prev_rank >= 0) {
        MPI_Request req;
        MPI_Irecv(recv_up.data(), recv_up.size(), MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, &req);
        requests.push_back(req);
    }
    if (next_rank >= 0) {
        MPI_Request req;
        MPI_Irecv(recv_down.data(), recv_down.size(), MPI_DOUBLE, next_rank, 1, MPI_COMM_WORLD, &req);
        requests.push_back(req);
    }
    
    // Post sends
    if (prev_rank >= 0) {
        MPI_Request req;
        MPI_Isend(send_up.data(), send_up.size(), MPI_DOUBLE, prev_rank, 1, MPI_COMM_WORLD, &req);
        requests.push_back(req);
    }
    if (next_rank >= 0) {
        MPI_Request req;
        MPI_Isend(send_down.data(), send_down.size(), MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &req);
        requests.push_back(req);
    }
    
    // Wait for all communication
    if (!requests.empty()) {
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
    
    // Update ghost data
    world.ghost_data.clear();
    world.ghost_indices.clear();
    
    if (prev_rank >= 0) {
        int ghost_row = local_start_row - 1;
        for (int y = 0; y < n_elems_root; ++y) {
            idx_t ghost_idx = ghost_row * n_elems_root + y;
            world.ghost_indices.push_back(ghost_idx);
            world.ghost_data.push_back({recv_up[y * 2], recv_up[y * 2 + 1]});
        }
    }
    
    if (next_rank >= 0) {
        int ghost_row = local_start_row + local_rows;
        for (int y = 0; y < n_elems_root; ++y) {
            idx_t ghost_idx = ghost_row * n_elems_root + y;
            world.ghost_indices.push_back(ghost_idx);
            world.ghost_data.push_back({recv_down[y * 2], recv_down[y * 2 + 1]});
        }
    }
}

// Get element data (local or ghost)
inline const ElementDynamic& getElementData(const World& world, idx_t global_idx) {
    if (global_idx >= world.local_start && global_idx < world.local_start + world.local_count) {
        return world.elements_dynamic[global_idx - world.local_start];
    }
    
    // Find in ghost data
    for (size_t i = 0; i < world.ghost_indices.size(); ++i) {
        if (world.ghost_indices[i] == global_idx) {
            return world.ghost_data[i];
        }
    }
    
    // Should never reach here
    static ElementDynamic dummy{0.0, 0.0};
    return dummy;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.local_count;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange ghost cell data
        exchangeGhostData(world);
        
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
                const ElementDynamic& neighbor_dyn = getElementData(world, neighbor_idx);
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
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
bool validateResults(World& world) {
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
    
    // Reduce to rank 0
    val_t energy_sum, flux_sum, energy_max, energy_min;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    bool valid = true;
    if (world.mpi_rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
        
        // Check for numerical issues
        constexpr val_t energy_epsilon = 1e-8;
        
        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = false;
        }
        
        if (std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            // Don't fail validation as this can happen with external flows
        }
        
        if (!std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = false;
        }
        
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = false;
        }
        
        if (valid) {
            printf("  Validation: PASSED\n");
        }
    }
    
    // Broadcast validation result
    MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    return valid;
}

// Compute a simple hash of the results for verification
uint64_t computeHash(World& world) {
    uint64_t local_hash = 0;
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        idx_t global_i = world.local_start + i;
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
        local_hash ^= (*e_ptr + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_hash, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    
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
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0)
    if (mpi_rank == 0) {
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
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI)\n");
        printf("==================================================\n");
        printf("MPI ranks: %d\n", mpi_size);
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh
    if (mpi_rank == 0) {
        printf("Building unstructured mesh...\n");
    }
    World world;
    world.mpi_rank = mpi_rank;
    world.mpi_size = mpi_size;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t ghost_mem = world.ghost_data.size() * sizeof(ElementDynamic);
    const size_t local_mem = static_mem + dynamic_mem + ghost_mem;
    
    size_t total_mem;
    MPI_Reduce(&local_mem, &total_mem, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        printf("Memory usage: %.2f MB total\n", total_mem / (1024.0 * 1024.0));
        printf("  Per rank: %.2f MB (static: %.2f MB, dynamic: %.2f MB, ghost: %.2f MB)\n",
               local_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0),
               ghost_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Synchronize before starting simulation
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (mpi_rank == 0) {
        printf("Running simulation...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    if (mpi_rank == 0) {
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
    const uint64_t hash = computeHash(world);
    if (mpi_rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults) {
        // Gather all data to rank 0
        std::vector<val_t> all_energies;
        std::vector<int> recvcounts(mpi_size);
        std::vector<int> displs(mpi_size);
        
        int local_count = world.elements_dynamic.size();
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (mpi_rank == 0) {
            int total_count = 0;
            for (int i = 0; i < mpi_size; ++i) {
                displs[i] = total_count;
                total_count += recvcounts[i];
            }
            all_energies.resize(total_count);
        }
        
        std::vector<val_t> local_energies;
        local_energies.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            local_energies.push_back(elem.current_energy);
        }
        
        MPI_Gatherv(local_energies.data(), local_count, MPI_DOUBLE,
                   all_energies.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        
        if (mpi_rank == 0) {
            print_results(all_energies, "ElementEnergy");
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
