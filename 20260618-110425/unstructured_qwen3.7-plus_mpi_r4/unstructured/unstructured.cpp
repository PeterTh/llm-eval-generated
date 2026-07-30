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

// Build a 2D square grid as an unstructured mesh (distributed version)
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int size) {
    // Calculate row distribution across processes
    int rows_per_proc = n_elems_root / size;
    int remainder = n_elems_root % size;
    
    int local_start_row = rank * rows_per_proc + std::min(rank, remainder);
    int local_end_row = local_start_row + rows_per_proc + (rank < remainder ? 1 : 0);
    int local_n_rows = local_end_row - local_start_row;
    
    const int n_elems = local_n_rows * n_elems_root;
    
    // Initialize materials (same on all processes)
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
    for (int local_x = 0; local_x < local_n_rows; ++local_x) {
        const int x = local_start_row + local_x;
        for (int y = 0; y < n_elems_root; ++y) {
            const int local_idx = local_x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    // Store global neighbor index temporarily
                    const int global_neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = global_neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    if (rank == 0) {
        const int last = n_elems_root - 1;
        world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
        world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    }
    if (rank == size - 1) {
        const int last = n_elems_root - 1;
        const int last_local_row = local_n_rows - 1;
        world.elements_static[last_local_row * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[last_local_row * n_elems_root + last].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations (distributed version with ghost rows)
void runSimulation(World& world, const int n_iters, const int n_elems_root, const int rank, const int size) {
    const size_t n_local_elems = world.elements_static.size();
    const int local_n_rows = n_local_elems / n_elems_root;
    
    // Calculate row distribution
    int rows_per_proc = n_elems_root / size;
    int remainder = n_elems_root % size;
    int local_start_row = rank * rows_per_proc + std::min(rank, remainder);
    int local_end_row = local_start_row + local_n_rows;
    
    // Determine neighbor ranks
    int up_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    
    // Allocate ghost row buffers
    std::vector<ElementDynamic> ghost_up(n_elems_root);
    std::vector<ElementDynamic> ghost_down(n_elems_root);
    
    // Convert global neighbor indices to local indices
    // Layout: [local elements][ghost_up][ghost_down]
    const size_t ghost_up_offset = n_local_elems;
    const size_t ghost_down_offset = n_local_elems + n_elems_root;
    
    for (size_t i = 0; i < n_local_elems; ++i) {
        ElementStatic& elem = world.elements_static[i];
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            idx_t global_idx = elem.connected_idx[j];
            int global_row = global_idx / n_elems_root;
            int col = global_idx % n_elems_root;
            
            if (global_row >= local_start_row && global_row < local_end_row) {
                // Local neighbor
                int local_row = global_row - local_start_row;
                elem.connected_idx[j] = local_row * n_elems_root + col;
            } else if (global_row == local_start_row - 1) {
                // Up ghost row
                elem.connected_idx[j] = ghost_up_offset + col;
            } else if (global_row == local_end_row) {
                // Down ghost row
                elem.connected_idx[j] = ghost_down_offset + col;
            }
        }
    }
    
    // Extended array for ghost rows
    std::vector<ElementDynamic> extended_dynamic(n_local_elems + 2 * n_elems_root);
    std::vector<ElementDynamic> extended_swap(n_local_elems + 2 * n_elems_root);
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Copy local data to extended array
        std::copy(world.elements_dynamic.begin(), world.elements_dynamic.end(), extended_dynamic.begin());
        
        // Exchange ghost rows using non-blocking communication
        MPI_Request requests[4];
        int num_requests = 0;
        
        // Send bottom row to down_rank, receive ghost from down_rank
        if (down_rank != MPI_PROC_NULL) {
            MPI_Isend(&world.elements_dynamic[n_local_elems - n_elems_root], 
                     n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
                     down_rank, 0, MPI_COMM_WORLD, &requests[num_requests++]);
            MPI_Irecv(&ghost_down[0], 
                     n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
                     down_rank, 1, MPI_COMM_WORLD, &requests[num_requests++]);
        }
        
        // Send top row to up_rank, receive ghost from up_rank
        if (up_rank != MPI_PROC_NULL) {
            MPI_Isend(&world.elements_dynamic[0], 
                     n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
                     up_rank, 1, MPI_COMM_WORLD, &requests[num_requests++]);
            MPI_Irecv(&ghost_up[0], 
                     n_elems_root * sizeof(ElementDynamic), MPI_BYTE,
                     up_rank, 0, MPI_COMM_WORLD, &requests[num_requests++]);
        }
        
        // Wait for all communication to complete
        MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
        
        // Copy ghost data to extended array
        std::copy(ghost_up.begin(), ghost_up.end(), extended_dynamic.begin() + ghost_up_offset);
        std::copy(ghost_down.begin(), ghost_down.end(), extended_dynamic.begin() + ghost_down_offset);
        
        // Update all local elements
        for (size_t i = 0; i < n_local_elems; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = extended_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic& neighbor_dyn = extended_dynamic[neighbor_idx];
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
            printUsage(argv[0]);
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
    
    // Build the unstructured mesh (distributed)
    if (rank == 0) {
        printf("Building distributed unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, rank, size);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) {
        printf("Memory usage per rank: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation
    if (rank == 0) {
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters, n_elems_root, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Compute local duration and get global max
    double local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double max_duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Gather all energy values to rank 0 for validation and results
    const int local_n_elems = world.elements_dynamic.size();
    
    // Gather counts for gatherv
    int* recv_counts = nullptr;
    int* displs = nullptr;
    if (rank == 0) {
        recv_counts = new int[size];
        displs = new int[size];
    }
    int send_count = local_n_elems;
    MPI_Gather(&send_count, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i-1] + recv_counts[i-1];
        }
    }
    
    // Gather energy data
    std::vector<double> all_energy;
    std::vector<double> all_flux;
    if (rank == 0) {
        all_energy.resize(n_elems);
        all_flux.resize(n_elems);
    }
    
    // Pack local data
    std::vector<double> local_energy(local_n_elems);
    std::vector<double> local_flux(local_n_elems);
    for (int i = 0; i < local_n_elems; ++i) {
        local_energy[i] = world.elements_dynamic[i].current_energy;
        local_flux[i] = world.elements_dynamic[i].total_flux;
    }
    
    MPI_Gatherv(local_energy.data(), local_n_elems, MPI_DOUBLE,
                all_energy.data(), recv_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_flux.data(), local_n_elems, MPI_DOUBLE,
                all_flux.data(), recv_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", max_duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = max_duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (max_duration_ms / 1000.0) / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        
        // Compute hash for verification using gathered data
        std::vector<ElementDynamic> gathered_elements(n_elems);
        for (int i = 0; i < n_elems; ++i) {
            gathered_elements[i].current_energy = all_energy[i];
            gathered_elements[i].total_flux = all_flux[i];
        }
        const uint64_t hash = computeHash(gathered_elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
        
        // Print results for external validation
        if (printResults) {
            print_results(all_energy, "ElementEnergy");
        }
        
        // Validation
        if (validate) {
            val_t energy_sum = 0.0;
            val_t flux_sum = 0.0;
            val_t energy_max = std::numeric_limits<val_t>::lowest();
            val_t energy_min = std::numeric_limits<val_t>::max();
            
            for (int i = 0; i < n_elems; ++i) {
                energy_sum += all_energy[i];
                flux_sum += all_flux[i];
                energy_max = std::max(all_energy[i], energy_max);
                energy_min = std::min(all_energy[i], energy_min);
            }
            
            printf("Validation results:\n");
            printf("  Energy sum: %.12f\n", energy_sum);
            printf("  Flux sum: %.2f\n", flux_sum);
            printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
            
            constexpr val_t energy_epsilon = 1e-8;
            bool valid = true;
            
            if (!std::isfinite(energy_sum)) {
                printf("  ERROR: Energy sum is not finite\n");
                valid = false;
            }
            if (std::abs(energy_sum) > energy_epsilon) {
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
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
        
        delete[] recv_counts;
        delete[] displs;
    }
    
    MPI_Finalize();
    return 0;
}
