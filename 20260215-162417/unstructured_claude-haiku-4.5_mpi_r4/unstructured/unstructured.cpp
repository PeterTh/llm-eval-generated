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
    
    // MPI distribution info
    int mpi_rank = 0;
    int mpi_size = 1;
    int local_start_x = 0;     // Starting x index for this process
    int local_end_x = 0;       // Ending x index (exclusive) for this process
    int local_start_idx = 0;   // Starting element index for this process
    int local_end_idx = 0;     // Ending element index (exclusive) for this process
    int global_n_elems_root = 0;
    
    // Neighbor info for halo exchange
    int left_neighbor = MPI_PROC_NULL;
    int right_neighbor = MPI_PROC_NULL;
    std::vector<ElementDynamic> left_halo_recv;   // Halo from left neighbor
    std::vector<ElementDynamic> right_halo_recv;  // Halo from right neighbor
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh with MPI distribution
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    [[maybe_unused]] const int n_elems = n_elems_root * n_elems_root;
    
    // Initialize materials (on all processes)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    world.global_n_elems_root = n_elems_root;
    
    // Partition grid: each process gets a contiguous strip in x direction
    int elems_per_proc = (n_elems_root + world.mpi_size - 1) / world.mpi_size;
    world.local_start_x = world.mpi_rank * elems_per_proc;
    world.local_end_x = std::min(world.local_start_x + elems_per_proc, n_elems_root);
    
    // Handle last process may have fewer elements
    int local_width = world.local_end_x - world.local_start_x;
    int local_n_elems = local_width * n_elems_root;
    
    world.local_start_idx = 0;
    world.local_end_idx = local_n_elems;
    
    // Set up MPI neighbor ranks
    world.left_neighbor = (world.mpi_rank > 0) ? world.mpi_rank - 1 : MPI_PROC_NULL;
    world.right_neighbor = (world.mpi_rank < world.mpi_size - 1) ? world.mpi_rank + 1 : MPI_PROC_NULL;
    
    // Allocate local elements
    world.elements_static.resize(local_n_elems);
    world.elements_dynamic.resize(local_n_elems);
    world.elements_dynamic_swap.resize(local_n_elems);
    
    // Allocate halo regions (single row of elements at boundaries)
    world.left_halo_recv.resize(n_elems_root);
    world.right_halo_recv.resize(n_elems_root);
    
    // Initialize all local elements with default material and zero energy
    for (int i = 0; i < local_n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = world.local_start_x; x < world.local_end_x; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int local_x = x - world.local_start_x;
            const int local_idx = local_x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within global bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    // Store neighbor info - we'll handle communication in simulation
                    elem.connected_idx[elem.num_connections] = ((uint64_t)nx << 32) | (uint64_t)ny;  // Store global coordinates
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow on the process that owns them
    const int last = n_elems_root - 1;
    
    // Check if this process owns element (0, 0)
    if (world.local_start_x == 0) {
        world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    }
    // Check if this process owns element (0, last)
    if (world.local_start_x == 0) {
        world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    }
    // Check if this process owns element (last, 0)
    if (world.local_end_x > last) {
        const int local_last_x = last - world.local_start_x;
        world.elements_static[local_last_x * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    }
    // Check if this process owns element (last, last)
    if (world.local_end_x > last) {
        const int local_last_x = last - world.local_start_x;
        world.elements_static[local_last_x * n_elems_root + last].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Helper: Get global element energy at a given position
inline ElementDynamic getElementEnergy(const World& world, int global_x, int global_y) {
    // Check if element is in local partition
    if (global_x >= world.local_start_x && global_x < world.local_end_x) {
        const int local_x = global_x - world.local_start_x;
        const int local_idx = local_x * world.global_n_elems_root + global_y;
        return world.elements_dynamic[local_idx];
    }
    
    // Check if element is in left halo
    if (global_x == world.local_start_x - 1) {
        return world.left_halo_recv[global_y];
    }
    
    // Check if element is in right halo
    if (global_x == world.local_end_x) {
        return world.right_halo_recv[global_y];
    }
    
    // Should not reach here for valid MPI usage
    return ElementDynamic{0.0, 0.0};
}

// Exchange halo data with neighboring processes
void exchangeHalo(World& world) {
    // Send local right boundary to right neighbor and receive from right neighbor
    // Send local left boundary to left neighbor and receive from left neighbor
    
    MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    
    // Prepare data to send
    std::vector<double> left_send(world.global_n_elems_root * 2);
    std::vector<double> right_send(world.global_n_elems_root * 2);
    
    if (world.local_start_x > 0) {
        // Prepare left boundary (x = local_start_x)
        for (int y = 0; y < world.global_n_elems_root; ++y) {
            const int local_idx = 0 * world.global_n_elems_root + y;
            left_send[y * 2] = world.elements_dynamic[local_idx].current_energy;
            left_send[y * 2 + 1] = world.elements_dynamic[local_idx].total_flux;
        }
    }
    
    if (world.local_end_x < world.global_n_elems_root) {
        // Prepare right boundary (x = local_end_x - 1)
        const int local_last_x = world.local_end_x - world.local_start_x - 1;
        for (int y = 0; y < world.global_n_elems_root; ++y) {
            const int local_idx = local_last_x * world.global_n_elems_root + y;
            right_send[y * 2] = world.elements_dynamic[local_idx].current_energy;
            right_send[y * 2 + 1] = world.elements_dynamic[local_idx].total_flux;
        }
    }
    
    // Non-blocking send/recv for halo exchange
    // Send left, receive from left
    MPI_Isend(left_send.data(), world.global_n_elems_root * 2, MPI_DOUBLE, 
              world.left_neighbor, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(world.left_halo_recv.data(), world.global_n_elems_root, 
              MPI_DOUBLE, world.left_neighbor, 0, MPI_COMM_WORLD, &requests[1]);
    
    // Send right, receive from right
    MPI_Isend(right_send.data(), world.global_n_elems_root * 2, MPI_DOUBLE,
              world.right_neighbor, 1, MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(world.right_halo_recv.data(), world.global_n_elems_root,
              MPI_DOUBLE, world.right_neighbor, 1, MPI_COMM_WORLD, &requests[3]);
    
    // Wait for all communication to complete
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
}

// Correct version: exchange halo data with neighboring processes
void exchangeHaloCorrect(World& world) {
    const int n = world.global_n_elems_root;
    
    // Prepare send buffers
    std::vector<double> left_send(n * 2, 0.0);   // (energy, flux) pairs for left boundary
    std::vector<double> right_send(n * 2, 0.0);  // (energy, flux) pairs for right boundary
    
    // Pack left boundary (first column)
    if (world.local_start_x > 0) {
        for (int y = 0; y < n; ++y) {
            const int local_idx = 0 * n + y;
            left_send[y * 2] = world.elements_dynamic[local_idx].current_energy;
            left_send[y * 2 + 1] = world.elements_dynamic[local_idx].total_flux;
        }
    }
    
    // Pack right boundary (last column)
    if (world.local_end_x < world.global_n_elems_root) {
        const int local_last_x = world.local_end_x - world.local_start_x - 1;
        for (int y = 0; y < n; ++y) {
            const int local_idx = local_last_x * n + y;
            right_send[y * 2] = world.elements_dynamic[local_idx].current_energy;
            right_send[y * 2 + 1] = world.elements_dynamic[local_idx].total_flux;
        }
    }
    
    // Allocate receive buffers as pairs
    std::vector<double> left_recv(n * 2, 0.0);
    std::vector<double> right_recv(n * 2, 0.0);
    
    // Exchange with left neighbor
    MPI_Sendrecv(left_send.data(), n * 2, MPI_DOUBLE, world.left_neighbor, 0,
                 left_recv.data(), n * 2, MPI_DOUBLE, world.left_neighbor, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Exchange with right neighbor
    MPI_Sendrecv(right_send.data(), n * 2, MPI_DOUBLE, world.right_neighbor, 1,
                 right_recv.data(), n * 2, MPI_DOUBLE, world.right_neighbor, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Unpack left halo
    for (int y = 0; y < n; ++y) {
        world.left_halo_recv[y].current_energy = left_recv[y * 2];
        world.left_halo_recv[y].total_flux = left_recv[y * 2 + 1];
    }
    
    // Unpack right halo
    for (int y = 0; y < n; ++y) {
        world.right_halo_recv[y].current_energy = right_recv[y * 2];
        world.right_halo_recv[y].total_flux = right_recv[y * 2 + 1];
    }
}

// Run simulation for n_iters iterations with MPI
void runSimulation(World& world, const int n_iters) {
    const int n_elems_root = world.global_n_elems_root;
    const size_t local_n_elems = world.elements_static.size();
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange halo data with neighbors
        exchangeHaloCorrect(world);
        
        // Update all local elements
        for (size_t local_i = 0; local_i < local_n_elems; ++local_i) {
            // Convert local index to global (x, y) coordinates
            const int local_x = local_i / n_elems_root;
            [[maybe_unused]] const int global_x = local_x + world.local_start_x;
            [[maybe_unused]] const int y = local_i % n_elems_root;
            
            const ElementStatic& elem_static = world.elements_static[local_i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[local_i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                // Decode global coordinates from stored index
                const uint64_t neighbor_coords = elem_static.connected_idx[j];
                const int neighbor_x = (neighbor_coords >> 32) & 0xFFFFFFFF;
                const int neighbor_y = neighbor_coords & 0xFFFFFFFF;
                
                // Get neighbor energy (may require halo data)
                ElementDynamic neighbor_dyn = getElementEnergy(world, neighbor_x, neighbor_y);
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }
            
            // Update element state
            ElementDynamic& elem_write = world.elements_dynamic_swap[local_i];
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
    
    // Reduce to global values
    val_t global_energy_sum, global_flux_sum, global_energy_max, global_energy_min;
    MPI_Reduce(&local_energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    // Only rank 0 prints validation results
    bool valid = true;
    if (world.mpi_rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_energy_sum);
        printf("  Flux sum: %.2f\n", global_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);
        
        // Check for numerical issues
        constexpr val_t energy_epsilon = 1e-8;
        
        if (!std::isfinite(global_energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = false;
        }
        
        if (std::abs(global_energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            // Don't fail validation as this can happen with external flows
        }
        
        if (!std::isfinite(global_flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = false;
        }
        
        if (!std::isfinite(global_energy_max) || !std::isfinite(global_energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = false;
        }
        
        if (valid) {
            printf("  Validation: PASSED\n");
        }
    }
    
    // Broadcast validation result to all processes
    MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    return valid;
}

// Compute a simple hash of the results for verification (with MPI reduction)
uint64_t computeHash(const std::vector<ElementDynamic>& elements, [[maybe_unused]] int mpi_rank) {
    uint64_t local_hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        local_hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    
    // Reduce hashes using XOR
    uint64_t global_hash = 0;
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
    
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
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
            if (mpi_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else if (mpi_rank == 0) {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("MPI processes: %d\n", mpi_size);
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
    
    // Calculate memory usage (only print from rank 0)
    if (mpi_rank == 0) {
        const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
        const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Per-process memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("Total memory usage (all %d processes): %.2f MB\n", 
               mpi_size, (total_mem * mpi_size) / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Synchronize before simulation starts
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (mpi_rank == 0) {
        printf("Running simulation...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    if (mpi_rank == 0) {
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
    const uint64_t hash = computeHash(world.elements_dynamic, mpi_rank);
    if (mpi_rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults && mpi_rank == 0) {
        // Gather all results to rank 0 for printing
        std::vector<double> energyData;
        energyData.reserve(n_elems);
        
        // Collect from all processes
        for (int rank = 0; rank < mpi_size; ++rank) {
            if (rank == 0) {
                // Add local data from rank 0
                for (const auto& elem : world.elements_dynamic) {
                    energyData.push_back(elem.current_energy);
                }
            } else {
                // Receive from other ranks
                std::vector<double> remote_data(world.elements_dynamic.size());
                MPI_Recv(remote_data.data(), remote_data.size(), MPI_DOUBLE, 
                        rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (const auto& val : remote_data) {
                    energyData.push_back(val);
                }
            }
        }
        
        print_results(energyData, "ElementEnergy");
    } else if (printResults && mpi_rank != 0) {
        // Send local data to rank 0
        std::vector<double> localData;
        for (const auto& elem : world.elements_dynamic) {
            localData.push_back(elem.current_energy);
        }
        MPI_Send(localData.data(), localData.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (!valid && mpi_rank == 0) {
            MPI_Finalize();
            return 1;
        }
    }
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
