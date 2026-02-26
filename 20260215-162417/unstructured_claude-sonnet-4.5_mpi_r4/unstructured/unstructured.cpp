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
    
    // MPI-related data
    int rank;
    int size;
    idx_t local_start;
    idx_t local_count;
    idx_t global_count;
    
    // Ghost elements for boundary communication
    std::vector<idx_t> ghost_indices;  // Global indices of ghost elements we need
    std::vector<ElementDynamic> ghost_data;  // Ghost element data received from other ranks
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh (MPI-parallel version)
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    world.global_count = n_elems;
    
    // Distribute elements across ranks
    const idx_t elems_per_rank = n_elems / world.size;
    const idx_t remainder = n_elems % world.size;
    
    world.local_start = world.rank * elems_per_rank + std::min((idx_t)world.rank, remainder);
    world.local_count = elems_per_rank + (world.rank < remainder ? 1 : 0);
    const idx_t local_end = world.local_start + world.local_count;
    
    // Initialize materials (same on all ranks)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate local elements
    world.elements_static.resize(world.local_count);
    world.elements_dynamic.resize(world.local_count);
    world.elements_dynamic_swap.resize(world.local_count);
    
    // Build connectivity and identify ghost elements
    std::vector<bool> is_ghost_needed(n_elems, false);
    
    for (idx_t local_i = 0; local_i < world.local_count; ++local_i) {
        const idx_t global_i = world.local_start + local_i;
        const int x = global_i / n_elems_root;
        const int y = global_i % n_elems_root;
        
        ElementStatic& elem = world.elements_static[local_i];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;
        
        world.elements_dynamic[local_i].current_energy = 0.0;
        world.elements_dynamic[local_i].total_flux = 0.0;
        
        // Connect to neighbors (up, down, left, right)
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        
        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];
            
            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_idx = nx * n_elems_root + ny;
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
                
                // Mark as ghost if neighbor is not in local range
                if (neighbor_idx < world.local_start || neighbor_idx >= local_end) {
                    is_ghost_needed[neighbor_idx] = true;
                }
            }
        }
    }
    
    // Build ghost index list
    for (idx_t i = 0; i < n_elems; ++i) {
        if (is_ghost_needed[i]) {
            world.ghost_indices.push_back(i);
        }
    }
    world.ghost_data.resize(world.ghost_indices.size());
    
    // Set corner elements as inflow/outflow (on appropriate ranks)
    const int last = n_elems_root - 1;
    const idx_t corners[4] = {
        0 * n_elems_root + 0,
        0 * n_elems_root + last,
        last * n_elems_root + 0,
        last * n_elems_root + last
    };
    const idx_t corner_mats[4] = {INFLOW_MAT_ID, OUTFLOW_MAT_ID, OUTFLOW_MAT_ID, INFLOW_MAT_ID};
    
    for (int c = 0; c < 4; ++c) {
        if (corners[c] >= world.local_start && corners[c] < local_end) {
            const idx_t local_idx = corners[c] - world.local_start;
            world.elements_static[local_idx].material_idx = corner_mats[c];
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Exchange ghost data between ranks
void exchangeGhostData(World& world) {
    // Build send requests: what data do other ranks need from us?
    std::vector<std::vector<idx_t>> send_indices(world.size);
    std::vector<std::vector<ElementDynamic>> send_data(world.size);
    
    // Collect requests from all ranks
    std::vector<int> recv_counts(world.size);
    std::vector<int> send_counts(world.size);
    int my_ghost_count = world.ghost_indices.size();
    
    MPI_Allgather(&my_ghost_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    std::vector<int> recv_displs(world.size + 1, 0);
    for (int i = 0; i < world.size; ++i) {
        recv_displs[i + 1] = recv_displs[i] + recv_counts[i];
    }
    
    std::vector<idx_t> all_ghost_requests(recv_displs[world.size]);
    MPI_Allgatherv(world.ghost_indices.data(), my_ghost_count, MPI_UINT64_T,
                   all_ghost_requests.data(), recv_counts.data(), recv_displs.data(),
                   MPI_UINT64_T, MPI_COMM_WORLD);
    
    // Determine what data to send to each rank
    for (int r = 0; r < world.size; ++r) {
        for (int i = recv_displs[r]; i < recv_displs[r + 1]; ++i) {
            idx_t requested_idx = all_ghost_requests[i];
            if (requested_idx >= world.local_start && 
                requested_idx < world.local_start + world.local_count) {
                send_indices[r].push_back(requested_idx);
                idx_t local_idx = requested_idx - world.local_start;
                send_data[r].push_back(world.elements_dynamic[local_idx]);
            }
        }
        send_counts[r] = send_data[r].size();
    }
    
    // Exchange data counts
    std::vector<int> my_recv_counts(world.size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, my_recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Prepare receive buffers
    std::vector<int> my_recv_displs(world.size + 1, 0);
    for (int i = 0; i < world.size; ++i) {
        my_recv_displs[i + 1] = my_recv_displs[i] + my_recv_counts[i];
    }
    
    std::vector<ElementDynamic> recv_buffer(my_recv_displs[world.size]);
    
    // Flatten send buffers
    std::vector<int> send_displs(world.size + 1, 0);
    for (int i = 0; i < world.size; ++i) {
        send_displs[i + 1] = send_displs[i] + send_counts[i];
    }
    std::vector<ElementDynamic> send_buffer(send_displs[world.size]);
    for (int r = 0; r < world.size; ++r) {
        std::copy(send_data[r].begin(), send_data[r].end(), 
                  send_buffer.begin() + send_displs[r]);
    }
    
    // Exchange element data
    MPI_Datatype mpi_elem_type;
    MPI_Type_contiguous(sizeof(ElementDynamic), MPI_BYTE, &mpi_elem_type);
    MPI_Type_commit(&mpi_elem_type);
    
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), mpi_elem_type,
                  recv_buffer.data(), my_recv_counts.data(), my_recv_displs.data(), mpi_elem_type,
                  MPI_COMM_WORLD);
    
    MPI_Type_free(&mpi_elem_type);
    
    // Map received data to ghost elements
    std::vector<idx_t> received_indices(recv_buffer.size());
    std::vector<int> send_idx_counts(world.size);
    for (int r = 0; r < world.size; ++r) {
        send_idx_counts[r] = send_indices[r].size();
    }
    
    std::vector<int> recv_idx_displs(world.size + 1, 0);
    for (int i = 0; i < world.size; ++i) {
        recv_idx_displs[i + 1] = recv_idx_displs[i] + my_recv_counts[i];
    }
    
    std::vector<idx_t> send_idx_buffer(send_displs[world.size]);
    for (int r = 0; r < world.size; ++r) {
        std::copy(send_indices[r].begin(), send_indices[r].end(),
                  send_idx_buffer.begin() + send_displs[r]);
    }
    
    MPI_Alltoallv(send_idx_buffer.data(), send_counts.data(), send_displs.data(), MPI_UINT64_T,
                  received_indices.data(), my_recv_counts.data(), my_recv_displs.data(), MPI_UINT64_T,
                  MPI_COMM_WORLD);
    
    // Build index-to-data mapping for ghost elements
    for (size_t i = 0; i < received_indices.size(); ++i) {
        idx_t global_idx = received_indices[i];
        for (size_t j = 0; j < world.ghost_indices.size(); ++j) {
            if (world.ghost_indices[j] == global_idx) {
                world.ghost_data[j] = recv_buffer[i];
                break;
            }
        }
    }
}

// Run simulation for n_iters iterations (MPI-parallel version)
void runSimulation(World& world, const int n_iters) {
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange ghost data before computing
        exchangeGhostData(world);
        
        // Update all local elements
        for (idx_t local_i = 0; local_i < world.local_count; ++local_i) {
            const ElementStatic& elem_static = world.elements_static[local_i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[local_i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_global_idx = elem_static.connected_idx[j];
                ElementDynamic neighbor_dyn = {0.0, 0.0};
                
                // Check if neighbor is local or ghost
                if (neighbor_global_idx >= world.local_start && 
                    neighbor_global_idx < world.local_start + world.local_count) {
                    // Local element
                    const idx_t neighbor_local_idx = neighbor_global_idx - world.local_start;
                    neighbor_dyn = world.elements_dynamic[neighbor_local_idx];
                } else {
                    // Ghost element - find in ghost data
                    for (size_t g = 0; g < world.ghost_indices.size(); ++g) {
                        if (world.ghost_indices[g] == neighbor_global_idx) {
                            neighbor_dyn = world.ghost_data[g];
                            break;
                        }
                    }
                }
                
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

// Validate simulation results (MPI-parallel version)
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
    
    // Global reductions
    val_t energy_sum, flux_sum, energy_max, energy_min;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    bool valid = true;
    if (world.rank == 0) {
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
    
    // Broadcast validation result to all ranks
    MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    return valid;
}

// Compute a simple hash of the results for verification (MPI-parallel version)
uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    for (size_t local_i = 0; local_i < world.elements_dynamic.size(); ++local_i) {
        idx_t global_i = world.local_start + local_i;
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[local_i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[local_i].total_flux);
        local_hash ^= (*e_ptr + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    
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
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (rank 0 only, then broadcast)
    if (rank == 0) {
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
                MPI_Abort(MPI_COMM_WORLD, 0);
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
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
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI)\n");
        printf("==================================================\n");
        printf("MPI processes: %d\n", size);
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
    world.size = size;
    buildSquare2D(world, n_elems_root);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        // Calculate memory usage
        const size_t static_mem = n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage (total): %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        
        const size_t local_static_mem = world.local_count * sizeof(ElementStatic);
        const size_t local_dynamic_mem = world.local_count * sizeof(ElementDynamic) * 2;
        const size_t local_total_mem = local_static_mem + local_dynamic_mem;
        printf("Memory usage (per rank avg): %.2f MB\n", local_total_mem / (1024.0 * 1024.0));
        printf("Local elements on rank 0: %lu\n", world.local_count);
        printf("\n");
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
        std::vector<double> global_energyData;
        std::vector<double> local_energyData;
        local_energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            local_energyData.push_back(elem.current_energy);
        }
        
        if (rank == 0) {
            global_energyData.resize(n_elems);
        }
        
        // Gather sizes
        std::vector<int> recv_counts(size);
        std::vector<int> recv_displs(size + 1, 0);
        int local_count = world.local_count;
        MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            for (int i = 0; i < size; ++i) {
                recv_displs[i + 1] = recv_displs[i] + recv_counts[i];
            }
        }
        
        MPI_Gatherv(local_energyData.data(), local_count, MPI_DOUBLE,
                    global_energyData.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            print_results(global_energyData, "ElementEnergy");
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
