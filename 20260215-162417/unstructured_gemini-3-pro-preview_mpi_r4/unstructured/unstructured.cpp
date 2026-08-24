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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (local or ghost index)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Communication structures
struct GhostInfo {
    idx_t global_idx;
    int rank;
    int buffer_idx; // Index in the receive buffer
};

struct ParallelContext {
    int rank;
    int size;
    idx_t local_start;
    idx_t local_end;
    idx_t global_n_elems;
    
    // Ghost communication
    std::vector<idx_t> ghosts_global_indices; // Global indices of ghosts needed
    std::vector<int> send_counts;
    std::vector<int> send_displs;
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    std::vector<idx_t> send_indices; // Local indices to pack and send
    std::vector<idx_t> recv_indices; // Indices in elements_dynamic where received data goes
    
    // Optimization for sparse communication
    std::vector<int> neighbor_ranks; // Ranks we communicate with (send or recv)
    
    std::vector<val_t> send_buffer;
    std::vector<val_t> recv_buffer;
};

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;   // Only local elements
    std::vector<ElementDynamic> elements_dynamic; // Local elements + Ghosts
    std::vector<ElementDynamic> elements_dynamic_swap; // Only local needed for update, but kept same size for simplicity
    ParallelContext ctx;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * n_elems_root;
    world.ctx.global_n_elems = n_elems;
    
    // Partitioning: slab decomposition
    idx_t elems_per_rank = n_elems / world.ctx.size;
    idx_t remainder = n_elems % world.ctx.size;
    
    world.ctx.local_start = world.ctx.rank * elems_per_rank + std::min(static_cast<idx_t>(world.ctx.rank), remainder);
    idx_t count = elems_per_rank + (static_cast<idx_t>(world.ctx.rank) < remainder ? 1 : 0);
    world.ctx.local_end = world.ctx.local_start + count;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(count);
    // Dynamic vectors will be resized after ghost identification
    
    std::map<idx_t, idx_t> global_to_ghost_map; // global_idx -> local_ghost_idx (offset from count)
    std::vector<idx_t> needed_ghosts;

    // Helper to find owner rank of a global index
    auto get_owner = [&](idx_t global_idx) -> int {
        idx_t per_rank = n_elems / world.ctx.size;
        idx_t rem = n_elems % world.ctx.size;
        if (global_idx < rem * (per_rank + 1)) {
            return global_idx / (per_rank + 1);
        } else {
            return rem + (global_idx - rem * (per_rank + 1)) / per_rank;
        }
    };

    // Initialize local elements
    for (idx_t i = 0; i < count; ++i) {
        idx_t global_idx = world.ctx.local_start + i;
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        
        // Reconstruct (x, y) from global_idx
        int x = global_idx / n_elems_root;
        int y = global_idx % n_elems_root;
        
        // Logic from original buildSquare2D
        if (global_idx == 0) world.elements_static[i].material_idx = INFLOW_MAT_ID;
        else if (global_idx == static_cast<idx_t>(n_elems_root) - 1) world.elements_static[i].material_idx = OUTFLOW_MAT_ID;
        else if (global_idx == n_elems - n_elems_root) world.elements_static[i].material_idx = OUTFLOW_MAT_ID;
        else if (global_idx == n_elems - 1) world.elements_static[i].material_idx = INFLOW_MAT_ID;

        // Connect to neighbors
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        
        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];
            
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                idx_t neighbor_global_idx = static_cast<idx_t>(nx) * n_elems_root + ny;
                
                idx_t local_idx;
                if (neighbor_global_idx >= world.ctx.local_start && neighbor_global_idx < world.ctx.local_end) {
                    local_idx = neighbor_global_idx - world.ctx.local_start;
                } else {
                    // It's a ghost
                    if (global_to_ghost_map.find(neighbor_global_idx) == global_to_ghost_map.end()) {
                        idx_t ghost_idx = count + needed_ghosts.size();
                        global_to_ghost_map[neighbor_global_idx] = ghost_idx;
                        needed_ghosts.push_back(neighbor_global_idx);
                    }
                    local_idx = global_to_ghost_map[neighbor_global_idx];
                }
                
                world.elements_static[i].connected_idx[world.elements_static[i].num_connections] = local_idx;
                world.elements_static[i].connected_flux[world.elements_static[i].num_connections] = 1.0;
                world.elements_static[i].num_connections++;
            }
        }
    }
    
    // Store ghost info
    world.ctx.ghosts_global_indices = needed_ghosts;
    
    // Resize dynamic arrays to hold local + ghosts
    size_t total_local_elements = count + needed_ghosts.size();
    world.elements_dynamic.resize(total_local_elements);
    world.elements_dynamic_swap.resize(total_local_elements);
    
    // Initialize elements
    for(size_t i=0; i<total_local_elements; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // --- Setup Communication Graph ---
    int comm_size = world.ctx.size;
    world.ctx.send_counts.assign(comm_size, 0);
    world.ctx.recv_counts.assign(comm_size, 0);
    world.ctx.send_displs.resize(comm_size);
    world.ctx.recv_displs.resize(comm_size);
    
    // 1. Tell everyone what I need (send requests)
    std::vector<int> requests_to_send(comm_size, 0);
    std::vector<std::vector<idx_t>> requested_indices(comm_size);
    
    for (idx_t ghost_global : needed_ghosts) {
        int owner = get_owner(ghost_global);
        requested_indices[owner].push_back(ghost_global);
        requests_to_send[owner]++;
    }
    
    // Exchange counts of requests
    std::vector<int> requests_to_recv(comm_size);
    MPI_Alltoall(requests_to_send.data(), 1, MPI_INT, requests_to_recv.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // 2. Exchange actual indices
    // Prepare send buffers for indices (I send indices I NEED to owners)
    // Owners receive indices they NEED TO SEND back to me later
    
    // Calculate displacements for alltoallv
    std::vector<int> idx_send_displs(comm_size, 0);
    std::vector<int> idx_recv_displs(comm_size, 0);
    int total_idx_send = 0;
    int total_idx_recv = 0;
    
    for(int i=0; i<comm_size; ++i) {
        idx_send_displs[i] = total_idx_send;
        total_idx_send += requests_to_send[i];
        
        idx_recv_displs[i] = total_idx_recv;
        total_idx_recv += requests_to_recv[i];
    }
    
    std::vector<idx_t> idx_send_buf;
    idx_send_buf.reserve(total_idx_send);
    for(int i=0; i<comm_size; ++i) {
        idx_send_buf.insert(idx_send_buf.end(), requested_indices[i].begin(), requested_indices[i].end());
    }
    
    std::vector<idx_t> idx_recv_buf(total_idx_recv);
    
    MPI_Alltoallv(idx_send_buf.data(), requests_to_send.data(), idx_send_displs.data(), MPI_UINT64_T,
                  idx_recv_buf.data(), requests_to_recv.data(), idx_recv_displs.data(), MPI_UINT64_T,
                  MPI_COMM_WORLD);
                  
    // 3. Process received indices to build send_list for future steps
    // idx_recv_buf contains global indices that others need from me.
    // I need to map them to my local indices.
    
    world.ctx.send_indices.clear();
    int current_recv_offset = 0;
    int total_data_to_send = 0;
    
    for(int i=0; i<comm_size; ++i) {
        world.ctx.send_counts[i] = requests_to_recv[i]; // I send X elements to rank i
        world.ctx.send_displs[i] = total_data_to_send;
        total_data_to_send += requests_to_recv[i];
        
        for(int j=0; j<requests_to_recv[i]; ++j) {
            idx_t requested_global = idx_recv_buf[current_recv_offset + j];
            // Verify ownership
            if (requested_global < world.ctx.local_start || requested_global >= world.ctx.local_end) {
                fprintf(stderr, "Rank %d Error: Requested global %lu not owned (range %lu-%lu)\n", 
                        world.ctx.rank, requested_global, world.ctx.local_start, world.ctx.local_end);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            world.ctx.send_indices.push_back(requested_global - world.ctx.local_start);
        }
        current_recv_offset += requests_to_recv[i];
    }
    
    world.ctx.send_buffer.resize(total_data_to_send);
    
    // 4. Setup Recv info for simulation steps
    // I receive data corresponding to 'needed_ghosts', grouped by rank as defined in step 1
    int total_data_to_recv = 0;
    for(int i=0; i<comm_size; ++i) {
        world.ctx.recv_counts[i] = requests_to_send[i];
        world.ctx.recv_displs[i] = total_data_to_recv;
        total_data_to_recv += requests_to_send[i];
    }
    world.ctx.recv_buffer.resize(total_data_to_recv);
    
    // Store where each received value should go in elements_dynamic
    world.ctx.recv_indices.reserve(total_data_to_recv);
    for(int i=0; i<comm_size; ++i) {
        for(idx_t ghost_global : requested_indices[i]) {
            world.ctx.recv_indices.push_back(global_to_ghost_map[ghost_global]);
        }
    }
}

// Exchange halo data between ranks using non-blocking communication
void exchange_halo(World& world) {
    // 1. Pack data: Collect values of local elements that others need
    for(size_t i=0; i<world.ctx.send_indices.size(); ++i) {
        // send_indices contains local index (0..count) of elements to send
        world.ctx.send_buffer[i] = world.elements_dynamic[world.ctx.send_indices[i]].current_energy;
    }
    
    std::vector<MPI_Request> requests;
    requests.reserve(world.ctx.size * 2); // Upper bound estimate
    
    // 2. Post Receives (non-blocking)
    for(int rank=0; rank < world.ctx.size; ++rank) {
        if (world.ctx.recv_counts[rank] > 0) {
            MPI_Request req;
            MPI_Irecv(&world.ctx.recv_buffer[world.ctx.recv_displs[rank]], 
                      world.ctx.recv_counts[rank], MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
    }
    
    // 3. Post Sends (non-blocking)
    for(int rank=0; rank < world.ctx.size; ++rank) {
        if (world.ctx.send_counts[rank] > 0) {
            MPI_Request req;
            MPI_Isend(&world.ctx.send_buffer[world.ctx.send_displs[rank]], 
                      world.ctx.send_counts[rank], MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
    }
    
    // 4. Wait for all communication to complete
    if (!requests.empty()) {
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
                  
    // 5. Unpack data: Update ghost elements
    for(size_t i=0; i<world.ctx.recv_indices.size(); ++i) {
        // recv_indices contains local index (count..total) where ghost data lives
        world.elements_dynamic[world.ctx.recv_indices[i]].current_energy = world.ctx.recv_buffer[i];
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems_local = world.elements_static.size();
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Communicate ghost cells before update
        exchange_halo(world);

        // Update all LOCAL elements
        for (size_t i = 0; i < n_elems_local; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements (local or ghost)
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                // connected_idx points to correct index in elements_dynamic (local or ghost)
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }
            
            // Update element state
            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
        
        // Swap buffers (only for local elements, ghosts are overwritten next iter)
        // Note: we only swap the *values* for local elements.
        // Actually, we can just swap the pointers/vectors, but ghosts in 'swap' vector are garbage.
        // The computeFlux uses 'elements_dynamic' (read-only for local, updated for ghost).
        // The update writes to 'elements_dynamic_swap'.
        // So swapping is correct. The next iteration will overwrite ghosts in 'elements_dynamic' (which was 'swap') anyway.
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    
    // Only iterate over owned elements
    size_t count = world.elements_static.size();
    for (size_t i = 0; i < count; ++i) {
        const auto& elem = world.elements_dynamic[i];
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(elem.current_energy, local_energy_max);
        local_energy_min = std::min(elem.current_energy, local_energy_min);
    }
    
    val_t global_energy_sum = 0.0;
    val_t global_flux_sum = 0.0;
    val_t global_energy_max = 0.0;
    val_t global_energy_min = 0.0;

    MPI_Reduce(&local_energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    if (world.ctx.rank == 0) {
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
    
    // Broadcast result to all
    // Actually just return true/false based on rank 0 decision?
    // Since we don't return bool across MPI here easily without Bcast, let's assume it passes if local passes (which is empty check).
    // Correct way: Bcast success.
    return true; 
}

// Compute a simple hash of the results for verification
uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    size_t count = world.elements_static.size();
    
    for (size_t i = 0; i < count; ++i) {
        // Simple hash combining energy and flux values
        // Important: use GLOBAL index for position dependence
        idx_t global_idx = world.ctx.local_start + i;
        
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
        local_hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }
    
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
    MPI_Init(&argc, &argv);
    
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI Parallel)\n");
        printf("========================================================\n");
        printf("Grid size: %d x %d = %lu elements\n", n_elems_root, n_elems_root, n_elems);
        printf("MPI Size: %d processes\n", size);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building distributed unstructured mesh...\n");
    }
    
    World world;
    world.ctx.rank = rank;
    world.ctx.size = size;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage (local)
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t local_mem = static_mem + dynamic_mem;
    
    size_t global_mem = 0;
    MPI_Reduce(&local_mem, &global_mem, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Total Memory usage: %.2f MB\n", global_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    const long long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);
        
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
    
    // Print results for external validation (MPI gather)
    if (printResults) {
        // Collect all energy values on rank 0
        std::vector<double> local_energy;
        size_t count = world.elements_static.size();
        local_energy.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            local_energy.push_back(world.elements_dynamic[i].current_energy);
        }
        
        // We need to gather variable sized arrays
        int my_count = static_cast<int>(count);
        std::vector<int> counts(size);
        MPI_Gather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        std::vector<int> displs(size);
        std::vector<double> global_energy;
        
        if (rank == 0) {
            displs[0] = 0;
            for(int i=1; i<size; ++i) displs[i] = displs[i-1] + counts[i-1];
            global_energy.resize(displs[size-1] + counts[size-1]);
        }
        
        MPI_Gatherv(local_energy.data(), my_count, MPI_DOUBLE,
                   global_energy.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
                   
        if (rank == 0) {
            print_results(global_energy, "ElementEnergy");
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
