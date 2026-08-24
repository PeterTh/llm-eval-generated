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

    // MPI related
    int rank;
    int size;
    idx_t local_start;
    idx_t local_end;
    idx_t local_count;
    
    // Ghost elements map: global_idx -> local_idx (in elements_dynamic)
    std::vector<idx_t> ghost_global_indices;
    std::vector<int> ghost_owners;
    
    // Communication lists
    // For each rank, a list of local indices to send
    std::vector<std::vector<idx_t>> send_indices;
    // For each rank, a list of local indices (ghosts) to receive into
    std::vector<std::vector<idx_t>> recv_indices;
    
    // Buffers for communication
    std::vector<std::vector<val_t>> send_buffers;
    std::vector<std::vector<val_t>> recv_buffers;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Helper to get owner rank of a global index
int get_owner(idx_t global_idx, int size, idx_t n_elems) {
    idx_t base = n_elems / size;
    idx_t remainder = n_elems % size;
    if (global_idx < remainder * (base + 1)) {
        return global_idx / (base + 1);
    } else {
        return remainder + (global_idx - remainder * (base + 1)) / base;
    }
}

// Build a 2D square grid as an unstructured mesh with MPI partitioning
void buildSquare2D(World& world, const int n_elems_root) {
    const idx_t n_elems = (idx_t)n_elems_root * n_elems_root;
    
    // Determine local range
    idx_t base = n_elems / world.size;
    idx_t remainder = n_elems % world.size;
    
    if (world.rank < remainder) {
        world.local_count = base + 1;
        world.local_start = world.rank * (base + 1);
    } else {
        world.local_count = base;
        world.local_start = remainder * (base + 1) + (world.rank - remainder) * base;
    }
    world.local_end = world.local_start + world.local_count;

    // Initialize materials (same on all ranks)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Temporary storage for ghosts mapping
    std::vector<idx_t> ghosts;
    // Map global index to local ghost index (offset from local_count)
    // Using a simple vector search for now as number of ghosts is small (boundary only)
    auto get_ghost_local_index = [&](idx_t global_idx) -> idx_t {
        for (size_t i = 0; i < ghosts.size(); ++i) {
            if (ghosts[i] == global_idx) return world.local_count + i;
        }
        ghosts.push_back(global_idx);
        return world.local_count + ghosts.size() - 1;
    };

    // Allocate local elements
    world.elements_static.resize(world.local_count);
    world.elements_dynamic.resize(world.local_count); 
    // We will resize elements_dynamic later to include ghosts
    
    // Initialize local elements
    for (idx_t i = 0; i < world.local_count; ++i) {
        idx_t global_idx = world.local_start + i;
        
        // Determine material based on global index
        int x = global_idx / n_elems_root;
        int y = global_idx % n_elems_root;
        
        idx_t mat_id = DEFAULT_MAT_ID;
        const int last = n_elems_root - 1;
        if (x == 0 && y == 0) mat_id = INFLOW_MAT_ID;
        else if (x == 0 && y == last) mat_id = OUTFLOW_MAT_ID;
        else if (x == last && y == 0) mat_id = OUTFLOW_MAT_ID;
        else if (x == last && y == last) mat_id = INFLOW_MAT_ID;

        world.elements_static[i].material_idx = mat_id;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        
        // Build connectivity
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        ElementStatic& elem = world.elements_static[i];
        
        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];
            
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_global_idx = nx * n_elems_root + ny;
                
                idx_t neighbor_local_idx;
                if (neighbor_global_idx >= world.local_start && neighbor_global_idx < world.local_end) {
                    neighbor_local_idx = neighbor_global_idx - world.local_start;
                } else {
                    neighbor_local_idx = get_ghost_local_index(neighbor_global_idx);
                }
                
                elem.connected_idx[elem.num_connections] = neighbor_local_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }
    
    // Finalize ghost setup
    world.ghost_global_indices = ghosts;
    world.elements_dynamic.resize(world.local_count + ghosts.size());
    world.elements_dynamic_swap.resize(world.local_count + ghosts.size());
    
    // Determine ghost owners
    world.ghost_owners.resize(ghosts.size());
    for(size_t i=0; i<ghosts.size(); ++i) {
        world.ghost_owners[i] = get_owner(ghosts[i], world.size, n_elems);
    }

    // Setup communication lists
    world.recv_indices.resize(world.size);
    world.send_indices.resize(world.size);
    world.recv_buffers.resize(world.size);
    world.send_buffers.resize(world.size);

    // Populate recv_indices (what I need from others)
    for (size_t i = 0; i < ghosts.size(); ++i) {
        int owner = world.ghost_owners[i];
        world.recv_indices[owner].push_back(world.local_count + i); // Local index where to put received data
    }
    
    // Use Alltoall to tell others what they need to send me
    // We need to send 'ghosts[i]' (which is a global index) to 'owner'.
    // The owner needs to know which of ITS local elements corresponds to 'ghosts[i]'.
    // That is 'ghosts[i] - owner_start'.
    
    // Exchange counts first
    std::vector<int> req_send_counts(world.size, 0);
    std::vector<int> req_recv_counts(world.size, 0);
    
    for(size_t i=0; i<ghosts.size(); ++i) {
        req_send_counts[world.ghost_owners[i]]++;
    }
    
    MPI_Alltoall(req_send_counts.data(), 1, MPI_INT, req_recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Prepare requests: I tell owner "please send me global index G"
    // The owner receives "please send global index G to rank R"
    
    std::vector<std::vector<idx_t>> requests_to_send(world.size);
    for(size_t i=0; i<ghosts.size(); ++i) {
        requests_to_send[world.ghost_owners[i]].push_back(ghosts[i]);
    }
    
    // Flatten for MPI
    std::vector<idx_t> flat_send_reqs;
    std::vector<int> sdispls(world.size, 0);
    int current_disp = 0;
    for(int r=0; r<world.size; ++r) {
        sdispls[r] = current_disp;
        flat_send_reqs.insert(flat_send_reqs.end(), requests_to_send[r].begin(), requests_to_send[r].end());
        current_disp += requests_to_send[r].size();
    }
    
    // Receive buffers for requests
    std::vector<idx_t> flat_recv_reqs;
    int total_recv = 0;
    std::vector<int> rdispls(world.size, 0);
    for(int r=0; r<world.size; ++r) {
        rdispls[r] = total_recv;
        total_recv += req_recv_counts[r];
    }
    flat_recv_reqs.resize(total_recv);
    
    // Check types
    static_assert(sizeof(idx_t) == 8, "idx_t must be 64-bit for MPI_UINT64_T");

    MPI_Alltoallv(flat_send_reqs.data(), req_send_counts.data(), sdispls.data(), MPI_UINT64_T,
                  flat_recv_reqs.data(), req_recv_counts.data(), rdispls.data(), MPI_UINT64_T,
                  MPI_COMM_WORLD);
                  
    // Process received requests: these are global indices that OTHERS need from ME.
    // I need to map them to my local indices and add to send_indices.
    // NOTE: The received requests come from rank r. This means rank r sent a request.
    // In Alltoallv, data received from rank r is placed in the buffer section corresponding to r.
    
    for(int r=0; r<world.size; ++r) {
        int count = req_recv_counts[r];
        int offset = rdispls[r];
        for(int k=0; k<count; ++k) {
            idx_t global_idx = flat_recv_reqs[offset + k];
            // Verify ownership
            if (global_idx >= world.local_start && global_idx < world.local_end) {
                 world.send_indices[r].push_back(global_idx - world.local_start);
            } else {
                printf("Rank %d Error: Requested global %lu not owned (range %lu-%lu)\n", 
                       world.rank, global_idx, world.local_start, world.local_end);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        // Resize buffers
        world.send_buffers[r].resize(world.send_indices[r].size());
        world.recv_buffers[r].resize(world.recv_indices[r].size());
        
        // Populate recv_indices (what I need from others) - actually we already did this, but let's double check order
        // The ghosts vector has elements ordered. We iterated over ghosts to create requests.
        // We need to match received data to ghosts.
        // My 'recv_indices' needs to store the LOCAL indices where I put the data.
        // world.recv_indices[owner] was populated earlier.
        // The order must match the order I sent requests?
        // Yes, if I send requests [G1, G2] to rank R, rank R will send back [Val(G1), Val(G2)].
        // So recv_indices[R] should correspond to ghosts owned by R in the order they appear in ghosts vector?
        // Wait, requests_to_send[R] preserves order of ghosts loop.
        // recv_indices[R] also preserves order of ghosts loop.
        // So they match.
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

void exchangeGhosts(World& world) {
    // 1. Pack data into send buffers
    for (int r = 0; r < world.size; ++r) {
        if (world.send_indices[r].empty()) continue;
        for (size_t i = 0; i < world.send_indices[r].size(); ++i) {
            world.send_buffers[r][i] = world.elements_dynamic[world.send_indices[r][i]].current_energy;
        }
    }

    // 2. Non-blocking sends and receives
    std::vector<MPI_Request> requests;
    requests.reserve(world.size * 2);

    for (int r = 0; r < world.size; ++r) {
        if (!world.recv_indices[r].empty()) {
            requests.resize(requests.size() + 1);
            MPI_Irecv(world.recv_buffers[r].data(), world.recv_indices[r].size(), MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &requests.back());
        }
    }
    
    for (int r = 0; r < world.size; ++r) {
        if (!world.send_indices[r].empty()) {
            requests.resize(requests.size() + 1);
            MPI_Isend(world.send_buffers[r].data(), world.send_indices[r].size(), MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &requests.back());
        }
    }
    
    // 3. Wait for all
    if (!requests.empty()) {
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
    
    // 4. Unpack received data
    for (int r = 0; r < world.size; ++r) {
        if (world.recv_indices[r].empty()) continue;
        for (size_t i = 0; i < world.recv_indices[r].size(); ++i) {
             world.elements_dynamic[world.recv_indices[r][i]].current_energy = world.recv_buffers[r][i];
        }
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    // Only process local elements
    const size_t n_elems = world.local_count;
    
    // Initial exchange to get ghosts for first iteration
    exchangeGhosts(world);
    
    for (int iter = 0; iter < n_iters; ++iter) {
        
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
                const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }
            
            // Update element state
            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
        
        // Exchange ghost data for next iteration
        exchangeGhosts(world);
    }
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    
    // Iterate only over owned elements
    for (size_t i = 0; i < world.local_count; ++i) {
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
        return true;
    }
    
    // Broadcast result from rank 0 (optional, but good for consistent return)
    // Actually, just return true on others?
    // If rank 0 returns false, others should probably know?
    // But main() only checks return value on rank 0 effectively if we exit/abort.
    // Let's broadcast success.
    
    // Actually, simply returning true on non-root is fine as long as only root prints.
    return true;
}

// Compute a simple hash of the results for verification
uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    // Iterate only over owned elements
    for (size_t i = 0; i < world.local_count; ++i) {
        const auto& elem = world.elements_dynamic[i];
        // Simple hash combining energy and flux values
        // We need to use global index for consistent hashing across ranks
        idx_t global_idx = world.local_start + i;
        
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elem.current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elem.total_flux);
        
        // Ensure hash order is independent of partition?
        // XOR is commutative. So just XORing everything is fine.
        // The original code: hash ^= (*e_ptr + i) * ...
        // We should use global_idx instead of i.
        
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI Size: %d\n", size);
        printf("\n");
        printf("Building unstructured mesh...\n");
    }
    
    // Build the unstructured mesh
    World world;
    world.rank = rank;
    world.size = size;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    size_t local_static_mem = world.elements_static.size() * sizeof(ElementStatic);
    size_t local_dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2; // dynamic + swap
    // Also MPI buffers
    size_t local_mpi_mem = 0;
    for(const auto& v : world.send_buffers) local_mpi_mem += v.capacity() * sizeof(val_t);
    for(const auto& v : world.recv_buffers) local_mpi_mem += v.capacity() * sizeof(val_t);
    for(const auto& v : world.send_indices) local_mpi_mem += v.capacity() * sizeof(idx_t);
    for(const auto& v : world.recv_indices) local_mpi_mem += v.capacity() * sizeof(idx_t);
    
    size_t local_mem = local_static_mem + local_dynamic_mem + local_mpi_mem;
    
    size_t total_mem = 0;
    size_t total_static = 0;
    size_t total_dynamic = 0;
    
    MPI_Reduce(&local_mem, &total_mem, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_static_mem, &total_static, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dynamic_mem, &total_dynamic, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Total Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               total_static / (1024.0 * 1024.0),
               total_dynamic / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    const long long local_duration_ms = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    long long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;
        
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
    if (printResults) {
        // Collect local energy data
        std::vector<double> local_energy;
        local_energy.reserve(world.local_count);
        for (size_t i = 0; i < world.local_count; ++i) {
            local_energy.push_back(world.elements_dynamic[i].current_energy);
        }
        
        // Gather counts
        int local_count_int = (int)world.local_count;
        std::vector<int> recv_counts(size);
        MPI_Gather(&local_count_int, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        std::vector<int> displs(size);
        std::vector<double> global_energy;
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recv_counts[i-1];
            }
            global_energy.resize(n_elems);
        }
        
        MPI_Gatherv(local_energy.data(), local_count_int, MPI_DOUBLE,
                   global_energy.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
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
