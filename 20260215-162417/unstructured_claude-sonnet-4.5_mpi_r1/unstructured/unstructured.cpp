#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <unordered_map>
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
    std::vector<idx_t> ghost_indices;  // Indices of ghost elements needed from other ranks
    std::vector<int> ghost_ranks;      // Which rank owns each ghost element
    std::unordered_map<idx_t, idx_t> global_to_local;  // Map global index to local index
    std::vector<idx_t> local_to_global;  // Map local index to global index
    idx_t num_local;  // Number of local (owned) elements
    idx_t num_ghost;  // Number of ghost elements
    
    // Pre-computed communication pattern
    struct CommPattern {
        int remote_rank;
        std::vector<idx_t> send_indices;  // Local indices to send
        std::vector<idx_t> recv_indices;  // Local indices to receive into
    };
    std::vector<CommPattern> comm_patterns;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh (MPI version)
// Each rank owns a horizontal strip of rows
void buildSquare2D(World& world, const int n_elems_root, int rank, int size) {
    // Initialize materials on all ranks
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Partition rows across ranks
    int rows_per_rank = n_elems_root / size;
    int extra_rows = n_elems_root % size;
    
    int start_row = rank * rows_per_rank + std::min(rank, extra_rows);
    int end_row = start_row + rows_per_rank + (rank < extra_rows ? 1 : 0);
    
    // Determine which global elements this rank owns
    std::vector<idx_t> owned_elements;
    for (int x = start_row; x < end_row; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            idx_t global_idx = x * n_elems_root + y;
            owned_elements.push_back(global_idx);
        }
    }
    
    world.num_local = owned_elements.size();
    
    // Build connectivity and identify ghost elements
    std::unordered_map<idx_t, bool> needed_ghosts;
    
    for (idx_t local_idx = 0; local_idx < world.num_local; ++local_idx) {
        idx_t global_idx = owned_elements[local_idx];
        world.local_to_global.push_back(global_idx);
        world.global_to_local[global_idx] = local_idx;
        
        int x = global_idx / n_elems_root;
        int y = global_idx % n_elems_root;
        
        // Check all 4 neighbors
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];
            
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                idx_t neighbor_global = nx * n_elems_root + ny;
                
                // Check if neighbor is owned by this rank
                bool is_local = (nx >= start_row && nx < end_row);
                
                if (!is_local && needed_ghosts.find(neighbor_global) == needed_ghosts.end()) {
                    needed_ghosts[neighbor_global] = true;
                }
            }
        }
    }
    
    // Add ghost elements to local structures
    for (const auto& kv : needed_ghosts) {
        idx_t ghost_global = kv.first;
        idx_t local_idx = world.num_local + world.ghost_indices.size();
        
        world.ghost_indices.push_back(ghost_global);
        world.local_to_global.push_back(ghost_global);
        world.global_to_local[ghost_global] = local_idx;
        
        // Determine which rank owns this ghost element
        int ghost_x = ghost_global / n_elems_root;
        int owner_rank = 0;
        int cumulative = 0;
        for (int r = 0; r < size; ++r) {
            int r_rows = rows_per_rank + (r < extra_rows ? 1 : 0);
            if (ghost_x < cumulative + r_rows) {
                owner_rank = r;
                break;
            }
            cumulative += r_rows;
        }
        world.ghost_ranks.push_back(owner_rank);
    }
    
    world.num_ghost = world.ghost_indices.size();
    
    const idx_t total_local = world.num_local + world.num_ghost;
    
    // Allocate elements
    world.elements_static.resize(total_local);
    world.elements_dynamic.resize(total_local);
    world.elements_dynamic_swap.resize(total_local);
    
    // Initialize all elements
    for (idx_t local_idx = 0; local_idx < total_local; ++local_idx) {
        idx_t global_idx = world.local_to_global[local_idx];
        
        world.elements_static[local_idx].material_idx = DEFAULT_MAT_ID;
        world.elements_static[local_idx].num_connections = 0;
        world.elements_dynamic[local_idx].current_energy = 0.0;
        world.elements_dynamic[local_idx].total_flux = 0.0;
        
        int x = global_idx / n_elems_root;
        int y = global_idx % n_elems_root;
        
        ElementStatic& elem = world.elements_static[local_idx];
        
        // Connect to neighbors using local indices
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        
        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];
            
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_global = nx * n_elems_root + ny;
                const idx_t neighbor_local = world.global_to_local[neighbor_global];
                elem.connected_idx[elem.num_connections] = neighbor_local;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }
    
    // Set corner elements as inflow/outflow
    const int last = n_elems_root - 1;
    idx_t corners[4] = {
        static_cast<idx_t>(0 * n_elems_root + 0),
        static_cast<idx_t>(0 * n_elems_root + last),
        static_cast<idx_t>(last * n_elems_root + 0),
        static_cast<idx_t>(last * n_elems_root + last)
    };
    
    idx_t corner_mats[4] = {INFLOW_MAT_ID, OUTFLOW_MAT_ID, OUTFLOW_MAT_ID, INFLOW_MAT_ID};
    
    for (int i = 0; i < 4; ++i) {
        auto it = world.global_to_local.find(corners[i]);
        if (it != world.global_to_local.end()) {
            world.elements_static[it->second].material_idx = corner_mats[i];
        }
    }
    
    // Build communication pattern
    std::unordered_map<int, std::vector<idx_t>> recv_from_rank;  // ghost local idx
    std::unordered_map<int, std::vector<idx_t>> send_to_rank_global;  // global indices
    
    for (idx_t i = 0; i < world.num_ghost; ++i) {
        int owner = world.ghost_ranks[i];
        idx_t local_idx = world.num_local + i;
        recv_from_rank[owner].push_back(local_idx);
    }
    
    // Exchange information about what to send
    for (int r = 0; r < size; ++r) {
        if (r == rank) continue;
        
        std::vector<idx_t> my_requests;
        auto it = recv_from_rank.find(r);
        if (it != recv_from_rank.end()) {
            for (idx_t local_idx : it->second) {
                my_requests.push_back(world.local_to_global[local_idx]);
            }
        }
        
        int send_count = my_requests.size();
        int recv_count;
        
        MPI_Sendrecv(&send_count, 1, MPI_INT, r, 0,
                     &recv_count, 1, MPI_INT, r, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::vector<idx_t> their_requests(recv_count);
        
        MPI_Sendrecv(my_requests.data(), send_count, MPI_UINT64_T, r, 1,
                     their_requests.data(), recv_count, MPI_UINT64_T, r, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if (recv_count > 0) {
            send_to_rank_global[r] = their_requests;
        }
    }
    
    // Build communication patterns
    for (const auto& kv : recv_from_rank) {
        World::CommPattern pattern;
        pattern.remote_rank = kv.first;
        pattern.recv_indices = kv.second;
        
        // Find corresponding send indices
        auto send_it = send_to_rank_global.find(kv.first);
        if (send_it != send_to_rank_global.end()) {
            for (idx_t global_idx : send_it->second) {
                pattern.send_indices.push_back(world.global_to_local[global_idx]);
            }
        }
        
        world.comm_patterns.push_back(pattern);
    }
    
    // Add patterns for ranks we only send to (not receive from)
    for (const auto& kv : send_to_rank_global) {
        if (recv_from_rank.find(kv.first) == recv_from_rank.end()) {
            World::CommPattern pattern;
            pattern.remote_rank = kv.first;
            for (idx_t global_idx : kv.second) {
                pattern.send_indices.push_back(world.global_to_local[global_idx]);
            }
            world.comm_patterns.push_back(pattern);
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Exchange ghost element data between ranks (using pre-computed pattern)
void exchangeGhostData(World& world) {
    std::vector<MPI_Request> requests;
    std::vector<std::vector<val_t>> send_buffers(world.comm_patterns.size());
    std::vector<std::vector<val_t>> recv_buffers(world.comm_patterns.size());
    
    // Post all receives and sends
    for (size_t i = 0; i < world.comm_patterns.size(); ++i) {
        const auto& pattern = world.comm_patterns[i];
        
        // Post receive if we have data to receive
        if (!pattern.recv_indices.empty()) {
            recv_buffers[i].resize(pattern.recv_indices.size());
            MPI_Request req;
            MPI_Irecv(recv_buffers[i].data(), pattern.recv_indices.size(), MPI_DOUBLE,
                     pattern.remote_rank, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
        
        // Pack and post send if we have data to send
        if (!pattern.send_indices.empty()) {
            send_buffers[i].resize(pattern.send_indices.size());
            for (size_t j = 0; j < pattern.send_indices.size(); ++j) {
                send_buffers[i][j] = world.elements_dynamic[pattern.send_indices[j]].current_energy;
            }
            MPI_Request req;
            MPI_Isend(send_buffers[i].data(), pattern.send_indices.size(), MPI_DOUBLE,
                     pattern.remote_rank, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
    }
    
    // Wait for all communication to complete
    if (!requests.empty()) {
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
    
    // Unpack received data
    for (size_t i = 0; i < world.comm_patterns.size(); ++i) {
        const auto& pattern = world.comm_patterns[i];
        if (!pattern.recv_indices.empty()) {
            for (size_t j = 0; j < pattern.recv_indices.size(); ++j) {
                world.elements_dynamic[pattern.recv_indices[j]].current_energy = recv_buffers[i][j];
            }
        }
    }
}

// Run simulation for n_iters iterations (MPI version)
void runSimulation(World& world, const int n_iters) {
    const size_t n_local = world.num_local;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange ghost data before computation
        exchangeGhostData(world);
        
        // Update only local elements
        for (size_t i = 0; i < n_local; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements (may include ghosts)
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
        
        // Swap buffers (only for local elements)
        for (size_t i = 0; i < n_local; ++i) {
            world.elements_dynamic[i] = world.elements_dynamic_swap[i];
        }
    }
}

// Validate simulation results (MPI version - gather to rank 0)
bool validateResults(const World& world, int rank, int size) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    // Compute local statistics
    for (idx_t i = 0; i < world.num_local; ++i) {
        const auto& elem = world.elements_dynamic[i];
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    
    // Reduce to rank 0
    val_t global_energy_sum, global_flux_sum, global_energy_max, global_energy_min;
    MPI_Reduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
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
    }
    
    return true;
}

// Compute a simple hash of the results for verification (MPI version)
uint64_t computeHash(const World& world, int rank, int size) {
    uint64_t hash = 0;
    
    // Compute local hash
    for (idx_t i = 0; i < world.num_local; ++i) {
        idx_t global_idx = world.local_to_global[i];
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
        hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }
    
    // XOR-reduce all hashes to rank 0
    uint64_t global_hash = 0;
    MPI_Reduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    
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
    buildSquare2D(world, n_elems_root, rank, size);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    
    size_t global_mem = 0;
    MPI_Reduce(&total_mem, &global_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Local elements per rank: %ld (avg)\n", n_elems / size);
        printf("Total memory usage: %.2f MB\n", global_mem / (1024.0 * 1024.0));
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
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
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
        
        // Compute hash for verification
        const uint64_t hash = computeHash(world, rank, size);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    } else {
        computeHash(world, rank, size);
    }
    
    // Gather results for printing
    if (printResults) {
        // Gather all local elements to rank 0
        std::vector<val_t> local_energies(world.num_local);
        for (idx_t i = 0; i < world.num_local; ++i) {
            local_energies[i] = world.elements_dynamic[i].current_energy;
        }
        
        std::vector<idx_t> local_globals(world.num_local);
        for (idx_t i = 0; i < world.num_local; ++i) {
            local_globals[i] = world.local_to_global[i];
        }
        
        // Gather counts
        int local_count = world.num_local;
        std::vector<int> all_counts;
        if (rank == 0) {
            all_counts.resize(size);
        }
        MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Compute displacements
        std::vector<int> displs;
        if (rank == 0) {
            displs.resize(size);
            displs[0] = 0;
            for (int r = 1; r < size; ++r) {
                displs[r] = displs[r-1] + all_counts[r-1];
            }
        }
        
        // Gather energies and indices
        std::vector<val_t> all_energies;
        std::vector<idx_t> all_globals;
        if (rank == 0) {
            all_energies.resize(n_elems);
            all_globals.resize(n_elems);
        }
        
        MPI_Gatherv(local_energies.data(), local_count, MPI_DOUBLE,
                    all_energies.data(), all_counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        MPI_Gatherv(local_globals.data(), local_count, MPI_UINT64_T,
                    all_globals.data(), all_counts.data(), displs.data(), MPI_UINT64_T,
                    0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            // Reorder by global index
            std::vector<val_t> ordered_energies(n_elems);
            for (int i = 0; i < n_elems; ++i) {
                ordered_energies[all_globals[i]] = all_energies[i];
            }
            print_results(ordered_energies, "ElementEnergy");
        }
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world, rank, size);
        if (!valid && rank == 0) {
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
