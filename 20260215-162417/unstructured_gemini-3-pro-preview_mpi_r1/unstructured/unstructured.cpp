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
#include <set>
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
    idx_t global_id; // Added for distributed tracking
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (Local or Ghost indices)
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
    
    // MPI related data
    idx_t global_n_elems;
    idx_t local_start_idx;
    idx_t local_n_elems;
    
    // Ghost communication
    std::vector<idx_t> ghosts_global_ids; // Map local ghost index -> global ID
    std::vector<int> send_counts;
    std::vector<int> send_displs;
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    std::vector<idx_t> send_indices; // Indices in elements_dynamic to send
    std::vector<val_t> send_buffer;
    std::vector<val_t> recv_buffer;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Helper to determine owner of a global index
int get_owner(idx_t global_idx, idx_t n_total, int n_ranks) {
    idx_t base = n_total / n_ranks;
    idx_t rem = n_total % n_ranks;
    if (global_idx < rem * (base + 1)) {
        return global_idx / (base + 1);
    } else {
        return rem + (global_idx - rem * (base + 1)) / base;
    }
}

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, int rank, int n_ranks) {
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * n_elems_root;
    world.global_n_elems = n_elems;
    
    // Determine local range
    idx_t base = n_elems / n_ranks;
    idx_t rem = n_elems % n_ranks;
    if (rank < rem) {
        world.local_n_elems = base + 1;
        world.local_start_idx = rank * (base + 1);
    } else {
        world.local_n_elems = base;
        world.local_start_idx = rem * (base + 1) + (rank - rem) * base;
    }
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements (local only initially)
    world.elements_static.resize(world.local_n_elems);
    world.elements_dynamic.resize(world.local_n_elems);
    world.elements_dynamic_swap.resize(world.local_n_elems);
    
    // Used to track which ghosts we need
    std::map<idx_t, idx_t> global_to_ghost_idx;
    std::vector<idx_t> ghosts_needed;

    // Initialize all elements
    for (idx_t i = 0; i < world.local_n_elems; ++i) {
        idx_t global_idx = world.local_start_idx + i;
        
        world.elements_static[i].global_id = global_idx;
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        
        // Geometry logic
        int x = global_idx / n_elems_root;
        int y = global_idx % n_elems_root;
        
        ElementStatic& elem = world.elements_static[i];
        
        // Connect to neighbors (up, down, left, right)
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        
        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];
            
            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_global_idx = nx * n_elems_root + ny;
                
                // Determine if neighbor is local or ghost
                idx_t neighbor_local_idx;
                if (neighbor_global_idx >= world.local_start_idx && 
                    neighbor_global_idx < world.local_start_idx + world.local_n_elems) {
                    // Local
                    neighbor_local_idx = neighbor_global_idx - world.local_start_idx;
                } else {
                    // Ghost
                    if (global_to_ghost_idx.find(neighbor_global_idx) == global_to_ghost_idx.end()) {
                        idx_t ghost_idx = world.local_n_elems + ghosts_needed.size();
                        global_to_ghost_idx[neighbor_global_idx] = ghost_idx;
                        ghosts_needed.push_back(neighbor_global_idx);
                    }
                    neighbor_local_idx = global_to_ghost_idx[neighbor_global_idx];
                }

                elem.connected_idx[elem.num_connections] = neighbor_local_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }
    
    // Set corner elements as inflow/outflow
    // We need to check if the corner is local
    auto set_material = [&](int r, int c, idx_t mat_id) {
        idx_t g_idx = r * n_elems_root + c;
        if (g_idx >= world.local_start_idx && g_idx < world.local_start_idx + world.local_n_elems) {
            world.elements_static[g_idx - world.local_start_idx].material_idx = mat_id;
        }
    };

    const int last = n_elems_root - 1;
    set_material(0, 0, INFLOW_MAT_ID);
    set_material(0, last, OUTFLOW_MAT_ID);
    set_material(last, 0, OUTFLOW_MAT_ID);
    set_material(last, last, INFLOW_MAT_ID);

    // Resize dynamic arrays to include ghosts
    idx_t total_with_ghosts = world.local_n_elems + ghosts_needed.size();
    world.elements_dynamic.resize(total_with_ghosts);
    // Swap buffer only needs to store local updates? 
    // Actually, runSimulation writes to elem_write = elements_dynamic_swap[i]
    // where i is local. So swap buffer only needs local size.
    // However, after swap, elements_dynamic becomes elements_dynamic_swap.
    // So both need to be big enough to hold ghosts after receiving?
    // Let's keep them same size for simplicity, though we only write to local part of swap.
    world.elements_dynamic_swap.resize(total_with_ghosts);
    world.ghosts_global_ids = ghosts_needed;

    // --- Setup MPI Communication ---
    
    // 1. Tell owners which ghosts we need
    std::vector<idx_t> send_reqs;
    std::vector<int> send_displs(n_ranks, 0);
    std::vector<int> send_counts_elems(n_ranks, 0);

    // Group needed ghosts by owner
    std::vector<std::vector<idx_t>> ghosts_by_owner(n_ranks);
    for (idx_t g_idx : ghosts_needed) {
        int owner = get_owner(g_idx, n_elems, n_ranks);
        ghosts_by_owner[owner].push_back(g_idx);
    }

    // Flatten for Alltoallv
    for (int r = 0; r < n_ranks; ++r) {
        send_counts_elems[r] = ghosts_by_owner[r].size();
        send_displs[r] = (r == 0) ? 0 : send_displs[r-1] + send_counts_elems[r-1];
        send_reqs.insert(send_reqs.end(), ghosts_by_owner[r].begin(), ghosts_by_owner[r].end());
    }
    
    // Exchange counts first to setup recv buffers
    std::vector<int> recv_counts_elems(n_ranks);
    MPI_Alltoall(send_counts_elems.data(), 1, MPI_INT, 
                 recv_counts_elems.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Prepare recv buffers (these are requests from others: "I need X")
    int total_recv_reqs = 0;
    std::vector<int> recv_displs_elems(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        recv_displs_elems[r] = (r == 0) ? 0 : recv_displs_elems[r-1] + recv_counts_elems[r-1];
        total_recv_reqs += recv_counts_elems[r];
    }
    
    std::vector<idx_t> recv_reqs(total_recv_reqs);
    MPI_Alltoallv(send_reqs.data(), send_counts_elems.data(), send_displs.data(), MPI_UINT64_T,
                  recv_reqs.data(), recv_counts_elems.data(), recv_displs_elems.data(), MPI_UINT64_T, MPI_COMM_WORLD);

    // Process requests from others
    // They sent GlobalIDs. We map them to our Local Indices for quick access later.
    world.send_indices.reserve(total_recv_reqs);
    for (idx_t g_idx : recv_reqs) {
        world.send_indices.push_back(g_idx - world.local_start_idx);
    }
    
    // Setup permanent communication structures
    // We send data to others (responding to their requests)
    world.send_counts = recv_counts_elems; // We send what they requested
    world.send_displs = recv_displs_elems;
    world.send_buffer.resize(total_recv_reqs);

    // We receive data from others (updates for our ghosts)
    world.recv_counts = send_counts_elems; // We receive what we requested
    world.recv_displs = send_displs; // displs are same relative logic
    world.recv_buffer.resize(ghosts_needed.size());
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_local = world.local_n_elems;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // --- Communication Phase ---
        // Pack data to send
        for (size_t i = 0; i < world.send_indices.size(); ++i) {
            world.send_buffer[i] = world.elements_dynamic[world.send_indices[i]].current_energy;
        }

        // Exchange data
        // We send `send_buffer` (values for others)
        // We receive `recv_buffer` (values for our ghosts)
        MPI_Alltoallv(world.send_buffer.data(), world.send_counts.data(), world.send_displs.data(), MPI_DOUBLE,
                      world.recv_buffer.data(), world.recv_counts.data(), world.recv_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Unpack ghosts
        // recv_buffer contains ghosts in the order we requested them (which is the order in ghosts_needed)
        for (size_t i = 0; i < world.ghosts_global_ids.size(); ++i) {
             world.elements_dynamic[n_local + i].current_energy = world.recv_buffer[i];
             // flux for ghosts doesn't matter for read, but we should init to 0? Not needed for logic.
        }

        // --- Computation Phase ---
        // Update all local elements
        for (size_t i = 0; i < n_local; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                // connected_idx is now Local Index (0..n_local-1) or Ghost Index (n_local..end)
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
    }
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    
    // Only iterate local elements
    for (size_t i = 0; i < world.local_n_elems; ++i) {
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

    MPI_Allreduce(&local_energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
        return true;
    }
    return true; // Non-root ranks just return true (or block? No, it's boolean return)
}

// Compute a simple hash of the results for verification
uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    for (size_t i = 0; i < world.local_n_elems; ++i) {
        // Use Global ID for stability
        idx_t global_idx = world.local_start_idx + i;
        const ElementDynamic& elem = world.elements_dynamic[i];

        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elem.current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elem.total_flux);
        
        local_hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }
    
    uint64_t global_hash = 0;
    // XOR reduction is safe
    // MPI_UNSIGNED_LONG_LONG corresponds to uint64_t usually
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, MPI_COMM_WORLD);
    
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
    
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI)\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %lu elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI Size:   %d\n", size);
        printf("\n");
        printf("Building distributed unstructured mesh...\n");
    }
    
    World world;
    buildSquare2D(world, n_elems_root, rank, size);
    
    // Calculate memory usage (local)
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    
    // Aggregate memory usage
    size_t global_total_mem = 0;
    MPI_Reduce(&total_mem, &global_total_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total Memory usage: %.2f MB\n", global_total_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long max_duration_ms = 0;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(max_duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / (max_duration_ms / 1000.0) / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration
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
        // Gather all data to rank 0
        std::vector<double> localData;
        localData.reserve(world.local_n_elems);
        for (size_t i=0; i<world.local_n_elems; ++i) {
            localData.push_back(world.elements_dynamic[i].current_energy);
        }

        // We need to gather variable size data
        std::vector<int> recv_counts(size);
        int local_count = world.local_n_elems;
        MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<double> globalData;
        std::vector<int> displs(size);
        if (rank == 0) {
            globalData.resize(n_elems);
            displs[0] = 0;
            for(int i=1; i<size; ++i) displs[i] = displs[i-1] + recv_counts[i-1];
        }

        MPI_Gatherv(localData.data(), local_count, MPI_DOUBLE, 
                    globalData.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(globalData, "ElementEnergy");
        }
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (rank == 0 && !valid) {
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
