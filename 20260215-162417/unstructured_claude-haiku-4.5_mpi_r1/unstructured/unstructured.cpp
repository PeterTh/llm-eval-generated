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
    
    // MPI domain decomposition
    int rank, num_ranks;
    size_t local_start, local_end;  // Elements assigned to this rank [local_start, local_end)
    size_t num_local_elems;
    int n_elems_root;  // Global grid dimensions
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    
    world.n_elems_root = n_elems_root;
    
    // Initialize materials on all ranks
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Distribute elements across ranks
    const size_t elems_per_rank = n_elems / world.num_ranks;
    const size_t remainder = n_elems % world.num_ranks;
    
    world.local_start = world.rank * elems_per_rank + std::min((size_t)world.rank, remainder);
    world.local_end = world.local_start + elems_per_rank + (world.rank < (int)remainder ? 1 : 0);
    world.num_local_elems = world.local_end - world.local_start;
    
    // Allocate local elements
    world.elements_static.resize(world.num_local_elems);
    world.elements_dynamic.resize(world.num_local_elems);
    world.elements_dynamic_swap.resize(world.num_local_elems);
    
    // Initialize local elements with default material and zero energy
    for (size_t i = 0; i < world.num_local_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const idx_t global_idx = x * n_elems_root + y;
            
            // Check if this element belongs to this rank
            if (global_idx < (idx_t)world.local_start || global_idx >= (idx_t)world.local_end) {
                continue;
            }
            
            const idx_t local_idx = global_idx - world.local_start;
            ElementStatic& elem = world.elements_static[local_idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const idx_t neighbor_global_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_global_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    const idx_t corners[4] = {
        (idx_t)(0 * n_elems_root + 0),
        (idx_t)(0 * n_elems_root + last),
        (idx_t)(last * n_elems_root + 0),
        (idx_t)(last * n_elems_root + last)
    };
    const idx_t corner_types[4] = {INFLOW_MAT_ID, OUTFLOW_MAT_ID, OUTFLOW_MAT_ID, INFLOW_MAT_ID};
    
    for (int c = 0; c < 4; ++c) {
        if (corners[c] >= (idx_t)world.local_start && corners[c] < (idx_t)world.local_end) {
            const idx_t local_idx = corners[c] - world.local_start;
            world.elements_static[local_idx].material_idx = corner_types[c];
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations with MPI
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.num_local_elems;
    const int n_elems_root = world.n_elems_root;
    
    // Pre-compute which neighbors are remote vs local for each element
    std::vector<std::vector<int>> local_neighbors(n_elems);
    std::vector<std::vector<int>> remote_neighbors(n_elems);
    std::vector<std::vector<int>> remote_neighbor_ranks(n_elems);
    std::vector<std::vector<idx_t>> remote_neighbor_global_idx(n_elems);
    
    for (size_t i = 0; i < n_elems; ++i) {
        const ElementStatic& elem_static = world.elements_static[i];
        
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_global_idx = elem_static.connected_idx[j];
            
            if (neighbor_global_idx >= (idx_t)world.local_start && neighbor_global_idx < (idx_t)world.local_end) {
                local_neighbors[i].push_back(j);
            } else {
                int neighbor_rank = 0;
                int n_elems = n_elems_root * n_elems_root;
                const size_t elems_per_rank = n_elems / world.num_ranks;
                const size_t remainder = n_elems % world.num_ranks;
                
                for (int r = 0; r < world.num_ranks; ++r) {
                    size_t rank_start = r * elems_per_rank + std::min((size_t)r, remainder);
                    size_t rank_end = rank_start + elems_per_rank + (r < (int)remainder ? 1 : 0);
                    if (neighbor_global_idx >= (idx_t)rank_start && neighbor_global_idx < (idx_t)rank_end) {
                        neighbor_rank = r;
                        break;
                    }
                }
                
                remote_neighbors[i].push_back(j);
                remote_neighbor_ranks[i].push_back(neighbor_rank);
                remote_neighbor_global_idx[i].push_back(neighbor_global_idx);
            }
        }
    }
    
    // Build global element map for remote access
    std::vector<ElementDynamic> global_elements_cache(world.num_local_elems);
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange ghost cell data (remote element energies)
        std::vector<std::vector<val_t>> send_buffers(world.num_ranks);
        std::vector<int> send_counts(world.num_ranks, 0);
        std::vector<int> recv_counts(world.num_ranks, 0);
        std::vector<std::vector<val_t>> recv_buffers(world.num_ranks);
        
        // Count what we need to send
        for (int r = 0; r < world.num_ranks; ++r) {
            for (size_t i = 0; i < n_elems; ++i) {
                for (int rr : remote_neighbor_ranks[i]) {
                    if (rr == r) send_counts[r]++;
                }
            }
        }
        
        // Use Alltoall to exchange counts
        std::vector<int> send_counts_copy = send_counts;
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, 
                     recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // Build send buffer
        std::vector<int> send_offsets(world.num_ranks, 0);
        for (int r = 0; r < world.num_ranks; ++r) {
            send_buffers[r].reserve(send_counts[r]);
        }
        
        for (size_t i = 0; i < n_elems; ++i) {
            for (size_t j = 0; j < remote_neighbor_ranks[i].size(); ++j) {
                int r = remote_neighbor_ranks[i][j];
                send_buffers[r].push_back(world.elements_dynamic[i].current_energy);
            }
        }
        
        // Communicate remote element energies using Alltoallv
        std::vector<int> send_displs(world.num_ranks + 1, 0);
        std::vector<int> recv_displs(world.num_ranks + 1, 0);
        for (int r = 0; r < world.num_ranks; ++r) {
            send_displs[r + 1] = send_displs[r] + send_counts[r];
            recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
        }
        
        int total_recv = recv_displs[world.num_ranks];
        std::vector<val_t> recv_flat(total_recv);
        std::vector<val_t> send_flat;
        for (int r = 0; r < world.num_ranks; ++r) {
            send_flat.insert(send_flat.end(), send_buffers[r].begin(), send_buffers[r].end());
        }
        
        MPI_Alltoallv(send_flat.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
                      recv_flat.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                      MPI_COMM_WORLD);
        
        // Unpack received data into a map for quick access
        std::vector<val_t> remote_energy_cache;
        std::vector<idx_t> remote_energy_global_idx;
        
        int recv_idx = 0;
        for (int r = 0; r < world.num_ranks; ++r) {
            for (int c = 0; c < recv_counts[r]; ++c) {
                remote_energy_cache.push_back(recv_flat[recv_displs[r] + c]);
                recv_idx++;
            }
        }
        
        // Rebuild mapping from global idx to cache position for this iteration
        std::vector<std::vector<val_t>> remote_energy_by_global(world.num_local_elems);
        recv_idx = 0;
        for (size_t i = 0; i < n_elems; ++i) {
            for (size_t j = 0; j < remote_neighbor_ranks[i].size(); ++j) {
                remote_energy_by_global[i].push_back(remote_energy_cache[recv_idx++]);
            }
        }
        
        // Update all local elements
        for (size_t i = 0; i < n_elems; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            val_t total_flux = mat.external_flow;
            
            // Process local neighbors
            for (int local_conn_idx : local_neighbors[i]) {
                const idx_t neighbor_idx = elem_static.connected_idx[local_conn_idx];
                if (neighbor_idx >= (idx_t)world.local_start && neighbor_idx < (idx_t)world.local_end) {
                    const idx_t local_neighbor_idx = neighbor_idx - world.local_start;
                    const ElementDynamic& neighbor_dyn = world.elements_dynamic[local_neighbor_idx];
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[local_conn_idx], neighbor_dyn);
                }
            }
            
            // Process remote neighbors
            for (size_t j = 0; j < remote_neighbors[i].size(); ++j) {
                int conn_idx = remote_neighbors[i][j];
                ElementDynamic remote_elem;
                remote_elem.current_energy = remote_energy_by_global[i][j];
                remote_elem.total_flux = 0.0;
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[conn_idx], remote_elem);
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

// Validate simulation results with MPI reductions
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
    MPI_Allreduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    
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
    
    // Broadcast validation result to all ranks
    bool valid = true;
    if (world.rank == 0) {
        valid = std::isfinite(energy_sum) && std::isfinite(flux_sum) && 
                std::isfinite(energy_max) && std::isfinite(energy_min);
    }
    MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    return valid;
}

// Compute a simple hash of the results for verification (with MPI)
uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
        const idx_t global_idx = world.local_start + i;
        local_hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }
    
    // Reduce hash across all ranks using XOR
    uint64_t global_hash = 0;
    MPI_Allreduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
    
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
    
    World world;
    MPI_Comm_rank(MPI_COMM_WORLD, &world.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world.num_ranks);
    
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse, but only use on rank 0 initially)
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
            if (world.rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (world.rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI)\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("MPI ranks: %d\n", world.num_ranks);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        
        printf("Building unstructured mesh...\n");
    }
    
    // Build the unstructured mesh (distributed)
    buildSquare2D(world, n_elems_root);
    
    if (world.rank == 0) {
        // Calculate memory usage (approximation for total)
        const size_t static_mem_per_rank = world.elements_static.size() * sizeof(ElementStatic);
        const size_t dynamic_mem_per_rank = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
        const size_t total_mem = (static_mem_per_rank + dynamic_mem_per_rank) * world.num_ranks;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               (static_mem_per_rank * world.num_ranks) / (1024.0 * 1024.0),
               (dynamic_mem_per_rank * world.num_ranks) / (1024.0 * 1024.0));
        printf("\n");
        
        printf("Running simulation...\n");
    }
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    if (world.rank == 0) {
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
    if (world.rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults && world.rank == 0) {
        std::vector<double> energyData;
        energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (!valid && world.rank == 0) {
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
