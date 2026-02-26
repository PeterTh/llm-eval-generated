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
#include <omp.h>

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
    int mpi_rank;
    int mpi_size;
    std::vector<idx_t> local_elem_indices;
    std::vector<idx_t> ghost_elem_indices;
    std::vector<int> remote_elem_map;
    std::vector<ElementDynamic> ghost_elements;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);
    
    // Initialize all elements with default material and zero energy
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
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
    world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
}

// Setup MPI domain decomposition using 1D row-based distribution
void setupMPIDomainDecomposition(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    const int elems_per_rank = n_elems / world.mpi_size;
    const int remainder = n_elems % world.mpi_size;
    
    // Compute local element range for this rank
    const int local_start = world.mpi_rank * elems_per_rank + std::min(world.mpi_rank, remainder);
    const int local_end = local_start + elems_per_rank + (world.mpi_rank < remainder ? 1 : 0);
    
    // Collect local elements
    for (int i = local_start; i < local_end; ++i) {
        world.local_elem_indices.push_back(i);
    }
    
    // Identify ghost elements (neighbors on other ranks)
    std::vector<bool> is_local(n_elems, false);
    for (idx_t idx : world.local_elem_indices) {
        is_local[idx] = true;
    }
    
    std::vector<bool> is_ghost(n_elems, false);
    for (idx_t local_idx : world.local_elem_indices) {
        const ElementStatic& elem = world.elements_static[local_idx];
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            const idx_t neighbor_idx = elem.connected_idx[j];
            if (!is_local[neighbor_idx] && !is_ghost[neighbor_idx]) {
                world.ghost_elem_indices.push_back(neighbor_idx);
                is_ghost[neighbor_idx] = true;
            }
        }
    }
    
    // Build remote element map for quick lookup
    world.remote_elem_map.assign(n_elems, -1);
    for (idx_t i = 0; i < static_cast<idx_t>(world.ghost_elem_indices.size()); ++i) {
        world.remote_elem_map[world.ghost_elem_indices[i]] = i;
    }
    
    world.ghost_elements.resize(world.ghost_elem_indices.size());
}

// Map global element index to owning rank
int getOwningRank(idx_t elem_idx, const int n_elems, const int n_ranks) {
    const int elems_per_rank = n_elems / n_ranks;
    const int remainder = n_elems % n_ranks;
    
    for (int rank = 0; rank < n_ranks; ++rank) {
        const idx_t rank_start = rank * elems_per_rank + std::min(rank, remainder);
        const idx_t rank_end = rank_start + elems_per_rank + (rank < remainder ? 1 : 0);
        if (elem_idx >= rank_start && elem_idx < rank_end) {
            return rank;
        }
    }
    return 0;
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations with MPI + OpenMP parallelization
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    const size_t local_n_elems = world.local_elem_indices.size();
    const size_t n_ghost = world.ghost_elem_indices.size();
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Update ghost elements with current energies
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_ghost; ++i) {
            const idx_t ghost_global_idx = world.ghost_elem_indices[i];
            world.ghost_elements[i].current_energy = world.elements_dynamic[ghost_global_idx].current_energy;
        }
        
        // Update all local elements in parallel with OpenMP
        #pragma omp parallel for schedule(static)
        for (size_t local_i = 0; local_i < local_n_elems; ++local_i) {
            const idx_t i = world.local_elem_indices[local_i];
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic* neighbor_dyn = nullptr;
                
                // Check if neighbor is ghost using remote map
                if (world.remote_elem_map[neighbor_idx] >= 0) {
                    neighbor_dyn = &world.ghost_elements[world.remote_elem_map[neighbor_idx]];
                } else {
                    // Local element
                    neighbor_dyn = &world.elements_dynamic[neighbor_idx];
                }
                
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], *neighbor_dyn);
            }
            
            // Update element state
            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
        
        // Synchronize ghost element updates across all ranks
        // Each rank sends its local updates to other ranks via Allgather
        std::vector<double> local_energies(n_elems);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < local_n_elems; ++i) {
            local_energies[world.local_elem_indices[i]] = world.elements_dynamic[world.local_elem_indices[i]].current_energy;
        }
        
        // Use MPI Allgather to synchronize element data for ghost boundaries
        std::vector<double> gathered(n_elems * world.mpi_size);
        MPI_Allgather(local_energies.data(), n_elems, MPI_DOUBLE,
                      gathered.data(), n_elems, MPI_DOUBLE, MPI_COMM_WORLD);
        
        // Merge received data into local copy (all ranks have all data)
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_elems; ++i) {
            if (world.remote_elem_map[i] >= 0) {
                // Update ghost elements with globally synchronized values
                world.ghost_elements[world.remote_elem_map[i]].current_energy = gathered[i];
            }
        }
    }
}

// Validate simulation results with MPI reduction
bool validateResults(const World& world) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    
    #pragma omp parallel for schedule(static) reduction(+:local_energy_sum, local_flux_sum) reduction(max:local_energy_max) reduction(min:local_energy_min)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const auto& elem = world.elements_dynamic[i];
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(elem.current_energy, local_energy_max);
        local_energy_min = std::min(elem.current_energy, local_energy_min);
    }
    
    // Reduce across all MPI ranks
    val_t global_energy_sum, global_flux_sum, global_energy_max, global_energy_min;
    MPI_Reduce(&local_energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    if (world.mpi_rank == 0) {
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
    
    // Broadcast validation result
    bool valid = true;
    if (world.mpi_rank == 0) {
        valid = std::isfinite(global_energy_sum) && std::isfinite(global_flux_sum) && 
                std::isfinite(global_energy_max) && std::isfinite(global_energy_min);
    }
    MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    return valid;
}

// Compute a simple hash of the results for verification (with MPI synchronization)
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t local_hash = 0;
    
    #pragma omp parallel for schedule(static) reduction(^:local_hash)
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        local_hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    
    // Combine hashes across all MPI ranks
    uint64_t global_hash = local_hash;
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
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    // Print info only on rank 0
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    if (mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI+OpenMP Hybrid)\n");
        printf("================================================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI Ranks: %d\n", mpi_size);
        printf("OpenMP Threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }
    
    // Build the unstructured mesh (collective on all ranks)
    World world;
    world.mpi_rank = mpi_rank;
    world.mpi_size = mpi_size;
    buildSquare2D(world, n_elems_root);
    setupMPIDomainDecomposition(world, n_elems_root);
    
    if (mpi_rank == 0) {
        // Calculate memory usage
        const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
        const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
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
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world.elements_dynamic);
    if (mpi_rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation (only on rank 0)
    if (printResults && mpi_rank == 0) {
        std::vector<double> energyData;
        energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation (synchronized across ranks)
    if (validate) {
        bool valid = validateResults(world);
        if (!valid && mpi_rank == 0) {
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
