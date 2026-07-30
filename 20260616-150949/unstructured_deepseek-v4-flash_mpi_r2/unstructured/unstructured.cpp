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
    // MPI decomposition info
    int rank = 0;
    int nprocs = 1;
    int n_elems_root_global = 0;
    int local_rows = 0;
    int start_row = 0;
    int end_row = 0;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh, distributed across MPI ranks
void buildSquare2D(World& world, const int n_elems_root_global) {
    world.n_elems_root_global = n_elems_root_global;
    const int n_elems_root = n_elems_root_global;
    const int rank = world.rank;
    const int nprocs = world.nprocs;
    
    // Compute 1D row-based decomposition across MPI ranks
    world.local_rows = n_elems_root / nprocs;
    const int remainder = n_elems_root % nprocs;
    if (rank < remainder) {
        world.local_rows++;
        world.start_row = rank * world.local_rows;
    } else {
        world.start_row = remainder * (world.local_rows + 1) + (rank - remainder) * world.local_rows;
    }
    world.end_row = world.start_row + world.local_rows;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements: static for owned only, dynamic includes halo cells
    const size_t n_local_owned = static_cast<size_t>(world.local_rows) * n_elems_root;
    const size_t n_local_with_halos = static_cast<size_t>(world.local_rows + 2) * n_elems_root;
    
    world.elements_static.resize(n_local_owned);
    world.elements_dynamic.resize(n_local_with_halos);
    world.elements_dynamic_swap.resize(n_local_with_halos);
    
    // Initialize owned elements with default material and zero energy
    // Dynamic array layout: [top_halo (1 row)] [owned rows] [bottom_halo (1 row)]
    for (int r = 0; r < world.local_rows; ++r) {
        for (int c = 0; c < n_elems_root; ++c) {
            const size_t static_idx = static_cast<size_t>(r) * n_elems_root + c;
            const size_t dyn_idx = static_cast<size_t>(r + 1) * n_elems_root + c;
            world.elements_static[static_idx].material_idx = DEFAULT_MAT_ID;
            world.elements_static[static_idx].num_connections = 0;
            world.elements_dynamic[dyn_idx].current_energy = 0.0;
            world.elements_dynamic[dyn_idx].total_flux = 0.0;
        }
    }
    
    // Initialize halo cells to zero
    for (int c = 0; c < n_elems_root; ++c) {
        world.elements_dynamic[c].current_energy = 0.0;
        world.elements_dynamic[c].total_flux = 0.0;
        const size_t bottom_idx = static_cast<size_t>(world.local_rows + 1) * n_elems_root + c;
        world.elements_dynamic[bottom_idx].current_energy = 0.0;
        world.elements_dynamic[bottom_idx].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    // connected_idx stores indices into the local extended dynamic array
    for (int r = 0; r < world.local_rows; ++r) {
        const int global_x = world.start_row + r;
        for (int c = 0; c < n_elems_root; ++c) {
            const size_t static_idx = static_cast<size_t>(r) * n_elems_root + c;
            ElementStatic& elem = world.elements_static[static_idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = global_x + offsets[n][0];
                const int ny = c + offsets[n][1];
                
                // Check if neighbor is within global bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    idx_t neighbor_dyn_idx;
                    if (nx >= world.start_row && nx < world.end_row) {
                        // Neighbor is owned by this rank
                        const int local_nr = nx - world.start_row;
                        neighbor_dyn_idx = static_cast<idx_t>(local_nr + 1) * n_elems_root + ny;
                    } else if (nx < world.start_row) {
                        // Neighbor is in the top halo (from rank above)
                        neighbor_dyn_idx = ny;
                    } else {
                        // Neighbor is in the bottom halo (from rank below)
                        neighbor_dyn_idx = static_cast<idx_t>(world.local_rows + 1) * n_elems_root + ny;
                    }
                    elem.connected_idx[elem.num_connections] = neighbor_dyn_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    // Only if the corner element is owned by this rank
    const int last = n_elems_root - 1;
    const int corners[4][2] = {{0, 0}, {0, last}, {last, 0}, {last, last}};
    const idx_t corner_mats[4] = {INFLOW_MAT_ID, OUTFLOW_MAT_ID, OUTFLOW_MAT_ID, INFLOW_MAT_ID};
    
    for (int i = 0; i < 4; ++i) {
        const int gx = corners[i][0];
        const int gy = corners[i][1];
        if (gx >= world.start_row && gx < world.end_row) {
            const int local_r = gx - world.start_row;
            const size_t static_idx = static_cast<size_t>(local_r) * n_elems_root + gy;
            world.elements_static[static_idx].material_idx = corner_mats[i];
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Exchange halo data with neighboring MPI ranks before each iteration
void exchangeHalos(World& world) {
    const int n_elems_root = world.n_elems_root_global;
    const int top_neighbor = (world.rank > 0) ? world.rank - 1 : MPI_PROC_NULL;
    const int bottom_neighbor = (world.rank < world.nprocs - 1) ? world.rank + 1 : MPI_PROC_NULL;
    const int count_bytes = n_elems_root * static_cast<int>(sizeof(ElementDynamic));
    
    // Exchange with top neighbor: send our first owned row, receive into top halo
    MPI_Sendrecv(
        &world.elements_dynamic[n_elems_root],
        count_bytes, MPI_BYTE,
        top_neighbor, 0,
        &world.elements_dynamic[0],
        count_bytes, MPI_BYTE,
        top_neighbor, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
    
    // Exchange with bottom neighbor: send our last owned row, receive into bottom halo
    MPI_Sendrecv(
        &world.elements_dynamic[static_cast<size_t>(world.local_rows) * n_elems_root],
        count_bytes, MPI_BYTE,
        bottom_neighbor, 0,
        &world.elements_dynamic[static_cast<size_t>(world.local_rows + 1) * n_elems_root],
        count_bytes, MPI_BYTE,
        bottom_neighbor, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
}

// Run simulation for n_iters iterations
// Dynamic array: [top_halo] [owned rows] [bottom_halo]
// Each owned element at static index i has dynamic index i + n_elems_root
void runSimulation(World& world, const int n_iters) {
    const int n_elems_root = world.n_elems_root_global;
    const size_t n_local_owned = static_cast<size_t>(world.local_rows) * n_elems_root;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange halo data with neighboring ranks
        exchangeHalos(world);
        
        // Update all owned elements
        for (size_t i = 0; i < n_local_owned; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const size_t dyn_idx = i + n_elems_root;
            
            const ElementDynamic& elem_dyn = world.elements_dynamic[dyn_idx];
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
            ElementDynamic& elem_write = world.elements_dynamic_swap[dyn_idx];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results on gathered data
bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : elements) {
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
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
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
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (world_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", world_size);
        printf("\n");
    }
    
    // Build the unstructured mesh
    if (world_rank == 0) printf("Building unstructured mesh...\n");
    World world;
    world.rank = world_rank;
    world.nprocs = world_size;
    buildSquare2D(world, n_elems_root);
    
    // Calculate per-rank memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    
    if (world_rank == 0) {
        printf("Memory usage per rank: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation
    if (world_rank == 0) printf("Running simulation...\n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Get maximum time across all ranks
    long long max_duration_ms;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (world_rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(max_duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (max_duration_ms / 1000.0) / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Gather results to rank 0 for hash, validation, and printing
    std::vector<ElementDynamic> all_dynamic;
    if (world_rank == 0) all_dynamic.resize(n_elems);
    
    // Compute gather counts and displacements
    std::vector<int> recvcounts(world_size);
    std::vector<int> displs(world_size);
    int offset_elems = 0;
    const int base_rows = n_elems_root / world_size;
    const int rem_rows = n_elems_root % world_size;
    for (int r = 0; r < world_size; ++r) {
        const int rows = base_rows + (r < rem_rows ? 1 : 0);
        recvcounts[r] = rows * n_elems_root * static_cast<int>(sizeof(ElementDynamic));
        displs[r] = offset_elems * static_cast<int>(sizeof(ElementDynamic));
        offset_elems += rows * n_elems_root;
    }
    
    MPI_Gatherv(
        &world.elements_dynamic[n_elems_root],
        world.local_rows * n_elems_root * static_cast<int>(sizeof(ElementDynamic)),
        MPI_BYTE,
        all_dynamic.data(),
        recvcounts.data(),
        displs.data(),
        MPI_BYTE,
        0,
        MPI_COMM_WORLD
    );
    
    if (world_rank == 0) {
        // Compute hash for verification
        const uint64_t hash = computeHash(all_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
        
        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(all_dynamic.size());
            for (const auto& elem : all_dynamic) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }
        
        // Validation
        if (validate) {
            if (!validateResults(all_dynamic)) {
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
