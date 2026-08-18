#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mpi.h>
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
static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t), "ElementDynamic must be tightly packed for MPI halo exchange");

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    int n_elems_root = 0;
    int first_row = 0;
    int local_rows = 0;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int active_ranks) {
    const int base_rows = n_elems_root / active_ranks;
    const int extra_rows = n_elems_root % active_ranks;
    const int local_rows = base_rows + (rank < extra_rows ? 1 : 0);
    const int first_row = rank * base_rows + std::min(rank, extra_rows);
    const int local_elems = local_rows * n_elems_root;
    const int ghost_elems = 2 * n_elems_root;

    world.n_elems_root = n_elems_root;
    world.first_row = first_row;
    world.local_rows = local_rows;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(local_elems);
    // The final two rows are receive-only halo cells for the vertical neighbors.
    world.elements_dynamic.resize(local_elems + ghost_elems);
    world.elements_dynamic_swap.resize(local_elems + ghost_elems);
    
    // Initialize all elements with default material and zero energy
    for (int i = 0; i < local_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int local_x = 0; local_x < local_rows; ++local_x) {
        const int x = first_row + local_x;
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = local_x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    int neighbor_idx;
                    if (nx < first_row) {
                        neighbor_idx = local_elems + ny; // top halo
                    } else if (nx >= first_row + local_rows) {
                        neighbor_idx = local_elems + n_elems_root + ny; // bottom halo
                    } else {
                        neighbor_idx = (nx - first_row) * n_elems_root + ny;
                    }
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    if (first_row == 0) {
        world.elements_static[0].material_idx = INFLOW_MAT_ID;
        world.elements_static[last].material_idx = OUTFLOW_MAT_ID;
    }
    if (first_row + local_rows == n_elems_root) {
        world.elements_static[(local_rows - 1) * n_elems_root].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[local_elems - 1].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, const int rank, const int active_ranks) {
    const size_t n_elems = world.elements_static.size();
    const int row_width = world.n_elems_root;
    const int previous = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < active_ranks ? rank + 1 : MPI_PROC_NULL;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange only the boundary rows.  ElementDynamic is contiguous, and the
        // unused halo total_flux field avoids an extra packing pass.
        MPI_Sendrecv(world.elements_dynamic.data(), 2 * row_width, MPI_DOUBLE, previous, 0,
                     world.elements_dynamic.data() + n_elems + row_width, 2 * row_width, MPI_DOUBLE,
                     next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(world.elements_dynamic.data() + n_elems - row_width, 2 * row_width, MPI_DOUBLE,
                     next, 1, world.elements_dynamic.data() + n_elems, 2 * row_width, MPI_DOUBLE,
                     previous, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
    }
}

// Validate simulation results
bool validateResults(const World& world, const int rank) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = 0; i < world.elements_static.size(); ++i) {
        const ElementDynamic& dynamic = world.elements_dynamic[i];
        energy_sum += dynamic.current_energy;
        flux_sum += dynamic.total_flux;
        energy_max = std::max(dynamic.current_energy, energy_max);
        energy_min = std::min(dynamic.current_energy, energy_min);
    }

    val_t global_energy_sum, global_flux_sum, global_energy_max, global_energy_min;
    MPI_Reduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank != 0) return true;
    energy_sum = global_energy_sum;
    flux_sum = global_flux_sum;
    energy_max = global_energy_max;
    energy_min = global_energy_min;
    
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
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    for (size_t i = 0; i < world.elements_static.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t global_i = static_cast<uint64_t>(world.first_row) * world.n_elems_root + i;
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
        hash ^= (*e_ptr + global_i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + global_i) * 0xbf58476d1ce4e5b9ULL;
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    if (n_elems_root <= 0 || n_iters < 0) {
        if (rank == 0) printf("Grid size must be positive and iteration count must be non-negative.\n");
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("MPI processes: %d\n", ranks);
    printf("\n");
    
    // Build the unstructured mesh
    printf("Building unstructured mesh...\n");
    }
    World world;
    const int active_ranks = std::min(ranks, n_elems_root);
    if (rank < active_ranks) buildSquare2D(world, n_elems_root, rank, active_ranks);
    
    // Calculate memory usage
    const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    if (rank == 0) printf("\n");
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    if (rank < active_ranks) runSimulation(world, n_iters, rank, active_ranks);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    long max_duration_ms = 0;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Computation time: %ld ms\n", max_duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(max_duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = max_duration_ms > 0 ? (n_measured_iters * n_elems) / (max_duration_ms / 1000.0) / 1e9 : 0.0;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    if (rank == 0) {
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification
    const uint64_t local_hash = computeHash(world);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> energyData;
        energyData.reserve(world.elements_static.size());
        for (size_t i = 0; i < world.elements_static.size(); ++i) energyData.push_back(world.elements_dynamic[i].current_energy);
        std::vector<int> counts, displacements;
        if (rank == 0) {
            counts.assign(ranks, 0); displacements.assign(ranks, 0);
            for (int p = 0; p < active_ranks; ++p) {
                const int rows = n_elems_root / active_ranks + (p < n_elems_root % active_ranks ? 1 : 0);
                const int first = p * (n_elems_root / active_ranks) + std::min(p, n_elems_root % active_ranks);
                counts[p] = rows * n_elems_root; displacements[p] = first * n_elems_root;
            }
        }
        std::vector<double> gathered(rank == 0 ? n_elems : 0);
        MPI_Gatherv(energyData.data(), static_cast<int>(energyData.size()), MPI_DOUBLE, gathered.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(gathered, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world, rank);
        int valid_int = valid ? 1 : 0;
        MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!valid_int) { MPI_Finalize(); return 1; }
    }
    MPI_Finalize();
    return 0;
}
