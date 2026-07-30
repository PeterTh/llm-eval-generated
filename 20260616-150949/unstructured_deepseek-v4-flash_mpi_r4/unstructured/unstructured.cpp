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
    std::vector<ElementDynamic> elements_dynamic;       // local + ghost rows at end
    std::vector<ElementDynamic> elements_dynamic_swap;  // same size as elements_dynamic
    
    // MPI partition info
    int rank = 0;
    int n_ranks = 1;
    int start_row = 0;
    int end_row = 0;        // exclusive
    int n_elems_root_global = 0;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries.
// Each MPI rank builds only its local row partition plus ghost regions.
void buildSquare2D(World& world, const int n_elems_root) {
    world.n_elems_root_global = n_elems_root;
    
    // Compute 1D row-based partition
    const int n_ranks = world.n_ranks;
    const int rank = world.rank;
    const int rows_per_rank = n_elems_root / n_ranks;
    const int remainder = n_elems_root % n_ranks;
    
    world.start_row = rank * rows_per_rank + std::min(rank, remainder);
    const int extra = (rank < remainder) ? 1 : 0;
    world.end_row = world.start_row + rows_per_rank + extra;
    const int local_rows = world.end_row - world.start_row;
    const int local_elems = local_rows * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate local elements + space for 2 ghost rows (top and bottom)
    world.elements_static.resize(local_elems);
    world.elements_dynamic.resize(local_elems + 2 * n_elems_root);
    world.elements_dynamic_swap.resize(local_elems + 2 * n_elems_root);
    
    // Initialize static elements
    for (int i = 0; i < local_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    // Initialize dynamic elements (local + ghost)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        world.elements_dynamic_swap[i].current_energy = 0.0;
        world.elements_dynamic_swap[i].total_flux = 0.0;
    }
    
    // Build connectivity for local elements using local indices
    // Ghost elements are appended after local: [n_local .. n_local+2*n_elems_root)
    // Bottom ghost (x+1 from rank+1): n_local + y
    // Top ghost (x-1 from rank-1):   n_local + n_elems_root + y
    const int bottom_ghost_base = local_elems;
    const int top_ghost_base = local_elems + n_elems_root;
    
    for (int x = world.start_row; x < world.end_row; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int local_idx = (x - world.start_row) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            
            // Down neighbor (x+1, y)  -- original offsets[0]
            if (x + 1 < n_elems_root) {
                if (x + 1 < world.end_row) {
                    elem.connected_idx[elem.num_connections] = local_idx + n_elems_root;
                } else {
                    elem.connected_idx[elem.num_connections] = bottom_ghost_base + y;
                }
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
            
            // Up neighbor (x-1, y)  -- original offsets[1]
            if (x - 1 >= 0) {
                if (x - 1 >= world.start_row) {
                    elem.connected_idx[elem.num_connections] = local_idx - n_elems_root;
                } else {
                    elem.connected_idx[elem.num_connections] = top_ghost_base + y;
                }
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
            
            // Right neighbor (x, y+1)  -- original offsets[2]
            if (y + 1 < n_elems_root) {
                elem.connected_idx[elem.num_connections] = local_idx + 1;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
            
            // Left neighbor (x, y-1)  -- original offsets[3]
            if (y - 1 >= 0) {
                elem.connected_idx[elem.num_connections] = local_idx - 1;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    if (world.start_row <= 0 && 0 < world.end_row) {
        world.elements_static[0].material_idx = INFLOW_MAT_ID;                    // (0, 0)
        world.elements_static[n_elems_root - 1].material_idx = OUTFLOW_MAT_ID;    // (0, last)
    }
    if (world.start_row <= last && last < world.end_row) {
        const int row_offset = (last - world.start_row) * n_elems_root;
        world.elements_static[row_offset].material_idx = OUTFLOW_MAT_ID;          // (last, 0)
        world.elements_static[row_offset + last].material_idx = INFLOW_MAT_ID;    // (last, last)
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations using MPI ghost exchange
void runSimulation(World& world, const int n_iters) {
    const size_t n_local = world.elements_static.size();
    const int n_elems_root = world.n_elems_root_global;
    const int rank = world.rank;
    const int n_ranks = world.n_ranks;
    const int bottom_ghost_base = static_cast<int>(n_local);
    const int top_ghost_base = static_cast<int>(n_local) + n_elems_root;
    const int row_stride = n_elems_root;
    const int local_rows = (n_local > 0) ? static_cast<int>(n_local) / n_elems_root : 0;
    
    MPI_Status status;
    const int elem_size = static_cast<int>(sizeof(ElementDynamic));
    const int row_bytes = n_elems_root * elem_size;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange ghost rows with neighbor ranks.
        // Uses same tag 0 for both exchanges (sequential, different partners).
        // Ranks with 0 local rows use dummy buffers so neighbors don't deadlock.
        if (rank < n_ranks - 1) {
            // Bottom ghost: send my last row to rank+1, receive their first row
            void* send_ptr;
            std::vector<char> dummy_send;
            if (local_rows > 0) {
                send_ptr = &world.elements_dynamic[(local_rows - 1) * row_stride];
            } else {
                dummy_send.resize(row_bytes, 0);
                send_ptr = dummy_send.data();
            }
            MPI_Sendrecv(
                send_ptr, row_bytes, MPI_BYTE,
                rank + 1, 0,
                &world.elements_dynamic[bottom_ghost_base],
                row_bytes, MPI_BYTE,
                rank + 1, 0,
                MPI_COMM_WORLD, &status
            );
        }
        
        if (rank > 0) {
            // Top ghost: send my first row to rank-1, receive their last row
            void* send_ptr;
            std::vector<char> dummy_send;
            if (local_rows > 0) {
                send_ptr = &world.elements_dynamic[0];
            } else {
                dummy_send.resize(row_bytes, 0);
                send_ptr = dummy_send.data();
            }
            MPI_Sendrecv(
                send_ptr, row_bytes, MPI_BYTE,
                rank - 1, 0,
                &world.elements_dynamic[top_ghost_base],
                row_bytes, MPI_BYTE,
                rank - 1, 0,
                MPI_COMM_WORLD, &status
            );
        }
        
        // Update all local elements (reads from elements_dynamic including ghosts)
        for (size_t i = 0; i < n_local; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements (local or ghost)
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
        
        // Swap buffers (both have same size: n_local + 2*n_elems_root)
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results with MPI reductions
bool validateResults(const World& world) {
    const size_t n_local = world.elements_static.size();
    
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    // Only iterate over local elements (ignore ghost region)
    for (size_t i = 0; i < n_local; ++i) {
        energy_sum += world.elements_dynamic[i].current_energy;
        flux_sum += world.elements_dynamic[i].total_flux;
        energy_max = std::max(world.elements_dynamic[i].current_energy, energy_max);
        energy_min = std::min(world.elements_dynamic[i].current_energy, energy_min);
    }
    
    // Global reductions
    val_t global_esum = 0.0, global_fsum = 0.0;
    val_t global_emax = 0.0, global_emin = 0.0;
    MPI_Reduce(&energy_sum, &global_esum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&flux_sum, &global_fsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_max, &global_emax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_min, &global_emin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    if (world.rank != 0) return true;
    
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", global_esum);
    printf("  Flux sum: %.2f\n", global_fsum);
    printf("  Energy range: [%.6f, %.6f]\n", global_emin, global_emax);
    
    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;
    
    if (!std::isfinite(global_esum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    
    if (std::abs(global_esum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        // Don't fail validation as this can happen with external flows
    }
    
    if (!std::isfinite(global_fsum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    
    if (!std::isfinite(global_emax) || !std::isfinite(global_emin)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }
    
    printf("  Validation: PASSED\n");
    return true;
}

// Compute a simple hash of the results for verification (parallel XOR reduction)
uint64_t computeHash(const World& world) {
    const size_t n_local = world.elements_static.size();
    const int n_elems_root = world.n_elems_root_global;
    const int start_row = world.start_row;
    const int local_rows = static_cast<int>(n_local / n_elems_root);
    
    uint64_t hash = 0;
    for (int r = 0; r < local_rows; ++r) {
        const int row_offset = r * n_elems_root;
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t idx = static_cast<size_t>(row_offset + y);
            const uint64_t global_i = static_cast<uint64_t>(start_row + r) * n_elems_root + y;
            // Simple hash combining energy and flux values
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[idx].current_energy);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[idx].total_flux);
            hash ^= (*e_ptr + global_i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (*f_ptr + global_i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    
    // XOR-reduce across all ranks (XOR is associative/commutative, so this matches sequential)
    uint64_t global_hash = 0;
    MPI_Reduce(&hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, MPI_COMM_WORLD);
    
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
    
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    int should_exit = 0;
    
    // Parse command line arguments (only rank 0 parses, then broadcasts)
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
                should_exit = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                should_exit = -1;
            }
        }
    }
    
    // Broadcast configuration to all ranks before any early return
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&should_exit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Handle help/error exit on all ranks
    if (should_exit != 0) {
        MPI_Finalize();
        return (should_exit < 0) ? 1 : 0;
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("MPI processes: %d\n", n_ranks);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh (each rank builds its local portion)
    if (rank == 0) printf("Building unstructured mesh (distributed)...\n");
    World world;
    world.rank = rank;
    world.n_ranks = n_ranks;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t local_static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t local_dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t local_total = local_static_mem + local_dynamic_mem;
    
    if (rank == 0) {
        printf("Memory per rank: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               local_total / (1024.0 * 1024.0),
               local_static_mem / (1024.0 * 1024.0),
               local_dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    
    // Synchronize and measure wall-clock time
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();
    
    runSimulation(world, n_iters);
    
    const double t_end = MPI_Wtime();
    const double local_duration_ms = (t_end - t_start) * 1000.0;
    
    // Take the max duration across all ranks (wall-clock time)
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification (parallel XOR reduction)
    const uint64_t hash = computeHash(world);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation (gather to rank 0)
    if (printResults) {
        const size_t n_local = world.elements_static.size();
        int my_count = static_cast<int>(n_local);
        std::vector<int> all_counts(n_ranks);
        MPI_Gather(&my_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        std::vector<int> displs(n_ranks, 0);
        std::vector<double> all_energies;
        if (rank == 0) {
            int total = 0;
            for (int i = 0; i < n_ranks; ++i) {
                displs[i] = total;
                total += all_counts[i];
            }
            all_energies.resize(total);
        }
        
        std::vector<double> local_energies(n_local);
        for (size_t i = 0; i < n_local; ++i) {
            local_energies[i] = world.elements_dynamic[i].current_energy;
        }
        
        MPI_Gatherv(local_energies.data(), my_count, MPI_DOUBLE,
                    all_energies.data(), all_counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            print_results(all_energies, "ElementEnergy");
        }
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world);
        int valid_int = valid ? 1 : 0;
        MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!valid_int) {
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
