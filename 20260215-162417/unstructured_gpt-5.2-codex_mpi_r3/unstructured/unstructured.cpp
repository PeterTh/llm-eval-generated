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
};

struct Decomposition {
    int n_elems_root = 0;
    int row_start = 0;
    int row_count = 0;
    int rank = 0;
    int size = 1;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

Decomposition buildDecomposition(const int n_elems_root, MPI_Comm comm) {
    Decomposition decomp;
    decomp.n_elems_root = n_elems_root;
    MPI_Comm_rank(comm, &decomp.rank);
    MPI_Comm_size(comm, &decomp.size);
    const int base = n_elems_root / decomp.size;
    const int rem = n_elems_root % decomp.size;
    decomp.row_count = base + (decomp.rank < rem ? 1 : 0);
    decomp.row_start = decomp.rank * base + std::min(decomp.rank, rem);
    return decomp;
}

inline idx_t globalIndex(const int row, const int col, const int n_elems_root) {
    return static_cast<idx_t>(row) * static_cast<idx_t>(n_elems_root) + static_cast<idx_t>(col);
}

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const Decomposition& decomp) {
    const int n_elems_root = decomp.n_elems_root;
    const int row_start = decomp.row_start;
    const int row_end = row_start + decomp.row_count;
    const int n_elems_local = decomp.row_count * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems_local);
    world.elements_dynamic.resize(n_elems_local);
    world.elements_dynamic_swap.resize(n_elems_local);
    
    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_elems_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = row_start; x < row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int local_row = x - row_start;
            const int idx = local_row * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const idx_t neighbor_idx = globalIndex(nx, ny, n_elems_root);
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    const int corner_rows[4] = {0, 0, last, last};
    const int corner_cols[4] = {0, last, 0, last};
    const idx_t corner_mats[4] = {INFLOW_MAT_ID, OUTFLOW_MAT_ID, OUTFLOW_MAT_ID, INFLOW_MAT_ID};
    for (int i = 0; i < 4; ++i) {
        const int cx = corner_rows[i];
        if (cx < row_start || cx >= row_end) {
            continue;
        }
        const int local_row = cx - row_start;
        const int idx = local_row * n_elems_root + corner_cols[i];
        world.elements_static[idx].material_idx = corner_mats[i];
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, val_t this_energy,
                        val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

inline val_t neighborEnergy(const std::vector<ElementDynamic>& local_elements,
                            const std::vector<val_t>& halo_up,
                            const std::vector<val_t>& halo_down,
                            const Decomposition& decomp,
                            const idx_t neighbor_idx) {
    const int n_elems_root = decomp.n_elems_root;
    const int row_end = decomp.row_start + decomp.row_count;
    const int row = static_cast<int>(neighbor_idx / static_cast<idx_t>(n_elems_root));
    const int col = static_cast<int>(neighbor_idx - static_cast<idx_t>(row) * n_elems_root);
    if (row >= decomp.row_start && row < row_end) {
        const size_t local_idx = static_cast<size_t>(row - decomp.row_start) * n_elems_root + col;
        return local_elements[local_idx].current_energy;
    }
    if (row == decomp.row_start - 1) {
        return halo_up[col];
    }
    if (row == row_end) {
        return halo_down[col];
    }
    return 0.0;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const Decomposition& decomp, const int n_iters, MPI_Comm comm) {
    const size_t n_elems = world.elements_static.size();
    const int n_elems_root = decomp.n_elems_root;
    const int row_end = decomp.row_start + decomp.row_count;
    const int neighbor_up = (decomp.row_count > 0 && decomp.row_start > 0) ? decomp.rank - 1 : MPI_PROC_NULL;
    const int neighbor_down = (decomp.row_count > 0 && row_end < n_elems_root) ? decomp.rank + 1 : MPI_PROC_NULL;
    std::vector<val_t> halo_up(n_elems_root);
    std::vector<val_t> halo_down(n_elems_root);
    std::vector<val_t> send_up(n_elems_root);
    std::vector<val_t> send_down(n_elems_root);
    
    for (int iter = 0; iter < n_iters; ++iter) {
        if (decomp.row_count > 0) {
            if (neighbor_up != MPI_PROC_NULL) {
                for (int col = 0; col < n_elems_root; ++col) {
                    send_up[col] = world.elements_dynamic[col].current_energy;
                }
                MPI_Sendrecv(send_up.data(), n_elems_root, MPI_DOUBLE, neighbor_up, 0,
                             halo_up.data(), n_elems_root, MPI_DOUBLE, neighbor_up, 1,
                             comm, MPI_STATUS_IGNORE);
            }
            if (neighbor_down != MPI_PROC_NULL) {
                const size_t last_row_offset = static_cast<size_t>(decomp.row_count - 1) * n_elems_root;
                for (int col = 0; col < n_elems_root; ++col) {
                    send_down[col] = world.elements_dynamic[last_row_offset + col].current_energy;
                }
                MPI_Sendrecv(send_down.data(), n_elems_root, MPI_DOUBLE, neighbor_down, 1,
                             halo_down.data(), n_elems_root, MPI_DOUBLE, neighbor_down, 0,
                             comm, MPI_STATUS_IGNORE);
            }
        }

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
                const val_t neighbor_energy = neighborEnergy(world.elements_dynamic,
                                                            halo_up,
                                                            halo_down,
                                                            decomp,
                                                            neighbor_idx);
                total_flux += computeFlux(mat,
                                          elem_dyn.current_energy,
                                          elem_static.connected_flux[j],
                                          neighbor_energy);
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
bool validateResults(const World& world, const Decomposition& decomp, MPI_Comm comm) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    int local_finite = 1;
    
    for (const auto& elem : world.elements_dynamic) {
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(elem.current_energy, local_energy_max);
        local_energy_min = std::min(elem.current_energy, local_energy_min);
        if (!std::isfinite(elem.current_energy) || !std::isfinite(elem.total_flux)) {
            local_finite = 0;
        }
    }

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    int global_finite = 0;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local_finite, &global_finite, 1, MPI_INT, MPI_LAND, 0, comm);
    
    int global_valid = 1;
    if (decomp.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
        
        // Check for numerical issues
        constexpr val_t energy_epsilon = 1e-8;
        
        if (!global_finite || !std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            global_valid = 0;
        }
        
        if (std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            // Don't fail validation as this can happen with external flows
        }
        
        if (!std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            global_valid = 0;
        }
        
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            global_valid = 0;
        }
        
        if (global_valid) {
            printf("  Validation: PASSED\n");
        }
    }

    MPI_Bcast(&global_valid, 1, MPI_INT, 0, comm);
    return global_valid == 1;
}

// Compute a simple hash of the results for verification
uint64_t computeLocalHash(const std::vector<ElementDynamic>& elements, const Decomposition& decomp) {
    uint64_t hash = 0;
    const int n_elems_root = decomp.n_elems_root;
    for (int local_row = 0; local_row < decomp.row_count; ++local_row) {
        const int global_row = decomp.row_start + local_row;
        const size_t row_offset = static_cast<size_t>(local_row) * n_elems_root;
        for (int col = 0; col < n_elems_root; ++col) {
            const size_t local_idx = row_offset + col;
            const idx_t global_idx = globalIndex(global_row, col, n_elems_root);
            // Simple hash combining energy and flux values
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[local_idx].current_energy);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[local_idx].total_flux);
            hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
            hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
        }
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
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

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
    const Decomposition decomp = buildDecomposition(n_elems_root, comm);
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
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
    buildSquare2D(world, decomp);
    
    // Calculate memory usage
    if (rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation
    if (rank == 0) {
        printf("Running simulation...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    runSimulation(world, decomp, n_iters, comm);
    MPI_Barrier(comm);
    const double end = MPI_Wtime();
    const double local_duration_ms = (end - start) * 1000.0;
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration_ms + 0.5));
    }
    
    if (rank == 0) {
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
    
    // Compute hash for verification
    const uint64_t local_hash = computeLocalHash(world.elements_dynamic, decomp);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> local_energy;
        local_energy.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            local_energy.push_back(elem.current_energy);
        }
        std::vector<int> counts(decomp.size);
        std::vector<int> displs(decomp.size);
        for (int r = 0; r < decomp.size; ++r) {
            const int base = n_elems_root / decomp.size;
            const int rem = n_elems_root % decomp.size;
            const int rows = base + (r < rem ? 1 : 0);
            counts[r] = rows * n_elems_root;
            displs[r] = (r == 0) ? 0 : (displs[r - 1] + counts[r - 1]);
        }

        std::vector<double> energyData;
        if (rank == 0) {
            energyData.resize(static_cast<size_t>(n_elems));
        }
        MPI_Gatherv(local_energy.data(),
                    static_cast<int>(local_energy.size()),
                    MPI_DOUBLE,
                    rank == 0 ? energyData.data() : nullptr,
                    counts.data(),
                    displs.data(),
                    MPI_DOUBLE,
                    0,
                    comm);
        if (rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }
    
    // Validation
    if (validate) {
        const bool valid = validateResults(world, decomp, comm);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
