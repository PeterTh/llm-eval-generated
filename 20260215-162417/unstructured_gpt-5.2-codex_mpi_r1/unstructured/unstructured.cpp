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

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

int rowsForRank(const int rank, const int size, const int n_elems_root) {
    const int base = n_elems_root / size;
    const int rem = n_elems_root % size;
    return base + (rank < rem ? 1 : 0);
}

int startRowForRank(const int rank, const int size, const int n_elems_root) {
    const int base = n_elems_root / size;
    const int rem = n_elems_root % size;
    return rank * base + std::min(rank, rem);
}

int prevRankWithRows(const int rank, const int size, const int n_elems_root) {
    for (int r = rank - 1; r >= 0; --r) {
        if (rowsForRank(r, size, n_elems_root) > 0) {
            return r;
        }
    }
    return MPI_PROC_NULL;
}

int nextRankWithRows(const int rank, const int size, const int n_elems_root) {
    for (int r = rank + 1; r < size; ++r) {
        if (rowsForRank(r, size, n_elems_root) > 0) {
            return r;
        }
    }
    return MPI_PROC_NULL;
}

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int start_row, const int local_rows) {
    const int n_elems = local_rows * n_elems_root;
    
    // Initialize materials
    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);
    
    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    const int last = n_elems_root - 1;
    const idx_t last_idx = static_cast<idx_t>(last);
    const idx_t last_row_offset = last_idx * n_elems_root;
    for (int local_x = 0; local_x < local_rows; ++local_x) {
        const int x = start_row + local_x;
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = local_x * n_elems_root + y;
            const idx_t global_idx = static_cast<idx_t>(x) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const idx_t neighbor_idx = static_cast<idx_t>(nx) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
            
            if (global_idx == 0 || global_idx == last_idx ||
                global_idx == last_row_offset || global_idx == last_row_offset + last_idx) {
                elem.material_idx = (global_idx == 0 || global_idx == last_row_offset + last_idx)
                    ? INFLOW_MAT_ID
                    : OUTFLOW_MAT_ID;
            }
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const val_t this_energy,
                        const val_t connection_flux, const val_t other_energy) {
    return (other_energy - this_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

inline val_t lookupEnergy(const World& world, const idx_t neighbor_idx, const idx_t global_start,
                          const idx_t global_end, const int n_elems_root, const int start_row,
                          const std::vector<val_t>& ghost_north,
                          const std::vector<val_t>& ghost_south) {
    if (neighbor_idx >= global_start && neighbor_idx < global_end) {
        return world.elements_dynamic[neighbor_idx - global_start].current_energy;
    }

    const int neighbor_row = static_cast<int>(neighbor_idx / n_elems_root);
    const int neighbor_col = static_cast<int>(neighbor_idx % n_elems_root);
    if (neighbor_row < start_row) {
        return ghost_north[neighbor_col];
    }
    return ghost_south[neighbor_col];
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, const int n_elems_root, const int start_row,
                   const int local_rows, const int north_rank, const int south_rank,
                   MPI_Comm comm) {
    const size_t n_elems = world.elements_static.size();
    const idx_t global_start = static_cast<idx_t>(start_row) * n_elems_root;
    const idx_t global_end = global_start + n_elems;
    std::vector<val_t> ghost_north(n_elems_root, 0.0);
    std::vector<val_t> ghost_south(n_elems_root, 0.0);
    std::vector<val_t> send_north(n_elems_root, 0.0);
    std::vector<val_t> send_south(n_elems_root, 0.0);
    
    for (int iter = 0; iter < n_iters; ++iter) {
        const int row_width = n_elems_root;
        const int exchange_count = local_rows > 0 ? row_width : 0;
        if (local_rows > 0) {
            for (int col = 0; col < row_width; ++col) {
                send_north[col] = world.elements_dynamic[col].current_energy;
                send_south[col] =
                    world.elements_dynamic[(local_rows - 1) * row_width + col].current_energy;
            }
        }

        MPI_Sendrecv(send_north.data(), exchange_count, MPI_DOUBLE, north_rank, 0,
                     ghost_north.data(), exchange_count, MPI_DOUBLE, north_rank, 1, comm,
                     MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_south.data(), exchange_count, MPI_DOUBLE, south_rank, 1,
                     ghost_south.data(), exchange_count, MPI_DOUBLE, south_rank, 0, comm,
                     MPI_STATUS_IGNORE);

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
                const val_t neighbor_energy = lookupEnergy(
                    world, neighbor_idx, global_start, global_end, n_elems_root, start_row,
                    ghost_north, ghost_south);
                total_flux += computeFlux(mat, elem_dyn.current_energy,
                                          elem_static.connected_flux[j], neighbor_energy);
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
bool validateResults(const World& world, MPI_Comm comm, const int rank) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : world.elements_dynamic) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }

    val_t global_energy_sum = 0.0;
    val_t global_flux_sum = 0.0;
    val_t global_energy_max = 0.0;
    val_t global_energy_min = 0.0;
    MPI_Allreduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    
    bool valid = true;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_energy_sum);
        printf("  Flux sum: %.2f\n", global_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);
        
        // Check for numerical issues
        constexpr val_t energy_epsilon = 1e-8;
        
        if (!std::isfinite(global_energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = false;
        }
        
        if (std::abs(global_energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            // Don't fail validation as this can happen with external flows
        }
        
        if (!std::isfinite(global_flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = false;
        }
        
        if (!std::isfinite(global_energy_max) || !std::isfinite(global_energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = false;
        }
        
        if (valid) {
            printf("  Validation: PASSED\n");
        }
    }

    int valid_int = valid ? 1 : 0;
    MPI_Bcast(&valid_int, 1, MPI_INT, 0, comm);
    return valid_int != 0;
}

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements, const int n_elems_root,
                     const int start_row, MPI_Comm comm) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        const idx_t global_idx =
            static_cast<idx_t>(start_row + static_cast<int>(i / n_elems_root)) * n_elems_root +
            static_cast<int>(i % n_elems_root);
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t global_hash = 0;
    MPI_Allreduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, comm);
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
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    int parse_status = 0;
    
    // Parse command line arguments
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
                parse_status = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parse_status = 2;
                break;
            }
        }
    }

    MPI_Bcast(&parse_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parse_status != 0) {
        MPI_Finalize();
        return parse_status == 1 ? 0 : 1;
    }

    int validate_int = validate ? 1 : 0;
    int printResults_int = printResults ? 1 : 0;
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = validate_int != 0;
    printResults = printResults_int != 0;
    
    const int n_elems = n_elems_root * n_elems_root;
    const int local_rows = rowsForRank(rank, size, n_elems_root);
    const int start_row = startRowForRank(rank, size, n_elems_root);
    const int north_rank =
        local_rows > 0 ? prevRankWithRows(rank, size, n_elems_root) : MPI_PROC_NULL;
    const int south_rank =
        local_rows > 0 ? nextRankWithRows(rank, size, n_elems_root) : MPI_PROC_NULL;
    
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
    buildSquare2D(world, n_elems_root, start_row, local_rows);
    
    // Calculate memory usage
    const unsigned long long local_static_mem =
        static_cast<unsigned long long>(world.elements_static.size()) * sizeof(ElementStatic);
    const unsigned long long local_dynamic_mem =
        static_cast<unsigned long long>(world.elements_dynamic.size()) * sizeof(ElementDynamic) * 2;
    unsigned long long global_static_mem = 0;
    unsigned long long global_dynamic_mem = 0;
    MPI_Reduce(&local_static_mem, &global_static_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_dynamic_mem, &global_dynamic_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        const unsigned long long total_mem = global_static_mem + global_dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               global_static_mem / (1024.0 * 1024.0),
               global_dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation
    if (rank == 0) {
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters, n_elems_root, start_row, local_rows, north_rank, south_rank,
                  MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    const double local_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double max_duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_duration_ms);
        printf("Computation time: %ld ms\n", duration_ms);
    
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = max_duration_ms / n_measured_iters;
        const double giga_elems_per_sec =
            (n_measured_iters * static_cast<double>(n_elems)) / (max_duration_ms / 1000.0) / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification
    const uint64_t hash =
        computeHash(world.elements_dynamic, n_elems_root, start_row, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> energyData;
        energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            energyData.push_back(elem.current_energy);
        }

        const int local_count = static_cast<int>(energyData.size());
        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            counts.resize(size);
        }
        MPI_Gather(&local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0,
                   MPI_COMM_WORLD);
        std::vector<double> globalEnergy;
        if (rank == 0) {
            displs.resize(size);
            displs[0] = 0;
            for (int r = 1; r < size; ++r) {
                displs[r] = displs[r - 1] + counts[r - 1];
            }
            globalEnergy.resize(static_cast<size_t>(n_elems));
        }
        MPI_Gatherv(energyData.data(), local_count, MPI_DOUBLE,
                    rank == 0 ? globalEnergy.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(globalEnergy, "ElementEnergy");
        }
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world, MPI_COMM_WORLD, rank);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
