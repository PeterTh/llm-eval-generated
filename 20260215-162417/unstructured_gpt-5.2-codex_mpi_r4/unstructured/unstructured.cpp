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
    int n_elems_root = 0;
    idx_t global_elems = 0;
    int start_row = 0;
    int local_rows = 0;
    idx_t local_start = 0;
    idx_t local_elems = 0;
    int rank = 0;
    int size = 1;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int size) {
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * n_elems_root;
    const int base_rows = n_elems_root / size;
    const int extra_rows = n_elems_root % size;
    const int local_rows = base_rows + (rank < extra_rows ? 1 : 0);
    const int start_row = rank * base_rows + std::min(rank, extra_rows);
    const idx_t local_elems = static_cast<idx_t>(local_rows) * n_elems_root;

    world.n_elems_root = n_elems_root;
    world.global_elems = n_elems;
    world.start_row = start_row;
    world.local_rows = local_rows;
    world.local_start = static_cast<idx_t>(start_row) * n_elems_root;
    world.local_elems = local_elems;
    world.rank = rank;
    world.size = size;
    
    // Initialize materials
    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(local_elems);
    world.elements_dynamic.resize(local_elems);
    world.elements_dynamic_swap.resize(local_elems);
    
    // Initialize all elements with default material and zero energy
    for (idx_t i = 0; i < local_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int local_x = 0; local_x < local_rows; ++local_x) {
        const int global_x = start_row + local_x;
        for (int y = 0; y < n_elems_root; ++y) {
            const idx_t local_idx = static_cast<idx_t>(local_x) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = global_x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const idx_t neighbor_idx = static_cast<idx_t>(nx) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }

            const int last = n_elems_root - 1;
            if (global_x == 0 && y == 0) {
                elem.material_idx = INFLOW_MAT_ID;
            } else if (global_x == 0 && y == last) {
                elem.material_idx = OUTFLOW_MAT_ID;
            } else if (global_x == last && y == 0) {
                elem.material_idx = OUTFLOW_MAT_ID;
            } else if (global_x == last && y == last) {
                elem.material_idx = INFLOW_MAT_ID;
            }
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, val_t this_energy,
                        val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, MPI_Comm comm) {
    const idx_t n_elems = world.local_elems;
    const int n_elems_root = world.n_elems_root;
    const idx_t local_start = world.local_start;
    const int start_row = world.start_row;
    const int end_row = start_row + world.local_rows;
    const bool has_up = start_row > 0;
    const bool has_down = end_row < n_elems_root;
    const int rank = world.rank;
    const int tag_to_up = 100;
    const int tag_to_down = 101;

    std::vector<val_t> halo_upper(has_up ? n_elems_root : 0);
    std::vector<val_t> halo_lower(has_down ? n_elems_root : 0);
    std::vector<val_t> send_upper(has_up ? n_elems_root : 0);
    std::vector<val_t> send_lower(has_down ? n_elems_root : 0);
    
    for (int iter = 0; iter < n_iters; ++iter) {
        if (n_elems > 0 && (has_up || has_down)) {
            if (has_up) {
                for (int y = 0; y < n_elems_root; ++y) {
                    send_upper[y] = world.elements_dynamic[y].current_energy;
                }
                MPI_Sendrecv(send_upper.data(), n_elems_root, MPI_DOUBLE, rank - 1, tag_to_up,
                             halo_upper.data(), n_elems_root, MPI_DOUBLE, rank - 1, tag_to_down,
                             comm, MPI_STATUS_IGNORE);
            }
            if (has_down) {
                const idx_t base = static_cast<idx_t>(world.local_rows - 1) * n_elems_root;
                for (int y = 0; y < n_elems_root; ++y) {
                    send_lower[y] = world.elements_dynamic[base + y].current_energy;
                }
                MPI_Sendrecv(send_lower.data(), n_elems_root, MPI_DOUBLE, rank + 1, tag_to_down,
                             halo_lower.data(), n_elems_root, MPI_DOUBLE, rank + 1, tag_to_up,
                             comm, MPI_STATUS_IGNORE);
            }
        }

        // Update all elements
        for (idx_t i = 0; i < n_elems; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                val_t neighbor_energy = 0.0;
                if (neighbor_idx >= local_start && neighbor_idx < local_start + n_elems) {
                    neighbor_energy = world.elements_dynamic[neighbor_idx - local_start].current_energy;
                } else {
                    const int neighbor_row = static_cast<int>(neighbor_idx / n_elems_root);
                    const int neighbor_col = static_cast<int>(neighbor_idx - static_cast<idx_t>(neighbor_row) * n_elems_root);
                    if (neighbor_row < start_row) {
                        neighbor_energy = halo_upper[neighbor_col];
                    } else {
                        neighbor_energy = halo_lower[neighbor_col];
                    }
                }
                total_flux += computeFlux(mat, elem_dyn.current_energy, elem_static.connected_flux[j], neighbor_energy);
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
bool validateResults(const World& world, MPI_Comm comm) {
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

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);

    int valid = 1;
    if (world.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    
        // Check for numerical issues
        constexpr val_t energy_epsilon = 1e-8;
        
        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        }
        
        if (std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            // Don't fail validation as this can happen with external flows
        }
        
        if (!std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }
        
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = 0;
        }
        
        if (valid) {
            printf("  Validation: PASSED\n");
        }
    }

    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    return valid != 0;
}

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements, idx_t base_index) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        const uint64_t global_i = static_cast<uint64_t>(base_index + i);
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
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
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    bool show_help = false;
    bool bad_option = false;
    const char* unknown_option = nullptr;
    
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
            show_help = true;
        } else {
            bad_option = true;
            if (unknown_option == nullptr) {
                unknown_option = argv[i];
            }
        }
    }

    if (show_help) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (bad_option) {
        if (rank == 0) {
            if (unknown_option != nullptr) {
                printf("Unknown option: %s\n", unknown_option);
            } else {
                printf("Unknown option in command line arguments\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
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
    buildSquare2D(world, n_elems_root, rank, size);
    
    // Calculate memory usage
    const size_t static_mem = static_cast<size_t>(world.global_elems) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(world.global_elems) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) {
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
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    runSimulation(world, n_iters, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double local_duration_ms = (end - start) * 1000.0;
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration_ms);
    }
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = duration_ms / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    if (rank == 0) {
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification
    const uint64_t local_hash = computeHash(world.elements_dynamic, world.local_start);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
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
        const int local_count = static_cast<int>(local_energy.size());
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> energyData;
        if (rank == 0) {
            counts.resize(size);
        }
        MPI_Gather(&local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            displs.resize(size);
            displs[0] = 0;
            for (int r = 1; r < size; ++r) {
                displs[r] = displs[r - 1] + counts[r - 1];
            }
            energyData.resize(displs[size - 1] + counts[size - 1]);
        }
        MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE,
                    rank == 0 ? energyData.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }
    
    // Validation
    int exit_code = 0;
    if (validate) {
        bool valid = validateResults(world, MPI_COMM_WORLD);
        if (!valid) {
            exit_code = 1;
        }
    }
    
    MPI_Finalize();
    return exit_code;
}
