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

static_assert(sizeof(ElementDynamic) == sizeof(val_t) * 2, "ElementDynamic must be two doubles");

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    idx_t n_elems_root = 0;
    idx_t n_elems = 0;
    idx_t row_start = 0;
    idx_t row_end = 0;
    idx_t local_rows = 0;
    idx_t local_elems = 0;
};

struct RowPartition {
    idx_t row_start;
    idx_t row_end;
    idx_t local_rows;
};

RowPartition computeRowPartition(const int n_rows, const int size, const int rank) {
    const idx_t base_rows = static_cast<idx_t>(n_rows / size);
    const idx_t remainder = static_cast<idx_t>(n_rows % size);
    RowPartition part{};
    if (static_cast<idx_t>(rank) < remainder) {
        part.local_rows = base_rows + 1;
        part.row_start = static_cast<idx_t>(rank) * (base_rows + 1);
    } else {
        part.local_rows = base_rows;
        part.row_start = remainder * (base_rows + 1) + static_cast<idx_t>(rank - remainder) * base_rows;
    }
    part.row_end = part.row_start + part.local_rows;
    return part;
}

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int size) {
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);
    const RowPartition part = computeRowPartition(n_elems_root, size, rank);
    world.n_elems_root = static_cast<idx_t>(n_elems_root);
    world.n_elems = n_elems;
    world.row_start = part.row_start;
    world.row_end = part.row_end;
    world.local_rows = part.local_rows;
    world.local_elems = part.local_rows * static_cast<idx_t>(n_elems_root);
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(world.local_elems);
    world.elements_dynamic.resize(world.local_elems);
    world.elements_dynamic_swap.resize(world.local_elems);
    
    // Initialize all elements with default material and zero energy
    for (idx_t i = 0; i < world.local_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (idx_t local_x = 0; local_x < world.local_rows; ++local_x) {
        const int x = static_cast<int>(world.row_start + local_x);
        for (int y = 0; y < n_elems_root; ++y) {
            const idx_t idx = local_x * static_cast<idx_t>(n_elems_root) + static_cast<idx_t>(y);
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const idx_t neighbor_idx =
                        static_cast<idx_t>(nx) * static_cast<idx_t>(n_elems_root) + static_cast<idx_t>(ny);
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const idx_t last = static_cast<idx_t>(n_elems_root - 1);
    auto set_material_if_local = [&](idx_t x, idx_t y, idx_t mat_id) {
        if (x >= world.row_start && x < world.row_end) {
            const idx_t local_x = x - world.row_start;
            const idx_t local_idx = local_x * static_cast<idx_t>(n_elems_root) + y;
            world.elements_static[local_idx].material_idx = mat_id;
        }
    };
    set_material_if_local(0, 0, INFLOW_MAT_ID);
    set_material_if_local(0, last, OUTFLOW_MAT_ID);
    set_material_if_local(last, 0, OUTFLOW_MAT_ID);
    set_material_if_local(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, const int rank, const int size) {
    (void)size;
    const idx_t n_elems_root = world.n_elems_root;
    const idx_t local_rows = world.local_rows;
    const size_t local_elems = world.elements_static.size();
    const bool has_prev = (local_rows > 0) && (world.row_start > 0);
    const bool has_next = (local_rows > 0) && (world.row_end < world.n_elems_root);
    const int prev_rank = rank - 1;
    const int next_rank = rank + 1;
    constexpr int TAG_ROW_UP = 100;
    constexpr int TAG_ROW_DOWN = 200;

    MPI_Datatype elem_dynamic_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &elem_dynamic_type);
    MPI_Type_commit(&elem_dynamic_type);

    std::vector<ElementDynamic> ghost_top;
    std::vector<ElementDynamic> ghost_bottom;
    if (has_prev) {
        ghost_top.resize(n_elems_root);
    }
    if (has_next) {
        ghost_bottom.resize(n_elems_root);
    }
    
    for (int iter = 0; iter < n_iters; ++iter) {
        if (has_prev) {
            MPI_Sendrecv(world.elements_dynamic.data(),
                         static_cast<int>(n_elems_root),
                         elem_dynamic_type,
                         prev_rank,
                         TAG_ROW_UP,
                         ghost_top.data(),
                         static_cast<int>(n_elems_root),
                         elem_dynamic_type,
                         prev_rank,
                         TAG_ROW_DOWN,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
        }
        if (has_next) {
            MPI_Sendrecv(world.elements_dynamic.data() +
                             static_cast<size_t>(local_rows - 1) * n_elems_root,
                         static_cast<int>(n_elems_root),
                         elem_dynamic_type,
                         next_rank,
                         TAG_ROW_DOWN,
                         ghost_bottom.data(),
                         static_cast<int>(n_elems_root),
                         elem_dynamic_type,
                         next_rank,
                         TAG_ROW_UP,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
        }

        // Update all elements
        for (size_t i = 0; i < local_elems; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const idx_t neighbor_row = neighbor_idx / n_elems_root;
                const idx_t neighbor_col = neighbor_idx - neighbor_row * n_elems_root;
                const ElementDynamic* neighbor_dyn = nullptr;
                if (neighbor_row >= world.row_start && neighbor_row < world.row_end) {
                    const idx_t local_row = neighbor_row - world.row_start;
                    const idx_t neighbor_local = local_row * n_elems_root + neighbor_col;
                    neighbor_dyn = &world.elements_dynamic[neighbor_local];
                } else if (neighbor_row < world.row_start) {
                    neighbor_dyn = &ghost_top[neighbor_col];
                } else {
                    neighbor_dyn = &ghost_bottom[neighbor_col];
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
    }

    MPI_Type_free(&elem_dynamic_type);
}

// Validate simulation results
bool validateResults(const World& world, const int rank, const int size) {
    (void)size;
    val_t energy_sum_local = 0.0;
    val_t flux_sum_local = 0.0;
    val_t energy_max_local = std::numeric_limits<val_t>::lowest();
    val_t energy_min_local = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : world.elements_dynamic) {
        energy_sum_local += elem.current_energy;
        flux_sum_local += elem.total_flux;
        energy_max_local = std::max(elem.current_energy, energy_max_local);
        energy_min_local = std::min(elem.current_energy, energy_min_local);
    }

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = 0.0;
    val_t energy_min = 0.0;
    MPI_Allreduce(&energy_sum_local, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&flux_sum_local, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&energy_max_local, &energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&energy_min_local, &energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    }
    
    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;
    bool valid = true;
    
    if (!std::isfinite(energy_sum)) {
        if (rank == 0) {
            printf("  ERROR: Energy sum is not finite\n");
        }
        valid = false;
    }
    
    if (std::abs(energy_sum) > energy_epsilon) {
        if (rank == 0) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        // Don't fail validation as this can happen with external flows
    }
    
    if (!std::isfinite(flux_sum)) {
        if (rank == 0) {
            printf("  ERROR: Flux sum is not finite\n");
        }
        valid = false;
    }
    
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        if (rank == 0) {
            printf("  ERROR: Energy extrema are not finite\n");
        }
        valid = false;
    }
    
    if (rank == 0) {
        if (valid) {
            printf("  Validation: PASSED\n");
        } else {
            printf("  Validation: FAILED\n");
        }
    }

    int valid_flag = valid ? 1 : 0;
    MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid_flag != 0;
}

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements, const idx_t global_offset) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        const uint64_t global_idx = static_cast<uint64_t>(global_offset + static_cast<idx_t>(i));
        hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
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
    int validate_flag = 0;
    int print_results_flag = 0;
    int show_help = 0;
    int parse_error = 0;
    
    // Parse command line arguments
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_flag = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                print_results_flag = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                show_help = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parse_error = 1;
                break;
            }
        }
    }

    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_results_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&show_help, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&parse_error, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (show_help) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    if (parse_error) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const bool validate = validate_flag != 0;
    const bool printResults = print_results_flag != 0;
    
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
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double duration = end - start;
    double max_duration = 0.0;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    if (rank == 0) {
        const double duration_ms = max_duration * 1000.0;
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * static_cast<double>(n_elems)) / max_duration / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Computation time: %.0f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification
    const uint64_t local_hash =
        computeHash(world.elements_dynamic, world.row_start * world.n_elems_root);
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", global_hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> local_energy(world.elements_dynamic.size());
        for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
            local_energy[i] = world.elements_dynamic[i].current_energy;
        }

        int local_count = static_cast<int>(local_energy.size());
        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            counts.resize(size);
            displs.resize(size);
        }
        int* counts_ptr = rank == 0 ? counts.data() : nullptr;
        int* displs_ptr = rank == 0 ? displs.data() : nullptr;
        MPI_Gather(&local_count, 1, MPI_INT, counts_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<double> global_energy;
        if (rank == 0) {
            int offset = 0;
            for (int r = 0; r < size; ++r) {
                displs[r] = offset;
                offset += counts[r];
            }
            global_energy.resize(static_cast<size_t>(n_elems));
        }
        double* global_ptr = rank == 0 ? global_energy.data() : nullptr;
        MPI_Gatherv(local_energy.data(),
                    local_count,
                    MPI_DOUBLE,
                    global_ptr,
                    counts_ptr,
                    displs_ptr,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(global_energy, "ElementEnergy");
        }
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world, rank, size);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
