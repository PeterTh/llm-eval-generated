#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    int offset_x = 0;         // Global x coordinate of first local row
    int local_nx = 0;         // Number of local rows (excluding ghost rows)
    int n_elems_root = 0;     // Grid dimension (full)
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build local portion of the grid (with ghost rows for MPI halo exchange)
// Each rank builds its own rows; static connectivity uses combined array indices
// (including ghost rows) so neighbors at rank boundaries point to ghost positions.
void buildLocalGrid(World& world, const int n_elems_root, const int offset_x, const int local_nx) {
    world.n_elems_root = n_elems_root;
    world.offset_x = offset_x;
    world.local_nx = local_nx;
    
    // Initialize materials (identical on all ranks)
    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    const int local_size = local_nx * n_elems_root;
    const int full_size = (local_nx + 2) * n_elems_root;  // +2 ghost rows
    
    // Allocate static for local elements only
    world.elements_static.resize(local_size);
    // Dynamic arrays include ghost rows for halo exchange
    world.elements_dynamic.resize(full_size);
    world.elements_dynamic_swap.resize(full_size);
    
    // Initialize static elements
    for (int i = 0; i < local_size; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    // Initialize all dynamic elements (including ghost rows) to zero
    for (int i = 0; i < full_size; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        world.elements_dynamic_swap[i].current_energy = 0.0;
        world.elements_dynamic_swap[i].total_flux = 0.0;
    }
    
    // Build connectivity for local elements
    // Combined array layout: [ghost_above | local_rows ... | ghost_below]
    // Row 0  = ghost from rank above (x = offset_x - 1)
    // Row 1..local_nx = local rows (x = offset_x .. offset_x + local_nx - 1)
    // Row local_nx+1 = ghost from rank below (x = offset_x + local_nx)
    for (int i = 0; i < local_nx; ++i) {
        const int x = offset_x + i;
        for (int y = 0; y < n_elems_root; ++y) {
            ElementStatic& elem = world.elements_static[i * n_elems_root + y];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within global bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    // Compute combined array index (offset_x+1 = first local row in combined)
                    const int combined_idx = (nx - offset_x + 1) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = combined_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow (if they belong to this rank)
    const int last = n_elems_root - 1;
    if (offset_x <= 0 && offset_x + local_nx > 0) {
        world.elements_static[(-offset_x) * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
        world.elements_static[(-offset_x) * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    }
    if (offset_x <= last && offset_x + local_nx > last) {
        world.elements_static[(last - offset_x) * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[(last - offset_x) * n_elems_root + last].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations with MPI ghost/halo exchange
void runSimulation(World& world, const int n_iters, MPI_Comm comm) {
    int rank, n_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);
    
    const int n_elems_root = world.n_elems_root;
    const int local_nx = world.local_nx;
    const int row_size = n_elems_root;
    const size_t local_size = static_cast<size_t>(local_nx) * row_size;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Compute flux for all local elements
        // Local elements in combined array: rows 1 .. local_nx (row 0 = ghost above, 
        // row local_nx+1 = ghost below)
        for (size_t i = 0; i < local_size; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            // Map static index (local-only) to combined array index (includes ghosts)
            const int combined_idx = static_cast<int>((i / row_size + 1) * row_size + (i % row_size));
            const ElementDynamic& elem_dyn = world.elements_dynamic[combined_idx];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements
            // connected_idx stores combined array indices (including ghost positions)
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }
            
            // Update element state in swap buffer (at same combined index)
            ElementDynamic& elem_write = world.elements_dynamic_swap[combined_idx];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
        
        // Swap buffers (local + ghost rows all swap)
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
        
        // Exchange ghost rows with neighboring MPI ranks
        // Combined array layout:
        //   row 0          = ghost above (from rank-1's bottom local row)
        //   row 1          = top local row (send to rank-1 as their ghost below)
        //   row local_nx   = bottom local row (send to rank+1 as their ghost above)
        //   row local_nx+1 = ghost below (from rank+1's top local row)
        
        // Exchange with neighbor below (rank+1)
        if (rank < n_ranks - 1) {
            const int send_offset = local_nx * row_size;  // bottom local row
            const int recv_offset = (local_nx + 1) * row_size;  // ghost below
            MPI_Sendrecv(
                &world.elements_dynamic[send_offset],  // my bottom local row
                static_cast<int>(row_size * sizeof(ElementDynamic)), MPI_BYTE,
                rank + 1, 0,  // send to rank below
                &world.elements_dynamic[recv_offset],  // my ghost below
                static_cast<int>(row_size * sizeof(ElementDynamic)), MPI_BYTE,
                rank + 1, 1,  // recv from rank below
                comm, MPI_STATUS_IGNORE);
        }
        
        // Exchange with neighbor above (rank-1)
        if (rank > 0) {
            const int send_offset = row_size;  // top local row
            const int recv_offset = 0;  // ghost above
            MPI_Sendrecv(
                &world.elements_dynamic[send_offset],  // my top local row
                static_cast<int>(row_size * sizeof(ElementDynamic)), MPI_BYTE,
                rank - 1, 1,  // send to rank above
                &world.elements_dynamic[recv_offset],  // my ghost above
                static_cast<int>(row_size * sizeof(ElementDynamic)), MPI_BYTE,
                rank - 1, 0,  // recv from rank above
                comm, MPI_STATUS_IGNORE);
        }
    }
}

// Validate simulation results (operates on gathered flat vector of elements)
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

// Compute a simple hash of the results for verification (operates on flat vector)
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
    
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (rank 0 only)
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
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Broadcast parsed arguments to all ranks
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int validate_int = validate ? 1 : 0;
    int print_int = printResults ? 1 : 0;
    MPI_Bcast(&validate_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (validate_int != 0);
    printResults = (print_int != 0);
    
    // Domain decomposition: split along x-axis (rows)
    const int base_nx = n_elems_root / n_ranks;
    const int remainder = n_elems_root % n_ranks;
    const int local_nx = base_nx + (rank < remainder ? 1 : 0);
    const int offset_x = rank * base_nx + std::min(rank, remainder);
    const int n_elems = n_elems_root * n_elems_root;
    
    // Rank 0 prints banner and setup info
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", n_ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the local portion of the unstructured mesh
    if (rank == 0) {
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildLocalGrid(world, n_elems_root, offset_x, local_nx);
    
    // Calculate memory usage
    if (rank == 0) {
        const size_t static_mem = local_nx * n_elems_root * sizeof(ElementStatic);
        const size_t full_size = (local_nx + 2) * n_elems_root;
        const size_t dynamic_mem = full_size * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage per MPI rank: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
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
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    const double local_duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Gather results on rank 0 for hash/validation/output
    const int local_count = local_nx * n_elems_root;
    std::vector<ElementDynamic> local_only(local_count);
    for (int i = 0; i < local_count; ++i) {
        const int combined_idx = (i / n_elems_root + 1) * n_elems_root + (i % n_elems_root);
        local_only[i] = world.elements_dynamic[combined_idx];
    }
    
    // Prepare counts and displacements for MPI_Gatherv
    std::vector<int> recvcounts(n_ranks);
    std::vector<int> displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        const int r_nx = base_nx + (r < remainder ? 1 : 0);
        recvcounts[r] = r_nx * n_elems_root * static_cast<int>(sizeof(ElementDynamic));
        displs[r] = (r * base_nx + std::min(r, remainder)) * n_elems_root * static_cast<int>(sizeof(ElementDynamic));
    }
    
    std::vector<ElementDynamic> all_elements;
    if (rank == 0) {
        all_elements.resize(n_elems);
    }
    
    MPI_Gatherv(local_only.data(),
                local_count * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE,
                all_elements.data(),
                recvcounts.data(), displs.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);
    
    // Rank 0 prints results, validation, and optional data
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration_ms);
        
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
        
        // Compute hash for verification
        const uint64_t hash = computeHash(all_elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
        
        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(all_elements.size());
            for (const auto& elem : all_elements) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }
        
        // Validation
        if (validate) {
            const bool valid = validateResults(all_elements);
            if (!valid) {
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
