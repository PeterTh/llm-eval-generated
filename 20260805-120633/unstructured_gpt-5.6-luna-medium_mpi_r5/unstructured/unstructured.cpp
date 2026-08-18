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
    int row_begin = 0;
    int row_end = 0;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root,
                   const int row_begin, const int row_end) {
    const int local_rows = row_end - row_begin;
    const size_t n_elems = static_cast<size_t>(local_rows) * n_elems_root;
    world.n_elems_root = n_elems_root;
    world.row_begin = row_begin;
    world.row_end = row_end;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);
    
    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = row_begin; x < row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = (x - row_begin) * n_elems_root + y;
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
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    if (local_rows > 0 && row_begin == 0) {
        world.elements_static[0].material_idx = INFLOW_MAT_ID;
        world.elements_static[last].material_idx = OUTFLOW_MAT_ID;
    }
    if (local_rows > 0 && row_end == n_elems_root) {
        world.elements_static[(local_rows - 1) * n_elems_root].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[(local_rows - 1) * n_elems_root + last].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
int ownerOfRow(const int row, const int n_rows, const int n_ranks) {
    const int base = n_rows / n_ranks;
    const int remainder = n_rows % n_ranks;
    if (base == 0) return row;
    const int large_rows = remainder * (base + 1);
    return row < large_rows ? row / (base + 1)
                            : remainder + (row - large_rows) / base;
}

void runSimulation(World& world, const int n_iters, MPI_Comm comm,
                   const int n_ranks) {
    const size_t n_elems = world.elements_static.size();
    const int n = world.n_elems_root;
    const int local_rows = world.row_end - world.row_begin;
    std::vector<val_t> top_ghost(static_cast<size_t>(n));
    std::vector<val_t> bottom_ghost(static_cast<size_t>(n));
    std::vector<val_t> top_send(static_cast<size_t>(n));
    std::vector<val_t> bottom_send(static_cast<size_t>(n));
    const int top_rank = world.row_begin > 0
        ? ownerOfRow(world.row_begin - 1, n, n_ranks) : MPI_PROC_NULL;
    const int bottom_rank = world.row_end < n
        ? ownerOfRow(world.row_end, n, n_ranks) : MPI_PROC_NULL;

    auto valueAt = [&](const idx_t global_idx) -> val_t {
        const int global_row = static_cast<int>(global_idx / n);
        const int column = static_cast<int>(global_idx % n);
        if (global_row < world.row_begin) return top_ghost[column];
        if (global_row >= world.row_end) return bottom_ghost[column];
        return world.elements_dynamic[(global_row - world.row_begin) * n + column].current_energy;
    };

    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange only the rows needed by the four-point connectivity.
        if (local_rows > 0) {
            for (int column = 0; column < n; ++column) {
                top_send[column] = world.elements_dynamic[column].current_energy;
                bottom_send[column] = world.elements_dynamic[(local_rows - 1) * n + column].current_energy;
            }
            MPI_Sendrecv(top_send.data(), n, MPI_DOUBLE, top_rank, 101,
                         top_ghost.data(), n, MPI_DOUBLE, top_rank, 102,
                         comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(bottom_send.data(), n, MPI_DOUBLE, bottom_rank, 102,
                         bottom_ghost.data(), n, MPI_DOUBLE,
                         bottom_rank, 101, comm, MPI_STATUS_IGNORE);
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
                ElementDynamic neighbor_dyn{valueAt(neighbor_idx), 0.0};
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
bool validateResults(const World& world) {
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
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

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
    
    if (n_elems_root <= 0 || n_iters < 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and iterations non-negative\n");
        MPI_Finalize();
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;

    const int base_rows = n_elems_root / n_ranks;
    const int extra_rows = n_elems_root % n_ranks;
    const int row_begin = rank * base_rows + std::min(rank, extra_rows);
    const int row_end = row_begin + base_rows + (rank < extra_rows ? 1 : 0);
    
    if (rank == 0) printf("Unstructured Mesh Energy Transfer Benchmark\n");
    if (rank == 0) {
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", n_ranks);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
    }
    
    // Build the unstructured mesh
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root, row_begin, row_end);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) printf("Memory usage per rank (rank 0): %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    if (rank == 0) printf("\n");
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    
    runSimulation(world, n_iters, comm, n_ranks);
    
    MPI_Barrier(comm);
    const double local_duration = MPI_Wtime() - start;
    double duration_seconds = 0.0;
    MPI_Reduce(&local_duration, &duration_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    const long duration_ms = static_cast<long>(duration_seconds * 1000.0);

    if (rank == 0) printf("Computation time: %ld ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    if (rank == 0) {
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Gather the distributed result in global order for identical reporting/hash semantics.
    std::vector<int> counts(n_ranks), displacements(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        const int rbegin = r * base_rows + std::min(r, extra_rows);
        const int rend = rbegin + base_rows + (r < extra_rows ? 1 : 0);
        counts[r] = (rend - rbegin) * n_elems_root * static_cast<int>(sizeof(ElementDynamic));
        displacements[r] = rbegin * n_elems_root * static_cast<int>(sizeof(ElementDynamic));
    }
    std::vector<ElementDynamic> global_elements(rank == 0 ? static_cast<size_t>(n_elems) : 0);
    MPI_Gatherv(world.elements_dynamic.data(), counts[rank], MPI_BYTE,
                rank == 0 ? global_elements.data() : nullptr, counts.data(), displacements.data(),
                MPI_BYTE, 0, comm);

    if (rank == 0) {
        const uint64_t result_hash = computeHash(global_elements);
        printf("  Result hash: %016lX\n\n", result_hash);
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        std::vector<double> energyData;
        energyData.reserve(global_elements.size());
        for (const auto& elem : global_elements) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    int valid_result = 1;
    if (rank == 0 && validate) {
        World validation_world;
        validation_world.elements_dynamic = std::move(global_elements);
        bool valid = validateResults(validation_world);
        valid_result = valid ? 1 : 0;
    }
    MPI_Bcast(&valid_result, 1, MPI_INT, 0, comm);
    MPI_Finalize();
    return valid_result ? 0 : 1;
}
