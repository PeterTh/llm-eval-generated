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
    idx_t global_offset = 0;
    int grid_width = 0;
    int local_rows = 0;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int rank,
                   const int n_ranks) {
    const int first_row = (n_elems_root * rank) / n_ranks;
    const int last_row = (n_elems_root * (rank + 1)) / n_ranks;
    const int n_elems = (last_row - first_row) * n_elems_root;
    world.global_offset = static_cast<idx_t>(first_row) * n_elems_root;
    world.grid_width = n_elems_root;
    world.local_rows = last_row - first_row;
    
    // Initialize materials
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
    for (int x = first_row; x < last_row; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int global_idx = x * n_elems_root + y;
            const int idx = global_idx - static_cast<int>(world.global_offset);
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
    const int corners[4] = {0, last, last * n_elems_root, last * n_elems_root + last};
    const idx_t corner_materials[4] = {INFLOW_MAT_ID, OUTFLOW_MAT_ID,
                                       OUTFLOW_MAT_ID, INFLOW_MAT_ID};
    for (int c = 0; c < 4; ++c) {
        const idx_t corner = static_cast<idx_t>(corners[c]);
        if (corner >= world.global_offset &&
            corner < world.global_offset + world.elements_static.size()) {
            world.elements_static[corners[c] - world.global_offset].material_idx = corner_materials[c];
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, const int rank, const int n_ranks) {
    const size_t n_elems = world.elements_static.size();
    const size_t row_size = static_cast<size_t>(world.grid_width);
    std::vector<ElementDynamic> top_halo(row_size), bottom_halo(row_size);

    auto update = [&](size_t i) {
        const ElementStatic& elem_static = world.elements_static[i];
        const ElementDynamic& elem_dyn = world.elements_dynamic[i];
        const Material& mat = world.materials[elem_static.material_idx];
        val_t total_flux = mat.external_flow;
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            const ElementDynamic* neighbor_dyn;
            if (neighbor_idx < world.global_offset)
                neighbor_dyn = &top_halo[neighbor_idx % world.grid_width];
            else if (neighbor_idx >= world.global_offset + n_elems)
                neighbor_dyn = &bottom_halo[neighbor_idx % world.grid_width];
            else
                neighbor_dyn = &world.elements_dynamic[neighbor_idx - world.global_offset];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], *neighbor_dyn);
        }
        ElementDynamic& elem_write = world.elements_dynamic_swap[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    };
    
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        int request_count = 0;
        if (world.local_rows > 0) {
            if (rank > 0) {
                MPI_Irecv(top_halo.data(), static_cast<int>(row_size * sizeof(ElementDynamic)), MPI_BYTE,
                          rank - 1, 1, MPI_COMM_WORLD, &requests[request_count++]);
                MPI_Isend(world.elements_dynamic.data(), static_cast<int>(row_size * sizeof(ElementDynamic)), MPI_BYTE,
                          rank - 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
            }
            if (rank + 1 < n_ranks) {
                MPI_Irecv(bottom_halo.data(), static_cast<int>(row_size * sizeof(ElementDynamic)), MPI_BYTE,
                          rank + 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
                MPI_Isend(world.elements_dynamic.data() + (world.local_rows - 1) * row_size,
                          static_cast<int>(row_size * sizeof(ElementDynamic)), MPI_BYTE,
                          rank + 1, 1, MPI_COMM_WORLD, &requests[request_count++]);
            }
            for (int row = 1; row + 1 < world.local_rows; ++row)
                for (size_t col = 0; col < row_size; ++col) update(row * row_size + col);
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
            if (world.local_rows == 1) {
                for (size_t col = 0; col < row_size; ++col) update(col);
            } else {
                for (size_t col = 0; col < row_size; ++col) update(col);
                const size_t base = (world.local_rows - 1) * row_size;
                for (size_t col = 0; col < row_size; ++col) update(base + col);
            }
        }
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results
bool validateResults(const World& world, int rank) {
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
    
    val_t global_energy_sum, global_flux_sum, global_energy_max, global_energy_min;
    MPI_Reduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank != 0) return true;
    energy_sum = global_energy_sum; flux_sum = global_flux_sum;
    energy_max = global_energy_max; energy_min = global_energy_min;
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
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (n_elems_root < 1 || n_iters < 0 || n_ranks > n_elems_root) {
        if (rank == 0) printf("Grid size must be positive, iterations non-negative, and MPI ranks must not exceed grid rows.\n");
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
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root, rank, n_ranks);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    size_t global_static_mem, global_dynamic_mem;
    MPI_Reduce(&static_mem, &global_static_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&dynamic_mem, &global_dynamic_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               (global_static_mem + global_dynamic_mem) / (1024.0 * 1024.0),
               global_static_mem / (1024.0 * 1024.0), global_dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters, rank, n_ranks);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long max_duration_ms;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
    duration_ms = max_duration_ms;
    
    printf("Computation time: %ld ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    // Compute hash for verification
    uint64_t local_hash = 0;
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const uint64_t* e = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
        const uint64_t* f = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
        const uint64_t global_i = world.global_offset + i;
        local_hash ^= (*e + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t hash;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("  Result hash: %016lX\n\n", hash);
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> local_energy(world.elements_dynamic.size());
        for (size_t i = 0; i < local_energy.size(); ++i) local_energy[i] = world.elements_dynamic[i].current_energy;
        std::vector<int> counts, displacements;
        std::vector<double> energyData;
        if (rank == 0) {
            counts.resize(n_ranks); displacements.resize(n_ranks); energyData.resize(n_elems);
            for (int r = 0; r < n_ranks; ++r) {
                const int first = n_elems_root * r / n_ranks;
                counts[r] = (n_elems_root * (r + 1) / n_ranks - first) * n_elems_root;
                displacements[r] = first * n_elems_root;
            }
        }
        MPI_Gatherv(local_energy.data(), static_cast<int>(local_energy.size()), MPI_DOUBLE,
                    rank == 0 ? energyData.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world, rank);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
