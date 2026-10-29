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
#include <climits>

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

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, int first_row, int rows) {
    const int n_elems = rows * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(static_cast<size_t>(n_elems) + 2ULL * n_elems_root);
    world.elements_dynamic_swap.resize(static_cast<size_t>(n_elems) + 2ULL * n_elems_root);
    
    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i + n_elems_root].current_energy = 0.0;
        world.elements_dynamic[i + n_elems_root].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = first_row; x < first_row + rows; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = (x - first_row) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = (nx - first_row + 1) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Apply corner assignments in the original order (also when N is one).
    const int last = n_elems_root - 1;
    auto set_corner = [&](int x, int y, idx_t material) {
        if (x >= first_row && x < first_row + rows)
            world.elements_static[(x - first_row) * n_elems_root + y].material_idx = material;
    };
    set_corner(0, 0, INFLOW_MAT_ID);
    set_corner(0, last, OUTFLOW_MAT_ID);
    set_corner(last, 0, OUTFLOW_MAT_ID);
    set_corner(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, int n_iters, int width, int rows, int rank, int active_ranks) {
    if (rows == 0) return;
    MPI_Datatype energy_row;
    MPI_Type_vector(width, 1, 2, MPI_DOUBLE, &energy_row);
    MPI_Type_commit(&energy_row);
    const int prev = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < active_ranks ? rank + 1 : MPI_PROC_NULL;
    auto update_rows = [&](int begin, int end) {
        for (int row = begin; row < end; ++row) {
            for (int y = 0; y < width; ++y) {
                const size_t i = static_cast<size_t>(row) * width + y;
                const ElementStatic& elem_static = world.elements_static[i];
                const ElementDynamic& elem_dyn = world.elements_dynamic[i + width];
                const Material& mat = world.materials[elem_static.material_idx];
                val_t total_flux = mat.external_flow;
                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const ElementDynamic& neighbor = world.elements_dynamic[elem_static.connected_idx[j]];
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor);
                }
                ElementDynamic& out = world.elements_dynamic_swap[i + width];
                out.current_energy = elem_dyn.current_energy + total_flux;
                out.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }
        }
    };
    for (int iter = 0; iter < n_iters; ++iter) {
        auto* data = world.elements_dynamic.data();
        MPI_Request requests[4];
        MPI_Irecv(&data[0].current_energy, 1, energy_row, prev, 1, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&data[(static_cast<size_t>(rows) + 1) * width].current_energy, 1, energy_row, next, 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(&data[width].current_energy, 1, energy_row, prev, 0, MPI_COMM_WORLD, &requests[2]);
        MPI_Isend(&data[static_cast<size_t>(rows) * width].current_energy, 1, energy_row, next, 1, MPI_COMM_WORLD, &requests[3]);
        if (rows > 2) update_rows(1, rows - 1);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        update_rows(0, 1);
        if (rows > 1) update_rows(rows - 1, rows);
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
    MPI_Type_free(&energy_row);
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
uint64_t computeHash(const ElementDynamic* elements, size_t count, size_t offset) {
    uint64_t hash = 0;
    for (size_t i = 0; i < count; ++i) {
        // Simple hash combining energy and flux values
        uint64_t e, f;
        std::memcpy(&e, &elements[i].current_energy, sizeof(e));
        std::memcpy(&f, &elements[i].total_flux, sizeof(f));
        hash ^= (e + i + offset) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + i + offset) * 0xbf58476d1ce4e5b9ULL;
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (n_elems_root <= 0 || static_cast<long long>(n_elems_root) * n_elems_root > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and contain at most INT_MAX elements\n");
        MPI_Finalize();
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;
    const int rows = n_elems_root / ranks + (rank < n_elems_root % ranks);
    const int first_row = rank * (n_elems_root / ranks) + std::min(rank, n_elems_root % ranks);
    const int active_ranks = std::min(ranks, n_elems_root);
    if (rank == 0) {
    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");
    
    // Build the unstructured mesh
    printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, first_row, rows);
    
    // Calculate memory usage
    const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) {
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    runSimulation(world, n_iters, n_elems_root, rows, rank, active_ranks);
    double elapsed = MPI_Wtime() - start, max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    auto duration_ms = static_cast<long>(max_elapsed * 1000);
    if (rank == 0) {
    
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
    uint64_t local_hash = computeHash(world.elements_dynamic.data() + n_elems_root,
                                      static_cast<size_t>(rows) * n_elems_root,
                                      static_cast<size_t>(first_row) * n_elems_root);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("  Result hash: %016lX\n\n", hash);

    if (printResults || validate) {
        // Gather in global order to retain the original serial output and summation order.
        MPI_Datatype element_type;
        MPI_Type_contiguous(2, MPI_DOUBLE, &element_type);
        MPI_Type_commit(&element_type);
        std::vector<int> counts(ranks), displacements(ranks);
        for (int r = 0; r < ranks; ++r) {
            counts[r] = (n_elems_root / ranks + (r < n_elems_root % ranks)) * n_elems_root;
            displacements[r] = (r * (n_elems_root / ranks) + std::min(r, n_elems_root % ranks)) * n_elems_root;
        }
        World result;
        if (rank == 0) result.elements_dynamic.resize(n_elems);
        MPI_Gatherv(world.elements_dynamic.data() + n_elems_root, rows * n_elems_root, element_type,
                    result.elements_dynamic.data(), counts.data(), displacements.data(), element_type, 0, MPI_COMM_WORLD);
        MPI_Type_free(&element_type);
        int valid = 1;
        if (rank == 0) {
            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(n_elems);
                for (const auto& elem : result.elements_dynamic) energyData.push_back(elem.current_energy);
                print_results(energyData, "ElementEnergy");
            }
            if (validate) valid = validateResults(result);
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    MPI_Finalize();
    return 0;
}
