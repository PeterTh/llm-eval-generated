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

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    // The dynamic vectors contain just this rank's rows.  Ghost rows carry
    // the energy values needed by connections crossing a rank boundary.
    std::vector<val_t> top_ghost;
    std::vector<val_t> bottom_ghost;
    int n_elems_root = 0;
    int first_row = 0;
    int local_rows = 0;
    int active_rank = -1;
    int active_size = 0;
    MPI_Comm comm = MPI_COMM_NULL;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, MPI_Comm comm) {
    world.comm = comm;
    world.n_elems_root = n_elems_root;
    MPI_Comm_rank(comm, &world.active_rank);
    MPI_Comm_size(comm, &world.active_size);
    world.first_row = n_elems_root * world.active_rank / world.active_size;
    const int end_row = n_elems_root * (world.active_rank + 1) / world.active_size;
    world.local_rows = end_row - world.first_row;
    const size_t local_elems = static_cast<size_t>(world.local_rows) * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(local_elems);
    world.elements_dynamic.resize(local_elems);
    world.elements_dynamic_swap.resize(local_elems);
    world.top_ghost.resize(n_elems_root);
    world.bottom_ghost.resize(n_elems_root);
    
    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < local_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = world.first_row; x < end_row; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t idx = static_cast<size_t>(x - world.first_row) * n_elems_root + y;
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
    const auto set_corner = [&](int x, int y, idx_t material) {
        if (x >= world.first_row && x < end_row) {
            world.elements_static[static_cast<size_t>(x - world.first_row) * n_elems_root + y]
                .material_idx = material;
        }
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
void runSimulation(World& world, const int n_iters) {
    const int n = world.n_elems_root;
    const int local_rows = world.local_rows;
    const idx_t local_base = static_cast<idx_t>(world.first_row) * n;
    const idx_t local_end = local_base + world.elements_dynamic.size();
    MPI_Datatype energy_row_type;
    MPI_Type_create_hvector(n, 1, sizeof(ElementDynamic), MPI_DOUBLE, &energy_row_type);
    MPI_Type_commit(&energy_row_type);
    const auto update_row = [&](int local_x) {
        const size_t row_offset = static_cast<size_t>(local_x) * n;
        for (int y = 0; y < n; ++y) {
            const size_t i = row_offset + y;
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                val_t neighbor_energy;
                if (neighbor_idx < local_base) {
                    neighbor_energy = world.top_ghost[neighbor_idx % n];
                } else if (neighbor_idx >= local_end) {
                    neighbor_energy = world.bottom_ghost[neighbor_idx % n];
                } else {
                    neighbor_energy = world.elements_dynamic[neighbor_idx - local_base].current_energy;
                }
                total_flux += (neighbor_energy - elem_dyn.current_energy) *
                              mat.transfer_coeff * elem_static.connected_flux[j] * 0.25;
            }
            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
    };

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        int request_count = 0;
        if (world.active_rank > 0) {
            MPI_Irecv(world.top_ghost.data(), n, MPI_DOUBLE, world.active_rank - 1, 0,
                      world.comm, &requests[request_count++]);
            MPI_Isend(world.elements_dynamic.data(), 1, energy_row_type,
                      world.active_rank - 1, 1, world.comm, &requests[request_count++]);
        }
        if (world.active_rank + 1 < world.active_size) {
            MPI_Irecv(world.bottom_ghost.data(), n, MPI_DOUBLE, world.active_rank + 1, 1,
                      world.comm, &requests[request_count++]);
            MPI_Isend(world.elements_dynamic.data() + static_cast<size_t>(local_rows - 1) * n,
                      1, energy_row_type, world.active_rank + 1, 0, world.comm,
                      &requests[request_count++]);
        }
        // Rows not touching a ghost can run while the boundary exchange is in flight.
        for (int local_x = 1; local_x + 1 < local_rows; ++local_x) {
            update_row(local_x);
        }
        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
        if (local_rows > 0) {
            update_row(0);
            if (local_rows > 1) update_row(local_rows - 1);
        }
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
    MPI_Type_free(&energy_row_type);
}

// Validate simulation results
bool validateResults(const World& world, int world_rank) {
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
    MPI_Reduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world.comm);
    MPI_Reduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, world.comm);
    MPI_Reduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
    MPI_Reduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, world.comm);
    if (world_rank != 0) return true;
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
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
        const uint64_t global_i = static_cast<uint64_t>(world.first_row) * world.n_elems_root + i;
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
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
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
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (n_elems_root <= 0 || n_iters < 0) {
        if (world_rank == 0) printf("Grid size must be positive and iterations cannot be negative\n");
        MPI_Finalize();
        return 1;
    }
    const size_t n_elems = static_cast<size_t>(n_elems_root) * n_elems_root;
    const int active_size = std::min(world_size, n_elems_root);
    MPI_Comm active_comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_size ? 0 : MPI_UNDEFINED,
                   world_rank, &active_comm);

    if (world_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI processes: %d (%d active)\n", world_size, active_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh
    if (world_rank == 0) printf("Building unstructured mesh...\n");
    World world;
    if (world_rank < active_size) buildSquare2D(world, n_elems_root, active_comm);
    
    // Calculate memory usage
    const size_t static_mem = n_elems * sizeof(ElementStatic);
    const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (world_rank == 0) printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    if (world_rank == 0) printf("\n");
    
    // Run simulation
    if (world_rank == 0) printf("Running simulation...\n");
    if (world_rank < active_size) MPI_Barrier(active_comm);
    const double start = MPI_Wtime();
    if (world_rank < active_size) runSimulation(world, n_iters);
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    if (world_rank < active_size) MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm);
    const double duration_ms = max_elapsed * 1000.0;
    if (world_rank == 0) printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = duration_ms / n_measured_iters;
    const double giga_elems_per_sec = duration_ms > 0.0
        ? (n_measured_iters * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9 : 0.0;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    if (world_rank == 0) {
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification
    uint64_t local_hash = 0, hash = 0;
    if (world_rank < active_size) local_hash = computeHash(world);
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        printf("  Result hash: %016" PRIX64 "\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> local_energy;
        if (world_rank < active_size) {
            local_energy.resize(world.elements_dynamic.size());
            for (size_t i = 0; i < local_energy.size(); ++i) local_energy[i] = world.elements_dynamic[i].current_energy;
        }
        std::vector<int> recv_counts, displacements;
        std::vector<double> energyData;
        if (world_rank == 0) {
            recv_counts.resize(world_size);
            displacements.resize(world_size);
            for (int rank = 0; rank < active_size; ++rank) {
                const int rows = n_elems_root * (rank + 1) / active_size - n_elems_root * rank / active_size;
                recv_counts[rank] = rows * n_elems_root;
                displacements[rank] = n_elems_root * (n_elems_root * rank / active_size);
            }
            for (int rank = active_size; rank < world_size; ++rank) recv_counts[rank] = displacements[rank] = 0;
            energyData.resize(n_elems);
        }
        MPI_Gatherv(local_energy.data(), static_cast<int>(local_energy.size()), MPI_DOUBLE,
                    energyData.data(), recv_counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (world_rank == 0) print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        bool valid = true;
        if (world_rank < active_size) valid = validateResults(world, world_rank);
        int valid_int = valid ? 1 : 0;
        MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!valid_int) { if (active_comm != MPI_COMM_NULL) MPI_Comm_free(&active_comm); MPI_Finalize(); return 1; }
    }
    if (active_comm != MPI_COMM_NULL) MPI_Comm_free(&active_comm);
    MPI_Finalize();
    return 0;
}
