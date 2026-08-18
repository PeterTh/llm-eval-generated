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
static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t));

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// The static mesh is partitioned in contiguous row blocks.  Dynamic arrays
// contain one read-only halo row at each end; static connectivity indexes that
// extended dynamic array while the static array itself contains owned cells only.
struct Partition {
    int first_row = 0;
    int local_rows = 0;
    int width = 0;
    int previous_rank = MPI_PROC_NULL;
    int next_rank = MPI_PROC_NULL;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const Partition& partition) {
    const size_t local_elems = static_cast<size_t>(partition.local_rows) * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(local_elems);
    world.elements_dynamic.assign(local_elems + 2 * n_elems_root, ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.assign(local_elems + 2 * n_elems_root, ElementDynamic{0.0, 0.0});
    
    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < local_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[n_elems_root + i].current_energy = 0.0;
        world.elements_dynamic[n_elems_root + i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int local_x = 0; local_x < partition.local_rows; ++local_x) {
        const int x = partition.first_row + local_x;
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = local_x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    // Dynamic data starts after the top halo row.  The
                    // neighbor may therefore be in either halo row.
                    const int neighbor_idx = (nx - partition.first_row + 1) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    if (partition.first_row == 0) {
        world.elements_static[0].material_idx = INFLOW_MAT_ID;
        world.elements_static[last].material_idx = OUTFLOW_MAT_ID;
    }
    if (partition.first_row + partition.local_rows == n_elems_root) {
        const size_t bottom = static_cast<size_t>(partition.local_rows - 1) * n_elems_root;
        world.elements_static[bottom].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[bottom + last].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, const Partition& partition, MPI_Comm comm) {
    const size_t n_elems = world.elements_static.size();
    const size_t offset = partition.width;
    MPI_Datatype halo_energy;
    MPI_Type_vector(partition.width, 1, 2, MPI_DOUBLE, &halo_energy);
    MPI_Type_commit(&halo_energy);

    const auto update_element = [&world, offset](size_t i) {
        const ElementStatic& elem_static = world.elements_static[i];
        const ElementDynamic& elem_dyn = world.elements_dynamic[offset + i];
        const Material& mat = world.materials[elem_static.material_idx];

        val_t total_flux = mat.external_flow;
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j],
                                      world.elements_dynamic[neighbor_idx]);
        }
        ElementDynamic& elem_write = world.elements_dynamic_swap[offset + i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    };
    
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        int request_count = 0;
        // Exchange only the values needed by the next update.  Posting all
        // operations before computation lets MPI make progress concurrently
        // with the strictly local interior work below.
        if (partition.previous_rank != MPI_PROC_NULL) {
            MPI_Irecv(&world.elements_dynamic[0], 1, halo_energy,
                      partition.previous_rank, 1, comm, &requests[request_count++]);
            MPI_Isend(&world.elements_dynamic[offset], 1, halo_energy,
                      partition.previous_rank, 0, comm, &requests[request_count++]);
        }
        if (partition.next_rank != MPI_PROC_NULL) {
            MPI_Irecv(&world.elements_dynamic[offset + n_elems], 1, halo_energy,
                      partition.next_rank, 0, comm, &requests[request_count++]);
            MPI_Isend(&world.elements_dynamic[offset + n_elems - partition.width], 1, halo_energy,
                      partition.next_rank, 1, comm, &requests[request_count++]);
        }
        // Interior rows do not touch halos and overlap communication.
        for (int row = 1; row + 1 < partition.local_rows; ++row) {
            const size_t begin = static_cast<size_t>(row) * partition.width;
            for (int column = 0; column < partition.width; ++column) update_element(begin + column);
        }
        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

        // Boundary rows depend on received halo data.  A single-row partition
        // has just one boundary row.
        for (int column = 0; column < partition.width; ++column) update_element(column);
        if (partition.local_rows > 1) {
            const size_t begin = static_cast<size_t>(partition.local_rows - 1) * partition.width;
            for (int column = 0; column < partition.width; ++column) update_element(begin + column);
        }
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
    MPI_Type_free(&halo_energy);
}

// Validate simulation results
bool validateResults(const World& world, const Partition& partition, MPI_Comm comm, int rank) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    const size_t offset = partition.width;
    for (size_t i = 0; i < world.elements_static.size(); ++i) {
        const auto& elem = world.elements_dynamic[offset + i];
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    
    val_t global_energy_sum, global_flux_sum, global_energy_max, global_energy_min;
    MPI_Reduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    if (rank != 0) return true;
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
uint64_t computeHash(const std::vector<ElementDynamic>& elements, size_t offset, size_t global_offset) {
    uint64_t hash = 0;
    for (size_t i = 0; i + 2 * offset < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elements[offset + i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[offset + i].total_flux, sizeof(flux_bits));
        const size_t global_i = global_offset + i;
        hash ^= (energy_bits + global_i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + global_i) * 0xbf58476d1ce4e5b9ULL;
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
        if (world_rank == 0) printf("Grid size must be positive and iterations must be non-negative\n");
        MPI_Finalize();
        return 1;
    }

    // Ranks with no rows are excluded from data collectives; the active ranks
    // retain their original ordering, so halo ranks are simply adjacent ranks.
    const int active_size = std::min(world_size, n_elems_root);
    MPI_Comm active_comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_size ? 0 : MPI_UNDEFINED,
                   world_rank, &active_comm);
    if (world_rank >= active_size) {
        MPI_Finalize();
        return 0;
    }

    int rank, size;
    MPI_Comm_rank(active_comm, &rank);
    MPI_Comm_size(active_comm, &size);
    const int base_rows = n_elems_root / size;
    const int extra_rows = n_elems_root % size;
    Partition partition;
    partition.first_row = rank * base_rows + std::min(rank, extra_rows);
    partition.local_rows = base_rows + (rank < extra_rows ? 1 : 0);
    partition.width = n_elems_root;
    partition.previous_rank = rank == 0 ? MPI_PROC_NULL : rank - 1;
    partition.next_rank = rank == size - 1 ? MPI_PROC_NULL : rank + 1;
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building distributed unstructured mesh...\n");
    }
    
    // Build the unstructured mesh
    World world;
    buildSquare2D(world, n_elems_root, partition);
    
    // Calculate memory usage
    const size_t local_static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t local_dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t local_total_mem = local_static_mem + local_dynamic_mem;
    uint64_t total_mem = 0, static_mem = 0, dynamic_mem = 0;
    const uint64_t local_total_mem_u64 = local_total_mem;
    const uint64_t local_static_mem_u64 = local_static_mem;
    const uint64_t local_dynamic_mem_u64 = local_dynamic_mem;
    MPI_Reduce(&local_total_mem_u64, &total_mem, 1, MPI_UINT64_T, MPI_SUM, 0, active_comm);
    MPI_Reduce(&local_static_mem_u64, &static_mem, 1, MPI_UINT64_T, MPI_SUM, 0, active_comm);
    MPI_Reduce(&local_dynamic_mem_u64, &dynamic_mem, 1, MPI_UINT64_T, MPI_SUM, 0, active_comm);
    if (rank == 0) {
        printf("Distributed memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               total_mem / (1024.0 * 1024.0), static_mem / (1024.0 * 1024.0), dynamic_mem / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    
    // Run simulation
    MPI_Barrier(active_comm);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters, partition, active_comm);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double local_duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm);
    
    if (rank == 0) printf("Computation time: %.3f ms\n", duration_ms);
    
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
    const uint64_t local_hash = computeHash(world.elements_dynamic, partition.width,
                                             static_cast<size_t>(partition.first_row) * partition.width);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, active_comm);
    if (rank == 0) printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
    
    // Print results for external validation
    if (printResults) {
        const int local_count = static_cast<int>(world.elements_static.size());
        std::vector<double> local_energy(local_count);
        for (int i = 0; i < local_count; ++i) {
            local_energy[i] = world.elements_dynamic[partition.width + i].current_energy;
        }
        std::vector<int> counts, displacements;
        std::vector<double> energyData;
        if (rank == 0) {
            counts.resize(size);
            displacements.resize(size);
            for (int r = 0; r < size; ++r) {
                const int rows = base_rows + (r < extra_rows ? 1 : 0);
                counts[r] = rows * n_elems_root;
                displacements[r] = (r * base_rows + std::min(r, extra_rows)) * n_elems_root;
            }
            energyData.resize(n_elems);
        }
        MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE, energyData.data(), counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, active_comm);
        if (rank == 0) print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world, partition, active_comm, rank);
        if (!valid) {
            MPI_Comm_free(&active_comm);
            MPI_Finalize();
            return 1;
        }
    }
    MPI_Comm_free(&active_comm);
    MPI_Finalize();
    return 0;
}
