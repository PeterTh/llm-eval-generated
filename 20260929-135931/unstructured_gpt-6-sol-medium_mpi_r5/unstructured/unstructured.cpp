#include <algorithm>
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
    int root_size;
    int first_row;
    int local_rows;
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
void buildSquare2D(World& world, const int n_elems_root, int first_row, int local_rows) {
    world.root_size = n_elems_root;
    world.first_row = first_row;
    world.local_rows = local_rows;
    const int n_elems = n_elems_root * local_rows;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    // Owned rows are bracketed by two ghost rows. Only energy is exchanged.
    world.elements_dynamic.resize(n_elems + 2 * n_elems_root);
    world.elements_dynamic_swap.resize(n_elems + 2 * n_elems_root);
    
    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = first_row; x < first_row + local_rows; ++x) {
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
                    elem.connected_idx[elem.num_connections] =
                        (nx - first_row + 1) * n_elems_root + ny;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    if (first_row == 0) {
        world.elements_static[0].material_idx = INFLOW_MAT_ID;
        world.elements_static[last].material_idx = OUTFLOW_MAT_ID;
    }
    if (first_row + local_rows == n_elems_root) {
        const int offset = (local_rows - 1) * n_elems_root;
        world.elements_static[offset].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[offset + last].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, MPI_Comm comm) {
    const size_t n_elems = world.elements_static.size();
    const int width = world.root_size;
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    MPI_Datatype energy_row;
    MPI_Type_create_hvector(width, 1, sizeof(ElementDynamic), MPI_DOUBLE, &energy_row);
    MPI_Type_commit(&energy_row);

    auto update_rows = [&](int first, int last) {
        for (int row = first; row < last; ++row) {
            for (int y = 0; y < width; ++y) {
                const size_t i = static_cast<size_t>(row) * width + y;
                const ElementStatic& elem_static = world.elements_static[i];
                const ElementDynamic& elem_dyn = world.elements_dynamic[i + width];
                const Material& mat = world.materials[elem_static.material_idx];
                val_t total_flux = mat.external_flow;
                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const ElementDynamic& neighbor_dyn = world.elements_dynamic[elem_static.connected_idx[j]];
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
                }
                ElementDynamic& elem_write = world.elements_dynamic_swap[i + width];
                elem_write.current_energy = elem_dyn.current_energy + total_flux;
                elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }
        }
    };
    
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        int count = 0;
        if (rank > 0) {
            MPI_Irecv(&world.elements_dynamic[0].current_energy, 1, energy_row,
                      rank - 1, 0, comm, &requests[count++]);
            MPI_Isend(&world.elements_dynamic[width].current_energy, 1, energy_row,
                      rank - 1, 1, comm, &requests[count++]);
        }
        if (rank + 1 < ranks) {
            MPI_Irecv(&world.elements_dynamic[n_elems + width].current_energy, 1, energy_row,
                      rank + 1, 1, comm, &requests[count++]);
            MPI_Isend(&world.elements_dynamic[n_elems].current_energy, 1, energy_row,
                      rank + 1, 0, comm, &requests[count++]);
        }
        update_rows(1, std::max(world.local_rows - 1, 1));
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        update_rows(0, 1);
        if (world.local_rows > 1) update_rows(world.local_rows - 1, world.local_rows);
        
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
    MPI_Type_free(&energy_row);
}

// Validate simulation results
bool validateResults(const World& world, MPI_Comm comm, int rank) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = world.root_size; i < world.root_size + world.elements_static.size(); ++i) {
        const auto& elem = world.elements_dynamic[i];
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    
    val_t local_sums[2] = {energy_sum, flux_sum};
    val_t sums[2];
    val_t global_max, global_min;
    MPI_Reduce(local_sums, sums, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&energy_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&energy_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    if (rank != 0) return true;
    energy_sum = sums[0];
    flux_sum = sums[1];
    energy_max = global_max;
    energy_min = global_min;
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
    for (size_t i = 0; i < world.elements_static.size(); ++i) {
        const size_t global_i = static_cast<size_t>(world.first_row) * world.root_size + i;
        const auto& elem = world.elements_dynamic[i + world.root_size];
        // Simple hash combining energy and flux values
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elem.current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elem.total_flux, sizeof(flux_bits));
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
    if (n_elems_root < 1 || n_elems_root > 46340 || n_iters < 0) {
        if (world_rank == 0) fprintf(stderr, "Grid size must be 1..46340 and iterations must be nonnegative\n");
        MPI_Finalize();
        return 1;
    }

    const int active_ranks = std::min(world_size, n_elems_root);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 0 : MPI_UNDEFINED,
                   world_rank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const int rank = world_rank;
    const int local_rows = n_elems_root / active_ranks + (rank < n_elems_root % active_ranks);
    const int first_row = rank * (n_elems_root / active_ranks) +
                          std::min(rank, n_elems_root % active_ranks);
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
    buildSquare2D(world, n_elems_root, first_row, local_rows);
    
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
    }
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    
    runSimulation(world, n_iters, comm);
    
    const double local_duration = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&local_duration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    const double duration_ms = duration * 1000.0;
    
    if (rank == 0) printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    if (rank == 0) {
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    // Compute hash for verification
    }
    const uint64_t local_hash = computeHash(world);
    uint64_t hash;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    if (rank == 0) {
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> local_energy(world.elements_static.size());
        for (size_t i = 0; i < local_energy.size(); ++i) {
            local_energy[i] = world.elements_dynamic[i + n_elems_root].current_energy;
        }
        std::vector<int> counts, offsets;
        std::vector<double> energyData;
        if (rank == 0) {
            counts.resize(active_ranks);
            offsets.resize(active_ranks);
            for (int r = 0; r < active_ranks; ++r) {
                const int rows = n_elems_root / active_ranks + (r < n_elems_root % active_ranks);
                const int first = r * (n_elems_root / active_ranks) + std::min(r, n_elems_root % active_ranks);
                counts[r] = rows * n_elems_root;
                offsets[r] = first * n_elems_root;
            }
            energyData.resize(n_elems);
        }
        MPI_Gatherv(local_energy.data(), static_cast<int>(local_energy.size()), MPI_DOUBLE,
                    energyData.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world, comm, rank);
        if (!valid) {
            MPI_Comm_free(&comm);
            MPI_Finalize();
            return 1;
        }
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return 0;
}
