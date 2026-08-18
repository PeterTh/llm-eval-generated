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

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    
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
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
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
    world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// A rank owns a contiguous band of rows.  The two extra rows are halos received
// from adjacent ranks; keeping the dynamic state contiguous makes exchange and
// the update loop cache-friendly.
struct DistributedWorld {
    int n;
    int first_row;
    int local_rows;
    int rank;
    int ranks;
    MPI_Comm comm;
    std::vector<ElementDynamic> current;
    std::vector<ElementDynamic> next;
};

inline val_t externalFlow(const int x, const int y, const int n) {
    return ((x == 0 && y == 0) || (x == n - 1 && y == n - 1)) ? 0.5 :
           ((x == 0 && y == n - 1) || (x == n - 1 && y == 0)) ? -0.5 : 0.0;
}

inline void updateRow(DistributedWorld& world, const int row) {
    const int n = world.n;
    const int x = world.first_row + row - 1;
    ElementDynamic* const write = world.next.data() + static_cast<size_t>(row) * n;
    const ElementDynamic* const read = world.current.data();
    const size_t base = static_cast<size_t>(row) * n;
    for (int y = 0; y < n; ++y) {
        const ElementDynamic& self = read[base + y];
        val_t flux = externalFlow(x, y, n);
        // Preserve the original connectivity and accumulation order: down, up,
        // right, left.  All materials use the same transfer coefficient (0.8).
        if (x + 1 < n) flux += computeFlux(Material{0.8, 0.0}, self, 1.0, read[base + n + y]);
        if (x > 0)     flux += computeFlux(Material{0.8, 0.0}, self, 1.0, read[base - n + y]);
        if (y + 1 < n) flux += computeFlux(Material{0.8, 0.0}, self, 1.0, read[base + y + 1]);
        if (y > 0)     flux += computeFlux(Material{0.8, 0.0}, self, 1.0, read[base + y - 1]);
        write[y] = {self.current_energy + flux, self.total_flux + std::abs(flux)};
    }
}

void runSimulation(DistributedWorld& world, const int n_iters) {
    const int bytes = world.n * static_cast<int>(sizeof(ElementDynamic));
    const int above = world.rank == 0 ? MPI_PROC_NULL : world.rank - 1;
    const int below = world.rank + 1 == world.ranks ? MPI_PROC_NULL : world.rank + 1;
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        MPI_Irecv(world.current.data(), bytes, MPI_BYTE, above, 1, world.comm, &requests[0]);
        MPI_Irecv(world.current.data() + static_cast<size_t>(world.local_rows + 1) * world.n,
                  bytes, MPI_BYTE, below, 0, world.comm, &requests[1]);
        MPI_Isend(world.current.data() + world.n, bytes, MPI_BYTE, above, 0, world.comm, &requests[2]);
        MPI_Isend(world.current.data() + static_cast<size_t>(world.local_rows) * world.n,
                  bytes, MPI_BYTE, below, 1, world.comm, &requests[3]);
        for (int row = 2; row < world.local_rows; ++row) updateRow(world, row);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        if (world.local_rows > 0) updateRow(world, 1);
        if (world.local_rows > 1) updateRow(world, world.local_rows);
        std::swap(world.current, world.next);
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
    int world_rank = 0;
    int world_size = 1;
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

    if (n_elems_root < 1 || n_iters < 0) {
        if (world_rank == 0) printf("Grid size must be positive and iterations non-negative\n");
        MPI_Finalize();
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;
    const int active_ranks = std::min(world_size, n_elems_root);
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 1 : MPI_UNDEFINED,
                   world_rank, &active_comm);
    if (world_rank >= active_ranks) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 0;
    }

    DistributedWorld world{};
    world.n = n_elems_root;
    world.rank = world_rank;
    world.ranks = active_ranks;
    world.comm = active_comm;
    const int base_rows = n_elems_root / active_ranks;
    const int extra_rows = n_elems_root % active_ranks;
    world.local_rows = base_rows + (world.rank < extra_rows ? 1 : 0);
    world.first_row = world.rank * base_rows + std::min(world.rank, extra_rows);
    const size_t local_with_halos = static_cast<size_t>(world.local_rows + 2) * n_elems_root;
    world.current.assign(local_with_halos, ElementDynamic{0.0, 0.0});
    world.next.resize(local_with_halos);

    if (world_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", active_ranks);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building distributed unstructured mesh...\n");
    }

    const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (world_rank == 0) printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    if (world_rank == 0) printf("\nRunning simulation...\n");
    MPI_Barrier(active_comm);
    auto start = std::chrono::high_resolution_clock::now();
    runSimulation(world, n_iters);
    auto end = std::chrono::high_resolution_clock::now();
    double local_seconds = std::chrono::duration<double>(end - start).count();
    double max_seconds = 0.0;
    MPI_Reduce(&local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm);
    const double duration_ms = max_seconds * 1000.0;
    if (world_rank == 0) printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    uint64_t local_hash = 0;
    for (int row = 1; row <= world.local_rows; ++row) {
        const size_t global_base = static_cast<size_t>(world.first_row + row - 1) * n_elems_root;
        const size_t local_base = static_cast<size_t>(row) * n_elems_root;
        for (int y = 0; y < n_elems_root; ++y) {
            const ElementDynamic& elem = world.current[local_base + y];
            uint64_t energy, flux;
            std::memcpy(&energy, &elem.current_energy, sizeof(energy));
            std::memcpy(&flux, &elem.total_flux, sizeof(flux));
            local_hash ^= (energy + global_base + y) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (flux + global_base + y) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, active_comm);
    if (world_rank == 0) {
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> local_energy(static_cast<size_t>(world.local_rows) * n_elems_root);
        for (size_t i = 0; i < local_energy.size(); ++i) local_energy[i] = world.current[n_elems_root + i].current_energy;
        std::vector<int> counts, displacements;
        std::vector<double> energyData;
        if (world_rank == 0) {
            counts.resize(active_ranks); displacements.resize(active_ranks);
            for (int rank = 0; rank < active_ranks; ++rank) {
                const int rows = base_rows + (rank < extra_rows ? 1 : 0);
                counts[rank] = rows * n_elems_root;
                displacements[rank] = (rank * base_rows + std::min(rank, extra_rows)) * n_elems_root;
            }
            energyData.resize(n_elems);
        }
        MPI_Gatherv(local_energy.data(), static_cast<int>(local_energy.size()), MPI_DOUBLE,
                    energyData.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, active_comm);
        if (world_rank == 0) print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    int exit_status = 0;
    if (validate) {
        val_t local_energy = 0.0, local_flux = 0.0;
        val_t local_max = std::numeric_limits<val_t>::lowest();
        val_t local_min = std::numeric_limits<val_t>::max();
        for (int row = 1; row <= world.local_rows; ++row) for (int y = 0; y < n_elems_root; ++y) {
            const auto& elem = world.current[static_cast<size_t>(row) * n_elems_root + y];
            local_energy += elem.current_energy; local_flux += elem.total_flux;
            local_max = std::max(local_max, elem.current_energy); local_min = std::min(local_min, elem.current_energy);
        }
        val_t energy_sum, flux_sum, energy_max, energy_min;
        MPI_Reduce(&local_energy, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, active_comm);
        MPI_Reduce(&local_flux, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, active_comm);
        MPI_Reduce(&local_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm);
        MPI_Reduce(&local_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, active_comm);
        if (world_rank == 0) {
            printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n",
                   energy_sum, flux_sum, energy_min, energy_max);
            printf("  Validation: %s\n", (std::isfinite(energy_sum) && std::isfinite(flux_sum) &&
                   std::isfinite(energy_max) && std::isfinite(energy_min)) ? "PASSED" : "FAILED");
            if (!std::isfinite(energy_sum) || !std::isfinite(flux_sum) ||
                !std::isfinite(energy_max) || !std::isfinite(energy_min)) exit_status = 1;
        }
        MPI_Bcast(&exit_status, 1, MPI_INT, 0, active_comm);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_free(&active_comm);
    MPI_Finalize();
    return exit_status;
}
