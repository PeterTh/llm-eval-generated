#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
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
static_assert(std::is_standard_layout_v<ElementDynamic> &&
              sizeof(ElementDynamic) == 2 * sizeof(double) &&
              offsetof(ElementDynamic, total_flux) == sizeof(double));

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    int row_start = 0;
    int row_count = 0;
    int root = 0;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, int row_start, int row_count) {
    world.row_start = row_start;
    world.row_count = row_count;
    world.root = n_elems_root;
    const size_t n_elems = static_cast<size_t>(row_count) * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems + (row_count ? 2 * static_cast<size_t>(n_elems_root) : 0));
    world.elements_dynamic_swap.resize(world.elements_dynamic.size());
    
    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = row_start; x < row_start + row_count; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = (x - row_start) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const size_t neighbor_idx =
                        static_cast<size_t>(nx - row_start + 1) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    if (row_start == 0 && row_count) {
        world.elements_static[0].material_idx = INFLOW_MAT_ID;
        world.elements_static[last].material_idx = OUTFLOW_MAT_ID;
    }
    if (row_start <= last && last < row_start + row_count) {
        const int base = (last - row_start) * n_elems_root;
        world.elements_static[base].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[base + last].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, int rank, int active_ranks) {
    const int n = world.root;
    const int rows = world.row_count;
    if (!rows) return;

    auto update_rows = [&](int first, int last) {
        for (int x = first; x < last; ++x) {
            for (int y = 0; y < n; ++y) {
                const size_t i = static_cast<size_t>(x) * n + y;
                const ElementStatic& elem_static = world.elements_static[i];
                const ElementDynamic& elem_dyn = world.elements_dynamic[i + n];
                const Material& mat = world.materials[elem_static.material_idx];
                val_t total_flux = mat.external_flow;
                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const ElementDynamic& neighbor_dyn = world.elements_dynamic[elem_static.connected_idx[j]];
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
                }
                ElementDynamic& elem_write = world.elements_dynamic_swap[i + n];
                elem_write.current_energy = elem_dyn.current_energy + total_flux;
                elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }
        }
    };

    if (active_ranks == 1) {
        for (int iter = 0; iter < n_iters; ++iter) {
            update_rows(0, rows);
            std::swap(world.elements_dynamic, world.elements_dynamic_swap);
        }
        return;
    }

    // A row of energy values is strided through ElementDynamic; flux history
    // never crosses a rank boundary.
    MPI_Datatype energy_row;
    MPI_Type_vector(n, 1, 2, MPI_DOUBLE, &energy_row);
    MPI_Type_commit(&energy_row);

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        int count = 0;
        if (rank > 0) {
            MPI_Irecv(&world.elements_dynamic[0].current_energy, 1, energy_row,
                      rank - 1, 1, MPI_COMM_WORLD, &requests[count++]);
            MPI_Isend(&world.elements_dynamic[n].current_energy, 1, energy_row,
                      rank - 1, 0, MPI_COMM_WORLD, &requests[count++]);
        }
        if (rank + 1 < active_ranks) {
            MPI_Irecv(&world.elements_dynamic[(static_cast<size_t>(rows) + 1) * n].current_energy, 1, energy_row,
                      rank + 1, 0, MPI_COMM_WORLD, &requests[count++]);
            MPI_Isend(&world.elements_dynamic[static_cast<size_t>(rows) * n].current_energy, 1, energy_row,
                      rank + 1, 1, MPI_COMM_WORLD, &requests[count++]);
        }

        update_rows(1, rows - 1);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        update_rows(0, 1);
        if (rows > 1) update_rows(rows - 1, rows);
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
    MPI_Type_free(&energy_row);
}

// Validate simulation results
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

// Compute a simple hash of the results for verification
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    const size_t base = static_cast<size_t>(world.row_start) * world.root;
    for (size_t i = 0; i < world.elements_static.size(); ++i) {
        // Simple hash combining energy and flux values
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &world.elements_dynamic[i + world.root].current_energy, sizeof(uint64_t));
        std::memcpy(&flux_bits, &world.elements_dynamic[i + world.root].total_flux, sizeof(uint64_t));
        hash ^= (energy_bits + base + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + base + i) * 0xbf58476d1ce4e5b9ULL;
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (n_elems_root < 1 || n_elems_root > 46340 || n_iters < 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be 1..46340 and iterations must be nonnegative\n");
        MPI_Finalize();
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;
    const int active_ranks = std::min(ranks, n_elems_root);
    const int row_start = rank < active_ranks ?
        static_cast<int>((static_cast<int64_t>(n_elems_root) * rank) / active_ranks) : n_elems_root;
    const int row_end = rank < active_ranks ?
        static_cast<int>((static_cast<int64_t>(n_elems_root) * (rank + 1)) / active_ranks) : n_elems_root;
    const int row_count = row_end - row_start;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\nBuilding unstructured mesh...\n");
    }
    
    World world;
    buildSquare2D(world, n_elems_root, row_start, row_count);
    
    const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               total_mem / (1024.0 * 1024.0), static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters, rank, active_ranks);
    const double elapsed = MPI_Wtime() - start;
    double duration_seconds;
    MPI_Reduce(&elapsed, &duration_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const uint64_t local_hash = computeHash(world);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double duration_ms = duration_seconds * 1000.0;
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double giga_elems_per_sec = duration_seconds > 0.0 ?
            (static_cast<double>(n_measured_iters) * n_elems) / duration_seconds / 1e9 : 0.0;
        printf("Computation time: %.3f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", duration_ms / n_measured_iters);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
    }

    std::vector<ElementDynamic> results;
    if (printResults || validate) {
        MPI_Datatype element_type;
        MPI_Type_contiguous(2, MPI_DOUBLE, &element_type);
        MPI_Type_commit(&element_type);
        std::vector<int> counts, displacements;
        if (rank == 0) {
            results.resize(n_elems);
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                const int begin = r < active_ranks ?
                    static_cast<int>((static_cast<int64_t>(n_elems_root) * r) / active_ranks) : n_elems_root;
                const int end = r < active_ranks ?
                    static_cast<int>((static_cast<int64_t>(n_elems_root) * (r + 1)) / active_ranks) : n_elems_root;
                counts[r] = (end - begin) * n_elems_root;
                displacements[r] = begin * n_elems_root;
            }
        }
        MPI_Gatherv(row_count ? world.elements_dynamic.data() + n_elems_root : nullptr,
                    row_count * n_elems_root, element_type,
                    rank == 0 ? results.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    element_type, 0, MPI_COMM_WORLD);
        MPI_Type_free(&element_type);
    }

    int valid = 1;
    if (rank == 0) {
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(results.size());
            for (const auto& elem : results) energyData.push_back(elem.current_energy);
            print_results(energyData, "ElementEnergy");
        }
        if (validate) valid = validateResults(results) ? 1 : 0;
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
