#include <algorithm>
#include <cmath>
#include <cstddef>
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

static_assert(offsetof(ElementDynamic, total_flux) == sizeof(val_t));
static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t));

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

// Build the rank's contiguous rows, with one dynamic halo row on each side.
void buildSquare2D(World& world, int n, int first_row, int rows) {
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(static_cast<size_t>(rows) * n);
    world.elements_dynamic.resize(static_cast<size_t>(rows + 2) * n);
    world.elements_dynamic_swap.resize(static_cast<size_t>(rows + 2) * n);

    for (int x = 0; x < rows; ++x) {
        const int global_x = first_row + x;
        for (int y = 0; y < n; ++y) {
            ElementStatic& elem = world.elements_static[static_cast<size_t>(x) * n + y];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            // Preserve the original neighbor order and arithmetic exactly.
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (const auto& offset : offsets) {
                const int nx = global_x + offset[0];
                const int ny = y + offset[1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    elem.connected_idx[elem.num_connections] =
                        static_cast<idx_t>(x + 1 + offset[0]) * n + ny;
                    elem.connected_flux[elem.num_connections++] = 1.0;
                }
            }
            // Match the sequential corner assignments, including a 1x1 grid.
            if (global_x == 0 && y == 0) elem.material_idx = INFLOW_MAT_ID;
            if (global_x == 0 && y == n - 1) elem.material_idx = OUTFLOW_MAT_ID;
            if (global_x == n - 1 && y == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (global_x == n - 1 && y == n - 1) elem.material_idx = INFLOW_MAT_ID;
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Exchange the boundary energy rows while updating rows that need no halo.
void runSimulation(World& world, int n, int rows, int rank, int active_ranks,
                   int n_iters, MPI_Datatype energy_row) {
    if (rows == 0) return;
    const int previous = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < active_ranks ? rank + 1 : MPI_PROC_NULL;
    auto update_row = [&](int x) {
        const size_t static_start = static_cast<size_t>(x) * n;
        const size_t dynamic_start = static_cast<size_t>(x + 1) * n;
        for (int y = 0; y < n; ++y) {
            const ElementStatic& elem_static = world.elements_static[static_start + y];
            const ElementDynamic& elem_dyn = world.elements_dynamic[dynamic_start + y];
            const Material& mat = world.materials[elem_static.material_idx];
            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const ElementDynamic& neighbor_dyn =
                    world.elements_dynamic[elem_static.connected_idx[j]];
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }
            ElementDynamic& result = world.elements_dynamic_swap[dynamic_start + y];
            result.current_energy = elem_dyn.current_energy + total_flux;
            result.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
    };

    for (int iter = 0; iter < n_iters; ++iter) {
        if (active_ranks == 1) {
            for (int x = 0; x < rows; ++x) update_row(x);
            std::swap(world.elements_dynamic, world.elements_dynamic_swap);
            continue;
        }
        MPI_Request requests[4];
        ElementDynamic* data = world.elements_dynamic.data();
        MPI_Irecv(&data[0].current_energy, 1, energy_row, previous, 1, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&data[static_cast<size_t>(rows + 1) * n].current_energy, 1, energy_row,
                  next, 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(&data[n].current_energy, 1, energy_row, previous, 0, MPI_COMM_WORLD, &requests[2]);
        MPI_Isend(&data[static_cast<size_t>(rows) * n].current_energy, 1, energy_row,
                  next, 1, MPI_COMM_WORLD, &requests[3]);
        for (int x = 1; x + 1 < rows; ++x) update_row(x);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        update_row(0);
        if (rows > 1) update_row(rows - 1);
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
uint64_t computeHash(const std::vector<ElementDynamic>& elements, size_t start,
                     size_t count, size_t global_start) {
    uint64_t hash = 0;
    for (size_t i = 0; i < count; ++i) {
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elements[start + i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[start + i].total_flux, sizeof(flux_bits));
        hash ^= (energy_bits + global_start + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + global_start + i) * 0xbf58476d1ce4e5b9ULL;
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
    if (n_elems_root <= 0 || static_cast<int64_t>(n_elems_root) * n_elems_root >
                                 std::numeric_limits<int>::max()) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and its square fit in an int\n");
        MPI_Finalize();
        return 1;
    }
    const int n = n_elems_root;
    const int n_elems = n * n;
    const int active_ranks = std::min(ranks, n);
    const int rows = n / ranks + (rank < n % ranks ? 1 : 0);
    const int first_row = rank * (n / ranks) + std::min(rank, n % ranks);
    const size_t local_count = static_cast<size_t>(rows) * n;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n, first_row, rows);
    MPI_Datatype energy_row, dynamic_type;
    MPI_Type_vector(n, 1, 2, MPI_DOUBLE, &energy_row);
    MPI_Type_commit(&energy_row);
    MPI_Type_contiguous(2, MPI_DOUBLE, &dynamic_type);
    MPI_Type_commit(&dynamic_type);

    if (rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (static_mem + dynamic_mem) / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0), dynamic_mem / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(world, n, rows, rank, active_ranks, n_iters, energy_row);
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    uint64_t local_hash = computeHash(world.elements_dynamic, n, local_count,
                                     static_cast<size_t>(first_row) * n);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) /
                                          (duration_ms / 1000.0) / 1e9;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
    }

    int valid = 1;
    if (printResults || validate) {
        std::vector<int> counts, displacements;
        World gathered;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                counts[r] = (n / ranks + (r < n % ranks ? 1 : 0)) * n;
                displacements[r] = (r * (n / ranks) + std::min(r, n % ranks)) * n;
            }
            gathered.elements_dynamic.resize(n_elems);
        }
        MPI_Gatherv(world.elements_dynamic.data() + n, static_cast<int>(local_count), dynamic_type,
                    rank == 0 ? gathered.elements_dynamic.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, dynamic_type, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(gathered.elements_dynamic.size());
                for (const auto& elem : gathered.elements_dynamic) energyData.push_back(elem.current_energy);
                print_results(energyData, "ElementEnergy");
            }
            if (validate) valid = validateResults(gathered) ? 1 : 0;
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&energy_row);
    MPI_Type_free(&dynamic_type);
    MPI_Finalize();
    return valid ? 0 : 1;
}
