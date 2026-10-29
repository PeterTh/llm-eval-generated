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
static_assert(sizeof(ElementDynamic) == 2 * sizeof(double));

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
void buildSquare2D(World& world, int n, int first_row, int rows) {
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    const size_t local_size = static_cast<size_t>(rows) * n;
    world.elements_static.resize(local_size);
    // One ghost row on either side; owned rows start at offset n.
    world.elements_dynamic.resize(local_size + 2 * static_cast<size_t>(n));
    world.elements_dynamic_swap.resize(local_size + 2 * static_cast<size_t>(n));
    for (int x = 0; x < rows; ++x) {
        const int gx = first_row + x;
        for (int y = 0; y < n; ++y) {
            auto& elem = world.elements_static[static_cast<size_t>(x) * n + y];
            elem.material_idx = DEFAULT_MAT_ID;
            // Match the original corner assignment order, including n == 1.
            if (gx == 0 && y == 0) elem.material_idx = INFLOW_MAT_ID;
            if (gx == 0 && y == n - 1) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n - 1 && y == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n - 1 && y == n - 1) elem.material_idx = INFLOW_MAT_ID;
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (const auto& offset : offsets) {
                int nx = gx + offset[0], ny = y + offset[1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    elem.connected_idx[elem.num_connections] =
                        static_cast<idx_t>(x + 1 + offset[0]) * n + ny;
                    elem.connected_flux[elem.num_connections++] = 1.0;
                }
            }
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
void runSimulation(World& world, int n_iters, int n, int rows, int rank, int active_ranks) {
    auto update = [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
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
    };
    if (rows == 0) return;
    const int prev = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < active_ranks ? rank + 1 : MPI_PROC_NULL;
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        auto* data = world.elements_dynamic.data();
        MPI_Irecv(data, n * 2, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(data + static_cast<size_t>(rows + 1) * n, n * 2, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(data + n, n * 2, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &requests[2]);
        MPI_Isend(data + static_cast<size_t>(rows) * n, n * 2, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &requests[3]);
        if (rows > 2) update(n, static_cast<size_t>(rows - 1) * n);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        update(0, n);
        if (rows > 1) update(static_cast<size_t>(rows - 1) * n, static_cast<size_t>(rows) * n);
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
uint64_t computeHash(const std::vector<ElementDynamic>& elements, size_t start, size_t count, size_t global_start) {
    uint64_t hash = 0;
    for (size_t i = 0; i < count; ++i) {
        // Simple hash combining energy and flux values
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
    
    if (n_elems_root <= 0 || static_cast<long long>(n_elems_root) * n_elems_root > std::numeric_limits<int>::max() / 2) {
        if (rank == 0) fprintf(stderr, "Invalid grid size\n");
        MPI_Finalize();
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;
    const int active_ranks = std::min(size, n_elems_root);
    const int rows = n_elems_root / size + (rank < n_elems_root % size);
    const int first_row = rank * (n_elems_root / size) + std::min(rank, n_elems_root % size);
    
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
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters, n_elems_root, rows, rank, active_ranks);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    long max_duration_ms = 0;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    duration_ms = max_duration_ms;
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
    
    // Compute hash for verification
    }
    const uint64_t local_hash = computeHash(world.elements_dynamic, n_elems_root, static_cast<size_t>(rows) * n_elems_root, static_cast<size_t>(first_row) * n_elems_root);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    }
    if (printResults || validate) {
        std::vector<int> counts(size), displacements(size);
        for (int r = 0; r < size; ++r) {
            int rrows = n_elems_root / size + (r < n_elems_root % size);
            int first = r * (n_elems_root / size) + std::min(r, n_elems_root % size);
            counts[r] = 2 * rrows * n_elems_root;
            displacements[r] = 2 * first * n_elems_root;
        }
        World gathered;
        if (rank == 0) gathered.elements_dynamic.resize(n_elems);
        MPI_Gatherv(world.elements_dynamic.data() + n_elems_root, 2 * rows * n_elems_root, MPI_DOUBLE,
                    gathered.elements_dynamic.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) world.elements_dynamic.swap(gathered.elements_dynamic);
    }
    int valid = 1;
    // Print results for external validation
    if (rank == 0 && printResults) {
        std::vector<double> energyData;
        energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (rank == 0 && validate) valid = validateResults(world);
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
