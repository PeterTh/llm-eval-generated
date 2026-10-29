#include <mpi.h>
#include <algorithm>
#include <climits>
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
void buildSquare2D(World& world, const int n, int first_row, int rows) {
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    const size_t count = static_cast<size_t>(rows) * n;
    world.elements_static.resize(count);
    // One ghost row at either end, even on edge ranks.
    world.elements_dynamic.resize(count + 2 * static_cast<size_t>(n));
    world.elements_dynamic_swap.resize(count + 2 * static_cast<size_t>(n));
    for (int x = 0; x < rows; ++x) {
        const int gx = first_row + x;
        for (int y = 0; y < n; ++y) {
            ElementStatic& elem = world.elements_static[static_cast<size_t>(x) * n + y];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            if ((gx == 0 || gx == n - 1) && (y == 0 || y == n - 1)) {
                // Preserve the original corner assignment order when n == 1.
                elem.material_idx = ((gx == 0 && y == 0) ||
                                     (gx == n - 1 && y == n - 1)) ? INFLOW_MAT_ID : OUTFLOW_MAT_ID;
            }
            const int dx[4] = {1, -1, 0, 0};
            const int dy[4] = {0, 0, 1, -1};
            for (int j = 0; j < 4; ++j) {
                int nx = gx + dx[j], ny = y + dy[j];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    elem.connected_idx[elem.num_connections] = static_cast<idx_t>(x + 1 + dx[j]) * n + ny;
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
void runSimulation(World& world, int n_iters, int n, int rows, int rank, int active) {
    MPI_Datatype energy_row;
    MPI_Type_vector(n, 1, 2, MPI_DOUBLE, &energy_row);
    MPI_Type_commit(&energy_row);
    auto update_row = [&](int x) {
        for (int y = 0; y < n; ++y) {
            size_t i = static_cast<size_t>(x) * n + y;
            const ElementStatic& elem = world.elements_static[i];
            const ElementDynamic& state = world.elements_dynamic[i + n];
            const Material& mat = world.materials[elem.material_idx];
            val_t flux = mat.external_flow;
            for (idx_t j = 0; j < elem.num_connections; ++j)
                flux += computeFlux(mat, state, elem.connected_flux[j], world.elements_dynamic[elem.connected_idx[j]]);
            ElementDynamic& out = world.elements_dynamic_swap[i + n];
            out.current_energy = state.current_energy + flux;
            out.total_flux = state.total_flux + std::abs(flux);
        }
    };
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        int nr = 0;
        if (rows && rank > 0) {
            MPI_Irecv(&world.elements_dynamic[0].current_energy, 1, energy_row, rank - 1, 1, MPI_COMM_WORLD, &requests[nr++]);
            MPI_Isend(&world.elements_dynamic[n].current_energy, 1, energy_row, rank - 1, 0, MPI_COMM_WORLD, &requests[nr++]);
        }
        if (rows && rank + 1 < active) {
            MPI_Irecv(&world.elements_dynamic[static_cast<size_t>(rows + 1) * n].current_energy, 1, energy_row, rank + 1, 0, MPI_COMM_WORLD, &requests[nr++]);
            MPI_Isend(&world.elements_dynamic[static_cast<size_t>(rows) * n].current_energy, 1, energy_row, rank + 1, 1, MPI_COMM_WORLD, &requests[nr++]);
        }
        for (int x = 1; x + 1 < rows; ++x) update_row(x);
        MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
        if (rows) update_row(0);
        if (rows > 1) update_row(rows - 1);
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
uint64_t computeHash(const std::vector<ElementDynamic>& elements, size_t begin, size_t count, size_t offset) {
    uint64_t hash = 0;
    for (size_t i = 0; i < count; ++i) {
        // Simple hash combining energy and flux values
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elements[begin + i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[begin + i].total_flux, sizeof(flux_bits));
        hash ^= (energy_bits + offset + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + offset + i) * 0xbf58476d1ce4e5b9ULL;
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
    int n = 512, n_iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (n <= 0 || static_cast<long long>(n) * n > INT_MAX / 2) {
        if (!rank) fprintf(stderr, "Grid size must be positive and fit MPI counts.\n");
        MPI_Finalize(); return 1;
    }
    const int n_elems = n * n;
    const int rows = n / size + (rank < n % size);
    const int first = rank * (n / size) + std::min(rank, n % size);
    const int active = std::min(n, size);
    if (!rank) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %d elements\nIterations: %d\nValidation: %s\n\n", n, n, n_elems, n_iters, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
        size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n", (static_mem + dynamic_mem) / 1048576.0, static_mem / 1048576.0, dynamic_mem / 1048576.0);
    }
    World world;
    buildSquare2D(world, n, first, rows);
    if (!rank) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    runSimulation(world, n_iters, n, rows, rank, active);
    double elapsed = MPI_Wtime() - start, max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    uint64_t local_hash = computeHash(world.elements_dynamic, n, static_cast<size_t>(rows) * n, static_cast<size_t>(first) * n);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (!rank) {
        long duration_ms = static_cast<long>(max_elapsed * 1000);
        printf("Computation time: %ld ms\n", duration_ms);
        int measured = std::max(n_iters - 1, 1);
        double rate = (measured * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9;
        printf("Performance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016lX\n\n", static_cast<double>(duration_ms) / measured, rate, rate * 22.0, static_cast<unsigned long>(hash));
    }
    int valid = 1;
    if (validate || printResults) {
        std::vector<int> counts(size), displacements(size);
        for (int r = 0; r < size; ++r) {
            int rrows = n / size + (r < n % size);
            int rfirst = r * (n / size) + std::min(r, n % size);
            counts[r] = 2 * rrows * n;
            displacements[r] = 2 * rfirst * n;
        }
        World global;
        if (!rank) global.elements_dynamic.resize(n_elems);
        MPI_Gatherv(world.elements_dynamic.data() + n, 2 * rows * n, MPI_DOUBLE,
                    rank == 0 ? global.elements_dynamic.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) {
            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(n_elems);
                for (const auto& elem : global.elements_dynamic) energyData.push_back(elem.current_energy);
                print_results(energyData, "ElementEnergy");
            }
            if (validate) valid = validateResults(global);
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
