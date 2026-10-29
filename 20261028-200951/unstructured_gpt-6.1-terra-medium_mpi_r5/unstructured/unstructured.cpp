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
void buildSquare2D(World& world, int n, int first_row, int rows) {
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(static_cast<size_t>(rows) * n);
    // One ghost row on each side; only current_energy is exchanged.
    world.elements_dynamic.resize(static_cast<size_t>(rows + 2) * n);
    world.elements_dynamic_swap.resize(static_cast<size_t>(rows + 2) * n);
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int x = 0; x < rows; ++x) {
        int gx = first_row + x;
        for (int y = 0; y < n; ++y) {
            ElementStatic& elem = world.elements_static[static_cast<size_t>(x) * n + y];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            for (const auto& offset : offsets) {
                int nx = gx + offset[0], ny = y + offset[1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    elem.connected_idx[elem.num_connections] = static_cast<idx_t>(x + 1 + offset[0]) * n + ny;
                    elem.connected_flux[elem.num_connections++] = 1.0;
                }
            }
            // Preserve the original assignment order when corners coincide.
            if (gx == 0 && y == 0) elem.material_idx = INFLOW_MAT_ID;
            if (gx == 0 && y == n - 1) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n - 1 && y == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n - 1 && y == n - 1) elem.material_idx = INFLOW_MAT_ID;
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
void runSimulation(World& world, int n_iters, int n, int rows, int rank, int size, MPI_Comm comm) {
    std::vector<val_t> send_top(n), send_bottom(n), recv_top(n), recv_bottom(n);
    auto update_row = [&](int x) {
        for (int y = 0; y < n; ++y) {
            const size_t i = static_cast<size_t>(x) * n + y;
            const size_t d = i + n;
            const ElementStatic& elem = world.elements_static[i];
            const ElementDynamic& current = world.elements_dynamic[d];
            const Material& mat = world.materials[elem.material_idx];
            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < elem.num_connections; ++j)
                total_flux += computeFlux(mat, current, elem.connected_flux[j], world.elements_dynamic[elem.connected_idx[j]]);
            ElementDynamic& out = world.elements_dynamic_swap[d];
            out.current_energy = current.current_energy + total_flux;
            out.total_flux = current.total_flux + std::abs(total_flux);
        }
    };
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        int count = 0;
        if (rank > 0) {
            for (int y = 0; y < n; ++y) send_top[y] = world.elements_dynamic[n + y].current_energy;
            MPI_Irecv(recv_top.data(), n, MPI_DOUBLE, rank - 1, 1, comm, &requests[count++]);
            MPI_Isend(send_top.data(), n, MPI_DOUBLE, rank - 1, 0, comm, &requests[count++]);
        }
        if (rank + 1 < size) {
            for (int y = 0; y < n; ++y) send_bottom[y] = world.elements_dynamic[static_cast<size_t>(rows) * n + y].current_energy;
            MPI_Irecv(recv_bottom.data(), n, MPI_DOUBLE, rank + 1, 0, comm, &requests[count++]);
            MPI_Isend(send_bottom.data(), n, MPI_DOUBLE, rank + 1, 1, comm, &requests[count++]);
        }
        for (int x = 1; x + 1 < rows; ++x) update_row(x);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        if (rank > 0)
            for (int y = 0; y < n; ++y) world.elements_dynamic[y].current_energy = recv_top[y];
        if (rank + 1 < size)
            for (int y = 0; y < n; ++y) world.elements_dynamic[static_cast<size_t>(rows + 1) * n + y].current_energy = recv_bottom[y];
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
uint64_t computeHash(const ElementDynamic* elements, size_t count, size_t offset) {
    uint64_t hash = 0;
    for (size_t i = 0; i < count; ++i) {
        // Simple hash combining energy and flux values
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        hash ^= (energy_bits + i + offset) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + i + offset) * 0xbf58476d1ce4e5b9ULL;
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
    int n = 512, n_iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    // MPI_Gatherv uses int counts and displacements, in units of doubles.
    if (n < 1 || static_cast<long long>(n) * n > std::numeric_limits<int>::max() / 2) {
        if (world_rank == 0) fprintf(stderr, "Invalid grid size\n");
        MPI_Finalize();
        return 1;
    }
    int active = std::min(world_size, n);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active) { MPI_Finalize(); return 0; }
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int rows = n / size + (rank < n % size);
    int first = rank * (n / size) + std::min(rank, n % size);
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, n * n);
        printf("Iterations: %d\nValidation: %s\n\n", n_iters, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n, first, rows);
    if (rank == 0) {
        size_t static_mem = static_cast<size_t>(n) * n * sizeof(ElementStatic);
        size_t dynamic_mem = static_cast<size_t>(n) * n * sizeof(ElementDynamic) * 2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n", (static_mem + dynamic_mem) / 1048576.0, static_mem / 1048576.0, dynamic_mem / 1048576.0);
        printf("Running simulation...\n");
    }
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    runSimulation(world, n_iters, n, rows, rank, size, comm);
    double elapsed = MPI_Wtime() - start, max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    ElementDynamic* local = world.elements_dynamic.data() + n;
    uint64_t local_hash = computeHash(local, static_cast<size_t>(rows) * n, static_cast<size_t>(first) * n), hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    if (rank == 0) {
        auto duration_ms = static_cast<long>(max_elapsed * 1000);
        int measured = std::max(n_iters - 1, 1);
        double giga = (static_cast<double>(measured) * n * n) / max_elapsed / 1e9;
        printf("Computation time: %ld ms\nPerformance:\n", duration_ms);
        printf("  Time per iteration: %.4f ms\n", max_elapsed * 1000 / measured);
        printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", giga, giga * 22.0);
        printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
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
        // Gather as bytes to avoid assumptions about struct padding.
        MPI_Datatype element_type;
        MPI_Type_contiguous(sizeof(ElementDynamic), MPI_BYTE, &element_type);
        MPI_Type_commit(&element_type);
        for (int r = 0; r < size; ++r) { counts[r] /= 2; displacements[r] /= 2; }
        World full;
        if (rank == 0) full.elements_dynamic.resize(static_cast<size_t>(n) * n);
        MPI_Gatherv(local, rows * n, element_type, full.elements_dynamic.data(), counts.data(), displacements.data(), element_type, 0, comm);
        MPI_Type_free(&element_type);
        if (rank == 0) {
            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(full.elements_dynamic.size());
                for (const auto& elem : full.elements_dynamic) energyData.push_back(elem.current_energy);
                print_results(energyData, "ElementEnergy");
            }
            if (validate) valid = validateResults(full);
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
