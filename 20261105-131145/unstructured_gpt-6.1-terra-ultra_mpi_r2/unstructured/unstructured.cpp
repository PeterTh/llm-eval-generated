#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <climits>
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
    int grid_size = 0;
    int first_row = 0;
    int local_rows = 0;
    int first_col = 0;
    int local_cols = 0;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n, const int rank, const int px, const int py) {
    world.grid_size = n;
    const int active_ranks = px * py;
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    if (rank >= active_ranks) return;
    const int rx = rank / py, ry = rank % py;
    const int bx = n / px, ex = n % px;
    const int by = n / py, ey = n % py;
    world.local_rows = bx + (rx < ex);
    world.local_cols = by + (ry < ey);
    world.first_row = rx * bx + std::min(rx, ex);
    world.first_col = ry * by + std::min(ry, ey);
    const int cols = world.local_cols;
    const size_t stride = static_cast<size_t>(cols) + 2;
    const size_t local_size = static_cast<size_t>(world.local_rows) * cols;
    world.elements_static.resize(local_size);
    // One ghost cell on each side holds the neighboring tiles' energies.
    world.elements_dynamic.resize((static_cast<size_t>(world.local_rows) + 2) * stride);
    world.elements_dynamic_swap.resize(world.elements_dynamic.size());

    for (int lx = 0; lx < world.local_rows; ++lx) {
        const int x = world.first_row + lx;
        for (int ly = 0; ly < cols; ++ly) {
            const int y = world.first_col + ly;
            ElementStatic& elem = world.elements_static[static_cast<size_t>(lx) * cols + ly];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (const auto& offset : offsets) {
                const int nx = x + offset[0], ny = y + offset[1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    elem.connected_idx[elem.num_connections] =
                        static_cast<idx_t>(lx + 1 + offset[0]) * stride + ly + 1 + offset[1];
                    elem.connected_flux[elem.num_connections++] = 1.0;
                }
            }
            if (x == 0 && y == 0) elem.material_idx = INFLOW_MAT_ID;
            if (x == 0 && y == n - 1) elem.material_idx = OUTFLOW_MAT_ID;
            if (x == n - 1 && y == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (x == n - 1 && y == n - 1) elem.material_idx = INFLOW_MAT_ID;
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
void runSimulation(World& world, const int n_iters, const int rank, const int px, const int py,
                   MPI_Datatype energy_row, MPI_Datatype energy_col) {
    const int rows = world.local_rows, cols = world.local_cols;
    if (rows == 0) return;
    const size_t stride = static_cast<size_t>(cols) + 2;
    const int rx = rank / py, ry = rank % py;
    const int up = rx > 0 ? rank - py : MPI_PROC_NULL;
    const int down = rx + 1 < px ? rank + py : MPI_PROC_NULL;
    const int left = ry > 0 ? rank - 1 : MPI_PROC_NULL;
    const int right = ry + 1 < py ? rank + 1 : MPI_PROC_NULL;

    auto update = [&](int x, int y) {
        const size_t i = static_cast<size_t>(x) * cols + y;
        const size_t d = static_cast<size_t>(x + 1) * stride + y + 1;
        const ElementStatic& elem_static = world.elements_static[i];
        const ElementDynamic& elem_dyn = world.elements_dynamic[d];
        const Material& mat = world.materials[elem_static.material_idx];
        val_t total_flux = mat.external_flow;
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const ElementDynamic& neighbor_dyn = world.elements_dynamic[elem_static.connected_idx[j]];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }
        ElementDynamic& elem_write = world.elements_dynamic_swap[d];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    };
    if (px == 1 && py == 1) {
        for (int iter = 0; iter < n_iters; ++iter) {
            for (int x = 0; x < rows; ++x)
                for (int y = 0; y < cols; ++y) update(x, y);
            std::swap(world.elements_dynamic, world.elements_dynamic_swap);
        }
        return;
    }
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange only energies; flux is local state. Compute the interior while
        // the four faces are in flight, then update each boundary cell once.
        MPI_Request requests[8];
        auto* data = world.elements_dynamic.data();
        MPI_Irecv(&data[1].current_energy, 1, energy_row, up, 1, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&data[static_cast<size_t>(rows + 1) * stride + 1].current_energy, 1, energy_row, down, 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Irecv(&data[stride].current_energy, 1, energy_col, left, 3, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(&data[stride + cols + 1].current_energy, 1, energy_col, right, 2, MPI_COMM_WORLD, &requests[3]);
        MPI_Isend(&data[stride + 1].current_energy, 1, energy_row, up, 0, MPI_COMM_WORLD, &requests[4]);
        MPI_Isend(&data[static_cast<size_t>(rows) * stride + 1].current_energy, 1, energy_row, down, 1, MPI_COMM_WORLD, &requests[5]);
        MPI_Isend(&data[stride + 1].current_energy, 1, energy_col, left, 2, MPI_COMM_WORLD, &requests[6]);
        MPI_Isend(&data[stride + cols].current_energy, 1, energy_col, right, 3, MPI_COMM_WORLD, &requests[7]);
        for (int x = 1; x < rows - 1; ++x)
            for (int y = 1; y < cols - 1; ++y) update(x, y);
        MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
        for (int y = 0; y < cols; ++y) update(0, y);
        if (rows > 1) for (int y = 0; y < cols; ++y) update(rows - 1, y);
        for (int x = 1; x < rows - 1; ++x) {
            update(x, 0);
            if (cols > 1) update(x, cols - 1);
        }
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
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
    const size_t stride = static_cast<size_t>(world.local_cols) + 2;
    for (int x = 0; x < world.local_rows; ++x) {
        for (int y = 0; y < world.local_cols; ++y) {
            const auto& elem = world.elements_dynamic[static_cast<size_t>(x + 1) * stride + y + 1];
            const uint64_t i = static_cast<uint64_t>(world.first_row + x) * world.grid_size + world.first_col + y;
            uint64_t energy_bits, flux_bits;
            std::memcpy(&energy_bits, &elem.current_energy, sizeof(energy_bits));
            std::memcpy(&flux_bits, &elem.total_flux, sizeof(flux_bits));
            hash ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
        }
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
    if (n_elems_root <= 0 || static_cast<int64_t>(n_elems_root) * n_elems_root > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and its square must fit in an int\n");
        MPI_Finalize();
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;
    // A nearly square process grid limits face traffic. Extra ranks are idle
    // when they cannot form another complete row of nonempty tiles.
    const int py = std::min(n_elems_root, std::max(1, static_cast<int>(std::sqrt(size))));
    const int px = std::min(n_elems_root, size / py);
    const int active_ranks = px * py;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, rank, px, py);
    if (rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (static_mem + dynamic_mem) / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0), dynamic_mem / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t));
    MPI_Datatype energy_row, energy_col, tile_type, dynamic_type;
    MPI_Type_vector(world.local_cols, 1, 2, MPI_DOUBLE, &energy_row);
    MPI_Type_commit(&energy_row);
    MPI_Type_vector(world.local_rows, 1, 2 * (world.local_cols + 2), MPI_DOUBLE, &energy_col);
    MPI_Type_commit(&energy_col);
    MPI_Type_contiguous(2, MPI_DOUBLE, &dynamic_type);
    MPI_Type_commit(&dynamic_type);
    MPI_Type_vector(world.local_rows, world.local_cols, world.local_cols + 2, dynamic_type, &tile_type);
    MPI_Type_commit(&tile_type);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters, rank, px, py, energy_row, energy_col);
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    uint64_t local_hash = computeHash(world), hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long duration_ms = static_cast<long>(seconds * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = seconds * 1000.0 / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / seconds / 1e9;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
    }
    int valid = 1;
    if (validate || printResults) {
        std::vector<int> counts, displacements;
        std::vector<ElementDynamic> all_elements, packed;
        if (rank == 0) {
            counts.resize(size);
            displacements.resize(size);
            int offset = 0;
            for (int r = 0; r < active_ranks; ++r) {
                const int rx = r / py, ry = r % py;
                const int rows = n_elems_root / px + (rx < n_elems_root % px);
                const int cols = n_elems_root / py + (ry < n_elems_root % py);
                counts[r] = rows * cols;
                displacements[r] = offset;
                offset += counts[r];
            }
            for (int r = active_ranks; r < size; ++r) displacements[r] = offset;
            packed.resize(n_elems);
            all_elements.resize(n_elems);
        }
        const ElementDynamic* local = world.local_rows ? world.elements_dynamic.data() + world.local_cols + 3 : nullptr;
        MPI_Gatherv(local, world.local_rows ? 1 : 0, tile_type,
                    packed.data(), counts.data(), displacements.data(), dynamic_type, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            // Restore global row-major order for the order-dependent output and validation.
            for (int r = 0; r < active_ranks; ++r) {
                const int rx = r / py, ry = r % py;
                const int bx = n_elems_root / px, ex = n_elems_root % px;
                const int by = n_elems_root / py, ey = n_elems_root % py;
                const int rows = bx + (rx < ex), cols = by + (ry < ey);
                const int first_row = rx * bx + std::min(rx, ex);
                const int first_col = ry * by + std::min(ry, ey);
                for (int x = 0; x < rows; ++x)
                    std::copy_n(packed.data() + displacements[r] + static_cast<size_t>(x) * cols, cols,
                                all_elements.data() + static_cast<size_t>(first_row + x) * n_elems_root + first_col);
            }
        }
        if (rank == 0) {
            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(all_elements.size());
                for (const auto& elem : all_elements) energyData.push_back(elem.current_energy);
                print_results(energyData, "ElementEnergy");
            }
            if (validate) valid = validateResults(all_elements);
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&energy_row);
    MPI_Type_free(&energy_col);
    MPI_Type_free(&tile_type);
    MPI_Type_free(&dynamic_type);
    MPI_Finalize();
    return valid ? 0 : 1;
}
