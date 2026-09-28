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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (local indices)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Domain decomposition: the global mesh is partitioned into contiguous row
// blocks (a "row" here is a ring of n_elems_root elements along the second
// grid axis). Each rank owns a contiguous range of rows [row_start, row_end)
// and stores a one-row halo above/below (when a neighbor rank exists) so
// that flux exchange with vertically adjacent elements requires no more
// than a single nearest-neighbor exchange per iteration.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;

    int n_elems_root = 0;

    // Global row range owned by this rank (halo excluded).
    int row_start = 0;
    int row_end = 0;

    // Local storage covers [ext_row_start, ext_row_end) globally, including
    // the halo rows (if present).
    int ext_row_start = 0;
    int ext_row_end = 0;

    bool has_halo_above = false;
    bool has_halo_below = false;

    int rank_above = MPI_PROC_NULL;
    int rank_below = MPI_PROC_NULL;

    int owned_row_offset() const { return row_start - ext_row_start; }
    int local_owned_rows() const { return row_end - row_start; }
    int local_ext_rows() const { return ext_row_end - ext_row_start; }
    size_t local_ext_elems() const {
        return static_cast<size_t>(local_ext_rows()) * n_elems_root;
    }

    // Map a global (x, y) coordinate to a local element index within this
    // rank's extended (halo-including) storage. Caller must ensure x is
    // within [ext_row_start, ext_row_end).
    idx_t localIndex(int x, int y) const {
        return static_cast<idx_t>(x - ext_row_start) * n_elems_root + y;
    }
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Compute this rank's owned row range for a 1D row decomposition of
// n_elems_root rows across `size` ranks.
void computeRowRange(int n_elems_root, int rank, int size, int& row_start, int& row_end) {
    const int base = n_elems_root / size;
    const int rem = n_elems_root % size;
    row_start = rank * base + std::min(rank, rem);
    const int rows = base + (rank < rem ? 1 : 0);
    row_end = row_start + rows;
}

// Build the local (row-decomposed, halo-including) portion of a 2D square
// grid mesh for this rank. Each rank only allocates and initializes the
// rows it owns plus a single halo row above/below where applicable.
void buildSquare2D(World& world, const int n_elems_root, int rank, int size) {
    world.n_elems_root = n_elems_root;

    // Initialize materials (small, replicated on every rank)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    computeRowRange(n_elems_root, rank, size, world.row_start, world.row_end);

    world.has_halo_above = world.row_start > 0;
    world.has_halo_below = world.row_end < n_elems_root;

    world.ext_row_start = world.row_start - (world.has_halo_above ? 1 : 0);
    world.ext_row_end = world.row_end + (world.has_halo_below ? 1 : 0);

    world.rank_above = world.has_halo_above ? rank - 1 : MPI_PROC_NULL;
    world.rank_below = world.has_halo_below ? rank + 1 : MPI_PROC_NULL;

    const size_t n_local = world.local_ext_elems();
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    // Initialize all locally-stored elements (owned + halo) with default
    // material and zero energy.
    for (size_t i = 0; i < n_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity only for owned elements (halo elements are never
    // updated, so their connectivity is irrelevant).
    for (int x = world.row_start; x < world.row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const idx_t local_idx = world.localIndex(x, y);
            ElementStatic& elem = world.elements_static[local_idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const idx_t neighbor_local_idx = world.localIndex(nx, ny);
                    elem.connected_idx[elem.num_connections] = neighbor_local_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    // (only applied by the rank(s) that own the corresponding rows).
    const int last = n_elems_root - 1;
    auto setCornerMaterial = [&](int x, int y, idx_t mat_id) {
        if (x >= world.row_start && x < world.row_end) {
            world.elements_static[world.localIndex(x, y)].material_idx = mat_id;
        }
    };
    setCornerMaterial(0, 0, INFLOW_MAT_ID);
    setCornerMaterial(0, last, OUTFLOW_MAT_ID);
    setCornerMaterial(last, 0, OUTFLOW_MAT_ID);
    setCornerMaterial(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Exchange halo rows with vertically-adjacent ranks. Sends this rank's
// boundary owned rows to its neighbors and receives their boundary owned
// rows into this rank's halo storage, so that flux computation always
// reads up-to-date neighbor state.
void exchangeHalos(World& world) {
    const int n_elems_root = world.n_elems_root;
    ElementDynamic* data = world.elements_dynamic.data();

    const int owned_offset = world.owned_row_offset();
    const int owned_rows = world.local_owned_rows();

    ElementDynamic* top_owned_row = data + static_cast<size_t>(owned_offset) * n_elems_root;
    ElementDynamic* bottom_owned_row =
        data + static_cast<size_t>(owned_offset + owned_rows - 1) * n_elems_root;
    ElementDynamic* halo_above_row = data;  // local row 0 when has_halo_above
    ElementDynamic* halo_below_row =
        data + static_cast<size_t>(world.local_ext_rows() - 1) * n_elems_root;

    // Exchange with the rank above: send our topmost owned row, receive
    // into our above-halo row.
    MPI_Sendrecv(top_owned_row, n_elems_root * 2, MPI_DOUBLE, world.rank_above, 0,
                 halo_above_row, n_elems_root * 2, MPI_DOUBLE, world.rank_above, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Exchange with the rank below: send our bottommost owned row, receive
    // into our below-halo row.
    MPI_Sendrecv(bottom_owned_row, n_elems_root * 2, MPI_DOUBLE, world.rank_below, 1,
                 halo_below_row, n_elems_root * 2, MPI_DOUBLE, world.rank_below, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const int owned_offset = world.owned_row_offset();
    const size_t n_owned = static_cast<size_t>(world.local_owned_rows()) * world.n_elems_root;
    const size_t owned_begin = static_cast<size_t>(owned_offset) * world.n_elems_root;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Refresh halo rows with the latest neighbor state before computing.
        if (world.rank_above != MPI_PROC_NULL || world.rank_below != MPI_PROC_NULL) {
            exchangeHalos(world);
        }

        // Update all owned elements
        for (size_t k = 0; k < n_owned; ++k) {
            const size_t i = owned_begin + k;
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];

            // Start with external flow
            val_t total_flux = mat.external_flow;

            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }

            // Update element state
            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }

        // Swap buffers (halo entries are stale until the next exchange,
        // which happens before they are read again).
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Gather each rank's owned elements into a single globally-ordered vector
// on rank 0.
std::vector<ElementDynamic> gatherGlobalResults(const World& world, int rank, int size) {
    const int n_elems_root = world.n_elems_root;
    const int n_elems = n_elems_root * n_elems_root;

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        int rs, re;
        computeRowRange(n_elems_root, r, size, rs, re);
        recvcounts[r] = (re - rs) * n_elems_root * 2;  // 2 doubles per element
        displs[r] = rs * n_elems_root * 2;
    }

    std::vector<ElementDynamic> global;
    if (rank == 0) {
        global.resize(n_elems);
    }

    const int owned_offset = world.owned_row_offset();
    const ElementDynamic* send_ptr =
        world.elements_dynamic.data() + static_cast<size_t>(owned_offset) * n_elems_root;
    const int send_count = world.local_owned_rows() * n_elems_root * 2;

    MPI_Gatherv(send_ptr, send_count, MPI_DOUBLE,
                global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    return global;
}

// Validate simulation results (rank 0 only, on gathered global data)
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

    int rank = 0;
    int size = 1;
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    if (n_elems_root < size) {
        if (rank == 0) {
            printf("Error: grid size (%d) must be >= number of MPI ranks (%d)\n",
                   n_elems_root, size);
        }
        MPI_Finalize();
        return 1;
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    // Build this rank's local (row-decomposed) portion of the mesh
    World world;
    buildSquare2D(world, n_elems_root, rank, size);

    // Calculate memory usage (aggregate across all ranks, excluding halo
    // duplication, matching the original single-process totals)
    const size_t local_static_mem = world.local_owned_rows() * static_cast<size_t>(n_elems_root) * sizeof(ElementStatic);
    const size_t local_dynamic_mem = world.local_owned_rows() * static_cast<size_t>(n_elems_root) * sizeof(ElementDynamic) * 2;
    size_t static_mem = 0;
    size_t dynamic_mem = 0;
    MPI_Reduce(&local_static_mem, &static_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dynamic_mem, &dynamic_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    auto end = std::chrono::high_resolution_clock::now();
    long local_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather full global results (needed for validation/hash/print output)
    std::vector<ElementDynamic> global_elements = gatherGlobalResults(world, rank, size);

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
        const uint64_t hash = computeHash(global_elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global_elements.size());
            for (const auto& elem : global_elements) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(global_elements);
            if (!valid) {
                MPI_Abort(MPI_COMM_WORLD, 1);
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
