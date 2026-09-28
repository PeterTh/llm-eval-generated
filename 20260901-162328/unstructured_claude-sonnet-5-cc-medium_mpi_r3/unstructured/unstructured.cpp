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

// World state.
//
// The global n_elems_root x n_elems_root grid is partitioned by contiguous
// row ranges ("row-block" domain decomposition) across MPI ranks. Since every
// element only connects to its immediate grid neighbors, a row-based split
// means that inter-rank dependencies only occur across row boundaries (the
// left/right neighbors within a row are always local to a rank). Each rank
// therefore only stores its own rows plus one ghost row above and below,
// which are refreshed from neighboring ranks every iteration via halo
// exchange. This keeps per-rank memory proportional to (local rows + 2)
// instead of the full global grid, giving genuine distributed-memory scaling.
struct World {
    std::vector<Material> materials;
    // Connectivity for owned rows only (compact, size = local_n_rows * n_elems_root).
    std::vector<ElementStatic> elements_static;
    // Dynamic state for owned rows plus one ghost row on each side
    // (padded, size = (local_n_rows + 2) * n_elems_root).
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;

    idx_t n_elems_root = 0;   // global grid dimension (NxN)
    idx_t x_start = 0;        // global row index of this rank's first owned row
    idx_t local_n_rows = 0;   // number of rows owned by this rank

    int prev_rank = MPI_PROC_NULL;  // rank owning the row above x_start (or PROC_NULL)
    int next_rank = MPI_PROC_NULL;  // rank owning the row below x_start+local_n_rows-1 (or PROC_NULL)
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Compute a block distribution of `n_rows` global rows across `size` ranks:
// the first (n_rows % size) ranks get one extra row. Ranks in excess of
// n_rows (when size > n_rows) simply receive zero rows.
void computeRowDistribution(int n_rows, int size, std::vector<int>& counts, std::vector<int>& starts) {
    counts.assign(size, 0);
    starts.assign(size, 0);
    const int base = n_rows / size;
    const int rem = n_rows % size;
    int cur = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        starts[r] = cur;
        cur += counts[r];
    }
}

// Build the local partition of a 2D square grid unstructured mesh.
// Each rank builds only the rows it owns (plus connectivity referencing
// ghost-row storage for neighbors across rank boundaries).
void buildSquare2D(World& world, const int n_elems_root, int rank, int size) {
    world.n_elems_root = static_cast<idx_t>(n_elems_root);

    std::vector<int> counts, starts;
    computeRowDistribution(n_elems_root, size, counts, starts);

    world.local_n_rows = static_cast<idx_t>(counts[rank]);
    world.x_start = static_cast<idx_t>(starts[rank]);

    world.prev_rank = MPI_PROC_NULL;
    for (int r = rank - 1; r >= 0; --r) {
        if (counts[r] > 0) { world.prev_rank = r; break; }
    }
    world.next_rank = MPI_PROC_NULL;
    for (int r = rank + 1; r < size; ++r) {
        if (counts[r] > 0) { world.next_rank = r; break; }
    }

    // Initialize materials (tiny, replicated on every rank)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    if (world.local_n_rows == 0) {
        return;  // This rank owns no elements (only possible if size > n_elems_root)
    }

    const idx_t padded_rows = world.local_n_rows + 2;
    const size_t n_root = static_cast<size_t>(n_elems_root);
    const size_t padded_n = static_cast<size_t>(padded_rows) * n_root;
    const size_t local_n = static_cast<size_t>(world.local_n_rows) * n_root;

    world.elements_static.resize(local_n);
    world.elements_dynamic.resize(padded_n);
    world.elements_dynamic_swap.resize(padded_n);

    for (size_t i = 0; i < padded_n; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    const int last = n_elems_root - 1;

    // Build connectivity: each element connects to its neighbors in the 2D grid.
    // Neighbor indices are expressed in the padded dynamic-state index space,
    // so up/down connections that cross a rank boundary transparently resolve
    // to this rank's ghost-row storage.
    for (idx_t p = 0; p < world.local_n_rows; ++p) {
        const idx_t gx = world.x_start + p;
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t static_idx = static_cast<size_t>(p) * n_root + static_cast<size_t>(y);
            ElementStatic& elem = world.elements_static[static_idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const long long nx = static_cast<long long>(gx) + offsets[n][0];
                const long long ny = static_cast<long long>(y) + offsets[n][1];

                // Check if neighbor is within global bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    // (p + 1) is this element's own row in padded ghost-row space;
                    // offsets[n][0] shifts by one row up/down within that space.
                    const idx_t neighbor_padded_row = p + 1 + static_cast<idx_t>(offsets[n][0]);
                    const size_t neighbor_idx = static_cast<size_t>(neighbor_padded_row) * n_root + static_cast<size_t>(ny);
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }

            // Set corner elements as inflow/outflow to create interesting dynamics
            if (gx == 0 && y == 0) {
                elem.material_idx = INFLOW_MAT_ID;
            } else if (gx == 0 && y == last) {
                elem.material_idx = OUTFLOW_MAT_ID;
            } else if (static_cast<int>(gx) == last && y == 0) {
                elem.material_idx = OUTFLOW_MAT_ID;
            } else if (static_cast<int>(gx) == last && y == last) {
                elem.material_idx = INFLOW_MAT_ID;
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

// Exchange ghost rows with neighboring ranks so that up/down connections
// crossing a rank boundary see up-to-date neighbor state. Uses MPI_PROC_NULL
// as a no-op for ranks at the global domain boundary.
void exchangeHalo(World& world) {
    if (world.local_n_rows == 0) {
        return;
    }

    const size_t n_root = static_cast<size_t>(world.n_elems_root);
    const size_t row_bytes = n_root * sizeof(ElementDynamic);
    const idx_t local_n_rows = world.local_n_rows;

    ElementDynamic* top_owned    = &world.elements_dynamic[1 * n_root];
    ElementDynamic* bottom_owned = &world.elements_dynamic[static_cast<size_t>(local_n_rows) * n_root];
    ElementDynamic* top_ghost    = &world.elements_dynamic[0];
    ElementDynamic* bottom_ghost = &world.elements_dynamic[static_cast<size_t>(local_n_rows + 1) * n_root];

    constexpr int TAG_UP = 0;
    constexpr int TAG_DOWN = 1;

    // Send our first owned row up to prev_rank; receive our bottom ghost
    // from next_rank's first owned row.
    MPI_Sendrecv(top_owned, static_cast<int>(row_bytes), MPI_BYTE, world.prev_rank, TAG_UP,
                 bottom_ghost, static_cast<int>(row_bytes), MPI_BYTE, world.next_rank, TAG_UP,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send our last owned row down to next_rank; receive our top ghost
    // from prev_rank's last owned row.
    MPI_Sendrecv(bottom_owned, static_cast<int>(row_bytes), MPI_BYTE, world.next_rank, TAG_DOWN,
                 top_ghost, static_cast<int>(row_bytes), MPI_BYTE, world.prev_rank, TAG_DOWN,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    if (world.local_n_rows == 0) {
        return;
    }

    const size_t n_root = static_cast<size_t>(world.n_elems_root);
    const size_t local_n = static_cast<size_t>(world.local_n_rows) * n_root;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Refresh ghost rows with the latest neighbor-owned state before computing
        exchangeHalo(world);

        // Update all owned elements
        for (size_t i = 0; i < local_n; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const size_t dyn_idx = i + n_root;  // shift into padded (ghost-aware) index space
            const ElementDynamic& elem_dyn = world.elements_dynamic[dyn_idx];
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
            ElementDynamic& elem_write = world.elements_dynamic_swap[dyn_idx];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }

        // Swap buffers (ghost rows are stale after the swap but get
        // refreshed again by exchangeHalo() at the start of the next iteration)
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (collective across all ranks; only rank 0 prints/returns meaningfully)
bool validateResults(const World& world, int rank) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    const size_t n_root = static_cast<size_t>(world.n_elems_root);
    for (idx_t p = 0; p < world.local_n_rows; ++p) {
        for (size_t y = 0; y < n_root; ++y) {
            const size_t dyn_idx = static_cast<size_t>(p + 1) * n_root + y;
            const ElementDynamic& elem = world.elements_dynamic[dyn_idx];
            energy_sum += elem.current_energy;
            flux_sum += elem.total_flux;
            energy_max = std::max(elem.current_energy, energy_max);
            energy_min = std::min(elem.current_energy, energy_min);
        }
    }

    val_t global_energy_sum = 0.0, global_flux_sum = 0.0;
    val_t global_energy_max = std::numeric_limits<val_t>::lowest();
    val_t global_energy_min = std::numeric_limits<val_t>::max();

    MPI_Reduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    int valid_flag = 1;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_energy_sum);
        printf("  Flux sum: %.2f\n", global_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);

        // Check for numerical issues
        constexpr val_t energy_epsilon = 1e-8;

        if (!std::isfinite(global_energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid_flag = 0;
        }

        if (valid_flag && std::abs(global_energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            // Don't fail validation as this can happen with external flows
        }

        if (valid_flag && !std::isfinite(global_flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid_flag = 0;
        }

        if (valid_flag && (!std::isfinite(global_energy_max) || !std::isfinite(global_energy_min))) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid_flag = 0;
        }

        if (valid_flag) {
            printf("  Validation: PASSED\n");
        }
    }

    MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid_flag != 0;
}

// Compute a hash of the results for verification (collective across all ranks;
// meaningful result is only valid on rank 0). Uses each element's global flat
// index (consistent with the original single-process indexing scheme) so the
// hash is independent of how rows are distributed across ranks.
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    const size_t n_root = static_cast<size_t>(world.n_elems_root);

    for (idx_t p = 0; p < world.local_n_rows; ++p) {
        const idx_t gx = world.x_start + p;
        for (size_t y = 0; y < n_root; ++y) {
            const uint64_t i = static_cast<uint64_t>(gx) * n_root + y;
            const size_t dyn_idx = static_cast<size_t>(p + 1) * n_root + y;
            const ElementDynamic& elem = world.elements_dynamic[dyn_idx];
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elem.current_energy);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elem.total_flux);
            hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }

    uint64_t global_hash = 0;
    MPI_Reduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    return global_hash;
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

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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

    const long long n_elems = static_cast<long long>(n_elems_root) * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("MPI ranks: %d\n", size);
        printf("Grid size: %d x %d = %lld elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    // Build this rank's local partition of the unstructured mesh
    World world;
    buildSquare2D(world, n_elems_root, rank, size);

    // Calculate memory usage (summed across ranks for a global picture)
    const size_t local_static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t local_dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) +
                                      world.elements_dynamic_swap.size() * sizeof(ElementDynamic);
    size_t total_static_mem = 0, total_dynamic_mem = 0;
    MPI_Reduce(&local_static_mem, &total_static_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dynamic_mem, &total_dynamic_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const size_t total_mem = total_static_mem + total_dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               total_static_mem / (1024.0 * 1024.0),
               total_dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    auto end = std::chrono::high_resolution_clock::now();
    const long long local_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Compute hash for verification (collective; all ranks must call this)
    const uint64_t hash = computeHash(world);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);

        // Calculate performance metrics (based on total problem size across all ranks)
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (n_measured_iters * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation (gather full global-order array to rank 0)
    if (printResults) {
        std::vector<int> counts, starts;
        computeRowDistribution(n_elems_root, size, counts, starts);

        std::vector<int> recvcounts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            recvcounts[r] = counts[r] * n_elems_root;
            displs[r] = starts[r] * n_elems_root;
        }

        const size_t n_root = static_cast<size_t>(n_elems_root);
        std::vector<double> localEnergy(static_cast<size_t>(world.local_n_rows) * n_root);
        for (idx_t p = 0; p < world.local_n_rows; ++p) {
            for (size_t y = 0; y < n_root; ++y) {
                const size_t dyn_idx = static_cast<size_t>(p + 1) * n_root + y;
                localEnergy[static_cast<size_t>(p) * n_root + y] = world.elements_dynamic[dyn_idx].current_energy;
            }
        }

        std::vector<double> energyData;
        if (rank == 0) {
            energyData.resize(static_cast<size_t>(n_elems));
        }
        MPI_Gatherv(localEnergy.data(), recvcounts[rank], MPI_DOUBLE,
                    energyData.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }

    // Validation (collective; all ranks must call this)
    bool valid = true;
    if (validate) {
        valid = validateResults(world, rank);
    }

    MPI_Finalize();

    if (validate && !valid) {
        return 1;
    }
    return 0;
}
