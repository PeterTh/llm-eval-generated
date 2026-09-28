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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices into the local (owned+halo) array
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state (holds this MPI rank's local slice of the mesh: owned rows plus
// one halo row on each side that mirrors the boundary rows of the neighbor ranks)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;

    // Global problem description
    int n_elems_root = 0;   // grid is n_elems_root x n_elems_root
    idx_t n_elems_global = 0;

    // Row-based domain decomposition (rows are the outer/x dimension, so each
    // row is fully contiguous in memory and owned by exactly one rank)
    int row_start = 0;      // first global row owned by this rank
    int local_rows = 0;     // number of rows owned by this rank
    idx_t local_n_elems = 0; // local_rows * n_elems_root

    bool has_top_halo = false;
    bool has_bottom_halo = false;
    idx_t halo_top_offset = 0;
    idx_t halo_bottom_offset = 0;

    int mpi_rank = 0;
    int mpi_size = 1;
    int prev_rank = MPI_PROC_NULL;
    int next_rank = MPI_PROC_NULL;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Map a global element index (row-major: x * n_elems_root + y) to a local
// storage index (owned range, or one of the two halo ranges) for this rank.
inline idx_t globalToLocal(const World& world, int gx, int gy) {
    if (gx >= world.row_start && gx < world.row_start + world.local_rows) {
        return static_cast<idx_t>(gx - world.row_start) * world.n_elems_root + gy;
    }
    if (gx == world.row_start - 1) {
        return world.halo_top_offset + gy;
    }
    // gx == world.row_start + world.local_rows
    return world.halo_bottom_offset + gy;
}

// Build a 2D square grid as an unstructured mesh, distributed across MPI ranks
// by contiguous row blocks. Each rank stores its owned rows plus one ghost
// row on each side (mirroring the adjacent rank's boundary row) so that
// neighbor lookups (up/down/left/right) never need cross-rank indirection
// during the compute step.
void buildSquare2D(World& world, const int n_elems_root, int mpi_rank, int mpi_size) {
    world.n_elems_root = n_elems_root;
    world.n_elems_global = static_cast<idx_t>(n_elems_root) * n_elems_root;
    world.mpi_rank = mpi_rank;
    world.mpi_size = mpi_size;

    // Distribute rows as evenly as possible: the first (n_elems_root % size)
    // ranks get one extra row.
    const int rows_per_rank = n_elems_root / mpi_size;
    const int remainder = n_elems_root % mpi_size;
    const int extra_before = std::min(mpi_rank, remainder);
    world.row_start = mpi_rank * rows_per_rank + extra_before;
    world.local_rows = rows_per_rank + (mpi_rank < remainder ? 1 : 0);
    world.local_n_elems = static_cast<idx_t>(world.local_rows) * n_elems_root;

    world.has_top_halo = (world.row_start > 0);
    world.has_bottom_halo = (world.row_start + world.local_rows < n_elems_root);

    world.prev_rank = world.has_top_halo ? mpi_rank - 1 : MPI_PROC_NULL;
    world.next_rank = world.has_bottom_halo ? mpi_rank + 1 : MPI_PROC_NULL;

    idx_t total_local = world.local_n_elems;
    world.halo_top_offset = total_local;
    if (world.has_top_halo) total_local += n_elems_root;
    world.halo_bottom_offset = total_local;
    if (world.has_bottom_halo) total_local += n_elems_root;

    // Initialize materials (replicated on every rank; negligible size)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements (owned rows + halo rows)
    world.elements_static.resize(world.local_n_elems);
    world.elements_dynamic.resize(total_local);
    world.elements_dynamic_swap.resize(total_local);

    // Initialize owned elements with default material and zero energy
    for (idx_t i = 0; i < world.local_n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    for (idx_t i = 0; i < total_local; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity: each owned element connects to its neighbors in the
    // 2D grid. Neighbors that fall outside this rank's owned rows resolve to
    // the appropriate halo slot.
    for (int local_x = 0; local_x < world.local_rows; ++local_x) {
        const int x = world.row_start + local_x;
        for (int y = 0; y < n_elems_root; ++y) {
            const idx_t idx = static_cast<idx_t>(local_x) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within global bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    elem.connected_idx[elem.num_connections] = globalToLocal(world, nx, ny);
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    // (only applied on the rank(s) that own the corresponding global row)
    const int last = n_elems_root - 1;
    auto setCornerMaterial = [&](int gx, int gy, idx_t mat_id) {
        if (gx >= world.row_start && gx < world.row_start + world.local_rows) {
            const idx_t idx = static_cast<idx_t>(gx - world.row_start) * n_elems_root + gy;
            world.elements_static[idx].material_idx = mat_id;
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

// Exchange boundary rows with neighbor ranks so that this rank's halo rows
// hold up-to-date current_energy values for the current iteration.
void exchangeHalos(World& world) {
    if (world.mpi_size == 1) return;

    const int n = world.n_elems_root;
    MPI_Datatype dyn_type = MPI_DOUBLE;
    // ElementDynamic is two contiguous doubles; send/recv as raw doubles.
    static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t), "unexpected padding");

    ElementDynamic* first_row = world.elements_dynamic.data();
    ElementDynamic* last_row = world.elements_dynamic.data() +
                               static_cast<idx_t>(world.local_rows - 1) * n;
    ElementDynamic* halo_top = world.elements_dynamic.data() + world.halo_top_offset;
    ElementDynamic* halo_bottom = world.elements_dynamic.data() + world.halo_bottom_offset;

    const int count = n * 2; // 2 doubles per element

    // Send our first owned row up / receive prev rank's last row into halo_top
    MPI_Sendrecv(first_row, count, dyn_type, world.prev_rank, 0,
                 halo_top, count, dyn_type, world.prev_rank, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send our last owned row down / receive next rank's first row into halo_bottom
    MPI_Sendrecv(last_row, count, dyn_type, world.next_rank, 1,
                 halo_bottom, count, dyn_type, world.next_rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const idx_t n_elems = world.local_n_elems;

    for (int iter = 0; iter < n_iters; ++iter) {
        exchangeHalos(world);

        // Update all owned elements
        for (idx_t i = 0; i < n_elems; ++i) {
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

        // Swap buffers (halo slots get refreshed by exchangeHalos next iteration)
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Gather each rank's owned elements (in global row order) into a single
// vector on rank 0, matching the layout of the original single-process code.
std::vector<ElementDynamic> gatherGlobalDynamic(const World& world) {
    std::vector<int> counts;
    std::vector<int> displs;
    if (world.mpi_rank == 0) {
        counts.resize(world.mpi_size);
        displs.resize(world.mpi_size);
    }

    const int local_count = static_cast<int>(world.local_n_elems) * 2; // doubles
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<ElementDynamic> global;
    if (world.mpi_rank == 0) {
        int offset = 0;
        for (int r = 0; r < world.mpi_size; ++r) {
            displs[r] = offset;
            offset += counts[r];
        }
        global.resize(world.n_elems_global);
    }

    MPI_Gatherv(world.elements_dynamic.data(), local_count, MPI_DOUBLE,
                world.mpi_rank == 0 ? global.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    return global;
}

// Validate simulation results (expects the full, gathered global state)
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

    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (n_elems_root < mpi_size) {
        if (mpi_rank == 0) {
            printf("Error: grid size (-n %d) must be >= number of MPI ranks (%d)\n",
                   n_elems_root, mpi_size);
        }
        MPI_Finalize();
        return 1;
    }

    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * n_elems_root;

    if (mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %llu elements\n", n_elems_root, n_elems_root,
               static_cast<unsigned long long>(n_elems));
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", mpi_size);
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    // Build this rank's slice of the unstructured mesh
    World world;
    buildSquare2D(world, n_elems_root, mpi_rank, mpi_size);

    // Calculate memory usage (global problem size, as in the original)
    if (mpi_rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // Run simulation (synchronize ranks before/after for a fair timing window)
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    long long duration_ms = 0;
    long long local_duration_ll = local_duration_ms;
    MPI_Allreduce(&local_duration_ll, &duration_ms, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Gather full result vector (in original global order) onto rank 0
    std::vector<ElementDynamic> global_dynamic = gatherGlobalDynamic(world);

    if (mpi_rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);

        // Calculate performance metrics
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

        // Compute hash for verification
        const uint64_t hash = computeHash(global_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global_dynamic.size());
            for (const auto& elem : global_dynamic) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(global_dynamic);
            if (!valid) {
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
