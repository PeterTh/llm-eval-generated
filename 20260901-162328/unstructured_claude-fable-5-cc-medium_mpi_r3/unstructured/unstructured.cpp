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

// World state (per-rank partition of the global mesh)
//
// The global grid rows [0, n_elems_root) are block-distributed across ranks.
// Each rank stores an extended slab of rows [ext_row_begin, ext_row_end) that
// includes one ghost row on each side where a neighboring rank exists. All
// element indices in the local connectivity are local to the extended slab.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;       // owned elements only
    std::vector<ElementDynamic> elements_dynamic;     // extended slab (with ghosts)
    std::vector<ElementDynamic> elements_dynamic_swap;

    int n_elems_root = 0;   // global grid dimension
    int row_begin = 0;      // first owned global row
    int row_end = 0;        // one past last owned global row
    int ext_row_begin = 0;  // first row of extended slab (incl. ghost)
    int prev_rank = MPI_PROC_NULL;  // rank owning row_begin - 1
    int next_rank = MPI_PROC_NULL;  // rank owning row_end

    int ownedRows() const { return row_end - row_begin; }
    // Local index of the first owned element within the extended slab
    size_t ownedOffset() const {
        return static_cast<size_t>(row_begin - ext_row_begin) * n_elems_root;
    }
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Block distribution of global rows across ranks: ranks < remainder get one
// extra row, and ranks owning rows are contiguous starting at rank 0.
static void rowRange(const int n_rows, const int size, const int rank,
                     int& begin, int& end) {
    const int base = n_rows / size;
    const int rem = n_rows % size;
    begin = rank * base + std::min(rank, rem);
    end = begin + base + (rank < rem ? 1 : 0);
}

// Build this rank's partition of a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int size) {
    world.n_elems_root = n_elems_root;
    rowRange(n_elems_root, size, rank, world.row_begin, world.row_end);

    const int n_owned_rows = world.ownedRows();
    world.prev_rank = (n_owned_rows > 0 && world.row_begin > 0) ? rank - 1 : MPI_PROC_NULL;
    world.next_rank = (n_owned_rows > 0 && world.row_end < n_elems_root) ? rank + 1 : MPI_PROC_NULL;

    world.ext_row_begin = world.row_begin - (world.prev_rank != MPI_PROC_NULL ? 1 : 0);
    const int ext_row_end = world.row_end + (world.next_rank != MPI_PROC_NULL ? 1 : 0);

    const size_t n_local = static_cast<size_t>(n_owned_rows) * n_elems_root;
    const size_t n_ext = static_cast<size_t>(ext_row_end - world.ext_row_begin) * n_elems_root;

    // Initialize materials (replicated on all ranks)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_ext);
    world.elements_dynamic_swap.resize(n_ext);

    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < n_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    for (size_t i = 0; i < n_ext; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        world.elements_dynamic_swap[i].current_energy = 0.0;
        world.elements_dynamic_swap[i].total_flux = 0.0;
    }

    // Build connectivity for owned elements: each element connects to its
    // neighbors in the 2D grid. Connection targets are indices into the
    // extended slab (owned + ghost rows).
    const size_t ext_offset = static_cast<size_t>(world.ext_row_begin) * n_elems_root;
    for (int x = world.row_begin; x < world.row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t local_idx =
                static_cast<size_t>(x - world.row_begin) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const size_t neighbor_idx =
                        static_cast<size_t>(nx) * n_elems_root + ny - ext_offset;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    auto setMaterial = [&](int x, int y, idx_t mat) {
        if (x >= world.row_begin && x < world.row_end) {
            world.elements_static[static_cast<size_t>(x - world.row_begin) * n_elems_root + y]
                .material_idx = mat;
        }
    };
    setMaterial(0, 0, INFLOW_MAT_ID);
    setMaterial(0, last, OUTFLOW_MAT_ID);
    setMaterial(last, 0, OUTFLOW_MAT_ID);
    setMaterial(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update the owned elements in local rows [row_lo, row_hi) (relative to the
// first owned row), reading from elements_dynamic and writing to the swap.
static void updateRows(World& world, const int row_lo, const int row_hi) {
    const size_t owned_off = world.ownedOffset();
    const size_t begin = static_cast<size_t>(row_lo) * world.n_elems_root;
    const size_t end = static_cast<size_t>(row_hi) * world.n_elems_root;

    for (size_t i = begin; i < end; ++i) {
        const ElementStatic& elem_static = world.elements_static[i];
        const ElementDynamic& elem_dyn = world.elements_dynamic[owned_off + i];
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
        ElementDynamic& elem_write = world.elements_dynamic_swap[owned_off + i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const int n_root = world.n_elems_root;
    const int n_owned_rows = world.ownedRows();
    const size_t owned_off = world.ownedOffset();
    const size_t row_bytes_count = static_cast<size_t>(n_root);

    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange ghost rows: send our first/last owned row, receive the
        // neighbor rows into the ghost slots. Communication is overlapped
        // with the update of interior rows.
        MPI_Request reqs[4];
        int n_reqs = 0;

        if (world.prev_rank != MPI_PROC_NULL) {
            MPI_Irecv(&world.elements_dynamic[owned_off - row_bytes_count],
                      2 * n_root, MPI_DOUBLE, world.prev_rank, 0,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Isend(&world.elements_dynamic[owned_off],
                      2 * n_root, MPI_DOUBLE, world.prev_rank, 1,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
        }
        if (world.next_rank != MPI_PROC_NULL) {
            MPI_Irecv(&world.elements_dynamic[owned_off + static_cast<size_t>(n_owned_rows) * n_root],
                      2 * n_root, MPI_DOUBLE, world.next_rank, 1,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Isend(&world.elements_dynamic[owned_off + static_cast<size_t>(n_owned_rows - 1) * n_root],
                      2 * n_root, MPI_DOUBLE, world.next_rank, 0,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
        }

        // Interior rows do not depend on ghost data
        const int interior_lo = std::min(1, n_owned_rows);
        const int interior_hi = std::max(n_owned_rows - 1, interior_lo);
        updateRows(world, interior_lo, interior_hi);

        if (n_reqs > 0) {
            MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
        }

        // Boundary rows (need up-to-date ghost data)
        if (n_owned_rows > 0) {
            updateRows(world, 0, interior_lo);
            updateRows(world, interior_hi, n_owned_rows);
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Gather the owned dynamic elements from all ranks onto rank 0, in global
// element order. Returns an empty vector on non-root ranks.
std::vector<ElementDynamic> gatherResults(const World& world, const int rank, const int size) {
    const int n_root = world.n_elems_root;
    const int n_owned = world.ownedRows() * n_root;

    std::vector<int> counts, displs;
    std::vector<ElementDynamic> global;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            int b, e;
            rowRange(n_root, size, r, b, e);
            counts[r] = (e - b) * n_root * 2;  // 2 doubles per element
            displs[r] = b * n_root * 2;
        }
        global.resize(static_cast<size_t>(n_root) * n_root);
    }

    MPI_Gatherv(world.elements_dynamic.data() + world.ownedOffset(),
                n_owned * 2, MPI_DOUBLE,
                rank == 0 ? global.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return global;
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
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

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, rank, size);

    if (rank == 0) {
        // Calculate memory usage (global totals across all ranks)
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
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

    runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Gather full results onto rank 0 for hashing/validation/output
    std::vector<ElementDynamic> global_dynamic = gatherResults(world, rank, size);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9;

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
    }

    // Validation
    int exit_code = 0;
    if (validate && rank == 0) {
        bool valid = validateResults(global_dynamic);
        if (!valid) {
            exit_code = 1;
        }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exit_code;
}
