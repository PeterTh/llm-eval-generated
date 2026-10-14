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

// The MPI halo exchange and the gather rely on this layout
static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t), "ElementDynamic must be two packed doubles");

// Local part of the mesh owned by one MPI rank.
// The grid is split into contiguous blocks of rows. Local storage is padded with one ghost row
// on each side of the owned rows:
//   local row 0          : ghost copy of global row x_begin-1 (owned by the rank above)
//   local rows 1..L      : owned global rows x_begin..x_end-1 (L = x_end - x_begin)
//   local row L+1        : ghost copy of global row x_end (owned by the rank below)
// Connectivity indices refer to this padded local layout.
struct World {
    int n_root = 0;   // Grid size (NxN elements), global
    int x_begin = 0;  // First owned global row
    int x_end = 0;    // One past the last owned global row
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Message tags for halo exchange
constexpr int TAG_TO_ABOVE = 0;  // Message sent to rank-1 (carries our first owned row)
constexpr int TAG_TO_BELOW = 1;  // Message sent to rank+1 (carries our last owned row)

// Block distribution of n_root grid rows over n_ranks ranks. Ranks beyond n_root get no rows.
void rowRange(int n_root, int n_ranks, int rank, int& x_begin, int& x_end) {
    const int base = n_root / n_ranks;
    const int rem = n_root % n_ranks;
    x_begin = rank * base + std::min(rank, rem);
    x_end = x_begin + base + (rank < rem ? 1 : 0);
}

// Local padded index of global element (x, y). x must lie in [x_begin-1, x_end].
inline idx_t localIndex(const World& world, int x, int y) {
    return static_cast<idx_t>(x - world.x_begin + 1) * world.n_root + y;
}

// Assign a material to a global grid element, if this rank owns it
void setMaterial(World& world, int x, int y, idx_t mat_id) {
    if (x >= world.x_begin && x < world.x_end) {
        world.elements_static[localIndex(world, x, y)].material_idx = mat_id;
    }
}

// Build the locally owned part of a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world) {
    const int n_elems_root = world.n_root;
    const int n_owned = world.x_end - world.x_begin;
    const size_t n_padded = static_cast<size_t>(n_owned + 2) * n_elems_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements (owned rows plus ghost rows)
    world.elements_static.resize(n_padded);
    world.elements_dynamic.resize(n_padded);
    world.elements_dynamic_swap.resize(n_padded);

    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < n_padded; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity: each owned element connects to its neighbors in 2D grid
    for (int x = world.x_begin; x < world.x_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            ElementStatic& elem = world.elements_static[localIndex(world, x, y)];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds (rows outside the owned block are ghost rows)
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    elem.connected_idx[elem.num_connections] = localIndex(world, nx, ny);
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    setMaterial(world, 0, 0, INFLOW_MAT_ID);
    setMaterial(world, 0, last, OUTFLOW_MAT_ID);
    setMaterial(world, last, 0, OUTFLOW_MAT_ID);
    setMaterial(world, last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Advance the owned elements in local index range [begin, end) by one iteration.
// Reads the current buffer (owned rows and ghost rows), writes the swap buffer.
void updateRange(World& world, size_t begin, size_t end) {
    const ElementStatic* statics = world.elements_static.data();
    const ElementDynamic* current = world.elements_dynamic.data();
    ElementDynamic* next = world.elements_dynamic_swap.data();
    const Material* materials = world.materials.data();

    for (size_t i = begin; i < end; ++i) {
        const ElementStatic& elem_static = statics[i];
        const ElementDynamic& elem_dyn = current[i];
        const Material& mat = materials[elem_static.material_idx];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            const ElementDynamic& neighbor_dyn = current[neighbor_idx];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }

        // Update element state
        ElementDynamic& elem_write = next[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations.
// Each iteration exchanges the current_energy of boundary rows with the neighboring ranks.
// Interior rows are updated while the exchange is in flight.
void runSimulation(World& world, const int n_iters, const int rank) {
    const size_t n = world.n_root;
    const size_t n_owned = world.x_end - world.x_begin;
    const bool has_above = n_owned > 0 && world.x_begin > 0;
    const bool has_below = n_owned > 0 && world.x_end < world.n_root;
    const int rank_above = rank - 1;
    const int rank_below = rank + 1;

    // Local rows: owned row k (1-based) starts at index k*n
    const size_t first_owned_row = n;
    const size_t last_owned_row = n_owned * n;
    const size_t ghost_below_row = (n_owned + 1) * n;

    // One grid row of current_energy values, strided over ElementDynamic
    MPI_Datatype row_type;
    MPI_Type_vector(static_cast<int>(n), 1, static_cast<int>(sizeof(ElementDynamic) / sizeof(val_t)),
                    MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    for (int iter = 0; iter < n_iters; ++iter) {
        // Halo exchange of the current buffer (ghost rows are filled by the neighbors)
        ElementDynamic* current = world.elements_dynamic.data();
        MPI_Request reqs[4];
        int n_reqs = 0;

        if (has_above) {
            MPI_Irecv(&current[0].current_energy, 1, row_type, rank_above, TAG_TO_BELOW,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Isend(&current[first_owned_row].current_energy, 1, row_type, rank_above, TAG_TO_ABOVE,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
        }
        if (has_below) {
            MPI_Irecv(&current[ghost_below_row].current_energy, 1, row_type, rank_below, TAG_TO_ABOVE,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Isend(&current[last_owned_row].current_energy, 1, row_type, rank_below, TAG_TO_BELOW,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
        }

        // Rows 2..L-1 only depend on owned rows: update them while the halo is in flight
        if (n_owned > 2) {
            updateRange(world, 2 * n, last_owned_row);
        }

        MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

        // Boundary rows depend on the ghost rows
        if (n_owned == 1) {
            updateRange(world, first_owned_row, 2 * n);
        } else if (n_owned >= 2) {
            updateRange(world, first_owned_row, 2 * n);
            updateRange(world, last_owned_row, ghost_below_row);
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }

    MPI_Type_free(&row_type);
}

// Gather the owned state of all ranks onto rank 0, in global element order
std::vector<ElementDynamic> gatherElements(const World& world, const int rank, const int n_ranks) {
    const int n = world.n_root;

    MPI_Datatype elem_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &elem_type);
    MPI_Type_commit(&elem_type);

    std::vector<int> counts(n_ranks);
    std::vector<int> displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        int x_begin = 0;
        int x_end = 0;
        rowRange(n, n_ranks, r, x_begin, x_end);
        counts[r] = (x_end - x_begin) * n;
        displs[r] = x_begin * n;
    }

    std::vector<ElementDynamic> full(rank == 0 ? static_cast<size_t>(n) * n : 0);
    const ElementDynamic* owned = world.elements_dynamic.data() + n;  // Skip ghost row above
    MPI_Gatherv(owned, counts[rank], elem_type, full.data(), counts.data(), displs.data(), elem_type,
                0, MPI_COMM_WORLD);

    MPI_Type_free(&elem_type);
    return full;
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
// The hash is an XOR over all elements, so each rank hashes its own elements (with their global
// indices) and the partial results are combined with a bitwise XOR reduction onto rank 0.
uint64_t computeHash(const World& world, const int rank) {
    const size_t n_owned = world.x_end - world.x_begin;
    const size_t n = world.n_root;
    const ElementDynamic* elements = world.elements_dynamic.data() + n;  // Skip ghost row above
    const size_t global_offset = static_cast<size_t>(world.x_begin) * n;

    uint64_t local_hash = 0;
    for (size_t i = 0; i < n_owned * n; ++i) {
        const size_t global_i = global_offset + i;
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        local_hash ^= (*e_ptr + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_i) * 0xbf58476d1ce4e5b9ULL;
    }

    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    return rank == 0 ? hash : 0;
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

// Benchmark driver. All ranks run it with the same arguments and return the same code.
int runBenchmark(int argc, char** argv, const int rank, const int n_ranks) {
    const bool is_root = rank == 0;

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
            if (is_root) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (is_root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (n_elems_root < 1) {
        if (is_root) {
            printf("Grid size must be at least 1\n");
            printUsage(argv[0]);
        }
        return 1;
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (is_root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("MPI ranks: %d\n", n_ranks);
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }

    World world;
    world.n_root = n_elems_root;
    rowRange(n_elems_root, n_ranks, rank, world.x_begin, world.x_end);
    buildSquare2D(world);

    if (is_root) {
        // Calculate memory usage (whole mesh)
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

    // Run simulation (timed as the slowest rank)
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, rank);

    auto end = std::chrono::high_resolution_clock::now();
    const double local_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (is_root) {
        const long duration_ms = static_cast<long>(max_ms);
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
    }

    // Compute hash for verification
    const uint64_t hash = computeHash(world, rank);
    if (is_root) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Gather the full state for result printing and validation (all ranks take the same branch)
    const bool need_full_state = printResults || validate;
    std::vector<ElementDynamic> full_state;
    if (need_full_state) {
        full_state = gatherElements(world, rank, n_ranks);
    }

    // Print results for external validation
    if (is_root && printResults) {
        std::vector<double> energyData;
        energyData.reserve(full_state.size());
        for (const auto& elem : full_state) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }

    // Validation
    int valid = 1;
    if (validate) {
        if (is_root) {
            valid = validateResults(full_state) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    const int status = runBenchmark(argc, argv, rank, n_ranks);

    MPI_Finalize();
    return status;
}
