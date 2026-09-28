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

// World state (distributed: each rank owns a contiguous block of grid rows)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;        // Owned elements only
    std::vector<ElementDynamic> elements_dynamic;      // Ghost row(s) + owned elements
    std::vector<ElementDynamic> elements_dynamic_swap;

    // Decomposition info
    int n_elems_root = 0;   // Global grid dimension
    int row_start = 0;      // First owned grid row (global)
    int n_local_rows = 0;   // Number of owned grid rows
    bool has_top = false;   // Ghost row from rank-1 present
    bool has_bottom = false;// Ghost row from rank+1 present
    int owned_offset = 0;   // Index of first owned element in elements_dynamic
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh, distributed by blocks of
// grid rows (x). Connectivity indices reference the local dynamic array,
// which holds an optional top ghost row, the owned rows, and an optional
// bottom ghost row.
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int size) {
    // Block row distribution: first `rem` ranks get one extra row
    const int base = n_elems_root / size;
    const int rem = n_elems_root % size;
    const int row_start = rank * base + std::min(rank, rem);
    const int n_local_rows = base + (rank < rem ? 1 : 0);
    const int row_end = row_start + n_local_rows;

    world.n_elems_root = n_elems_root;
    world.row_start = row_start;
    world.n_local_rows = n_local_rows;
    world.has_top = (n_local_rows > 0) && (row_start > 0);
    world.has_bottom = (n_local_rows > 0) && (row_end < n_elems_root);
    world.owned_offset = (world.has_top ? 1 : 0) * n_elems_root;

    const int n_local = n_local_rows * n_elems_root;
    const int n_dyn = n_local + (world.has_top ? n_elems_root : 0) +
                      (world.has_bottom ? n_elems_root : 0);

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_dyn);
    world.elements_dynamic_swap.resize(n_dyn);

    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    for (int i = 0; i < n_dyn; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Local dynamic index of a global grid cell (x in [row_start-1, row_end])
    const auto localDynIdx = [&](int x, int y) {
        return (x - row_start) * n_elems_root + y + world.owned_offset;
    };

    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = row_start; x < row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = (x - row_start) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    elem.connected_idx[elem.num_connections] = localDynIdx(nx, ny);
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    const auto setCorner = [&](int x, int y, idx_t mat) {
        if (x >= row_start && x < row_end) {
            world.elements_static[(x - row_start) * n_elems_root + y].material_idx = mat;
        }
    };
    setCorner(0, 0, INFLOW_MAT_ID);
    setCorner(0, last, OUTFLOW_MAT_ID);
    setCorner(last, 0, OUTFLOW_MAT_ID);
    setCorner(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update elements in local range [lo, hi) (indices into elements_static)
static inline void updateRange(World& world, const size_t lo, const size_t hi) {
    const int off = world.owned_offset;
    for (size_t i = lo; i < hi; ++i) {
        const ElementStatic& elem_static = world.elements_static[i];
        const ElementDynamic& elem_dyn = world.elements_dynamic[i + off];
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
        ElementDynamic& elem_write = world.elements_dynamic_swap[i + off];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations with halo exchange each iteration.
// Communication of boundary rows is overlapped with interior computation.
void runSimulation(World& world, const int n_iters, const int rank) {
    const int n_root = world.n_elems_root;
    const size_t n_local = world.elements_static.size();
    const int off = world.owned_offset;

    // Packed energy buffers for halo exchange (one grid row each)
    std::vector<val_t> send_top(n_root), send_bottom(n_root);
    std::vector<val_t> recv_top(n_root), recv_bottom(n_root);

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request reqs[4];
        int n_reqs = 0;

        // Post halo exchange of boundary-row energies
        if (world.has_top) {
            MPI_Irecv(recv_top.data(), n_root, MPI_DOUBLE, rank - 1, 0,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
            for (int y = 0; y < n_root; ++y) {
                send_top[y] = world.elements_dynamic[off + y].current_energy;
            }
            MPI_Isend(send_top.data(), n_root, MPI_DOUBLE, rank - 1, 1,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
        }
        if (world.has_bottom) {
            MPI_Irecv(recv_bottom.data(), n_root, MPI_DOUBLE, rank + 1, 1,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
            const int last_row = off + (world.n_local_rows - 1) * n_root;
            for (int y = 0; y < n_root; ++y) {
                send_bottom[y] = world.elements_dynamic[last_row + y].current_energy;
            }
            MPI_Isend(send_bottom.data(), n_root, MPI_DOUBLE, rank + 1, 0,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
        }

        // Compute interior elements (not adjacent to ghost rows) while
        // communication is in flight
        const size_t top_edge = world.has_top ? static_cast<size_t>(n_root) : 0;
        const size_t bottom_edge =
            world.has_bottom ? n_local - static_cast<size_t>(n_root) : n_local;
        updateRange(world, top_edge, bottom_edge);

        // Finish communication and unpack ghost energies
        if (n_reqs > 0) {
            MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
        }
        if (world.has_top) {
            for (int y = 0; y < n_root; ++y) {
                world.elements_dynamic[y].current_energy = recv_top[y];
            }
        }
        if (world.has_bottom) {
            const int ghost_bottom = off + world.n_local_rows * n_root;
            for (int y = 0; y < n_root; ++y) {
                world.elements_dynamic[ghost_bottom + y].current_energy = recv_bottom[y];
            }
        }

        // Compute boundary elements that depend on ghost rows
        if (world.has_top) {
            updateRange(world, 0, top_edge);
        }
        if (world.has_bottom) {
            updateRange(world, bottom_edge, n_local);
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (rank 0, on the gathered global state)
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
    int rank = 0, size = 1;
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

    // Calculate memory usage (aggregated over all ranks)
    const size_t local_static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t local_dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    unsigned long long local_mem[2] = {local_static_mem, local_dynamic_mem};
    unsigned long long global_mem[2] = {0, 0};
    MPI_Reduce(local_mem, global_mem, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const unsigned long long total_mem = global_mem[0] + global_mem[1];
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               global_mem[0] / (1024.0 * 1024.0),
               global_mem[1] / (1024.0 * 1024.0));
        printf("\n");

        // Run simulation
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long global_duration_ms = 0;
    MPI_Reduce(&duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the full dynamic state on rank 0 in global element order
    // (row blocks are contiguous and rank-ordered, so a Gatherv suffices)
    const int local_count = static_cast<int>(world.elements_static.size()) * 2;
    std::vector<int> counts(size), displs(size);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<ElementDynamic> global_dynamic;
    if (rank == 0) {
        int disp = 0;
        for (int r = 0; r < size; ++r) {
            displs[r] = disp;
            disp += counts[r];
        }
        global_dynamic.resize(n_elems);
    }
    MPI_Gatherv(reinterpret_cast<const double*>(world.elements_dynamic.data() + world.owned_offset),
                local_count, MPI_DOUBLE,
                reinterpret_cast<double*>(global_dynamic.data()),
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exit_code = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(global_duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (global_duration_ms / 1000.0) / 1e9;

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
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
