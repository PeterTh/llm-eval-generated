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
    idx_t connected_idx[MAX_CONNECTIONS];     // Global indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state (per-rank local view)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;       // local elements
    std::vector<ElementDynamic> elements_dynamic;     // local
    std::vector<ElementDynamic> elements_dynamic_swap;// local

    // partition info
    int rank = 0;
    int size = 1;
    int start_row = 0;       // global start row
    int local_rows = 0;      // number of rows owned by this rank
    int ncols = 0;           // global number of columns (== n_elems_root)
    idx_t start_idx = 0;     // global index of first local element
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh for this rank only
void buildSquare2D(World& world, const int n_elems_root, int rank, int size) {
    world.rank = rank;
    world.size = size;
    world.ncols = n_elems_root;

    // Partition rows among ranks (first 'remainder' ranks get an extra row)
    const int base_rows = n_elems_root / size;
    const int remainder = n_elems_root % size;
    world.local_rows = base_rows + (rank < remainder ? 1 : 0);
    world.start_row = rank * base_rows + std::min(rank, remainder);

    const int local_n_elems = world.local_rows * n_elems_root;
    world.start_idx = static_cast<idx_t>(world.start_row) * n_elems_root;

    // Initialize materials (replicated on each rank)
    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate local elements
    world.elements_static.resize(local_n_elems);
    world.elements_dynamic.resize(local_n_elems);
    world.elements_dynamic_swap.resize(local_n_elems);

    // Initialize local elements
    for (int i = 0; i < local_n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity for local elements using global indices for neighbors
    for (int lx = 0; lx < world.local_rows; ++lx) {
        int x = world.start_row + lx; // global row
        for (int y = 0; y < n_elems_root; ++y) {
            const int local_idx = lx * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const idx_t neighbor_idx = static_cast<idx_t>(nx) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx; // global
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner materials if they belong to this rank
    const int last = n_elems_root - 1;
    const idx_t corners[4] = {
        0 * n_elems_root + 0,
        0 * n_elems_root + last,
        last * n_elems_root + 0,
        last * n_elems_root + last
    };
    const idx_t local_start = world.start_idx;
    const idx_t local_end_exclusive = world.start_idx + local_n_elems;
    for (int c = 0; c < 4; ++c) {
        idx_t g = corners[c];
        if (g >= local_start && g < local_end_exclusive) {
            idx_t local_idx = g - local_start;
            if (c == 0 || c == 3) world.elements_static[local_idx].material_idx = INFLOW_MAT_ID;
            else world.elements_static[local_idx].material_idx = OUTFLOW_MAT_ID;
        }
    }
}

// Helper to check if a global index is local
inline bool isLocal(const World& world, idx_t global_idx) {
    return global_idx >= world.start_idx && global_idx < world.start_idx + world.elements_static.size();
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations with halo exchanges between ranks
void runSimulation(World& world, const int n_iters) {
    const int rank = world.rank;
    const int size = world.size;
    const int ncols = world.ncols;
    const int local_rows = world.local_rows;
    const idx_t local_n = static_cast<idx_t>(local_rows) * ncols;

    // Buffers for exchanging boundary row energies with neighbor ranks
    std::vector<val_t> send_up, recv_up, send_down, recv_down;
    if (local_rows > 0) {
        send_up.resize(ncols);
        recv_up.resize(ncols);
        send_down.resize(ncols);
        recv_down.resize(ncols);
    }

    for (int iter = 0; iter < n_iters; ++iter) {
        // Pack boundary rows
        if (local_rows > 0) {
            // first local row -> send_up
            for (int y = 0; y < ncols; ++y) {
                const idx_t local_idx = 0 * ncols + y;
                send_up[y] = world.elements_dynamic[local_idx].current_energy;
            }
            // last local row -> send_down
            for (int y = 0; y < ncols; ++y) {
                const idx_t local_idx = (local_rows - 1) * ncols + y;
                send_down[y] = world.elements_dynamic[local_idx].current_energy;
            }
        }

        // Exchange with up neighbor (rank-1)
        if (rank > 0 && local_rows > 0) {
            MPI_Sendrecv(send_up.data(), ncols, MPI_DOUBLE, rank - 1, 0,
                         recv_up.data(), ncols, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Exchange with down neighbor (rank+1)
        if (rank < size - 1 && local_rows > 0) {
            MPI_Sendrecv(send_down.data(), ncols, MPI_DOUBLE, rank + 1, 0,
                         recv_down.data(), ncols, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Update local elements
        for (idx_t local_i = 0; local_i < local_n; ++local_i) {
            const ElementStatic& elem_static = world.elements_static[local_i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[local_i];
            const Material& mat = world.materials[elem_static.material_idx];

            val_t total_flux = mat.external_flow;

            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_global = elem_static.connected_idx[j];
                val_t neighbor_energy = 0.0;

                if (isLocal(world, neighbor_global)) {
                    const idx_t neighbor_local = neighbor_global - world.start_idx;
                    neighbor_energy = world.elements_dynamic[neighbor_local].current_energy;
                } else {
                    // determine whether neighbor is in the row above or below
                    const int neighbor_row = static_cast<int>(neighbor_global / ncols);
                    if (neighbor_row == world.start_row - 1) {
                        // from recv_up
                        const int col = static_cast<int>(neighbor_global % ncols);
                        neighbor_energy = recv_up[col];
                    } else if (neighbor_row == world.start_row + local_rows) {
                        // from recv_down
                        const int col = static_cast<int>(neighbor_global % ncols);
                        neighbor_energy = recv_down[col];
                    } else {
                        // Should not happen in row-partitioning; fallback to zero
                        neighbor_energy = 0.0;
                    }
                }

                // construct a temporary ElementDynamic-like view
                ElementDynamic neighbor_view{neighbor_energy, 0.0};
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_view);
            }

            ElementDynamic& elem_write = world.elements_dynamic_swap[local_i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }

        // Swap local buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results across ranks (printed on rank 0)
bool validateResultsDistributed(const World& world) {
    // local reductions
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : world.elements_dynamic) {
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(elem.current_energy, local_energy_max);
        local_energy_min = std::min(elem.current_energy, local_energy_min);
    }

    val_t global_energy_sum = 0.0;
    val_t global_flux_sum = 0.0;
    val_t global_energy_max = 0.0;
    val_t global_energy_min = 0.0;

    MPI_Reduce(&local_energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    int local_finite = (std::isfinite(local_energy_sum) && std::isfinite(local_flux_sum) &&
                        std::isfinite(local_energy_max) && std::isfinite(local_energy_min)) ? 1 : 0;
    int global_finite = 0;
    MPI_Reduce(&local_finite, &global_finite, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (world.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_energy_sum);
        printf("  Flux sum: %.2f\n", global_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);

        constexpr val_t energy_epsilon = 1e-8;
        if (!global_finite) {
            printf("  ERROR: Non-finite values detected in global reduction\n");
            return false;
        }
        if (std::abs(global_energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        printf("  Validation: PASSED\n");
    }
    return global_finite != 0;
}

// Compute a distributed hash (XOR of per-element contributions) and return final hash on rank 0
uint64_t computeHashDistributed(const World& world) {
    uint64_t local_hash = 0;
    const idx_t local_n = world.elements_dynamic.size();
    for (idx_t i = 0; i < local_n; ++i) {
        uint64_t e_bits = *reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
        uint64_t f_bits = *reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
        uint64_t global_i = static_cast<uint64_t>(world.start_idx + i);
        local_hash ^= (e_bits + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (f_bits + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
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
    // Initialize MPI unconditionally
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (do this on all ranks)
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

    const int total_elems = n_elems_root * n_elems_root;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, total_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("\n");
    }

    // Build the unstructured mesh locally
    World world;
    buildSquare2D(world, n_elems_root, rank, size);

    // Calculate memory usage per rank and global
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem_local = static_mem + dynamic_mem;
    size_t total_mem_global = 0;
    MPI_Reduce(&total_mem_local, &total_mem_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Memory usage (total across ranks): %.2f MB\n", total_mem_global / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // Barrier before timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Compute wall-clock duration (max across ranks)
    long local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time (max across ranks): %ld ms\n", max_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(max_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(total_elems)) / (max_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute distributed hash and print on rank 0
    uint64_t global_hash = computeHashDistributed(world);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", global_hash);
        printf("\n");
    }

    // Print results for external validation (gather to rank 0)
    if (printResults) {
        // Prepare local energy array
        const idx_t local_n = world.elements_dynamic.size();
        std::vector<double> local_energy(local_n);
        for (idx_t i = 0; i < local_n; ++i) local_energy[i] = world.elements_dynamic[i].current_energy;

        // compute counts/displacements (all ranks can compute deterministically)
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            int base = n_elems_root / size;
            int rem = n_elems_root % size;
            int rows = base + (r < rem ? 1 : 0);
            counts[r] = rows * n_elems_root;
            displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
        }

        std::vector<double> gathered;
        if (rank == 0) gathered.resize(total_elems);

        MPI_Gatherv(local_energy.data(), static_cast<int>(local_n), MPI_DOUBLE,
                    gathered.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(gathered, "ElementEnergy");
        }
    }

    // Validation across ranks
    if (validate) {
        bool ok = validateResultsDistributed(world);
        if (!ok) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
