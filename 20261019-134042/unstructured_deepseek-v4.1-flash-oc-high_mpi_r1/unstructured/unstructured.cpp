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
    std::vector<val_t> energy_current;
    std::vector<val_t> flux_current;
    std::vector<val_t> energy_swap;
    std::vector<val_t> flux_swap;
    int n_root = 0;
    int x_start = 0;
    int x_end = 0;
    int local_rows = 0;
    int rank = 0;
    int size = 1;
    int global_elems = 0;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

inline void computeRowRange(const int rank, const int size, const int n_root,
                            int& start, int& end) {
    if (n_root <= 0) {
        start = 0;
        end = 0;
        return;
    }

    if (size <= n_root) {
        start = static_cast<int>((static_cast<long long>(rank) * n_root) / size);
        end = static_cast<int>((static_cast<long long>(rank + 1) * n_root) / size);
    } else {
        start = std::min(rank, n_root);
        end = std::min(rank + 1, n_root);
    }
}

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root,
                   const int rank, const int size) {
    world.n_root = n_elems_root;
    world.rank = rank;
    world.size = size;
    computeRowRange(rank, size, n_elems_root, world.x_start, world.x_end);
    world.local_rows = world.x_end - world.x_start;
    world.global_elems = n_elems_root * n_elems_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    const int n = n_elems_root;
    const size_t local_elems = static_cast<size_t>(world.local_rows) * n;
    const size_t ext_elems = static_cast<size_t>(world.local_rows + 2) * n;

    // Allocate elements
    world.elements_static.resize(local_elems);
    world.energy_current.assign(ext_elems, 0.0);
    world.flux_current.assign(ext_elems, 0.0);
    world.energy_swap.assign(ext_elems, 0.0);
    world.flux_swap.assign(ext_elems, 0.0);

    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    const int last = n - 1;

    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int r = 0; r < world.local_rows; ++r) {
        const int x = world.x_start + r;
        for (int y = 0; y < n; ++y) {
            const size_t local_idx = static_cast<size_t>(r) * n + y;
            ElementStatic& elem = world.elements_static[local_idx];

            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            // Connect to neighbors (up, down, left, right)
            for (int c = 0; c < 4; ++c) {
                const int nx = x + offsets[c][0];
                const int ny = y + offsets[c][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    size_t neighbor_idx;
                    if (nx >= world.x_start && nx < world.x_end) {
                        neighbor_idx = static_cast<size_t>(nx - world.x_start + 1) * n + ny;
                    } else if (nx < world.x_start) {
                        neighbor_idx = static_cast<size_t>(ny);
                    } else {
                        neighbor_idx = static_cast<size_t>(world.local_rows + 1) * n + ny;
                    }

                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }

            // Set corner elements as inflow/outflow to create interesting dynamics
            if (x == 0 && y == 0) {
                elem.material_idx = INFLOW_MAT_ID;
            } else if (x == 0 && y == last) {
                elem.material_idx = OUTFLOW_MAT_ID;
            } else if (x == last && y == 0) {
                elem.material_idx = OUTFLOW_MAT_ID;
            } else if (x == last && y == last) {
                elem.material_idx = INFLOW_MAT_ID;
            }
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, val_t this_energy,
                         val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const int n = world.n_root;
    const int rows = world.local_rows;

    if (n <= 0 || rows <= 0) {
        return;
    }

    const bool has_prev = world.x_start > 0;
    const bool has_next = world.x_end < world.n_root;
    const int prev = world.rank - 1;
    const int next = world.rank + 1;

    constexpr int TAG_UP = 10;
    constexpr int TAG_DOWN = 11;

    for (int iter = 0; iter < n_iters; ++iter) {
        val_t* cur_e = world.energy_current.data();
        val_t* cur_f = world.flux_current.data();
        val_t* new_e = world.energy_swap.data();
        val_t* new_f = world.flux_swap.data();

        if (has_prev) {
            MPI_Sendrecv(cur_e + n, n, MPI_DOUBLE, prev, TAG_UP,
                         cur_e, n, MPI_DOUBLE, prev, TAG_DOWN,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (has_next) {
            MPI_Sendrecv(cur_e + static_cast<size_t>(rows) * n, n,
                         MPI_DOUBLE, next, TAG_DOWN,
                         cur_e + static_cast<size_t>(rows + 1) * n, n,
                         MPI_DOUBLE, next, TAG_UP,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Update all elements
        for (int r = 0; r < rows; ++r) {
            const ElementStatic* elem_static_row =
                world.elements_static.data() + static_cast<size_t>(r) * n;
            const size_t ext_base = static_cast<size_t>(r + 1) * n;

            for (int y = 0; y < n; ++y) {
                const ElementStatic& elem_static = elem_static_row[y];
                const Material& mat = world.materials[elem_static.material_idx];
                const size_t idx = ext_base + y;
                const val_t this_energy = cur_e[idx];

                // Start with external flow
                val_t total_flux = mat.external_flow;

                // Add flux from all connected elements
                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const size_t neighbor_idx = elem_static.connected_idx[j];
                    total_flux += computeFlux(mat, this_energy,
                                              elem_static.connected_flux[j],
                                              cur_e[neighbor_idx]);
                }

                // Update element state
                new_e[idx] = this_energy + total_flux;
                new_f[idx] = cur_f[idx] + std::abs(total_flux);
            }
        }

        // Swap buffers
        std::swap(world.energy_current, world.energy_swap);
        std::swap(world.flux_current, world.flux_swap);
    }
}

// Validate simulation results
bool validateResults(const std::vector<val_t>& energy,
                     const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
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
uint64_t computeLocalHash(const World& world) {
    uint64_t hash = 0;
    const size_t local_elems =
        static_cast<size_t>(world.local_rows) * world.n_root;
    const idx_t global_base =
        static_cast<idx_t>(world.x_start) * world.n_root;
    const size_t ext_offset = static_cast<size_t>(world.n_root);

    for (size_t l = 0; l < local_elems; ++l) {
        const idx_t i = global_base + l;
        const uint64_t* e_ptr =
            reinterpret_cast<const uint64_t*>(&world.energy_current[ext_offset + l]);
        const uint64_t* f_ptr =
            reinterpret_cast<const uint64_t*>(&world.flux_current[ext_offset + l]);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

void gatherResults(const World& world,
                   std::vector<val_t>& energy,
                   std::vector<val_t>& flux) {
    const int n = world.n_root;
    const int local_count = world.local_rows * n;

    std::vector<int> counts(world.size, 0);
    std::vector<int> displs(world.size, 0);
    for (int r = 0; r < world.size; ++r) {
        int start, end;
        computeRowRange(r, world.size, n, start, end);
        counts[r] = (end - start) * n;
        displs[r] = start * n;
    }

    if (world.rank == 0) {
        energy.resize(world.global_elems);
        flux.resize(world.global_elems);
    }

    const val_t* send_e =
        local_count > 0 ? world.energy_current.data() + n : nullptr;
    const val_t* send_f =
        local_count > 0 ? world.flux_current.data() + n : nullptr;

    MPI_Gatherv(send_e, local_count, MPI_DOUBLE,
                world.rank == 0 ? energy.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(send_f, local_count, MPI_DOUBLE,
                world.rank == 0 ? flux.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
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
    int exit_code = 0;

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
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, rank, size);

    if (rank == 0) {
        // Calculate memory usage
        const size_t static_mem = static_cast<size_t>(world.global_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(world.global_elems) * sizeof(ElementDynamic) * 2;
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

    const uint64_t local_hash = computeLocalHash(world);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    std::vector<val_t> energy_full;
    std::vector<val_t> flux_full;
    if (printResults || validate) {
        gatherResults(world, energy_full, flux_full);
    }

    if (rank == 0) {
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

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
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            print_results(energy_full, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(energy_full, flux_full);
            if (!valid) {
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
