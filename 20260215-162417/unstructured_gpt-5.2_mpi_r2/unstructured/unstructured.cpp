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
    int n_elems_root = 0;
    int start_row = 0;   // global row index of first owned row
    int local_rows = 0;  // number of owned rows

    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh (distributed by contiguous rows)
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int start_row, const int local_rows) {
    world.n_elems_root = n_elems_root;
    world.start_row = start_row;
    world.local_rows = local_rows;

    // Initialize materials (replicated)
    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});   // Default material
    world.materials.emplace_back(Material{0.8, 0.5});   // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});  // Outflow material

    const size_t n_local = static_cast<size_t>(local_rows) * static_cast<size_t>(n_elems_root);

    // Allocate local elements
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    // Initialize all local elements with default material and zero energy
    for (size_t i = 0; i < n_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build local connectivity: each element connects to its neighbors in 2D grid
    for (int lx = 0; lx < local_rows; ++lx) {
        const int gx = start_row + lx;
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t lidx = static_cast<size_t>(lx) * static_cast<size_t>(n_elems_root) + static_cast<size_t>(y);
            ElementStatic& elem = world.elements_static[lidx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; ++n) {
                const int nx = gx + offsets[n][0];
                const int ny = y + offsets[n][1];
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const idx_t neighbor_idx = static_cast<idx_t>(nx) * static_cast<idx_t>(n_elems_root) + static_cast<idx_t>(ny);
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }

            // Set corner elements as inflow/outflow to create interesting dynamics
            const int last = n_elems_root - 1;
            if (gx == 0 && y == 0) elem.material_idx = INFLOW_MAT_ID;
            if (gx == 0 && y == last) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == last && y == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == last && y == last) elem.material_idx = INFLOW_MAT_ID;
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

static inline void computeRowDecomp(const int n_rows, const int rank, const int size, int& start_row, int& local_rows) {
    const int base = n_rows / size;
    const int rem = n_rows % size;
    local_rows = base + (rank < rem ? 1 : 0);
    start_row = rank * base + (rank < rem ? rank : rem);
}

// Run simulation for n_iters iterations (MPI distributed memory)
void runSimulation(World& world, const int n_iters, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    const int n = world.n_elems_root;
    const int start_row = world.start_row;
    const int local_rows = world.local_rows;

    const size_t n_local = world.elements_static.size();
    if (n_local == 0) {
        return;
    }

    // Halo buffers for neighbor energies (one row each)
    std::vector<val_t> halo_top(static_cast<size_t>(n), 0.0);
    std::vector<val_t> halo_bottom(static_cast<size_t>(n), 0.0);
    std::vector<val_t> send_top(static_cast<size_t>(n), 0.0);
    std::vector<val_t> send_bottom(static_cast<size_t>(n), 0.0);

    const int prev_rank = (start_row == 0) ? MPI_PROC_NULL : (rank - 1);
    const int next_rank = (start_row + local_rows == n) ? MPI_PROC_NULL : (rank + 1);

    for (int iter = 0; iter < n_iters; ++iter) {
        // Pack boundary rows
        for (int y = 0; y < n; ++y) {
            send_top[static_cast<size_t>(y)] = world.elements_dynamic[static_cast<size_t>(y)].current_energy;
            send_bottom[static_cast<size_t>(y)] =
                world.elements_dynamic[static_cast<size_t>(local_rows - 1) * static_cast<size_t>(n) + static_cast<size_t>(y)].current_energy;
        }

        // Exchange halos (energies only)
        MPI_Sendrecv(send_top.data(), n, MPI_DOUBLE, prev_rank, 100,
                     halo_bottom.data(), n, MPI_DOUBLE, next_rank, 100,
                     comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_bottom.data(), n, MPI_DOUBLE, next_rank, 200,
                     halo_top.data(), n, MPI_DOUBLE, prev_rank, 200,
                     comm, MPI_STATUS_IGNORE);

        // Update all local elements
        for (size_t i = 0; i < n_local; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];

            const val_t this_energy = elem_dyn.current_energy;

            // Start with external flow
            val_t total_flux = mat.external_flow;

            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const int nx = static_cast<int>(neighbor_idx / static_cast<idx_t>(n));
                const int ny = static_cast<int>(neighbor_idx - static_cast<idx_t>(nx) * static_cast<idx_t>(n));

                val_t neighbor_energy = 0.0;
                if (nx < start_row) {
                    neighbor_energy = halo_top[static_cast<size_t>(ny)];
                } else if (nx >= start_row + local_rows) {
                    neighbor_energy = halo_bottom[static_cast<size_t>(ny)];
                } else {
                    const size_t lnx = static_cast<size_t>(nx - start_row);
                    neighbor_energy = world.elements_dynamic[lnx * static_cast<size_t>(n) + static_cast<size_t>(ny)].current_energy;
                }

                total_flux += (neighbor_energy - this_energy) * mat.transfer_coeff * elem_static.connected_flux[j] * 0.25;
            }

            // Update element state
            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = this_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }

        // Swap buffers
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

// Compute a simple hash of the results for verification (local chunk)
uint64_t computeHash(const std::vector<ElementDynamic>& elements, const idx_t global_start_idx) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        const uint64_t gi = static_cast<uint64_t>(global_start_idx + static_cast<idx_t>(i));
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + gi) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + gi) * 0xbf58476d1ce4e5b9ULL;
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
    int validate = 0;
    int printResults = 0;
    int exit_now = 0;
    int exit_code = 0;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                exit_now = 1;
                exit_code = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exit_now = 1;
                exit_code = 1;
                break;
            }
        }
    }

    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exit_now, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (exit_now) {
        MPI_Finalize();
        return exit_code;
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

    int start_row = 0, local_rows = 0;
    computeRowDecomp(n_elems_root, rank, size, start_row, local_rows);

    World world;
    buildSquare2D(world, n_elems_root, start_row, local_rows);

    if (rank == 0) {
        // Calculate memory usage (serial-equivalent problem size)
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
    const double t0 = MPI_Wtime();

    runSimulation(world, n_iters, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double local_ms = (t1 - t0) * 1000.0;
    double duration_ms = 0.0;
    MPI_Reduce(&local_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration_ms));

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute hash for verification (distributed, exact match)
    const idx_t global_start_idx = static_cast<idx_t>(start_row) * static_cast<idx_t>(n_elems_root);
    const uint64_t local_hash = computeHash(world.elements_dynamic, global_start_idx);
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("  Result hash: %016lX\n", global_hash);
        printf("\n");
    }

    // Print results for external validation (gather to rank 0)
    if (printResults) {
        const int local_n = static_cast<int>(world.elements_dynamic.size());
        std::vector<double> local_energy(static_cast<size_t>(local_n));
        for (int i = 0; i < local_n; ++i) {
            local_energy[static_cast<size_t>(i)] = world.elements_dynamic[static_cast<size_t>(i)].current_energy;
        }

        std::vector<int> recv_counts;
        if (rank == 0) recv_counts.resize(static_cast<size_t>(size));
        MPI_Gather(&local_n, 1, MPI_INT, rank == 0 ? recv_counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> displs;
        std::vector<double> energy_all;
        if (rank == 0) {
            displs.resize(static_cast<size_t>(size));
            int off = 0;
            for (int r = 0; r < size; ++r) {
                displs[static_cast<size_t>(r)] = off;
                off += recv_counts[static_cast<size_t>(r)];
            }
            energy_all.resize(static_cast<size_t>(off));
        }

        MPI_Gatherv(local_energy.data(), local_n, MPI_DOUBLE,
                    rank == 0 ? energy_all.data() : nullptr,
                    rank == 0 ? recv_counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(energy_all, "ElementEnergy");
        }
    }

    // Validation (gather to rank 0 to preserve serial-equivalent validation semantics/output)
    if (validate) {
        const int local_n = static_cast<int>(world.elements_dynamic.size());
        std::vector<double> local_energy(static_cast<size_t>(local_n));
        std::vector<double> local_flux(static_cast<size_t>(local_n));
        for (int i = 0; i < local_n; ++i) {
            local_energy[static_cast<size_t>(i)] = world.elements_dynamic[static_cast<size_t>(i)].current_energy;
            local_flux[static_cast<size_t>(i)] = world.elements_dynamic[static_cast<size_t>(i)].total_flux;
        }

        std::vector<int> recv_counts;
        if (rank == 0) recv_counts.resize(static_cast<size_t>(size));
        MPI_Gather(&local_n, 1, MPI_INT, rank == 0 ? recv_counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> displs;
        std::vector<double> energy_all;
        std::vector<double> flux_all;
        if (rank == 0) {
            displs.resize(static_cast<size_t>(size));
            int off = 0;
            for (int r = 0; r < size; ++r) {
                displs[static_cast<size_t>(r)] = off;
                off += recv_counts[static_cast<size_t>(r)];
            }
            energy_all.resize(static_cast<size_t>(off));
            flux_all.resize(static_cast<size_t>(off));
        }

        MPI_Gatherv(local_energy.data(), local_n, MPI_DOUBLE,
                    rank == 0 ? energy_all.data() : nullptr,
                    rank == 0 ? recv_counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        MPI_Gatherv(local_flux.data(), local_n, MPI_DOUBLE,
                    rank == 0 ? flux_all.data() : nullptr,
                    rank == 0 ? recv_counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            World world_all;
            world_all.elements_dynamic.resize(static_cast<size_t>(n_elems));
            for (int i = 0; i < n_elems; ++i) {
                world_all.elements_dynamic[static_cast<size_t>(i)].current_energy = energy_all[static_cast<size_t>(i)];
                world_all.elements_dynamic[static_cast<size_t>(i)].total_flux = flux_all[static_cast<size_t>(i)];
            }

            const bool valid_ok = validateResults(world_all);
            if (!valid_ok) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
