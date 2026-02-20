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
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh (distributed by rows).
// This represents computation on arbitrarily-shaped geometries.
void buildSquare2D(World& world, const int n_elems_root, const int start_row, const int local_rows) {
    const int n_local_elems = local_rows * n_elems_root;

    // Initialize materials (replicated)
    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate local elements
    world.elements_static.resize(n_local_elems);
    world.elements_dynamic.resize(n_local_elems);
    world.elements_dynamic_swap.resize(n_local_elems);

    // Initialize all local elements with default material and zero energy
    for (int i = 0; i < n_local_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity for local rows: each element connects to its neighbors in 2D grid
    for (int gx = start_row; gx < start_row + local_rows; ++gx) {
        const int lx = gx - start_row;
        for (int y = 0; y < n_elems_root; ++y) {
            const int lidx = lx * n_elems_root + y;
            ElementStatic& elem = world.elements_static[lidx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = gx + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = static_cast<idx_t>(neighbor_idx);
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }

            // Set corner elements as inflow/outflow to create interesting dynamics
            const int last = n_elems_root - 1;
            if ((gx == 0 && y == 0) || (gx == last && y == last)) {
                elem.material_idx = INFLOW_MAT_ID;
            } else if ((gx == 0 && y == last) || (gx == last && y == 0)) {
                elem.material_idx = OUTFLOW_MAT_ID;
            }
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const val_t this_energy,
                         const val_t connection_flux, const val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations (MPI row decomposition with halo exchange).
void runSimulation(World& world, const int n_iters, const int n_elems_root, const int start_row,
                   const int local_rows, const int rank, MPI_Datatype elem_dyn_type, MPI_Comm comm) {
    if (local_rows <= 0) {
        return;
    }

    const size_t start_idx = static_cast<size_t>(start_row) * static_cast<size_t>(n_elems_root);
    const size_t local_elems = static_cast<size_t>(local_rows) * static_cast<size_t>(n_elems_root);
    const size_t end_idx = start_idx + local_elems;

    const int up_rank = (start_row == 0) ? MPI_PROC_NULL : rank - 1;
    const int down_rank = (start_row + local_rows == n_elems_root) ? MPI_PROC_NULL : rank + 1;

    std::vector<ElementDynamic> ghost_top;
    std::vector<ElementDynamic> ghost_bottom;
    if (up_rank != MPI_PROC_NULL) {
        ghost_top.resize(static_cast<size_t>(n_elems_root));
    }
    if (down_rank != MPI_PROC_NULL) {
        ghost_bottom.resize(static_cast<size_t>(n_elems_root));
    }

    auto compute_rows = [&](const int row_begin, const int row_end) {
        for (int lx = row_begin; lx < row_end; ++lx) {
            const size_t row_offset = static_cast<size_t>(lx) * static_cast<size_t>(n_elems_root);
            for (int y = 0; y < n_elems_root; ++y) {
                const size_t i = row_offset + static_cast<size_t>(y);
                const ElementStatic& elem_static = world.elements_static[i];
                const ElementDynamic& elem_dyn = world.elements_dynamic[i];
                const Material& mat = world.materials[elem_static.material_idx];

                // Start with external flow
                val_t total_flux = mat.external_flow;

                // Add flux from all connected elements
                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const size_t neighbor_idx = static_cast<size_t>(elem_static.connected_idx[j]);
                    val_t neighbor_energy;
                    if (neighbor_idx >= start_idx && neighbor_idx < end_idx) {
                        neighbor_energy = world.elements_dynamic[neighbor_idx - start_idx].current_energy;
                    } else if (neighbor_idx < start_idx) {
                        // Neighbor is in the row above our local domain
                        const size_t yy = neighbor_idx + static_cast<size_t>(n_elems_root) - start_idx;
                        neighbor_energy = ghost_top[yy].current_energy;
                    } else {
                        // Neighbor is in the row below our local domain
                        const size_t yy = neighbor_idx - end_idx;
                        neighbor_energy = ghost_bottom[yy].current_energy;
                    }
                    total_flux += computeFlux(mat, elem_dyn.current_energy, elem_static.connected_flux[j], neighbor_energy);
                }

                // Update element state
                ElementDynamic& elem_write = world.elements_dynamic_swap[i];
                elem_write.current_energy = elem_dyn.current_energy + total_flux;
                elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }
        }
    };

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request reqs[4];
        int nreq = 0;

        if (up_rank != MPI_PROC_NULL) {
            MPI_Irecv(ghost_top.data(), n_elems_root, elem_dyn_type, up_rank, 1, comm, &reqs[nreq++]);
            MPI_Isend(world.elements_dynamic.data(), n_elems_root, elem_dyn_type, up_rank, 0, comm, &reqs[nreq++]);
        }
        if (down_rank != MPI_PROC_NULL) {
            MPI_Irecv(ghost_bottom.data(), n_elems_root, elem_dyn_type, down_rank, 0, comm, &reqs[nreq++]);
            MPI_Isend(world.elements_dynamic.data() + static_cast<size_t>(local_rows - 1) * static_cast<size_t>(n_elems_root),
                      n_elems_root, elem_dyn_type, down_rank, 1, comm, &reqs[nreq++]);
        }

        if (local_rows == 1) {
            if (nreq) {
                MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
            }
            compute_rows(0, 1);
        } else {
            const int interior_begin = (up_rank == MPI_PROC_NULL) ? 0 : 1;
            const int interior_end = (down_rank == MPI_PROC_NULL) ? local_rows : (local_rows - 1);
            if (interior_begin < interior_end) {
                compute_rows(interior_begin, interior_end);
            }

            if (nreq) {
                MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
            }

            if (up_rank != MPI_PROC_NULL) {
                compute_rows(0, 1);
            }
            if (down_rank != MPI_PROC_NULL) {
                compute_rows(local_rows - 1, local_rows);
            }
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
    int validate = 0;
    int printResults = 0;

    int should_exit = 0;
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
                should_exit = 1;
                exit_code = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                should_exit = 1;
                exit_code = 1;
                break;
            }
        }
    }

    MPI_Bcast(&should_exit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (should_exit) {
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

    // Row decomposition across ranks (contiguous blocks)
    const int base_rows = n_elems_root / size;
    const int rem_rows = n_elems_root % size;
    const int local_rows = base_rows + (rank < rem_rows ? 1 : 0);
    const int start_row = rank * base_rows + std::min(rank, rem_rows);

    World world;
    buildSquare2D(world, n_elems_root, start_row, local_rows);

    // Calculate (distributed) memory usage and report global totals
    const unsigned long long static_mem_local =
        static_cast<unsigned long long>(world.elements_static.size() * sizeof(ElementStatic));
    const unsigned long long dynamic_mem_local =
        static_cast<unsigned long long>(world.elements_dynamic.size() * sizeof(ElementDynamic) * 2);

    unsigned long long static_mem = 0;
    unsigned long long dynamic_mem = 0;
    MPI_Reduce(&static_mem_local, &static_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&dynamic_mem_local, &dynamic_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        // Run simulation
        printf("Running simulation...\n");
    }

    MPI_Datatype elem_dyn_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &elem_dyn_type);
    MPI_Type_commit(&elem_dyn_type);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    runSimulation(world, n_iters, n_elems_root, start_row, local_rows, rank, elem_dyn_type, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double local_time = t1 - t0;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather final results to rank 0 for deterministic hashing/validation/output
    std::vector<ElementDynamic> global_dyn;
    std::vector<int> recvcounts;
    std::vector<int> displs;

    if (rank == 0) {
        global_dyn.resize(static_cast<size_t>(n_elems));
        recvcounts.resize(static_cast<size_t>(size));
        displs.resize(static_cast<size_t>(size));

        for (int r = 0; r < size; ++r) {
            const int r_rows = base_rows + (r < rem_rows ? 1 : 0);
            const int r_start = r * base_rows + std::min(r, rem_rows);
            recvcounts[static_cast<size_t>(r)] = r_rows * n_elems_root;
            displs[static_cast<size_t>(r)] = r_start * n_elems_root;
        }
    }

    MPI_Gatherv(world.elements_dynamic.data(), static_cast<int>(world.elements_dynamic.size()), elem_dyn_type,
                rank == 0 ? global_dyn.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                elem_dyn_type, 0, MPI_COMM_WORLD);

    MPI_Type_free(&elem_dyn_type);

    int retcode = 0;

    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_time * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Compute hash for verification (global, deterministic order)
        const uint64_t hash = computeHash(global_dyn);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global_dyn.size());
            for (const auto& elem : global_dyn) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            World global_world;
            global_world.elements_dynamic = std::move(global_dyn);
            const bool valid = validateResults(global_world);
            if (!valid) {
                retcode = 1;
            }
        }
    }

    MPI_Bcast(&retcode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return retcode;
}
