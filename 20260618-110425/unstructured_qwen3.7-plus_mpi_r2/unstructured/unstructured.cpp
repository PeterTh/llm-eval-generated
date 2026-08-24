#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
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

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse identically)
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
            printUsage(argv[0]);
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

    // ---- Domain decomposition (1D block along rows) ----
    int row_start, local_nrows;
    {
        int base = n_elems_root / nprocs;
        int rem = n_elems_root % nprocs;
        if (rank < rem) {
            local_nrows = base + 1;
            row_start = rank * (base + 1);
        } else {
            local_nrows = base;
            row_start = rem * (base + 1) + (rank - rem) * base;
        }
    }

    // Ghost layer flags
    int n_ghost_top = (row_start > 0 && local_nrows > 0) ? 1 : 0;
    int n_ghost_bottom = (row_start + local_nrows < n_elems_root && local_nrows > 0) ? 1 : 0;
    int local_total_rows = local_nrows + n_ghost_top + n_ghost_bottom;
    int local_n_elems = local_nrows * n_elems_root;
    int local_total_elems = local_total_rows * n_elems_root;

    // Find neighbor ranks for ghost exchange (handle ranks with 0 rows)
    int neighbor_up = MPI_PROC_NULL, neighbor_down = MPI_PROC_NULL;
    if (local_nrows > 0) {
        for (int r = rank - 1; r >= 0; --r) {
            int b = n_elems_root / nprocs, rm = n_elems_root % nprocs;
            int ln = (r < rm) ? b + 1 : b;
            if (ln > 0) { neighbor_up = r; break; }
        }
        for (int r = rank + 1; r < nprocs; ++r) {
            int b = n_elems_root / nprocs, rm = n_elems_root % nprocs;
            int ln = (r < rm) ? b + 1 : b;
            if (ln > 0) { neighbor_down = r; break; }
        }
    }

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI)\n");
        printf("============================================\n");
        printf("MPI ranks: %d\n", nprocs);
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // ---- Build local mesh ----
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;

    if (local_nrows > 0) {
        // Initialize materials
        materials.emplace_back(Material{0.8, 0.0});    // Default material
        materials.emplace_back(Material{0.8, 0.5});    // Inflow material
        materials.emplace_back(Material{0.8, -0.5});   // Outflow material

        // Allocate: static for owned only, dynamic for owned + ghost
        elements_static.resize(local_n_elems);
        elements_dynamic.resize(local_total_elems, ElementDynamic{0.0, 0.0});
        elements_dynamic_swap.resize(local_total_elems, ElementDynamic{0.0, 0.0});

        // Build connectivity for owned elements
        // Local row lr (0..local_nrows-1) maps to global row (row_start + lr)
        // In elements_dynamic, owned row lr is at local row (lr + n_ghost_top)
        for (int lr = 0; lr < local_nrows; ++lr) {
            const int gr = row_start + lr;  // global row
            for (int c = 0; c < n_elems_root; ++c) {
                ElementStatic& elem = elements_static[lr * n_elems_root + c];

                // Material assignment (corners)
                elem.material_idx = DEFAULT_MAT_ID;
                const int last = n_elems_root - 1;
                if (gr == 0 && c == 0) elem.material_idx = INFLOW_MAT_ID;
                else if (gr == 0 && c == last) elem.material_idx = OUTFLOW_MAT_ID;
                else if (gr == last && c == 0) elem.material_idx = OUTFLOW_MAT_ID;
                else if (gr == last && c == last) elem.material_idx = INFLOW_MAT_ID;

                elem.num_connections = 0;

                // Neighbors in same order as original: (x+1,y), (x-1,y), (x,y+1), (x,y-1)
                // i.e., down, up, right, left

                // Down neighbor (gr+1, c)
                if (gr + 1 < n_elems_root) {
                    idx_t ni = static_cast<idx_t>(lr + n_ghost_top + 1) * n_elems_root + c;
                    elem.connected_idx[elem.num_connections] = ni;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }

                // Up neighbor (gr-1, c)
                if (gr > 0) {
                    idx_t ni = static_cast<idx_t>(lr + n_ghost_top - 1) * n_elems_root + c;
                    elem.connected_idx[elem.num_connections] = ni;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }

                // Right neighbor (gr, c+1)
                if (c + 1 < n_elems_root) {
                    idx_t ni = static_cast<idx_t>(lr + n_ghost_top) * n_elems_root + (c + 1);
                    elem.connected_idx[elem.num_connections] = ni;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }

                // Left neighbor (gr, c-1)
                if (c > 0) {
                    idx_t ni = static_cast<idx_t>(lr + n_ghost_top) * n_elems_root + (c - 1);
                    elem.connected_idx[elem.num_connections] = ni;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    if (rank == 0) {
        const size_t static_mem = n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // ---- Run simulation with halo exchange ----
    if (rank == 0) {
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < n_iters; ++iter) {
        // Halo exchange: update ghost rows of elements_dynamic
        if (local_nrows > 0) {
            const int top_owned_row = n_ghost_top;
            const int bottom_owned_row = n_ghost_top + local_nrows - 1;
            const int top_ghost_row = 0;
            const int bottom_ghost_row = n_ghost_top + local_nrows;
            const int row_bytes = n_elems_root * static_cast<int>(sizeof(ElementDynamic));

            // Send bottom owned row down, receive top ghost row from above
            MPI_Sendrecv(
                &elements_dynamic[bottom_owned_row * n_elems_root], row_bytes, MPI_BYTE,
                neighbor_down, 0,
                &elements_dynamic[top_ghost_row * n_elems_root], row_bytes, MPI_BYTE,
                neighbor_up, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Send top owned row up, receive bottom ghost row from below
            MPI_Sendrecv(
                &elements_dynamic[top_owned_row * n_elems_root], row_bytes, MPI_BYTE,
                neighbor_up, 1,
                &elements_dynamic[bottom_ghost_row * n_elems_root], row_bytes, MPI_BYTE,
                neighbor_down, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Compute fluxes for owned elements
        for (int i = 0; i < local_n_elems; ++i) {
            const ElementStatic& elem_static = elements_static[i];
            const int local_idx = i + n_ghost_top * n_elems_root;
            const ElementDynamic& elem_dyn = elements_dynamic[local_idx];
            const Material& mat = materials[elem_static.material_idx];

            val_t total_flux = mat.external_flow;

            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic& neighbor_dyn = elements_dynamic[neighbor_idx];
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }

            ElementDynamic& elem_write = elements_dynamic_swap[local_idx];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }

        std::swap(elements_dynamic, elements_dynamic_swap);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long max_duration_ms = 0;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---- Gather results to rank 0 ----
    // Compute send counts and displacements (in bytes) for Gatherv
    std::vector<int> gather_counts(nprocs), gather_displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        int b = n_elems_root / nprocs, rm = n_elems_root % nprocs;
        int ln = (r < rm) ? b + 1 : b;
        gather_counts[r] = ln * n_elems_root * static_cast<int>(sizeof(ElementDynamic));
    }
    gather_displs[0] = 0;
    for (int r = 1; r < nprocs; ++r) {
        gather_displs[r] = gather_displs[r - 1] + gather_counts[r - 1];
    }

    std::vector<ElementDynamic> global_elements;
    if (rank == 0) {
        global_elements.resize(n_elems);
    }

    const ElementDynamic* send_ptr = (local_nrows > 0)
        ? &elements_dynamic[n_ghost_top * n_elems_root]
        : nullptr;

    MPI_Gatherv(
        send_ptr, local_n_elems * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE,
        rank == 0 ? global_elements.data() : nullptr,
        gather_counts.data(), gather_displs.data(), MPI_BYTE,
        0, MPI_COMM_WORLD);

    // ---- Post-processing on rank 0 ----
    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(max_duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (double)n_elems) / (max_duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Compute hash for verification
        uint64_t hash_val = 0;
        for (int i = 0; i < n_elems; ++i) {
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&global_elements[i].current_energy);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&global_elements[i].total_flux);
            hash_val ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
            hash_val ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
        }
        printf("  Result hash: %016lX\n", hash_val);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(n_elems);
            for (int i = 0; i < n_elems; ++i) {
                energyData.push_back(global_elements[i].current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            val_t energy_sum = 0.0;
            val_t flux_sum = 0.0;
            val_t energy_max = std::numeric_limits<val_t>::lowest();
            val_t energy_min = std::numeric_limits<val_t>::max();

            for (int i = 0; i < n_elems; ++i) {
                energy_sum += global_elements[i].current_energy;
                flux_sum += global_elements[i].total_flux;
                energy_max = std::max(global_elements[i].current_energy, energy_max);
                energy_min = std::min(global_elements[i].current_energy, energy_min);
            }

            printf("Validation results:\n");
            printf("  Energy sum: %.12f\n", energy_sum);
            printf("  Flux sum: %.2f\n", flux_sum);
            printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

            constexpr val_t energy_epsilon = 1e-8;

            bool valid = true;
            if (!std::isfinite(energy_sum)) {
                printf("  ERROR: Energy sum is not finite\n");
                valid = false;
            }
            if (std::abs(energy_sum) > energy_epsilon) {
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            }
            if (!std::isfinite(flux_sum)) {
                printf("  ERROR: Flux sum is not finite\n");
                valid = false;
            }
            if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
                printf("  ERROR: Energy extrema are not finite\n");
                valid = false;
            }

            if (valid) {
                printf("  Validation: PASSED\n");
            }

            if (!valid) {
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
