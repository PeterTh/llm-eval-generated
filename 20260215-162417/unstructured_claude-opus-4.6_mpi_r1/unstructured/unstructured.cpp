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

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
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
    
    constexpr val_t energy_epsilon = 1e-8;
    
    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    
    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
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
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
    const int n_cols = n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("MPI ranks: %d\n", nprocs);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Row-based domain decomposition
    const int base_rows = n_elems_root / nprocs;
    const int extra = n_elems_root % nprocs;
    const int my_start_row = (rank < extra) ? rank * (base_rows + 1)
                                            : extra * (base_rows + 1) + (rank - extra) * base_rows;
    const int my_nrows = (rank < extra) ? base_rows + 1 : base_rows;
    const int my_end_row = my_start_row + my_nrows;
    const int my_start_elem = my_start_row * n_cols;
    const int my_nelems = my_nrows * n_cols;

    // Ghost rows for halo exchange
    const int ghost_top_n = (my_nrows > 0 && my_start_row > 0) ? n_cols : 0;
    const int ghost_bot_n = (my_nrows > 0 && my_end_row < n_elems_root) ? n_cols : 0;
    const int total_n = my_nelems + ghost_top_n + ghost_bot_n;

    const int top_neighbor = (ghost_top_n > 0) ? rank - 1 : -1;
    const int bot_neighbor = (ghost_bot_n > 0) ? rank + 1 : -1;

    // Materials (replicated on all ranks)
    std::vector<Material> materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};

    // Build local mesh with connectivity remapped to local indices
    // Layout: [0, my_nelems) = local, [my_nelems, +ghost_top_n) = ghost top,
    //         [my_nelems+ghost_top_n, total_n) = ghost bottom
    if (rank == 0) printf("Building unstructured mesh...\n");

    std::vector<ElementStatic> local_static(my_nelems);
    std::vector<ElementDynamic> dyn_a(total_n, {0.0, 0.0});
    std::vector<ElementDynamic> dyn_b(total_n, {0.0, 0.0});

    const int last = n_elems_root - 1;
    for (int x = my_start_row; x < my_end_row; ++x) {
        for (int y = 0; y < n_cols; ++y) {
            const int li = (x - my_start_row) * n_cols + y;
            auto& es = local_static[li];

            es.material_idx = DEFAULT_MAT_ID;
            if (x == 0 && y == 0) es.material_idx = INFLOW_MAT_ID;
            else if (x == 0 && y == last) es.material_idx = OUTFLOW_MAT_ID;
            else if (x == last && y == 0) es.material_idx = OUTFLOW_MAT_ID;
            else if (x == last && y == last) es.material_idx = INFLOW_MAT_ID;

            es.num_connections = 0;
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int gn = nx * n_cols + ny;
                    idx_t ln;
                    if (gn >= my_start_elem && gn < my_start_elem + my_nelems) {
                        ln = gn - my_start_elem;
                    } else if (ghost_top_n > 0 && gn >= (my_start_row - 1) * n_cols
                               && gn < my_start_row * n_cols) {
                        ln = my_nelems + (gn - (my_start_row - 1) * n_cols);
                    } else {
                        ln = my_nelems + ghost_top_n + (gn - my_end_row * n_cols);
                    }
                    es.connected_idx[es.num_connections] = ln;
                    es.connected_flux[es.num_connections] = 1.0;
                    es.num_connections++;
                }
            }
        }
    }

    if (rank == 0) {
        const size_t static_mem = (size_t)n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = (size_t)n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Element ranges for communication-computation overlap
    const int interior_start = (ghost_top_n > 0) ? n_cols : 0;
    const int interior_end = (ghost_bot_n > 0) ? my_nelems - n_cols : my_nelems;
    const int safe_interior_start = std::min(interior_start, interior_end);

    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Start non-blocking halo exchange
        MPI_Request reqs[4];
        int nreqs = 0;

        if (top_neighbor >= 0) {
            MPI_Irecv(&dyn_a[my_nelems], ghost_top_n * (int)sizeof(ElementDynamic),
                      MPI_BYTE, top_neighbor, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Isend(&dyn_a[0], n_cols * (int)sizeof(ElementDynamic),
                      MPI_BYTE, top_neighbor, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (bot_neighbor >= 0) {
            MPI_Irecv(&dyn_a[my_nelems + ghost_top_n], ghost_bot_n * (int)sizeof(ElementDynamic),
                      MPI_BYTE, bot_neighbor, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Isend(&dyn_a[(my_nrows - 1) * n_cols], n_cols * (int)sizeof(ElementDynamic),
                      MPI_BYTE, bot_neighbor, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }

        // Compute interior elements (no ghost dependency)
        for (int i = safe_interior_start; i < interior_end; ++i) {
            const auto& es = local_static[i];
            const auto& ed = dyn_a[i];
            const auto& mat = materials[es.material_idx];
            val_t flux = mat.external_flow;
            for (idx_t j = 0; j < es.num_connections; ++j) {
                flux += computeFlux(mat, ed, es.connected_flux[j], dyn_a[es.connected_idx[j]]);
            }
            dyn_b[i].current_energy = ed.current_energy + flux;
            dyn_b[i].total_flux = ed.total_flux + std::abs(flux);
        }

        // Wait for halo data
        if (nreqs > 0) MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

        // Compute boundary elements (depend on ghost data)
        for (int i = 0; i < safe_interior_start; ++i) {
            const auto& es = local_static[i];
            const auto& ed = dyn_a[i];
            const auto& mat = materials[es.material_idx];
            val_t flux = mat.external_flow;
            for (idx_t j = 0; j < es.num_connections; ++j) {
                flux += computeFlux(mat, ed, es.connected_flux[j], dyn_a[es.connected_idx[j]]);
            }
            dyn_b[i].current_energy = ed.current_energy + flux;
            dyn_b[i].total_flux = ed.total_flux + std::abs(flux);
        }
        for (int i = interior_end; i < my_nelems; ++i) {
            const auto& es = local_static[i];
            const auto& ed = dyn_a[i];
            const auto& mat = materials[es.material_idx];
            val_t flux = mat.external_flow;
            for (idx_t j = 0; j < es.num_connections; ++j) {
                flux += computeFlux(mat, ed, es.connected_flux[j], dyn_a[es.connected_idx[j]]);
            }
            dyn_b[i].current_energy = ed.current_energy + flux;
            dyn_b[i].total_flux = ed.total_flux + std::abs(flux);
        }

        std::swap(dyn_a, dyn_b);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Gather results on rank 0
    std::vector<int> recvcounts(nprocs), displs(nprocs);
    {
        int my_bytes = my_nelems * (int)sizeof(ElementDynamic);
        MPI_Allgather(&my_bytes, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        displs[0] = 0;
        for (int i = 1; i < nprocs; ++i) displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    std::vector<ElementDynamic> global_dyn;
    if (rank == 0) global_dyn.resize(n_elems);

    MPI_Gatherv(dyn_a.data(), my_nelems * (int)sizeof(ElementDynamic), MPI_BYTE,
                rank == 0 ? global_dyn.data() : nullptr,
                recvcounts.data(), displs.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);

    int retval = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        const uint64_t hash = computeHash(global_dyn);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(n_elems);
            for (const auto& elem : global_dyn) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            if (!validateResults(global_dyn)) {
                retval = 1;
            }
        }
    }

    MPI_Bcast(&retval, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return retval;
}
