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
// connected_idx: < local_n_elems => owned neighbor index
//                >= local_n_elems => halo index (offset into halo_energy)
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // ---- Argument parsing (rank 0) + broadcast ----
    int n_elems_root = 512;
    int n_iters = 10;
    int validate_int = 0;
    int printResults_int = 0;
    int exit_code = 0;  // 0=continue, 1=help, 2=error

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_int = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_int = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
                printf("  -i <num>     Number of simulation iterations (default: 10)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
                exit_code = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
                printf("  -i <num>     Number of simulation iterations (default: 10)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
                exit_code = 2;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exit_code != 0) {
        MPI_Finalize();
        return exit_code == 2 ? 1 : 0;
    }

    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_int, 1, MPI_INT, 0, MPI_COMM_WORLD);

    bool validate = validate_int != 0;
    bool printResults = printResults_int != 0;
    const int n_elems = n_elems_root * n_elems_root;

    // ---- Domain decomposition (1D along x / rows) ----
    int rows_per_rank = n_elems_root / num_ranks;
    int remainder = n_elems_root % num_ranks;
    int x_start = rank * rows_per_rank + std::min(rank, remainder);
    int x_end = (rank + 1) * rows_per_rank + std::min(rank + 1, remainder);
    int local_n_x = x_end - x_start;
    size_t local_n_elems = (size_t)local_n_x * n_elems_root;

    // Determine MPI neighbors (skip ranks with 0 rows)
    int upper_rank = MPI_PROC_NULL;
    int lower_rank = MPI_PROC_NULL;
    if (rank > 0) {
        int u_xs = (rank - 1) * rows_per_rank + std::min(rank - 1, remainder);
        int u_xe = rank * rows_per_rank + std::min(rank, remainder);
        if (u_xs < u_xe) upper_rank = rank - 1;
    }
    if (rank < num_ranks - 1) {
        int l_xs = (rank + 1) * rows_per_rank + std::min(rank + 1, remainder);
        int l_xe = (rank + 2) * rows_per_rank + std::min(rank + 2, remainder);
        if (l_xs < l_xe) lower_rank = rank + 1;
    }

    // ---- Build local mesh ----
    std::vector<Material> materials;
    materials.reserve(3);
    materials.push_back({0.8, 0.0});  // Default
    materials.push_back({0.8, 0.5});  // Inflow
    materials.push_back({0.8, -0.5}); // Outflow

    std::vector<ElementStatic> elements_static(local_n_elems);
    for (size_t i = 0; i < local_n_elems; ++i) {
        elements_static[i].material_idx = 0;
        elements_static[i].num_connections = 0;
    }

    for (int x = x_start; x < x_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            size_t li = (size_t)(x - x_start) * n_elems_root + y;
            ElementStatic& elem = elements_static[li];

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; ++n) {
                int nx = x + offsets[n][0];
                int ny = y + offsets[n][1];
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    size_t ni;
                    if (nx >= x_start && nx < x_end) {
                        ni = (size_t)(nx - x_start) * n_elems_root + ny;
                    } else if (nx == x_start - 1) {
                        ni = local_n_elems + (size_t)ny;
                    } else { // nx == x_end
                        ni = local_n_elems + (size_t)n_elems_root + ny;
                    }
                    elem.connected_idx[elem.num_connections] = ni;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner materials (convert global to local indices)
    int last = n_elems_root - 1;
    if (x_start <= 0 && 0 < x_end) {
        elements_static[0].material_idx = 1; // INFLOW (0,0)
        elements_static[last].material_idx = 2; // OUTFLOW (0,last)
    }
    if (x_start <= last && last < x_end) {
        size_t local_last_row = (size_t)(last - x_start) * n_elems_root;
        elements_static[local_last_row].material_idx = 2; // OUTFLOW (last,0)
        elements_static[local_last_row + last].material_idx = 1; // INFLOW (last,last)
    }

    // ---- Initialize dynamic data ----
    std::vector<ElementDynamic> owned_dynamic(local_n_elems, {0.0, 0.0});
    std::vector<ElementDynamic> owned_swap(local_n_elems, {0.0, 0.0});

    // Halo: upper row (x_start-1) + lower row (x_end), each n_elems_root cols
    std::vector<val_t> halo_energy(2 * (size_t)n_elems_root, 0.0);

    // Send buffers for boundary data (contiguous current_energy)
    std::vector<val_t> send_upper(n_elems_root);
    std::vector<val_t> send_lower(n_elems_root);

    // ---- Print info (rank 0) ----
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("MPI ranks: %d\n", num_ranks);
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");

        size_t static_mem = (size_t)n_elems * sizeof(ElementStatic);
        size_t dynamic_mem = (size_t)n_elems * sizeof(ElementDynamic) * 2;
        size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // ---- Run simulation ----
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    MPI_Request requests[4];
    int pending_nreq = 0;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Wait for previous iteration's halo communication
        if (pending_nreq > 0) {
            MPI_Waitall(pending_nreq, requests, MPI_STATUSES_IGNORE);
            pending_nreq = 0;
        }

        // Compute all owned elements
        for (size_t i = 0; i < local_n_elems; ++i) {
            const ElementStatic& es = elements_static[i];
            const ElementDynamic& ed = owned_dynamic[i];
            const Material& mat = materials[es.material_idx];

            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < es.num_connections; ++j) {
                idx_t ni = es.connected_idx[j];
                val_t n_energy = (ni < local_n_elems)
                    ? owned_dynamic[ni].current_energy
                    : halo_energy[ni - local_n_elems];
                total_flux += (n_energy - ed.current_energy) *
                              mat.transfer_coeff * es.connected_flux[j] * 0.25;
            }

            owned_swap[i].current_energy = ed.current_energy + total_flux;
            owned_swap[i].total_flux = ed.total_flux + std::abs(total_flux);
        }

        // Swap buffers
        std::swap(owned_dynamic, owned_swap);

        // Post non-blocking halo exchange for next iteration
        if (iter < n_iters - 1 && local_n_x > 0) {
            pending_nreq = 0;

            if (upper_rank != MPI_PROC_NULL) {
                for (int y = 0; y < n_elems_root; ++y)
                    send_upper[y] = owned_dynamic[y].current_energy;
                MPI_Isend(send_upper.data(), n_elems_root, MPI_DOUBLE,
                          upper_rank, 0, MPI_COMM_WORLD, &requests[pending_nreq++]);
                MPI_Irecv(&halo_energy[0], n_elems_root, MPI_DOUBLE,
                          upper_rank, 1, MPI_COMM_WORLD, &requests[pending_nreq++]);
            }
            if (lower_rank != MPI_PROC_NULL) {
                size_t lb = (size_t)(local_n_x - 1) * n_elems_root;
                for (int y = 0; y < n_elems_root; ++y)
                    send_lower[y] = owned_dynamic[lb + y].current_energy;
                MPI_Isend(send_lower.data(), n_elems_root, MPI_DOUBLE,
                          lower_rank, 1, MPI_COMM_WORLD, &requests[pending_nreq++]);
                MPI_Irecv(&halo_energy[n_elems_root], n_elems_root, MPI_DOUBLE,
                          lower_rank, 0, MPI_COMM_WORLD, &requests[pending_nreq++]);
            }
        }
    }

    // Wait for final pending communication
    if (pending_nreq > 0) {
        MPI_Waitall(pending_nreq, requests, MPI_STATUSES_IGNORE);
    }

    auto end = std::chrono::high_resolution_clock::now();
    long local_duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (local_duration_ms == 0) local_duration_ms = 1;
    long duration_ms;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---- Hash (XOR-reduce across ranks) ----
    uint64_t local_hash = 0;
    idx_t global_offset = (idx_t)x_start * n_elems_root;
    for (size_t i = 0; i < local_n_elems; ++i) {
        idx_t gi = global_offset + i;
        const uint64_t* ep = reinterpret_cast<const uint64_t*>(&owned_dynamic[i].current_energy);
        const uint64_t* fp = reinterpret_cast<const uint64_t*>(&owned_dynamic[i].total_flux);
        local_hash ^= (*ep + gi) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*fp + gi) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t global_hash;
    MPI_Allreduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BOR, MPI_COMM_WORLD);

    // ---- Print results (rank 0) ----
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
        printf("  Result hash: %016lX\n", global_hash);
        printf("\n");
    }

    // ---- Print results for external validation ----
    if (printResults) {
        std::vector<double> energy_data;
        if (rank == 0) energy_data.resize(n_elems);

        std::vector<double> local_energy(local_n_elems);
        for (size_t i = 0; i < local_n_elems; ++i)
            local_energy[i] = owned_dynamic[i].current_energy;

        std::vector<int> counts(num_ranks);
        std::vector<int> displs(num_ranks);
        for (int r = 0; r < num_ranks; ++r) {
            int r_xs = r * rows_per_rank + std::min(r, remainder);
            int r_xe = (r + 1) * rows_per_rank + std::min(r + 1, remainder);
            counts[r] = (r_xe - r_xs) * n_elems_root;
            displs[r] = r_xs * n_elems_root;
        }

        MPI_Gatherv(local_energy.data(), (int)local_n_elems, MPI_DOUBLE,
                    rank == 0 ? energy_data.data() : nullptr,
                    counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(energy_data, "ElementEnergy");
        }
    }

    // ---- Validation ----
    if (validate) {
        val_t local_energy_sum = 0.0, local_flux_sum = 0.0;
        val_t local_energy_max = -std::numeric_limits<val_t>::infinity();
        val_t local_energy_min = std::numeric_limits<val_t>::infinity();

        for (size_t i = 0; i < local_n_elems; ++i) {
            local_energy_sum += owned_dynamic[i].current_energy;
            local_flux_sum += owned_dynamic[i].total_flux;
            local_energy_max = std::max(owned_dynamic[i].current_energy, local_energy_max);
            local_energy_min = std::min(owned_dynamic[i].current_energy, local_energy_min);
        }

        val_t g_energy_sum, g_flux_sum, g_energy_max, g_energy_min;
        MPI_Allreduce(&local_energy_sum, &g_energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_flux_sum, &g_flux_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_energy_max, &g_energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_energy_min, &g_energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validation results:\n");
            printf("  Energy sum: %.12f\n", g_energy_sum);
            printf("  Flux sum: %.2f\n", g_flux_sum);
            printf("  Energy range: [%.6f, %.6f]\n", g_energy_min, g_energy_max);

            constexpr val_t energy_epsilon = 1e-8;
            bool valid = true;

            if (!std::isfinite(g_energy_sum)) {
                printf("  ERROR: Energy sum is not finite\n");
                valid = false;
            }
            if (std::abs(g_energy_sum) > energy_epsilon) {
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            }
            if (!std::isfinite(g_flux_sum)) {
                printf("  ERROR: Flux sum is not finite\n");
                valid = false;
            }
            if (!std::isfinite(g_energy_max) || !std::isfinite(g_energy_min)) {
                printf("  ERROR: Energy extrema are not finite\n");
                valid = false;
            }
            if (valid) printf("  Validation: PASSED\n");

            if (!valid) {
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
