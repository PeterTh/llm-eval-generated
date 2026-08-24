#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
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
    idx_t connected_idx[MAX_CONNECTIONS];     // Local indices of connected elements
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

// Validate simulation results (rank 0 only, on gathered data)
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
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Materials (identical on all ranks)
    std::vector<Material> materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};

    // Contiguous partitioning of elements across ranks
    std::vector<int> part_start(nprocs + 1);
    for (int r = 0; r <= nprocs; ++r) {
        part_start[r] = (int)((long long)r * n_elems / nprocs);
    }
    int local_start = part_start[rank];
    int local_end = part_start[rank + 1];
    int n_local = local_end - local_start;

    // Material index for a global element
    auto get_material = [&](int gi) -> idx_t {
        int last = n_elems_root - 1;
        if (gi == 0) return INFLOW_MAT_ID;
        if (gi == last) return OUTFLOW_MAT_ID;
        if (gi == last * n_elems_root) return OUTFLOW_MAT_ID;
        if (gi == last * n_elems_root + last) return INFLOW_MAT_ID;
        return DEFAULT_MAT_ID;
    };

    // Identify ghost elements needed from other ranks
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    std::set<int> ghost_set;

    for (int li = 0; li < n_local; ++li) {
        int gi = local_start + li;
        int x = gi / n_elems_root;
        int y = gi % n_elems_root;
        for (int n = 0; n < 4; ++n) {
            int nx = x + offsets[n][0];
            int ny = y + offsets[n][1];
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                int ngi = nx * n_elems_root + ny;
                if (ngi < local_start || ngi >= local_end) {
                    ghost_set.insert(ngi);
                }
            }
        }
    }

    std::vector<int> ghost_global(ghost_set.begin(), ghost_set.end());
    int n_ghost = (int)ghost_global.size();

    // Global-to-local index mapping for ghost elements
    std::map<int, int> ghost_g2l;
    for (int i = 0; i < n_ghost; ++i) {
        ghost_g2l[ghost_global[i]] = n_local + i;
    }

    auto g2l = [&](int gi) -> idx_t {
        if (gi >= local_start && gi < local_end) return (idx_t)(gi - local_start);
        return (idx_t)ghost_g2l.at(gi);
    };

    // Build local element connectivity with local indices
    int n_total = n_local + n_ghost;
    std::vector<ElementStatic> elements_static(n_local);
    std::vector<ElementDynamic> elements_dynamic(n_total, {0.0, 0.0});
    std::vector<ElementDynamic> elements_dynamic_swap(n_total, {0.0, 0.0});

    for (int li = 0; li < n_local; ++li) {
        int gi = local_start + li;
        int x = gi / n_elems_root;
        int y = gi % n_elems_root;

        elements_static[li].material_idx = get_material(gi);
        elements_static[li].num_connections = 0;

        for (int n = 0; n < 4; ++n) {
            int nx = x + offsets[n][0];
            int ny = y + offsets[n][1];
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                int ngi = nx * n_elems_root + ny;
                idx_t conn = elements_static[li].num_connections;
                elements_static[li].connected_idx[conn] = g2l(ngi);
                elements_static[li].connected_flux[conn] = 1.0;
                elements_static[li].num_connections++;
            }
        }
    }

    // Classify elements as interior (all-local connections) or boundary (has ghost connections)
    // for computation-communication overlap
    std::vector<int> interior_elems, boundary_elems;
    interior_elems.reserve(n_local);
    boundary_elems.reserve(n_local);
    for (int i = 0; i < n_local; ++i) {
        bool is_boundary = false;
        for (idx_t j = 0; j < elements_static[i].num_connections; ++j) {
            if (elements_static[i].connected_idx[j] >= (idx_t)n_local) {
                is_boundary = true;
                break;
            }
        }
        if (is_boundary) boundary_elems.push_back(i);
        else interior_elems.push_back(i);
    }

    // Setup MPI halo exchange communication structures
    auto owner_of = [&](int gi) -> int {
        int lo = 0, hi = nprocs - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (gi < part_start[mid + 1]) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    };

    // Group ghost elements by the rank that owns them
    std::map<int, std::vector<int>> recv_map;
    for (int gi : ghost_global) {
        recv_map[owner_of(gi)].push_back(gi);
    }

    std::vector<int> neighbor_ranks;
    for (auto& [r, _] : recv_map) {
        neighbor_ranks.push_back(r);
    }
    int n_neighbors = (int)neighbor_ranks.size();

    std::vector<int> recv_counts(n_neighbors), send_counts(n_neighbors);
    for (int i = 0; i < n_neighbors; ++i) {
        recv_counts[i] = (int)recv_map[neighbor_ranks[i]].size();
    }

    // Exchange counts with neighbors
    std::vector<MPI_Request> reqs;
    if (n_neighbors > 0) {
        reqs.resize(2 * n_neighbors);
        for (int i = 0; i < n_neighbors; ++i) {
            MPI_Isend(&recv_counts[i], 1, MPI_INT, neighbor_ranks[i], 0, MPI_COMM_WORLD, &reqs[i]);
            MPI_Irecv(&send_counts[i], 1, MPI_INT, neighbor_ranks[i], 0, MPI_COMM_WORLD, &reqs[n_neighbors + i]);
        }
        MPI_Waitall(2 * n_neighbors, reqs.data(), MPI_STATUSES_IGNORE);
    }

    // Exchange index lists: send what I need, receive what they need from me
    std::vector<std::vector<int>> send_global(n_neighbors);
    for (int i = 0; i < n_neighbors; ++i) {
        send_global[i].resize(send_counts[i]);
    }

    if (n_neighbors > 0) {
        reqs.resize(2 * n_neighbors);
        for (int i = 0; i < n_neighbors; ++i) {
            MPI_Isend(recv_map[neighbor_ranks[i]].data(), recv_counts[i], MPI_INT,
                      neighbor_ranks[i], 1, MPI_COMM_WORLD, &reqs[i]);
            MPI_Irecv(send_global[i].data(), send_counts[i], MPI_INT,
                      neighbor_ranks[i], 1, MPI_COMM_WORLD, &reqs[n_neighbors + i]);
        }
        MPI_Waitall(2 * n_neighbors, reqs.data(), MPI_STATUSES_IGNORE);
    }

    // Convert indices: send list to local offsets, recv list to ghost positions
    std::vector<std::vector<int>> send_local(n_neighbors), recv_local(n_neighbors);
    for (int i = 0; i < n_neighbors; ++i) {
        send_local[i].resize(send_counts[i]);
        for (int j = 0; j < send_counts[i]; ++j) {
            send_local[i][j] = send_global[i][j] - local_start;
        }
        recv_local[i].resize(recv_counts[i]);
        for (int j = 0; j < recv_counts[i]; ++j) {
            recv_local[i][j] = ghost_g2l[recv_map[neighbor_ranks[i]][j]];
        }
    }

    // Pre-allocate communication buffers
    std::vector<std::vector<val_t>> send_bufs(n_neighbors), recv_bufs(n_neighbors);
    for (int i = 0; i < n_neighbors; ++i) {
        send_bufs[i].resize(send_counts[i]);
        recv_bufs[i].resize(recv_counts[i]);
    }

    if (rank == 0) {
        printf("Building unstructured mesh...\n");
        const size_t static_mem = (size_t)n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = (size_t)n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Simulation loop with computation-communication overlap
    for (int iter = 0; iter < n_iters; ++iter) {
        // Start non-blocking ghost exchange
        if (n_neighbors > 0) {
            reqs.resize(2 * n_neighbors);
            for (int i = 0; i < n_neighbors; ++i) {
                for (int j = 0; j < send_counts[i]; ++j) {
                    send_bufs[i][j] = elements_dynamic[send_local[i][j]].current_energy;
                }
                MPI_Isend(send_bufs[i].data(), send_counts[i], MPI_DOUBLE,
                          neighbor_ranks[i], 2, MPI_COMM_WORLD, &reqs[i]);
                MPI_Irecv(recv_bufs[i].data(), recv_counts[i], MPI_DOUBLE,
                          neighbor_ranks[i], 2, MPI_COMM_WORLD, &reqs[n_neighbors + i]);
            }
        }

        // Compute interior elements while communication is in flight
        for (int idx : interior_elems) {
            const ElementStatic& es = elements_static[idx];
            const ElementDynamic& ed = elements_dynamic[idx];
            const Material& mat = materials[es.material_idx];

            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < es.num_connections; ++j) {
                total_flux += computeFlux(mat, ed, es.connected_flux[j],
                                          elements_dynamic[es.connected_idx[j]]);
            }

            elements_dynamic_swap[idx].current_energy = ed.current_energy + total_flux;
            elements_dynamic_swap[idx].total_flux = ed.total_flux + std::abs(total_flux);
        }

        // Wait for ghost exchange to complete
        if (n_neighbors > 0) {
            MPI_Waitall(2 * n_neighbors, reqs.data(), MPI_STATUSES_IGNORE);
            for (int i = 0; i < n_neighbors; ++i) {
                for (int j = 0; j < recv_counts[i]; ++j) {
                    elements_dynamic[recv_local[i][j]].current_energy = recv_bufs[i][j];
                }
            }
        }

        // Compute boundary elements that depend on ghost data
        for (int idx : boundary_elems) {
            const ElementStatic& es = elements_static[idx];
            const ElementDynamic& ed = elements_dynamic[idx];
            const Material& mat = materials[es.material_idx];

            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < es.num_connections; ++j) {
                total_flux += computeFlux(mat, ed, es.connected_flux[j],
                                          elements_dynamic[es.connected_idx[j]]);
            }

            elements_dynamic_swap[idx].current_energy = ed.current_energy + total_flux;
            elements_dynamic_swap[idx].total_flux = ed.total_flux + std::abs(total_flux);
        }

        // Swap buffers
        std::swap(elements_dynamic, elements_dynamic_swap);
    }

    auto end = std::chrono::high_resolution_clock::now();
    long long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to rank 0
    std::vector<double> local_energy(n_local), local_flux(n_local);
    for (int i = 0; i < n_local; ++i) {
        local_energy[i] = elements_dynamic[i].current_energy;
        local_flux[i] = elements_dynamic[i].total_flux;
    }

    std::vector<int> gather_counts(nprocs), gather_displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        gather_counts[r] = part_start[r + 1] - part_start[r];
        gather_displs[r] = part_start[r];
    }

    std::vector<double> all_energy, all_flux;
    if (rank == 0) {
        all_energy.resize(n_elems);
        all_flux.resize(n_elems);
    }

    MPI_Gatherv(local_energy.data(), n_local, MPI_DOUBLE,
                all_energy.data(), gather_counts.data(), gather_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_flux.data(), n_local, MPI_DOUBLE,
                all_flux.data(), gather_counts.data(), gather_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int ret = 0;
    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (double)n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Reconstruct full element array for hash/validation
        std::vector<ElementDynamic> all_elements(n_elems);
        for (int i = 0; i < n_elems; ++i) {
            all_elements[i].current_energy = all_energy[i];
            all_elements[i].total_flux = all_flux[i];
        }

        // Compute hash for verification
        const uint64_t hash = computeHash(all_elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData(all_energy.begin(), all_energy.end());
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            if (!validateResults(all_elements)) {
                ret = 1;
            }
        }
    }

    MPI_Finalize();
    return ret;
}
