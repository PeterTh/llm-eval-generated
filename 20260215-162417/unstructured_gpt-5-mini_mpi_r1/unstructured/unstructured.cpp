#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <unordered_map>
#include <numeric>

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

// World state (local partition)
struct World {
    std::vector<Material> materials; // small, replicated
    std::vector<ElementStatic> elements_static; // local chunk
    std::vector<ElementDynamic> elements_dynamic; // local chunk
    std::vector<ElementDynamic> elements_dynamic_swap; // local chunk
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Helper: compute owner rank for a global index given a block distribution
static void compute_block_partition(idx_t total_count, int nprocs, idx_t rank,
                                    idx_t& out_start, idx_t& out_count) {
    idx_t base = total_count / nprocs;
    idx_t rem = total_count % nprocs;
    if ((idx_t)rank < rem) {
        out_start = rank * (base + 1);
        out_count = base + 1;
    } else {
        out_start = rem * (base + 1) + (rank - rem) * base;
        out_count = base;
    }
}

static int owner_of(idx_t idx, idx_t total_count, int nprocs) {
    idx_t base = total_count / nprocs;
    idx_t rem = total_count % nprocs;
    if (idx < (base + 1) * rem) {
        return static_cast<int>(idx / (base + 1));
    } else {
        return static_cast<int>(rem + (idx - (base + 1) * rem) / base);
    }
}

// Build a 2D square grid as an unstructured mesh for a local partition
void buildSquare2D(World& world, const int n_elems_root, idx_t global_start = 0, idx_t local_count = 0) {
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * n_elems_root;
    bool build_all = (local_count == 0 || local_count == n_elems);

    // Initialize materials (replicated)
    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    if (build_all) {
        // Allocate and initialize full world (fallback)
        world.elements_static.resize(n_elems);
        world.elements_dynamic.resize(n_elems);
        world.elements_dynamic_swap.resize(n_elems);

        for (idx_t i = 0; i < n_elems; ++i) {
            world.elements_static[i].material_idx = DEFAULT_MAT_ID;
            world.elements_static[i].num_connections = 0;
            world.elements_dynamic[i].current_energy = 0.0;
            world.elements_dynamic[i].total_flux = 0.0;
        }

        for (int x = 0; x < n_elems_root; ++x) {
            for (int y = 0; y < n_elems_root; ++y) {
                const idx_t idx = static_cast<idx_t>(x) * n_elems_root + y;
                ElementStatic& elem = world.elements_static[idx];
                const int offsets[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
                for (int n = 0; n < 4; ++n) {
                    const int nx = x + offsets[n][0];
                    const int ny = y + offsets[n][1];
                    if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                        const idx_t neighbor_idx = static_cast<idx_t>(nx) * n_elems_root + ny;
                        elem.connected_idx[elem.num_connections] = neighbor_idx;
                        elem.connected_flux[elem.num_connections] = 1.0;
                        elem.num_connections++;
                    }
                }
            }
        }

        const int last = n_elems_root - 1;
        world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
        world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
        return;
    }

    // Local-only allocation
    world.elements_static.resize(local_count);
    world.elements_dynamic.resize(local_count);
    world.elements_dynamic_swap.resize(local_count);

    for (idx_t local_i = 0; local_i < local_count; ++local_i) {
        world.elements_static[local_i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[local_i].num_connections = 0;
        world.elements_dynamic[local_i].current_energy = 0.0;
        world.elements_dynamic[local_i].total_flux = 0.0;
    }

    for (idx_t local_i = 0; local_i < local_count; ++local_i) {
        idx_t gidx = global_start + local_i;
        int x = static_cast<int>(gidx / n_elems_root);
        int y = static_cast<int>(gidx % n_elems_root);
        ElementStatic& elem = world.elements_static[local_i];
        const int offsets[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_idx = static_cast<idx_t>(nx) * n_elems_root + ny;
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }

    // Set corners if present in this partition
    const idx_t last = static_cast<idx_t>(n_elems_root - 1);
    auto set_material_if_owned = [&](idx_t gx, idx_t gy, idx_t mat) {
        idx_t gidx = gx * n_elems_root + gy;
        if (gidx >= global_start && gidx < global_start + local_count) {
            world.elements_static[gidx - global_start].material_idx = mat;
        }
    };
    set_material_if_owned(0,0,INFLOW_MAT_ID);
    set_material_if_owned(0,last,OUTFLOW_MAT_ID);
    set_material_if_owned(last,0,OUTFLOW_MAT_ID);
    set_material_if_owned(last,last,INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux_local(const Material& mat, val_t this_energy,
                               val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// Run parallel simulation for n_iters iterations (distributed world)
void runSimulationParallel(World& world, const int n_iters,
                           const int n_elems_root, idx_t global_start, idx_t global_count,
                           idx_t total_global_elems, int rank, int nprocs) {
    // Prepare communication patterns: for each peer, which global indices we need from them
    std::vector<std::vector<idx_t>> need_from_peer(nprocs);

    for (idx_t local_i = 0; local_i < global_count; ++local_i) {
        const ElementStatic& es = world.elements_static[local_i];
        for (idx_t j = 0; j < es.num_connections; ++j) {
            idx_t neighbor = es.connected_idx[j];
            int owner = owner_of(neighbor, total_global_elems, nprocs);
            if (owner != rank) need_from_peer[owner].push_back(neighbor);
        }
    }

    // Remove duplicates and sort for deterministic ordering
    for (int p = 0; p < nprocs; ++p) {
        auto &v = need_from_peer[p];
        if (v.empty()) continue;
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    // Exchange lists using Alltoallv so each peer knows which indices others want from it
    std::vector<int> sendcounts(nprocs), recvcounts(nprocs);
    for (int p = 0; p < nprocs; ++p) sendcounts[p] = static_cast<int>(need_from_peer[p].size());
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> sdispls(nprocs+1), rdispls(nprocs+1);
    sdispls[0] = 0; rdispls[0] = 0;
    for (int p = 0; p < nprocs; ++p) {
        sdispls[p+1] = sdispls[p] + sendcounts[p];
        rdispls[p+1] = rdispls[p] + recvcounts[p];
    }
    std::vector<idx_t> sendbuf(sdispls.back());
    for (int p = 0; p < nprocs; ++p) {
        std::copy(need_from_peer[p].begin(), need_from_peer[p].end(), sendbuf.begin() + sdispls[p]);
    }
    std::vector<idx_t> recvbuf(rdispls.back());

    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_UINT64_T,
                  recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_UINT64_T, MPI_COMM_WORLD);

    // Now build send list: peers_want_from_me[p] contains global indices that peer p expects from this rank
    std::vector<std::vector<idx_t>> peers_want_from_me(nprocs);
    for (int p = 0; p < nprocs; ++p) {
        int cnt = recvcounts[p];
        if (cnt == 0) continue;
        peers_want_from_me[p].assign(recvbuf.data() + rdispls[p], recvbuf.data() + rdispls[p] + cnt);
    }

    // For fast lookup during updates, map requested indices -> local offsets for sends
    std::vector<std::vector<int>> send_local_offsets(nprocs);
    for (int p = 0; p < nprocs; ++p) {
        auto &v = peers_want_from_me[p];
        send_local_offsets[p].reserve(v.size());
        for (idx_t gidx : v) {
            send_local_offsets[p].push_back(static_cast<int>(gidx - global_start));
        }
    }

    // Similarly, map recv lists (need_from_peer) to local storage indices for unpacking
    std::vector<std::vector<idx_t>> recv_global_lists = need_from_peer; // copy

    // Buffers for communication per peer
    std::vector<std::vector<val_t>> send_buffers(nprocs);
    std::vector<std::vector<val_t>> recv_buffers(nprocs);
    std::vector<MPI_Request> requests;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Prepare send buffers using current local energies
        for (int p = 0; p < nprocs; ++p) {
            if (send_local_offsets[p].empty()) continue;
            auto &buf = send_buffers[p];
            buf.resize(send_local_offsets[p].size());
            for (size_t k = 0; k < send_local_offsets[p].size(); ++k) {
                int local_off = send_local_offsets[p][k];
                buf[k] = world.elements_dynamic[local_off].current_energy;
            }
            // prepare recv buffer sized to what we requested from p
            recv_buffers[p].resize(recv_global_lists[p].size());
        }

        // Post receives
        requests.clear();
        for (int p = 0; p < nprocs; ++p) {
            if (recv_buffers[p].empty()) continue;
            MPI_Request req;
            MPI_Irecv(recv_buffers[p].data(), static_cast<int>(recv_buffers[p].size()), MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
        // Post sends
        for (int p = 0; p < nprocs; ++p) {
            if (send_buffers[p].empty()) continue;
            MPI_Request req;
            MPI_Isend(send_buffers[p].data(), static_cast<int>(send_buffers[p].size()), MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }

        // Compute updates but need remote neighbor energies: wait for receives first
        if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

        // Build a map from global index -> received energy
        std::unordered_map<idx_t, val_t> remote_energy;
        for (int p = 0; p < nprocs; ++p) {
            auto &rlist = recv_global_lists[p];
            auto &rbuf = recv_buffers[p];
            for (size_t k = 0; k < rlist.size(); ++k) {
                remote_energy[rlist[k]] = rbuf[k];
            }
        }

        // Update local elements
        for (idx_t local_i = 0; local_i < global_count; ++local_i) {
            const ElementStatic& elem_static = world.elements_static[local_i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[local_i];
            const Material& mat = world.materials[elem_static.material_idx];

            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                int owner = owner_of(neighbor_idx, total_global_elems, nprocs);
                val_t neighbor_energy;
                if (owner == rank) {
                    neighbor_energy = world.elements_dynamic[neighbor_idx - global_start].current_energy;
                } else {
                    neighbor_energy = remote_energy[neighbor_idx];
                }
                total_flux += computeFlux_local(mat, elem_dyn.current_energy, elem_static.connected_flux[j], neighbor_energy);
            }

            ElementDynamic& elem_write = world.elements_dynamic_swap[local_i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }

        // Swap local buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (same as before but operates on a gathered full world)
bool validateResultsFull(const std::vector<ElementDynamic>& elements) {
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse)
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

    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %llu elements\n", n_elems_root, n_elems_root, (unsigned long long)n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Processes: %d\n", nprocs);
        printf("\n");
    }

    // Determine local partition
    idx_t local_start = 0, local_count = 0;
    compute_block_partition(n_elems, nprocs, rank, local_start, local_count);

    if (rank == 0) printf("Building unstructured mesh (distributed)...\n");
    World world;
    buildSquare2D(world, n_elems_root, local_start, local_count);

    // Local memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) {
        printf("Memory usage per rank (approx): %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0), static_mem / (1024.0 * 1024.0), dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Warm up barrier, then timed simulation
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    runSimulationParallel(world, n_iters, n_elems_root, local_start, local_count, n_elems, rank, nprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double duration_ms = (end_time - start_time) * 1000.0;

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration_ms);
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Gather results to rank 0 for hashing/validation/printing
    // Pack local current_energy and total_flux into contiguous buffer
    std::vector<val_t> local_pack(2 * local_count);
    for (idx_t i = 0; i < local_count; ++i) {
        local_pack[2*i] = world.elements_dynamic[i].current_energy;
        local_pack[2*i+1] = world.elements_dynamic[i].total_flux;
    }

    // Gather counts
    std::vector<int> local_counts(nprocs);
    int my_count = static_cast<int>(local_count);
    MPI_Allgather(&my_count, 1, MPI_INT, local_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> recvcounts(nprocs), displs(nprocs);
    for (int p = 0; p < nprocs; ++p) recvcounts[p] = local_counts[p] * 2; // two doubles per element
    displs[0] = 0;
    for (int p = 1; p < nprocs; ++p) displs[p] = displs[p-1] + recvcounts[p-1];
    int total_recv = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);

    std::vector<val_t> gathered;
    if (rank == 0) gathered.resize(total_recv);

    MPI_Gatherv(local_pack.data(), static_cast<int>(local_pack.size()), MPI_DOUBLE,
                gathered.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Reconstruct full ElementDynamic vector
        std::vector<ElementDynamic> full_elements;
        full_elements.resize(n_elems);
        int pos = 0;
        for (int p = 0; p < nprocs; ++p) {
            int cnt = local_counts[p];
            // Determine start index for rank p
            idx_t pstart = 0, pcount = 0;
            compute_block_partition(n_elems, nprocs, p, pstart, pcount);
            for (int i = 0; i < cnt; ++i) {
                ElementDynamic ed;
                ed.current_energy = gathered[pos++];
                ed.total_flux = gathered[pos++];
                full_elements[pstart + i] = ed;
            }
        }

        uint64_t hash = computeHash(full_elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(full_elements.size());
            for (const auto& e : full_elements) energyData.push_back(e.current_energy);
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            bool ok = validateResultsFull(full_elements);
            if (!ok) {
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
