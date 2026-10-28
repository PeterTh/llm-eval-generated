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

// Tag used for halo-exchange messages
constexpr int HALO_TAG = 0x5A1;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element.
// connected_idx refers to the rank-local extended index space:
//   [0, n_local)                -> locally owned elements
//   [n_local, n_local + n_ghost) -> ghost copies of remotely owned elements
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

// Start of the contiguous block of global indices owned by `rank`.
// Blocks differ in size by at most one element.
idx_t blockBegin(idx_t n_global, int nranks, int rank) {
    const idx_t base = n_global / static_cast<idx_t>(nranks);
    const idx_t rem = n_global % static_cast<idx_t>(nranks);
    const idx_t r = static_cast<idx_t>(rank);
    return r * base + std::min<idx_t>(r, rem);
}

// Rank-local portion of the distributed unstructured mesh and its halo plan.
struct LocalMesh {
    idx_t n_global = 0;   // Total number of elements in the mesh
    idx_t begin = 0;      // First global index owned by this rank
    idx_t n_local = 0;    // Number of elements owned by this rank
    idx_t n_ghost = 0;    // Number of remote elements referenced by owned elements
    // Local elements [0, n_interior) need no ghost data and are updated while the
    // halo exchange is in flight; [n_interior, n_local) reference ghosts.
    idx_t n_interior = 0;

    // Map from (global index - begin) to local element index
    std::vector<idx_t> global_to_local;

    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;   // n_local entries

    // Current and next-step state, extended with ghost slots [n_local, n_local + n_ghost)
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;

    // Halo plan: values this rank sends to / receives from each peer each iteration
    std::vector<int> send_peers;
    std::vector<int> send_counts;
    std::vector<int> send_displs;
    std::vector<idx_t> send_list;   // local indices, grouped by peer
    std::vector<int> recv_peers;
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;   // offset into ghost slots (and recv buffer)

    std::vector<val_t> send_buf;
    std::vector<val_t> recv_buf;
    std::vector<MPI_Request> send_reqs;
    std::vector<MPI_Request> recv_reqs;
};

// Build the rank-local part of the 2D square grid mesh, plus the halo plan.
// Global element index for (x, y) is x * n + y; this matches the original layout.
void buildLocalMesh(LocalMesh& m, const int n_elems_root, int rank, int nranks, MPI_Comm comm) {
    const idx_t n = static_cast<idx_t>(n_elems_root);
    const idx_t n_global = n * n;
    const idx_t begin = blockBegin(n_global, nranks, rank);
    const idx_t end = blockBegin(n_global, nranks, rank + 1);

    m.n_global = n_global;
    m.begin = begin;
    m.n_local = end - begin;

    // Initialize materials
    m.materials.clear();
    m.materials.push_back(Material{0.8, 0.0});    // Default material
    m.materials.push_back(Material{0.8, 0.5});    // Inflow material
    m.materials.push_back(Material{0.8, -0.5});   // Outflow material

    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    // Global index of the neighbor of g in direction k, if it lies inside the mesh
    auto neighbor = [&](idx_t g, int k, idx_t& ng) -> bool {
        const long long nx = static_cast<long long>(g / n) + offsets[k][0];
        const long long ny = static_cast<long long>(g % n) + offsets[k][1];
        if (nx < 0 || nx >= n_elems_root || ny < 0 || ny >= n_elems_root) return false;
        ng = static_cast<idx_t>(nx) * n + static_cast<idx_t>(ny);
        return true;
    };

    // Classify owned elements and collect remote neighbors (ghosts)
    std::vector<uint8_t> has_ghost(m.n_local, 0);
    std::vector<idx_t> ghosts;
    for (idx_t li = 0; li < m.n_local; ++li) {
        const idx_t g = begin + li;
        for (int k = 0; k < 4; ++k) {
            idx_t ng;
            if (neighbor(g, k, ng) && (ng < begin || ng >= end)) {
                has_ghost[li] = 1;
                ghosts.push_back(ng);
            }
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    m.n_ghost = static_cast<idx_t>(ghosts.size());

    // Local numbering: interior elements first, then boundary elements, each in global order
    m.n_interior = static_cast<idx_t>(m.n_local - std::count(has_ghost.begin(), has_ghost.end(), 1));
    m.global_to_local.assign(m.n_local, 0);
    idx_t next_interior = 0;
    idx_t next_boundary = m.n_interior;
    for (idx_t li = 0; li < m.n_local; ++li) {
        m.global_to_local[li] = has_ghost[li] ? next_boundary++ : next_interior++;
    }

    // Build static connectivity (neighbor order identical to the original)
    m.elements_static.assign(m.n_local, ElementStatic{});
    for (idx_t li = 0; li < m.n_local; ++li) {
        const idx_t g = begin + li;
        ElementStatic& elem = m.elements_static[m.global_to_local[li]];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;
        for (int k = 0; k < 4; ++k) {
            idx_t ng;
            if (!neighbor(g, k, ng)) continue;
            idx_t ext;
            if (ng >= begin && ng < end) {
                ext = m.global_to_local[ng - begin];
            } else {
                const auto it = std::lower_bound(ghosts.begin(), ghosts.end(), ng);
                ext = m.n_local + static_cast<idx_t>(it - ghosts.begin());
            }
            elem.connected_idx[elem.num_connections] = ext;
            elem.connected_flux[elem.num_connections] = 1.0;
            elem.num_connections++;
        }
    }

    // Set corner elements as inflow/outflow (applied in the original order)
    const idx_t last = n - 1;
    const idx_t corner_g[4] = {0 * n + 0, 0 * n + last, last * n + 0, last * n + last};
    const idx_t corner_m[4] = {INFLOW_MAT_ID, OUTFLOW_MAT_ID, OUTFLOW_MAT_ID, INFLOW_MAT_ID};
    for (int c = 0; c < 4; ++c) {
        if (corner_g[c] >= begin && corner_g[c] < end) {
            m.elements_static[m.global_to_local[corner_g[c] - begin]].material_idx = corner_m[c];
        }
    }

    // Ghosts are sorted by global index, so they are grouped by owner rank.
    std::vector<idx_t> starts(nranks + 1);
    for (int r = 0; r <= nranks; ++r) {
        starts[r] = blockBegin(n_global, nranks, r);
    }

    m.recv_peers.clear();
    m.recv_counts.clear();
    m.recv_displs.clear();
    std::vector<int> recv_count_all(nranks, 0);
    for (idx_t gi = 0; gi < m.n_ghost; ++gi) {
        const idx_t g = ghosts[gi];
        // Owner: last rank r with starts[r] <= g (always non-empty for such r)
        const int owner = static_cast<int>(std::upper_bound(starts.begin(), starts.end(), g) - starts.begin()) - 1;
        if (m.recv_peers.empty() || m.recv_peers.back() != owner) {
            m.recv_peers.push_back(owner);
            m.recv_counts.push_back(0);
            m.recv_displs.push_back(static_cast<int>(gi));
        }
        m.recv_counts.back()++;
        recv_count_all[owner]++;
    }

    // Exchange requests so each owner learns which of its elements to send.
    std::vector<int> send_count_all(nranks, 0);
    MPI_Alltoall(recv_count_all.data(), 1, MPI_INT, send_count_all.data(), 1, MPI_INT, comm);

    std::vector<int> sdispl(nranks, 0), rdispl(nranks, 0);
    for (int r = 1; r < nranks; ++r) {
        sdispl[r] = sdispl[r - 1] + recv_count_all[r - 1];
        rdispl[r] = rdispl[r - 1] + send_count_all[r - 1];
    }
    const int total_requested = nranks > 0 ? rdispl[nranks - 1] + send_count_all[nranks - 1] : 0;
    std::vector<idx_t> requested(total_requested);
    MPI_Alltoallv(ghosts.data(), recv_count_all.data(), sdispl.data(), MPI_UINT64_T,
                  requested.data(), send_count_all.data(), rdispl.data(), MPI_UINT64_T, comm);

    m.send_peers.clear();
    m.send_counts.clear();
    m.send_displs.clear();
    m.send_list.clear();
    for (int r = 0; r < nranks; ++r) {
        if (send_count_all[r] == 0) continue;
        m.send_peers.push_back(r);
        m.send_counts.push_back(send_count_all[r]);
        m.send_displs.push_back(static_cast<int>(m.send_list.size()));
        for (int k = 0; k < send_count_all[r]; ++k) {
            m.send_list.push_back(m.global_to_local[requested[rdispl[r] + k] - begin]);
        }
    }

    // Allocate state buffers (ghost slots included), zero-initialized
    const size_t n_ext = static_cast<size_t>(m.n_local + m.n_ghost);
    m.elements_dynamic.assign(n_ext, ElementDynamic{0.0, 0.0});
    m.elements_dynamic_swap.assign(n_ext, ElementDynamic{0.0, 0.0});

    m.send_buf.assign(m.send_list.size(), 0.0);
    m.recv_buf.assign(static_cast<size_t>(m.n_ghost), 0.0);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                         val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update owned element i: read the current state `cur`, write the next state to `nxt[i]`
inline void updateElement(const ElementStatic* __restrict statics, const Material* __restrict materials,
                          const ElementDynamic* __restrict cur, ElementDynamic* __restrict nxt, idx_t i) {
    const ElementStatic& elem_static = statics[i];
    const ElementDynamic& elem_dyn = cur[i];
    const Material& mat = materials[elem_static.material_idx];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const idx_t neighbor_idx = elem_static.connected_idx[j];
        const ElementDynamic& neighbor_dyn = cur[neighbor_idx];
        total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
    }

    // Update element state
    nxt[i].current_energy = elem_dyn.current_energy + total_flux;
    nxt[i].total_flux = elem_dyn.total_flux + std::abs(total_flux);
}

// Run simulation for n_iters iterations. Ghost energies are refreshed from
// neighboring ranks before every update; communication overlaps interior work.
// On return, m.elements_dynamic holds the final state.
void runSimulation(LocalMesh& m, const int n_iters, MPI_Comm comm) {
    const int n_recv = static_cast<int>(m.recv_peers.size());
    const int n_send = static_cast<int>(m.send_peers.size());
    const idx_t n_local = m.n_local;
    const idx_t n_interior = m.n_interior;
    const ElementStatic* statics = m.elements_static.data();
    const Material* materials = m.materials.data();

    for (int iter = 0; iter < n_iters; ++iter) {
        ElementDynamic* cur = m.elements_dynamic.data();
        ElementDynamic* nxt = m.elements_dynamic_swap.data();

        // Post receives for ghost energies
        m.recv_reqs.resize(n_recv);
        for (int k = 0; k < n_recv; ++k) {
            MPI_Irecv(m.recv_buf.data() + m.recv_displs[k], m.recv_counts[k], MPI_DOUBLE,
                      m.recv_peers[k], HALO_TAG, comm, &m.recv_reqs[k]);
        }

        // Pack and send owned energies needed by peers
        for (size_t j = 0; j < m.send_list.size(); ++j) {
            m.send_buf[j] = cur[m.send_list[j]].current_energy;
        }
        m.send_reqs.resize(n_send);
        for (int k = 0; k < n_send; ++k) {
            MPI_Isend(m.send_buf.data() + m.send_displs[k], m.send_counts[k], MPI_DOUBLE,
                      m.send_peers[k], HALO_TAG, comm, &m.send_reqs[k]);
        }

        // Interior elements do not depend on ghosts: overlap with communication
        for (idx_t i = 0; i < n_interior; ++i) {
            updateElement(statics, materials, cur, nxt, i);
        }

        // Ghost data has arrived: install it and finish boundary elements
        MPI_Waitall(n_recv, m.recv_reqs.data(), MPI_STATUSES_IGNORE);
        for (size_t j = 0; j < m.recv_buf.size(); ++j) {
            cur[n_local + j].current_energy = m.recv_buf[j];
        }
        for (idx_t i = n_interior; i < n_local; ++i) {
            updateElement(statics, materials, cur, nxt, i);
        }

        MPI_Waitall(n_send, m.send_reqs.data(), MPI_STATUSES_IGNORE);

        // Swap buffers
        std::swap(m.elements_dynamic, m.elements_dynamic_swap);
    }
}

// Gather owned values (given in global index order) to rank 0, which receives
// the full array in global order
std::vector<val_t> gatherToRoot(const LocalMesh& m, const std::vector<val_t>& local,
                                int rank, int nranks, MPI_Comm comm) {
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        const idx_t b = blockBegin(m.n_global, nranks, r);
        const idx_t e = blockBegin(m.n_global, nranks, r + 1);
        counts[r] = static_cast<int>(e - b);
        displs[r] = static_cast<int>(b);
    }
    std::vector<val_t> out;
    if (rank == 0) out.resize(m.n_global);
    MPI_Gatherv(local.data(), static_cast<int>(m.n_local), MPI_DOUBLE,
                out.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    return out;
}

// Validate simulation results (operates on global, index-ordered arrays on rank 0)
bool validateResults(const std::vector<val_t>& energies, const std::vector<val_t>& fluxes) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energies.size(); ++i) {
        energy_sum += energies[i];
        flux_sum += fluxes[i];
        energy_max = std::max(energies[i], energy_max);
        energy_min = std::min(energies[i], energy_min);
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

// Partial hash over owned elements, using global indices. XOR is
// order-independent, so the XOR-reduction across ranks equals the serial hash.
uint64_t computeLocalHash(const LocalMesh& m) {
    uint64_t hash = 0;
    for (idx_t k = 0; k < m.n_local; ++k) {
        const idx_t i = m.begin + k;   // global index
        const ElementDynamic& elem = m.elements_dynamic[m.global_to_local[k]];
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elem.current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elem.total_flux);
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

int runBenchmark(int argc, char** argv, int rank, int nranks) {
    const bool root = (rank == 0);
    MPI_Comm comm = MPI_COMM_WORLD;

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (n_elems_root < 1) {
        if (root) {
            printf("Error: grid size must be at least 1\n");
            printUsage(argv[0]);
        }
        return 1;
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        printf("Building unstructured mesh...\n");
    }

    // Build the distributed unstructured mesh (each rank holds only its block)
    LocalMesh mesh;
    buildLocalMesh(mesh, n_elems_root, rank, nranks, comm);

    // Calculate memory usage (global totals, as in the serial version)
    if (root) {
        const size_t static_mem = static_cast<size_t>(mesh.n_global) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(mesh.n_global) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        printf("Running simulation...\n");
    }

    // Run simulation (timed across all ranks; reported time is the slowest rank)
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(mesh, n_iters, comm);

    auto end = std::chrono::high_resolution_clock::now();
    const int64_t local_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    int64_t max_ns = 0;
    MPI_Allreduce(&local_ns, &max_ns, 1, MPI_INT64_T, MPI_MAX, comm);
    const long duration_ms = static_cast<long>(max_ns / 1000000);

    if (root) {
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
    }

    // Compute hash for verification (global XOR reduction)
    uint64_t local_hash = computeLocalHash(mesh);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    if (root) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Gather owned state on rank 0 in global order for output / validation
    int exit_code = 0;
    if (printResults || validate) {
        std::vector<val_t> energies(mesh.n_local);
        std::vector<val_t> fluxes(mesh.n_local);
        for (idx_t k = 0; k < mesh.n_local; ++k) {   // global index order
            const ElementDynamic& elem = mesh.elements_dynamic[mesh.global_to_local[k]];
            energies[k] = elem.current_energy;
            fluxes[k] = elem.total_flux;
        }
        std::vector<val_t> energy_all = gatherToRoot(mesh, energies, rank, nranks, comm);

        // Print results for external validation
        if (printResults && root) {
            print_results(energy_all, "ElementEnergy");
        }

        if (validate) {
            std::vector<val_t> flux_all = gatherToRoot(mesh, fluxes, rank, nranks, comm);
            if (root) {
                exit_code = validateResults(energy_all, flux_all) ? 0 : 1;
            }
            // Make every rank exit with the same status
            MPI_Bcast(&exit_code, 1, MPI_INT, 0, comm);
        }
    }

    return exit_code;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const int status = runBenchmark(argc, argv, rank, nranks);

    MPI_Finalize();
    return status;
}
