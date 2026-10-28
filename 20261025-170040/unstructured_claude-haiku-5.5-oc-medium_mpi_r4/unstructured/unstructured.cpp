#include <mpi.h>

#include <algorithm>
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
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element (used for final gathered results)
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

std::vector<Material> makeMaterials() {
    return {Material{0.8, 0.0}, Material{0.8, 0.5}, Material{0.8, -0.5}};
}

// ---------------------------------------------------------------------------
// Global mesh helpers. The mesh is a 2D square grid with global index
// g = x * n + y. It is never materialised globally; each rank only derives
// the connectivity of the elements it owns.
// ---------------------------------------------------------------------------

// Contiguous block partition of the global element index space.
// Rank r owns the half-open range [blockBegin(r), blockBegin(r + 1)).
idx_t blockBegin(idx_t n_elems, int n_ranks, int rank) {
    const idx_t base = n_elems / static_cast<idx_t>(n_ranks);
    const idx_t rem = n_elems % static_cast<idx_t>(n_ranks);
    const idx_t r = static_cast<idx_t>(rank);
    return r * base + std::min(r, rem);
}

// Rank that owns global element g under the block partition above.
int ownerOf(idx_t g, idx_t n_elems, int n_ranks) {
    const idx_t base = n_elems / static_cast<idx_t>(n_ranks);
    const idx_t rem = n_elems % static_cast<idx_t>(n_ranks);
    const idx_t big = rem * (base + 1);
    if (g < big) return static_cast<int>(g / (base + 1));
    return static_cast<int>(rem + (g - big) / base);
}

// Neighbours of global element g, in the same order as the serial buildSquare2D
// (+x, -x, +y, -y). Returns the number of in-bounds neighbours written to out.
int globalNeighbors(idx_t n, idx_t g, idx_t out[4]) {
    static constexpr long long dx[4] = {1, -1, 0, 0};
    static constexpr long long dy[4] = {0, 0, 1, -1};
    const long long nn = static_cast<long long>(n);
    const long long x = static_cast<long long>(g) / nn;
    const long long y = static_cast<long long>(g) % nn;
    int count = 0;
    for (int k = 0; k < 4; ++k) {
        const long long nx = x + dx[k];
        const long long ny = y + dy[k];
        if (nx >= 0 && nx < nn && ny >= 0 && ny < nn) {
            out[count++] = static_cast<idx_t>(nx * nn + ny);
        }
    }
    return count;
}

// Material of global element g, reproducing the serial assignment order
// (the later assignments win when they alias, e.g. for n == 1).
idx_t materialOf(idx_t n, idx_t g) {
    const idx_t last = n - 1;
    idx_t mat = DEFAULT_MAT_ID;
    if (g == 0) mat = INFLOW_MAT_ID;                  // (0, 0)
    if (g == last) mat = OUTFLOW_MAT_ID;              // (0, last)
    if (g == last * n) mat = OUTFLOW_MAT_ID;          // (last, 0)
    if (g == last * n + last) mat = INFLOW_MAT_ID;    // (last, last)
    return mat;
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, val_t self_energy,
                         val_t connection_flux, val_t other_energy) {
    return (other_energy - self_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// Update one owned element: reads current state (owned and ghost slots),
// writes next state for the owned element only.
inline void updateElement(idx_t i, const ElementStatic& es, const Material* materials,
                          const val_t* e_cur, val_t* e_next,
                          const val_t* f_cur, val_t* f_next) {
    const Material& mat = materials[es.material_idx];
    const val_t self = e_cur[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    for (idx_t j = 0; j < es.num_connections; ++j) {
        total_flux += computeFlux(mat, self, es.connected_flux[j], e_cur[es.connected_idx[j]]);
    }

    e_next[i] = self + total_flux;
    f_next[i] = f_cur[i] + std::abs(total_flux);
}

// ---------------------------------------------------------------------------
// Distributed mesh
// ---------------------------------------------------------------------------

// Half-open range of local element indices [begin, end)
struct IndexRange {
    idx_t begin;
    idx_t end;
};

struct LocalMesh {
    idx_t lo = 0;        // first owned global index
    idx_t n_owned = 0;   // number of owned elements
    idx_t n_ghost = 0;   // number of remote elements referenced by owned elements

    // Owned elements. Connections are stored in the extended local index space:
    // [0, n_owned) owned, [n_owned, n_owned + n_ghost) ghost.
    std::vector<ElementStatic> elems;
    // Owned elements split into runs of consecutive local indices [begin, end):
    // interior runs have only owned neighbours, boundary runs have >= 1 ghost neighbour
    std::vector<IndexRange> interior;
    std::vector<IndexRange> boundary;

    // Halo exchange pattern
    std::vector<idx_t> ghost_global;          // sorted global ids of ghosts (grouped by owner)
    std::vector<int> recv_count, recv_disp;   // ghost values received from each rank
    std::vector<int> send_count, send_disp;   // owned values sent to each rank
    std::vector<idx_t> send_list;             // local owned indices to pack, grouped by destination
    std::vector<val_t> send_buf;

    // Double-buffered state: energy has n_owned + n_ghost entries, flux has n_owned.
    // Ghost slots of energy[b] are the receive targets for parity b.
    std::vector<val_t> energy[2];
    std::vector<val_t> flux[2];

    // Persistent communication requests
    std::vector<MPI_Request> send_reqs;
    std::vector<MPI_Request> recv_reqs[2];
};

// Set up halo exchange: determine send lists from peers' requests and create
// persistent requests. Must run after ghost_global / recv_count are known.
void setupHaloExchange(LocalMesh& m, int n_ranks, MPI_Comm comm) {
    m.recv_disp.assign(n_ranks, 0);
    for (int r = 1; r < n_ranks; ++r) m.recv_disp[r] = m.recv_disp[r - 1] + m.recv_count[r - 1];

    // Learn how many values each peer needs from us
    m.send_count.assign(n_ranks, 0);
    MPI_Alltoall(m.recv_count.data(), 1, MPI_INT, m.send_count.data(), 1, MPI_INT, comm);

    m.send_disp.assign(n_ranks, 0);
    for (int r = 1; r < n_ranks; ++r) m.send_disp[r] = m.send_disp[r - 1] + m.send_count[r - 1];
    const int total_send = m.send_disp[n_ranks - 1] + m.send_count[n_ranks - 1];

    // Exchange the global ids we need; receive the ids each peer needs from us
    std::vector<idx_t> requested(static_cast<size_t>(total_send));
    MPI_Alltoallv(m.ghost_global.data(), m.recv_count.data(), m.recv_disp.data(), MPI_UINT64_T,
                  requested.data(), m.send_count.data(), m.send_disp.data(), MPI_UINT64_T, comm);

    m.send_list.resize(static_cast<size_t>(total_send));
    for (size_t j = 0; j < m.send_list.size(); ++j) {
        m.send_list[j] = requested[j] - m.lo;
    }
    m.send_buf.assign(static_cast<size_t>(total_send), 0.0);

    m.energy[0].assign(static_cast<size_t>(m.n_owned + m.n_ghost), 0.0);
    m.energy[1].assign(static_cast<size_t>(m.n_owned + m.n_ghost), 0.0);
    m.flux[0].assign(static_cast<size_t>(m.n_owned), 0.0);
    m.flux[1].assign(static_cast<size_t>(m.n_owned), 0.0);

    // Persistent requests (buffers are fixed for the lifetime of the mesh)
    m.send_reqs.clear();
    for (int r = 0; r < n_ranks; ++r) {
        if (m.send_count[r] == 0) continue;
        MPI_Request req;
        MPI_Send_init(m.send_buf.data() + m.send_disp[r], m.send_count[r], MPI_DOUBLE,
                      r, 0, comm, &req);
        m.send_reqs.push_back(req);
    }
    for (int b = 0; b < 2; ++b) {
        m.recv_reqs[b].clear();
        for (int r = 0; r < n_ranks; ++r) {
            if (m.recv_count[r] == 0) continue;
            MPI_Request req;
            MPI_Recv_init(m.energy[b].data() + m.n_owned + m.recv_disp[r], m.recv_count[r],
                          MPI_DOUBLE, r, 0, comm, &req);
            m.recv_reqs[b].push_back(req);
        }
    }
}

// Build the owned part of the mesh and its communication pattern.
LocalMesh buildLocalMesh(idx_t n, idx_t n_elems, int rank, int n_ranks, MPI_Comm comm) {
    LocalMesh m;
    const idx_t lo = blockBegin(n_elems, n_ranks, rank);
    const idx_t hi = blockBegin(n_elems, n_ranks, rank + 1);
    m.lo = lo;
    m.n_owned = hi - lo;

    // Pass 1: collect remote neighbours (ghosts)
    idx_t nb[4];
    for (idx_t g = lo; g < hi; ++g) {
        const int cnt = globalNeighbors(n, g, nb);
        for (int c = 0; c < cnt; ++c) {
            if (nb[c] < lo || nb[c] >= hi) m.ghost_global.push_back(nb[c]);
        }
    }
    std::sort(m.ghost_global.begin(), m.ghost_global.end());
    m.ghost_global.erase(std::unique(m.ghost_global.begin(), m.ghost_global.end()),
                         m.ghost_global.end());
    m.n_ghost = static_cast<idx_t>(m.ghost_global.size());

    m.recv_count.assign(n_ranks, 0);
    for (idx_t g : m.ghost_global) ++m.recv_count[ownerOf(g, n_elems, n_ranks)];

    // Pass 2: build owned elements with extended local connectivity
    m.elems.resize(static_cast<size_t>(m.n_owned));
    for (idx_t k = 0; k < m.n_owned; ++k) {
        const idx_t g = lo + k;
        ElementStatic& es = m.elems[k];
        es.material_idx = materialOf(n, g);
        es.num_connections = 0;

        const int cnt = globalNeighbors(n, g, nb);
        bool has_ghost = false;
        for (int c = 0; c < cnt; ++c) {
            idx_t local;
            if (nb[c] >= lo && nb[c] < hi) {
                local = nb[c] - lo;
            } else {
                has_ghost = true;
                const auto it = std::lower_bound(m.ghost_global.begin(), m.ghost_global.end(), nb[c]);
                local = m.n_owned + static_cast<idx_t>(it - m.ghost_global.begin());
            }
            es.connected_idx[es.num_connections] = local;
            es.connected_flux[es.num_connections] = 1.0;
            es.num_connections++;
        }

        // Append to the current run of this class, or start a new one
        std::vector<IndexRange>& runs = has_ghost ? m.boundary : m.interior;
        if (!runs.empty() && runs.back().end == k) {
            runs.back().end = k + 1;
        } else {
            runs.push_back(IndexRange{k, k + 1});
        }
    }

    setupHaloExchange(m, n_ranks, comm);
    return m;
}

// Run the simulation. Each iteration: pack and send owned energies, update
// interior elements while messages are in flight, wait for ghosts, then
// update boundary elements. Returns the buffer index holding the final state.
int runSimulation(LocalMesh& m, const std::vector<Material>& materials, int n_iters) {
    const Material* mats = materials.data();
    const ElementStatic* elems = m.elems.data();
    const int n_send = static_cast<int>(m.send_reqs.size());

    for (int iter = 0; iter < n_iters; ++iter) {
        const int cur = iter & 1;
        const int nxt = cur ^ 1;
        const val_t* e_cur = m.energy[cur].data();
        val_t* e_nxt = m.energy[nxt].data();
        const val_t* f_cur = m.flux[cur].data();
        val_t* f_nxt = m.flux[nxt].data();

        // Pack owned energies required by peers and start the halo exchange
        const size_t n_pack = m.send_list.size();
        for (size_t j = 0; j < n_pack; ++j) {
            m.send_buf[j] = e_cur[m.send_list[j]];
        }
        if (n_send > 0) MPI_Startall(n_send, m.send_reqs.data());
        const int n_recv = static_cast<int>(m.recv_reqs[cur].size());
        if (n_recv > 0) MPI_Startall(n_recv, m.recv_reqs[cur].data());

        // Interior elements depend only on owned data: overlap with communication
        for (const IndexRange& r : m.interior) {
            for (idx_t i = r.begin; i < r.end; ++i) {
                updateElement(i, elems[i], mats, e_cur, e_nxt, f_cur, f_nxt);
            }
        }

        if (n_send > 0) MPI_Waitall(n_send, m.send_reqs.data(), MPI_STATUSES_IGNORE);
        if (n_recv > 0) MPI_Waitall(n_recv, m.recv_reqs[cur].data(), MPI_STATUSES_IGNORE);

        // Boundary elements need ghost energies from neighbouring ranks
        for (const IndexRange& r : m.boundary) {
            for (idx_t i = r.begin; i < r.end; ++i) {
                updateElement(i, elems[i], mats, e_cur, e_nxt, f_cur, f_nxt);
            }
        }
    }

    return n_iters > 0 ? (n_iters & 1) : 0;
}

// Validate simulation results (rank 0 only)
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

int runBenchmark(int argc, char** argv, int rank, int n_ranks) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const bool root = (rank == 0);

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
        if (root) printf("Grid size must be at least 1\n");
        return 1;
    }

    const idx_t n = static_cast<idx_t>(n_elems_root);
    const idx_t n_elems = n * n;

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %llu elements\n", n_elems_root, n_elems_root,
               static_cast<unsigned long long>(n_elems));
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", n_ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        printf("Building unstructured mesh...\n");
    }

    const std::vector<Material> materials = makeMaterials();
    LocalMesh mesh = buildLocalMesh(n, n_elems, rank, n_ranks, comm);

    if (root) {
        // Memory usage of the (global) mesh representation
        const double static_mem = static_cast<double>(n_elems) * sizeof(ElementStatic);
        const double dynamic_mem = static_cast<double>(n_elems) * sizeof(ElementDynamic) * 2;
        const double total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        printf("Running simulation...\n");
    }

    // Run simulation (timed across all ranks; reported time is the slowest rank)
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();
    const int final_buf = runSimulation(mesh, materials, n_iters);
    const double local_elapsed = MPI_Wtime() - t0;

    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    const long long duration_ms = static_cast<long long>(elapsed * 1000.0);

    if (root) {
        printf("Computation time: %lld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * static_cast<double>(n_elems)) /
            (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Gather final state on rank 0 in global element order
    std::vector<int> counts(n_ranks), displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        const idx_t b = blockBegin(n_elems, n_ranks, r);
        const idx_t e = blockBegin(n_elems, n_ranks, r + 1);
        counts[r] = static_cast<int>(e - b);
        displs[r] = static_cast<int>(b);
    }
    std::vector<val_t> all_energy, all_flux;
    if (root) {
        all_energy.resize(static_cast<size_t>(n_elems));
        all_flux.resize(static_cast<size_t>(n_elems));
    }
    MPI_Gatherv(mesh.energy[final_buf].data(), static_cast<int>(mesh.n_owned), MPI_DOUBLE,
                all_energy.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(mesh.flux[final_buf].data(), static_cast<int>(mesh.n_owned), MPI_DOUBLE,
                all_flux.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

    int rc = 0;
    if (root) {
        std::vector<ElementDynamic> elements(static_cast<size_t>(n_elems));
        for (size_t i = 0; i < elements.size(); ++i) {
            elements[i].current_energy = all_energy[i];
            elements[i].total_flux = all_flux[i];
        }

        // Compute hash for verification
        const uint64_t hash = computeHash(elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(elements.size());
            for (const auto& elem : elements) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate && !validateResults(elements)) {
            rc = 1;
        }
    }

    // All ranks report the same exit status
    MPI_Bcast(&rc, 1, MPI_INT, 0, comm);

    for (auto& req : mesh.send_reqs) MPI_Request_free(&req);
    for (int b = 0; b < 2; ++b) {
        for (auto& req : mesh.recv_reqs[b]) MPI_Request_free(&req);
    }

    return rc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    const int rc = runBenchmark(argc, argv, rank, n_ranks);

    MPI_Finalize();
    return rc;
}
