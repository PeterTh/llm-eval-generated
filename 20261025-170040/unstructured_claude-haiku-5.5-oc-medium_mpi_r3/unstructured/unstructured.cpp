#include <mpi.h>

#include <algorithm>
#include <climits>
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

// Static connectivity information for each element.
// connected_idx holds LOCAL indices: [0, count) are owned elements and
// [count, count + n_ghost) are halo copies of elements owned by other ranks.
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element (global result layout)
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Contiguous block partition of the global element index space.
// Rank r owns global indices [starts[r], starts[r+1]).
struct Partition {
    std::vector<idx_t> starts;

    Partition(idx_t n_elems, int nranks) : starts(static_cast<size_t>(nranks) + 1) {
        for (int r = 0; r <= nranks; ++r) {
            starts[r] = (n_elems * static_cast<idx_t>(r)) / static_cast<idx_t>(nranks);
        }
    }

    int owner(idx_t g) const {
        return static_cast<int>(std::upper_bound(starts.begin(), starts.end(), g) - starts.begin()) - 1;
    }
};

// Neighbors of global element g in an n x n grid, in the same order as the reference
// (+x, -x, +y, -y), restricted to in-bounds positions.
inline int gridNeighbors(idx_t g, int n, idx_t out[4]) {
    const int x = static_cast<int>(g / static_cast<idx_t>(n));
    const int y = static_cast<int>(g % static_cast<idx_t>(n));
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int count = 0;
    for (int k = 0; k < 4; ++k) {
        const int nx = x + offsets[k][0];
        const int ny = y + offsets[k][1];
        if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
            out[count++] = static_cast<idx_t>(nx) * static_cast<idx_t>(n) + static_cast<idx_t>(ny);
        }
    }
    return count;
}

// Material of global element g. Corner assignments are applied in the same
// order as the reference, so later assignments win when corners coincide (n == 1).
inline idx_t materialIdFor(idx_t g, int n) {
    const idx_t N = static_cast<idx_t>(n);
    const idx_t last = N - 1;
    idx_t mat = DEFAULT_MAT_ID;
    if (g == 0) mat = INFLOW_MAT_ID;                  // (0, 0)
    if (g == last) mat = OUTFLOW_MAT_ID;              // (0, last)
    if (g == last * N) mat = OUTFLOW_MAT_ID;          // (last, 0)
    if (g == last * N + last) mat = INFLOW_MAT_ID;    // (last, last)
    return mat;
}

// Local (per-rank) portion of the mesh and its solver state
struct LocalMesh {
    idx_t begin = 0;       // first owned global index
    idx_t count = 0;       // number of owned elements
    idx_t n_ghost = 0;     // number of halo elements

    std::vector<ElementStatic> elements_static;  // [count]
    std::vector<idx_t> interior;                 // owned elements with no halo neighbor
    std::vector<idx_t> boundary;                 // owned elements with >= 1 halo neighbor

    // Double-buffered state. Energy buffers include the halo tail; flux is owned only.
    std::vector<val_t> energy[2];  // [count + n_ghost]
    std::vector<val_t> flux[2];    // [count]
    int read = 0;

    // Halo exchange
    MPI_Comm graph = MPI_COMM_NULL;
    std::vector<idx_t> send_local;   // owned local indices to send, grouped by destination
    std::vector<val_t> send_buf;     // packed energies for sending
    std::vector<int> send_counts;    // per destination neighbor
    std::vector<int> send_displs;    // per destination neighbor (into send_buf)
    std::vector<int> recv_counts;    // per source neighbor
    std::vector<int> recv_displs;    // per source neighbor (into halo region)
};

// Build the local mesh for `rank`, including halo layout and communication pattern.
void buildLocalMesh(LocalMesh& m, int n, const Partition& part, int rank, int nranks) {
    m.begin = part.starts[rank];
    const idx_t end = part.starts[rank + 1];
    m.count = end - m.begin;

    // Collect off-rank neighbors (halo), sorted and unique.
    std::vector<idx_t> ghosts;
    for (idx_t g = m.begin; g < end; ++g) {
        idx_t nb[4];
        const int k = gridNeighbors(g, n, nb);
        for (int j = 0; j < k; ++j) {
            if (nb[j] < m.begin || nb[j] >= end) ghosts.push_back(nb[j]);
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    m.n_ghost = static_cast<idx_t>(ghosts.size());

    // Static connectivity in local indexing
    m.elements_static.resize(m.count);
    for (idx_t l = 0; l < m.count; ++l) {
        const idx_t g = m.begin + l;
        ElementStatic& es = m.elements_static[l];
        es.material_idx = materialIdFor(g, n);
        es.num_connections = 0;

        idx_t nb[4];
        const int k = gridNeighbors(g, n, nb);
        bool has_halo = false;
        for (int j = 0; j < k; ++j) {
            const idx_t gn = nb[j];
            idx_t ln;
            if (gn >= m.begin && gn < end) {
                ln = gn - m.begin;
            } else {
                const auto it = std::lower_bound(ghosts.begin(), ghosts.end(), gn);
                ln = m.count + static_cast<idx_t>(it - ghosts.begin());
                has_halo = true;
            }
            es.connected_idx[es.num_connections] = ln;
            es.connected_flux[es.num_connections] = 1.0;
            es.num_connections++;
        }
        if (has_halo) {
            m.boundary.push_back(l);
        } else {
            m.interior.push_back(l);
        }
    }

    // State buffers (zero-initialized, as in the reference)
    for (int b = 0; b < 2; ++b) {
        m.energy[b].assign(static_cast<size_t>(m.count + m.n_ghost), 0.0);
        m.flux[b].assign(static_cast<size_t>(m.count), 0.0);
    }

    // Receive side: ghosts are sorted, and owners are contiguous ascending ranges,
    // so each owner's halo block is contiguous and ordered by rank.
    std::vector<int> rcnt(nranks, 0), rdsp(nranks, 0);
    for (idx_t gh : ghosts) rcnt[part.owner(gh)]++;
    for (int r = 1; r < nranks; ++r) rdsp[r] = rdsp[r - 1] + rcnt[r - 1];

    // Send side: learn what every other rank needs from this rank.
    std::vector<int> scnt(nranks, 0), sdsp(nranks, 0);
    MPI_Alltoall(rcnt.data(), 1, MPI_INT, scnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int r = 1; r < nranks; ++r) sdsp[r] = sdsp[r - 1] + scnt[r - 1];
    const int total_send = nranks > 0 ? sdsp[nranks - 1] + scnt[nranks - 1] : 0;

    std::vector<idx_t> requests(static_cast<size_t>(total_send));
    MPI_Alltoallv(ghosts.data(), rcnt.data(), rdsp.data(), MPI_UINT64_T,
                  requests.data(), scnt.data(), sdsp.data(), MPI_UINT64_T, MPI_COMM_WORLD);

    m.send_local.resize(static_cast<size_t>(total_send));
    for (int k = 0; k < total_send; ++k) {
        m.send_local[k] = requests[k] - m.begin;  // requested elements are owned by this rank
    }
    m.send_buf.assign(static_cast<size_t>(total_send), 0.0);

    // Neighbor lists for the distributed-graph communicator (ascending rank order).
    std::vector<int> srcs, dsts;
    for (int r = 0; r < nranks; ++r) {
        if (rcnt[r] > 0) {
            srcs.push_back(r);
            m.recv_counts.push_back(rcnt[r]);
            m.recv_displs.push_back(rdsp[r]);
        }
    }
    for (int r = 0; r < nranks; ++r) {
        if (scnt[r] > 0) {
            dsts.push_back(r);
            m.send_counts.push_back(scnt[r]);
            m.send_displs.push_back(sdsp[r]);
        }
    }
    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                   static_cast<int>(srcs.size()), srcs.data(), MPI_UNWEIGHTED,
                                   static_cast<int>(dsts.size()), dsts.data(), MPI_UNWEIGHTED,
                                   MPI_INFO_NULL, 0, &m.graph);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, val_t this_energy,
                         val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update one owned element from the read buffers into the write buffers
inline void updateElement(idx_t l, const ElementStatic* statics, const Material* materials,
                          const val_t* E, const val_t* F, val_t* EN, val_t* FN) {
    const ElementStatic& es = statics[l];
    const Material& mat = materials[es.material_idx];
    const val_t self = E[l];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements (owned or halo)
    for (idx_t j = 0; j < es.num_connections; ++j) {
        total_flux += computeFlux(mat, self, es.connected_flux[j], E[es.connected_idx[j]]);
    }

    EN[l] = self + total_flux;
    FN[l] = F[l] + std::abs(total_flux);
}

// Run simulation for n_iters iterations (halo exchange overlapped with interior work)
void runSimulation(LocalMesh& m, const std::vector<Material>& materials, const int n_iters) {
    const ElementStatic* statics = m.elements_static.data();
    const Material* mats = materials.data();

    for (int iter = 0; iter < n_iters; ++iter) {
        const int rd = m.read;
        const int wr = 1 - rd;

        val_t* E = m.energy[rd].data();
        val_t* EN = m.energy[wr].data();
        const val_t* F = m.flux[rd].data();
        val_t* FN = m.flux[wr].data();

        // Pack boundary energies and start halo exchange; received values land in the halo tail
        for (size_t k = 0; k < m.send_local.size(); ++k) {
            m.send_buf[k] = E[m.send_local[k]];
        }
        MPI_Request req;
        MPI_Ineighbor_alltoallv(m.send_buf.data(), m.send_counts.data(), m.send_displs.data(), MPI_DOUBLE,
                                E + m.count, m.recv_counts.data(), m.recv_displs.data(), MPI_DOUBLE,
                                m.graph, &req);

        // Interior elements do not depend on halo values
        for (idx_t l : m.interior) {
            updateElement(l, statics, mats, E, F, EN, FN);
        }

        MPI_Wait(&req, MPI_STATUS_IGNORE);

        // Boundary elements need halo values
        for (idx_t l : m.boundary) {
            updateElement(l, statics, mats, E, F, EN, FN);
        }

        m.read = wr;
    }
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

static int runBenchmark(int argc, char** argv, int rank, int nranks) {
    const bool root = (rank == 0);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks agree on exit paths)
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

    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);
    if (n_elems > (idx_t(1) << 30)) {
        if (root) printf("Grid size too large\n");
        return 1;
    }

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, static_cast<int>(n_elems));
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nranks);
        printf("\n");

        printf("Building unstructured mesh...\n");
    }

    // Partition and build this rank's portion of the mesh
    const Partition part(n_elems, nranks);
    LocalMesh mesh;
    buildLocalMesh(mesh, n_elems_root, part, rank, nranks);

    const std::vector<Material> materials = {
        Material{0.8, 0.0},    // Default material
        Material{0.8, 0.5},    // Inflow material
        Material{0.8, -0.5},   // Outflow material
    };

    if (root) {
        // Calculate memory usage (global problem size)
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        printf("Running simulation...\n");
    }

    // Run simulation; time is the slowest rank's wall time
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    runSimulation(mesh, materials, n_iters);
    const double local_elapsed = MPI_Wtime() - t0;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the final global state on rank 0 (owned ranges are contiguous and ordered by rank)
    std::vector<ElementDynamic> local_dyn(static_cast<size_t>(mesh.count));
    const val_t* E_final = mesh.energy[mesh.read].data();
    const val_t* F_final = mesh.flux[mesh.read].data();
    for (idx_t l = 0; l < mesh.count; ++l) {
        local_dyn[l].current_energy = E_final[l];
        local_dyn[l].total_flux = F_final[l];
    }
    std::vector<ElementDynamic> world_dyn;
    std::vector<int> gcnt(nranks), gdsp(nranks);
    for (int r = 0; r < nranks; ++r) {
        gcnt[r] = static_cast<int>(2 * (part.starts[r + 1] - part.starts[r]));
        gdsp[r] = static_cast<int>(2 * part.starts[r]);
    }
    if (root) world_dyn.resize(static_cast<size_t>(n_elems));
    MPI_Gatherv(local_dyn.data(), static_cast<int>(2 * mesh.count), MPI_DOUBLE,
                root ? world_dyn.data() : nullptr, gcnt.data(), gdsp.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int rc = 0;
    if (root) {
        const long duration_ms = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

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

        // Compute hash for verification
        const uint64_t hash = computeHash(world_dyn);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(world_dyn.size());
            for (const auto& elem : world_dyn) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            const bool valid = validateResults(world_dyn);
            if (!valid) rc = 1;
        }
    }

    // Share the exit status so every rank returns the same code
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&mesh.graph);
    return rc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const int rc = runBenchmark(argc, argv, rank, nranks);

    MPI_Finalize();
    return rc;
}
