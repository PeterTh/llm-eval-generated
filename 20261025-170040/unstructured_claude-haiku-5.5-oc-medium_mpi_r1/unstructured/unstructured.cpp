#include <mpi.h>

#include <algorithm>
#include <cinttypes>
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

// Maximum number of connections per element (for a 2D grid: 4 neighbors).
// Used only to report the logical per-element footprint, as in the serial benchmark.
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Logical static connectivity record (reporting only; the solver uses a CSR layout)
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
static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t), "ElementDynamic must be two packed doubles");

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Block distribution of the global element range [0, n) over MPI ranks.
// The first `rem` ranks own `base + 1` elements, the rest own `base`.
struct Partition {
    idx_t n;
    idx_t base;
    idx_t rem;

    Partition(idx_t n_elems, int nranks)
        : n(n_elems), base(n_elems / static_cast<idx_t>(nranks)), rem(n_elems % static_cast<idx_t>(nranks)) {}

    idx_t begin(int r) const { return static_cast<idx_t>(r) * base + std::min<idx_t>(static_cast<idx_t>(r), rem); }
    idx_t end(int r) const { return begin(r) + base + (static_cast<idx_t>(r) < rem ? 1 : 0); }

    int owner(idx_t g) const {
        const idx_t boundary = rem * (base + 1);
        return g < boundary ? static_cast<int>(g / (base + 1))
                            : static_cast<int>(rem + (g - boundary) / base);
    }
};

// Global grid neighbours of element g, in the same order as the serial builder
// (+x, -x, +y, -y). Returns the number of in-bounds neighbours written to out.
inline int gridNeighbors(idx_t g, int n_root, idx_t out[4]) {
    const int x = static_cast<int>(g / static_cast<idx_t>(n_root));
    const int y = static_cast<int>(g % static_cast<idx_t>(n_root));
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int count = 0;
    for (int n = 0; n < 4; ++n) {
        const int nx = x + offsets[n][0];
        const int ny = y + offsets[n][1];
        if (nx >= 0 && nx < n_root && ny >= 0 && ny < n_root) {
            out[count++] = static_cast<idx_t>(nx) * static_cast<idx_t>(n_root) + static_cast<idx_t>(ny);
        }
    }
    return count;
}

// Material of global element g (corners are inflow/outflow, applied in the serial order)
inline idx_t materialOf(idx_t g, int n_root) {
    const idx_t n = static_cast<idx_t>(n_root);
    const idx_t last = n - 1;
    idx_t mat = DEFAULT_MAT_ID;
    if (g == 0) mat = INFLOW_MAT_ID;                  // (0, 0)
    if (g == last) mat = OUTFLOW_MAT_ID;              // (0, last)
    if (g == last * n) mat = OUTFLOW_MAT_ID;          // (last, 0)
    if (g == last * n + last) mat = INFLOW_MAT_ID;    // (last, last)
    return mat;
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, val_t this_energy, val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// Rank-local portion of the mesh plus its halo-exchange plan.
//
// Energy storage per rank is a single array of length n_local + n_ghost:
//   [0, n_local)                 owned elements, in global order
//   [n_local, n_local + n_ghost) copies of remote neighbour energies, grouped by owner rank
struct LocalMesh {
    idx_t lo = 0;                 // first owned global index
    idx_t hi = 0;                 // one past last owned global index
    size_t n_local = 0;
    size_t n_ghost = 0;

    std::vector<Material> materials;
    std::vector<idx_t> material_idx;   // per owned element
    std::vector<size_t> conn_begin;    // CSR offsets, size n_local + 1
    std::vector<size_t> conn_slot;     // index into the energy array
    std::vector<val_t> conn_flux;      // connection flux coefficient

    // Owned elements in [boundary_lo, boundary_hi) are treated as boundary (computed after the
    // halo arrives); everything outside that range has no ghost neighbours and is computed
    // while messages are in flight. For an empty boundary both are n_local / 0.
    size_t boundary_lo = 0;
    size_t boundary_hi = 0;

    // Halo plan (per rank)
    std::vector<int> recv_count;       // number of ghost values received from each rank
    std::vector<int> recv_disp;        // offset of those values in the ghost region
    std::vector<int> send_count;       // number of owned values sent to each rank
    std::vector<int> send_disp;        // offset of those values in the send buffer
    std::vector<size_t> send_local;    // owned local indices to pack, grouped by destination
    std::vector<int> recv_peers;       // ranks with recv_count > 0
    std::vector<int> send_peers;       // ranks with send_count > 0
};

// Build this rank's elements, connectivity, and halo exchange plan.
void buildLocalMesh(LocalMesh& m, int n_root, const Partition& part, int rank, int nranks, MPI_Comm comm) {
    m.lo = part.begin(rank);
    m.hi = part.end(rank);
    m.n_local = static_cast<size_t>(m.hi - m.lo);

    m.materials = {
        Material{0.8, 0.0},    // Default material
        Material{0.8, 0.5},    // Inflow material
        Material{0.8, -0.5},   // Outflow material
    };

    // Pass 1: collect remote neighbours (sorted, unique => grouped by owner ascending)
    std::vector<idx_t> ghosts;
    idx_t nb[4];
    for (idx_t g = m.lo; g < m.hi; ++g) {
        const int cnt = gridNeighbors(g, n_root, nb);
        for (int k = 0; k < cnt; ++k) {
            if (nb[k] < m.lo || nb[k] >= m.hi) ghosts.push_back(nb[k]);
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    m.n_ghost = ghosts.size();

    // Receive plan: ghost values from each owner form a contiguous block
    m.recv_count.assign(nranks, 0);
    m.recv_disp.assign(nranks, 0);
    for (idx_t g : ghosts) m.recv_count[part.owner(g)]++;
    for (int r = 1; r < nranks; ++r) m.recv_disp[r] = m.recv_disp[r - 1] + m.recv_count[r - 1];
    for (int r = 0; r < nranks; ++r) {
        if (m.recv_count[r] > 0) m.recv_peers.push_back(r);
    }

    // Pass 2: CSR connectivity in the serial neighbour order
    m.material_idx.resize(m.n_local);
    m.conn_begin.resize(m.n_local + 1);
    m.conn_slot.reserve(m.n_local * 4);
    m.conn_flux.reserve(m.n_local * 4);
    size_t first_boundary = m.n_local;
    size_t last_boundary = 0;
    bool any_boundary = false;
    for (size_t i = 0; i < m.n_local; ++i) {
        const idx_t g = m.lo + i;
        m.material_idx[i] = materialOf(g, n_root);
        m.conn_begin[i] = m.conn_slot.size();
        const int cnt = gridNeighbors(g, n_root, nb);
        bool touches_ghost = false;
        for (int k = 0; k < cnt; ++k) {
            size_t slot;
            if (nb[k] >= m.lo && nb[k] < m.hi) {
                slot = static_cast<size_t>(nb[k] - m.lo);
            } else {
                const auto it = std::lower_bound(ghosts.begin(), ghosts.end(), nb[k]);
                slot = m.n_local + static_cast<size_t>(it - ghosts.begin());
                touches_ghost = true;
            }
            m.conn_slot.push_back(slot);
            m.conn_flux.push_back(1.0);
        }
        if (touches_ghost) {
            if (!any_boundary) first_boundary = i;
            last_boundary = i;
            any_boundary = true;
        }
    }
    m.conn_begin[m.n_local] = m.conn_slot.size();
    if (any_boundary) {
        m.boundary_lo = first_boundary;
        m.boundary_hi = last_boundary + 1;
    } else {
        m.boundary_lo = m.n_local;
        m.boundary_hi = m.n_local;
    }

    // Exchange requests: tell each owner which of its elements we need
    m.send_count.assign(nranks, 0);
    MPI_Alltoall(m.recv_count.data(), 1, MPI_INT, m.send_count.data(), 1, MPI_INT, comm);

    m.send_disp.assign(nranks, 0);
    for (int r = 1; r < nranks; ++r) m.send_disp[r] = m.send_disp[r - 1] + m.send_count[r - 1];
    const int total_send = m.send_disp[nranks - 1] + m.send_count[nranks - 1];

    std::vector<idx_t> requested(static_cast<size_t>(total_send));
    MPI_Alltoallv(ghosts.data(), m.recv_count.data(), m.recv_disp.data(), MPI_UINT64_T,
                  requested.data(), m.send_count.data(), m.send_disp.data(), MPI_UINT64_T, comm);

    m.send_local.resize(requested.size());
    for (size_t k = 0; k < requested.size(); ++k) {
        m.send_local[k] = static_cast<size_t>(requested[k] - m.lo);
    }
    for (int r = 0; r < nranks; ++r) {
        if (m.send_count[r] > 0) m.send_peers.push_back(r);
    }
}

// Run n_iters iterations of the energy-transfer update on this rank's elements.
// Halo energies are exchanged each iteration; interior elements are computed
// while messages are in flight.
void runSimulation(const LocalMesh& m, std::vector<val_t>& cur, std::vector<val_t>& nxt,
                   std::vector<val_t>& tot, std::vector<val_t>& send_buf, int n_iters, MPI_Comm comm) {
    std::vector<MPI_Request> reqs(m.recv_peers.size() + m.send_peers.size());

    const size_t* conn_begin = m.conn_begin.data();
    const size_t* conn_slot = m.conn_slot.data();
    const val_t* conn_flux = m.conn_flux.data();
    const idx_t* material_idx = m.material_idx.data();
    const Material* materials = m.materials.data();
    const size_t* send_local = m.send_local.data();

    // Update elements [begin, end) of the owned range; reads cur (owned + ghost), writes nxt/tot
    auto updateRange = [&](size_t begin, size_t end, const val_t* __restrict energy,
                           val_t* __restrict energy_out, val_t* __restrict flux_acc) {
        for (size_t i = begin; i < end; ++i) {
            const Material& mat = materials[material_idx[i]];
            const val_t self = energy[i];

            // Start with external flow, then add fluxes from all connections (same order as serial)
            val_t total_flux = mat.external_flow;
            for (size_t j = conn_begin[i]; j < conn_begin[i + 1]; ++j) {
                total_flux += computeFlux(mat, self, conn_flux[j], energy[conn_slot[j]]);
            }

            energy_out[i] = self + total_flux;
            flux_acc[i] = flux_acc[i] + std::abs(total_flux);
        }
    };

    const size_t n_local = m.n_local;
    const size_t b_lo = m.boundary_lo;
    const size_t b_hi = m.boundary_hi;

    for (int iter = 0; iter < n_iters; ++iter) {
        size_t r = 0;

        // Post halo receives directly into the ghost region of the current buffer
        for (int p : m.recv_peers) {
            MPI_Irecv(cur.data() + n_local + m.recv_disp[p], m.recv_count[p], MPI_DOUBLE, p, 0, comm, &reqs[r++]);
        }

        // Pack and send owned energies needed by peers
        const size_t total_send = m.send_local.size();
        for (size_t k = 0; k < total_send; ++k) {
            send_buf[k] = cur[send_local[k]];
        }
        for (int p : m.send_peers) {
            MPI_Isend(send_buf.data() + m.send_disp[p], m.send_count[p], MPI_DOUBLE, p, 0, comm, &reqs[r++]);
        }

        // Overlap: elements with no ghost neighbours depend only on owned data
        updateRange(0, b_lo, cur.data(), nxt.data(), tot.data());
        updateRange(b_hi, n_local, cur.data(), nxt.data(), tot.data());

        MPI_Waitall(static_cast<int>(r), reqs.data(), MPI_STATUSES_IGNORE);

        // Boundary elements need the halo
        updateRange(b_lo, b_hi, cur.data(), nxt.data(), tot.data());

        // Swap buffers
        std::swap(cur, nxt);
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

int runBenchmark(int argc, char** argv) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    const bool root = (rank == 0);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks agree on early exits)
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

    if (n_elems_root <= 0) {
        if (root) printf("Grid size must be positive\n");
        return 1;
    }

    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);
    if (n_elems > static_cast<idx_t>(std::numeric_limits<int>::max())) {
        if (root) printf("Grid size too large for MPI gather counts\n");
        return 1;
    }

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        printf("Building unstructured mesh...\n");
    }

    // Build this rank's share of the unstructured mesh
    const Partition part(n_elems, nranks);
    LocalMesh mesh;
    buildLocalMesh(mesh, n_elems_root, part, rank, nranks, comm);

    std::vector<val_t> cur(mesh.n_local + mesh.n_ghost, 0.0);
    std::vector<val_t> nxt(mesh.n_local + mesh.n_ghost, 0.0);
    std::vector<val_t> tot(mesh.n_local, 0.0);
    std::vector<val_t> send_buf(mesh.send_local.size());

    if (root) {
        // Calculate memory usage (logical footprint of the full mesh, as in the serial benchmark)
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

    // Run simulation (timed across all ranks)
    MPI_Barrier(comm);
    const double t_start = MPI_Wtime();

    runSimulation(mesh, cur, nxt, tot, send_buf, n_iters, comm);

    MPI_Barrier(comm);
    const double t_end = MPI_Wtime();

    // Gather final state in global element order on rank 0
    std::vector<ElementDynamic> local_out(mesh.n_local);
    for (size_t i = 0; i < mesh.n_local; ++i) {
        local_out[i].current_energy = cur[i];
        local_out[i].total_flux = tot[i];
    }

    std::vector<ElementDynamic> world_dyn;
    std::vector<int> counts;
    std::vector<int> displs;
    if (root) {
        world_dyn.resize(static_cast<size_t>(n_elems));
        counts.resize(nranks);
        displs.resize(nranks);
        for (int r = 0; r < nranks; ++r) {
            counts[r] = static_cast<int>(part.end(r) - part.begin(r));
            displs[r] = static_cast<int>(part.begin(r));
        }
    }

    MPI_Datatype pair_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &pair_type);
    MPI_Type_commit(&pair_type);
    MPI_Gatherv(local_out.data(), static_cast<int>(mesh.n_local), pair_type,
                world_dyn.data(), counts.data(), displs.data(), pair_type, 0, comm);
    MPI_Type_free(&pair_type);

    int rc = 0;
    if (root) {
        const double elapsed_s = t_end - t_start;
        const auto duration_ms = static_cast<long>(elapsed_s * 1000.0);

        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9;

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

    // Propagate the outcome so every rank exits with the same status
    MPI_Bcast(&rc, 1, MPI_INT, 0, comm);
    return rc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = runBenchmark(argc, argv);
    MPI_Finalize();
    return rc;
}
