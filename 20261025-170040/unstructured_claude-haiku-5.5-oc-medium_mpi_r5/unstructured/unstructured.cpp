#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
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

// Static connectivity information for each element (logical per-element layout;
// used for reporting the problem's memory footprint)
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
constexpr uint8_t DEFAULT_MAT_ID = 0;
constexpr uint8_t INFLOW_MAT_ID = 1;
constexpr uint8_t OUTFLOW_MAT_ID = 2;

// Distributed mesh partition. Global element indices are split into contiguous
// blocks, one per MPI rank. Each rank stores only its owned elements plus ghost
// copies of remote neighbours referenced by its elements.
//
// Owned elements are stored in local order: first the interior elements (all
// neighbours owned locally), then the boundary elements (at least one ghost
// neighbour). This makes the update kernel two contiguous loops.
struct Mesh {
    idx_t lo = 0;                 // first owned global index
    idx_t hi = 0;                 // one past last owned global index
    idx_t n_local = 0;            // number of owned elements
    idx_t n_ghost = 0;            // number of ghost elements
    idx_t n_interior = 0;         // local ids [0, n_interior) are interior

    std::vector<uint8_t> material_idx;    // per local element
    std::vector<uint8_t> num_conn;        // per local element
    std::vector<uint32_t> conn_slot;      // per local element * MAX_CONNECTIONS: local or ghost slot
    std::vector<val_t> conn_flux;         // per local element * MAX_CONNECTIONS

    std::vector<idx_t> local_of_gid;      // local id of owned element with global index lo + j

    std::vector<idx_t> ghost_gid;         // sorted global indices of ghost elements

    // Communication plan (grouped by peer rank)
    std::vector<int> recv_cnt, recv_disp;     // per rank: ghosts received from it
    std::vector<int> send_cnt, send_disp;     // per rank: owned values sent to it
    std::vector<uint32_t> send_local;         // local ids to pack, grouped by destination
    std::vector<int> recv_ranks, send_ranks;  // peers with nonzero counts
};

// Global index where each rank's contiguous block starts (starts[nprocs] == n_total)
std::vector<idx_t> computePartitionStarts(idx_t n_total, int nprocs) {
    std::vector<idx_t> starts(nprocs + 1);
    const idx_t base = n_total / static_cast<idx_t>(nprocs);
    const idx_t rem = n_total % static_cast<idx_t>(nprocs);
    for (int r = 0; r <= nprocs; ++r) {
        const idx_t ur = static_cast<idx_t>(r);
        starts[r] = ur * base + std::min(ur, rem);
    }
    return starts;
}

// Rank that owns global element g
int ownerOf(const std::vector<idx_t>& starts, idx_t g) {
    const auto it = std::upper_bound(starts.begin(), starts.end(), g);
    return static_cast<int>(it - starts.begin()) - 1;
}

// Compute the flux for one connection (same arithmetic as the serial version)
inline val_t computeFlux(const Material& mat, val_t this_energy,
                         val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// Build this rank's portion of the 2D square grid unstructured mesh, including
// the communication plan needed for ghost exchange.
void buildLocalMesh(Mesh& m, const int n_elems_root, const std::vector<idx_t>& starts,
                    int rank, int nprocs, MPI_Comm comm) {
    const idx_t n = static_cast<idx_t>(n_elems_root);
    const idx_t last = n - 1;

    m.lo = starts[rank];
    m.hi = starts[rank + 1];
    m.n_local = m.hi - m.lo;
    const idx_t nloc = m.n_local;

    // Connectivity in global-order temporaries
    std::vector<uint8_t> mat_tmp(nloc, DEFAULT_MAT_ID);
    std::vector<uint8_t> nconn_tmp(nloc, 0);
    std::vector<idx_t> gid_tmp(nloc * MAX_CONNECTIONS, 0);
    std::vector<val_t> flux_tmp(nloc * MAX_CONNECTIONS, 0.0);

    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (idx_t i = 0; i < nloc; ++i) {
        const idx_t g = m.lo + i;
        const idx_t x = g / n;
        const idx_t y = g % n;

        // Corner materials: applied in the same sequence as the serial code
        uint8_t mat = DEFAULT_MAT_ID;
        if (g == 0) mat = INFLOW_MAT_ID;
        if (g == last) mat = OUTFLOW_MAT_ID;
        if (g == last * n) mat = OUTFLOW_MAT_ID;
        if (g == last * n + last) mat = INFLOW_MAT_ID;
        mat_tmp[i] = mat;

        int c = 0;
        for (int k = 0; k < 4; ++k) {
            const long long nx = static_cast<long long>(x) + offsets[k][0];
            const long long ny = static_cast<long long>(y) + offsets[k][1];
            if (nx >= 0 && nx < static_cast<long long>(n) &&
                ny >= 0 && ny < static_cast<long long>(n)) {
                gid_tmp[i * MAX_CONNECTIONS + c] = static_cast<idx_t>(nx) * n + static_cast<idx_t>(ny);
                flux_tmp[i * MAX_CONNECTIONS + c] = 1.0;
                ++c;
            }
        }
        nconn_tmp[i] = static_cast<uint8_t>(c);
    }

    // Ghost set: remote neighbours referenced by owned elements
    std::vector<idx_t> ghosts;
    std::vector<uint8_t> has_ghost(nloc, 0);
    for (idx_t i = 0; i < nloc; ++i) {
        for (int c = 0; c < nconn_tmp[i]; ++c) {
            const idx_t gid = gid_tmp[i * MAX_CONNECTIONS + c];
            if (gid < m.lo || gid >= m.hi) {
                ghosts.push_back(gid);
                has_ghost[i] = 1;
            }
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    m.ghost_gid = ghosts;
    m.n_ghost = static_cast<idx_t>(ghosts.size());

    // Local ordering: interior elements first, then boundary (each in global order)
    m.local_of_gid.assign(nloc, 0);
    idx_t next_id = 0;
    for (idx_t i = 0; i < nloc; ++i) {
        if (!has_ghost[i]) m.local_of_gid[i] = next_id++;
    }
    m.n_interior = next_id;
    for (idx_t i = 0; i < nloc; ++i) {
        if (has_ghost[i]) m.local_of_gid[i] = next_id++;
    }

    m.material_idx.assign(nloc, DEFAULT_MAT_ID);
    m.num_conn.assign(nloc, 0);
    m.conn_slot.assign(nloc * MAX_CONNECTIONS, 0);
    m.conn_flux.assign(nloc * MAX_CONNECTIONS, 0.0);
    for (idx_t i = 0; i < nloc; ++i) {
        const idx_t k = m.local_of_gid[i];
        m.material_idx[k] = mat_tmp[i];
        m.num_conn[k] = nconn_tmp[i];
        for (int c = 0; c < nconn_tmp[i]; ++c) {
            const idx_t gid = gid_tmp[i * MAX_CONNECTIONS + c];
            idx_t slot;
            if (gid >= m.lo && gid < m.hi) {
                slot = m.local_of_gid[gid - m.lo];
            } else {
                const auto it = std::lower_bound(ghosts.begin(), ghosts.end(), gid);
                slot = nloc + static_cast<idx_t>(it - ghosts.begin());
            }
            m.conn_slot[k * MAX_CONNECTIONS + c] = static_cast<uint32_t>(slot);
            m.conn_flux[k * MAX_CONNECTIONS + c] = flux_tmp[i * MAX_CONNECTIONS + c];
        }
    }

    // Receive plan: ghosts are sorted, so they are grouped by owner in rank order
    m.recv_cnt.assign(nprocs, 0);
    m.recv_disp.assign(nprocs, 0);
    for (idx_t gid : ghosts) m.recv_cnt[ownerOf(starts, gid)]++;
    for (int r = 1; r < nprocs; ++r) m.recv_disp[r] = m.recv_disp[r - 1] + m.recv_cnt[r - 1];

    // Send plan: tell each owner which of its elements we need
    m.send_cnt.assign(nprocs, 0);
    MPI_Alltoall(m.recv_cnt.data(), 1, MPI_INT, m.send_cnt.data(), 1, MPI_INT, comm);
    m.send_disp.assign(nprocs, 0);
    for (int r = 1; r < nprocs; ++r) m.send_disp[r] = m.send_disp[r - 1] + m.send_cnt[r - 1];
    const int total_send = m.send_disp[nprocs - 1] + m.send_cnt[nprocs - 1];

    std::vector<idx_t> requested(total_send);
    MPI_Alltoallv(ghosts.data(), m.recv_cnt.data(), m.recv_disp.data(), MPI_UINT64_T,
                  requested.data(), m.send_cnt.data(), m.send_disp.data(), MPI_UINT64_T, comm);

    m.send_local.resize(total_send);
    for (int k = 0; k < total_send; ++k) {
        m.send_local[k] = static_cast<uint32_t>(m.local_of_gid[requested[k] - m.lo]);
    }

    m.recv_ranks.clear();
    m.send_ranks.clear();
    for (int r = 0; r < nprocs; ++r) {
        if (m.recv_cnt[r] > 0) m.recv_ranks.push_back(r);
        if (m.send_cnt[r] > 0) m.send_ranks.push_back(r);
    }
}

// Update local elements [begin, end): reads the current state (owned + ghost),
// writes the next energy, and accumulates total flux in place.
inline void updateRange(idx_t begin, idx_t end,
                        const uint8_t* __restrict num_conn,
                        const uint8_t* __restrict material_idx,
                        const uint32_t* __restrict conn_slot,
                        const val_t* __restrict conn_flux,
                        const Material* __restrict materials,
                        const val_t* __restrict cur,
                        val_t* __restrict nxt,
                        val_t* __restrict tflux) {
    for (idx_t i = begin; i < end; ++i) {
        const Material& mat = materials[material_idx[i]];
        const val_t self = cur[i];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        const uint32_t* slots = conn_slot + i * MAX_CONNECTIONS;
        const val_t* fluxes = conn_flux + i * MAX_CONNECTIONS;
        const int nc = num_conn[i];
        for (int j = 0; j < nc; ++j) {
            total_flux += computeFlux(mat, self, fluxes[j], cur[slots[j]]);
        }

        nxt[i] = self + total_flux;
        tflux[i] = tflux[i] + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations with a halo exchange per iteration.
// Communication overlaps with the update of interior elements.
void runSimulation(const Mesh& m, const std::vector<Material>& materials,
                   std::vector<val_t>& cur, std::vector<val_t>& nxt,
                   std::vector<val_t>& tflux, int n_iters, MPI_Comm comm) {
    const idx_t nloc = m.n_local;
    std::vector<val_t> sendbuf(m.send_local.size());
    std::vector<MPI_Request> reqs;
    reqs.reserve(m.recv_ranks.size() + m.send_ranks.size());

    for (int iter = 0; iter < n_iters; ++iter) {
        reqs.clear();

        // Post receives directly into ghost slots of the current buffer
        for (int r : m.recv_ranks) {
            MPI_Request q;
            MPI_Irecv(cur.data() + nloc + m.recv_disp[r], m.recv_cnt[r], MPI_DOUBLE,
                      r, 0, comm, &q);
            reqs.push_back(q);
        }

        // Pack and send boundary energies to peers
        for (size_t k = 0; k < m.send_local.size(); ++k) {
            sendbuf[k] = cur[m.send_local[k]];
        }
        for (int r : m.send_ranks) {
            MPI_Request q;
            MPI_Isend(sendbuf.data() + m.send_disp[r], m.send_cnt[r], MPI_DOUBLE,
                      r, 0, comm, &q);
            reqs.push_back(q);
        }

        // Interior elements need no remote data: overlap with communication
        updateRange(0, m.n_interior, m.num_conn.data(), m.material_idx.data(),
                    m.conn_slot.data(), m.conn_flux.data(), materials.data(),
                    cur.data(), nxt.data(), tflux.data());

        if (!reqs.empty()) {
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        }

        // Boundary elements depend on received ghost energies
        updateRange(m.n_interior, nloc, m.num_conn.data(), m.material_idx.data(),
                    m.conn_slot.data(), m.conn_flux.data(), materials.data(),
                    cur.data(), nxt.data(), tflux.data());

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

// Hash contribution of a single element with global index i. Contributions are
// combined by XOR, so per-rank partial hashes can be reduced in any order.
inline uint64_t elementHashTerm(val_t energy, val_t flux, uint64_t i) {
    uint64_t e_bits, f_bits;
    std::memcpy(&e_bits, &energy, sizeof(e_bits));
    std::memcpy(&f_bits, &flux, sizeof(f_bits));
    uint64_t h = 0;
    h ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
    h ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
    return h;
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

int runBenchmark(int argc, char** argv, int rank, int nprocs) {
    const bool root = (rank == 0);
    MPI_Comm comm = MPI_COMM_WORLD;

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

    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);
    if (n_elems_root < 1 || n_elems > static_cast<idx_t>(std::numeric_limits<int>::max())) {
        if (root) {
            printf("Grid size must be between 1 and %d\n",
                   static_cast<int>(std::sqrt(static_cast<double>(std::numeric_limits<int>::max()))));
            printUsage(argv[0]);
        }
        return 1;
    }

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %llu elements\n", n_elems_root, n_elems_root,
               static_cast<unsigned long long>(n_elems));
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        printf("Building unstructured mesh...\n");
    }

    // Build the distributed unstructured mesh
    const std::vector<idx_t> starts = computePartitionStarts(n_elems, nprocs);
    Mesh mesh;
    buildLocalMesh(mesh, n_elems_root, starts, rank, nprocs, comm);

    const std::vector<Material> materials = {
        Material{0.8, 0.0},   // Default material
        Material{0.8, 0.5},   // Inflow material
        Material{0.8, -0.5},  // Outflow material
    };

    // Dynamic state: current and next energy (owned + ghost), total flux (owned)
    std::vector<val_t> cur(mesh.n_local + mesh.n_ghost, 0.0);
    std::vector<val_t> nxt(mesh.n_local + mesh.n_ghost, 0.0);
    std::vector<val_t> tflux(mesh.n_local, 0.0);

    if (root) {
        // Memory usage of the logical global problem
        const size_t static_mem = n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        printf("Running simulation...\n");
    }

    // Run simulation; time the slowest rank
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();
    runSimulation(mesh, materials, cur, nxt, tflux, n_iters, comm);
    const double local_elapsed = MPI_Wtime() - t0;
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (root) {
        const long long duration_ms = static_cast<long long>(max_elapsed * 1000.0);
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

    // Hash over all elements in global order (XOR-combined, so reducible)
    uint64_t local_hash = 0;
    for (idx_t j = 0; j < mesh.n_local; ++j) {
        const idx_t k = mesh.local_of_gid[j];
        local_hash ^= elementHashTerm(cur[k], tflux[k], mesh.lo + j);
    }
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);

    int status = 0;
    if (root) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Gather full state on root only when external output or validation needs it
    if (printResults || validate) {
        std::vector<val_t> energy_local(mesh.n_local), flux_local(mesh.n_local);
        for (idx_t j = 0; j < mesh.n_local; ++j) {
            const idx_t k = mesh.local_of_gid[j];
            energy_local[j] = cur[k];
            flux_local[j] = tflux[k];
        }

        std::vector<int> counts(nprocs), displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            counts[r] = static_cast<int>(starts[r + 1] - starts[r]);
            displs[r] = static_cast<int>(starts[r]);
        }

        std::vector<val_t> energy_all, flux_all;
        if (root) {
            energy_all.resize(n_elems);
            flux_all.resize(n_elems);
        }
        MPI_Gatherv(energy_local.data(), static_cast<int>(mesh.n_local), MPI_DOUBLE,
                    root ? energy_all.data() : nullptr, counts.data(), displs.data(),
                    MPI_DOUBLE, 0, comm);
        MPI_Gatherv(flux_local.data(), static_cast<int>(mesh.n_local), MPI_DOUBLE,
                    root ? flux_all.data() : nullptr, counts.data(), displs.data(),
                    MPI_DOUBLE, 0, comm);

        if (root) {
            // Print results for external validation
            if (printResults) {
                print_results(energy_all, "ElementEnergy");
            }

            // Validation
            if (validate) {
                std::vector<ElementDynamic> elements(n_elems);
                for (idx_t i = 0; i < n_elems; ++i) {
                    elements[i] = ElementDynamic{energy_all[i], flux_all[i]};
                }
                if (!validateResults(elements)) {
                    status = 1;
                }
            }
        }
    }

    // Make the exit status consistent across all ranks
    MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int status = runBenchmark(argc, argv, rank, nprocs);

    MPI_Finalize();
    return status;
}
