#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
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

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Block partition of the global element index space across MPI ranks
struct Partition {
    int64_t n_global;
    int n_ranks;
    int64_t begin(int r) const {
        const int64_t base = n_global / n_ranks, rem = n_global % n_ranks;
        return r * base + std::min<int64_t>(r, rem);
    }
    int64_t end(int r) const { return begin(r + 1); }
    int owner(int64_t g) const {
        const int64_t base = n_global / n_ranks, rem = n_global % n_ranks;
        const int64_t split = rem * (base + 1);
        if (g < split) return static_cast<int>(g / (base + 1));
        return static_cast<int>(rem + (g - split) / base);
    }
};

// Build the locally owned part [g_begin, g_end) of a 2D square grid as an
// unstructured mesh. Connectivity uses global element indices.
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int64_t g_begin, const int64_t g_end) {
    const int64_t n_local = g_end - g_begin;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);
    
    // Initialize all elements with default material and zero energy
    for (int64_t i = 0; i < n_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int64_t g = g_begin; g < g_end; ++g) {
        const int64_t x = g / n_elems_root;
        const int64_t y = g % n_elems_root;
        ElementStatic& elem = world.elements_static[g - g_begin];
        
        // Connect to neighbors (up, down, left, right)
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        
        for (int n = 0; n < 4; ++n) {
            const int64_t nx = x + offsets[n][0];
            const int64_t ny = y + offsets[n][1];
            
            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const int64_t neighbor_idx = nx * n_elems_root + ny;
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int64_t last = n_elems_root - 1;
    auto setMat = [&](int64_t g, idx_t mat) {
        if (g >= g_begin && g < g_end) world.elements_static[g - g_begin].material_idx = mat;
    };
    setMat(0 * n_elems_root + 0, INFLOW_MAT_ID);
    setMat(0 * n_elems_root + last, OUTFLOW_MAT_ID);
    setMat(last * n_elems_root + 0, OUTFLOW_MAT_ID);
    setMat(last * n_elems_root + last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                        val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

// Compact local mesh representation with halo (ghost) exchange plan
struct LocalMesh {
    int n_local = 0;
    int n_ghost = 0;
    // CSR connectivity with local indices (ghosts are numbered n_local..)
    std::vector<int32_t> conn_off;
    std::vector<int32_t> conn_idx;
    std::vector<val_t> conn_flux;
    std::vector<val_t> coeff;     // per-element material transfer coefficient
    std::vector<val_t> ext_flow;  // per-element material external flow
    // Elements whose neighbors are all owned, as contiguous ranges; and the rest
    std::vector<std::pair<int32_t, int32_t>> interior_ranges;
    std::vector<int32_t> boundary;
    // Communication plan
    std::vector<int> recv_ranks, recv_offs, recv_cnts;  // ghost region slices
    std::vector<int> send_ranks, send_offs, send_cnts;
    std::vector<int32_t> send_idx;                       // local indices to pack
};

LocalMesh buildLocalMesh(const World& world, const Partition& part, int rank, MPI_Comm comm) {
    LocalMesh m;
    const int64_t g_begin = part.begin(rank), g_end = part.end(rank);
    m.n_local = static_cast<int>(g_end - g_begin);
    const int P = part.n_ranks;

    // Collect ghost global indices
    std::vector<int64_t> ghosts;
    for (int i = 0; i < m.n_local; ++i) {
        const ElementStatic& es = world.elements_static[i];
        for (idx_t j = 0; j < es.num_connections; ++j) {
            const int64_t g = static_cast<int64_t>(es.connected_idx[j]);
            if (g < g_begin || g >= g_end) ghosts.push_back(g);
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    m.n_ghost = static_cast<int>(ghosts.size());
    // Sorted by global index => grouped by owner rank (block partition)

    std::vector<int> req_cnt(P, 0), req_off(P, 0);
    for (int64_t g : ghosts) req_cnt[part.owner(g)]++;
    for (int r = 1; r < P; ++r) req_off[r] = req_off[r - 1] + req_cnt[r - 1];
    for (int r = 0; r < P; ++r) {
        if (req_cnt[r] > 0) {
            m.recv_ranks.push_back(r);
            m.recv_offs.push_back(m.n_local + req_off[r]);
            m.recv_cnts.push_back(req_cnt[r]);
        }
    }

    // Tell owners which indices we need
    std::vector<int> srv_cnt(P, 0), srv_off(P, 0);
    MPI_Alltoall(req_cnt.data(), 1, MPI_INT, srv_cnt.data(), 1, MPI_INT, comm);
    for (int r = 1; r < P; ++r) srv_off[r] = srv_off[r - 1] + srv_cnt[r - 1];
    const int n_send = srv_off[P - 1] + srv_cnt[P - 1];
    std::vector<int64_t> requested(n_send);
    MPI_Alltoallv(ghosts.data(), req_cnt.data(), req_off.data(), MPI_INT64_T,
                  requested.data(), srv_cnt.data(), srv_off.data(), MPI_INT64_T, comm);
    m.send_idx.resize(n_send);
    for (int k = 0; k < n_send; ++k) m.send_idx[k] = static_cast<int32_t>(requested[k] - g_begin);
    for (int r = 0; r < P; ++r) {
        if (srv_cnt[r] > 0) {
            m.send_ranks.push_back(r);
            m.send_offs.push_back(srv_off[r]);
            m.send_cnts.push_back(srv_cnt[r]);
        }
    }

    // Local CSR connectivity
    m.conn_off.resize(m.n_local + 1);
    m.coeff.resize(m.n_local);
    m.ext_flow.resize(m.n_local);
    m.conn_off[0] = 0;
    int32_t range_start = -1;
    for (int i = 0; i < m.n_local; ++i) {
        const ElementStatic& es = world.elements_static[i];
        const Material& mat = world.materials[es.material_idx];
        m.coeff[i] = mat.transfer_coeff;
        m.ext_flow[i] = mat.external_flow;
        bool is_interior = true;
        for (idx_t j = 0; j < es.num_connections; ++j) {
            const int64_t g = static_cast<int64_t>(es.connected_idx[j]);
            int32_t li;
            if (g >= g_begin && g < g_end) {
                li = static_cast<int32_t>(g - g_begin);
            } else {
                is_interior = false;
                li = m.n_local + static_cast<int32_t>(
                    std::lower_bound(ghosts.begin(), ghosts.end(), g) - ghosts.begin());
            }
            m.conn_idx.push_back(li);
            m.conn_flux.push_back(es.connected_flux[j]);
        }
        m.conn_off[i + 1] = static_cast<int32_t>(m.conn_idx.size());
        if (is_interior) {
            if (range_start < 0) range_start = i;
        } else {
            if (range_start >= 0) m.interior_ranges.emplace_back(range_start, i);
            range_start = -1;
            m.boundary.push_back(i);
        }
    }
    if (range_start >= 0) m.interior_ranges.emplace_back(range_start, m.n_local);
    return m;
}

// Update a single element
static inline void updateElement(const LocalMesh& m, const val_t* __restrict e,
                                 const val_t* __restrict tf, val_t* __restrict e_new,
                                 val_t* __restrict tf_new, int i) {
    const val_t this_e = e[i];
    const val_t coeff = m.coeff[i];
    // Start with external flow
    val_t total_flux = m.ext_flow[i];
    // Add flux from all connected elements
    const int32_t jb = m.conn_off[i], je = m.conn_off[i + 1];
    for (int32_t j = jb; j < je; ++j) {
        total_flux += computeFlux(coeff, this_e, m.conn_flux[j], e[m.conn_idx[j]]);
    }
    // Update element state
    e_new[i] = this_e + total_flux;
    tf_new[i] = tf[i] + std::abs(total_flux);
}

// Run simulation for n_iters iterations; energy/flux hold the local state on return
void runSimulation(const LocalMesh& m, std::vector<val_t>& energy, std::vector<val_t>& flux,
                   const int n_iters, MPI_Comm comm) {
    const int n_total = m.n_local + m.n_ghost;
    std::vector<val_t> e(n_total, 0.0), e_new(n_total, 0.0);
    std::vector<val_t> tf(m.n_local, 0.0), tf_new(m.n_local, 0.0);
    for (int i = 0; i < m.n_local; ++i) { e[i] = energy[i]; tf[i] = flux[i]; }

    std::vector<val_t> send_buf(m.send_idx.size());
    const size_t n_recv = m.recv_ranks.size(), n_sendr = m.send_ranks.size();
    std::vector<MPI_Request> reqs(n_recv + n_sendr);

    for (int iter = 0; iter < n_iters; ++iter) {
        val_t* __restrict ep = e.data();
        // Halo exchange of current energies
        for (size_t k = 0; k < n_recv; ++k) {
            MPI_Irecv(ep + m.recv_offs[k], m.recv_cnts[k], MPI_DOUBLE, m.recv_ranks[k], 0,
                      comm, &reqs[k]);
        }
        for (size_t k = 0; k < m.send_idx.size(); ++k) send_buf[k] = ep[m.send_idx[k]];
        for (size_t k = 0; k < n_sendr; ++k) {
            MPI_Isend(send_buf.data() + m.send_offs[k], m.send_cnts[k], MPI_DOUBLE,
                      m.send_ranks[k], 0, comm, &reqs[n_recv + k]);
        }

        // Interior elements overlap with communication
        for (const auto& rg : m.interior_ranges) {
            for (int32_t i = rg.first; i < rg.second; ++i) {
                updateElement(m, ep, tf.data(), e_new.data(), tf_new.data(), i);
            }
        }

        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

        // Boundary elements need ghost values
        for (int32_t i : m.boundary) {
            updateElement(m, ep, tf.data(), e_new.data(), tf_new.data(), i);
        }

        // Swap buffers
        std::swap(e, e_new);
        std::swap(tf, tf_new);
    }

    for (int i = 0; i < m.n_local; ++i) { energy[i] = e[i]; flux[i] = tf[i]; }
}

// Validate simulation results
bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
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
// (local contribution; XOR-combined across ranks; g_offset = global index of first element)
uint64_t computeHash(const std::vector<val_t>& energy, const std::vector<val_t>& flux,
                     uint64_t g_offset) {
    uint64_t hash = 0;
    for (size_t li = 0; li < energy.size(); ++li) {
        const uint64_t i = g_offset + li;
        // Simple hash combining energy and flux values
        uint64_t e_bits, f_bits;
        std::memcpy(&e_bits, &energy[li], sizeof(e_bits));
        std::memcpy(&f_bits, &flux[li], sizeof(f_bits));
        hash ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
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
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, n_ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);
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
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    
        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    const Partition part{static_cast<int64_t>(n_elems), n_ranks};
    const int64_t g_begin = part.begin(rank), g_end = part.end(rank);
    LocalMesh mesh;
    std::vector<val_t> energy, flux;
    {
        World world;
        buildSquare2D(world, n_elems_root, g_begin, g_end);
        mesh = buildLocalMesh(world, part, rank, comm);
        energy.resize(world.elements_dynamic.size());
        flux.resize(world.elements_dynamic.size());
        for (size_t i = 0; i < energy.size(); ++i) {
            energy[i] = world.elements_dynamic[i].current_energy;
            flux[i] = world.elements_dynamic[i].total_flux;
        }
    }
    
    // Calculate memory usage (global, as for the full mesh)
    const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (root) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    
        // Run simulation
        printf("Running simulation...\n");
        fflush(stdout);
    }
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(mesh, energy, flux, n_iters, comm);
    
    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();
    long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, comm);
    
    // Compute hash for verification
    const uint64_t local_hash = computeHash(energy, flux, static_cast<uint64_t>(g_begin));
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);

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
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Gather full results on rank 0 when needed
    std::vector<val_t> all_energy, all_flux;
    if (printResults || validate) {
        std::vector<int> counts(n_ranks), displs(n_ranks);
        for (int r = 0; r < n_ranks; ++r) {
            counts[r] = static_cast<int>(part.end(r) - part.begin(r));
            displs[r] = static_cast<int>(part.begin(r));
        }
        if (root) all_energy.resize(n_elems);
        MPI_Gatherv(energy.data(), static_cast<int>(energy.size()), MPI_DOUBLE,
                    all_energy.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
        if (validate) {
            if (root) all_flux.resize(n_elems);
            MPI_Gatherv(flux.data(), static_cast<int>(flux.size()), MPI_DOUBLE,
                        all_flux.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
        }
    }

    // Print results for external validation
    if (printResults && root) {
        print_results(all_energy, "ElementEnergy");
    }
    
    // Validation
    int status = 0;
    if (validate) {
        if (root) {
            bool valid = validateResults(all_energy, all_flux);
            status = valid ? 0 : 1;
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    }
    
    MPI_Finalize();
    return status;
}
