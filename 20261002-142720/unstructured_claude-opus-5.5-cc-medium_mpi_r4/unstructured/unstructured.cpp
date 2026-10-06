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

// Block distribution of global element indices over ranks
struct Partition {
    int nranks = 1;
    std::vector<int64_t> begin;  // size nranks+1

    void init(int64_t n_global, int p) {
        nranks = p;
        begin.resize(p + 1);
        for (int r = 0; r <= p; ++r) begin[r] = (n_global * r) / p;
    }
    int owner(int64_t g) const {
        // last r with begin[r] <= g
        return static_cast<int>(std::upper_bound(begin.begin(), begin.end(), g) - begin.begin()) - 1;
    }
};

// Distributed world state (local block + ghost layer)
struct World {
    std::vector<Material> materials;

    int64_t global_begin = 0;     // first owned global index
    uint32_t n_local = 0;         // number of owned elements
    uint32_t n_ghost = 0;         // number of ghost elements

    // Local static connectivity (CSR, local indices; ghosts are >= n_local)
    std::vector<uint8_t> material_idx;
    std::vector<uint32_t> conn_offset;  // n_local + 1
    std::vector<uint32_t> conn_idx;
    std::vector<val_t> conn_flux;
    bool unit_flux = true;              // all connection flux coefficients == 1.0

    // Dynamic state (SoA); energy arrays include ghost storage
    std::vector<val_t> energy;
    std::vector<val_t> energy_swap;
    std::vector<val_t> total_flux;

    // Interior (no ghost dependency) and boundary element ranges [first, second)
    std::vector<std::pair<uint32_t, uint32_t>> interior_ranges;
    std::vector<std::pair<uint32_t, uint32_t>> boundary_ranges;

    // Halo exchange description
    std::vector<int> recv_ranks, recv_offsets, recv_counts;  // offsets into ghost area
    std::vector<int> send_ranks, send_offsets, send_counts;  // offsets into send_idx
    std::vector<uint32_t> send_idx;                          // local indices to send
    std::vector<val_t> send_buf;
    std::vector<MPI_Request> requests;
};

// Build the locally owned part of a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const Partition& part, int rank) {
    const int64_t n_root = n_elems_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    const int64_t lo = part.begin[rank];
    const int64_t hi = part.begin[rank + 1];
    world.global_begin = lo;
    world.n_local = static_cast<uint32_t>(hi - lo);
    const uint32_t n_local = world.n_local;

    world.material_idx.assign(n_local, static_cast<uint8_t>(DEFAULT_MAT_ID));
    world.conn_offset.resize(n_local + 1);
    std::vector<int64_t> conn_global;
    conn_global.reserve(static_cast<size_t>(n_local) * 4);
    world.conn_flux.clear();
    world.conn_flux.reserve(static_cast<size_t>(n_local) * 4);

    // Build connectivity: each element connects to its neighbors in 2D grid
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int64_t g = lo; g < hi; ++g) {
        const int64_t x = g / n_root;
        const int64_t y = g % n_root;
        world.conn_offset[g - lo] = static_cast<uint32_t>(conn_global.size());
        int count = 0;
        for (int n = 0; n < 4; ++n) {
            const int64_t nx = x + offsets[n][0];
            const int64_t ny = y + offsets[n][1];
            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_root && ny >= 0 && ny < n_root && count < MAX_CONNECTIONS) {
                conn_global.push_back(nx * n_root + ny);
                world.conn_flux.push_back(1.0);
                ++count;
            }
        }
    }
    world.conn_offset[n_local] = static_cast<uint32_t>(conn_global.size());

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int64_t last = n_root - 1;
    auto setMat = [&](int64_t g, idx_t m) {
        if (g >= lo && g < hi) world.material_idx[g - lo] = static_cast<uint8_t>(m);
    };
    // Same order as original (later assignments win on overlap)
    setMat(0 * n_root + 0, INFLOW_MAT_ID);
    setMat(0 * n_root + last, OUTFLOW_MAT_ID);
    setMat(last * n_root + 0, OUTFLOW_MAT_ID);
    setMat(last * n_root + last, INFLOW_MAT_ID);

    // ---- Ghost discovery (generic: driven purely by connectivity) ----
    std::vector<int64_t> ghosts;
    for (int64_t c : conn_global)
        if (c < lo || c >= hi) ghosts.push_back(c);
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    world.n_ghost = static_cast<uint32_t>(ghosts.size());

    // Map connectivity to local indices
    world.conn_idx.resize(conn_global.size());
    for (size_t k = 0; k < conn_global.size(); ++k) {
        const int64_t c = conn_global[k];
        if (c >= lo && c < hi) {
            world.conn_idx[k] = static_cast<uint32_t>(c - lo);
        } else {
            const size_t pos = std::lower_bound(ghosts.begin(), ghosts.end(), c) - ghosts.begin();
            world.conn_idx[k] = n_local + static_cast<uint32_t>(pos);
        }
    }
    for (val_t f : world.conn_flux)
        if (f != 1.0) world.unit_flux = false;

    // Ghosts are sorted by global index, hence grouped contiguously by owner
    const int P = part.nranks;
    std::vector<int> req_count(P, 0), req_disp(P, 0);
    for (int64_t g : ghosts) req_count[part.owner(g)]++;
    for (int r = 1; r < P; ++r) req_disp[r] = req_disp[r - 1] + req_count[r - 1];

    std::vector<int> srv_count(P, 0), srv_disp(P, 0);
    MPI_Alltoall(req_count.data(), 1, MPI_INT, srv_count.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int r = 1; r < P; ++r) srv_disp[r] = srv_disp[r - 1] + srv_count[r - 1];
    const int total_srv = P > 0 ? srv_disp[P - 1] + srv_count[P - 1] : 0;

    std::vector<int64_t> requested(std::max(total_srv, 1));
    MPI_Alltoallv(ghosts.data(), req_count.data(), req_disp.data(), MPI_INT64_T,
                  requested.data(), srv_count.data(), srv_disp.data(), MPI_INT64_T,
                  MPI_COMM_WORLD);

    for (int r = 0; r < P; ++r) {
        if (req_count[r] > 0) {
            world.recv_ranks.push_back(r);
            world.recv_offsets.push_back(req_disp[r]);
            world.recv_counts.push_back(req_count[r]);
        }
        if (srv_count[r] > 0) {
            world.send_ranks.push_back(r);
            world.send_offsets.push_back(srv_disp[r]);
            world.send_counts.push_back(srv_count[r]);
        }
    }
    world.send_idx.resize(total_srv);
    for (int k = 0; k < total_srv; ++k)
        world.send_idx[k] = static_cast<uint32_t>(requested[k] - lo);
    world.send_buf.resize(total_srv);
    world.requests.resize(world.recv_ranks.size() + world.send_ranks.size());

    // Classify elements as interior (only local neighbors) or boundary
    for (uint32_t i = 0; i < n_local;) {
        auto isBoundary = [&](uint32_t e) {
            for (uint32_t k = world.conn_offset[e]; k < world.conn_offset[e + 1]; ++k)
                if (world.conn_idx[k] >= n_local) return true;
            return false;
        };
        const bool b = isBoundary(i);
        uint32_t j = i + 1;
        while (j < n_local && isBoundary(j) == b) ++j;
        (b ? world.boundary_ranges : world.interior_ranges).emplace_back(i, j);
        i = j;
    }

    // Initialize all elements with zero energy
    world.energy.assign(static_cast<size_t>(n_local) + world.n_ghost, 0.0);
    world.energy_swap.assign(static_cast<size_t>(n_local) + world.n_ghost, 0.0);
    world.total_flux.assign(n_local, 0.0);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, val_t this_energy,
                        val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update elements in [first, last)
template <bool UNIT_FLUX>
static inline void updateRange(const World& world, const val_t* __restrict__ cur,
                               val_t* __restrict__ next, val_t* __restrict__ tflux,
                               uint32_t first, uint32_t last) {
    const Material* __restrict__ mats = world.materials.data();
    const uint8_t* __restrict__ mat_idx = world.material_idx.data();
    const uint32_t* __restrict__ off = world.conn_offset.data();
    const uint32_t* __restrict__ nbr = world.conn_idx.data();
    const val_t* __restrict__ cflux = world.conn_flux.data();

    for (uint32_t i = first; i < last; ++i) {
        const Material mat = mats[mat_idx[i]];
        const val_t e = cur[i];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        const uint32_t kend = off[i + 1];
        for (uint32_t k = off[i]; k < kend; ++k) {
            // connection_flux == 1.0 multiplies exactly, so skipping it is bit-identical
            const val_t cf = UNIT_FLUX ? 1.0 : cflux[k];
            total_flux += computeFlux(mat, e, cf, cur[nbr[k]]);
        }

        // Update element state
        next[i] = e + total_flux;
        tflux[i] = tflux[i] + std::abs(total_flux);
    }
}

template <bool UNIT_FLUX>
static void runSimulationImpl(World& world, const int n_iters) {
    const uint32_t n_local = world.n_local;
    const size_t n_recv = world.recv_ranks.size();
    const size_t n_send = world.send_ranks.size();

    for (int iter = 0; iter < n_iters; ++iter) {
        val_t* cur = world.energy.data();
        val_t* next = world.energy_swap.data();
        val_t* tflux = world.total_flux.data();

        // Start halo exchange: receive ghosts directly into ghost area
        for (size_t r = 0; r < n_recv; ++r) {
            MPI_Irecv(cur + n_local + world.recv_offsets[r], world.recv_counts[r], MPI_DOUBLE,
                      world.recv_ranks[r], 0, MPI_COMM_WORLD, &world.requests[r]);
        }
        for (size_t s = 0; s < n_send; ++s) {
            val_t* buf = world.send_buf.data() + world.send_offsets[s];
            const uint32_t* idx = world.send_idx.data() + world.send_offsets[s];
            const int cnt = world.send_counts[s];
            for (int k = 0; k < cnt; ++k) buf[k] = cur[idx[k]];
            MPI_Isend(buf, cnt, MPI_DOUBLE, world.send_ranks[s], 0, MPI_COMM_WORLD,
                      &world.requests[n_recv + s]);
        }

        // Overlap: update interior elements while halos are in flight
        for (const auto& rg : world.interior_ranges)
            updateRange<UNIT_FLUX>(world, cur, next, tflux, rg.first, rg.second);

        if (!world.requests.empty())
            MPI_Waitall(static_cast<int>(world.requests.size()), world.requests.data(),
                        MPI_STATUSES_IGNORE);

        for (const auto& rg : world.boundary_ranges)
            updateRange<UNIT_FLUX>(world, cur, next, tflux, rg.first, rg.second);

        // Swap buffers
        std::swap(world.energy, world.energy_swap);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    if (world.unit_flux)
        runSimulationImpl<true>(world, n_iters);
    else
        runSimulationImpl<false>(world, n_iters);
}

// Validate simulation results (global arrays, on root)
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

// Compute a simple hash of the local results (XOR-combinable across ranks)
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    for (uint32_t li = 0; li < world.n_local; ++li) {
        const uint64_t i = static_cast<uint64_t>(world.global_begin) + li;
        // Simple hash combining energy and flux values
        uint64_t e_bits, f_bits;
        std::memcpy(&e_bits, &world.energy[li], sizeof(e_bits));
        std::memcpy(&f_bits, &world.total_flux[li], sizeof(f_bits));
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
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
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
    Partition part;
    part.init(n_elems, nranks);
    World world;
    buildSquare2D(world, n_elems_root, part, rank);

    if (root) {
        // Calculate memory usage (global, reference layout)
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        // Run simulation
        printf("Running simulation...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Compute hash for verification (XOR is order-independent)
    const uint64_t local_hash = computeHash(world);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

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

    // Gather global results on root for output/validation (exact sequential reductions)
    int exit_code = 0;
    if (printResults || validate) {
        std::vector<int> counts(nranks), displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            counts[r] = static_cast<int>(part.begin[r + 1] - part.begin[r]);
            displs[r] = static_cast<int>(part.begin[r]);
        }
        std::vector<val_t> energy_all, flux_all;
        if (root) energy_all.resize(n_elems);
        MPI_Gatherv(world.energy.data(), static_cast<int>(world.n_local), MPI_DOUBLE,
                    energy_all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (validate) {
            if (root) flux_all.resize(n_elems);
            MPI_Gatherv(world.total_flux.data(), static_cast<int>(world.n_local), MPI_DOUBLE,
                        flux_all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        if (root) {
            // Print results for external validation
            if (printResults) print_results(energy_all, "ElementEnergy");

            // Validation
            if (validate && !validateResults(energy_all, flux_all)) exit_code = 1;
        }
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exit_code;
}
