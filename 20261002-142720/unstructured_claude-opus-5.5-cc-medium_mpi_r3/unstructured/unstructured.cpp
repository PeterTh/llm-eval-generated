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
using lidx_t = uint32_t;  // Rank-local element index (local elements + ghosts)

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element (global indices, used while building)
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

// Distributed world state. Each rank owns a contiguous block of global element
// indices [elem_begin, elem_end); local element i is global element elem_begin + i.
// Ghost (halo) copies of remote neighbors are stored after the local elements.
// Static connectivity is kept in a compact CSR layout with local indices.
struct World {
    std::vector<Material> materials;

    // CSR connectivity of local elements
    std::vector<uint8_t> material_idx;     // Per local element
    std::vector<lidx_t> conn_ptr;          // Size n_local + 1
    std::vector<lidx_t> conn_idx;          // Local indices of connected elements
    std::vector<val_t> conn_flux;          // Flux coefficients for each connection

    std::vector<ElementDynamic> elements_dynamic;       // Local elements followed by ghosts
    std::vector<ElementDynamic> elements_dynamic_swap;

    idx_t n_global = 0;
    idx_t elem_begin = 0;
    idx_t elem_end = 0;
    size_t n_local = 0;
    size_t n_ghost = 0;

    // Halo exchange description
    std::vector<int> send_ranks;
    std::vector<size_t> send_offsets;      // Into send_list, size send_ranks + 1
    std::vector<lidx_t> send_list;         // Local indices of elements to send
    std::vector<int> recv_ranks;
    std::vector<size_t> recv_offsets;      // Into ghost range, size recv_ranks + 1
    std::vector<val_t> send_buf;
    std::vector<val_t> recv_buf;

    // Elements with ghost neighbors (updated after the halo exchange completes)
    // and runs [a, b) of interior elements (updated while communicating)
    std::vector<lidx_t> boundary_elems;
    std::vector<std::pair<size_t, size_t>> interior_runs;
};

static int g_rank = 0;
static int g_size = 1;

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Block distribution of n elements over p ranks
static inline idx_t blockBegin(idx_t n, int r, int p) {
    const idx_t base = n / p, rem = n % p;
    return base * r + std::min<idx_t>(r, rem);
}
static inline int ownerOf(idx_t g, idx_t n, int p) {
    const idx_t base = n / p, rem = n % p;
    const idx_t split = (base + 1) * rem;
    if (g < split) return static_cast<int>(g / (base + 1));
    return static_cast<int>(rem + (g - split) / base);
}

// Build the locally owned part of a 2D square grid as an unstructured mesh
// (global connectivity identical to the serial construction), then convert it
// to local indices and set up the halo exchange pattern from the connectivity.
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const idx_t nr = static_cast<idx_t>(n_elems_root);
    const idx_t n_elems = nr * nr;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.n_global = n_elems;
    world.elem_begin = blockBegin(n_elems, g_rank, g_size);
    world.elem_end = blockBegin(n_elems, g_rank + 1, g_size);
    const size_t n_local = world.elem_end - world.elem_begin;
    world.n_local = n_local;

    // Build connectivity with global indices: each element connects to its neighbors in 2D grid
    std::vector<ElementStatic> elements_static(n_local);
    const idx_t last = nr - 1;
    for (size_t i = 0; i < n_local; ++i) {
        const idx_t g = world.elem_begin + i;
        const int64_t x = static_cast<int64_t>(g / nr);
        const int64_t y = static_cast<int64_t>(g % nr);
        ElementStatic& elem = elements_static[i];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;

        // Connect to neighbors (up, down, left, right)
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        for (int n = 0; n < 4; ++n) {
            const int64_t nx = x + offsets[n][0];
            const int64_t ny = y + offsets[n][1];

            // Check if neighbor is within bounds
            if (nx >= 0 && nx < static_cast<int64_t>(nr) && ny >= 0 && ny < static_cast<int64_t>(nr)) {
                elem.connected_idx[elem.num_connections] = static_cast<idx_t>(nx) * nr + static_cast<idx_t>(ny);
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }

        // Set corner elements as inflow/outflow to create interesting dynamics
        // (same assignment order as the serial code, relevant when corners coincide)
        if (g == 0 * nr + 0) elem.material_idx = INFLOW_MAT_ID;
        if (g == 0 * nr + last) elem.material_idx = OUTFLOW_MAT_ID;
        if (g == last * nr + 0) elem.material_idx = OUTFLOW_MAT_ID;
        if (g == last * nr + last) elem.material_idx = INFLOW_MAT_ID;
    }

    // Collect remote neighbors (ghosts); sorting by global index groups them by owner
    std::vector<idx_t> ghosts;
    for (const ElementStatic& e : elements_static) {
        for (idx_t j = 0; j < e.num_connections; ++j) {
            const idx_t g = e.connected_idx[j];
            if (g < world.elem_begin || g >= world.elem_end) ghosts.push_back(g);
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    world.n_ghost = ghosts.size();

    if (n_local + world.n_ghost >= std::numeric_limits<lidx_t>::max()) {
        fprintf(stderr, "Rank %d: too many local elements (%zu); use more MPI ranks\n", g_rank, n_local);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Convert to CSR with local indices; classify interior/boundary elements
    world.material_idx.resize(n_local);
    world.conn_ptr.resize(n_local + 1);
    world.conn_ptr[0] = 0;
    std::vector<char> is_boundary(n_local, 0);
    for (size_t i = 0; i < n_local; ++i) {
        const ElementStatic& e = elements_static[i];
        world.material_idx[i] = static_cast<uint8_t>(e.material_idx);
        for (idx_t j = 0; j < e.num_connections; ++j) {
            const idx_t g = e.connected_idx[j];
            lidx_t l;
            if (g >= world.elem_begin && g < world.elem_end) {
                l = static_cast<lidx_t>(g - world.elem_begin);
            } else {
                l = static_cast<lidx_t>(n_local + (std::lower_bound(ghosts.begin(), ghosts.end(), g) - ghosts.begin()));
                is_boundary[i] = 1;
            }
            world.conn_idx.push_back(l);
            world.conn_flux.push_back(e.connected_flux[j]);
        }
        world.conn_ptr[i + 1] = static_cast<lidx_t>(world.conn_idx.size());
    }
    elements_static.clear();
    elements_static.shrink_to_fit();

    for (size_t i = 0; i < n_local;) {
        if (is_boundary[i]) {
            world.boundary_elems.push_back(static_cast<lidx_t>(i));
            ++i;
            continue;
        }
        size_t j = i;
        while (j < n_local && !is_boundary[j]) ++j;
        world.interior_runs.emplace_back(i, j);
        i = j;
    }

    // Receive side: ghosts grouped by owner rank
    std::vector<int> recv_counts(g_size, 0);
    world.recv_offsets.push_back(0);
    for (size_t k = 0; k < ghosts.size();) {
        const int owner = ownerOf(ghosts[k], n_elems, g_size);
        size_t k2 = k;
        while (k2 < ghosts.size() && ownerOf(ghosts[k2], n_elems, g_size) == owner) ++k2;
        world.recv_ranks.push_back(owner);
        world.recv_offsets.push_back(k2);
        recv_counts[owner] = static_cast<int>(k2 - k);
        k = k2;
    }

    // Send side: tell owners which of their elements we need
    std::vector<int> send_counts(g_size, 0);
    MPI_Alltoall(recv_counts.data(), 1, MPI_INT, send_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> rdispl(g_size + 1, 0), sdispl(g_size + 1, 0);
    for (int r = 0; r < g_size; ++r) {
        rdispl[r + 1] = rdispl[r] + recv_counts[r];
        sdispl[r + 1] = sdispl[r] + send_counts[r];
    }
    std::vector<uint64_t> requested(sdispl[g_size]);
    MPI_Alltoallv(ghosts.data(), recv_counts.data(), rdispl.data(), MPI_UINT64_T,
                  requested.data(), send_counts.data(), sdispl.data(), MPI_UINT64_T, MPI_COMM_WORLD);
    world.send_offsets.push_back(0);
    for (int r = 0; r < g_size; ++r) {
        if (send_counts[r] == 0) continue;
        world.send_ranks.push_back(r);
        for (int k = sdispl[r]; k < sdispl[r + 1]; ++k)
            world.send_list.push_back(static_cast<lidx_t>(requested[k] - world.elem_begin));
        world.send_offsets.push_back(world.send_list.size());
    }
    world.send_buf.resize(world.send_list.size());
    world.recv_buf.resize(world.n_ghost);

    // Initialize all elements (and ghosts) with zero energy
    world.elements_dynamic.assign(n_local + world.n_ghost, ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.assign(n_local + world.n_ghost, ElementDynamic{0.0, 0.0});
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update a range of local elements [begin, end) (or a list, via idx)
template <typename IndexFn>
static inline void updateElements(const World& world, const ElementDynamic* __restrict dyn,
                                  ElementDynamic* __restrict out, size_t count, IndexFn index) {
    const Material* __restrict mats = world.materials.data();
    const uint8_t* __restrict mat_idx = world.material_idx.data();
    const lidx_t* __restrict ptr = world.conn_ptr.data();
    const lidx_t* __restrict cidx = world.conn_idx.data();
    const val_t* __restrict cflux = world.conn_flux.data();

    for (size_t k = 0; k < count; ++k) {
        const size_t i = index(k);
        const ElementDynamic& elem_dyn = dyn[i];
        const Material& mat = mats[mat_idx[i]];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        for (lidx_t j = ptr[i]; j < ptr[i + 1]; ++j) {
            total_flux += computeFlux(mat, elem_dyn, cflux[j], dyn[cidx[j]]);
        }

        // Update element state
        ElementDynamic& elem_write = out[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_local = world.n_local;
    const size_t n_send = world.send_ranks.size();
    const size_t n_recv = world.recv_ranks.size();
    std::vector<MPI_Request> reqs(n_send + n_recv);

    for (int iter = 0; iter < n_iters; ++iter) {
        ElementDynamic* dyn = world.elements_dynamic.data();
        ElementDynamic* out = world.elements_dynamic_swap.data();

        // Start halo exchange of current energies
        int nreq = 0;
        for (size_t r = 0; r < n_recv; ++r) {
            const size_t b = world.recv_offsets[r], e = world.recv_offsets[r + 1];
            MPI_Irecv(world.recv_buf.data() + b, static_cast<int>(e - b), MPI_DOUBLE,
                      world.recv_ranks[r], 0, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        for (size_t r = 0; r < n_send; ++r) {
            const size_t b = world.send_offsets[r], e = world.send_offsets[r + 1];
            for (size_t k = b; k < e; ++k) world.send_buf[k] = dyn[world.send_list[k]].current_energy;
            MPI_Isend(world.send_buf.data() + b, static_cast<int>(e - b), MPI_DOUBLE,
                      world.send_ranks[r], 0, MPI_COMM_WORLD, &reqs[nreq++]);
        }

        // Update interior elements while communication is in flight
        for (const auto& run : world.interior_runs) {
            const size_t first = run.first;
            updateElements(world, dyn, out, run.second - first, [first](size_t k) { return first + k; });
        }

        MPI_Waitall(nreq, reqs.data(), MPI_STATUSES_IGNORE);
        for (size_t k = 0; k < world.n_ghost; ++k) dyn[n_local + k].current_energy = world.recv_buf[k];

        // Update elements depending on ghost values
        const lidx_t* bnd = world.boundary_elems.data();
        updateElements(world, dyn, out, world.boundary_elems.size(), [bnd](size_t k) { return bnd[k]; });

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Global sum in global element order: the running sum is passed from rank to
// rank, so the result is bitwise identical to the sequential summation.
template <typename Getter>
static val_t orderedGlobalSum(const World& world, Getter get) {
    val_t sum = 0.0;
    if (g_rank > 0) MPI_Recv(&sum, 1, MPI_DOUBLE, g_rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (size_t i = 0; i < world.n_local; ++i) sum += get(world.elements_dynamic[i]);
    if (g_rank + 1 < g_size) MPI_Send(&sum, 1, MPI_DOUBLE, g_rank + 1, 1, MPI_COMM_WORLD);
    MPI_Bcast(&sum, 1, MPI_DOUBLE, g_size - 1, MPI_COMM_WORLD);
    return sum;
}

// Validate simulation results (collective; rank 0 prints)
bool validateResults(const World& world) {
    const val_t energy_sum = orderedGlobalSum(world, [](const ElementDynamic& e) { return e.current_energy; });
    const val_t flux_sum = orderedGlobalSum(world, [](const ElementDynamic& e) { return e.total_flux; });
    val_t local_ext[2] = {std::numeric_limits<val_t>::lowest(), -std::numeric_limits<val_t>::max()};

    for (size_t i = 0; i < world.n_local; ++i) {
        const val_t e = world.elements_dynamic[i].current_energy;
        local_ext[0] = std::max(e, local_ext[0]);
        local_ext[1] = std::max(-e, local_ext[1]);
    }
    val_t global_ext[2];
    MPI_Allreduce(local_ext, global_ext, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const val_t energy_max = global_ext[0];
    const val_t energy_min = -global_ext[1];
    const bool root = (g_rank == 0);

    if (root) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    }

    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;

    if (!std::isfinite(energy_sum)) {
        if (root) printf("  ERROR: Energy sum is not finite\n");
        return false;
    }

    if (std::abs(energy_sum) > energy_epsilon) {
        if (root) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        // Don't fail validation as this can happen with external flows
    }

    if (!std::isfinite(flux_sum)) {
        if (root) printf("  ERROR: Flux sum is not finite\n");
        return false;
    }

    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        if (root) printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    if (root) printf("  Validation: PASSED\n");
    return true;
}

// Compute a simple hash of the results for verification (collective).
// The XOR combination is order independent, so partial hashes over global
// indices are combined with a bitwise-XOR reduction.
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    for (size_t li = 0; li < world.n_local; ++li) {
        const uint64_t i = world.elem_begin + li;
        // Simple hash combining energy and flux values
        uint64_t e_bits, f_bits;
        std::memcpy(&e_bits, &world.elements_dynamic[li].current_energy, sizeof(e_bits));
        std::memcpy(&f_bits, &world.elements_dynamic[li].total_flux, sizeof(f_bits));
        hash ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t global_hash = 0;
    MPI_Allreduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
    return global_hash;
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
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    const bool root = (g_rank == 0);

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
    World world;
    buildSquare2D(world, n_elems_root);

    // Calculate memory usage (of the equivalent serial data structures)
    const size_t static_mem = world.n_global * sizeof(ElementStatic);
    const size_t dynamic_mem = world.n_global * sizeof(ElementDynamic) * 2;
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
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long global_duration_ms = 0;
    MPI_Reduce(&duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Compute hash for verification
    const uint64_t hash = computeHash(world);

    if (root) {
        printf("Computation time: %ld ms\n", global_duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(global_duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (global_duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation (gather energies to rank 0)
    if (printResults) {
        std::vector<double> localData(world.n_local);
        for (size_t i = 0; i < world.n_local; ++i) localData[i] = world.elements_dynamic[i].current_energy;
        std::vector<int> counts, displs;
        std::vector<double> energyData;
        if (root) {
            counts.resize(g_size);
            displs.resize(g_size);
            for (int r = 0; r < g_size; ++r) {
                const idx_t b = blockBegin(world.n_global, r, g_size);
                counts[r] = static_cast<int>(blockBegin(world.n_global, r + 1, g_size) - b);
                displs[r] = static_cast<int>(b);
            }
            energyData.resize(world.n_global);
        }
        MPI_Gatherv(localData.data(), static_cast<int>(world.n_local), MPI_DOUBLE,
                    energyData.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (root) print_results(energyData, "ElementEnergy");
    }

    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
