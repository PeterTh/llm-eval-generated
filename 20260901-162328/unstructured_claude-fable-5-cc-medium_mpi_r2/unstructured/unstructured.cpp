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

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif

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

// World state (each rank holds only its local block of elements plus ghosts)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;        // n_local elements
    std::vector<ElementDynamic> elements_dynamic;      // n_local + n_ghost elements
    std::vector<ElementDynamic> elements_dynamic_swap; // n_local + n_ghost elements
};

// Contiguous block distribution of the global element index space
struct Decomp {
    int rank = 0;
    int nprocs = 1;
    idx_t n_global = 0;
    idx_t begin = 0;  // first owned global index
    idx_t end = 0;    // one past last owned global index
};

// Halo exchange plan derived from the mesh connectivity
struct Halo {
    idx_t n_local = 0;
    idx_t n_ghost = 0;
    std::vector<int> send_ranks;
    std::vector<int> send_counts;
    std::vector<int> send_displs;
    std::vector<idx_t> send_idx;   // local indices to pack, flattened per peer
    std::vector<int> recv_ranks;
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;  // offsets into the ghost region
    std::vector<val_t> send_buf;
    std::vector<val_t> recv_buf;
    using Range = std::pair<idx_t, idx_t>;  // [first, last) of local indices
    std::vector<Range> interior;   // local elements with no ghost neighbors
    std::vector<Range> boundary;   // local elements touching at least one ghost
    std::vector<MPI_Request> recv_reqs;
    std::vector<MPI_Request> send_reqs;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

static idx_t blockBegin(idx_t n, int nprocs, int r) {
    const idx_t base = n / static_cast<idx_t>(nprocs);
    const idx_t rem = n % static_cast<idx_t>(nprocs);
    return static_cast<idx_t>(r) * base + std::min<idx_t>(static_cast<idx_t>(r), rem);
}

static int ownerOf(idx_t g, idx_t n, int nprocs) {
    const idx_t base = n / static_cast<idx_t>(nprocs);
    const idx_t rem = n % static_cast<idx_t>(nprocs);
    const idx_t cutoff = rem * (base + 1);
    if (g < cutoff) {
        return static_cast<int>(g / (base + 1));
    }
    return static_cast<int>(rem + (g - cutoff) / base);
}

// Build the local block of a 2D square grid as an unstructured mesh.
// Connectivity is stored with global indices; setupHalo remaps them to
// local/ghost indices afterwards.
void buildSquare2D(World& world, const int n_elems_root, const Decomp& decomp) {
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    const idx_t n_local = decomp.end - decomp.begin;
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    // Corner elements are inflow/outflow to create interesting dynamics
    const idx_t last = static_cast<idx_t>(n_elems_root) - 1;
    const idx_t corner_in_a = 0;
    const idx_t corner_out_a = last;
    const idx_t corner_out_b = last * static_cast<idx_t>(n_elems_root);
    const idx_t corner_in_b = last * static_cast<idx_t>(n_elems_root) + last;

    for (idx_t g = decomp.begin; g < decomp.end; ++g) {
        const idx_t i = g - decomp.begin;
        ElementStatic& elem = world.elements_static[i];
        elem.material_idx = DEFAULT_MAT_ID;
        if (g == corner_in_a || g == corner_in_b) elem.material_idx = INFLOW_MAT_ID;
        if (g == corner_out_a || g == corner_out_b) elem.material_idx = OUTFLOW_MAT_ID;
        elem.num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;

        const int x = static_cast<int>(g / static_cast<idx_t>(n_elems_root));
        const int y = static_cast<int>(g % static_cast<idx_t>(n_elems_root));

        // Connect to neighbors (up, down, left, right)
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];

            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_idx =
                    static_cast<idx_t>(nx) * static_cast<idx_t>(n_elems_root) + static_cast<idx_t>(ny);
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }
}

// Derive the halo exchange plan from the connectivity, remap the global
// neighbor indices to local/ghost indices, and split local elements into
// interior/boundary sets for communication/computation overlap.
void setupHalo(World& world, const Decomp& decomp, Halo& halo) {
    const idx_t n_local = decomp.end - decomp.begin;
    halo.n_local = n_local;

    // Collect the set of remote global indices referenced by local elements
    std::vector<idx_t> ghost_globals;
    for (const ElementStatic& elem : world.elements_static) {
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            const idx_t g = elem.connected_idx[j];
            if (g < decomp.begin || g >= decomp.end) {
                ghost_globals.push_back(g);
            }
        }
    }
    std::sort(ghost_globals.begin(), ghost_globals.end());
    ghost_globals.erase(std::unique(ghost_globals.begin(), ghost_globals.end()), ghost_globals.end());
    halo.n_ghost = ghost_globals.size();

    // Remap connectivity to local indices; ghosts live at [n_local, n_local + n_ghost)
    for (ElementStatic& elem : world.elements_static) {
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            const idx_t g = elem.connected_idx[j];
            if (g >= decomp.begin && g < decomp.end) {
                elem.connected_idx[j] = g - decomp.begin;
            } else {
                const auto it = std::lower_bound(ghost_globals.begin(), ghost_globals.end(), g);
                elem.connected_idx[j] = n_local + static_cast<idx_t>(it - ghost_globals.begin());
            }
        }
    }

    // Count ghosts per owning rank; since ghost_globals is sorted and block
    // ownership is monotonic in the global index, each owner's ghosts are a
    // contiguous slice of the ghost region.
    std::vector<int> req_counts(decomp.nprocs, 0);
    for (const idx_t g : ghost_globals) {
        req_counts[ownerOf(g, decomp.n_global, decomp.nprocs)]++;
    }

    std::vector<int> peer_counts(decomp.nprocs, 0);
    MPI_Alltoall(req_counts.data(), 1, MPI_INT, peer_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Tell each owner which of its elements we need
    std::vector<int> req_displs(decomp.nprocs, 0);
    std::vector<int> peer_displs(decomp.nprocs, 0);
    for (int r = 1; r < decomp.nprocs; ++r) {
        req_displs[r] = req_displs[r - 1] + req_counts[r - 1];
        peer_displs[r] = peer_displs[r - 1] + peer_counts[r - 1];
    }
    const int total_send = peer_displs[decomp.nprocs - 1] + peer_counts[decomp.nprocs - 1];
    std::vector<idx_t> requested_globals(static_cast<size_t>(total_send));
    MPI_Alltoallv(ghost_globals.data(), req_counts.data(), req_displs.data(), MPI_UINT64_T,
                  requested_globals.data(), peer_counts.data(), peer_displs.data(), MPI_UINT64_T,
                  MPI_COMM_WORLD);

    // Build the send side (values we provide to peers each iteration)
    halo.send_idx.reserve(requested_globals.size());
    for (int r = 0; r < decomp.nprocs; ++r) {
        if (peer_counts[r] == 0) continue;
        halo.send_ranks.push_back(r);
        halo.send_counts.push_back(peer_counts[r]);
        halo.send_displs.push_back(static_cast<int>(halo.send_idx.size()));
        for (int k = 0; k < peer_counts[r]; ++k) {
            halo.send_idx.push_back(requested_globals[peer_displs[r] + k] - decomp.begin);
        }
    }
    halo.send_buf.resize(halo.send_idx.size());

    // Build the receive side (our ghost region, grouped by owner)
    for (int r = 0; r < decomp.nprocs; ++r) {
        if (req_counts[r] == 0) continue;
        halo.recv_ranks.push_back(r);
        halo.recv_counts.push_back(req_counts[r]);
        halo.recv_displs.push_back(req_displs[r]);
    }
    halo.recv_buf.resize(halo.n_ghost);
    halo.recv_reqs.resize(halo.recv_ranks.size());
    halo.send_reqs.resize(halo.send_ranks.size());

    // Split local elements into interior (no ghost neighbors) and boundary,
    // stored as contiguous ranges so the update loops stay vectorizable
    for (idx_t i = 0; i < n_local; ++i) {
        const ElementStatic& elem = world.elements_static[i];
        bool touches_ghost = false;
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            if (elem.connected_idx[j] >= n_local) {
                touches_ghost = true;
                break;
            }
        }
        auto& ranges = touches_ghost ? halo.boundary : halo.interior;
        if (!ranges.empty() && ranges.back().second == i) {
            ranges.back().second = i + 1;
        } else {
            ranges.emplace_back(i, i + 1);
        }
    }

    // Extend the dynamic buffers with ghost slots (energy filled each iteration)
    world.elements_dynamic.resize(n_local + halo.n_ghost, ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.resize(n_local + halo.n_ghost, ElementDynamic{0.0, 0.0});
}

// Give a buffer an explicit local-preferred memory policy. A non-default
// policy makes the kernel's automatic NUMA balancing skip these pages, whose
// periodic hint faults otherwise stall the compute loop; placement itself
// is unchanged (first-touch pages stay where they are). Failure is harmless.
template <typename T>
void pinMemoryPolicy(std::vector<T>& v) {
#if defined(__linux__) && defined(SYS_mbind)
    if (v.empty()) return;
    constexpr uintptr_t page = 4096;
    const uintptr_t lo = reinterpret_cast<uintptr_t>(v.data()) & ~(page - 1);
    const uintptr_t hi =
        (reinterpret_cast<uintptr_t>(v.data() + v.size()) + page - 1) & ~(page - 1);
    constexpr int MPOL_PREFERRED_LOCAL = 1;  // MPOL_PREFERRED with empty node mask
    syscall(SYS_mbind, lo, hi - lo, MPOL_PREFERRED_LOCAL, nullptr, 0UL, 0U);
#else
    (void)v;
#endif
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update a contiguous range of local elements
// (reads elements_dynamic, writes the swap buffer)
static void updateRange(World& world, const idx_t first, const idx_t last) {
    const ElementStatic* const statics = world.elements_static.data();
    const ElementDynamic* const dynamics = world.elements_dynamic.data();
    ElementDynamic* const dynamics_out = world.elements_dynamic_swap.data();
    const Material* const materials = world.materials.data();

    for (idx_t i = first; i < last; ++i) {
        const ElementStatic& elem_static = statics[i];
        const ElementDynamic& elem_dyn = dynamics[i];
        const Material& mat = materials[elem_static.material_idx];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            const ElementDynamic& neighbor_dyn = dynamics[neighbor_idx];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }

        // Update element state
        ElementDynamic& elem_write = dynamics_out[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations with halo exchange overlapped
// with the interior update
void runSimulation(World& world, Halo& halo, const int n_iters) {
    for (int iter = 0; iter < n_iters; ++iter) {
        // Post receives for ghost energies
        for (size_t p = 0; p < halo.recv_ranks.size(); ++p) {
            MPI_Irecv(halo.recv_buf.data() + halo.recv_displs[p], halo.recv_counts[p], MPI_DOUBLE,
                      halo.recv_ranks[p], 0, MPI_COMM_WORLD, &halo.recv_reqs[p]);
        }

        // Pack and send boundary energies
        for (size_t k = 0; k < halo.send_idx.size(); ++k) {
            halo.send_buf[k] = world.elements_dynamic[halo.send_idx[k]].current_energy;
        }
        for (size_t p = 0; p < halo.send_ranks.size(); ++p) {
            MPI_Isend(halo.send_buf.data() + halo.send_displs[p], halo.send_counts[p], MPI_DOUBLE,
                      halo.send_ranks[p], 0, MPI_COMM_WORLD, &halo.send_reqs[p]);
        }

        // Update interior elements while the halo exchange is in flight
        for (const auto& [first, last] : halo.interior) {
            updateRange(world, first, last);
        }

        // Unpack ghost energies, then update boundary elements
        MPI_Waitall(static_cast<int>(halo.recv_reqs.size()), halo.recv_reqs.data(), MPI_STATUSES_IGNORE);
        for (idx_t k = 0; k < halo.n_ghost; ++k) {
            world.elements_dynamic[halo.n_local + k].current_energy = halo.recv_buf[k];
        }
        for (const auto& [first, last] : halo.boundary) {
            updateRange(world, first, last);
        }

        MPI_Waitall(static_cast<int>(halo.send_reqs.size()), halo.send_reqs.data(), MPI_STATUSES_IGNORE);

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (runs on rank 0 over the gathered global state)
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

// Gather the distributed dynamic state onto rank 0 in global index order
std::vector<ElementDynamic> gatherResults(const World& world, const Decomp& decomp) {
    const int n_local = static_cast<int>(decomp.end - decomp.begin);
    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<ElementDynamic> global;
    if (decomp.rank == 0) {
        counts.resize(decomp.nprocs);
        displs.resize(decomp.nprocs);
        for (int r = 0; r < decomp.nprocs; ++r) {
            const idx_t b = blockBegin(decomp.n_global, decomp.nprocs, r);
            const idx_t e = blockBegin(decomp.n_global, decomp.nprocs, r + 1);
            counts[r] = static_cast<int>((e - b) * sizeof(ElementDynamic));
            displs[r] = static_cast<int>(b * sizeof(ElementDynamic));
        }
        global.resize(decomp.n_global);
    }
    MPI_Gatherv(world.elements_dynamic.data(), static_cast<int>(n_local * sizeof(ElementDynamic)),
                MPI_BYTE, global.data(), counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    return global;
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

    int rank = 0;
    int nprocs = 1;
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
        printf("MPI ranks: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Build the local block of the unstructured mesh
    if (rank == 0) printf("Building unstructured mesh...\n");
    Decomp decomp;
    decomp.rank = rank;
    decomp.nprocs = nprocs;
    decomp.n_global = static_cast<idx_t>(n_elems);
    decomp.begin = blockBegin(decomp.n_global, nprocs, rank);
    decomp.end = blockBegin(decomp.n_global, nprocs, rank + 1);

    World world;
    buildSquare2D(world, n_elems_root, decomp);

    Halo halo;
    setupHalo(world, decomp, halo);

    pinMemoryPolicy(world.elements_static);
    pinMemoryPolicy(world.elements_dynamic);
    pinMemoryPolicy(world.elements_dynamic_swap);

    // Calculate memory usage (aggregate over all ranks, excluding ghosts)
    if (rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();

    runSimulation(world, halo, n_iters);

    const double t_local = MPI_Wtime() - t_start;
    double t_max = 0.0;
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long duration_ms = static_cast<long>(t_max * 1000.0);

    // Gather the global state onto rank 0 for hashing/validation/output
    const std::vector<ElementDynamic> global_dynamic = gatherResults(world, decomp);

    int exit_code = 0;
    if (rank == 0) {
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

        // Compute hash for verification
        const uint64_t hash = computeHash(global_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global_dynamic.size());
            for (const auto& elem : global_dynamic) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(global_dynamic);
            if (!valid) {
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
