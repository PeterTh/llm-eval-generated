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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (local numbering)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// MPI decomposition: the global element index space [0, n_global) is split into
// contiguous blocks, one per rank. Each rank stores its owned elements first
// (local indices [0, n_owned)), followed by halo (ghost) copies of remote
// elements that its owned elements are connected to. Connectivity is stored in
// local numbering so the simulation kernel never needs a lookup.
struct Decomposition {
    int rank = 0;
    int size = 1;
    idx_t n_global = 0;          // total number of elements
    idx_t owned_begin = 0;       // global index of first owned element
    idx_t n_owned = 0;           // number of owned elements

    // Halo exchange description. For each neighbor rank we send a contiguous
    // run of owned elements and receive a contiguous run into the halo region.
    struct Exchange {
        int peer;
        idx_t send_begin;        // local index of first element to send
        idx_t send_count;
        idx_t recv_begin;        // local index of first halo slot to fill
        idx_t recv_count;
    };
    std::vector<Exchange> exchanges;

    // Boundary elements: owned elements that depend on at least one halo value.
    // Stored as a contiguous local index range [bnd_lo_end) and [bnd_hi_begin, n_owned).
    idx_t bnd_lo_end = 0;        // owned elements [0, bnd_lo_end) touch the lower halo
    idx_t bnd_hi_begin = 0;      // owned elements [bnd_hi_begin, n_owned) touch the upper halo
};

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;     // owned elements only
    std::vector<ElementDynamic> elements_dynamic;   // owned + halo
    std::vector<ElementDynamic> elements_dynamic_swap;
    Decomposition decomp;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Compute the contiguous block of global indices owned by `rank`.
static inline void blockRange(idx_t n_global, int size, int rank, idx_t& begin, idx_t& end) {
    const idx_t base = n_global / size;
    const idx_t rem = n_global % size;
    const idx_t r = static_cast<idx_t>(rank);
    begin = r * base + std::min(r, rem);
    end = begin + base + (r < rem ? 1 : 0);
}

// Build a 2D square grid as an unstructured mesh, distributed across ranks.
// This represents computation on arbitrarily-shaped geometries.
// Each rank only constructs the elements it owns plus the halo it needs.
void buildSquare2D(World& world, const int n_elems_root) {
    Decomposition& d = world.decomp;
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);

    const idx_t n_root = static_cast<idx_t>(n_elems_root);
    d.n_global = n_root * n_root;

    // Owned block: a contiguous range of global indices. Because the grid is
    // numbered row-major (idx = x * n + y), a contiguous block is a set of
    // whole/partial rows and all cross-block connections lie within one row
    // (n_root elements) on either side of the block.
    idx_t owned_end;
    blockRange(d.n_global, d.size, d.rank, d.owned_begin, owned_end);
    d.n_owned = owned_end - d.owned_begin;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Halo layout: [owned][lower halo][upper halo]. The lower halo covers global
    // indices [owned_begin - halo_lo, owned_begin), the upper halo covers
    // [owned_end, owned_end + halo_hi). Neighbor offsets are +-1 (same row) and
    // +-n_root (adjacent rows), so at most n_root elements are needed per side.
    // A rank that owns no elements needs no halo at all.
    const idx_t halo_lo = d.n_owned ? std::min<idx_t>(d.owned_begin, n_root) : 0;
    const idx_t halo_hi = d.n_owned ? std::min<idx_t>(d.n_global - owned_end, n_root) : 0;
    const idx_t lo_halo_base = d.n_owned;               // local index of first lower-halo slot
    const idx_t hi_halo_base = d.n_owned + halo_lo;     // local index of first upper-halo slot
    const idx_t n_local = d.n_owned + halo_lo + halo_hi;

    // Allocate elements
    world.elements_static.resize(d.n_owned);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    // Initialize all elements with default material and zero energy
    for (idx_t i = 0; i < d.n_owned; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    for (idx_t i = 0; i < n_local; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        world.elements_dynamic_swap[i].current_energy = 0.0;
        world.elements_dynamic_swap[i].total_flux = 0.0;
    }

    // Map a global index (known to be owned or in the halo) to local numbering.
    auto toLocal = [&](idx_t g) -> idx_t {
        if (g >= d.owned_begin && g < owned_end) return g - d.owned_begin;
        if (g < d.owned_begin) return lo_halo_base + (g - (d.owned_begin - halo_lo));
        return hi_halo_base + (g - owned_end);
    };

    // Build connectivity: each element connects to its neighbors in 2D grid
    const idx_t x_begin = d.owned_begin / n_root;
    const idx_t x_end = d.n_owned > 0 ? (owned_end - 1) / n_root + 1 : x_begin;
    for (idx_t x = x_begin; x < x_end; ++x) {
        const idx_t y_begin = (x == x_begin) ? d.owned_begin - x * n_root : 0;
        const idx_t y_end = (x + 1 == x_end) ? owned_end - x * n_root : n_root;
        for (idx_t y = y_begin; y < y_end; ++y) {
            const idx_t idx = x * n_root + y;
            ElementStatic& elem = world.elements_static[idx - d.owned_begin];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const long nx = static_cast<long>(x) + offsets[n][0];
                const long ny = static_cast<long>(y) + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const idx_t neighbor_idx = static_cast<idx_t>(nx) * n_root + static_cast<idx_t>(ny);
                    elem.connected_idx[elem.num_connections] = toLocal(neighbor_idx);
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const idx_t last = n_root - 1;
    auto setMaterial = [&](idx_t g, idx_t mat) {
        if (g >= d.owned_begin && g < owned_end) {
            world.elements_static[g - d.owned_begin].material_idx = mat;
        }
    };
    setMaterial(0 * n_root + 0, INFLOW_MAT_ID);
    setMaterial(0 * n_root + last, OUTFLOW_MAT_ID);
    setMaterial(last * n_root + 0, OUTFLOW_MAT_ID);
    setMaterial(last * n_root + last, INFLOW_MAT_ID);

    // Describe the halo exchanges. Global indices [owned_begin - halo_lo, owned_begin)
    // may be spread over several lower ranks (if blocks are smaller than a row),
    // so walk the owning ranks of each halo range. Each peer both sends us its
    // part of our halo and receives from us the part of its halo that we own.
    // The two counts can differ when block sizes are uneven, so both are kept.
    d.exchanges.clear();
    // Lower side: we receive [owned_begin - halo_lo, owned_begin) and send the
    // matching upper halo of each lower peer, which is [owned_begin, min(owned_end, peer_end + n_root)).
    for (int peer = d.rank - 1; peer >= 0; --peer) {
        idx_t pb, pe;
        blockRange(d.n_global, d.size, peer, pb, pe);
        const idx_t recv_lo = std::max(pb, d.owned_begin - halo_lo);
        const idx_t recv_hi = std::min(pe, d.owned_begin);
        // Peer's upper halo is [pe, pe + peer_halo_hi); the part we own:
        const idx_t peer_halo_hi = (pe > pb) ? std::min<idx_t>(d.n_global - pe, n_root) : 0;
        const idx_t send_lo = std::max(pe, d.owned_begin);
        const idx_t send_hi = std::min(pe + peer_halo_hi, owned_end);
        const idx_t n_recv = recv_hi > recv_lo ? recv_hi - recv_lo : 0;
        const idx_t n_send = send_hi > send_lo ? send_hi - send_lo : 0;
        if (n_recv == 0 && n_send == 0) break;  // no further lower peers overlap
        d.exchanges.push_back(Decomposition::Exchange{
            peer, n_send ? send_lo - d.owned_begin : 0, n_send,
            n_recv ? toLocal(recv_lo) : 0, n_recv});
    }
    // Upper side: we receive [owned_end, owned_end + halo_hi) and send the matching
    // lower halo of each upper peer.
    for (int peer = d.rank + 1; peer < d.size; ++peer) {
        idx_t pb, pe;
        blockRange(d.n_global, d.size, peer, pb, pe);
        const idx_t recv_lo = std::max(pb, owned_end);
        const idx_t recv_hi = std::min(pe, owned_end + halo_hi);
        const idx_t peer_halo_lo = (pe > pb) ? std::min<idx_t>(pb, n_root) : 0;
        const idx_t send_lo = std::max(pb - peer_halo_lo, d.owned_begin);
        const idx_t send_hi = std::min(pb, owned_end);
        const idx_t n_recv = recv_hi > recv_lo ? recv_hi - recv_lo : 0;
        const idx_t n_send = send_hi > send_lo ? send_hi - send_lo : 0;
        if (n_recv == 0 && n_send == 0) break;  // no further upper peers overlap
        d.exchanges.push_back(Decomposition::Exchange{
            peer, n_send ? send_lo - d.owned_begin : 0, n_send,
            n_recv ? toLocal(recv_lo) : 0, n_recv});
    }

    // Boundary element ranges: owned elements within n_root of either block edge
    // are the only ones that can reference halo values.
    d.bnd_lo_end = (halo_lo > 0) ? std::min<idx_t>(n_root, d.n_owned) : 0;
    d.bnd_hi_begin = (halo_hi > 0) ? d.n_owned - std::min<idx_t>(n_root, d.n_owned) : d.n_owned;
    if (d.bnd_hi_begin < d.bnd_lo_end) d.bnd_hi_begin = d.bnd_lo_end;
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update owned elements in local index range [begin, end)
static inline void updateRange(World& world, idx_t begin, idx_t end) {
    const ElementStatic* __restrict__ elements_static = world.elements_static.data();
    const ElementDynamic* __restrict__ elements_dynamic = world.elements_dynamic.data();
    ElementDynamic* __restrict__ elements_swap = world.elements_dynamic_swap.data();
    const Material* __restrict__ materials = world.materials.data();

    for (idx_t i = begin; i < end; ++i) {
        const ElementStatic& elem_static = elements_static[i];
        const ElementDynamic& elem_dyn = elements_dynamic[i];
        const Material& mat = materials[elem_static.material_idx];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            const ElementDynamic& neighbor_dyn = elements_dynamic[neighbor_idx];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }

        // Update element state
        ElementDynamic& elem_write = elements_swap[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const Decomposition& d = world.decomp;
    const idx_t n_owned = d.n_owned;
    const size_t n_ex = d.exchanges.size();
    std::vector<MPI_Request> requests(2 * n_ex);

    // Only current_energy of halo elements is ever read by the kernel, but the
    // ElementDynamic struct is small and contiguous, so exchange it whole.
    MPI_Datatype elem_type;
    MPI_Type_contiguous(static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE, &elem_type);
    MPI_Type_commit(&elem_type);

    for (int iter = 0; iter < n_iters; ++iter) {
        ElementDynamic* dyn = world.elements_dynamic.data();

        // Post halo exchange of the current state (nonblocking)
        for (size_t e = 0; e < n_ex; ++e) {
            const Decomposition::Exchange& ex = d.exchanges[e];
            MPI_Irecv(dyn + ex.recv_begin, static_cast<int>(ex.recv_count), elem_type,
                      ex.peer, 0, MPI_COMM_WORLD, &requests[2 * e]);
            MPI_Isend(dyn + ex.send_begin, static_cast<int>(ex.send_count), elem_type,
                      ex.peer, 0, MPI_COMM_WORLD, &requests[2 * e + 1]);
        }

        // Update interior elements (no halo dependency) while communication proceeds
        updateRange(world, d.bnd_lo_end, d.bnd_hi_begin);

        // Wait for halo data, then update boundary elements
        if (n_ex > 0) {
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        }
        updateRange(world, 0, d.bnd_lo_end);
        updateRange(world, d.bnd_hi_begin, n_owned);

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }

    MPI_Type_free(&elem_type);
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

// Compute a simple hash of the results for verification.
// The hash is an XOR of per-element terms indexed by the global element index,
// so each rank computes the terms for its owned elements and the partial hashes
// are combined with an XOR reduction, giving a result identical to the serial code.
uint64_t computeHash(const World& world) {
    const Decomposition& d = world.decomp;
    uint64_t hash = 0;
    for (idx_t li = 0; li < d.n_owned; ++li) {
        const size_t i = static_cast<size_t>(d.owned_begin + li);
        const ElementDynamic& elem = world.elements_dynamic[li];
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elem.current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elem.total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t global_hash = 0;
    MPI_Reduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    return global_hash;
}

// Gather the owned dynamic state of all ranks onto rank 0 in global order.
// Only used for the (untimed) result printing and validation paths.
std::vector<ElementDynamic> gatherDynamic(const World& world) {
    const Decomposition& d = world.decomp;
    std::vector<int> counts(d.size), displs(d.size);
    for (int r = 0; r < d.size; ++r) {
        idx_t b, e;
        blockRange(d.n_global, d.size, r, b, e);
        counts[r] = static_cast<int>(e - b);
        displs[r] = static_cast<int>(b);
    }
    std::vector<ElementDynamic> all;
    if (d.rank == 0) all.resize(d.n_global);

    MPI_Datatype elem_type;
    MPI_Type_contiguous(static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE, &elem_type);
    MPI_Type_commit(&elem_type);
    MPI_Gatherv(world.elements_dynamic.data(), static_cast<int>(d.n_owned), elem_type,
                all.data(), counts.data(), displs.data(), elem_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&elem_type);
    return all;
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool is_root = (rank == 0);

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
            if (is_root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (is_root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (is_root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("\n");
    }

    // Build the unstructured mesh
    if (is_root) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);

    // Calculate memory usage (global totals across all ranks)
    const size_t local_static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t local_dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    unsigned long long local_mem[2] = {local_static_mem, local_dynamic_mem};
    unsigned long long global_mem[2] = {0, 0};
    MPI_Reduce(local_mem, global_mem, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (is_root) {
        const double static_mem = static_cast<double>(global_mem[0]);
        const double dynamic_mem = static_cast<double>(global_mem[1]);
        const double total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run simulation
    if (is_root) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();

    runSimulation(world, n_iters);

    const double t_local = MPI_Wtime() - t_start;
    double t_max = t_local;
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long duration_ms = static_cast<long>(t_max * 1000.0);

    if (is_root) printf("Computation time: %ld ms\n", duration_ms);

    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9;

    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;

    // Compute hash for verification (collective)
    const uint64_t hash = computeHash(world);

    if (is_root) {
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    int exit_code = 0;
    if (printResults || validate) {
        // Gather the full state on rank 0 so output is identical to the serial code
        std::vector<ElementDynamic> all = gatherDynamic(world);

        // Print results for external validation
        if (printResults && is_root) {
            std::vector<double> energyData;
            energyData.reserve(all.size());
            for (const auto& elem : all) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            int valid = 1;
            if (is_root) valid = validateResults(all) ? 1 : 0;
            MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!valid) exit_code = 1;
        }
    }

    MPI_Finalize();
    return exit_code;
}
