// Only the C MPI API is used; skip the C++ bindings (Open MPI honors OMPI_SKIP_MPICXX)
#define OMPI_SKIP_MPICXX 1
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

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};
static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t), "ElementDynamic must be two packed values");

// A contiguous block of values exchanged with one peer rank
struct Link {
    int peer;
    size_t offset;  // Offset into the halo buffer (recv) or the send index list (send)
    size_t count;
};

// Distributed world: each rank owns a contiguous block of global element indices.
// Energy buffers hold the owned elements followed by halo copies of remote neighbors.
struct World {
    idx_t global_begin = 0;                      // First global element index owned by this rank
    idx_t n_local = 0;                           // Number of owned elements
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;  // Owned elements; connected_idx indexes energy[]
    std::vector<val_t> total_flux;               // Accumulated flux magnitude per owned element
    std::vector<val_t> energy;                   // Current energy: owned, then halo
    std::vector<val_t> energy_swap;              // Next energy (same layout as energy)
    std::vector<uint8_t> is_boundary;            // 1 if an owned element has a halo neighbor
    std::vector<idx_t> boundary;                 // Owned elements with a halo neighbor (is_boundary set)
    std::vector<Link> recv_links;
    std::vector<Link> send_links;
    std::vector<idx_t> send_indices;             // Local indices of owned values sent to peers
    std::vector<val_t> send_buf;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Owned range boundaries: rank r owns global indices [bounds[r], bounds[r+1])
std::vector<idx_t> makeBounds(idx_t n_elems, int nranks) {
    std::vector<idx_t> bounds(nranks + 1);
    for (int r = 0; r <= nranks; ++r) {
        bounds[r] = n_elems * static_cast<idx_t>(r) / static_cast<idx_t>(nranks);
    }
    return bounds;
}

// Rank that owns global element g
int ownerOf(const std::vector<idx_t>& bounds, idx_t g) {
    return static_cast<int>(std::upper_bound(bounds.begin(), bounds.end(), g) - bounds.begin()) - 1;
}

// Material of global element g. Corner assignments are applied in the same order as the serial builder.
idx_t materialOf(idx_t g, idx_t n) {
    const idx_t last = n - 1;
    idx_t mat = DEFAULT_MAT_ID;
    if (g == 0) mat = INFLOW_MAT_ID;
    if (g == last) mat = OUTFLOW_MAT_ID;
    if (g == last * n) mat = OUTFLOW_MAT_ID;
    if (g == last * n + last) mat = INFLOW_MAT_ID;
    return mat;
}

// Global neighbors of element g in a 2D square grid (up, down, left, right, in-bounds only)
int neighborsOf(idx_t g, idx_t n, idx_t out[4]) {
    const long long x = static_cast<long long>(g / n);
    const long long y = static_cast<long long>(g % n);
    const long long offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int count = 0;
    for (int k = 0; k < 4; ++k) {
        const long long nx = x + offsets[k][0];
        const long long ny = y + offsets[k][1];
        if (nx >= 0 && nx < static_cast<long long>(n) && ny >= 0 && ny < static_cast<long long>(n)) {
            out[count++] = static_cast<idx_t>(nx) * n + static_cast<idx_t>(ny);
        }
    }
    return count;
}

// Build this rank's share of the 2D square grid unstructured mesh and its communication pattern
void buildSquare2D(World& world, const int n_elems_root, const std::vector<idx_t>& bounds, const int rank) {
    const idx_t n = static_cast<idx_t>(n_elems_root);
    const int nranks = static_cast<int>(bounds.size()) - 1;
    const idx_t gbeg = bounds[rank];
    const idx_t gend = bounds[rank + 1];
    const idx_t n_local = gend - gbeg;

    world.global_begin = gbeg;
    world.n_local = n_local;

    // Initialize materials
    world.materials = {Material{0.8, 0.0},    // Default material
                       Material{0.8, 0.5},    // Inflow material
                       Material{0.8, -0.5}};  // Outflow material

    // Collect remote neighbors needed by owned elements, sorted and unique.
    // Sorting groups them by owner, so each peer's values form one contiguous halo slice.
    std::vector<idx_t> remote;
    for (idx_t g = gbeg; g < gend; ++g) {
        idx_t nb[4];
        const int count = neighborsOf(g, n, nb);
        for (int k = 0; k < count; ++k) {
            if (nb[k] < gbeg || nb[k] >= gend) remote.push_back(nb[k]);
        }
    }
    std::sort(remote.begin(), remote.end());
    remote.erase(std::unique(remote.begin(), remote.end()), remote.end());

    // Receive side: how many values come from each peer
    std::vector<int> recv_cnt(nranks, 0);
    for (idx_t g : remote) recv_cnt[ownerOf(bounds, g)]++;

    world.recv_links.clear();
    size_t halo_off = 0;
    for (int q = 0; q < nranks; ++q) {
        if (recv_cnt[q] > 0) {
            world.recv_links.push_back(Link{q, halo_off, static_cast<size_t>(recv_cnt[q])});
        }
        halo_off += static_cast<size_t>(recv_cnt[q]);
    }

    // Send side: each owner learns which of its elements the other ranks need
    std::vector<int> send_cnt(nranks, 0);
    MPI_Alltoall(recv_cnt.data(), 1, MPI_INT, send_cnt.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> recv_disp(nranks, 0), send_disp(nranks, 0);
    int recv_total = 0, send_total = 0;
    for (int q = 0; q < nranks; ++q) {
        recv_disp[q] = recv_total;
        send_disp[q] = send_total;
        recv_total += recv_cnt[q];
        send_total += send_cnt[q];
    }

    std::vector<idx_t> wanted(send_total);
    MPI_Alltoallv(remote.data(), recv_cnt.data(), recv_disp.data(), MPI_UINT64_T,
                  wanted.data(), send_cnt.data(), send_disp.data(), MPI_UINT64_T, MPI_COMM_WORLD);

    world.send_links.clear();
    world.send_indices.resize(send_total);
    for (int q = 0; q < nranks; ++q) {
        if (send_cnt[q] > 0) {
            world.send_links.push_back(Link{q, static_cast<size_t>(send_disp[q]), static_cast<size_t>(send_cnt[q])});
        }
    }
    for (int k = 0; k < send_total; ++k) {
        world.send_indices[k] = wanted[k] - gbeg;  // Owned by this rank by construction
    }
    world.send_buf.assign(send_total, 0.0);

    // Local element state and connectivity
    world.elements_static.assign(n_local, ElementStatic{});
    world.is_boundary.assign(n_local, 0);
    world.boundary.clear();
    for (idx_t i = 0; i < n_local; ++i) {
        const idx_t g = gbeg + i;
        ElementStatic& elem = world.elements_static[i];
        elem.material_idx = materialOf(g, n);
        elem.num_connections = 0;

        // Same neighbor order as the serial builder; remote neighbors map to halo slots
        idx_t nb[4];
        const int count = neighborsOf(g, n, nb);
        bool has_halo = false;
        for (int k = 0; k < count; ++k) {
            idx_t local;
            if (nb[k] >= gbeg && nb[k] < gend) {
                local = nb[k] - gbeg;
            } else {
                local = n_local + static_cast<idx_t>(std::lower_bound(remote.begin(), remote.end(), nb[k]) - remote.begin());
                has_halo = true;
            }
            elem.connected_idx[elem.num_connections] = local;
            elem.connected_flux[elem.num_connections] = 1.0;
            elem.num_connections++;
        }
        if (has_halo) {
            world.is_boundary[i] = 1;
            world.boundary.push_back(i);
        }
    }

    world.total_flux.assign(n_local, 0.0);
    world.energy.assign(n_local + remote.size(), 0.0);
    world.energy_swap.assign(n_local + remote.size(), 0.0);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, val_t this_energy,
                         val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update one owned element from the current energies into the next buffer.
// Works on raw pointers so the hot loop keeps its base addresses in registers.
inline void updateElement(idx_t i, const ElementStatic* elems, const Material* mats,
                          val_t* total_flux, const val_t* cur, val_t* nxt) {
    const ElementStatic& elem_static = elems[i];
    const Material& mat = mats[elem_static.material_idx];
    const val_t self = cur[i];

    // Start with external flow
    val_t flux = mat.external_flow;

    // Add flux from all connected elements
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        flux += computeFlux(mat, self, elem_static.connected_flux[j], cur[elem_static.connected_idx[j]]);
    }

    nxt[i] = self + flux;
    total_flux[i] += std::abs(flux);
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, MPI_Comm comm) {
    const idx_t n_local = world.n_local;
    const ElementStatic* elems = world.elements_static.data();
    const Material* mats = world.materials.data();
    val_t* total_flux = world.total_flux.data();
    const uint8_t* is_boundary = world.is_boundary.data();

    std::vector<MPI_Request> reqs(world.recv_links.size() + world.send_links.size());

    for (int iter = 0; iter < n_iters; ++iter) {
        val_t* cur = world.energy.data();
        val_t* nxt = world.energy_swap.data();

        // Post halo receives and sends of the current energies
        int nreq = 0;
        for (const Link& link : world.recv_links) {
            MPI_Irecv(cur + n_local + link.offset, static_cast<int>(link.count), MPI_DOUBLE,
                      link.peer, 0, comm, &reqs[nreq++]);
        }
        for (size_t k = 0; k < world.send_indices.size(); ++k) {
            world.send_buf[k] = cur[world.send_indices[k]];
        }
        for (const Link& link : world.send_links) {
            MPI_Isend(world.send_buf.data() + link.offset, static_cast<int>(link.count), MPI_DOUBLE,
                      link.peer, 0, comm, &reqs[nreq++]);
        }

        // Interior elements need no remote data, so compute them while the halo is in flight
        for (idx_t i = 0; i < n_local; ++i) {
            if (!is_boundary[i]) updateElement(i, elems, mats, total_flux, cur, nxt);
        }

        MPI_Waitall(nreq, reqs.data(), MPI_STATUSES_IGNORE);

        for (idx_t i : world.boundary) {
            updateElement(i, elems, mats, total_flux, cur, nxt);
        }

        // Swap buffers
        std::swap(world.energy, world.energy_swap);
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

// Compute a simple hash of the results for verification.
// Each term depends only on its global index and XOR is order-independent, so
// partial hashes of disjoint ranges combine exactly into the full hash.
uint64_t computeHash(const std::vector<ElementDynamic>& elements, idx_t global_offset) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        const idx_t gi = global_offset + i;
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + gi) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + gi) * 0xbf58476d1ce4e5b9ULL;
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

int run(int argc, char** argv) {
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool root = (rank == 0);

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

    if (n_elems_root <= 0) {
        if (root) {
            printf("Grid size must be positive\n");
            printUsage(argv[0]);
        }
        return 1;
    }

    const int n_elems = n_elems_root * n_elems_root;
    const idx_t n_total = static_cast<idx_t>(n_elems);

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
    const std::vector<idx_t> bounds = makeBounds(n_total, nranks);
    World world;
    buildSquare2D(world, n_elems_root, bounds, rank);

    // Calculate memory usage (global, as in the serial version)
    if (root) {
        const size_t static_mem = n_total * sizeof(ElementStatic);
        const size_t dynamic_mem = n_total * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        printf("Running simulation...\n");
    }

    // Run simulation; the reported time is the slowest rank
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    runSimulation(world, n_iters, MPI_COMM_WORLD);

    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Final owned state
    std::vector<ElementDynamic> local_state(world.n_local);
    for (idx_t i = 0; i < world.n_local; ++i) {
        local_state[i].current_energy = world.energy[i];
        local_state[i].total_flux = world.total_flux[i];
    }

    if (root) {
        const long duration_ms = static_cast<long>(max_elapsed * 1000.0);
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

    // Compute hash for verification
    uint64_t hash = 0;
    const uint64_t local_hash = computeHash(local_state, world.global_begin);
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (root) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Gather the full state on rank 0 only when the serial-order statistics need it
    int status = 0;
    if (validate || printResults) {
        std::vector<ElementDynamic> all_state;
        std::vector<int> counts(nranks), displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            counts[r] = static_cast<int>(2 * (bounds[r + 1] - bounds[r]));
            displs[r] = static_cast<int>(2 * bounds[r]);
        }
        if (root) all_state.resize(n_total);
        MPI_Gatherv(local_state.data(), static_cast<int>(2 * world.n_local), MPI_DOUBLE,
                    all_state.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (root) {
            // Print results for external validation
            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(all_state.size());
                for (const auto& elem : all_state) {
                    energyData.push_back(elem.current_energy);
                }
                print_results(energyData, "ElementEnergy");
            }

            // Validation
            if (validate && !validateResults(all_state)) {
                status = 1;
            }
        }
    }

    // All ranks report the same exit status
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int status = run(argc, argv);
    MPI_Finalize();
    return status;
}
