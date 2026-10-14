#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices into the local+ghost element array
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Ghost exchange pattern of one rank, built once during setup
struct HaloExchange {
    std::vector<idx_t> ghosts;      // Global indices of remote neighbors (ascending, grouped by owner)
    std::vector<idx_t> send_list;   // Local indices of owned elements sent to others (grouped by destination)
    std::vector<int> sources;       // Ranks we receive ghosts from (ascending)
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    std::vector<int> destinations;  // Ranks we send our boundary elements to (ascending)
    std::vector<int> send_counts;
    std::vector<int> send_displs;
    std::vector<val_t> send_buf;
    std::vector<val_t> recv_buf;
    MPI_Comm comm = MPI_COMM_NULL;  // Distributed graph communicator over the neighbor ranks
};

// World state (the portion owned by this MPI rank)
struct World {
    idx_t begin = 0;                                // First global element index owned by this rank
    std::vector<Material> materials;                // Replicated on every rank
    std::vector<ElementStatic> elements_static;     // Owned elements
    std::vector<ElementDynamic> elements_dynamic;   // Owned elements followed by ghost copies
    std::vector<ElementDynamic> elements_dynamic_swap;
    std::vector<idx_t> interior;                    // Owned elements without remote neighbors
    std::vector<idx_t> boundary;                    // Owned elements with at least one remote neighbor
    HaloExchange halo;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// First global index of the contiguous block owned by `rank` (rank == n_ranks gives n_total)
idx_t blockBegin(const idx_t n_total, const int n_ranks, const int rank) {
    const idx_t q = n_total / n_ranks;
    const idx_t rem = n_total % n_ranks;
    return q * rank + rem * rank / n_ranks;
}

// Exclusive prefix sum of counts
std::vector<int> displacements(const std::vector<int>& counts) {
    std::vector<int> displs(counts.size());
    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
    return displs;
}

// Build the part of a 2D square grid owned by this rank as an unstructured mesh.
// Owned elements are a contiguous block of global indices; neighbors owned by other
// ranks are recorded as ghosts and stored after the owned elements.
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int n_ranks) {
    const idx_t n_root = static_cast<idx_t>(n_elems_root);
    const int64_t side = n_elems_root;
    const idx_t n_elems = n_root * n_root;
    const idx_t begin = blockBegin(n_elems, n_ranks, rank);
    const idx_t end = blockBegin(n_elems, n_ranks, rank + 1);
    const idx_t n_local = end - begin;
    const idx_t last = n_root - 1;
    world.begin = begin;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.elements_static.resize(n_local);

    // Build connectivity: each element connects to its neighbors in 2D grid.
    // Neighbors are first stored as global indices; remote ones are collected as ghosts.
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (idx_t li = 0; li < n_local; ++li) {
        const idx_t idx = begin + li;
        const int64_t x = static_cast<int64_t>(idx / n_root);
        const int64_t y = static_cast<int64_t>(idx % n_root);
        ElementStatic& elem = world.elements_static[li];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;

        // Connect to neighbors (up, down, left, right)
        for (int n = 0; n < 4; ++n) {
            const int64_t nx = x + offsets[n][0];
            const int64_t ny = y + offsets[n][1];

            // Check if neighbor is within bounds
            if (nx >= 0 && nx < side && ny >= 0 && ny < side) {
                const idx_t neighbor_idx = static_cast<idx_t>(nx) * n_root + static_cast<idx_t>(ny);
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
                if (neighbor_idx < begin || neighbor_idx >= end) {
                    world.halo.ghosts.push_back(neighbor_idx);
                }
            }
        }

        // Set corner elements as inflow/outflow to create interesting dynamics
        // (applied in the same order as the serial construction)
        if (idx == 0) elem.material_idx = INFLOW_MAT_ID;
        if (idx == last) elem.material_idx = OUTFLOW_MAT_ID;
        if (idx == last * n_root) elem.material_idx = OUTFLOW_MAT_ID;
        if (idx == last * n_root + last) elem.material_idx = INFLOW_MAT_ID;
    }

    // Ghosts are sorted, so they are grouped by owning rank in ascending order
    std::sort(world.halo.ghosts.begin(), world.halo.ghosts.end());
    world.halo.ghosts.erase(std::unique(world.halo.ghosts.begin(), world.halo.ghosts.end()),
                            world.halo.ghosts.end());
    const size_t n_ghost = world.halo.ghosts.size();

    // Remap neighbor references to the local+ghost layout and classify elements
    world.interior.clear();
    world.boundary.clear();
    for (idx_t li = 0; li < n_local; ++li) {
        ElementStatic& elem = world.elements_static[li];
        bool has_ghost = false;
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            const idx_t g = elem.connected_idx[j];
            if (g >= begin && g < end) {
                elem.connected_idx[j] = g - begin;
            } else {
                const auto it = std::lower_bound(world.halo.ghosts.begin(), world.halo.ghosts.end(), g);
                elem.connected_idx[j] = n_local + static_cast<idx_t>(it - world.halo.ghosts.begin());
                has_ghost = true;
            }
        }
        (has_ghost ? world.boundary : world.interior).push_back(li);
    }

    // Initialize all elements with zero energy (owned and ghost slots)
    world.elements_dynamic.assign(n_local + n_ghost, ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.assign(n_local + n_ghost, ElementDynamic{0.0, 0.0});
}

// Create the ghost exchange pattern: which ghosts we read from each owner, and which
// of our elements each other rank reads. Only neighbor ranks are involved.
void setupHalo(World& world, const idx_t n_elems, const int n_ranks, MPI_Comm base_comm) {
    HaloExchange& halo = world.halo;

    // Rank r owns global indices [starts[r], starts[r + 1])
    std::vector<idx_t> starts(n_ranks + 1);
    for (int r = 0; r <= n_ranks; ++r) {
        starts[r] = blockBegin(n_elems, n_ranks, r);
    }

    // How many ghosts we need from each owner
    std::vector<int> req_counts(n_ranks, 0);
    for (const idx_t g : halo.ghosts) {
        const int owner = static_cast<int>(std::upper_bound(starts.begin(), starts.end(), g) - starts.begin()) - 1;
        req_counts[owner]++;
    }
    const std::vector<int> req_displs = displacements(req_counts);

    // How many elements every other rank needs from us
    std::vector<int> need_counts(n_ranks);
    MPI_Alltoall(req_counts.data(), 1, MPI_INT, need_counts.data(), 1, MPI_INT, base_comm);
    const std::vector<int> need_displs = displacements(need_counts);
    const int total_need = std::accumulate(need_counts.begin(), need_counts.end(), 0);

    // Tell each owner which of its elements we need (global indices)
    std::vector<idx_t> requested(total_need);
    MPI_Alltoallv(halo.ghosts.data(), req_counts.data(), req_displs.data(), MPI_UINT64_T,
                  requested.data(), need_counts.data(), need_displs.data(), MPI_UINT64_T, base_comm);

    halo.send_list.resize(total_need);
    for (int k = 0; k < total_need; ++k) {
        halo.send_list[k] = requested[k] - world.begin;
    }

    // Neighbor lists for the distributed graph; displacements index the same packed layouts
    for (int r = 0; r < n_ranks; ++r) {
        if (req_counts[r] > 0) {
            halo.sources.push_back(r);
            halo.recv_counts.push_back(req_counts[r]);
            halo.recv_displs.push_back(req_displs[r]);
        }
        if (need_counts[r] > 0) {
            halo.destinations.push_back(r);
            halo.send_counts.push_back(need_counts[r]);
            halo.send_displs.push_back(need_displs[r]);
        }
    }

    halo.send_buf.resize(total_need);
    halo.recv_buf.resize(halo.ghosts.size());

    MPI_Dist_graph_create_adjacent(base_comm,
                                   static_cast<int>(halo.sources.size()), halo.sources.data(), MPI_UNWEIGHTED,
                                   static_cast<int>(halo.destinations.size()), halo.destinations.data(), MPI_UNWEIGHTED,
                                   MPI_INFO_NULL, 0, &halo.comm);
}

// Post the exchange of boundary energies (non-blocking)
void startHaloExchange(World& world, MPI_Request& req) {
    HaloExchange& halo = world.halo;
    for (size_t k = 0; k < halo.send_list.size(); ++k) {
        halo.send_buf[k] = world.elements_dynamic[halo.send_list[k]].current_energy;
    }
    MPI_Ineighbor_alltoallv(halo.send_buf.data(), halo.send_counts.data(), halo.send_displs.data(), MPI_DOUBLE,
                            halo.recv_buf.data(), halo.recv_counts.data(), halo.recv_displs.data(), MPI_DOUBLE,
                            halo.comm, &req);
}

// Complete the exchange and store the received energies in the ghost slots
void finishHaloExchange(World& world, MPI_Request& req) {
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    const idx_t n_local = world.elements_static.size();
    const HaloExchange& halo = world.halo;
    for (size_t g = 0; g < halo.recv_buf.size(); ++g) {
        world.elements_dynamic[n_local + g].current_energy = halo.recv_buf[g];
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update the listed owned elements: reads `cur` (including ghosts), writes `nxt`
void updateElements(const World& world, const ElementDynamic* cur, ElementDynamic* nxt,
                    const std::vector<idx_t>& list) {
    const ElementStatic* elements_static = world.elements_static.data();
    const Material* materials = world.materials.data();

    for (const idx_t i : list) {
        const ElementStatic& elem_static = elements_static[i];
        const ElementDynamic& elem_dyn = cur[i];
        const Material& mat = materials[elem_static.material_idx];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const ElementDynamic& neighbor_dyn = cur[elem_static.connected_idx[j]];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }

        // Update element state
        ElementDynamic& elem_write = nxt[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    MPI_Request halo_req;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Overlap: interior elements are computed while boundary energies are in flight
        startHaloExchange(world, halo_req);
        updateElements(world, world.elements_dynamic.data(), world.elements_dynamic_swap.data(), world.interior);
        finishHaloExchange(world, halo_req);
        updateElements(world, world.elements_dynamic.data(), world.elements_dynamic_swap.data(), world.boundary);

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (global reduction over all ranks)
bool validateResults(const World& world, const int rank, MPI_Comm comm) {
    const size_t n_local = world.elements_static.size();
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < n_local; ++i) {
        const ElementDynamic& elem = world.elements_dynamic[i];
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }

    val_t sums[2] = {energy_sum, flux_sum};
    MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, comm);
    energy_sum = sums[0];
    flux_sum = sums[1];
    MPI_Allreduce(MPI_IN_PLACE, &energy_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &energy_min, 1, MPI_DOUBLE, MPI_MIN, comm);

    const bool root = rank == 0;
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
        if (root) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
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

// Compute a simple hash of the results for verification.
// Each rank hashes its owned elements using global indices; the XOR combination is
// order-independent, so the reduced value equals the serial hash.
uint64_t computeHash(const World& world, MPI_Comm comm) {
    uint64_t hash = 0;
    const size_t n_local = world.elements_static.size();
    for (size_t li = 0; li < n_local; ++li) {
        const size_t i = world.begin + li;
        // Simple hash combining energy and flux values
        const ElementDynamic& elem = world.elements_dynamic[li];
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elem.current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elem.total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    MPI_Allreduce(MPI_IN_PLACE, &hash, 1, MPI_UINT64_T, MPI_BXOR, comm);
    return hash;
}

// Gather all element energies on rank 0 in global index order (empty on other ranks)
std::vector<val_t> gatherEnergies(const World& world, const idx_t n_elems, const int rank, const int n_ranks,
                                  MPI_Comm comm) {
    const size_t n_local = world.elements_static.size();
    std::vector<val_t> local(n_local);
    for (size_t li = 0; li < n_local; ++li) {
        local[li] = world.elements_dynamic[li].current_energy;
    }

    if (rank != 0) {
        MPI_Send(local.data(), static_cast<int>(n_local), MPI_DOUBLE, 0, 0, comm);
        return {};
    }

    std::vector<val_t> all(n_elems);
    std::copy(local.begin(), local.end(), all.begin());
    for (int r = 1; r < n_ranks; ++r) {
        const idx_t begin = blockBegin(n_elems, n_ranks, r);
        const idx_t count = blockBegin(n_elems, n_ranks, r + 1) - begin;
        MPI_Recv(all.data() + begin, static_cast<int>(count), MPI_DOUBLE, r, 0, comm, MPI_STATUS_IGNORE);
    }
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

int runBenchmark(int argc, char** argv, const int rank, const int n_ranks) {
    const bool root = rank == 0;
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

    if (n_elems_root < 1) {
        if (root) printf("Grid size must be at least 1\n");
        return 1;
    }

    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * n_elems_root;

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %llu elements\n", n_elems_root, n_elems_root,
               static_cast<unsigned long long>(n_elems));
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, rank, n_ranks);
    setupHalo(world, n_elems, n_ranks, MPI_COMM_WORLD);

    // Calculate memory usage (global totals, as in the serial version)
    if (root) {
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

    // Run simulation (timed across all ranks; the slowest rank determines the result)
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    runSimulation(world, n_iters);

    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        const auto duration_ms = static_cast<long>(seconds * 1000.0);
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
    const uint64_t hash = computeHash(world, MPI_COMM_WORLD);
    if (root) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation
    if (printResults) {
        const std::vector<val_t> energyData = gatherEnergies(world, n_elems, rank, n_ranks, MPI_COMM_WORLD);
        if (root) {
            print_results(energyData, "ElementEnergy");
        }
    }

    int rc = 0;
    if (validate) {
        if (!validateResults(world, rank, MPI_COMM_WORLD)) {
            rc = 1;
        }
    }

    MPI_Comm_free(&world.halo.comm);
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
