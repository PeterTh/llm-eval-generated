#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>
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

    // Elements are stored in global-index order in the half-open interval
    // [global_begin, global_begin + elements_static.size()).
    idx_t global_begin = 0;

    // A connection reference is a local element index when it is less than
    // elements_dynamic.size(); otherwise it is local_size + halo_slot.
    std::vector<idx_t> connection_refs;
    std::vector<val_t> halo_energy;

    // Communication is grouped by peer.  The offsets index halo_energy or
    // send_buffer, respectively.
    struct PeerPlan {
        int rank;
        std::size_t offset;
        std::size_t count;
    };
    std::vector<PeerPlan> recv_peers;
    std::vector<PeerPlan> send_peers;
    std::vector<std::size_t> send_local_indices;
    std::vector<val_t> send_buffer;

    // Interior work can run while halo messages are in flight.
    std::vector<std::size_t> interior_elements;
    std::vector<std::size_t> boundary_elements;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root,
                   const idx_t global_begin, const idx_t global_end) {
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.global_begin = global_begin;
    const std::size_t n_local = static_cast<std::size_t>(global_end - global_begin);

    // Allocate elements
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    // Initialize all elements with default material and zero energy
    for (std::size_t i = 0; i < n_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity: each element connects to its neighbors in 2D grid
    for (std::size_t local_idx = 0; local_idx < n_local; ++local_idx) {
        const idx_t global_idx = global_begin + static_cast<idx_t>(local_idx);
        const idx_t x = global_idx / static_cast<idx_t>(n_elems_root);
        const idx_t y = global_idx % static_cast<idx_t>(n_elems_root);
        ElementStatic& elem = world.elements_static[local_idx];

        // Connect to neighbors (up, down, left, right)
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        for (int n = 0; n < 4; ++n) {
            // Check if neighbor is within bounds.  The signed checks avoid
            // unsigned underflow for the first row and column.
            const int64_t signed_nx = static_cast<int64_t>(x) + offsets[n][0];
            const int64_t signed_ny = static_cast<int64_t>(y) + offsets[n][1];
            if (signed_nx >= 0 && signed_nx < n_elems_root &&
                signed_ny >= 0 && signed_ny < n_elems_root) {
                const idx_t neighbor_idx =
                    static_cast<idx_t>(signed_nx) * static_cast<idx_t>(n_elems_root) +
                    static_cast<idx_t>(signed_ny);
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    const idx_t last_row = static_cast<idx_t>(last) * static_cast<idx_t>(n_elems_root);
    for (std::size_t local_idx = 0; local_idx < n_local; ++local_idx) {
        const idx_t global_idx = global_begin + static_cast<idx_t>(local_idx);
        if (global_idx == 0) {
            world.elements_static[local_idx].material_idx = INFLOW_MAT_ID;
        }
        if (global_idx == static_cast<idx_t>(last)) {
            world.elements_static[local_idx].material_idx = OUTFLOW_MAT_ID;
        }
        if (global_idx == last_row) {
            world.elements_static[local_idx].material_idx = OUTFLOW_MAT_ID;
        }
        if (global_idx == last_row + static_cast<idx_t>(last)) {
            world.elements_static[local_idx].material_idx = INFLOW_MAT_ID;
        }
    }
}

std::vector<idx_t> makePartitionBounds(const idx_t n_elems, const int n_ranks) {
    std::vector<idx_t> bounds(static_cast<std::size_t>(n_ranks) + 1, 0);
    const idx_t base = n_elems / static_cast<idx_t>(n_ranks);
    const idx_t remainder = n_elems % static_cast<idx_t>(n_ranks);

    for (int rank = 0; rank < n_ranks; ++rank) {
        bounds[static_cast<std::size_t>(rank) + 1] =
            bounds[static_cast<std::size_t>(rank)] + base +
            (static_cast<idx_t>(rank) < remainder ? 1 : 0);
    }
    return bounds;
}

int ownerOf(const idx_t global_idx, const std::vector<idx_t>& bounds) {
    const auto it = std::upper_bound(bounds.begin(), bounds.end(), global_idx);
    return static_cast<int>(it - bounds.begin()) - 1;
}

// Construct the one-time halo exchange plan.  Each rank advertises the
// global IDs it needs; owners receive those requests and return the matching
// local values every iteration.  This also works for arbitrary connectivity,
// even though the benchmark's generated mesh has at most two peer ranks per
// contiguous partition in practice.
void setupCommunication(World& world, const std::vector<idx_t>& bounds,
                        const int n_ranks, MPI_Comm comm) {
    const idx_t n_local = static_cast<idx_t>(world.elements_static.size());
    const idx_t global_end = world.global_begin + n_local;
    const std::size_t n_local_size = static_cast<std::size_t>(n_local);

    world.connection_refs.assign(n_local_size * MAX_CONNECTIONS, 0);
    std::vector<std::vector<idx_t>> requests_by_owner(static_cast<std::size_t>(n_ranks));
    std::unordered_map<idx_t, unsigned char> remote_seen;
    remote_seen.reserve(n_local_size * 2 + 1);

    for (std::size_t local_idx = 0; local_idx < n_local_size; ++local_idx) {
        const ElementStatic& elem = world.elements_static[local_idx];
        const std::size_t ref_base = local_idx * MAX_CONNECTIONS;
        for (idx_t connection = 0; connection < elem.num_connections; ++connection) {
            const idx_t global_neighbor = elem.connected_idx[connection];
            if (global_neighbor >= world.global_begin && global_neighbor < global_end) {
                world.connection_refs[ref_base + static_cast<std::size_t>(connection)] =
                    global_neighbor - world.global_begin;
            } else {
                const int peer = ownerOf(global_neighbor, bounds);
                if (remote_seen.emplace(global_neighbor, 0).second) {
                    requests_by_owner[static_cast<std::size_t>(peer)].push_back(global_neighbor);
                }
            }
        }
    }

    // Assign contiguous halo slots in peer order.  The grouped layout lets
    // every receive write directly into the halo array without unpacking.
    std::unordered_map<idx_t, idx_t> halo_slot_by_global;
    halo_slot_by_global.reserve(remote_seen.size() + 1);
    world.recv_peers.clear();
    world.halo_energy.clear();
    for (int peer = 0; peer < n_ranks; ++peer) {
        const auto& requests = requests_by_owner[static_cast<std::size_t>(peer)];
        if (requests.empty()) {
            continue;
        }
        const std::size_t offset = world.halo_energy.size();
        world.recv_peers.push_back(World::PeerPlan{peer, offset, requests.size()});
        for (const idx_t global_idx : requests) {
            halo_slot_by_global.emplace(global_idx,
                                        static_cast<idx_t>(world.halo_energy.size()));
            world.halo_energy.push_back(0.0);
        }
    }

    // Convert remote global IDs in each connection into direct halo indices.
    for (std::size_t local_idx = 0; local_idx < n_local_size; ++local_idx) {
        const ElementStatic& elem = world.elements_static[local_idx];
        const std::size_t ref_base = local_idx * MAX_CONNECTIONS;
        for (idx_t connection = 0; connection < elem.num_connections; ++connection) {
            const idx_t global_neighbor = elem.connected_idx[connection];
            if (global_neighbor < world.global_begin || global_neighbor >= global_end) {
                const auto slot = halo_slot_by_global.find(global_neighbor);
                if (slot == halo_slot_by_global.end()) {
                    MPI_Abort(comm, 1);
                    return;
                }
                world.connection_refs[ref_base + static_cast<std::size_t>(connection)] =
                    n_local + slot->second;
            }
        }
    }

    world.interior_elements.clear();
    world.boundary_elements.clear();
    world.interior_elements.reserve(n_local_size);
    world.boundary_elements.reserve(n_local_size);
    for (std::size_t local_idx = 0; local_idx < n_local_size; ++local_idx) {
        const ElementStatic& elem = world.elements_static[local_idx];
        const std::size_t ref_base = local_idx * MAX_CONNECTIONS;
        bool has_remote_neighbor = false;
        for (idx_t connection = 0; connection < elem.num_connections; ++connection) {
            if (world.connection_refs[ref_base + static_cast<std::size_t>(connection)] >= n_local) {
                has_remote_neighbor = true;
                break;
            }
        }
        (has_remote_neighbor ? world.boundary_elements : world.interior_elements).push_back(local_idx);
    }

    // Exchange request lists.  This is a setup-only all-to-all; the hot loop
    // uses only the sparse peer lists created below.
    std::vector<int> send_counts(static_cast<std::size_t>(n_ranks), 0);
    std::vector<int> recv_counts(static_cast<std::size_t>(n_ranks), 0);
    for (int peer = 0; peer < n_ranks; ++peer) {
        const std::size_t count = requests_by_owner[static_cast<std::size_t>(peer)].size();
        if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            MPI_Abort(comm, 1);
            return;
        }
        send_counts[static_cast<std::size_t>(peer)] = static_cast<int>(count);
    }

    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    std::vector<int> send_displacements(static_cast<std::size_t>(n_ranks), 0);
    std::vector<int> recv_displacements(static_cast<std::size_t>(n_ranks), 0);
    int send_total = 0;
    int recv_total = 0;
    for (int peer = 0; peer < n_ranks; ++peer) {
        send_displacements[static_cast<std::size_t>(peer)] = send_total;
        recv_displacements[static_cast<std::size_t>(peer)] = recv_total;
        send_total += send_counts[static_cast<std::size_t>(peer)];
        recv_total += recv_counts[static_cast<std::size_t>(peer)];
    }

    std::vector<idx_t> send_ids(static_cast<std::size_t>(send_total));
    for (int peer = 0; peer < n_ranks; ++peer) {
        const auto& requests = requests_by_owner[static_cast<std::size_t>(peer)];
        std::copy(requests.begin(), requests.end(),
                  send_ids.begin() + send_displacements[static_cast<std::size_t>(peer)]);
    }
    std::vector<idx_t> received_ids(static_cast<std::size_t>(recv_total));
    MPI_Alltoallv(send_ids.empty() ? nullptr : send_ids.data(), send_counts.data(),
                  send_displacements.data(), MPI_UINT64_T,
                  received_ids.empty() ? nullptr : received_ids.data(), recv_counts.data(),
                  recv_displacements.data(), MPI_UINT64_T, comm);

    world.send_peers.clear();
    world.send_local_indices.clear();
    world.send_local_indices.reserve(static_cast<std::size_t>(recv_total));
    for (int peer = 0; peer < n_ranks; ++peer) {
        const int count = recv_counts[static_cast<std::size_t>(peer)];
        if (count == 0) {
            continue;
        }
        const std::size_t offset = world.send_local_indices.size();
        world.send_peers.push_back(World::PeerPlan{peer, offset, static_cast<std::size_t>(count)});
        for (int i = 0; i < count; ++i) {
            const idx_t global_idx = received_ids[
                static_cast<std::size_t>(recv_displacements[static_cast<std::size_t>(peer)] + i)];
            if (global_idx < world.global_begin || global_idx >= global_end) {
                MPI_Abort(comm, 1);
                return;
            }
            world.send_local_indices.push_back(
                static_cast<std::size_t>(global_idx - world.global_begin));
        }
    }
    world.send_buffer.resize(world.send_local_indices.size());
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void updateElements(World& world, const std::vector<std::size_t>& indices) {
    const std::size_t n_local = world.elements_static.size();
    const ElementDynamic* current = world.elements_dynamic.data();
    ElementDynamic* next = world.elements_dynamic_swap.data();

    for (const std::size_t i : indices) {
        const ElementStatic& elem_static = world.elements_static[i];
        const ElementDynamic& elem_dyn = current[i];
        const Material& mat = world.materials[elem_static.material_idx];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements.  Local references index the
        // current state directly; remote references index the received halo.
        const std::size_t ref_base = i * MAX_CONNECTIONS;
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t ref = world.connection_refs[ref_base + static_cast<std::size_t>(j)];
            const val_t neighbor_energy =
                ref < n_local ? current[static_cast<std::size_t>(ref)].current_energy
                              : world.halo_energy[static_cast<std::size_t>(ref - n_local)];
            total_flux += (neighbor_energy - elem_dyn.current_energy) *
                          mat.transfer_coeff * elem_static.connected_flux[j] * 0.25;
        }

        // Update element state
        next[i].current_energy = elem_dyn.current_energy + total_flux;
        next[i].total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

void runSimulation(World& world, const int n_iters, MPI_Comm comm) {
    std::vector<MPI_Request> requests;
    requests.reserve(world.recv_peers.size() + world.send_peers.size());

    for (int iter = 0; iter < n_iters; ++iter) {
        requests.clear();

        // Post receives first, then pack and post sends.  Each peer has one
        // message in each direction, so the fixed tag is safe after the
        // wait at the end of this iteration.
        for (const World::PeerPlan& peer : world.recv_peers) {
            MPI_Request request;
            MPI_Irecv(world.halo_energy.data() + peer.offset,
                      static_cast<int>(peer.count), MPI_DOUBLE, peer.rank, 0, comm, &request);
            requests.push_back(request);
        }

        for (std::size_t i = 0; i < world.send_local_indices.size(); ++i) {
            world.send_buffer[i] =
                world.elements_dynamic[world.send_local_indices[i]].current_energy;
        }
        for (const World::PeerPlan& peer : world.send_peers) {
            MPI_Request request;
            MPI_Isend(world.send_buffer.data() + peer.offset,
                      static_cast<int>(peer.count), MPI_DOUBLE, peer.rank, 0, comm, &request);
            requests.push_back(request);
        }

        // Interior elements do not depend on the halo and overlap useful
        // computation with network progress.
        updateElements(world, world.interior_elements);

        if (!requests.empty()) {
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        }
        updateElements(world, world.boundary_elements);

        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : world.elements_dynamic) {
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
uint64_t computeHash(const std::vector<ElementDynamic>& elements, const idx_t global_begin) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        const uint64_t global_idx = global_begin + static_cast<idx_t>(i);
        hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

std::vector<ElementDynamic> gatherDynamic(const World& world,
                                          const std::vector<idx_t>& bounds,
                                          const int rank, const int n_ranks,
                                          MPI_Comm comm) {
    const idx_t n_elems = bounds.back();
    if (n_elems > static_cast<idx_t>(std::numeric_limits<int>::max())) {
        MPI_Abort(comm, 1);
        return {};
    }

    std::vector<int> counts(static_cast<std::size_t>(n_ranks), 0);
    std::vector<int> displacements(static_cast<std::size_t>(n_ranks), 0);
    for (int peer = 0; peer < n_ranks; ++peer) {
        const idx_t count = bounds[static_cast<std::size_t>(peer) + 1] -
                            bounds[static_cast<std::size_t>(peer)];
        counts[static_cast<std::size_t>(peer)] = static_cast<int>(count);
        displacements[static_cast<std::size_t>(peer)] =
            static_cast<int>(bounds[static_cast<std::size_t>(peer)]);
    }

    static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t));
    MPI_Datatype dynamic_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &dynamic_type);
    MPI_Type_commit(&dynamic_type);

    std::vector<ElementDynamic> all_elements;
    if (rank == 0) {
        all_elements.resize(static_cast<std::size_t>(n_elems));
    }
    MPI_Gatherv(world.elements_dynamic.empty() ? nullptr : world.elements_dynamic.data(),
                counts[static_cast<std::size_t>(rank)], dynamic_type,
                rank == 0 && !all_elements.empty() ? all_elements.data() : nullptr,
                counts.data(), displacements.data(), dynamic_type, 0, comm);

    MPI_Type_free(&dynamic_type);
    return all_elements;
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
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    if (n_elems_root <= 0) {
        if (rank == 0) {
            printf("Grid size must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }

    const idx_t n_elems = static_cast<idx_t>(n_elems_root) *
                          static_cast<idx_t>(n_elems_root);
    const std::vector<idx_t> bounds = makePartitionBounds(n_elems, n_ranks);
    const idx_t global_begin = bounds[static_cast<std::size_t>(rank)];
    const idx_t global_end = bounds[static_cast<std::size_t>(rank) + 1];

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n",
               n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", n_ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Build the unstructured mesh
    if (rank == 0) {
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, global_begin, global_end);
    setupCommunication(world, bounds, n_ranks, MPI_COMM_WORLD);

    // Calculate memory usage
    const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) {
        printf("Memory usage (aggregate): %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        printf("Running simulation...\n");
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters, MPI_COMM_WORLD);
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Compute hash for verification
    const uint64_t local_hash = computeHash(world.elements_dynamic, world.global_begin);
    uint64_t result_hash = 0;
    MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long duration_ms = std::max(1L, static_cast<long>(elapsed * 1000.0));
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
        printf("  Result hash: %016" PRIX64 "\n", result_hash);
        printf("\n");
    }

    std::vector<ElementDynamic> all_elements;
    // Print results for external validation
    if (printResults || validate) {
        all_elements = gatherDynamic(world, bounds, rank, n_ranks, MPI_COMM_WORLD);
    }
    if (rank == 0 && printResults) {
        std::vector<double> energyData;
        energyData.reserve(all_elements.size());
        for (const auto& elem : all_elements) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }

    // Validation
    int valid = 1;
    if (validate) {
        if (rank == 0) {
            World gathered_world;
            gathered_world.elements_dynamic = std::move(all_elements);
            valid = validateResults(gathered_world) ? 1 : 0;
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return valid == 1 ? 0 : 1;
}
