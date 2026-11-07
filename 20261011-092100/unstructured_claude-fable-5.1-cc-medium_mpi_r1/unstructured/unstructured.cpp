#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

#include <mpi.h>
#include <sys/syscall.h>
#include <unistd.h>

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

// Halo exchange pattern: which owned elements to send to / ghosts to receive from each neighbor rank
struct HaloExchange {
    std::vector<int> send_ranks;                   // Neighbor ranks we send to
    std::vector<int> send_offsets;                 // Prefix offsets into send_local_idx (size+1)
    std::vector<idx_t> send_local_idx;             // Local indices of owned elements to send
    std::vector<val_t> send_buf;                   // Packed send buffer

    std::vector<int> recv_ranks;                   // Neighbor ranks we receive from
    std::vector<int> recv_offsets;                 // Prefix offsets into ghost region (size+1)

    std::vector<MPI_Request> requests;             // Persistent requests (recv then send)
};

// World state (distributed): each rank owns a contiguous range of global element indices
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;        // Owned elements only
    std::vector<ElementDynamic> elements_dynamic;      // Owned elements followed by ghost elements
    std::vector<ElementDynamic> elements_dynamic_swap;

    idx_t n_global = 0;        // Total number of elements across all ranks
    idx_t global_begin = 0;    // First global index owned by this rank
    idx_t n_owned = 0;         // Number of owned elements
    idx_t n_ghost = 0;         // Number of ghost elements

    std::vector<std::pair<idx_t, idx_t>> interior_ranges;  // Contiguous [begin,end) runs of owned
                                                           // elements with only owned neighbors
    std::vector<idx_t> boundary_elems;   // Owned elements with at least one ghost neighbor

    HaloExchange halo;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Contiguous block partitioning of [0, n_global) over ranks
inline idx_t partitionBegin(idx_t n_global, int size, int rank) {
    return (n_global * static_cast<idx_t>(rank)) / static_cast<idx_t>(size);
}

inline int ownerOf(idx_t gidx, idx_t n_global, int size) {
    // Initial guess from proportional mapping, then correct for rounding
    int r = static_cast<int>((static_cast<__uint128_t>(gidx) * static_cast<idx_t>(size)) / n_global);
    if (r >= size) r = size - 1;
    while (r > 0 && partitionBegin(n_global, size, r) > gidx) --r;
    while (r + 1 < size && partitionBegin(n_global, size, r + 1) <= gidx) ++r;
    return r;
}

// Build the halo exchange pattern from the (global-indexed) connectivity of the owned elements,
// renumber connectivity to local indices, and classify owned elements as interior/boundary.
void setupHalo(World& world, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const idx_t gbegin = world.global_begin;
    const idx_t gend = gbegin + world.n_owned;

    // Collect ghost global indices per owner rank
    std::vector<std::vector<idx_t>> ghost_by_rank(size);
    for (idx_t i = 0; i < world.n_owned; ++i) {
        const ElementStatic& es = world.elements_static[i];
        for (idx_t j = 0; j < es.num_connections; ++j) {
            const idx_t g = es.connected_idx[j];
            if (g < gbegin || g >= gend) {
                ghost_by_rank[ownerOf(g, world.n_global, size)].push_back(g);
            }
        }
    }

    // Sorted unique ghost list per rank; assign ghost local indices in rank order
    HaloExchange& h = world.halo;
    std::unordered_map<idx_t, idx_t> ghost_local;  // global -> local index
    idx_t next_local = world.n_owned;
    h.recv_offsets.push_back(0);
    std::vector<int> recv_counts(size, 0);
    std::vector<idx_t> recv_global_all;
    for (int r = 0; r < size; ++r) {
        auto& v = ghost_by_rank[r];
        if (v.empty()) continue;
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        h.recv_ranks.push_back(r);
        for (idx_t g : v) {
            ghost_local.emplace(g, next_local++);
            recv_global_all.push_back(g);
        }
        recv_counts[r] = static_cast<int>(v.size());
        h.recv_offsets.push_back(static_cast<int>(recv_global_all.size()));
    }
    world.n_ghost = next_local - world.n_owned;

    // Tell owners which of their elements we need
    std::vector<int> send_counts(size, 0);
    MPI_Alltoall(recv_counts.data(), 1, MPI_INT, send_counts.data(), 1, MPI_INT, comm);

    std::vector<int> recv_displs(size + 1, 0), send_displs(size + 1, 0);
    for (int r = 0; r < size; ++r) {
        recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
        send_displs[r + 1] = send_displs[r] + send_counts[r];
    }
    std::vector<idx_t> send_global_all(send_displs[size]);
    MPI_Alltoallv(recv_global_all.data(), recv_counts.data(), recv_displs.data(), MPI_UINT64_T,
                  send_global_all.data(), send_counts.data(), send_displs.data(), MPI_UINT64_T, comm);

    h.send_offsets.push_back(0);
    for (int r = 0; r < size; ++r) {
        if (send_counts[r] == 0) continue;
        h.send_ranks.push_back(r);
        for (int k = send_displs[r]; k < send_displs[r + 1]; ++k) {
            h.send_local_idx.push_back(send_global_all[k] - gbegin);
        }
        h.send_offsets.push_back(static_cast<int>(h.send_local_idx.size()));
    }
    h.send_buf.resize(h.send_local_idx.size());

    // Renumber connectivity to local indices and classify elements
    idx_t run_begin = 0;
    bool in_run = false;
    for (idx_t i = 0; i < world.n_owned; ++i) {
        ElementStatic& es = world.elements_static[i];
        bool has_ghost = false;
        for (idx_t j = 0; j < es.num_connections; ++j) {
            const idx_t g = es.connected_idx[j];
            if (g >= gbegin && g < gend) {
                es.connected_idx[j] = g - gbegin;
            } else {
                es.connected_idx[j] = ghost_local[g];
                has_ghost = true;
            }
        }
        if (has_ghost) {
            world.boundary_elems.push_back(i);
            if (in_run) {
                world.interior_ranges.emplace_back(run_begin, i);
                in_run = false;
            }
        } else if (!in_run) {
            run_begin = i;
            in_run = true;
        }
    }
    if (in_run) world.interior_ranges.emplace_back(run_begin, world.n_owned);

    // Resize dynamic arrays to include ghosts (ghosts initialized to zero energy)
    world.elements_dynamic.resize(world.n_owned + world.n_ghost, ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.resize(world.n_owned + world.n_ghost, ElementDynamic{0.0, 0.0});
}

// Build a 2D square grid as an unstructured mesh (local portion of a block partition)
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);
    world.n_global = n_elems;
    world.global_begin = partitionBegin(n_elems, size, rank);
    world.n_owned = partitionBegin(n_elems, size, rank + 1) - world.global_begin;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate owned elements
    world.elements_static.resize(world.n_owned);
    world.elements_dynamic.resize(world.n_owned);
    world.elements_dynamic_swap.resize(world.n_owned);

    // Initialize all owned elements with default material and zero energy
    for (idx_t i = 0; i < world.n_owned; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity (global indices): each element connects to its neighbors in 2D grid
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (idx_t i = 0; i < world.n_owned; ++i) {
        const idx_t gidx = world.global_begin + i;
        const int x = static_cast<int>(gidx / n_elems_root);
        const int y = static_cast<int>(gidx % n_elems_root);
        ElementStatic& elem = world.elements_static[i];

        // Connect to neighbors (up, down, left, right)
        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];

            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_idx = static_cast<idx_t>(nx) * n_elems_root + ny;
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const idx_t last = n_elems_root - 1;
    auto setMaterial = [&](idx_t gidx, idx_t mat) {
        if (gidx >= world.global_begin && gidx < world.global_begin + world.n_owned) {
            world.elements_static[gidx - world.global_begin].material_idx = mat;
        }
    };
    setMaterial(0 * n_elems_root + 0, INFLOW_MAT_ID);
    setMaterial(0 * n_elems_root + last, OUTFLOW_MAT_ID);
    setMaterial(last * n_elems_root + 0, OUTFLOW_MAT_ID);
    setMaterial(last * n_elems_root + last, INFLOW_MAT_ID);

    setupHalo(world, comm);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update a contiguous range [begin,end) of owned elements
// (reads dyn_read, writes dyn_write; pointers are hoisted so the loop body matches the serial code)
inline void updateRange(const ElementStatic* elements_static,
                        const ElementDynamic* dyn_read,
                        ElementDynamic* dyn_write,
                        const Material* materials,
                        const idx_t begin, const idx_t end) {
    for (idx_t i = begin; i < end; ++i) {
        const ElementStatic& elem_static = elements_static[i];
        const ElementDynamic& elem_dyn = dyn_read[i];
        const Material& mat = materials[elem_static.material_idx];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            const ElementDynamic& neighbor_dyn = dyn_read[neighbor_idx];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }

        // Update element state
        ElementDynamic& elem_write = dyn_write[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Ghost current_energy values live in the dynamic arrays after the owned elements; exchange them
// using the derived datatype (stride of ElementDynamic) directly into/out of the given buffer.
void startHaloExchange(World& world, MPI_Comm comm) {
    HaloExchange& h = world.halo;
    const int n_recv = static_cast<int>(h.recv_ranks.size());
    const int n_send = static_cast<int>(h.send_ranks.size());
    h.requests.resize(n_recv + n_send);

    // Pack send buffer with owned current_energy values
    const ElementDynamic* dyn = world.elements_dynamic.data();
    for (size_t k = 0; k < h.send_local_idx.size(); ++k) {
        h.send_buf[k] = dyn[h.send_local_idx[k]].current_energy;
    }

    // Ghost energies are received directly into the ghost slots of elements_dynamic (this
    // iteration's read buffer) using a strided derived type, avoiding an unpack pass.
    static MPI_Datatype ghost_type = MPI_DATATYPE_NULL;
    if (ghost_type == MPI_DATATYPE_NULL) {
        MPI_Datatype tmp;
        MPI_Type_create_resized(MPI_DOUBLE, 0, sizeof(ElementDynamic), &tmp);
        MPI_Type_commit(&tmp);
        ghost_type = tmp;
    }

    for (int k = 0; k < n_recv; ++k) {
        const int count = h.recv_offsets[k + 1] - h.recv_offsets[k];
        MPI_Irecv(&world.elements_dynamic[world.n_owned + h.recv_offsets[k]].current_energy, count,
                  ghost_type, h.recv_ranks[k], 0, comm, &h.requests[k]);
    }
    for (int k = 0; k < n_send; ++k) {
        const int count = h.send_offsets[k + 1] - h.send_offsets[k];
        MPI_Isend(h.send_buf.data() + h.send_offsets[k], count, MPI_DOUBLE, h.send_ranks[k], 0, comm,
                  &h.requests[n_recv + k]);
    }
}

void finishHaloExchange(World& world) {
    HaloExchange& h = world.halo;
    if (!h.requests.empty()) {
        MPI_Waitall(static_cast<int>(h.requests.size()), h.requests.data(), MPI_STATUSES_IGNORE);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, MPI_Comm comm) {
    for (int iter = 0; iter < n_iters; ++iter) {
        // Start exchanging ghost energies for this iteration
        startHaloExchange(world, comm);

        const ElementStatic* es = world.elements_static.data();
        const ElementDynamic* dyn_read = world.elements_dynamic.data();
        ElementDynamic* dyn_write = world.elements_dynamic_swap.data();
        const Material* mats = world.materials.data();

        // Update interior elements while communication is in flight
        for (const auto& [rbegin, rend] : world.interior_ranges) {
            updateRange(es, dyn_read, dyn_write, mats, rbegin, rend);
        }

        // Wait for ghosts, then update boundary elements
        finishHaloExchange(world);
        for (const idx_t i : world.boundary_elems) {
            updateRange(es, dyn_read, dyn_write, mats, i, i + 1);
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (on the gathered global element array)
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
// Computed locally over owned elements using global indices; the per-element terms are combined
// with XOR, so partial hashes are reduced across ranks with MPI_BXOR to obtain the exact serial hash.
uint64_t computeHash(const std::vector<ElementDynamic>& elements, const idx_t n_owned, const idx_t global_begin) {
    uint64_t hash = 0;
    for (idx_t li = 0; li < n_owned; ++li) {
        const idx_t i = global_begin + li;
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[li].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[li].total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

// Gather the owned dynamic state of all ranks into a global array on rank 0
std::vector<ElementDynamic> gatherDynamic(const World& world, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<int> counts(size), displs(size + 1, 0);
    for (int r = 0; r < size; ++r) {
        counts[r] = static_cast<int>((partitionBegin(world.n_global, size, r + 1) -
                                      partitionBegin(world.n_global, size, r)) * 2);
        displs[r + 1] = displs[r] + counts[r];
    }
    std::vector<ElementDynamic> all;
    if (rank == 0) all.resize(world.n_global);
    MPI_Gatherv(world.elements_dynamic.data(), static_cast<int>(world.n_owned * 2), MPI_DOUBLE,
                all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
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
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const bool is_root = (rank == 0);

    // Pin this rank's memory policy to "allocate on the local NUMA node" (MPOL_LOCAL = 4).
    // Ranks are normally core-bound and first-touch their own data, so the kernel's automatic
    // NUMA balancing only adds periodic hint faults and page migrations to the hot loop.
    // Without an explicit policy the balancer scans the process; with one it leaves it alone.
#ifdef SYS_set_mempolicy
    syscall(SYS_set_mempolicy, 4 /* MPOL_LOCAL */, nullptr, 0UL);
#endif

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

    // Build the unstructured mesh (distributed)
    if (is_root) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root, comm);

    // Calculate memory usage (summed over all ranks)
    const size_t static_mem_local = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem_local = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    unsigned long long mem_local[2] = {static_mem_local, dynamic_mem_local};
    unsigned long long mem_total[2] = {0, 0};
    MPI_Reduce(mem_local, mem_total, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    if (is_root) {
        const size_t static_mem = mem_total[0];
        const size_t dynamic_mem = mem_total[1];
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run simulation
    if (is_root) printf("Running simulation...\n");
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, comm);

    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Compute hash for verification (exact XOR-reduction of per-rank partial hashes)
    const uint64_t local_hash = computeHash(world.elements_dynamic, world.n_owned, world.global_begin);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);

    if (is_root) {
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

    int ret = 0;
    if (printResults || validate) {
        // Gather the full element state to rank 0 so that the output is identical to the serial code
        std::vector<ElementDynamic> all = gatherDynamic(world, comm);

        if (is_root) {
            // Print results for external validation
            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(all.size());
                for (const auto& elem : all) {
                    energyData.push_back(elem.current_energy);
                }
                print_results(energyData, "ElementEnergy");
            }

            // Validation
            if (validate) {
                bool valid = validateResults(all);
                if (!valid) {
                    ret = 1;
                }
            }
        }
        MPI_Bcast(&ret, 1, MPI_INT, 0, comm);
    }

    MPI_Finalize();
    return ret;
}
