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

// Static connectivity information for each owned element
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Local indices (owned or ghost) of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Exchange pattern with one peer rank for ghost energies
struct Peer {
    int rank;
    int send_offset;  // Offset into Halo::send_buf
    int send_count;
    int recv_offset;  // Offset into Halo::recv_buf (ghost slots, in ghost order)
    int recv_count;
};

// Halo exchange plan for ghost elements
struct Halo {
    std::vector<Peer> peers;
    std::vector<idx_t> send_local;  // Owned local indices to pack, grouped by peer
    std::vector<val_t> send_buf;
    std::vector<val_t> recv_buf;
};

// Distributed world state: this rank owns the contiguous global range [first, first + n_owned)
struct World {
    idx_t n_global = 0;
    idx_t first = 0;
    idx_t n_owned = 0;
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;      // Owned elements
    std::vector<ElementDynamic> elements_dynamic;    // Owned elements followed by ghost copies
    std::vector<ElementDynamic> elements_dynamic_swap;
    std::vector<idx_t> interior;  // Owned elements whose neighbors are all owned
    std::vector<idx_t> boundary;  // Owned elements with at least one ghost neighbor
    Halo halo;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// First global element index owned by rank p in a balanced contiguous split of n elements
idx_t partitionStart(idx_t n, int nranks, int p) {
    const idx_t q = n / nranks;
    const idx_t rem = n % nranks;
    return static_cast<idx_t>(p) * q + std::min(static_cast<idx_t>(p), rem);
}

// Exchange ghost requests with peers so every rank knows which owned energies to send each iteration
void setupHalo(World& world, const std::vector<idx_t>& ghosts, int nranks) {
    // Ghosts are sorted by global index and owners are contiguous, so each owner's ghosts form one run
    std::vector<int> need_count(nranks, 0);
    int owner = 0;
    for (idx_t g : ghosts) {
        while (partitionStart(world.n_global, nranks, owner + 1) <= g) {
            ++owner;
        }
        need_count[owner]++;
    }

    std::vector<int> need_disp(nranks, 0);
    for (int r = 1; r < nranks; ++r) {
        need_disp[r] = need_disp[r - 1] + need_count[r - 1];
    }

    // Tell each owner which of its elements we need
    std::vector<int> give_count(nranks, 0);
    MPI_Alltoall(need_count.data(), 1, MPI_INT, give_count.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> give_disp(nranks, 0);
    for (int r = 1; r < nranks; ++r) {
        give_disp[r] = give_disp[r - 1] + give_count[r - 1];
    }
    const int total_give = give_disp[nranks - 1] + give_count[nranks - 1];

    std::vector<idx_t> requested(total_give);
    MPI_Alltoallv(ghosts.data(), need_count.data(), need_disp.data(), MPI_UINT64_T,
                  requested.data(), give_count.data(), give_disp.data(), MPI_UINT64_T,
                  MPI_COMM_WORLD);

    Halo& halo = world.halo;
    halo.send_local.resize(requested.size());
    for (size_t k = 0; k < requested.size(); ++k) {
        halo.send_local[k] = requested[k] - world.first;
    }
    halo.send_buf.resize(requested.size());
    halo.recv_buf.resize(ghosts.size());

    halo.peers.clear();
    for (int r = 0; r < nranks; ++r) {
        if (need_count[r] > 0 || give_count[r] > 0) {
            halo.peers.push_back(Peer{r, give_disp[r], give_count[r], need_disp[r], need_count[r]});
        }
    }
}

// Build the owned part of a 2D square grid as an unstructured mesh, with ghost copies of remote neighbors
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int nranks) {
    const idx_t n = n_elems_root;
    const idx_t last = n - 1;
    const idx_t n_global = n * n;
    const idx_t first = partitionStart(n_global, nranks, rank);
    const idx_t n_owned = partitionStart(n_global, nranks, rank + 1) - first;

    world.n_global = n_global;
    world.first = first;
    world.n_owned = n_owned;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate owned elements
    world.elements_static.resize(n_owned);

    // Build connectivity: each element connects to its neighbors in 2D grid
    // Neighbor indices are kept global until all ghosts are known
    std::vector<idx_t> ghosts;
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (idx_t i = 0; i < n_owned; ++i) {
        const idx_t g = first + i;
        const idx_t x = g / n;
        const idx_t y = g % n;
        ElementStatic& elem = world.elements_static[i];

        // Initialize with default material; corner elements are set as inflow/outflow below
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;

        // Connect to neighbors (up, down, left, right)
        for (const auto& off : offsets) {
            const long long nx = static_cast<long long>(x) + off[0];
            const long long ny = static_cast<long long>(y) + off[1];

            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_idx = static_cast<idx_t>(nx) * n + static_cast<idx_t>(ny);
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
                if (neighbor_idx < first || neighbor_idx >= first + n_owned) {
                    ghosts.push_back(neighbor_idx);
                }
            }
        }

        // Set corner elements as inflow/outflow to create interesting dynamics (same assignment order as serial)
        if (g == 0 * n + 0) elem.material_idx = INFLOW_MAT_ID;
        if (g == 0 * n + last) elem.material_idx = OUTFLOW_MAT_ID;
        if (g == last * n + 0) elem.material_idx = OUTFLOW_MAT_ID;
        if (g == last * n + last) elem.material_idx = INFLOW_MAT_ID;
    }

    // Ghost copies of remote neighbors, sorted by global index
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());

    // Translate neighbor indices to local indices: owned elements first, then ghosts
    for (idx_t i = 0; i < n_owned; ++i) {
        ElementStatic& elem = world.elements_static[i];
        bool has_ghost = false;
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            const idx_t g = elem.connected_idx[j];
            if (g >= first && g < first + n_owned) {
                elem.connected_idx[j] = g - first;
            } else {
                const auto it = std::lower_bound(ghosts.begin(), ghosts.end(), g);
                elem.connected_idx[j] = n_owned + static_cast<idx_t>(it - ghosts.begin());
                has_ghost = true;
            }
        }
        (has_ghost ? world.boundary : world.interior).push_back(i);
    }

    // Initialize all elements with zero energy
    world.elements_dynamic.assign(n_owned + ghosts.size(), ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.assign(n_owned + ghosts.size(), ElementDynamic{0.0, 0.0});

    setupHalo(world, ghosts, nranks);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update the listed owned elements from the current buffer into the swap buffer
void updateElements(World& world, const std::vector<idx_t>& list) {
    const ElementDynamic* cur = world.elements_dynamic.data();
    ElementDynamic* nxt = world.elements_dynamic_swap.data();

    for (idx_t i : list) {
        const ElementStatic& elem_static = world.elements_static[i];
        const ElementDynamic& elem_dyn = cur[i];
        const Material& mat = world.materials[elem_static.material_idx];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], cur[elem_static.connected_idx[j]]);
        }

        // Update element state
        ElementDynamic& elem_write = nxt[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations
// Ghost energies are exchanged with peers every iteration; interior elements are computed while messages are in flight
void runSimulation(World& world, const int n_iters) {
    Halo& halo = world.halo;
    std::vector<MPI_Request> requests(2 * halo.peers.size());

    for (int iter = 0; iter < n_iters; ++iter) {
        // Pack current energies that peers need
        for (size_t k = 0; k < halo.send_local.size(); ++k) {
            halo.send_buf[k] = world.elements_dynamic[halo.send_local[k]].current_energy;
        }

        int n_req = 0;
        for (const Peer& p : halo.peers) {
            if (p.recv_count > 0) {
                MPI_Irecv(halo.recv_buf.data() + p.recv_offset, p.recv_count, MPI_DOUBLE,
                          p.rank, 0, MPI_COMM_WORLD, &requests[n_req++]);
            }
        }
        for (const Peer& p : halo.peers) {
            if (p.send_count > 0) {
                MPI_Isend(halo.send_buf.data() + p.send_offset, p.send_count, MPI_DOUBLE,
                          p.rank, 0, MPI_COMM_WORLD, &requests[n_req++]);
            }
        }

        // Interior elements do not read ghosts, so update them while halos are in flight
        updateElements(world, world.interior);

        MPI_Waitall(n_req, requests.data(), MPI_STATUSES_IGNORE);

        for (size_t k = 0; k < halo.recv_buf.size(); ++k) {
            world.elements_dynamic[world.n_owned + k].current_energy = halo.recv_buf[k];
        }
        updateElements(world, world.boundary);

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (operates on the complete, globally ordered state)
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

// Hash contribution of one element; XOR-combining these over all elements gives the global hash
uint64_t elementHash(const ElementDynamic& elem, idx_t global_idx) {
    uint64_t e_bits;
    uint64_t f_bits;
    memcpy(&e_bits, &elem.current_energy, sizeof(e_bits));
    memcpy(&f_bits, &elem.total_flux, sizeof(f_bits));
    return ((e_bits + global_idx) * 0x9e3779b97f4a7c15ULL) ^ ((f_bits + global_idx) * 0xbf58476d1ce4e5b9ULL);
}

// Collect a block-distributed array on rank 0 in global index order
std::vector<val_t> gatherOnRoot(const std::vector<val_t>& local, idx_t n_global, int rank, int nranks, int tag) {
    constexpr idx_t kMaxChunk = idx_t{1} << 30;  // Keep each message within MPI's int count limit
    std::vector<val_t> global;

    if (rank == 0) {
        global.resize(n_global);
        std::copy(local.begin(), local.end(), global.begin());
        for (int r = 1; r < nranks; ++r) {
            const idx_t begin = partitionStart(n_global, nranks, r);
            const idx_t count = partitionStart(n_global, nranks, r + 1) - begin;
            for (idx_t done = 0; done < count; done += kMaxChunk) {
                const idx_t chunk = std::min(kMaxChunk, count - done);
                MPI_Recv(global.data() + begin + done, static_cast<int>(chunk), MPI_DOUBLE,
                         r, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        const idx_t count = static_cast<idx_t>(local.size());
        for (idx_t done = 0; done < count; done += kMaxChunk) {
            const idx_t chunk = std::min(kMaxChunk, count - done);
            MPI_Send(local.data() + done, static_cast<int>(chunk), MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        }
    }
    return global;
}

// Gather the full element state onto rank 0 (diagnostics only; empty on other ranks)
std::vector<ElementDynamic> gatherElements(const World& world, int rank, int nranks) {
    std::vector<val_t> energy(world.n_owned);
    std::vector<val_t> flux(world.n_owned);
    for (idx_t i = 0; i < world.n_owned; ++i) {
        energy[i] = world.elements_dynamic[i].current_energy;
        flux[i] = world.elements_dynamic[i].total_flux;
    }

    const std::vector<val_t> all_energy = gatherOnRoot(energy, world.n_global, rank, nranks, 1);
    const std::vector<val_t> all_flux = gatherOnRoot(flux, world.n_global, rank, nranks, 2);
    if (rank != 0) {
        return {};
    }

    std::vector<ElementDynamic> elements(world.n_global);
    for (idx_t i = 0; i < world.n_global; ++i) {
        elements[i] = ElementDynamic{all_energy[i], all_flux[i]};
    }
    return elements;
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

int runBenchmark(int argc, char** argv, int rank, int nranks) {
    const bool root = rank == 0;
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

    if (n_elems_root < 1) {
        if (root) printf("Grid size must be at least 1\n");
        return 1;
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Ranks: %d\n", nranks);
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    // Build the distributed unstructured mesh
    World world;
    buildSquare2D(world, n_elems_root, rank, nranks);

    // Calculate memory usage (for the complete mesh)
    const size_t static_mem = world.n_global * sizeof(ElementStatic);
    const size_t dynamic_mem = world.n_global * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (root) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        printf("Running simulation...\n");
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    runSimulation(world, n_iters);

    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long duration_ms = static_cast<long>(elapsed * 1000.0);

    if (root) {
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute hash for verification: XOR is order independent, so per-rank partial hashes combine exactly
    uint64_t local_hash = 0;
    for (idx_t i = 0; i < world.n_owned; ++i) {
        local_hash ^= elementHash(world.elements_dynamic[i], world.first + i);
    }
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (root) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    if (printResults || validate) {
        const std::vector<ElementDynamic> elements = gatherElements(world, rank, nranks);

        // Print results for external validation
        if (root && printResults) {
            std::vector<double> energyData;
            energyData.reserve(elements.size());
            for (const auto& elem : elements) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            int valid = (root && validateResults(elements)) ? 1 : 0;
            MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!valid) {
                return 1;
            }
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const int status = runBenchmark(argc, argv, rank, nranks);

    MPI_Finalize();
    return status;
}
