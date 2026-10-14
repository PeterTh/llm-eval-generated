#include <algorithm>
#include <cinttypes>
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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (local slots)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Contiguous range [first, last) of owned element indices
struct Span {
    idx_t first;
    idx_t last;
};

// Communication with one peer rank
// For recv halos, offset indexes recv_buf; for send halos, it indexes send_indices and send_buf
struct Halo {
    int peer;
    idx_t offset;
    idx_t count;
};

// World state, restricted to the contiguous block of elements owned by this rank
struct World {
    std::vector<idx_t> block_start;  // First global element of each rank, followed by n_global
    idx_t begin = 0;                 // First global element owned by this rank
    idx_t n_local = 0;               // Number of elements owned by this rank
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    // Owned elements followed by ghost slots holding copies of remote neighbor energies
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    // Owned elements whose neighbors are all local (computed while halos are in flight) and the rest
    std::vector<Span> interior;
    std::vector<Span> boundary;
    // Halo exchange plan
    std::vector<Halo> recv_halos;
    std::vector<Halo> send_halos;
    std::vector<idx_t> send_indices;
    std::vector<val_t> send_buf;
    std::vector<val_t> recv_buf;     // Ghost energies, in ghost slot order
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Rank that owns global element g
int ownerOf(const std::vector<idx_t>& block_start, idx_t g) {
    return static_cast<int>(std::upper_bound(block_start.begin(), block_start.end(), g) - block_start.begin()) - 1;
}

// Set up the halo exchange plan. ghost lists the remote global elements this rank reads, sorted
void setupHaloExchange(World& world, const std::vector<idx_t>& ghost, const int n_ranks) {
    // Ghosts are sorted by global index and owners are contiguous, so each owner's ghosts form one range
    std::vector<int> request_counts(n_ranks, 0);  // Number of elements we need from each owner
    world.recv_halos.clear();
    size_t k = 0;
    while (k < ghost.size()) {
        const int owner = ownerOf(world.block_start, ghost[k]);
        const size_t start = k;
        while (k < ghost.size() && ghost[k] < world.block_start[owner + 1]) {
            ++k;
        }
        world.recv_halos.push_back(Halo{owner, start, k - start});
        request_counts[owner] = static_cast<int>(k - start);
    }

    // Tell each owner which of its elements we need, and learn which of ours other ranks need
    std::vector<int> serve_counts(n_ranks, 0);
    MPI_Alltoall(request_counts.data(), 1, MPI_INT, serve_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> request_displs(n_ranks, 0);
    std::vector<int> serve_displs(n_ranks, 0);
    for (int r = 1; r < n_ranks; ++r) {
        request_displs[r] = request_displs[r - 1] + request_counts[r - 1];
        serve_displs[r] = serve_displs[r - 1] + serve_counts[r - 1];
    }
    const int total_serve = serve_displs[n_ranks - 1] + serve_counts[n_ranks - 1];

    std::vector<idx_t> requested(total_serve);
    MPI_Alltoallv(ghost.data(), request_counts.data(), request_displs.data(), MPI_UINT64_T,
                  requested.data(), serve_counts.data(), serve_displs.data(), MPI_UINT64_T, MPI_COMM_WORLD);

    world.send_indices.resize(total_serve);
    for (int i = 0; i < total_serve; ++i) {
        world.send_indices[i] = requested[i] - world.begin;
    }

    world.send_halos.clear();
    for (int r = 0; r < n_ranks; ++r) {
        if (serve_counts[r] > 0) {
            world.send_halos.push_back(Halo{r, static_cast<idx_t>(serve_displs[r]), static_cast<idx_t>(serve_counts[r])});
        }
    }

    world.send_buf.assign(total_serve, 0.0);
    world.recv_buf.assign(ghost.size(), 0.0);
}

// Build the local block of a 2D square grid as an unstructured mesh
// Each rank builds only the contiguous range of elements it owns
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int n_ranks) {
    const idx_t n = static_cast<idx_t>(n_elems_root);
    const idx_t n_global = n * n;

    // Initialize materials (replicated on every rank)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Contiguous block partition of the global element range
    world.block_start.resize(n_ranks + 1);
    for (int r = 0; r <= n_ranks; ++r) {
        world.block_start[r] = n_global * static_cast<idx_t>(r) / n_ranks;
    }
    world.begin = world.block_start[rank];
    world.n_local = world.block_start[rank + 1] - world.begin;
    const idx_t lo = world.begin;
    const idx_t hi = lo + world.n_local;
    const idx_t n_local = world.n_local;

    world.elements_static.resize(n_local);

    // Build connectivity with global indices; remote neighbors are remapped to ghost slots below
    std::vector<idx_t> ghost;
    const long long n_signed = n_elems_root;
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (idx_t li = 0; li < n_local; ++li) {
        const idx_t g = lo + li;
        const long long x = static_cast<long long>(g / n);
        const long long y = static_cast<long long>(g % n);
        ElementStatic& elem = world.elements_static[li];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;

        // Connect to neighbors (up, down, left, right)
        for (int d = 0; d < 4; ++d) {
            const long long nx = x + offsets[d][0];
            const long long ny = y + offsets[d][1];

            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_signed && ny >= 0 && ny < n_signed) {
                const idx_t neighbor_idx = static_cast<idx_t>(nx * n_signed + ny);
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
                if (neighbor_idx < lo || neighbor_idx >= hi) {
                    ghost.push_back(neighbor_idx);
                }
            }
        }
    }

    // Ghost slots follow the owned elements, in global index order
    std::sort(ghost.begin(), ghost.end());
    ghost.erase(std::unique(ghost.begin(), ghost.end()), ghost.end());
    const idx_t n_ghost = ghost.size();

    for (idx_t li = 0; li < n_local; ++li) {
        ElementStatic& elem = world.elements_static[li];
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            const idx_t neighbor_idx = elem.connected_idx[j];
            if (neighbor_idx >= lo && neighbor_idx < hi) {
                elem.connected_idx[j] = neighbor_idx - lo;
            } else {
                const auto it = std::lower_bound(ghost.begin(), ghost.end(), neighbor_idx);
                elem.connected_idx[j] = n_local + static_cast<idx_t>(it - ghost.begin());
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const idx_t last = n - 1;
    const idx_t corners[4] = {0 * n + 0, 0 * n + last, last * n + 0, last * n + last};
    const idx_t corner_mat[4] = {INFLOW_MAT_ID, OUTFLOW_MAT_ID, OUTFLOW_MAT_ID, INFLOW_MAT_ID};
    for (int c = 0; c < 4; ++c) {
        if (corners[c] >= lo && corners[c] < hi) {
            world.elements_static[corners[c] - lo].material_idx = corner_mat[c];
        }
    }

    // Allocate elements, initialized with zero energy
    world.elements_dynamic.assign(n_local + n_ghost, ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.assign(n_local + n_ghost, ElementDynamic{0.0, 0.0});

    // Split owned elements by whether they depend on ghost values, stored as contiguous spans
    for (idx_t li = 0; li < n_local; ++li) {
        const ElementStatic& elem = world.elements_static[li];
        bool has_ghost = false;
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            has_ghost = has_ghost || elem.connected_idx[j] >= n_local;
        }
        std::vector<Span>& spans = has_ghost ? world.boundary : world.interior;
        if (!spans.empty() && spans.back().last == li) {
            spans.back().last = li + 1;
        } else {
            spans.push_back(Span{li, li + 1});
        }
    }

    setupHaloExchange(world, ghost, n_ranks);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update one owned element from the current state, writing into the next state
inline void updateElement(const ElementStatic* statics, const Material* materials, idx_t i,
                          const ElementDynamic* cur, ElementDynamic* nxt) {
    const ElementStatic& elem_static = statics[i];
    const ElementDynamic& elem_dyn = cur[i];
    const Material& mat = materials[elem_static.material_idx];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const idx_t neighbor_idx = elem_static.connected_idx[j];
        total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], cur[neighbor_idx]);
    }

    // Update element state
    ElementDynamic& elem_write = nxt[i];
    elem_write.current_energy = elem_dyn.current_energy + total_flux;
    elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
}

// Update all elements in the given spans
void updateSpans(const World& world, const std::vector<Span>& spans, const ElementDynamic* cur, ElementDynamic* nxt) {
    const ElementStatic* statics = world.elements_static.data();
    const Material* materials = world.materials.data();
    for (const Span& span : spans) {
        for (idx_t i = span.first; i < span.last; ++i) {
            updateElement(statics, materials, i, cur, nxt);
        }
    }
}

// Start sending owned energies to neighbors and receiving their ghost energies
void postHaloExchange(World& world, std::vector<MPI_Request>& requests) {
    requests.clear();
    for (const Halo& h : world.recv_halos) {
        requests.emplace_back();
        MPI_Irecv(world.recv_buf.data() + h.offset, static_cast<int>(h.count), MPI_DOUBLE,
                  h.peer, 0, MPI_COMM_WORLD, &requests.back());
    }
    const ElementDynamic* cur = world.elements_dynamic.data();
    for (const Halo& h : world.send_halos) {
        for (idx_t k = 0; k < h.count; ++k) {
            world.send_buf[h.offset + k] = cur[world.send_indices[h.offset + k]].current_energy;
        }
        requests.emplace_back();
        MPI_Isend(world.send_buf.data() + h.offset, static_cast<int>(h.count), MPI_DOUBLE,
                  h.peer, 0, MPI_COMM_WORLD, &requests.back());
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const idx_t n_ghost = world.recv_buf.size();
    std::vector<MPI_Request> requests;
    requests.reserve(world.recv_halos.size() + world.send_halos.size());

    for (int iter = 0; iter < n_iters; ++iter) {
        postHaloExchange(world, requests);

        // Elements without remote neighbors do not need to wait for the exchange
        ElementDynamic* cur = world.elements_dynamic.data();
        ElementDynamic* nxt = world.elements_dynamic_swap.data();
        updateSpans(world, world.interior, cur, nxt);

        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

        // Refresh ghost slots with the neighbors' energies from this iteration
        for (idx_t k = 0; k < n_ghost; ++k) {
            cur[world.n_local + k].current_energy = world.recv_buf[k];
        }

        updateSpans(world, world.boundary, cur, nxt);

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Gather the owned state of all ranks onto rank 0 in global element order
// Only rank 0 receives a populated vector
std::vector<ElementDynamic> gatherDynamic(const World& world, const int rank, const int n_ranks) {
    MPI_Datatype elem_type;
    MPI_Type_contiguous(static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE, &elem_type);
    MPI_Type_commit(&elem_type);

    std::vector<int> counts(n_ranks);
    std::vector<int> displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        counts[r] = static_cast<int>(world.block_start[r + 1] - world.block_start[r]);
        displs[r] = static_cast<int>(world.block_start[r]);
    }

    std::vector<ElementDynamic> full(rank == 0 ? world.block_start[n_ranks] : 0);
    MPI_Gatherv(world.elements_dynamic.data(), counts[rank], elem_type,
                full.data(), counts.data(), displs.data(), elem_type, 0, MPI_COMM_WORLD);

    MPI_Type_free(&elem_type);
    return full;
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

// Compute a simple hash of the results for verification
// Each term is XORed into the hash, so the hash of a block of elements can be combined with others in any order
uint64_t computeHash(const ElementDynamic* elements, const idx_t count, const idx_t offset) {
    uint64_t hash = 0;
    for (idx_t i = 0; i < count; ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + offset + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + offset + i) * 0xbf58476d1ce4e5b9ULL;
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

int runBenchmark(int argc, char** argv, const int rank, const int n_ranks) {
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (n_elems_root <= 0) {
        if (rank == 0) {
            printf("Grid size must be positive\n");
            printUsage(argv[0]);
        }
        return 1;
    }

    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, rank, n_ranks);

    // Calculate memory usage of the full mesh
    if (rank == 0) {
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

    // Run simulation, timed by the slowest rank
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    runSimulation(world, n_iters);

    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long duration_ms = static_cast<long>(elapsed * 1000.0);

    // Compute hash for verification (XOR across ranks)
    const uint64_t local_hash = computeHash(world.elements_dynamic.data(), world.n_local, world.begin);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
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
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Gather full results for external printing or validation
    std::vector<ElementDynamic> full;
    if (printResults || validate) {
        full = gatherDynamic(world, rank, n_ranks);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        std::vector<double> energyData;
        energyData.reserve(full.size());
        for (const auto& elem : full) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }

    // Validation
    int exit_code = 0;
    if (validate) {
        int valid = rank == 0 ? validateResults(full) : 0;
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!valid) {
            exit_code = 1;
        }
    }

    return exit_code;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    const int status = runBenchmark(argc, argv, rank, n_ranks);

    MPI_Finalize();
    return status;
}
