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

// World state
//
// The mesh is distributed across MPI ranks with a 1D row decomposition of the
// underlying grid (element index = x * n_elems_root + y, so a "row" of constant
// x is a contiguous block of elements). Each rank stores the elements it owns
// plus one halo row on each side that is shared with its neighbouring ranks.
// Element connectivity is expressed entirely in local indices, so the compute
// kernel is identical to the serial one.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;

    // Decomposition description
    int n_elems_root = 0;   // global grid edge length
    int row_begin = 0;      // first globally owned row
    int row_end = 0;        // one past last globally owned row
    int n_rows = 0;         // number of owned rows (row_end - row_begin)
    int halo_lo = 0;        // 1 if a halo row is stored below the owned rows
    int halo_hi = 0;        // 1 if a halo row is stored above the owned rows
    int rank_lo = MPI_PROC_NULL;  // rank owning the row below our first row
    int rank_hi = MPI_PROC_NULL;  // rank owning the row above our last row

    // Halo exchange buffers (one grid row of energies each)
    std::vector<val_t> send_lo, send_hi, recv_lo, recv_hi;

    // Local element index of the first owned element
    size_t ownedOffset() const { return static_cast<size_t>(halo_lo) * n_elems_root; }
    // Number of locally stored elements (owned + halo)
    size_t localSize() const {
        return static_cast<size_t>(halo_lo + n_rows + halo_hi) * n_elems_root;
    }
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Block distribution of `n` rows over `n_ranks` ranks: rank r owns rows
// [rowBegin(r), rowBegin(r+1)). The first `n % n_ranks` ranks get one extra row.
static inline int rowBegin(int n, int n_ranks, int r) {
    const int base = n / n_ranks;
    const int rem = n % n_ranks;
    return r * base + std::min(r, rem);
}

// Rank owning global row x (x must be in [0, n))
static inline int rowOwner(int n, int n_ranks, int x) {
    const int base = n / n_ranks;
    const int rem = n % n_ranks;
    if (base == 0) return x;  // more ranks than rows: rank r owns row r
    const int split = rem * (base + 1);
    return (x < split) ? x / (base + 1) : rem + (x - split) / base;
}

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int n_ranks) {
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Determine this rank's slab of grid rows
    world.n_elems_root = n_elems_root;
    world.row_begin = rowBegin(n_elems_root, n_ranks, rank);
    world.row_end = rowBegin(n_elems_root, n_ranks, rank + 1);
    world.n_rows = world.row_end - world.row_begin;

    if (world.n_rows > 0) {
        world.halo_lo = (world.row_begin > 0) ? 1 : 0;
        world.halo_hi = (world.row_end < n_elems_root) ? 1 : 0;
        if (world.halo_lo) world.rank_lo = rowOwner(n_elems_root, n_ranks, world.row_begin - 1);
        if (world.halo_hi) world.rank_hi = rowOwner(n_elems_root, n_ranks, world.row_end);
    }

    const size_t n_local = world.localSize();

    // Allocate elements
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < n_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        world.elements_dynamic_swap[i].current_energy = 0.0;
        world.elements_dynamic_swap[i].total_flux = 0.0;
    }

    // Build connectivity: each element connects to its neighbors in 2D grid.
    // Only owned elements need connectivity; their neighbours are always either
    // owned or stored in one of the halo rows.
    const int base_row = world.row_begin - world.halo_lo;
    for (int x = world.row_begin; x < world.row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t idx = static_cast<size_t>(x - base_row) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const size_t neighbor_idx =
                        static_cast<size_t>(nx - base_row) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    const auto setMaterial = [&](int x, int y, idx_t mat) {
        if (x >= world.row_begin && x < world.row_end) {
            const size_t idx = static_cast<size_t>(x - base_row) * n_elems_root + y;
            world.elements_static[idx].material_idx = mat;
        }
    };
    setMaterial(0, 0, INFLOW_MAT_ID);
    setMaterial(0, last, OUTFLOW_MAT_ID);
    setMaterial(last, 0, OUTFLOW_MAT_ID);
    setMaterial(last, last, INFLOW_MAT_ID);

    // Halo exchange buffers
    if (world.halo_lo) {
        world.send_lo.resize(n_elems_root);
        world.recv_lo.resize(n_elems_root);
    }
    if (world.halo_hi) {
        world.send_hi.resize(n_elems_root);
        world.recv_hi.resize(n_elems_root);
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update the elements in the local index range [begin, end)
static inline void updateElements(World& world, const size_t begin, const size_t end) {
    const ElementStatic* __restrict statics = world.elements_static.data();
    const ElementDynamic* __restrict dynamics = world.elements_dynamic.data();
    ElementDynamic* __restrict writes = world.elements_dynamic_swap.data();
    const Material* __restrict materials = world.materials.data();

    for (size_t i = begin; i < end; ++i) {
        const ElementStatic& elem_static = statics[i];
        const ElementDynamic& elem_dyn = dynamics[i];
        const Material& mat = materials[elem_static.material_idx];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        const idx_t n_conn = elem_static.num_connections;
        for (idx_t j = 0; j < n_conn; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            const ElementDynamic& neighbor_dyn = dynamics[neighbor_idx];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }

        // Update element state
        ElementDynamic& elem_write = writes[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Post the halo exchange of the freshly computed boundary rows held in
// elements_dynamic_swap. Only current_energy is read from neighbouring
// elements, so that is all we communicate.
static inline void startHaloExchange(World& world, MPI_Request* requests, int& n_requests) {
    n_requests = 0;
    if (world.halo_lo == 0 && world.halo_hi == 0) return;

    const int n = world.n_elems_root;
    const ElementDynamic* swap = world.elements_dynamic_swap.data();
    const size_t owned = world.ownedOffset();
    const size_t last_row = owned + static_cast<size_t>(world.n_rows - 1) * n;

    if (world.halo_lo) {
        for (int y = 0; y < n; ++y) world.send_lo[y] = swap[owned + y].current_energy;
        MPI_Irecv(world.recv_lo.data(), n, MPI_DOUBLE, world.rank_lo, 0, MPI_COMM_WORLD,
                  &requests[n_requests++]);
        MPI_Isend(world.send_lo.data(), n, MPI_DOUBLE, world.rank_lo, 1, MPI_COMM_WORLD,
                  &requests[n_requests++]);
    }
    if (world.halo_hi) {
        for (int y = 0; y < n; ++y) world.send_hi[y] = swap[last_row + y].current_energy;
        MPI_Irecv(world.recv_hi.data(), n, MPI_DOUBLE, world.rank_hi, 1, MPI_COMM_WORLD,
                  &requests[n_requests++]);
        MPI_Isend(world.send_hi.data(), n, MPI_DOUBLE, world.rank_hi, 0, MPI_COMM_WORLD,
                  &requests[n_requests++]);
    }
}

// Wait for the halo exchange and store the received energies in the halo rows
static inline void finishHaloExchange(World& world, MPI_Request* requests, int n_requests) {
    if (n_requests == 0) return;
    MPI_Waitall(n_requests, requests, MPI_STATUSES_IGNORE);

    const int n = world.n_elems_root;
    ElementDynamic* swap = world.elements_dynamic_swap.data();
    if (world.halo_lo) {
        for (int y = 0; y < n; ++y) swap[y].current_energy = world.recv_lo[y];
    }
    if (world.halo_hi) {
        const size_t halo_hi_off =
            static_cast<size_t>(world.halo_lo + world.n_rows) * n;
        for (int y = 0; y < n; ++y) swap[halo_hi_off + y].current_energy = world.recv_hi[y];
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    if (world.n_rows == 0) return;  // rank without work (more ranks than grid rows)

    const int n = world.n_elems_root;
    const size_t owned = world.ownedOffset();
    const size_t n_owned = static_cast<size_t>(world.n_rows) * n;

    // Rows whose stencil touches a halo row have to be computed before the
    // halo exchange can be started; the remaining interior rows are computed
    // while the communication is in flight.
    const size_t first_row_end = owned + static_cast<size_t>(n);
    const size_t last_row_begin = owned + static_cast<size_t>(world.n_rows - 1) * n;
    const bool has_interior = world.n_rows > 2;

    MPI_Request requests[4];
    int n_requests = 0;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Boundary rows first, so their new values can be communicated early
        updateElements(world, owned, std::min(first_row_end, owned + n_owned));
        if (world.n_rows > 1) {
            updateElements(world, last_row_begin, owned + n_owned);
        }

        startHaloExchange(world, requests, n_requests);

        // Interior rows: no halo dependency
        if (has_interior) {
            updateElements(world, first_row_end, last_row_begin);
        }

        finishHaloExchange(world, requests, n_requests);

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Gather the owned element values of all ranks into global arrays on rank 0
static void gatherResults(const World& world, int rank, int n_ranks,
                          std::vector<val_t>& energy, std::vector<val_t>& flux) {
    const int n = world.n_elems_root;
    const size_t owned = world.ownedOffset();
    const int n_owned = world.n_rows * n;

    std::vector<val_t> local_energy(n_owned), local_flux(n_owned);
    for (int i = 0; i < n_owned; ++i) {
        local_energy[i] = world.elements_dynamic[owned + i].current_energy;
        local_flux[i] = world.elements_dynamic[owned + i].total_flux;
    }

    std::vector<int> counts, displs;
    if (rank == 0) {
        counts.resize(n_ranks);
        displs.resize(n_ranks);
        for (int r = 0; r < n_ranks; ++r) {
            counts[r] = (rowBegin(n, n_ranks, r + 1) - rowBegin(n, n_ranks, r)) * n;
            displs[r] = rowBegin(n, n_ranks, r) * n;
        }
        energy.resize(static_cast<size_t>(n) * n);
        flux.resize(static_cast<size_t>(n) * n);
    }

    MPI_Gatherv(local_energy.data(), n_owned, MPI_DOUBLE,
                rank == 0 ? energy.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_flux.data(), n_owned, MPI_DOUBLE,
                rank == 0 ? flux.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// Validate simulation results
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

// Compute a simple hash of the results for verification.
// The hash combines per-element contributions with XOR, so the distributed
// partial hashes can simply be reduced with a global XOR.
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    const size_t owned = world.ownedOffset();
    const size_t n_owned = static_cast<size_t>(world.n_rows) * world.n_elems_root;
    const size_t first = static_cast<size_t>(world.row_begin) * world.n_elems_root;
    for (size_t k = 0; k < n_owned; ++k) {
        const size_t i = first + k;  // global element index
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr =
            reinterpret_cast<const uint64_t*>(&world.elements_dynamic[owned + k].current_energy);
        const uint64_t* f_ptr =
            reinterpret_cast<const uint64_t*>(&world.elements_dynamic[owned + k].total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
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

    int rank = 0, n_ranks = 1;
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

    if (n_elems_root < 1) {
        if (rank == 0) printf("Grid size must be at least 1\n");
        MPI_Finalize();
        return 1;
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", n_ranks);
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, rank, n_ranks);

    if (rank == 0) {
        // Calculate memory usage (aggregated over the whole distributed mesh)
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
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long max_duration_ms = duration_ms;
    MPI_Allreduce(&duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    duration_ms = max_duration_ms;

    // Compute hash for verification (collective)
    const uint64_t hash = computeHash(world);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Gather the distributed state for output / validation
    std::vector<val_t> energyData, fluxData;
    if (printResults || validate) {
        gatherResults(world, rank, n_ranks, energyData, fluxData);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(energyData, "ElementEnergy");
    }

    // Validation
    int failed = 0;
    if (validate) {
        if (rank == 0) {
            failed = validateResults(energyData, fluxData) ? 0 : 1;
        }
        MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return failed;
}
