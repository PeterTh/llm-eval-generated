#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include <sys/mman.h>

#include <mpi.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Allocator for the large mesh arrays. The kernel performs indirect (gather)
// accesses over hundreds of MB per rank, so it is very sensitive to TLB
// pressure; backing the arrays with transparent huge pages measurably reduces
// the miss cost. It also keeps the allocations independent of the malloc
// tuning that the MPI runtime applies to the process heap.
template <typename T>
struct HugeAllocator {
    using value_type = T;

    HugeAllocator() = default;
    template <typename U>
    HugeAllocator(const HugeAllocator<U>&) {}

    static constexpr size_t HUGE_PAGE = 2u << 20;

    T* allocate(size_t count) {
        const size_t bytes = count * sizeof(T);
        const size_t rounded = (bytes + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1);
        void* p = nullptr;
        if (posix_memalign(&p, HUGE_PAGE, rounded) != 0) {
            throw std::bad_alloc();
        }
        madvise(p, rounded, MADV_HUGEPAGE);
        return static_cast<T*>(p);
    }

    void deallocate(T* p, size_t) { free(p); }

    template <typename U>
    bool operator==(const HugeAllocator<U>&) const { return true; }
    template <typename U>
    bool operator!=(const HugeAllocator<U>&) const { return false; }
};

template <typename T>
using huge_vector = std::vector<T, HugeAllocator<T>>;

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

// Distributed world state.
//
// The mesh is partitioned across MPI ranks with a 1D block decomposition over
// the first grid coordinate (x), i.e. each rank owns a contiguous band of
// "rows" of the linearised element numbering (idx = x * n_elems_root + y).
// The dynamic state is stored in structure-of-arrays form and padded with one
// ghost row on each side so that the connectivity of locally owned elements can
// address off-rank neighbours through plain (local) indices, exactly like it
// addresses on-rank neighbours.
struct World {
    std::vector<Material> materials;

    // Connectivity of the locally owned elements only (n_local_elems entries),
    // indexed by local owned index `xl * row_len + y`. `connected_idx` refers to
    // indices into the padded dynamic arrays below.
    huge_vector<ElementStatic> elements_static;

    // Padded dynamic state: ((n_local_rows + 2) * row_len) entries.
    // Padded index of owned element (xl, y) is (xl + 1) * row_len + y.
    huge_vector<val_t> energy;
    huge_vector<val_t> energy_swap;
    huge_vector<val_t> flux;
    huge_vector<val_t> flux_swap;

    // Decomposition description
    idx_t row_len = 0;        // elements per row (== n_elems_root)
    idx_t x_begin = 0;        // first owned row (global x)
    idx_t n_local_rows = 0;   // number of owned rows
    idx_t n_local_elems = 0;  // n_local_rows * row_len
    idx_t padded_offset = 0;  // padded index of first owned element (== row_len)

    // Neighbour ranks in the row chain (MPI_PROC_NULL if none)
    int prev_rank = MPI_PROC_NULL;
    int next_rank = MPI_PROC_NULL;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Block distribution of `n` rows over `size` ranks: rows [begin, end) for rank r
static inline idx_t rowBegin(idx_t n, int size, int r) {
    return (n * static_cast<idx_t>(r)) / static_cast<idx_t>(size);
}
static inline idx_t rowCount(idx_t n, int size, int r) {
    return rowBegin(n, size, r + 1) - rowBegin(n, size, r);
}

// Build the local part of a 2D square grid represented as an unstructured mesh.
// Only the elements owned by this rank are instantiated; their connectivity
// points into the padded local dynamic arrays (including ghost rows).
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int size) {
    const idx_t n = static_cast<idx_t>(n_elems_root);

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.row_len = n;
    world.x_begin = rowBegin(n, size, rank);
    world.n_local_rows = rowCount(n, size, rank);
    world.n_local_elems = world.n_local_rows * n;
    world.padded_offset = n;

    // Determine the neighbours in the chain of ranks that actually own rows.
    // (With more ranks than rows some ranks own nothing and are skipped.)
    if (world.n_local_rows > 0) {
        for (int r = rank - 1; r >= 0; --r) {
            if (rowCount(n, size, r) > 0) { world.prev_rank = r; break; }
        }
        for (int r = rank + 1; r < size; ++r) {
            if (rowCount(n, size, r) > 0) { world.next_rank = r; break; }
        }
    }

    // Allocate elements (padded by one ghost row on each side)
    const size_t padded = static_cast<size_t>(world.n_local_rows + 2) * static_cast<size_t>(n);
    world.elements_static.resize(world.n_local_elems);
    world.energy.assign(padded, 0.0);
    world.energy_swap.assign(padded, 0.0);
    world.flux.assign(padded, 0.0);
    world.flux_swap.assign(padded, 0.0);

    // Build connectivity: each element connects to its neighbors in the 2D grid.
    // The neighbour visiting order matches the serial reference implementation so
    // that the flux accumulation order (and hence the floating-point result) is
    // bit-for-bit identical.
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (idx_t xl = 0; xl < world.n_local_rows; ++xl) {
        const int64_t x = static_cast<int64_t>(world.x_begin + xl);
        for (idx_t y = 0; y < n; ++y) {
            ElementStatic& elem = world.elements_static[xl * n + y];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            for (int c = 0; c < 4; ++c) {
                const int64_t nx = x + offsets[c][0];
                const int64_t ny = static_cast<int64_t>(y) + offsets[c][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < static_cast<int64_t>(n) && ny >= 0 && ny < static_cast<int64_t>(n)) {
                    // Padded local index of the neighbour: rows outside the owned
                    // band land in the ghost rows.
                    const idx_t neighbor_idx =
                        static_cast<idx_t>(nx - static_cast<int64_t>(world.x_begin) + 1) * n +
                        static_cast<idx_t>(ny);
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const idx_t last = n - 1;
    const auto set_material = [&](idx_t x, idx_t y, idx_t mat) {
        if (x >= world.x_begin && x < world.x_begin + world.n_local_rows) {
            world.elements_static[(x - world.x_begin) * n + y].material_idx = mat;
        }
    };
    set_material(0, 0, INFLOW_MAT_ID);
    set_material(0, last, OUTFLOW_MAT_ID);
    set_material(last, 0, OUTFLOW_MAT_ID);
    set_material(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, val_t this_energy,
                        val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Compute the new state of the single element with local index `li`
static inline void updateElement(const ElementStatic* __restrict statics,
                                 const Material* __restrict materials, idx_t row_len, idx_t li,
                                 const val_t* __restrict energy_in, const val_t* __restrict flux_in,
                                 val_t& new_energy, val_t& new_flux) {
    const ElementStatic& elem_static = statics[li];
    const idx_t pi = li + row_len;
    const val_t this_energy = energy_in[pi];
    const Material& mat = materials[elem_static.material_idx];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    const idx_t nconn = elem_static.num_connections;
    for (idx_t j = 0; j < nconn; ++j) {
        const idx_t neighbor_idx = elem_static.connected_idx[j];
        total_flux += computeFlux(mat, this_energy, elem_static.connected_flux[j],
                                  energy_in[neighbor_idx]);
    }

    // Update element state
    new_energy = this_energy + total_flux;
    new_flux = flux_in[pi] + std::abs(total_flux);
}

// Update the owned rows [xl_begin, xl_end) of the local band
static void updateRows(const World& world, idx_t xl_begin, idx_t xl_end,
                       const val_t* __restrict energy_in, const val_t* __restrict flux_in,
                       val_t* __restrict energy_out, val_t* __restrict flux_out) {
    const idx_t row_len = world.row_len;
    const ElementStatic* __restrict statics = world.elements_static.data();
    const Material* __restrict materials = world.materials.data();

    // Flat sweep over the owned elements of the requested row range; the padded
    // index of owned element `li` is simply `li + row_len`.
    const idx_t li_begin = xl_begin * row_len;
    const idx_t li_end = xl_end * row_len;

    for (idx_t li = li_begin; li < li_end; ++li) {
        updateElement(statics, materials, row_len, li, energy_in, flux_in,
                      energy_out[li + row_len], flux_out[li + row_len]);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const idx_t row_len = world.row_len;
    const idx_t n_rows = world.n_local_rows;
    const int row_count = static_cast<int>(row_len);

    for (int iter = 0; iter < n_iters; ++iter) {
        val_t* energy_in = world.energy.data();
        val_t* flux_in = world.flux.data();
        val_t* energy_out = world.energy_swap.data();
        val_t* flux_out = world.flux_swap.data();

        if (n_rows == 0) {
            continue;
        }

        // Exchange the halo rows of the current energy field with the neighbouring
        // ranks; the flux field is never read across element boundaries.
        MPI_Request requests[4];
        int n_requests = 0;
        if (world.prev_rank != MPI_PROC_NULL) {
            MPI_Irecv(energy_in, row_count, MPI_DOUBLE, world.prev_rank, 0,
                      MPI_COMM_WORLD, &requests[n_requests++]);
            MPI_Isend(energy_in + row_len, row_count, MPI_DOUBLE, world.prev_rank, 1,
                      MPI_COMM_WORLD, &requests[n_requests++]);
        }
        if (world.next_rank != MPI_PROC_NULL) {
            MPI_Irecv(energy_in + (n_rows + 1) * row_len, row_count, MPI_DOUBLE,
                      world.next_rank, 1, MPI_COMM_WORLD, &requests[n_requests++]);
            MPI_Isend(energy_in + n_rows * row_len, row_count, MPI_DOUBLE,
                      world.next_rank, 0, MPI_COMM_WORLD, &requests[n_requests++]);
        }

        // Overlap: update the interior rows (which do not touch the halo) while
        // the halo exchange is in flight.
        if (n_rows > 2) {
            updateRows(world, 1, n_rows - 1, energy_in, flux_in, energy_out, flux_out);
        }

        if (n_requests > 0) {
            MPI_Waitall(n_requests, requests, MPI_STATUSES_IGNORE);
        }

        // Update the boundary rows now that the halo is available.
        updateRows(world, 0, 1, energy_in, flux_in, energy_out, flux_out);
        if (n_rows > 1) {
            updateRows(world, n_rows - 1, n_rows, energy_in, flux_in, energy_out, flux_out);
        }

        // Swap buffers
        std::swap(world.energy, world.energy_swap);
        std::swap(world.flux, world.flux_swap);
    }
}

// Gather the owned dynamic values of `field` (padded local array) into a
// globally ordered array on rank 0.
static void gatherField(const World& world, const huge_vector<val_t>& field,
                        std::vector<val_t>& out, int rank, int size) {
    const int local_count = static_cast<int>(world.n_local_elems);
    std::vector<int> counts, displs;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
    }
    MPI_Gather(&local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);
    size_t total = 0;
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            displs[r] = static_cast<int>(total);
            total += static_cast<size_t>(counts[r]);
        }
        out.resize(total);
    }
    MPI_Gatherv(field.data() + world.padded_offset, local_count, MPI_DOUBLE,
                rank == 0 ? out.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// Validate simulation results (executed on rank 0 over the gathered fields, so
// that the reduction order matches the serial reference exactly)
bool validateResults(const std::vector<val_t>& energies, const std::vector<val_t>& fluxes) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energies.size(); ++i) {
        energy_sum += energies[i];
        flux_sum += fluxes[i];
        energy_max = std::max(energies[i], energy_max);
        energy_min = std::min(energies[i], energy_min);
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

// Compute a simple hash of the local results. The hash combines per-element
// contributions with XOR (using the *global* element index), so the distributed
// partial hashes can be combined with a bitwise-XOR reduction to yield exactly
// the same digest as the serial version.
uint64_t computeLocalHash(const World& world) {
    uint64_t hash = 0;
    const val_t* energy = world.energy.data() + world.padded_offset;
    const val_t* flux = world.flux.data() + world.padded_offset;
    const size_t global_base = static_cast<size_t>(world.x_begin) * static_cast<size_t>(world.row_len);

    for (size_t k = 0; k < static_cast<size_t>(world.n_local_elems); ++k) {
        const size_t i = global_base + k;
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&energy[k]);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&flux[k]);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    if (n_elems_root <= 0) {
        if (rank == 0) printf("Grid size must be positive\n");
        MPI_Finalize();
        return 1;
    }

    const size_t n_elems = static_cast<size_t>(n_elems_root) * static_cast<size_t>(n_elems_root);

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root,
               static_cast<int>(n_elems));
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("\n");
    }

    // Build the unstructured mesh (each rank builds only its own partition)
    if (rank == 0) {
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, rank, size);

    // Calculate (global) memory usage
    if (rank == 0) {
        const size_t static_mem = n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
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

    auto end = std::chrono::high_resolution_clock::now();
    long long local_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    long long max_ns = local_ns;
    MPI_Allreduce(&local_ns, &max_ns, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    const long duration_ms = static_cast<long>(max_ns / 1000000);

    // Compute hash for verification (XOR reduction over the per-rank partials)
    const uint64_t local_hash = computeLocalHash(world);
    uint64_t hash = local_hash;
    MPI_Allreduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);

    if (rank == 0) {
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
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation
    if (printResults) {
        std::vector<val_t> energyData;
        gatherField(world, world.energy, energyData, rank, size);
        if (rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }

    // Validation
    int exit_code = 0;
    if (validate) {
        std::vector<val_t> energyData, fluxData;
        gatherField(world, world.energy, energyData, rank, size);
        gatherField(world, world.flux, fluxData, rank, size);
        if (rank == 0) {
            exit_code = validateResults(energyData, fluxData) ? 0 : 1;
        }
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exit_code;
}
