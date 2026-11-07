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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (local numbering)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Distributed-memory decomposition of the global element array.
//
// The global element ordering is idx = x * n_elems_root + y. The grid is split
// into contiguous row blocks [row_begin, row_end) per rank, so every rank owns
// a contiguous slice of the global element array. Locally, the owned rows are
// stored with one halo (ghost) row before and one after them, so that the
// connectivity of owned elements can reference neighbors in adjacent ranks
// through ordinary local indices:
//
//   local row 0            : halo for global row row_begin - 1 (from rank_up)
//   local rows 1..n_rows   : owned rows row_begin .. row_end - 1
//   local row n_rows + 1   : halo for global row row_end (from rank_down)
struct Decomposition {
    int rank = 0;
    int size = 1;
    int n_elems_root = 0;
    int row_begin = 0;        // first owned global row
    int row_end = 0;          // one past last owned global row
    int n_rows = 0;           // number of owned rows
    int rank_up = MPI_PROC_NULL;    // owner of row row_begin - 1
    int rank_down = MPI_PROC_NULL;  // owner of row row_end
    size_t local_offset = 0;  // local index of first owned element
    size_t n_local = 0;       // number of owned elements
    size_t n_total = 0;       // owned + halo elements
    idx_t global_begin = 0;   // global index of first owned element
};

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;        // owned elements only
    std::vector<ElementDynamic> elements_dynamic;      // owned + halo rows
    std::vector<ElementDynamic> elements_dynamic_swap; // owned + halo rows
    Decomposition decomp;
};

// Pin this rank's memory allocations to the NUMA node it is running on.
// Each rank first-touches only its own data, so a local-node policy is the
// natural placement. Setting an explicit policy also exempts the process from
// automatic NUMA balancing (hinting page faults), which otherwise slows the
// memory-bound kernel noticeably once MPI's helper threads exist.
static void preferLocalNumaMemory() {
#ifdef __linux__
    constexpr int MPOL_PREFERRED_MODE = 1;  // MPOL_PREFERRED, empty mask = local node
    syscall(SYS_set_mempolicy, MPOL_PREFERRED_MODE, nullptr, 0UL);
#endif
}

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Block distribution of rows: the first (n % size) ranks get one extra row
static inline void rowRange(const int n_rows_global, const int size, const int rank,
                            int& begin, int& end) {
    const int base = n_rows_global / size;
    const int rem = n_rows_global % size;
    begin = rank * base + std::min(rank, rem);
    end = begin + base + (rank < rem ? 1 : 0);
}

static inline int rowOwner(const int n_rows_global, const int size, const int row) {
    const int base = n_rows_global / size;
    const int rem = n_rows_global % size;
    const int split = rem * (base + 1);
    if (row < split) return row / (base + 1);
    return rem + (row - split) / base;
}

// Build a 2D square grid as an unstructured mesh (local part of this rank)
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int size) {
    Decomposition& d = world.decomp;
    d.rank = rank;
    d.size = size;
    d.n_elems_root = n_elems_root;
    rowRange(n_elems_root, size, rank, d.row_begin, d.row_end);
    d.n_rows = d.row_end - d.row_begin;
    d.n_local = static_cast<size_t>(d.n_rows) * n_elems_root;
    d.local_offset = static_cast<size_t>(n_elems_root);  // one halo row in front
    d.n_total = d.n_local + 2 * static_cast<size_t>(n_elems_root);
    d.global_begin = static_cast<idx_t>(d.row_begin) * n_elems_root;
    if (d.n_rows > 0) {
        if (d.row_begin > 0) d.rank_up = rowOwner(n_elems_root, size, d.row_begin - 1);
        if (d.row_end < n_elems_root) d.rank_down = rowOwner(n_elems_root, size, d.row_end);
    }

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements
    world.elements_static.resize(d.n_local);
    world.elements_dynamic.resize(d.n_total);
    world.elements_dynamic_swap.resize(d.n_total);

    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < d.n_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    for (size_t i = 0; i < d.n_total; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        world.elements_dynamic_swap[i].current_energy = 0.0;
        world.elements_dynamic_swap[i].total_flux = 0.0;
    }

    // Build connectivity: each element connects to its neighbors in 2D grid.
    // Connections are stored as local indices into elements_dynamic (which
    // includes the halo rows), preserving the original neighbor ordering.
    for (int x = d.row_begin; x < d.row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t idx = static_cast<size_t>(x - d.row_begin) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const size_t neighbor_local =
                        static_cast<size_t>(nx - d.row_begin + 1) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_local;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    auto setMaterial = [&](const int x, const int y, const idx_t mat) {
        if (x >= d.row_begin && x < d.row_end) {
            world.elements_static[static_cast<size_t>(x - d.row_begin) * n_elems_root + y].material_idx = mat;
        }
    };
    setMaterial(0, 0, INFLOW_MAT_ID);
    setMaterial(0, last, OUTFLOW_MAT_ID);
    setMaterial(last, 0, OUTFLOW_MAT_ID);
    setMaterial(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update owned elements in local range [begin, end)
static inline void updateRange(World& world, const size_t begin, const size_t end) {
    const ElementStatic* es = world.elements_static.data();
    const ElementDynamic* dyn = world.elements_dynamic.data();
    const ElementDynamic* dyn_owned = dyn + world.decomp.local_offset;
    ElementDynamic* out_owned = world.elements_dynamic_swap.data() + world.decomp.local_offset;
    const Material* mats = world.materials.data();

    for (size_t i = begin; i < end; ++i) {
        const ElementStatic& elem_static = es[i];
        const ElementDynamic& elem_dyn = dyn_owned[i];
        const Material& mat = mats[elem_static.material_idx];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            const ElementDynamic& neighbor_dyn = dyn[neighbor_idx];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }

        // Update element state
        ElementDynamic& elem_write = out_owned[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const Decomposition& d = world.decomp;
    const int n = d.n_elems_root;
    const size_t row_elems = static_cast<size_t>(n);
    const size_t n_local = d.n_local;
    // Only current_energy of the halo is read, but exchanging whole rows of
    // ElementDynamic keeps the buffers contiguous and copy-free.
    const int row_bytes = static_cast<int>(row_elems * sizeof(ElementDynamic));

    // Boundary (first/last owned row) vs. interior split for overlap:
    //   [0, first_row_end)              first owned row (needs upper halo)
    //   [first_row_end, last_row_begin) interior rows (no halo dependency)
    //   [last_row_begin, n_local)       last owned row (needs lower halo), if n_rows >= 2
    const size_t first_row_end = std::min(row_elems, n_local);
    const size_t last_row_begin = (d.n_rows >= 2) ? n_local - row_elems : n_local;

    for (int iter = 0; iter < n_iters; ++iter) {
        ElementDynamic* dyn = world.elements_dynamic.data();
        MPI_Request reqs[4];
        int n_reqs = 0;

        if (n_local > 0) {
            // Receive halo rows into local rows 0 and n_rows + 1
            if (d.rank_up != MPI_PROC_NULL) {
                MPI_Irecv(dyn, row_bytes, MPI_BYTE, d.rank_up, 0, MPI_COMM_WORLD, &reqs[n_reqs++]);
            }
            if (d.rank_down != MPI_PROC_NULL) {
                MPI_Irecv(dyn + d.local_offset + n_local, row_bytes, MPI_BYTE, d.rank_down, 1,
                          MPI_COMM_WORLD, &reqs[n_reqs++]);
            }
            // Send first owned row up and last owned row down
            if (d.rank_up != MPI_PROC_NULL) {
                MPI_Isend(dyn + d.local_offset, row_bytes, MPI_BYTE, d.rank_up, 1, MPI_COMM_WORLD,
                          &reqs[n_reqs++]);
            }
            if (d.rank_down != MPI_PROC_NULL) {
                MPI_Isend(dyn + d.local_offset + n_local - row_elems, row_bytes, MPI_BYTE,
                          d.rank_down, 0, MPI_COMM_WORLD, &reqs[n_reqs++]);
            }
        }

        // Update interior elements (independent of halo data) while halos are in flight
        if (last_row_begin > first_row_end) {
            updateRange(world, first_row_end, last_row_begin);
        }

        if (n_reqs > 0) {
            MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
        }

        // Update boundary rows that depend on halo data
        updateRange(world, 0, first_row_end);
        if (last_row_begin < n_local) {
            updateRange(world, last_row_begin, n_local);
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Gather the owned part of the dynamic state of all ranks on rank 0, in global order
static std::vector<ElementDynamic> gatherDynamic(const World& world) {
    const Decomposition& d = world.decomp;
    std::vector<int> counts(d.size), displs(d.size);
    for (int r = 0; r < d.size; ++r) {
        int b, e;
        rowRange(d.n_elems_root, d.size, r, b, e);
        counts[r] = static_cast<int>(static_cast<size_t>(e - b) * d.n_elems_root * sizeof(ElementDynamic));
        displs[r] = static_cast<int>(static_cast<size_t>(b) * d.n_elems_root * sizeof(ElementDynamic));
    }
    std::vector<ElementDynamic> all;
    if (d.rank == 0) {
        all.resize(static_cast<size_t>(d.n_elems_root) * d.n_elems_root);
    }
    MPI_Gatherv(world.elements_dynamic.data() + d.local_offset,
                static_cast<int>(d.n_local * sizeof(ElementDynamic)), MPI_BYTE,
                all.data(), counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    return all;
}

// Validate simulation results (called on rank 0 with the gathered global state)
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
// The hash is an XOR of per-element terms keyed by the global index, so each
// rank hashes its owned elements and the partial hashes are combined with
// MPI_BXOR, giving a result identical to the sequential version.
uint64_t computeHash(const World& world) {
    const Decomposition& d = world.decomp;
    const ElementDynamic* elements = world.elements_dynamic.data() + d.local_offset;
    uint64_t hash = 0;
    for (size_t li = 0; li < d.n_local; ++li) {
        const uint64_t i = d.global_begin + li;
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[li].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[li].total_flux);
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool is_root = (rank == 0);
    preferLocalNumaMemory();

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

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, rank, size);

    // Calculate memory usage (global, across all ranks, excluding halo rows)
    const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (is_root) {
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
    long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Compute hash for verification (collective)
    const uint64_t hash = computeHash(world);

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

    int exit_code = 0;
    if (printResults || validate) {
        // Gather the global state on rank 0 so that sums are evaluated in the
        // same sequential order as the original program
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
            if (is_root) {
                bool valid = validateResults(all);
                if (!valid) exit_code = 1;
            }
            MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return exit_code;
}
