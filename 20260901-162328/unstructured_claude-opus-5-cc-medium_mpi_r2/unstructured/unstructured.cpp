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

// World state.
//
// The mesh is distributed over MPI ranks with a 1D row-block decomposition
// (element index is x * n + y, so a block of x-rows is contiguous in memory
// and every cut plane is a single row of n elements).
//
// The dynamic state is kept as separate energy/flux arrays: only the energy of
// the boundary rows has to be communicated, and in this layout such a row is
// already contiguous, so halos can be exchanged straight out of the arrays
// without any packing.
//
// The dynamic arrays are padded with one ghost row below and above the owned
// rows. Padded index of owned element k is simply k + n.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;  // owned elements only
    std::vector<val_t> energy;                   // padded with ghost rows
    std::vector<val_t> flux;                     // padded with ghost rows
    std::vector<val_t> energy_swap;
    std::vector<val_t> flux_swap;

    int n = 0;            // grid edge length (n_elems_root)
    int row_begin = 0;    // first owned x-row (global)
    int local_rows = 0;   // number of owned x-rows
    size_t n_local = 0;   // number of owned elements
    int rank_down = MPI_PROC_NULL;  // owner of row row_begin - 1
    int rank_up = MPI_PROC_NULL;    // owner of row row_begin + local_rows
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Block distribution of n rows over n_ranks ranks: the first (n % n_ranks)
// ranks get one extra row. If there are more ranks than rows, the trailing
// ranks stay empty.
static inline int rowsBegin(int n, int n_ranks, int rank) {
    const int base = n / n_ranks;
    const int rem = n % n_ranks;
    return rank * base + std::min(rank, rem);
}

// Rank owning global row x (x must be a valid row index).
static inline int rowOwner(int n, int n_ranks, int x) {
    const int base = n / n_ranks;
    const int rem = n % n_ranks;
    if (base == 0) {
        return x;  // one row per rank, trailing ranks empty
    }
    const int split = rem * (base + 1);
    return x < split ? x / (base + 1) : rem + (x - split) / base;
}

// Build the local part of a 2D square grid represented as an unstructured mesh.
// This represents computation on arbitrarily-shaped geometries.
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int n_ranks) {
    const int n = n_elems_root;

    world.n = n;
    world.row_begin = rowsBegin(n, n_ranks, rank);
    world.local_rows = rowsBegin(n, n_ranks, rank + 1) - world.row_begin;
    world.n_local = static_cast<size_t>(world.local_rows) * n;

    const int row_end = world.row_begin + world.local_rows;
    if (world.local_rows > 0) {
        if (world.row_begin > 0) {
            world.rank_down = rowOwner(n, n_ranks, world.row_begin - 1);
        }
        if (row_end < n) {
            world.rank_up = rowOwner(n, n_ranks, row_end);
        }
    }

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements (dynamic state padded by one ghost row on each side)
    const size_t padded = world.n_local + 2 * static_cast<size_t>(n);
    world.elements_static.resize(world.n_local);
    world.energy.assign(padded, 0.0);
    world.flux.assign(padded, 0.0);
    world.energy_swap.assign(padded, 0.0);
    world.flux_swap.assign(padded, 0.0);

    // Build connectivity: each element connects to its neighbors in 2D grid.
    // Connection indices refer to the padded local dynamic arrays; x-neighbors
    // outside the owned rows land in a ghost row.
    for (int x = world.row_begin; x < row_end; ++x) {
        for (int y = 0; y < n; ++y) {
            const size_t idx = static_cast<size_t>(x - world.row_begin) * n + y;
            ElementStatic& elem = world.elements_static[idx];

            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int nn = 0; nn < 4; ++nn) {
                const int nx = x + offsets[nn][0];
                const int ny = y + offsets[nn][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    const size_t neighbor_idx =
                        static_cast<size_t>(nx - world.row_begin + 1) * n + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n - 1;
    const auto setMaterial = [&](int x, int y, idx_t mat) {
        if (x >= world.row_begin && x < row_end) {
            world.elements_static[static_cast<size_t>(x - world.row_begin) * n + y].material_idx = mat;
        }
    };
    setMaterial(0, 0, INFLOW_MAT_ID);
    setMaterial(0, last, OUTFLOW_MAT_ID);
    setMaterial(last, 0, OUTFLOW_MAT_ID);
    setMaterial(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, val_t this_energy,
                        val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update the owned rows [row_lo, row_hi) (local row numbering)
static void updateRows(World& world, const int row_lo, const int row_hi) {
    const int n = world.n;
    const ElementStatic* __restrict elements_static = world.elements_static.data();
    const Material* __restrict materials = world.materials.data();
    const val_t* __restrict energy = world.energy.data();
    const val_t* __restrict flux = world.flux.data();
    val_t* __restrict energy_out = world.energy_swap.data();
    val_t* __restrict flux_out = world.flux_swap.data();

    const size_t begin = static_cast<size_t>(row_lo) * n;
    const size_t end = static_cast<size_t>(row_hi) * n;

    for (size_t i = begin; i < end; ++i) {
        const ElementStatic& elem_static = elements_static[i];
        const Material& mat = materials[elem_static.material_idx];
        const val_t this_energy = energy[i + n];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            total_flux += computeFlux(mat, this_energy, elem_static.connected_flux[j],
                                      energy[neighbor_idx]);
        }

        // Update element state
        energy_out[i + n] = this_energy + total_flux;
        flux_out[i + n] = flux[i + n] + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const int n = world.n;
    const int local_rows = world.local_rows;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange the energy of the boundary rows with the neighbor ranks.
        MPI_Request requests[4];
        int n_requests = 0;
        if (local_rows > 0) {
            // Ghost row below / above
            MPI_Irecv(world.energy.data(), n, MPI_DOUBLE, world.rank_down, 0,
                      MPI_COMM_WORLD, &requests[n_requests++]);
            MPI_Irecv(world.energy.data() + static_cast<size_t>(local_rows + 1) * n, n,
                      MPI_DOUBLE, world.rank_up, 1, MPI_COMM_WORLD, &requests[n_requests++]);
            // First / last owned row
            MPI_Isend(world.energy.data() + n, n, MPI_DOUBLE, world.rank_down, 1,
                      MPI_COMM_WORLD, &requests[n_requests++]);
            MPI_Isend(world.energy.data() + static_cast<size_t>(local_rows) * n, n,
                      MPI_DOUBLE, world.rank_up, 0, MPI_COMM_WORLD, &requests[n_requests++]);
        }

        // Interior rows do not depend on the halo: overlap them with the exchange
        if (local_rows > 2) {
            updateRows(world, 1, local_rows - 1);
        }

        if (n_requests > 0) {
            MPI_Waitall(n_requests, requests, MPI_STATUSES_IGNORE);
        }

        // Boundary rows
        if (local_rows > 0) {
            updateRows(world, 0, std::min(local_rows, 1));
            if (local_rows > 1) {
                updateRows(world, local_rows - 1, local_rows);
            }
        }

        // Swap buffers
        std::swap(world.energy, world.energy_swap);
        std::swap(world.flux, world.flux_swap);
    }
}

// Validate simulation results (global energy/flux data, rank 0 only)
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

// Compute a simple hash of the local results for verification.
// The per-element contributions are XOR-combined and carry the global element
// index, so the distributed partial hashes can simply be XOR-reduced.
uint64_t computeLocalHash(const World& world) {
    uint64_t hash = 0;
    const size_t global_offset = static_cast<size_t>(world.row_begin) * world.n;
    for (size_t k = 0; k < world.n_local; ++k) {
        // Simple hash combining energy and flux values
        const val_t energy = world.energy[k + world.n];
        const val_t flux = world.flux[k + world.n];
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&flux);
        const size_t i = global_offset + k;
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

// Gather an owned dynamic array to rank 0 in global element order
static void gatherToRoot(const World& world, const std::vector<val_t>& local,
                         std::vector<val_t>& global, const int rank, const int n_ranks) {
    std::vector<int> counts, displs;
    if (rank == 0) {
        counts.resize(n_ranks);
        displs.resize(n_ranks);
        for (int r = 0; r < n_ranks; ++r) {
            const int begin = rowsBegin(world.n, n_ranks, r);
            counts[r] = (rowsBegin(world.n, n_ranks, r + 1) - begin) * world.n;
            displs[r] = begin * world.n;
        }
        global.resize(static_cast<size_t>(world.n) * world.n);
    }
    MPI_Gatherv(local.data() + world.n, static_cast<int>(world.n_local), MPI_DOUBLE,
                rank == 0 ? global.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
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

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, rank, n_ranks);

    if (rank == 0) {
        // Calculate memory usage (aggregated over all ranks, ghost rows excluded)
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * 2 * sizeof(val_t) * 2;
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
    long duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Compute hash for verification
    uint64_t hash = computeLocalHash(world);
    MPI_Allreduce(MPI_IN_PLACE, &hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);

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

    // Print results for external validation
    if (printResults) {
        std::vector<val_t> energyData;
        gatherToRoot(world, world.energy, energyData, rank, n_ranks);
        if (rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }

    // Validation
    if (validate) {
        std::vector<val_t> energyData, fluxData;
        gatherToRoot(world, world.energy, energyData, rank, n_ranks);
        gatherToRoot(world, world.flux, fluxData, rank, n_ranks);
        bool valid = true;
        if (rank == 0) {
            valid = validateResults(energyData, fluxData);
        }
        int valid_flag = valid ? 1 : 0;
        MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!valid_flag) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
