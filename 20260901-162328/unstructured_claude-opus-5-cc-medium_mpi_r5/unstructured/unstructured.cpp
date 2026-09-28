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

// Distributed world state.
//
// The mesh is partitioned across MPI ranks by contiguous blocks of rows (the
// first grid coordinate 'x'). Every rank stores the elements of its own rows
// plus one ghost row on each side, which mirrors the boundary rows of the
// neighbouring ranks and is refreshed by a halo exchange every iteration.
//
// The dynamic state is kept as separate energy/flux arrays (rather than an
// array of structs) so that a ghost row is a contiguous run of doubles and can
// be sent/received without packing.
//
// Local storage index of global element (x, y) is (x - x0 + 1) * n + y, i.e.
// local row 0 is the lower ghost row and local row nrows_local + 1 the upper one.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;  // nrows_local * n entries, row x0 first
    std::vector<val_t> energy;                   // (nrows_local + 2) * n entries
    std::vector<val_t> energy_swap;              // (nrows_local + 2) * n entries
    std::vector<val_t> flux;                     // (nrows_local + 2) * n entries

    int n_elems_root = 0;   // global grid edge length
    int x0 = 0;             // first row owned by this rank
    int nrows_local = 0;    // number of rows owned by this rank
    int rank_up = MPI_PROC_NULL;    // owner of row x0 - 1
    int rank_down = MPI_PROC_NULL;  // owner of row x0 + nrows_local
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Row range [x0, x0 + nrows) owned by a given rank (block distribution).
inline void rowRange(int n_elems_root, int n_ranks, int rank, int& x0, int& nrows) {
    const int base = n_elems_root / n_ranks;
    const int rem = n_elems_root % n_ranks;
    if (rank < rem) {
        nrows = base + 1;
        x0 = rank * nrows;
    } else {
        nrows = base;
        x0 = rem * (base + 1) + (rank - rem) * base;
    }
}

// Rank owning a given row, or MPI_PROC_NULL if the row is outside the grid.
inline int rowOwner(int n_elems_root, int n_ranks, int x) {
    if (x < 0 || x >= n_elems_root) return MPI_PROC_NULL;
    const int base = n_elems_root / n_ranks;
    const int rem = n_elems_root % n_ranks;
    const int split = rem * (base + 1);
    if (x < split) return x / (base + 1);
    return rem + (x - split) / base;  // base > 0 here, since x >= split implies base >= 1
}

// Build this rank's share of a 2D square grid represented as an unstructured mesh.
// This represents computation on arbitrarily-shaped geometries.
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int n_ranks) {
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.n_elems_root = n_elems_root;
    rowRange(n_elems_root, n_ranks, rank, world.x0, world.nrows_local);

    const int n = n_elems_root;
    const int x0 = world.x0;
    const int nrows = world.nrows_local;

    world.rank_up = (nrows > 0) ? rowOwner(n, n_ranks, x0 - 1) : MPI_PROC_NULL;
    world.rank_down = (nrows > 0) ? rowOwner(n, n_ranks, x0 + nrows) : MPI_PROC_NULL;

    // Allocate elements (local rows plus one ghost row on each side)
    const size_t n_local = static_cast<size_t>(nrows) * n;
    const size_t n_stored = static_cast<size_t>(nrows + 2) * n;
    world.elements_static.resize(n_local);
    world.energy.assign(n_stored, 0.0);
    world.energy_swap.assign(n_stored, 0.0);
    world.flux.assign(n_stored, 0.0);

    // Build connectivity: each element connects to its neighbors in the 2D grid.
    // Neighbour indices refer to local storage slots; the neighbour order matches
    // the serial construction order {+x, -x, +y, -y}.
    const int last = n - 1;
    for (int x = x0; x < x0 + nrows; ++x) {
        for (int y = 0; y < n; ++y) {
            const size_t local_idx = static_cast<size_t>(x - x0) * n + y;
            ElementStatic& elem = world.elements_static[local_idx];

            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int c = 0; c < 4; ++c) {
                const int nx = x + offsets[c][0];
                const int ny = y + offsets[c][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    const size_t neighbor_store_idx =
                        static_cast<size_t>(nx - x0 + 1) * n + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_store_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }

            // Set corner elements as inflow/outflow to create interesting dynamics
            if ((x == 0 && y == 0) || (x == last && y == last)) {
                elem.material_idx = INFLOW_MAT_ID;
            } else if ((x == 0 && y == last) || (x == last && y == 0)) {
                elem.material_idx = OUTFLOW_MAT_ID;
            }
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, val_t this_energy,
                         val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// Update the local rows [r0, r1) (relative to this rank's first row).
static void updateRows(World& world, const int r0, const int r1) {
    const int n = world.n_elems_root;
    const Material* __restrict materials = world.materials.data();
    const ElementStatic* __restrict elems = world.elements_static.data();
    const val_t* __restrict energy = world.energy.data();
    val_t* __restrict energy_new = world.energy_swap.data();
    val_t* __restrict flux = world.flux.data();

    const size_t begin = static_cast<size_t>(r0) * n;
    const size_t end = static_cast<size_t>(r1) * n;

    for (size_t i = begin; i < end; ++i) {
        const ElementStatic& elem_static = elems[i];
        const size_t store_idx = i + n;  // skip the lower ghost row
        const val_t this_energy = energy[store_idx];
        const Material& mat = materials[elem_static.material_idx];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            total_flux += computeFlux(mat, this_energy, elem_static.connected_flux[j],
                                      energy[neighbor_idx]);
        }

        // Update element state
        energy_new[store_idx] = this_energy + total_flux;
        flux[store_idx] += std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const int n = world.n_elems_root;
    const int nrows = world.nrows_local;

    if (nrows == 0) return;  // rank without work; no halo partner refers to it

    const size_t lower_ghost = 0;
    const size_t upper_ghost = static_cast<size_t>(nrows + 1) * n;
    const size_t first_row = static_cast<size_t>(n);
    const size_t last_row = static_cast<size_t>(nrows) * n;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange the boundary rows with the neighbouring ranks, overlapping
        // the transfer with the update of the interior rows.
        MPI_Request reqs[4];
        int n_reqs = 0;
        if (world.rank_up != MPI_PROC_NULL) {
            MPI_Irecv(&world.energy[lower_ghost], n, MPI_DOUBLE, world.rank_up, 0,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Isend(&world.energy[first_row], n, MPI_DOUBLE, world.rank_up, 1,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
        }
        if (world.rank_down != MPI_PROC_NULL) {
            MPI_Irecv(&world.energy[upper_ghost], n, MPI_DOUBLE, world.rank_down, 1,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Isend(&world.energy[last_row], n, MPI_DOUBLE, world.rank_down, 0,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
        }

        // Interior rows do not read any ghost data
        if (nrows > 2) {
            updateRows(world, 1, nrows - 1);
        }

        MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

        // Boundary rows
        updateRows(world, 0, std::min(1, nrows));
        if (nrows > 1) {
            updateRows(world, nrows - 1, nrows);
        }

        // Swap buffers
        std::swap(world.energy, world.energy_swap);
    }
}

// Validate simulation results (rank 0 only, on the gathered global state)
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
// The per-element contributions are combined with XOR, so the distributed
// partial hashes can be reduced to the serial result independently of order.
uint64_t computeHash(const World& world) {
    const int n = world.n_elems_root;
    uint64_t hash = 0;
    const size_t first = static_cast<size_t>(world.x0) * n;
    const size_t count = static_cast<size_t>(world.nrows_local) * n;
    for (size_t k = 0; k < count; ++k) {
        const size_t i = first + k;  // global element index
        // Simple hash combining energy and flux values
        const val_t e = world.energy[k + n];
        const val_t f = world.flux[k + n];
        uint64_t e_bits, f_bits;
        memcpy(&e_bits, &e, sizeof(uint64_t));
        memcpy(&f_bits, &f, sizeof(uint64_t));
        hash ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }

    uint64_t global_hash = 0;
    MPI_Allreduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
    return global_hash;
}

// Collect a distributed per-element field on rank 0 in global element order.
static void gatherField(const World& world, const val_t* local_data,
                        const std::vector<int>& counts, const std::vector<int>& displs,
                        std::vector<val_t>& out) {
    const int n = world.n_elems_root;
    const int local_count = world.nrows_local * n;
    MPI_Gatherv(local_data + n, local_count, MPI_DOUBLE, out.data(), counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
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

    // Parse command line arguments (identically on every rank)
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

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("MPI ranks: %d\n", n_ranks);

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, rank, n_ranks);

    // Calculate memory usage (aggregated over all ranks)
    if (rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * (2 * sizeof(val_t)) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        // Run simulation
        printf("Running simulation...\n");
    }

    auto start = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Compute hash for verification (collective)
    const uint64_t hash = computeHash(world);

    if (rank == 0) {
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

    // Results output and validation operate on the full state in global element
    // order, so the distributed fields are collected on rank 0 first.
    if (printResults || validate) {
        std::vector<int> counts, displs;
        if (rank == 0) {
            counts.resize(n_ranks);
            displs.resize(n_ranks);
            for (int r = 0; r < n_ranks; ++r) {
                int rx0, rrows;
                rowRange(n_elems_root, n_ranks, r, rx0, rrows);
                counts[r] = rrows * n_elems_root;
                displs[r] = rx0 * n_elems_root;
            }
        }

        std::vector<val_t> energyData;
        if (rank == 0) energyData.resize(n_elems);
        gatherField(world, world.energy.data(), counts, displs, energyData);

        if (printResults && rank == 0) {
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            std::vector<val_t> fluxData;
            if (rank == 0) fluxData.resize(n_elems);
            gatherField(world, world.flux.data(), counts, displs, fluxData);

            int valid = 1;
            if (rank == 0) {
                valid = validateResults(energyData, fluxData) ? 1 : 0;
            }
            MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!valid) {
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
