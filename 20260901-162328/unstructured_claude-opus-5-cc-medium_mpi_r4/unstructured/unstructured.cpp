#include <algorithm>
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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (into the local
                                              // energy array, halo included)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Distributed world state.
//
// The 2D grid (element idx = x * n + y) is decomposed into contiguous blocks of rows (x).
// Every rank stores the static connectivity of its own rows plus one halo row of dynamic
// state on each side that is owned by a neighboring rank.
//
// Dynamic state is kept in structure-of-arrays form: the energy is the only quantity read by
// neighboring elements, so the halo exchange moves contiguous rows of energy values. The
// accumulated flux is purely element-local and is updated in place.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;  // n_local_rows * n_elems_root entries
    std::vector<val_t> energy;                   // (halo_lo + n_local_rows + halo_hi) * n
    std::vector<val_t> energy_swap;              // same layout as energy
    std::vector<val_t> total_flux;               // n_local_rows * n_elems_root entries

    int n_elems_root = 0;
    int row_begin = 0;      // first globally owned row
    int n_local_rows = 0;   // number of owned rows
    int halo_lo = 0;        // 1 if a halo row exists below (lower x)
    int halo_hi = 0;        // 1 if a halo row exists above (higher x)
    int rank_lo = MPI_PROC_NULL;  // rank owning the lower halo row
    int rank_hi = MPI_PROC_NULL;  // rank owning the upper halo row
    size_t owned_offset = 0;      // index of the first owned element in energy[]
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Block row decomposition: rank r owns rows [begin, begin + count)
static inline void rowRange(int n_rows, int rank, int n_ranks, int& begin, int& count) {
    const int base = n_rows / n_ranks;
    const int rem = n_rows % n_ranks;
    count = base + (rank < rem ? 1 : 0);
    begin = rank * base + std::min(rank, rem);
}

// Build the local part of a 2D square grid represented as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int n_ranks) {
    world.n_elems_root = n_elems_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    int row_begin = 0, n_local_rows = 0;
    rowRange(n_elems_root, rank, n_ranks, row_begin, n_local_rows);
    world.row_begin = row_begin;
    world.n_local_rows = n_local_rows;

    const int row_end = row_begin + n_local_rows;
    world.halo_lo = (n_local_rows > 0 && row_begin > 0) ? 1 : 0;
    world.halo_hi = (n_local_rows > 0 && row_end < n_elems_root) ? 1 : 0;
    world.rank_lo = world.halo_lo ? rank - 1 : MPI_PROC_NULL;
    world.rank_hi = world.halo_hi ? rank + 1 : MPI_PROC_NULL;

    const size_t n = static_cast<size_t>(n_elems_root);
    const size_t n_owned = static_cast<size_t>(n_local_rows) * n;
    const size_t n_dyn = (static_cast<size_t>(n_local_rows) + world.halo_lo + world.halo_hi) * n;
    world.owned_offset = static_cast<size_t>(world.halo_lo) * n;

    world.elements_static.resize(n_owned);
    world.energy.assign(n_dyn, 0.0);
    world.energy_swap.assign(n_dyn, 0.0);
    world.total_flux.assign(n_owned, 0.0);

    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int lx = 0; lx < n_local_rows; ++lx) {
        const int x = row_begin + lx;
        for (int y = 0; y < n_elems_root; ++y) {
            ElementStatic& elem = world.elements_static[static_cast<size_t>(lx) * n + y];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int i = 0; i < 4; ++i) {
                const int nx = x + offsets[i][0];
                const int ny = y + offsets[i][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    // Local index into the energy array (halo rows included)
                    const size_t neighbor_idx =
                        static_cast<size_t>(nx - row_begin + world.halo_lo) * n + ny;
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
        if (x >= row_begin && x < row_end) {
            world.elements_static[static_cast<size_t>(x - row_begin) * n + y].material_idx = mat;
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
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// Update all elements of the local rows [lx_begin, lx_end)
static inline void updateRows(World& world, int lx_begin, int lx_end) {
    const size_t n = static_cast<size_t>(world.n_elems_root);
    const ElementStatic* __restrict elems = world.elements_static.data();
    const val_t* __restrict energy = world.energy.data();
    val_t* __restrict energy_out = world.energy_swap.data();
    val_t* __restrict flux_acc = world.total_flux.data();
    const Material* __restrict materials = world.materials.data();
    const size_t owned_offset = world.owned_offset;

    const size_t begin = static_cast<size_t>(lx_begin) * n;
    const size_t end = static_cast<size_t>(lx_end) * n;

    for (size_t i = begin; i < end; ++i) {
        const ElementStatic& elem_static = elems[i];
        const val_t this_energy = energy[owned_offset + i];
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
        energy_out[owned_offset + i] = this_energy + total_flux;
        flux_acc[i] += std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n = static_cast<size_t>(world.n_elems_root);
    const int n_local_rows = world.n_local_rows;
    if (n_local_rows == 0) {
        return;
    }

    const size_t owned_offset = world.owned_offset;
    const size_t last_row_offset = owned_offset + static_cast<size_t>(n_local_rows - 1) * n;
    const size_t hi_halo_offset = owned_offset + static_cast<size_t>(n_local_rows) * n;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange halo rows of the energy field with the neighboring ranks
        MPI_Request requests[4];
        int n_requests = 0;
        val_t* energy = world.energy.data();

        if (world.halo_lo) {
            MPI_Irecv(energy, static_cast<int>(n), MPI_DOUBLE, world.rank_lo, 0,
                      MPI_COMM_WORLD, &requests[n_requests++]);
        }
        if (world.halo_hi) {
            MPI_Irecv(energy + hi_halo_offset, static_cast<int>(n), MPI_DOUBLE, world.rank_hi, 1,
                      MPI_COMM_WORLD, &requests[n_requests++]);
        }
        if (world.halo_lo) {
            MPI_Isend(energy + owned_offset, static_cast<int>(n), MPI_DOUBLE, world.rank_lo, 1,
                      MPI_COMM_WORLD, &requests[n_requests++]);
        }
        if (world.halo_hi) {
            MPI_Isend(energy + last_row_offset, static_cast<int>(n), MPI_DOUBLE, world.rank_hi, 0,
                      MPI_COMM_WORLD, &requests[n_requests++]);
        }

        // Update the interior rows, which do not depend on the halo, while the exchange is
        // in flight
        if (n_local_rows > 2) {
            updateRows(world, 1, n_local_rows - 1);
        }

        if (n_requests > 0) {
            MPI_Waitall(n_requests, requests, MPI_STATUSES_IGNORE);
        }

        // Update the rows touching the halo
        updateRows(world, 0, 1);
        if (n_local_rows > 1) {
            updateRows(world, n_local_rows - 1, n_local_rows);
        }

        // Swap buffers
        std::swap(world.energy, world.energy_swap);
    }
}

// Validate simulation results (on the gathered global state)
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

// Compute a simple hash of the local results; the global hash is the XOR reduction of all
// local hashes, which matches the sequential result as the hash is a commutative XOR fold
uint64_t computeLocalHash(const World& world) {
    const size_t n = static_cast<size_t>(world.n_elems_root);
    const size_t global_offset = static_cast<size_t>(world.row_begin) * n;
    const size_t n_owned = static_cast<size_t>(world.n_local_rows) * n;
    const val_t* energy = world.energy.data() + world.owned_offset;
    const val_t* flux = world.total_flux.data();

    uint64_t hash = 0;
    for (size_t i = 0; i < n_owned; ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&energy[i]);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&flux[i]);
        const uint64_t gi = global_offset + i;
        hash ^= (*e_ptr + gi) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + gi) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

// Gather a distributed, row-block-distributed field onto rank 0 (in global element order)
void gatherField(const World& world, const val_t* local_data, std::vector<val_t>& out,
                 int rank, int n_ranks) {
    const size_t n = static_cast<size_t>(world.n_elems_root);
    const int local_count = static_cast<int>(static_cast<size_t>(world.n_local_rows) * n);

    std::vector<int> counts, displs;
    if (rank == 0) {
        counts.resize(n_ranks);
        displs.resize(n_ranks);
        int displ = 0;
        for (int r = 0; r < n_ranks; ++r) {
            int rb = 0, rc = 0;
            rowRange(world.n_elems_root, r, n_ranks, rb, rc);
            counts[r] = static_cast<int>(static_cast<size_t>(rc) * n);
            displs[r] = displ;
            displ += counts[r];
        }
        out.resize(static_cast<size_t>(displ));
    }

    MPI_Gatherv(local_data, local_count, MPI_DOUBLE,
                rank == 0 ? out.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
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
        // Calculate (global) memory usage
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
    const double start = MPI_Wtime();

    runSimulation(world, n_iters);

    double elapsed = MPI_Wtime() - start;
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const long duration_ms = static_cast<long>(elapsed * 1000.0);

    // Compute hash for verification (XOR reduction over all ranks)
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

    // Gather the global state on rank 0 if it is needed for output or validation
    std::vector<val_t> global_energy, global_flux;
    if (printResults || validate) {
        gatherField(world, world.energy.data() + world.owned_offset, global_energy, rank, n_ranks);
    }
    if (validate) {
        gatherField(world, world.total_flux.data(), global_flux, rank, n_ranks);
    }

    int exit_code = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(global_energy, "ElementEnergy");
        }

        // Validation
        if (validate) {
            if (!validateResults(global_energy, global_flux)) {
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exit_code;
}
