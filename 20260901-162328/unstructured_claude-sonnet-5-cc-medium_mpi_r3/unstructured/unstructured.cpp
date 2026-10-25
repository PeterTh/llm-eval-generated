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

// Static connectivity information for each element. connected_idx entries are
// offsets into World::energy_buffer (the local extended-with-halo domain),
// not global mesh indices, so the hot loop never needs to branch/translate.
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Offsets into energy_buffer
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state: holds only the elements owned by this MPI rank, plus the
// halo (ghost) metadata needed to exchange boundary energies with
// neighboring ranks each iteration.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;

    int n_elems_root = 0;

    // Rows of the global n_elems_root x n_elems_root grid owned by this rank.
    uint64_t row_start = 0;
    uint64_t row_end = 0;

    // Rows including the halo row borrowed from each neighbor (if any).
    uint64_t ext_row_start = 0;
    uint64_t ext_row_end = 0;

    int prev_rank = MPI_PROC_NULL;
    int next_rank = MPI_PROC_NULL;

    // Energy values for the extended (local + halo) row range, indexed by
    // (global_idx - ext_row_start * n_elems_root).
    std::vector<val_t> energy_buffer;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Compute the contiguous row range [row_start, row_end) owned by `rank` out
// of `n_elems_root` total rows, split as evenly as possible across `size`
// ranks (first `n_elems_root % size` ranks get one extra row).
void computeRowRange(int n_elems_root, int rank, int size, uint64_t& row_start, uint64_t& row_end) {
    const uint64_t base = static_cast<uint64_t>(n_elems_root) / size;
    const uint64_t rem = static_cast<uint64_t>(n_elems_root) % size;
    row_start = base * rank + std::min<uint64_t>(rank, rem);
    row_end = row_start + base + (static_cast<uint64_t>(rank) < rem ? 1 : 0);
}

// Build a 2D square grid as an unstructured mesh, partitioned into
// contiguous row-blocks across MPI ranks for distributed-memory parallelism.
// Each rank stores only its owned elements; connectivity crossing a rank
// boundary is resolved through a halo exchange of a single ghost row.
void buildSquare2D(World& world, const int n_elems_root, int rank, int size) {
    world.n_elems_root = n_elems_root;

    computeRowRange(n_elems_root, rank, size, world.row_start, world.row_end);

    // Find the nearest ranks (possibly skipping idle ranks with zero rows)
    // that own the row immediately before/after this rank's range.
    world.prev_rank = MPI_PROC_NULL;
    if (world.row_start > 0) {
        for (int r = rank - 1; r >= 0; --r) {
            uint64_t rs, re;
            computeRowRange(n_elems_root, r, size, rs, re);
            if (re > rs) {
                world.prev_rank = r;
                break;
            }
        }
    }
    world.next_rank = MPI_PROC_NULL;
    if (world.row_end < static_cast<uint64_t>(n_elems_root)) {
        for (int r = rank + 1; r < size; ++r) {
            uint64_t rs, re;
            computeRowRange(n_elems_root, r, size, rs, re);
            if (re > rs) {
                world.next_rank = r;
                break;
            }
        }
    }

    world.ext_row_start = (world.row_start > 0) ? world.row_start - 1 : world.row_start;
    world.ext_row_end = (world.row_end < static_cast<uint64_t>(n_elems_root)) ? world.row_end + 1 : world.row_end;

    const uint64_t local_n = (world.row_end - world.row_start) * n_elems_root;
    const uint64_t ext_n = (world.ext_row_end - world.ext_row_start) * n_elems_root;

    // Initialize materials (identical on every rank)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate local elements and the extended halo energy buffer
    world.elements_static.resize(local_n);
    world.elements_dynamic.resize(local_n);
    world.elements_dynamic_swap.resize(local_n);
    world.energy_buffer.assign(ext_n, 0.0);

    for (uint64_t i = 0; i < local_n; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity: each element connects to its neighbors in the 2D
    // grid. Neighbor references are stored as offsets into energy_buffer.
    const uint64_t buffer_base = world.ext_row_start * static_cast<uint64_t>(n_elems_root);
    for (uint64_t x = world.row_start; x < world.row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const uint64_t local_idx = (x - world.row_start) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int64_t nx = static_cast<int64_t>(x) + offsets[n][0];
                const int64_t ny = static_cast<int64_t>(y) + offsets[n][1];

                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const uint64_t neighbor_global_idx = static_cast<uint64_t>(nx) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_global_idx - buffer_base;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const uint64_t last = n_elems_root - 1;
    auto setCornerMaterial = [&](uint64_t gx, uint64_t gy, idx_t mat_id) {
        if (gx >= world.row_start && gx < world.row_end) {
            const uint64_t local_idx = (gx - world.row_start) * n_elems_root + gy;
            world.elements_static[local_idx].material_idx = mat_id;
        }
    };
    setCornerMaterial(0, 0, INFLOW_MAT_ID);
    setCornerMaterial(0, last, OUTFLOW_MAT_ID);
    setCornerMaterial(last, 0, OUTFLOW_MAT_ID);
    setCornerMaterial(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, val_t other_energy) {
    return (other_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Compute the flux update for a single local grid row, reading neighbor
// energies from the (already up to date) energy_buffer.
inline void computeRow(World& world, uint64_t local_row) {
    const int n_elems_root = world.n_elems_root;
    const uint64_t row_base = local_row * n_elems_root;

    for (int y = 0; y < n_elems_root; ++y) {
        const uint64_t i = row_base + y;
        const ElementStatic& elem_static = world.elements_static[i];
        const ElementDynamic& elem_dyn = world.elements_dynamic[i];
        const Material& mat = world.materials[elem_static.material_idx];

        val_t total_flux = mat.external_flow;

        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const val_t neighbor_energy = world.energy_buffer[elem_static.connected_idx[j]];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_energy);
        }

        ElementDynamic& elem_write = world.elements_dynamic_swap[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations, exchanging a single halo row with
// neighboring ranks each iteration and overlapping that communication with
// the computation of interior rows that do not depend on the halo.
void runSimulation(World& world, const int n_iters, MPI_Comm comm) {
    const int n_elems_root = world.n_elems_root;
    const uint64_t local_n_rows = world.row_end - world.row_start;
    if (local_n_rows == 0) {
        return;  // Idle rank: owns no rows
    }

    const bool has_prev = (world.prev_rank != MPI_PROC_NULL);
    const bool has_next = (world.next_rank != MPI_PROC_NULL);
    const uint64_t local_offset = (world.row_start - world.ext_row_start) * n_elems_root;
    const uint64_t local_n = local_n_rows * n_elems_root;

    constexpr int TAG_FORWARD = 1;   // messages traveling towards +x
    constexpr int TAG_BACKWARD = 2;  // messages traveling towards -x

    for (int iter = 0; iter < n_iters; ++iter) {
        // Publish this rank's current energies into the shared buffer so
        // that both local neighbor lookups and the halo exchange see them.
        for (uint64_t i = 0; i < local_n; ++i) {
            world.energy_buffer[local_offset + i] = world.elements_dynamic[i].current_energy;
        }

        MPI_Request reqs[4];
        int nreq = 0;

        if (has_next) {
            val_t* last_row = &world.energy_buffer[local_offset + local_n - n_elems_root];
            val_t* ghost_after = &world.energy_buffer[local_offset + local_n];
            MPI_Isend(last_row, n_elems_root, MPI_DOUBLE, world.next_rank, TAG_FORWARD, comm, &reqs[nreq++]);
            MPI_Irecv(ghost_after, n_elems_root, MPI_DOUBLE, world.next_rank, TAG_BACKWARD, comm, &reqs[nreq++]);
        }
        if (has_prev) {
            val_t* first_row = &world.energy_buffer[local_offset];
            val_t* ghost_before = &world.energy_buffer[0];
            MPI_Isend(first_row, n_elems_root, MPI_DOUBLE, world.prev_rank, TAG_BACKWARD, comm, &reqs[nreq++]);
            MPI_Irecv(ghost_before, n_elems_root, MPI_DOUBLE, world.prev_rank, TAG_FORWARD, comm, &reqs[nreq++]);
        }

        // Interior rows only reference local energies, so compute them while
        // the halo exchange is in flight.
        if (local_n_rows >= 3) {
            for (uint64_t r = 1; r + 1 < local_n_rows; ++r) {
                computeRow(world, r);
            }
        }

        if (nreq > 0) {
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        }

        // Boundary rows need the freshly-arrived halo data.
        computeRow(world, 0);
        if (local_n_rows > 1) {
            computeRow(world, local_n_rows - 1);
        }

        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
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
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
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

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool is_root = (rank == 0);

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
            if (is_root) {
                printUsage(argv[0]);
            }
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
        printf("Building unstructured mesh...\n");
    }

    // Build this rank's partition of the unstructured mesh
    World world;
    buildSquare2D(world, n_elems_root, rank, size);

    // Calculate memory usage (summed across all ranks, matching the total
    // footprint of the distributed dataset)
    const size_t local_static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t local_dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2
                                    + world.energy_buffer.size() * sizeof(val_t);
    size_t static_mem = 0, dynamic_mem = 0;
    MPI_Reduce(&local_static_mem, &static_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dynamic_mem, &dynamic_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (is_root) {
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the full global result (in original element order) onto the
    // root rank so that hashing/validation/printing remain bit-identical to
    // the single-process implementation regardless of rank count.
    std::vector<ElementDynamic> global_elements;
    std::vector<int> recvcounts, displs;
    if (is_root) {
        global_elements.resize(n_elems);
        recvcounts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            uint64_t rs, re;
            computeRowRange(n_elems_root, r, size, rs, re);
            recvcounts[r] = static_cast<int>((re - rs) * n_elems_root * sizeof(ElementDynamic));
            displs[r] = static_cast<int>(rs * n_elems_root * sizeof(ElementDynamic));
        }
    }
    const int local_send_bytes = static_cast<int>(world.elements_dynamic.size() * sizeof(ElementDynamic));
    MPI_Gatherv(world.elements_dynamic.data(), local_send_bytes, MPI_BYTE,
                is_root ? global_elements.data() : nullptr,
                is_root ? recvcounts.data() : nullptr,
                is_root ? displs.data() : nullptr,
                MPI_BYTE, 0, MPI_COMM_WORLD);

    int valid_flag = 1;

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

        // Compute hash for verification
        const uint64_t hash = computeHash(global_elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global_elements.size());
            for (const auto& elem : global_elements) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            valid_flag = validateResults(global_elements) ? 1 : 0;
        }
    }

    MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    if (validate && !valid_flag) {
        return 1;
    }
    return 0;
}
