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
// connected_idx entries refer to indices in this rank's *local extended*
// element arrays (which include one ghost row on each side owned by
// neighboring ranks), not to global mesh indices.
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (local, extended)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Describes how the global n_root x n_root grid is decomposed by row-blocks
// across MPI ranks (1D domain decomposition along x).
struct Decomposition {
    int mpi_rank = 0;
    int mpi_size = 1;
    int n_root = 0;

    int row_start = 0;   // first global row owned by this rank
    int row_end = 0;     // one past the last global row owned by this rank
    bool has_top = false;    // a ghost row above (owned by rank-1) exists
    bool has_bottom = false; // a ghost row below (owned by rank+1) exists

    int local_rows() const { return row_end - row_start; }
    bool active() const { return row_end > row_start; }

    // Offset (in rows) of the first ghost/real row in the local extended arrays.
    int ext_row_offset() const { return has_top ? 1 : 0; }
    int extended_rows() const { return local_rows() + (has_top ? 1 : 0) + (has_bottom ? 1 : 0); }

    // Map a global row index to a local extended-array row index.
    int ext_row(int global_x) const { return global_x - row_start + ext_row_offset(); }
    int ext_idx(int global_x, int global_y) const { return ext_row(global_x) * n_root + global_y; }

    // First/last local (real, non-ghost) extended row indices.
    int first_real_ext_row() const { return ext_row_offset(); }
    int last_real_ext_row() const { return ext_row_offset() + local_rows() - 1; }
};

// Compute the [row_start, row_end) row-block owned by a given rank for an
// n_root x n_root grid split across mpi_size ranks.
static void computeRowRange(int n_root, int mpi_size, int rank, int& row_start, int& row_end) {
    const int rows_per_rank = n_root / mpi_size;
    const int remainder = n_root % mpi_size;
    const int rows = rows_per_rank + (rank < remainder ? 1 : 0);
    row_start = rank * rows_per_rank + std::min(rank, remainder);
    row_end = row_start + rows;
}

static Decomposition makeDecomposition(int n_root, int mpi_rank, int mpi_size) {
    Decomposition d;
    d.mpi_rank = mpi_rank;
    d.mpi_size = mpi_size;
    d.n_root = n_root;
    computeRowRange(n_root, mpi_size, mpi_rank, d.row_start, d.row_end);

    // Active ranks (those owning at least one row) always form a contiguous
    // prefix [0, R) of rank ids, since rows are assigned greedily starting
    // from rank 0. This lets us determine halo neighbors trivially.
    const bool active = d.active();
    d.has_top = active && d.row_start > 0;
    d.has_bottom = active && (d.row_end < n_root);
    return d;
}

// World state (local to this rank, including one ghost row of neighbors on
// each side when applicable)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    Decomposition decomp;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build this rank's local slice of a 2D square grid unstructured mesh.
// Each rank owns a contiguous block of rows [row_start, row_end) plus, when
// applicable, one ghost row above and/or below owned by the neighboring
// rank(s), used to hold halo data exchanged every iteration.
void buildSquare2D(World& world, const Decomposition& decomp) {
    world.decomp = decomp;
    const int n_root = decomp.n_root;
    const int extended = decomp.extended_rows();
    const int n_local = extended * n_root;

    // Initialize materials (identical on every rank; tiny and read-only)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements (including ghost rows)
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity for real (non-ghost) local elements only.
    for (int x = decomp.row_start; x < decomp.row_end; ++x) {
        for (int y = 0; y < n_root; ++y) {
            const int idx = decomp.ext_idx(x, y);
            ElementStatic& elem = world.elements_static[idx];

            // Connect to neighbors (down, up, right, left) in global grid coords
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within global grid bounds
                if (nx >= 0 && nx < n_root && ny >= 0 && ny < n_root) {
                    const int neighbor_idx = decomp.ext_idx(nx, ny);
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    // (only if this rank owns the corresponding row)
    const int last = n_root - 1;
    auto setCornerMaterial = [&](int x, int y, idx_t mat_id) {
        if (x >= decomp.row_start && x < decomp.row_end) {
            world.elements_static[decomp.ext_idx(x, y)].material_idx = mat_id;
        }
    };
    setCornerMaterial(0, 0, INFLOW_MAT_ID);
    setCornerMaterial(0, last, OUTFLOW_MAT_ID);
    setCornerMaterial(last, 0, OUTFLOW_MAT_ID);
    setCornerMaterial(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Exchange one row of ghost data with vertical neighbor ranks so that this
// rank's halo rows reflect the neighbors' current elements_dynamic state.
void exchangeHalos(World& world, MPI_Datatype elem_dynamic_type) {
    const Decomposition& d = world.decomp;
    if (!d.active()) return;
    const int n_root = d.n_root;

    MPI_Request requests[4];
    int nreq = 0;

    ElementDynamic* base = world.elements_dynamic.data();
    ElementDynamic* first_real_row = base + static_cast<size_t>(d.first_real_ext_row()) * n_root;
    ElementDynamic* last_real_row = base + static_cast<size_t>(d.last_real_ext_row()) * n_root;
    ElementDynamic* top_ghost_row = base;  // row 0 when has_top
    ElementDynamic* bottom_ghost_row = base + static_cast<size_t>(d.extended_rows() - 1) * n_root;

    if (d.has_top) {
        const int neighbor = d.mpi_rank - 1;
        MPI_Irecv(top_ghost_row, n_root, elem_dynamic_type, neighbor, 0, MPI_COMM_WORLD, &requests[nreq++]);
        MPI_Isend(first_real_row, n_root, elem_dynamic_type, neighbor, 1, MPI_COMM_WORLD, &requests[nreq++]);
    }
    if (d.has_bottom) {
        const int neighbor = d.mpi_rank + 1;
        MPI_Irecv(bottom_ghost_row, n_root, elem_dynamic_type, neighbor, 1, MPI_COMM_WORLD, &requests[nreq++]);
        MPI_Isend(last_real_row, n_root, elem_dynamic_type, neighbor, 0, MPI_COMM_WORLD, &requests[nreq++]);
    }

    if (nreq > 0) {
        MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, MPI_Datatype elem_dynamic_type) {
    const Decomposition& d = world.decomp;
    if (!d.active()) {
        // Ranks without any local rows still participate in the halo
        // exchange (as a no-op) so collective-free point-to-point calls
        // made by their neighbors line up correctly.
        for (int iter = 0; iter < n_iters; ++iter) {
            exchangeHalos(world, elem_dynamic_type);
        }
        return;
    }

    const int n_root = d.n_root;
    const int row_lo = d.first_real_ext_row();
    const int row_hi = d.last_real_ext_row();

    for (int iter = 0; iter < n_iters; ++iter) {
        // Refresh ghost rows with the neighbors' latest state before computing this iteration
        exchangeHalos(world, elem_dynamic_type);

        // Update all real (non-ghost) local elements
        for (int row = row_lo; row <= row_hi; ++row) {
            for (int y = 0; y < n_root; ++y) {
                const size_t i = static_cast<size_t>(row) * n_root + y;
                const ElementStatic& elem_static = world.elements_static[i];
                const ElementDynamic& elem_dyn = world.elements_dynamic[i];
                const Material& mat = world.materials[elem_static.material_idx];

                // Start with external flow
                val_t total_flux = mat.external_flow;

                // Add flux from all connected elements
                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const idx_t neighbor_idx = elem_static.connected_idx[j];
                    const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
                }

                // Update element state
                ElementDynamic& elem_write = world.elements_dynamic_swap[i];
                elem_write.current_energy = elem_dyn.current_energy + total_flux;
                elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (global, MPI-Allreduce'd across ranks). Only
// the calling rank's local partial data (real elements only) is expected;
// each rank must call this collectively.
bool validateResults(const std::vector<ElementDynamic>& local_elements, bool is_root) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : local_elements) {
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(elem.current_energy, local_energy_max);
        local_energy_min = std::min(elem.current_energy, local_energy_min);
    }

    val_t energy_sum = 0.0, flux_sum = 0.0, energy_max = 0.0, energy_min = 0.0;
    MPI_Allreduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    if (is_root) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    }

    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;

    if (!std::isfinite(energy_sum)) {
        if (is_root) printf("  ERROR: Energy sum is not finite\n");
        return false;
    }

    if (is_root && std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        // Don't fail validation as this can happen with external flows
    }

    if (!std::isfinite(flux_sum)) {
        if (is_root) printf("  ERROR: Flux sum is not finite\n");
        return false;
    }

    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        if (is_root) printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    if (is_root) printf("  Validation: PASSED\n");
    return true;
}

// Compute a simple hash of the results for verification. Each rank hashes
// its own real (non-ghost) elements using their *global* index, then the
// per-element XOR contributions are combined across ranks with MPI_BXOR;
// since XOR is commutative/associative this yields a value identical to
// hashing the whole array on a single process.
uint64_t computeHash(const std::vector<ElementDynamic>& local_elements, int global_index_offset) {
    uint64_t hash = 0;
    for (size_t k = 0; k < local_elements.size(); ++k) {
        const size_t i = global_index_offset + k;
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&local_elements[k].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&local_elements[k].total_flux);
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
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    const bool is_root = (mpi_rank == 0);

    MPI_Datatype elem_dynamic_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &elem_dynamic_type);
    MPI_Type_commit(&elem_dynamic_type);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (mpirun distributes identical argv to every rank)
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
            MPI_Type_free(&elem_dynamic_type);
            MPI_Finalize();
            return 0;
        } else {
            if (is_root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Type_free(&elem_dynamic_type);
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
        printf("MPI ranks: %d\n", mpi_size);
        printf("\n");
    }

    // Build this rank's local slice of the unstructured mesh
    if (is_root) printf("Building unstructured mesh...\n");
    World world;
    const Decomposition decomp = makeDecomposition(n_elems_root, mpi_rank, mpi_size);
    buildSquare2D(world, decomp);

    // Calculate memory usage (aggregated across all ranks)
    const size_t local_static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t local_dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
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
    }

    // Run simulation
    if (is_root) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, elem_dynamic_type);

    auto end = std::chrono::high_resolution_clock::now();
    auto local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (is_root) printf("Computation time: %ld ms\n", duration_ms);

    // Calculate performance metrics (aggregate throughput across the whole cluster)
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9;

    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;

    // Gather the real (non-ghost) elements from every rank, in global index
    // order, so hash/print/validation logic exactly matches single-process semantics.
    const int local_rows = decomp.local_rows();
    const int local_count = local_rows * n_elems_root;
    std::vector<ElementDynamic> local_real(local_count);
    if (local_count > 0) {
        const ElementDynamic* src = world.elements_dynamic.data() +
            static_cast<size_t>(decomp.first_real_ext_row()) * n_elems_root;
        std::copy(src, src + local_count, local_real.begin());
    }

    const uint64_t hash = computeHash(local_real, decomp.row_start * n_elems_root);

    if (is_root) {
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation: gather the global energy array
    // (in global index order) onto rank 0 so output matches single-process semantics exactly.
    if (printResults) {
        std::vector<int> recvcounts, displs;
        if (is_root) {
            recvcounts.resize(mpi_size);
            displs.resize(mpi_size);
            for (int r = 0; r < mpi_size; ++r) {
                int rs, re;
                computeRowRange(n_elems_root, mpi_size, r, rs, re);
                recvcounts[r] = (re - rs) * n_elems_root;
                displs[r] = rs * n_elems_root;
            }
        }

        std::vector<ElementDynamic> global_elements(is_root ? n_elems : 0);
        MPI_Gatherv(local_real.data(), local_count, elem_dynamic_type,
                    is_root ? global_elements.data() : nullptr,
                    is_root ? recvcounts.data() : nullptr,
                    is_root ? displs.data() : nullptr,
                    elem_dynamic_type, 0, MPI_COMM_WORLD);

        if (is_root) {
            std::vector<double> energyData;
            energyData.reserve(global_elements.size());
            for (const auto& elem : global_elements) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }
    }

    // Validation (each rank contributes only its own real elements;
    // validateResults combines them across ranks internally via MPI_Allreduce).
    if (validate) {
        bool valid = validateResults(local_real, is_root);
        if (!valid) {
            MPI_Type_free(&elem_dynamic_type);
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Type_free(&elem_dynamic_type);
    MPI_Finalize();
    return 0;
}
