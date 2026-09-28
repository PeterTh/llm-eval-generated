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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (local indices)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state (holds this rank's local slice of the mesh, plus halo rows)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Distributed decomposition info: the global mesh is a 2D grid partitioned
// into contiguous row blocks, one block per MPI rank. Each rank stores its
// owned rows plus (up to) one halo row on each side for neighbor exchange.
struct Decomposition {
    int n_elems_root = 0;
    int local_x_start = 0;   // first owned global row index
    int local_n_rows = 0;    // number of owned rows
    int ghost_before = 0;    // 1 if a halo row precedes owned rows (0 or 1)
    int ghost_after = 0;     // 1 if a halo row follows owned rows (0 or 1)
    int ext_rows = 0;        // local_n_rows + ghost_before + ghost_after
    int prev_rank = -1;
    int next_rank = -1;
};

Decomposition computeDecomposition(int n_elems_root, int comm_rank, int comm_size) {
    Decomposition d;
    d.n_elems_root = n_elems_root;

    const int base = n_elems_root / comm_size;
    const int rem = n_elems_root % comm_size;
    d.local_n_rows = base + (comm_rank < rem ? 1 : 0);
    d.local_x_start = comm_rank * base + std::min(comm_rank, rem);

    d.ghost_before = (d.local_x_start > 0) ? 1 : 0;
    d.ghost_after = (d.local_x_start + d.local_n_rows < n_elems_root) ? 1 : 0;
    d.ext_rows = d.local_n_rows + d.ghost_before + d.ghost_after;

    d.prev_rank = comm_rank - 1;
    d.next_rank = comm_rank + 1;

    return d;
}

// Build the local slice (owned rows + halo rows) of a 2D square grid
// unstructured mesh. Connectivity indices are local to this rank's storage.
void buildSquare2D(World& world, const Decomposition& d) {
    const int n_elems_root = d.n_elems_root;
    const int n_elems = d.ext_rows * n_elems_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements (local slice only)
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);

    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity only for owned rows (halo rows are never updated,
    // so their static connectivity is never used).
    for (int rl = d.ghost_before; rl < d.ghost_before + d.local_n_rows; ++rl) {
        const int x = d.local_x_start - d.ghost_before + rl;
        for (int y = 0; y < n_elems_root; ++y) {
            const int local_idx = rl * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within global bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    // Row offsets map 1:1 to local row offsets; halo rows
                    // guarantee any in-range neighbor row is locally present.
                    const int nrl = rl + offsets[n][0];
                    const int neighbor_local_idx = nrl * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_local_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    // (only meaningful for rows this rank owns).
    const int last = n_elems_root - 1;
    auto maybeSetCorner = [&](int gx, int gy, idx_t mat_id) {
        if (gx < d.local_x_start || gx >= d.local_x_start + d.local_n_rows) return;
        const int rl = gx - d.local_x_start + d.ghost_before;
        world.elements_static[rl * n_elems_root + gy].material_idx = mat_id;
    };
    maybeSetCorner(0, 0, INFLOW_MAT_ID);
    maybeSetCorner(0, last, OUTFLOW_MAT_ID);
    maybeSetCorner(last, 0, OUTFLOW_MAT_ID);
    maybeSetCorner(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Exchange halo rows with neighboring ranks (non-blocking).
void exchangeHaloStart(World& world, const Decomposition& d, MPI_Comm comm,
                        MPI_Request reqs[4], int& nreq) {
    nreq = 0;
    ElementDynamic* base = world.elements_dynamic.data();
    const int n_elems_root = d.n_elems_root;

    if (d.ghost_before) {
        MPI_Isend(&base[d.ghost_before * n_elems_root], n_elems_root * sizeof(ElementDynamic),
                   MPI_BYTE, d.prev_rank, 0, comm, &reqs[nreq++]);
        MPI_Irecv(&base[0], n_elems_root * sizeof(ElementDynamic),
                   MPI_BYTE, d.prev_rank, 1, comm, &reqs[nreq++]);
    }
    if (d.ghost_after) {
        const int last_owned = d.ghost_before + d.local_n_rows - 1;
        const int ghost_idx = d.ghost_before + d.local_n_rows;
        MPI_Isend(&base[last_owned * n_elems_root], n_elems_root * sizeof(ElementDynamic),
                   MPI_BYTE, d.next_rank, 1, comm, &reqs[nreq++]);
        MPI_Irecv(&base[ghost_idx * n_elems_root], n_elems_root * sizeof(ElementDynamic),
                   MPI_BYTE, d.next_rank, 0, comm, &reqs[nreq++]);
    }
}

// Run simulation for n_iters iterations on this rank's local slice,
// overlapping halo communication with interior-row computation.
void runSimulation(World& world, const Decomposition& d, const int n_iters, MPI_Comm comm) {
    const int n_elems_root = d.n_elems_root;
    if (d.local_n_rows == 0) return;

    auto updateRow = [&](int rl) {
        const int row_base = rl * n_elems_root;
        for (int y = 0; y < n_elems_root; ++y) {
            const int i = row_base + y;
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
    };

    const int first_owned = d.ghost_before;
    const int last_owned = d.ghost_before + d.local_n_rows - 1;

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request reqs[4];
        int nreq = 0;
        exchangeHaloStart(world, d, comm, reqs, nreq);

        // Overlap: compute interior rows that don't need halo data
        for (int rl = first_owned + 1; rl < last_owned; ++rl) {
            updateRow(rl);
        }

        if (nreq) {
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        }

        // Now compute the boundary rows that depend on halo data
        updateRow(first_owned);
        if (last_owned != first_owned) {
            updateRow(last_owned);
        }

        // Swap buffers
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

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n_elems_root = 512;
    int n_iters = 10;
    int validate = 0;
    int printResults = 0;
    int status = 0;  // 0 = continue, 1 = exit success, 2 = exit error

    // Parse command line arguments on rank 0 only
    if (world_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                status = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                status = 2;
                break;
            }
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0) {
        MPI_Finalize();
        return status == 1 ? 0 : 1;
    }

    int params[4] = {n_elems_root, n_iters, validate, printResults};
    MPI_Bcast(params, 4, MPI_INT, 0, MPI_COMM_WORLD);
    n_elems_root = params[0];
    n_iters = params[1];
    validate = params[2];
    printResults = params[3];

    const int n_elems = n_elems_root * n_elems_root;

    // Restrict the compute communicator to at most n_elems_root ranks
    // (one row minimum per rank); extra ranks idle.
    const int comm_size = std::min(world_size, std::max(n_elems_root, 1));
    const int color = (world_rank < comm_size) ? 0 : MPI_UNDEFINED;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &comm);

    if (world_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (using %d for computation)\n", world_size, comm_size);
        printf("\n");
    }

    if (comm != MPI_COMM_NULL) {
        int rank = 0, size = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        const Decomposition decomp = computeDecomposition(n_elems_root, rank, size);

        if (world_rank == 0) {
            printf("Building unstructured mesh...\n");
        }
        World world;
        buildSquare2D(world, decomp);

        // Calculate memory usage (aggregate, informational)
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        if (world_rank == 0) {
            printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
                   total_mem / (1024.0 * 1024.0),
                   static_mem / (1024.0 * 1024.0),
                   dynamic_mem / (1024.0 * 1024.0));
            printf("\n");

            printf("Running simulation...\n");
        }

        MPI_Barrier(comm);
        auto start = std::chrono::high_resolution_clock::now();

        runSimulation(world, decomp, n_iters, comm);

        MPI_Barrier(comm);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, comm);

        // Gather owned-row results into global order on rank 0
        std::vector<int> counts, displs;
        if (rank == 0) {
            counts.resize(size);
            displs.resize(size);
            const int base = n_elems_root / size;
            const int rem = n_elems_root % size;
            for (int r = 0; r < size; ++r) {
                const int rows = base + (r < rem ? 1 : 0);
                const int start_row = r * base + std::min(r, rem);
                counts[r] = rows * n_elems_root * static_cast<int>(sizeof(ElementDynamic));
                displs[r] = start_row * n_elems_root * static_cast<int>(sizeof(ElementDynamic));
            }
        }

        std::vector<ElementDynamic> global_dynamic;
        if (rank == 0) {
            global_dynamic.resize(n_elems);
        }
        const ElementDynamic* send_ptr = world.elements_dynamic.data() + decomp.ghost_before * n_elems_root;
        const int send_count = decomp.local_n_rows * n_elems_root * static_cast<int>(sizeof(ElementDynamic));
        MPI_Gatherv(send_ptr, send_count, MPI_BYTE,
                    rank == 0 ? global_dynamic.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_BYTE, 0, comm);

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

            // Compute hash for verification
            const uint64_t hash = computeHash(global_dynamic);
            printf("  Result hash: %016lX\n", hash);
            printf("\n");

            // Print results for external validation
            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(global_dynamic.size());
                for (const auto& elem : global_dynamic) {
                    energyData.push_back(elem.current_energy);
                }
                print_results(energyData, "ElementEnergy");
            }
        }

        int valid = 1;
        if (validate && rank == 0) {
            valid = validateResults(global_dynamic) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);

        MPI_Comm_free(&comm);

        if (validate && !valid) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
