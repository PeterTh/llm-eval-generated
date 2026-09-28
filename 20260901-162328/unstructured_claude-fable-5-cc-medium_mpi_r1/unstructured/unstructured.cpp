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

// World state (per-rank partition of the global mesh)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;      // owned elements only
    std::vector<ElementDynamic> elements_dynamic;    // owned elements + halo rows
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// MPI decomposition state: block distribution of grid rows (x dimension)
struct Partition {
    int rank;
    int size;
    int n_root;      // global grid dimension
    int x_start;     // first owned row
    int local_nx;    // number of owned rows
    int prev_rank;   // rank owning row x_start-1 (or MPI_PROC_NULL)
    int next_rank;   // rank owning row x_start+local_nx (or MPI_PROC_NULL)
};

Partition makePartition(int rank, int size, int n_root) {
    Partition p;
    p.rank = rank;
    p.size = size;
    p.n_root = n_root;
    const int base = n_root / size;
    const int rem = n_root % size;
    p.local_nx = base + (rank < rem ? 1 : 0);
    p.x_start = rank * base + std::min(rank, rem);
    p.prev_rank = (p.x_start > 0 && p.local_nx > 0) ? rank - 1 : MPI_PROC_NULL;
    p.next_rank = (p.local_nx > 0 && p.x_start + p.local_nx < n_root) ? rank + 1 : MPI_PROC_NULL;
    return p;
}

// Build this rank's block of the 2D square grid as an unstructured mesh.
// Dynamic arrays hold local_nx+2 rows: row 0 and row local_nx+1 are halo rows
// mirroring the neighbor ranks' boundary rows. Connectivity indices point into
// this halo-padded dynamic array.
void buildSquare2D(World& world, const Partition& p) {
    const int n_root = p.n_root;
    const int local_elems = p.local_nx * n_root;
    const int padded_elems = (p.local_nx + 2) * n_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements
    world.elements_static.resize(local_elems);
    world.elements_dynamic.resize(padded_elems);
    world.elements_dynamic_swap.resize(padded_elems);

    // Initialize all elements with default material and zero energy
    for (int i = 0; i < local_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    for (int i = 0; i < padded_elems; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        world.elements_dynamic_swap[i].current_energy = 0.0;
        world.elements_dynamic_swap[i].total_flux = 0.0;
    }

    // Build connectivity: each element connects to its neighbors in 2D grid.
    // Neighbor order matches the serial code ({1,0},{-1,0},{0,1},{0,-1}) so
    // flux summation order (and thus bitwise results) is identical.
    for (int lx = 0; lx < p.local_nx; ++lx) {
        const int gx = p.x_start + lx;
        for (int y = 0; y < n_root; ++y) {
            ElementStatic& elem = world.elements_static[lx * n_root + y];

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int ngx = gx + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within global bounds
                if (ngx >= 0 && ngx < n_root && ny >= 0 && ny < n_root) {
                    // Index into the halo-padded dynamic array
                    const int neighbor_idx = (ngx - p.x_start + 1) * n_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_root - 1;
    auto setMaterial = [&](int gx, int y, idx_t mat) {
        if (gx >= p.x_start && gx < p.x_start + p.local_nx) {
            world.elements_static[(gx - p.x_start) * n_root + y].material_idx = mat;
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

// Update the owned rows [lx_begin, lx_end) for one iteration
inline void updateRows(World& world, const Partition& p, int lx_begin, int lx_end) {
    const int n_root = p.n_root;
    for (int lx = lx_begin; lx < lx_end; ++lx) {
        const size_t row_static = static_cast<size_t>(lx) * n_root;
        const size_t row_dyn = static_cast<size_t>(lx + 1) * n_root;
        for (int y = 0; y < n_root; ++y) {
            const ElementStatic& elem_static = world.elements_static[row_static + y];
            const ElementDynamic& elem_dyn = world.elements_dynamic[row_dyn + y];
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
            ElementDynamic& elem_write = world.elements_dynamic_swap[row_dyn + y];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
    }
}

// Run simulation for n_iters iterations with halo exchange each step.
// Communication of boundary-row energies is overlapped with the update of
// the interior rows.
void runSimulation(World& world, const Partition& p, const int n_iters) {
    const int n_root = p.n_root;
    const int local_nx = p.local_nx;

    // Only current_energy of halo neighbors is ever read, so exchange just
    // the energies via packed buffers.
    std::vector<val_t> send_lo(n_root), send_hi(n_root);
    std::vector<val_t> recv_lo(n_root), recv_hi(n_root);

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request reqs[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                               MPI_REQUEST_NULL, MPI_REQUEST_NULL};

        if (local_nx > 0) {
            MPI_Irecv(recv_lo.data(), n_root, MPI_DOUBLE, p.prev_rank, 0,
                      MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(recv_hi.data(), n_root, MPI_DOUBLE, p.next_rank, 1,
                      MPI_COMM_WORLD, &reqs[1]);

            const ElementDynamic* first_row = &world.elements_dynamic[n_root];
            const ElementDynamic* last_row = &world.elements_dynamic[static_cast<size_t>(local_nx) * n_root];
            for (int y = 0; y < n_root; ++y) {
                send_lo[y] = first_row[y].current_energy;
                send_hi[y] = last_row[y].current_energy;
            }
            MPI_Isend(send_lo.data(), n_root, MPI_DOUBLE, p.prev_rank, 1,
                      MPI_COMM_WORLD, &reqs[2]);
            MPI_Isend(send_hi.data(), n_root, MPI_DOUBLE, p.next_rank, 0,
                      MPI_COMM_WORLD, &reqs[3]);
        }

        // Update interior rows while boundary energies are in flight
        if (local_nx > 2) {
            updateRows(world, p, 1, local_nx - 1);
        }

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        if (local_nx > 0) {
            // Unpack halo energies
            if (p.prev_rank != MPI_PROC_NULL) {
                ElementDynamic* halo = &world.elements_dynamic[0];
                for (int y = 0; y < n_root; ++y) halo[y].current_energy = recv_lo[y];
            }
            if (p.next_rank != MPI_PROC_NULL) {
                ElementDynamic* halo = &world.elements_dynamic[static_cast<size_t>(local_nx + 1) * n_root];
                for (int y = 0; y < n_root; ++y) halo[y].current_energy = recv_hi[y];
            }

            // Update boundary rows
            updateRows(world, p, 0, std::min(1, local_nx));
            if (local_nx > 1) {
                updateRows(world, p, local_nx - 1, local_nx);
            }
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Gather the owned elements from all ranks into a single global array
// (in global row order) on rank 0.
std::vector<ElementDynamic> gatherResults(const World& world, const Partition& p) {
    const int n_root = p.n_root;
    std::vector<int> counts(p.size), displs(p.size);
    for (int r = 0; r < p.size; ++r) {
        const Partition rp = makePartition(r, p.size, n_root);
        counts[r] = rp.local_nx * n_root * 2;  // two doubles per element
        displs[r] = rp.x_start * n_root * 2;
    }

    std::vector<ElementDynamic> global;
    if (p.rank == 0) {
        global.resize(static_cast<size_t>(n_root) * n_root);
    }
    MPI_Gatherv(&world.elements_dynamic[n_root], p.local_nx * n_root * 2, MPI_DOUBLE,
                p.rank == 0 ? global.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return global;
}

// Validate simulation results (rank 0, on the gathered global state)
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
    int rank, size;
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

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
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

    const Partition part = makePartition(rank, size, n_elems_root);
    World world;
    buildSquare2D(world, part);

    if (rank == 0) {
        // Calculate memory usage (global totals across all ranks)
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

    runSimulation(world, part, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Gather the full result on rank 0 for hashing/validation/output
    std::vector<ElementDynamic> global_dynamic = gatherResults(world, part);

    int exit_code = 0;
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

        // Validation
        if (validate) {
            bool valid = validateResults(global_dynamic);
            if (!valid) {
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
