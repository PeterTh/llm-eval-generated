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

// World state (MPI-parallel, with ghost rows)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;  // local owned elements only
    std::vector<ElementDynamic> elements_dynamic; // owned + ghost rows
    std::vector<ElementDynamic> elements_dynamic_swap; // owned + ghost rows

    // MPI domain decomposition info
    int rank, nprocs;
    int n_elems_root;        // global grid dimension
    int local_nx;            // number of owned rows on this rank
    int global_x_start;      // global x-index of first owned row
    int up_rank;             // rank owning the row above (x-1), or MPI_PROC_NULL
    int down_rank;           // rank owning the row below (x+1), or MPI_PROC_NULL
    int ny;                  // grid width (== n_elems_root)

    // Indices of ghost rows in elements_dynamic
    int ghost_up_idx;        // index of ghost row from up neighbor (x-1 direction)
    int ghost_down_idx;      // index of ghost row from down neighbor (x+1 direction)
    bool has_ghost_up;
    bool has_ghost_down;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh (MPI-parallel, 1D decomposition along x)
void buildSquare2D(World& world, const int n_elems_root, int rank, int nprocs) {
    world.rank = rank;
    world.nprocs = nprocs;
    world.n_elems_root = n_elems_root;
    world.ny = n_elems_root;

    // Distribute rows among ranks as evenly as possible
    int base_rows = n_elems_root / nprocs;
    int extra = n_elems_root % nprocs;
    // First 'extra' ranks get one additional row
    int local_nx, global_x_start;
    if (rank < extra) {
        local_nx = base_rows + 1;
        global_x_start = rank * (base_rows + 1);
    } else {
        local_nx = base_rows;
        global_x_start = extra * (base_rows + 1) + (rank - extra) * base_rows;
    }
    world.local_nx = local_nx;
    world.global_x_start = global_x_start;

    // Neighbor ranks
    world.up_rank = (global_x_start > 0) ? rank - 1 : MPI_PROC_NULL;
    world.down_rank = (global_x_start + local_nx < n_elems_root) ? rank + 1 : MPI_PROC_NULL;
    // Handle case where rank has no up/down neighbor due to decomposition
    if (rank == 0) world.up_rank = MPI_PROC_NULL;
    if (rank == nprocs - 1) world.down_rank = MPI_PROC_NULL;
    // More robust: recompute based on actual ownership
    world.up_rank = (global_x_start > 0) ? (rank > 0 ? rank - 1 : MPI_PROC_NULL) : MPI_PROC_NULL;
    world.down_rank = (global_x_start + local_nx < n_elems_root) ? (rank < nprocs - 1 ? rank + 1 : MPI_PROC_NULL) : MPI_PROC_NULL;

    world.has_ghost_up = (world.up_rank != MPI_PROC_NULL);
    world.has_ghost_down = (world.down_rank != MPI_PROC_NULL);

    // Layout: [ghost_up] [owned rows] [ghost_down]
    // ghost_up is at index 0 if present, owned starts at offset ghost_up ? 1 : 0
    int owned_offset = world.has_ghost_up ? 1 : 0;
    world.ghost_up_idx = 0;
    world.ghost_down_idx = owned_offset + local_nx;

    int total_rows = local_nx + (world.has_ghost_up ? 1 : 0) + (world.has_ghost_down ? 1 : 0);
    int total_elems = total_rows * n_elems_root;

    // Initialize materials (same on all ranks)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate: static only for owned, dynamic for owned+ghost
    world.elements_static.resize(local_nx * n_elems_root);
    world.elements_dynamic.resize(total_elems);
    world.elements_dynamic_swap.resize(total_elems);

    // Initialize all dynamic to zero
    for (int i = 0; i < total_elems; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        world.elements_dynamic_swap[i].current_energy = 0.0;
        world.elements_dynamic_swap[i].total_flux = 0.0;
    }

    // Build static connectivity for owned elements
    // Local indexing: owned row r (0..local_nx-1) -> global x = global_x_start + r
    // Local flat index for owned: r * ny + y
    // Dynamic array index for owned row r: (owned_offset + r) * ny + y
    for (int r = 0; r < local_nx; ++r) {
        const int gx = global_x_start + r; // global x coordinate
        for (int y = 0; y < n_elems_root; ++y) {
            const int local_idx = r * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            // Neighbors: up(gx-1,y), down(gx+1,y), left(gx,y-1), right(gx,y+1)
            // For up/down neighbors, we store the index into elements_dynamic
            // For left/right, same row so also into elements_dynamic

            const int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

            for (int n = 0; n < 4; ++n) {
                const int ngx = gx + offsets[n][0];
                const int ngy = y + offsets[n][1];

                if (ngx >= 0 && ngx < n_elems_root && ngy >= 0 && ngy < n_elems_root) {
                    // Compute dynamic array index for neighbor
                    int dyn_idx;
                    if (ngx >= global_x_start && ngx < global_x_start + local_nx) {
                        // Owned neighbor
                        int lr = ngx - global_x_start;
                        dyn_idx = (owned_offset + lr) * n_elems_root + ngy;
                    } else if (ngx == global_x_start - 1) {
                        // Ghost up
                        dyn_idx = world.ghost_up_idx * n_elems_root + ngy;
                    } else if (ngx == global_x_start + local_nx) {
                        // Ghost down
                        dyn_idx = world.ghost_down_idx * n_elems_root + ngy;
                    } else {
                        continue; // shouldn't happen for 4-connected grid
                    }
                    elem.connected_idx[elem.num_connections] = dyn_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow (only if this rank owns them)
    auto set_material = [&](int gx, int gy, idx_t mat_id) {
        if (gx >= global_x_start && gx < global_x_start + local_nx) {
            int lr = gx - global_x_start;
            int local_idx = lr * n_elems_root + gy;
            world.elements_static[local_idx].material_idx = mat_id;
        }
    };
    const int last = n_elems_root - 1;
    set_material(0, 0, INFLOW_MAT_ID);
    set_material(0, last, OUTFLOW_MAT_ID);
    set_material(last, 0, OUTFLOW_MAT_ID);
    set_material(last, last, INFLOW_MAT_ID);
}

// Exchange ghost rows with neighbors
void exchangeGhostRows(World& world) {
    const int ny = world.ny;
    const int owned_offset = world.has_ghost_up ? 1 : 0;
    const int row_size = ny; // number of elements per row

    // We need to send/receive current_energy for ghost rows
    // Pack send buffers from boundary owned rows, unpack into ghost rows

    // Send down (last owned row) to down_rank, receive into ghost_up from up_rank
    // Send up (first owned row) to up_rank, receive into ghost_down from down_rank

    std::vector<val_t> send_down_buf(row_size), send_up_buf(row_size);
    std::vector<val_t> recv_up_buf(row_size), recv_down_buf(row_size);

    // Pack: last owned row -> send_down
    int last_owned_row = owned_offset + world.local_nx - 1;
    int first_owned_row = owned_offset;

    for (int y = 0; y < ny; ++y) {
        send_down_buf[y] = world.elements_dynamic[last_owned_row * ny + y].current_energy;
        send_up_buf[y] = world.elements_dynamic[first_owned_row * ny + y].current_energy;
    }

    // Exchange: send down to down_rank, receive from up_rank into recv_up_buf (goes into ghost_up)
    //           send up to up_rank, receive from down_rank into recv_down_buf (goes into ghost_down)
    MPI_Request reqs[4];
    int nreqs = 0;

    if (world.down_rank != MPI_PROC_NULL) {
        MPI_Isend(send_down_buf.data(), row_size, MPI_DOUBLE, world.down_rank, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        MPI_Irecv(recv_down_buf.data(), row_size, MPI_DOUBLE, world.down_rank, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    if (world.up_rank != MPI_PROC_NULL) {
        MPI_Isend(send_up_buf.data(), row_size, MPI_DOUBLE, world.up_rank, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        MPI_Irecv(recv_up_buf.data(), row_size, MPI_DOUBLE, world.up_rank, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
    }

    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

    // Unpack received ghost rows
    if (world.has_ghost_up) {
        for (int y = 0; y < ny; ++y) {
            world.elements_dynamic[world.ghost_up_idx * ny + y].current_energy = recv_up_buf[y];
        }
    }
    if (world.has_ghost_down) {
        for (int y = 0; y < ny; ++y) {
            world.elements_dynamic[world.ghost_down_idx * ny + y].current_energy = recv_down_buf[y];
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations (MPI parallel)
void runSimulation(World& world, const int n_iters) {
    const int local_owned = world.local_nx * world.ny;
    const int owned_offset = world.has_ghost_up ? 1 : 0;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange ghost rows before computation
        exchangeGhostRows(world);

        // Update all owned elements
        for (int i = 0; i < local_owned; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            // Dynamic index: owned elements start at owned_offset in the row dimension
            int r = i / world.ny;
            int y = i % world.ny;
            int dyn_i = (owned_offset + r) * world.ny + y;
            const ElementDynamic& elem_dyn = world.elements_dynamic[dyn_i];
            const Material& mat = world.materials[elem_static.material_idx];

            // Start with external flow
            val_t total_flux = mat.external_flow;

            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }

            // Update element state (write to swap buffer at same position)
            ElementDynamic& elem_write = world.elements_dynamic_swap[dyn_i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (MPI global reduction)
bool validateResults(const World& world) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    const int owned_offset = world.has_ghost_up ? 1 : 0;
    for (int r = 0; r < world.local_nx; ++r) {
        for (int y = 0; y < world.ny; ++y) {
            int dyn_i = (owned_offset + r) * world.ny + y;
            const ElementDynamic& elem = world.elements_dynamic[dyn_i];
            local_energy_sum += elem.current_energy;
            local_flux_sum += elem.total_flux;
            local_energy_max = std::max(elem.current_energy, local_energy_max);
            local_energy_min = std::min(elem.current_energy, local_energy_min);
        }
    }

    val_t energy_sum, flux_sum, energy_max, energy_min;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    bool valid = true;
    if (world.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        constexpr val_t energy_epsilon = 1e-8;

        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = false;
        }

        if (std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }

        if (!std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = false;
        }

        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = false;
        }

        if (valid) {
            printf("  Validation: PASSED\n");
        }
    }

    MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    return valid;
}

// Compute a simple hash of the results for verification (MPI global)
uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    const int owned_offset = world.has_ghost_up ? 1 : 0;
    for (int r = 0; r < world.local_nx; ++r) {
        int global_r = world.global_x_start + r;
        for (int y = 0; y < world.ny; ++y) {
            int dyn_i = (owned_offset + r) * world.ny + y;
            // Global index in the original flat array
            size_t global_i = (size_t)global_r * world.ny + y;
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[dyn_i].current_energy);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[dyn_i].total_flux);
            local_hash ^= (*e_ptr + global_i) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (*f_ptr + global_i) * 0xbf58476d1ce4e5b9ULL;
        }
    }

    // Reduce: XOR all partial hashes together on rank 0
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    return global_hash;
}

// Gather all energy values to rank 0 for results printing
void gatherEnergyToRank0(const World& world, std::vector<double>& global_energy) {
    const int owned_offset = world.has_ghost_up ? 1 : 0;
    const int ny = world.ny;
    const int local_owned_count = world.local_nx * ny;

    // Pack local owned energy into contiguous buffer (row-major, global order)
    std::vector<double> local_buf(local_owned_count);
    for (int r = 0; r < world.local_nx; ++r) {
        for (int y = 0; y < ny; ++y) {
            int dyn_i = (owned_offset + r) * ny + y;
            local_buf[r * ny + y] = world.elements_dynamic[dyn_i].current_energy;
        }
    }

    if (world.nprocs == 1) {
        global_energy = std::move(local_buf);
        return;
    }

    // Compute displacements in terms of doubles
    std::vector<int> recv_counts(world.nprocs);
    std::vector<int> displs(world.nprocs);
    // Each rank's local_nx may differ; we need to gather these
    std::vector<int> all_local_nx(world.nprocs);
    MPI_Allgather(&world.local_nx, 1, MPI_INT, all_local_nx.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total = 0;
    for (int p = 0; p < world.nprocs; ++p) {
        recv_counts[p] = all_local_nx[p] * ny;
        displs[p] = total;
        total += recv_counts[p];
    }

    global_energy.resize(total);
    MPI_Gatherv(local_buf.data(), local_owned_count, MPI_DOUBLE,
                global_energy.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse identically)
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI)\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("MPI ranks: %d\n", nprocs);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Build the unstructured mesh (distributed)
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root, rank, nprocs);

    // Calculate memory usage (local)
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;

    if (rank == 0) {
        size_t global_total_mem;
        MPI_Reduce(&total_mem, &global_total_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        printf("Memory usage: %.2f MB total across all ranks\n", global_total_mem / (1024.0 * 1024.0));
    } else {
        MPI_Reduce(&total_mem, nullptr, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) printf("\n");

    // Barrier to synchronize start
    MPI_Barrier(MPI_COMM_WORLD);

    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    double start_time = MPI_Wtime();

    runSimulation(world, n_iters);

    double end_time = MPI_Wtime();
    double duration_ms_local = (end_time - start_time) * 1000.0;

    // Get max time across all ranks for performance reporting
    double max_time_ms;
    MPI_Reduce(&duration_ms_local, &max_time_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)max_time_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = max_time_ms / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (max_time_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute hash for verification
    const uint64_t hash = computeHash(world);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> energyData;
        gatherEnergyToRank0(world, energyData);
        if (rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }

    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
