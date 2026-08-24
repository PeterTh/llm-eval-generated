#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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
    idx_t connected_idx[MAX_CONNECTIONS];     // Local indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state with MPI domain decomposition metadata
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    int n_elems_root;
    int local_x_start;       // First owned x in local coordinates
    int local_x_end;         // Last owned x in local coordinates (exclusive)
    int local_x_total;       // Total x-slices including halo
    int n_local_owned;       // Number of owned elements
    int global_x_start;      // First owned global x
    int halo_x_start;        // Global x of first local x-slice (including halo)
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build local portion of the 2D square grid with halo layers
// Domain is decomposed along the x-axis across MPI ranks
void buildLocalMesh(World& world, const int n_elems_root, const int rank, const int size) {
    // Decompose grid along x-axis; last rank gets remainder
    const int x_per_rank = n_elems_root / size;
    const int global_x_start = rank * x_per_rank;
    const int global_x_end = (rank == size - 1) ? n_elems_root : (rank + 1) * x_per_rank;

    // Halo extent: one x-slice on each side (clamped to grid bounds)
    const int halo_x_start = std::max(0, global_x_start - 1);
    const int halo_x_end   = std::min(n_elems_root, global_x_end + 1);

    world.n_elems_root  = n_elems_root;
    world.global_x_start = global_x_start;
    world.halo_x_start   = halo_x_start;
    world.local_x_start  = global_x_start - halo_x_start;
    world.local_x_end    = global_x_end   - halo_x_start;
    world.local_x_total  = halo_x_end     - halo_x_start;
    world.n_local_owned  = (global_x_end - global_x_start) * n_elems_root;

    const int local_n_elems = world.local_x_total * n_elems_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow

    // Allocate and zero-initialize
    world.elements_static.resize(local_n_elems);
    world.elements_dynamic.resize(local_n_elems);
    world.elements_dynamic_swap.resize(local_n_elems);

    for (int i = 0; i < local_n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Set corner materials (inflow/outflow)
    const int last = n_elems_root - 1;
    if (halo_x_start <= 0 && 0 < halo_x_end) {
        world.elements_static[0 * n_elems_root + 0].material_idx       = INFLOW_MAT_ID;
        world.elements_static[0 * n_elems_root + last].material_idx    = OUTFLOW_MAT_ID;
    }
    if (halo_x_start <= last && last < halo_x_end) {
        const int lx = last - halo_x_start;
        world.elements_static[lx * n_elems_root + 0].material_idx      = OUTFLOW_MAT_ID;
        world.elements_static[lx * n_elems_root + last].material_idx   = INFLOW_MAT_ID;
    }

    // Build connectivity using local indices; neighbors outside halo are excluded
    for (int lx = 0; lx < world.local_x_total; ++lx) {
        const int gx = lx + halo_x_start;
        for (int y = 0; y < n_elems_root; ++y) {
            const int local_idx = lx * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; ++n) {
                const int ngx = gx + offsets[n][0];
                const int ngy = y  + offsets[n][1];
                if (ngx >= halo_x_start && ngx < halo_x_end &&
                    ngy >= 0           && ngy < n_elems_root) {
                    const int nlocal = (ngx - halo_x_start) * n_elems_root + ngy;
                    elem.connected_idx[elem.num_connections] = nlocal;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                         val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation with MPI halo exchange for boundary data
void runSimulation(World& world, const int n_iters, const int rank, const int size) {
    const int n = world.n_elems_root;

    // Temporary buffers for halo exchange (only current_energy)
    std::vector<val_t> send_buf(n);
    std::vector<val_t> recv_buf(n);

    for (int iter = 0; iter < n_iters; ++iter) {
        // ---- Halo exchange: left boundary (with rank-1) ----
        if (rank > 0) {
            const int send_lx = world.local_x_start;
            for (int y = 0; y < n; ++y)
                send_buf[y] = world.elements_dynamic[send_lx * n + y].current_energy;

            MPI_Sendrecv(send_buf.data(), n, MPI_DOUBLE, rank - 1, 0,
                         recv_buf.data(), n, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int y = 0; y < n; ++y)
                world.elements_dynamic[y].current_energy = recv_buf[y];
        }

        // ---- Halo exchange: right boundary (with rank+1) ----
        if (rank < size - 1) {
            const int send_lx = world.local_x_end - 1;
            for (int y = 0; y < n; ++y)
                send_buf[y] = world.elements_dynamic[send_lx * n + y].current_energy;

            const int recv_lx = world.local_x_end;
            MPI_Sendrecv(send_buf.data(), n, MPI_DOUBLE, rank + 1, 0,
                         recv_buf.data(), n, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int y = 0; y < n; ++y)
                world.elements_dynamic[recv_lx * n + y].current_energy = recv_buf[y];
        }

        // ---- Compute flux for owned elements ----
        for (int lx = world.local_x_start; lx < world.local_x_end; ++lx) {
            for (int y = 0; y < n; ++y) {
                const int local_idx = lx * n + y;
                const ElementStatic&  elem_static = world.elements_static[local_idx];
                const ElementDynamic& elem_dyn    = world.elements_dynamic[local_idx];
                const Material&       mat         = world.materials[elem_static.material_idx];

                val_t total_flux = mat.external_flow;

                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const idx_t neighbor_idx = elem_static.connected_idx[j];
                    const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                    total_flux += computeFlux(mat, elem_dyn,
                                              elem_static.connected_flux[j], neighbor_dyn);
                }

                ElementDynamic& elem_write = world.elements_dynamic_swap[local_idx];
                elem_write.current_energy = elem_dyn.current_energy + total_flux;
                elem_write.total_flux     = elem_dyn.total_flux     + std::abs(total_flux);
            }
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results using MPI reductions
bool validateResults(const World& world, const int rank, const int /* size */) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum   = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    int   local_valid      = 1;

    const int n = world.n_elems_root;
    for (int lx = world.local_x_start; lx < world.local_x_end; ++lx) {
        for (int y = 0; y < n; ++y) {
            const int local_idx = lx * n + y;
            const auto& elem = world.elements_dynamic[local_idx];
            local_energy_sum += elem.current_energy;
            local_flux_sum   += elem.total_flux;
            local_energy_max  = std::max(elem.current_energy, local_energy_max);
            local_energy_min  = std::min(elem.current_energy, local_energy_min);
        }
    }

    val_t energy_sum, flux_sum, energy_max, energy_min;
    MPI_Allreduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_flux_sum,   &flux_sum,   1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        constexpr val_t energy_epsilon = 1e-8;

        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            local_valid = 0;
        }
        if (std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            local_valid = 0;
        }
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            local_valid = 0;
        }

        printf("  Validation: %s\n", local_valid ? "PASSED" : "FAILED");
    }

    int global_valid;
    MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    return global_valid != 0;
}

// Compute hash of results using global indices, then XOR-reduce across ranks
uint64_t computeHash(const World& world, const int /* rank */, const int /* size */) {
    uint64_t local_hash = 0;
    const int n = world.n_elems_root;

    for (int lx = world.local_x_start; lx < world.local_x_end; ++lx) {
        const int gx = lx + world.halo_x_start;
        for (int y = 0; y < n; ++y) {
            const int local_idx  = lx * n + y;
            const int global_idx = gx * n + y;

            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[local_idx].current_energy);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[local_idx].total_flux);
            local_hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
        }
    }

    uint64_t global_hash;
    MPI_Allreduce(&local_hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BOR, MPI_COMM_WORLD);
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
        printf("MPI ranks: %d\n", size);
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Build local portion of the unstructured mesh
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildLocalMesh(world, n_elems_root, rank, size);

    // Calculate memory usage (local)
    if (rank == 0) {
        const size_t static_mem  = world.elements_static.size()  * sizeof(ElementStatic);
        const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
        const size_t total_mem   = static_mem + dynamic_mem;
        printf("Memory usage (local): %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem   / (1024.0 * 1024.0),
               static_mem  / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, rank, size);

    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    const long long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long duration_ms;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute hash for verification (reduced across all ranks)
    const uint64_t hash = computeHash(world, rank, size);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation (gather to rank 0)
    if (printResults) {
        const int n = world.n_elems_root;

        // Extract local energy data in global order
        std::vector<val_t> local_energy(world.n_local_owned);
        for (int lx = world.local_x_start; lx < world.local_x_end; ++lx) {
            for (int y = 0; y < n; ++y) {
                const int local_idx = lx * n + y;
                const int owned_idx = (lx - world.local_x_start) * n + y;
                local_energy[owned_idx] = world.elements_dynamic[local_idx].current_energy;
            }
        }

        // Gather to rank 0 using Gatherv (ranks may have different counts)
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        std::vector<val_t> global_energy;

        if (rank == 0) {
            global_energy.resize(n_elems);
            const int x_per_rank = n_elems_root / size;
            for (int r = 0; r < size; ++r) {
                const int gs = r * x_per_rank;
                const int ge = (r == size - 1) ? n_elems_root : (r + 1) * x_per_rank;
                recvcounts[r] = (ge - gs) * n;
                displs[r]     = gs * n;
            }
        }

        MPI_Gatherv(local_energy.data(), world.n_local_owned, MPI_DOUBLE,
                    global_energy.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(global_energy, "ElementEnergy");
        }
    }

    // Validation
    if (validate) {
        bool valid = validateResults(world, rank, size);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
