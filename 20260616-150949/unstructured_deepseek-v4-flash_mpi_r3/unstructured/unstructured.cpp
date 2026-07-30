#include <algorithm>
#include <cassert>
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

// World state (includes MPI decomposition metadata)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;   // Replicated on all ranks
    
    // Dynamic arrays shared across all ranks — each rank stores its own local
    // portion plus ghost/halo rows for neighbor data at domain boundaries.
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    
    // MPI metadata
    int mpi_rank;
    int mpi_size;
    int n_elems_root;          // Full grid dimension
    int first_row;             // First global row owned by this rank
    int last_row;              // Last global row owned by this rank (inclusive)
    int n_local_rows;          // Number of rows owned locally
    int n_local_elems;         // Number of elements owned locally
    int n_storage_rows;        // Total storage rows (local + ghosts)
    int n_storage_elems;       // Total storage elements
    
    // Per-rank info for gather/scatter
    std::vector<int> all_n_local_rows;
    std::vector<int> all_first_rows;
    std::vector<int> all_n_local_elems;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// ---------------------------------------------------------------------------
// MPI domain decomposition helpers
// ---------------------------------------------------------------------------

// Convert a global (x, y) grid coordinate to a linear storage index for the
// dynamic arrays (which include ghost rows).  Storage layout:
//   rank 0:  local rows [0 .. n_local_rows-1]; ghost below at n_local_rows
//   other ranks: ghost above at 0; local rows [1 .. n_local_rows]; ghost below at n_local_rows+1
static inline idx_t globalToStorage(const int x, const int y, const World& w) noexcept {
    const int ghost_above = (w.mpi_rank > 0) ? 1 : 0;
    return (static_cast<idx_t>(x) - static_cast<idx_t>(w.first_row) + ghost_above)
           * static_cast<idx_t>(w.n_elems_root) + static_cast<idx_t>(y);
}

// ---------------------------------------------------------------------------
// Build the full unstructured mesh (replicated on every rank)
// ---------------------------------------------------------------------------

void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    
    // Store grid dimension for later use
    world.n_elems_root = n_elems_root;
    
    // Initialize materials (replicated)
    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Build static connectivity (replicated)
    world.elements_static.resize(n_elems);
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    
    // Build connectivity: each element connects to its 2D grid neighbors
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    elem.connected_idx[elem.num_connections] =
                        static_cast<idx_t>(nx) * n_elems_root + static_cast<idx_t>(ny);
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow
    const int last = n_elems_root - 1;
    world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
    
    // --- 1D row-based domain decomposition ---
    
    // Compute number of rows per rank (ceil division)
    const int rows_per_rank = (n_elems_root + world.mpi_size - 1) / world.mpi_size;
    world.first_row = world.mpi_rank * rows_per_rank;
    world.last_row  = std::min(world.first_row + rows_per_rank - 1, n_elems_root - 1);
    world.n_local_rows = world.last_row - world.first_row + 1;
    
    // Count ghost rows (present except at mesh boundaries)
    world.n_storage_rows = world.n_local_rows
                           + (world.mpi_rank > 0              ? 1 : 0)   // ghost above
                           + (world.mpi_rank < world.mpi_size - 1 ? 1 : 0);  // ghost below
    
    world.n_local_elems  = world.n_local_rows * n_elems_root;
    world.n_storage_elems = world.n_storage_rows * n_elems_root;
    
    // Allocate distributed dynamic arrays (local + ghosts)
    world.elements_dynamic.resize(world.n_storage_elems);
    world.elements_dynamic_swap.resize(world.n_storage_elems);
    
    // Initialize owned elements to zero energy
    std::memset(world.elements_dynamic.data(), 0,
                world.n_storage_elems * sizeof(ElementDynamic));
    std::memset(world.elements_dynamic_swap.data(), 0,
                world.n_storage_elems * sizeof(ElementDynamic));
    
    // Build per-rank metadata arrays (for collectives)
    world.all_n_local_rows.resize(world.mpi_size);
    world.all_first_rows.resize(world.mpi_size);
    world.all_n_local_elems.resize(world.mpi_size);
    MPI_Allgather(&world.n_local_rows,   1, MPI_INT,
                  world.all_n_local_rows.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&world.first_row,      1, MPI_INT,
                  world.all_first_rows.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&world.n_local_elems,  1, MPI_INT,
                  world.all_n_local_elems.data(), 1, MPI_INT, MPI_COMM_WORLD);
}

// ---------------------------------------------------------------------------
// Ghost/halo exchange using non-blocking point-to-point messages
// ---------------------------------------------------------------------------

static void exchangeGhosts(World& world) {
    const int n         = world.n_elems_root;
    const int rank      = world.mpi_rank;
    const int size      = world.mpi_size;
    const int n_local   = world.n_local_rows;
    const int ghost_above = (rank > 0) ? 1 : 0;
    
    // Storage row indices:
    //   local rows begin at storage row  ghost_above
    //   last local storage row = ghost_above + n_local - 1
    //   ghost below = ghost_above + n_local
    
    MPI_Request reqs[4];
    int nreq = 0;
    
    // Exchange with neighbour below (rank+1)
    if (rank < size - 1) {
        const int send_row  = ghost_above + n_local - 1;   // last local row
        const int recv_row  = ghost_above + n_local;        // ghost-below row
        MPI_Isend(world.elements_dynamic.data() + send_row * n, n * sizeof(ElementDynamic),
                  MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(world.elements_dynamic.data() + recv_row * n, n * sizeof(ElementDynamic),
                  MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    
    // Exchange with neighbour above (rank-1)
    if (rank > 0) {
        const int send_row  = ghost_above;                   // first local row
        const int recv_row  = 0;                              // ghost-above row
        MPI_Isend(world.elements_dynamic.data() + send_row * n, n * sizeof(ElementDynamic),
                  MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(world.elements_dynamic.data() + recv_row * n, n * sizeof(ElementDynamic),
                  MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    
    if (nreq > 0)
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
}

// ---------------------------------------------------------------------------
// Flux computation
// ---------------------------------------------------------------------------

static inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                                val_t connection_flux, const ElementDynamic& other_elem) noexcept {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// ---------------------------------------------------------------------------
// Simulation kernel – with MPI ghost exchange before every iteration
// ---------------------------------------------------------------------------

void runSimulation(World& world, const int n_iters) {
    const int n            = world.n_elems_root;
    const int n_local_rows = world.n_local_rows;
    const int first_row    = world.first_row;
    const int ghost_above  = (world.mpi_rank > 0) ? 1 : 0;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Ensure ghost data is up-to-date for boundary elements
        exchangeGhosts(world);
        
        // Update all locally-owned elements
        for (int r = 0; r < n_local_rows; ++r) {
            const int global_x = first_row + r;
            const int storage_base = (ghost_above + r) * n;
            
            for (int y = 0; y < n; ++y) {
                const int storage_idx = storage_base + y;
                const int global_idx  = global_x * n + y;
                
                const ElementStatic& elem_static = world.elements_static[global_idx];
                const ElementDynamic& elem_dyn   = world.elements_dynamic[storage_idx];
                const Material& mat              = world.materials[elem_static.material_idx];
                
                // Start with external flow
                val_t total_flux = mat.external_flow;
                
                // Add flux from all connected elements
                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const idx_t neighbor_global = elem_static.connected_idx[j];
                    const int nx = static_cast<int>(neighbor_global / n);
                    const int ny = static_cast<int>(neighbor_global % n);
                    const idx_t neighbor_storage = globalToStorage(nx, ny, world);
                    const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_storage];
                    
                    total_flux += computeFlux(mat, elem_dyn,
                                              elem_static.connected_flux[j], neighbor_dyn);
                }
                
                // Write updated values into swap buffer
                ElementDynamic& elem_write = world.elements_dynamic_swap[storage_idx];
                elem_write.current_energy = elem_dyn.current_energy + total_flux;
                elem_write.total_flux     = elem_dyn.total_flux + std::abs(total_flux);
            }
        }
        
        // Swap buffers for next iteration
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// ---------------------------------------------------------------------------
// Parallel validation – reduces across all ranks
// ---------------------------------------------------------------------------

bool validateResults(const World& world) {
    const int ghost_above   = (world.mpi_rank > 0) ? 1 : 0;
    const int n = world.n_elems_root;
    
    // Compute local statistics from owned elements (skip ghost rows)
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum   = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    
    for (int r = 0; r < world.n_local_rows; ++r) {
        const int base = (ghost_above + r) * n;
        for (int y = 0; y < n; ++y) {
            const auto& elem = world.elements_dynamic[base + y];
            local_energy_sum += elem.current_energy;
            local_flux_sum   += elem.total_flux;
            local_energy_max  = std::max(elem.current_energy, local_energy_max);
            local_energy_min  = std::min(elem.current_energy, local_energy_min);
        }
    }
    
    // Global reduction
    val_t global_energy_sum = 0.0;
    val_t global_flux_sum   = 0.0;
    val_t global_energy_max = 0.0;
    val_t global_energy_min = 0.0;
    
    MPI_Reduce(&local_energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum,   &global_flux_sum,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    bool all_valid = true;
    if (world.mpi_rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_energy_sum);
        printf("  Flux sum: %.2f\n", global_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);
        
        constexpr val_t energy_epsilon = 1e-8;
        if (!std::isfinite(global_energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            all_valid = false;
        }
        if (std::abs(global_energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(global_flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            all_valid = false;
        }
        if (!std::isfinite(global_energy_max) || !std::isfinite(global_energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            all_valid = false;
        }
        if (all_valid) {
            printf("  Validation: PASSED\n");
        }
    }
    // Broadcast result so all ranks agree on return value
    MPI_Bcast(&all_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    return all_valid;
}

// ---------------------------------------------------------------------------
// Parallel hash – each rank hashes its owned elements (with global indices)
// then all contributions are XOR-reduced on rank 0.
// ---------------------------------------------------------------------------

uint64_t computeHash(const std::vector<ElementDynamic>& elements,
                     const World& world) {
    const int n          = world.n_elems_root;
    const int ghost_above = (world.mpi_rank > 0) ? 1 : 0;
    
    uint64_t local_hash = 0;
    for (int r = 0; r < world.n_local_rows; ++r) {
        const int global_x  = world.first_row + r;
        const int base      = (ghost_above + r) * n;
        for (int y = 0; y < n; ++y) {
            const int storage_idx = base + y;
            const uint64_t i      = static_cast<uint64_t>(global_x) * n + y;
            
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(
                &elements[storage_idx].current_energy);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(
                &elements[storage_idx].total_flux);
            
            local_hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    return global_hash;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    World world;
    MPI_Comm_rank(MPI_COMM_WORLD, &world.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world.mpi_size);
    
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
            if (world.mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world.mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (world.mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", world.mpi_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh and set up decomposition
    if (world.mpi_rank == 0) printf("Building unstructured mesh...\n");
    buildSquare2D(world, n_elems_root);
    
    if (world.mpi_rank == 0) {
        const size_t static_mem  = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem   = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation (timed on all ranks; rank 0 reports wall time)
    if (world.mpi_rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();
    
    runSimulation(world, n_iters);
    
    const double t_end = MPI_Wtime();
    const double duration_ms = (t_end - t_start) * 1.0e3;
    
    // Report timing and performance (only rank 0)
    if (world.mpi_rank == 0) {
        printf("Computation time: %.0f ms\n", duration_ms);
        
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification (global across all ranks)
    const uint64_t hash = computeHash(world.elements_dynamic, world);
    if (world.mpi_rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation (gather to rank 0)
    if (printResults && world.mpi_rank == 0) {
        std::vector<double> all_energy(n_elems, 0.0);
        std::vector<double> local_energy(world.n_local_elems);
        
        const int ghost_above = (world.mpi_rank > 0) ? 1 : 0;
        const int n = world.n_elems_root;
        for (int r = 0; r < world.n_local_rows; ++r) {
            const int base = (ghost_above + r) * n;
            for (int y = 0; y < n; ++y) {
                local_energy[r * n + y] = world.elements_dynamic[base + y].current_energy;
            }
        }
        
        // Gather all local energy arrays onto rank 0 in global index order
        std::vector<int> displs(world.mpi_size, 0);
        for (int i = 1; i < world.mpi_size; ++i) {
            displs[i] = displs[i-1] + world.all_n_local_elems[i-1];
        }
        MPI_Gatherv(local_energy.data(), world.n_local_elems, MPI_DOUBLE,
                    all_energy.data(), world.all_n_local_elems.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // The gathered data is in rank order (first rank 0's rows, then rank 1's, etc.)
        // We need to reorder to match the original element ordering which is
        // row-major (row 0 first, then row 1, ...). Since we use 1D row
        // decomposition in rank order, this is already in the correct order.
        print_results(all_energy, "ElementEnergy");
    } else if (printResults) {
        std::vector<double> local_energy(world.n_local_elems);
        const int ghost_above = (world.mpi_rank > 0) ? 1 : 0;
        const int n = world.n_elems_root;
        for (int r = 0; r < world.n_local_rows; ++r) {
            const int base = (ghost_above + r) * n;
            for (int y = 0; y < n; ++y) {
                local_energy[r * n + y] = world.elements_dynamic[base + y].current_energy;
            }
        }
        std::vector<int> displs(world.mpi_size, 0);
        for (int i = 1; i < world.mpi_size; ++i) {
            displs[i] = displs[i-1] + world.all_n_local_elems[i-1];
        }
        MPI_Gatherv(local_energy.data(), world.n_local_elems, MPI_DOUBLE,
                    nullptr, world.all_n_local_elems.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    // Validation (collective operation)
    bool valid = true;
    if (validate) {
        valid = validateResults(world);
    }
    
    MPI_Finalize();
    return (validate && !valid) ? 1 : 0;
}
