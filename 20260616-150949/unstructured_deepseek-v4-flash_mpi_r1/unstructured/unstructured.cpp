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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices into extended dynamic array
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state with MPI decomposition
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;       // local elements only (flat, row-major)
    std::vector<ElementDynamic> elements_dynamic;     // extended: (n_my_rows+2) * n_elems_root
    std::vector<ElementDynamic> elements_dynamic_swap; // same layout
    
    // Domain decomposition info
    int n_elems_root;  // grid dimension (both axes)
    int n_my_rows;     // number of rows owned by this rank
    int start_row;     // first global row index owned by this rank
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Decompose grid rows among MPI ranks
static void computeRowDecomposition(int n_elems_root, int n_procs, int my_rank,
                                     int& start_row, int& n_my_rows) {
    const int base = n_elems_root / n_procs;
    const int rem = n_elems_root % n_procs;
    n_my_rows = base + (my_rank < rem ? 1 : 0);
    start_row = my_rank * base + std::min(my_rank, rem);
}

// Convert global grid position (nx, ny) to extended dynamic array index.
// Extended array layout: [top_halo_row(0..n-1) | local_rows(0..n_my_rows-1) | bottom_halo_row(0..n-1)]
// Local row r (0-indexed) maps to extended row (r+1).
inline int globalToExtended(int nx, int ny, int n_elems_root,
                            int start_row, int n_my_rows) {
    if (nx < 0 || nx >= n_elems_root || ny < 0 || ny >= n_elems_root) return -1;
    if (nx >= start_row && nx < start_row + n_my_rows) {
        // Local element
        return (nx - start_row + 1) * n_elems_root + ny;
    } else {
        // Halo element: row 0 (top) or (n_my_rows+1) (bottom)
        const int halo_row = (nx < start_row) ? 0 : (n_my_rows + 1);
        return halo_row * n_elems_root + ny;
    }
}

// Build the local portion of the unstructured mesh for this MPI rank
void buildSquare2DMPI(World& world, int n_elems_root, int my_rank, int n_procs) {
    computeRowDecomposition(n_elems_root, n_procs, my_rank,
                            world.start_row, world.n_my_rows);
    world.n_elems_root = n_elems_root;
    
    const int n_my_rows = world.n_my_rows;
    const int start_row = world.start_row;
    const int n_my_elems = n_my_rows * n_elems_root;
    const int extended_size = (n_my_rows + 2) * n_elems_root;
    
    // Initialize materials (replicated on all ranks)
    world.materials = {
        {0.8, 0.0},   // DEFAULT: neutral
        {0.8, 0.5},   // INFLOW: positive source
        {0.8, -0.5}   // OUTFLOW: negative sink
    };
    
    // Allocate elements (only local portion for static, extended for dynamic)
    world.elements_static.resize(n_my_elems);
    world.elements_dynamic.resize(extended_size);
    world.elements_dynamic_swap.resize(extended_size);
    
    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_my_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    for (auto& e : world.elements_dynamic) {
        e.current_energy = 0.0;
        e.total_flux = 0.0;
    }
    for (auto& e : world.elements_dynamic_swap) {
        e.current_energy = 0.0;
        e.total_flux = 0.0;
    }
    
    // Build connectivity for local elements using extended array indices
    for (int r = 0; r < n_my_rows; ++r) {
        const int x = start_row + r;  // global x coordinate
        for (int y = 0; y < n_elems_root; ++y) {
            const int local_idx = r * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                const int ext_idx = globalToExtended(nx, ny, n_elems_root,
                                                     start_row, n_my_rows);
                if (ext_idx >= 0) {
                    elem.connected_idx[elem.num_connections] = static_cast<idx_t>(ext_idx);
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow (if this rank owns them)
    const int last = n_elems_root - 1;
    const struct { int x, y; idx_t mat; } corners[4] = {
        {0, 0, INFLOW_MAT_ID},
        {0, last, OUTFLOW_MAT_ID},
        {last, 0, OUTFLOW_MAT_ID},
        {last, last, INFLOW_MAT_ID}
    };
    for (const auto& c : corners) {
        if (c.x >= start_row && c.x < start_row + n_my_rows) {
            const int local_idx = (c.x - start_row) * n_elems_root + c.y;
            world.elements_static[local_idx].material_idx = c.mat;
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                         val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Exchange halo rows of current_energy with neighboring ranks
static void exchangeHalos(std::vector<ElementDynamic>& dyn, int n_my_rows,
                           int n_elems_root, int my_rank, int n_procs) {
    // Pack/unpack buffers for current_energy values of one row
    std::vector<val_t> send_buf(n_elems_root);
    std::vector<val_t> recv_buf(n_elems_root);
    
    // Exchange with upper neighbor (top halo)
    if (my_rank > 0) {
        for (int i = 0; i < n_elems_root; ++i) {
            send_buf[i] = dyn[1 * n_elems_root + i].current_energy;
        }
        MPI_Sendrecv(send_buf.data(), n_elems_root, MPI_DOUBLE, my_rank - 1, 0,
                     recv_buf.data(), n_elems_root, MPI_DOUBLE, my_rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < n_elems_root; ++i) {
            dyn[0 * n_elems_root + i].current_energy = recv_buf[i];
        }
    }
    
    // Exchange with lower neighbor (bottom halo)
    if (my_rank < n_procs - 1) {
        for (int i = 0; i < n_elems_root; ++i) {
            send_buf[i] = dyn[n_my_rows * n_elems_root + i].current_energy;
        }
        MPI_Sendrecv(send_buf.data(), n_elems_root, MPI_DOUBLE, my_rank + 1, 0,
                     recv_buf.data(), n_elems_root, MPI_DOUBLE, my_rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < n_elems_root; ++i) {
            dyn[(n_my_rows + 1) * n_elems_root + i].current_energy = recv_buf[i];
        }
    }
}

// Run simulation for n_iters iterations with MPI halo exchange
void runSimulationMPI(World& world, int n_iters, int my_rank, int n_procs) {
    const int n_my_rows = world.n_my_rows;
    const int n_elems_root = world.n_elems_root;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange halo current_energy data with neighboring ranks
        exchangeHalos(world.elements_dynamic, n_my_rows, n_elems_root,
                      my_rank, n_procs);
        
        // Update all local elements
        for (int r = 0; r < n_my_rows; ++r) {
            for (int c = 0; c < n_elems_root; ++c) {
                const int local_idx = r * n_elems_root + c;
                // Extended index: local row r maps to extended row (r+1)
                // (row 0 = top halo, row n_my_rows+1 = bottom halo)
                const int ext_idx = (r + 1) * n_elems_root + c;
                
                const ElementStatic& elem_static = world.elements_static[local_idx];
                const ElementDynamic& elem_dyn = world.elements_dynamic[ext_idx];
                const Material& mat = world.materials[elem_static.material_idx];
                
                // Start with external flow
                val_t total_flux = mat.external_flow;
                
                // Add flux from all connected elements
                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const ElementDynamic& neighbor_dyn =
                        world.elements_dynamic[elem_static.connected_idx[j]];
                    total_flux += computeFlux(mat, elem_dyn,
                                              elem_static.connected_flux[j],
                                              neighbor_dyn);
                }
                
                // Update element state (write to swap buffer)
                ElementDynamic& elem_write = world.elements_dynamic_swap[ext_idx];
                elem_write.current_energy = elem_dyn.current_energy + total_flux;
                elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }
        }
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (rank 0 only, with full gathered data)
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : world.elements_dynamic) {
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
    
    int my_rank, n_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    
    // Parse arguments on rank 0, then broadcast
    int params[5] = {512, 10, 0, 0, 0}; // {n_elems_root, n_iters, do_validate, print_results, status}
    // status: 0=ok, 1=help, 2=error
    
    if (my_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                params[0] = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                params[1] = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                params[2] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                params[3] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                params[4] = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                params[4] = 2;
            }
        }
    }
    
    MPI_Bcast(params, 5, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (params[4]) {
        MPI_Finalize();
        return (params[4] == 1) ? 0 : 1;
    }
    
    const int n_elems_root = params[0];
    const int n_iters = params[1];
    const bool do_validate = params[2] != 0;
    const bool do_print_results = params[3] != 0;
    const int n_elems = n_elems_root * n_elems_root;
    
    // Rank 0 prints header info
    if (my_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", n_procs);
        printf("Validation: %s\n", do_validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the local portion of the unstructured mesh
    World world;
    buildSquare2DMPI(world, n_elems_root, my_rank, n_procs);
    
    // Calculate memory usage (report per-rank and total on rank 0)
    {
        const long long local_static = static_cast<long long>(world.n_my_rows * n_elems_root) *
                                        static_cast<long long>(sizeof(ElementStatic));
        const long long local_dynamic = static_cast<long long>((world.n_my_rows + 2) * n_elems_root) *
                                         static_cast<long long>(sizeof(ElementDynamic)) * 2;
        const long long local_total = local_static + local_dynamic;
        
        long long total_mem = 0;
        MPI_Reduce(&local_total, &total_mem, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (my_rank == 0) {
            printf("Memory usage (per rank, total across all ranks):\n");
            printf("  Static: %.2f MB per rank\n", local_static / (1024.0 * 1024.0));
            printf("  Dynamic: %.2f MB per rank\n", local_dynamic / (1024.0 * 1024.0));
            printf("  Total across %d ranks: %.2f MB\n", n_procs, total_mem / (1024.0 * 1024.0));
            printf("\n");
        }
    }
    
    // Run simulation (all ranks participate)
    if (my_rank == 0) {
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    const double start_time = MPI_Wtime();
    runSimulationMPI(world, n_iters, my_rank, n_procs);
    const double end_time = MPI_Wtime();
    
    const double duration_ms = (end_time - start_time) * 1000.0;
    
    // Gather results to rank 0 for output
    const int n_my_rows = world.n_my_rows;
    const int n_my_elems = n_my_rows * n_elems_root;
    
    // Extract local portion from extended array into contiguous buffer
    std::vector<ElementDynamic> gather_send(n_my_elems);
    for (int r = 0; r < n_my_rows; ++r) {
        std::memcpy(&gather_send[r * n_elems_root],
                    &world.elements_dynamic[(r + 1) * n_elems_root],
                    n_elems_root * sizeof(ElementDynamic));
    }
    
    // Prepare gather counts/displacements on rank 0
    std::vector<ElementDynamic> all_dynamic;
    std::vector<int> recvcounts, displs;
    if (my_rank == 0) {
        all_dynamic.resize(n_elems);
        recvcounts.resize(n_procs);
        displs.resize(n_procs);
        for (int r = 0; r < n_procs; ++r) {
            int rs, rn;
            computeRowDecomposition(n_elems_root, n_procs, r, rs, rn);
            recvcounts[r] = rn * n_elems_root;
            displs[r] = rs * n_elems_root;
        }
    }
    
    // Gather using MPI_BYTE (ElementDynamic is POD)
    {
        std::vector<int> recv_bytes, disp_bytes;
        if (my_rank == 0) {
            recv_bytes.resize(n_procs);
            disp_bytes.resize(n_procs);
            for (int r = 0; r < n_procs; ++r) {
                recv_bytes[r] = recvcounts[r] * static_cast<int>(sizeof(ElementDynamic));
                disp_bytes[r] = displs[r] * static_cast<int>(sizeof(ElementDynamic));
            }
        }
        
        const int send_bytes = n_my_elems * static_cast<int>(sizeof(ElementDynamic));
        MPI_Gatherv(gather_send.data(), send_bytes, MPI_BYTE,
                    all_dynamic.data(), recv_bytes.data(), disp_bytes.data(), MPI_BYTE,
                    0, MPI_COMM_WORLD);
    }
    
    // Rank 0: output results
    if (my_rank == 0) {
        printf("Computation time: %.0f ms\n", duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        
        // Compute hash for verification
        const uint64_t hash = computeHash(all_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
        
        // Print results for external validation
        if (do_print_results) {
            std::vector<double> energyData;
            energyData.reserve(all_dynamic.size());
            for (const auto& elem : all_dynamic) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }
        
        // Validation
        if (do_validate) {
            // Create a temporary World with the full data for validation
            World full_world;
            full_world.elements_dynamic = std::move(all_dynamic);
            bool valid = validateResults(full_world);
            if (!valid) {
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
