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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state
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

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, int rank, int n_ranks, int& local_x_start, int& local_x_end) {
    // Determine local range of x
    int rows_per_rank = n_elems_root / n_ranks;
    int remainder = n_elems_root % n_ranks;
    
    local_x_start = rank * rows_per_rank + std::min(rank, remainder);
    local_x_end = local_x_start + rows_per_rank + (rank < remainder ? 1 : 0);
    
    int local_rows = local_x_end - local_x_start;
    
    // Ghost rows: one before (if not first rank), one after (if not last rank)
    bool has_top_ghost = (rank > 0);
    bool has_bottom_ghost = (rank < n_ranks - 1);
    
    int total_local_rows = local_rows + (has_top_ghost ? 1 : 0) + (has_bottom_ghost ? 1 : 0);
    const int n_elems_local = total_local_rows * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems_local);
    world.elements_dynamic.resize(n_elems_local);
    world.elements_dynamic_swap.resize(n_elems_local);
    
    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_elems_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity
    // We iterate over LOCAL rows plus ghost rows to set up connectivity correctly for local computation
    // The "logical" x coordinate for storage index i is:
    // logical_x = local_x_start - (has_top_ghost ? 1 : 0) + (i / n_elems_root)
    
    int start_store_x = local_x_start - (has_top_ghost ? 1 : 0);
    int end_store_x = local_x_end + (has_bottom_ghost ? 1 : 0);
    
    for (int x = start_store_x; x < end_store_x; ++x) {
        // Compute storage index base for this row
        int row_idx_in_storage = x - start_store_x;
        
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = row_idx_in_storage * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within global bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    // Check if neighbor is within local storage bounds (including ghosts)
                    if (nx >= start_store_x && nx < end_store_x) {
                        int neighbor_row_in_storage = nx - start_store_x;
                        const int neighbor_idx = neighbor_row_in_storage * n_elems_root + ny;
                        
                        elem.connected_idx[elem.num_connections] = neighbor_idx;
                        elem.connected_flux[elem.num_connections] = 1.0;
                        elem.num_connections++;
                    }
                }
            }
            
            // Set corner elements
            if (x == 0 && y == 0) elem.material_idx = INFLOW_MAT_ID;
            if (x == 0 && y == n_elems_root - 1) elem.material_idx = OUTFLOW_MAT_ID;
            if (x == n_elems_root - 1 && y == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (x == n_elems_root - 1 && y == n_elems_root - 1) elem.material_idx = INFLOW_MAT_ID;
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, int n_elems_root, int rank, int n_ranks, int local_x_start, int local_x_end) {
    
    // Ghost rows exchange setup
    bool has_top_ghost = (rank > 0);
    bool has_bottom_ghost = (rank < n_ranks - 1);
    
    // Calculate pointers for ghost exchange
    // Top ghost row is at index 0 (if exists)
    // Bottom ghost row is at index (local_rows + (has_top_ghost?1:0)) * n_elems_root (if exists)
    // Wait, let's be precise about storage layout:
    // If rank 0: [Own Rows][Bottom Ghost]
    // If rank mid: [Top Ghost][Own Rows][Bottom Ghost]
    // If rank last: [Top Ghost][Own Rows]
    
    int row_offset = (has_top_ghost ? 1 : 0);
    int num_own_rows = local_x_end - local_x_start;
    
    // Pointers to data to send/recv
    // We send our first own row to Top
    // We send our last own row to Bottom
    // We recv Top Ghost from Top
    // We recv Bottom Ghost from Bottom
    
    ElementDynamic* send_top_ptr = &world.elements_dynamic[row_offset * n_elems_root];
    ElementDynamic* recv_top_ptr = &world.elements_dynamic[0]; // Into top ghost slot
    
    ElementDynamic* send_bot_ptr = &world.elements_dynamic[(row_offset + num_own_rows - 1) * n_elems_root];
    ElementDynamic* recv_bot_ptr = &world.elements_dynamic[(row_offset + num_own_rows) * n_elems_root]; // Into bottom ghost slot
    
    size_t row_size_bytes = n_elems_root * sizeof(ElementDynamic);
    
    for (int iter = 0; iter < n_iters; ++iter) {
        
        // Exchange ghost rows
        MPI_Request reqs[4];
        int n_reqs = 0;
        
        // Send to Top (rank-1), Recv from Top (rank-1)
        if (has_top_ghost) {
            MPI_Isend(send_top_ptr, row_size_bytes, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Irecv(recv_top_ptr, row_size_bytes, MPI_BYTE, rank - 1, 1, MPI_COMM_WORLD, &reqs[n_reqs++]);
        }
        
        // Send to Bottom (rank+1), Recv from Bottom (rank+1)
        if (has_bottom_ghost) {
            MPI_Isend(send_bot_ptr, row_size_bytes, MPI_BYTE, rank + 1, 1, MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Irecv(recv_bot_ptr, row_size_bytes, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD, &reqs[n_reqs++]);
        }
        
        MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
        
        // Update all elements (skip ghost rows for update, but use them for flux)
        // We only update our OWN rows.
        // Storage index for own rows starts at row_offset * n_elems_root
        // and ends at (row_offset + num_own_rows) * n_elems_root
        
        size_t start_idx = row_offset * n_elems_root;
        size_t end_idx = start_idx + num_own_rows * n_elems_root;
        
        for (size_t i = start_idx; i < end_idx; ++i) {
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
        
        // Swap buffers (pointers only for dynamic state)
        // Note: vector swap swaps internal pointers, which invalidates our send/recv pointers if we stored them.
        // But we re-calculate them or they are just indices.
        // Wait, I calculated pointers outside the loop.
        // If I swap vectors, the data moves (or rather the pointers inside vector move).
        // Since I recalculate pointers inside loop or don't use them inside loop except for MPI,
        // and MPI is done before swap.
        // But wait, next iteration MPI needs valid pointers.
        // If I swap, `elements_dynamic` now points to the buffer that was `swap`.
        // The pointers `send_top_ptr` etc are into the OLD buffer.
        // So I must recalculate pointers inside loop.
        
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
        
        // Re-calculate pointers because of swap
        send_top_ptr = &world.elements_dynamic[row_offset * n_elems_root];
        recv_top_ptr = &world.elements_dynamic[0];
        send_bot_ptr = &world.elements_dynamic[(row_offset + num_own_rows - 1) * n_elems_root];
        recv_bot_ptr = &world.elements_dynamic[(row_offset + num_own_rows) * n_elems_root];
    }
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
    int rank, n_ranks;
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
        printf("MPI Ranks: %d\n", n_ranks);
        printf("\n");
    }
    
    // Build the unstructured mesh
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    int local_x_start, local_x_end;
    buildSquare2D(world, n_elems_root, rank, n_ranks, local_x_start, local_x_end);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    
    if (rank == 0) {
        // Just print rank 0 usage as an estimate/example
        printf("Memory usage (Rank 0): %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters, n_elems_root, rank, n_ranks, local_x_start, local_x_end);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long global_duration_ms = 0;
    MPI_Reduce(&duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1); // Not really skipping first iter here but formula matches original
        const double time_per_iter = static_cast<double>(global_duration_ms) / n_measured_iters; // original code divided by total iters? No, it assumed n_iters in loop.
        // Original: const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        // Wait, original code:
        // const int n_measured_iters = std::max(n_iters - 1, 1);
        // The loop ran n_iters. If n_iters=10, loop ran 10 times.
        // n_measured_iters = 9. Why 9? Maybe warmup? But there is no warmup code.
        // It's just a metric calculation choice in original code. I should preserve it.
        
        const double giga_elems_per_sec = (n_measured_iters * (double)n_elems) / (global_duration_ms / 1000.0) / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification
    // Hash needs to be aggregated from all ranks.
    // We compute local hash for own elements only.
    
    // Identify own elements range in storage
    bool has_top_ghost = (rank > 0);
    int row_offset = (has_top_ghost ? 1 : 0);
    int num_own_rows = local_x_end - local_x_start;
    
    uint64_t local_hash = 0;
    // We need to iterate over OWN elements to compute hash
    // Original computeHash iterates over ALL elements in vector.
    // If we pass a vector of own elements, we need to copy them or modify computeHash.
    // Let's modify computeHash to take range.
    // Or just manually loop here.
    
    for (int r = 0; r < num_own_rows; ++r) {
        int store_idx_start = (row_offset + r) * n_elems_root;
        for (int c = 0; c < n_elems_root; ++c) {
            int i = store_idx_start + c; // Index in local storage
            // Global index for hash calculation consistency?
            // Original hash:
            // hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
            // The 'i' was the index in the vector. Since vector was global, 'i' was global index.
            // So we need global index here for 'i'.
            
            idx_t global_idx = (local_x_start + r) * n_elems_root + c;
            
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
            local_hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("  Result hash: %016lX\n", global_hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults) {
        // Collect all data to rank 0 (not scalable but okay for verification of small runs)
        // Or write to file in parallel (complex).
        // For simplicity, let's just gather to rank 0.
        // Assuming fit in memory.
        
        std::vector<double> local_energy;
        local_energy.reserve(num_own_rows * n_elems_root);
        for (int r = 0; r < num_own_rows; ++r) {
             int store_idx_start = (row_offset + r) * n_elems_root;
             for (int c = 0; c < n_elems_root; ++c) {
                 local_energy.push_back(world.elements_dynamic[store_idx_start + c].current_energy);
             }
        }
        
        std::vector<double> global_energy;
        if (rank == 0) {
            global_energy.resize(n_elems);
        }
        
        // Use MPI_Gatherv because counts might differ
        std::vector<int> recv_counts(n_ranks);
        std::vector<int> displs(n_ranks);
        
        int local_count = local_energy.size();
        MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < n_ranks; ++i) {
                displs[i] = displs[i-1] + recv_counts[i-1];
            }
        }
        
        MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE,
                    global_energy.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
                    
        if (rank == 0) {
            print_results(global_energy, "ElementEnergy");
        }
    }
    
    // Validation
    if (validate) {
        // Compute local sums/min/max
        val_t local_energy_sum = 0.0;
        val_t local_flux_sum = 0.0;
        val_t local_energy_max = std::numeric_limits<val_t>::lowest();
        val_t local_energy_min = std::numeric_limits<val_t>::max();
        
        for (int r = 0; r < num_own_rows; ++r) {
             int store_idx_start = (row_offset + r) * n_elems_root;
             for (int c = 0; c < n_elems_root; ++c) {
                 const auto& elem = world.elements_dynamic[store_idx_start + c];
                 local_energy_sum += elem.current_energy;
                 local_flux_sum += elem.total_flux;
                 local_energy_max = std::max(elem.current_energy, local_energy_max);
                 local_energy_min = std::min(elem.current_energy, local_energy_min);
             }
        }
        
        val_t global_energy_sum, global_flux_sum, global_energy_max, global_energy_min;
        
        MPI_Reduce(&local_energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            // Validation Logic (copied from validateResults but using globals)
             printf("Validation results:\n");
            printf("  Energy sum: %.12f\n", global_energy_sum);
            printf("  Flux sum: %.2f\n", global_flux_sum);
            printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);
            
            // Check for numerical issues
            constexpr val_t energy_epsilon = 1e-8;
            
            bool passed = true;
            if (!std::isfinite(global_energy_sum)) {
                printf("  ERROR: Energy sum is not finite\n");
                passed = false;
            }
            
            if (std::abs(global_energy_sum) > energy_epsilon) {
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            }
            
            if (!std::isfinite(global_flux_sum)) {
                printf("  ERROR: Flux sum is not finite\n");
                passed = false;
            }
            
            if (!std::isfinite(global_energy_max) || !std::isfinite(global_energy_min)) {
                printf("  ERROR: Energy extrema are not finite\n");
                passed = false;
            }
            
            if (passed) printf("  Validation: PASSED\n");
            else {
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
