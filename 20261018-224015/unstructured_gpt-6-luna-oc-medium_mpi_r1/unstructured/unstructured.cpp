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
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
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
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Update all elements
        for (size_t i = 0; i < n_elems; ++i) {
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
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    const int base = n_elems / n_ranks;
    const int remainder = n_elems % n_ranks;
    const int local_count = base + (rank < remainder ? 1 : 0);
    const int local_start = rank * base + std::min(rank, remainder);
    
    if (rank == 0) printf("Unstructured Mesh Energy Transfer Benchmark\n");
    if (rank == 0) printf("============================================\n");
    if (rank == 0) { printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);
    printf("Validation: %s\n", validate ? "enabled" : "disabled"); printf("\n"); }
    
    // Build the unstructured mesh
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    // Connectivity is generated only for this rank's contiguous partition.
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(local_count);
    world.elements_dynamic.resize(local_count);
    world.elements_dynamic_swap.resize(local_count);
    for (int li = 0; li < local_count; ++li) {
        int g = local_start + li, x = g / n_elems_root, y = g % n_elems_root;
        auto& e = world.elements_static[li];
        e.material_idx = DEFAULT_MAT_ID; e.num_connections = 0;
        world.elements_dynamic[li] = {0.0, 0.0};
        const int dx[] = {1,-1,0,0}, dy[] = {0,0,1,-1};
        for (int d = 0; d < 4; ++d) {
            int nx=x+dx[d], ny=y+dy[d];
            if (nx>=0 && nx<n_elems_root && ny>=0 && ny<n_elems_root) {
                auto j=e.num_connections++; e.connected_idx[j]=nx*n_elems_root+ny; e.connected_flux[j]=1.0;
            }
        }
        if (g == 0 || g == n_elems-1) e.material_idx=INFLOW_MAT_ID;
        if (g == n_elems_root-1 || g == n_elems-n_elems_root) e.material_idx=OUTFLOW_MAT_ID;
    }
    
    // Calculate memory usage
    const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    if (rank == 0) printf("\n");
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<double> global_energy(n_elems), global_flux(n_elems);
    std::vector<double> local_energy(local_count), local_flux(local_count);
    std::vector<int> counts(n_ranks), displs(n_ranks);
    for (int r=0, off=0; r<n_ranks; ++r) { counts[r]=n_elems/n_ranks+(r<n_elems%n_ranks); displs[r]=off; off+=counts[r]; }
    for (int iter=0; iter<n_iters; ++iter) {
        for (int li=0; li<local_count; ++li) local_energy[li]=world.elements_dynamic[li].current_energy;
        MPI_Allgatherv(local_energy.data(), local_count, MPI_DOUBLE, global_energy.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        for (int li=0; li<local_count; ++li) {
            const auto& s=world.elements_static[li]; const auto& d=world.elements_dynamic[li];
            val_t f=world.materials[s.material_idx].external_flow;
            for (idx_t j=0;j<s.num_connections;++j) {
                const int gi=s.connected_idx[j];
                f += (global_energy[gi]-d.current_energy)*world.materials[s.material_idx].transfer_coeff*s.connected_flux[j]*0.25;
            }
            world.elements_dynamic_swap[li]={d.current_energy+f,d.total_flux+std::abs(f)};
        }
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
    for (int li=0; li<local_count; ++li) { local_energy[li]=world.elements_dynamic[li].current_energy; local_flux[li]=world.elements_dynamic[li].total_flux; }
    MPI_Allgatherv(local_energy.data(), local_count, MPI_DOUBLE, global_energy.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(local_flux.data(), local_count, MPI_DOUBLE, global_flux.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long local_ms = duration_ms, max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    duration_ms = max_ms;
    
    if (rank == 0) printf("Computation time: %ld ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    if (rank == 0) printf("Performance:\n");
    if (rank == 0) { printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops); }
    
    // Compute hash for verification
    std::vector<ElementDynamic> full_state;
    if (rank == 0) full_state.resize(n_elems);
    for (int r=0; r<n_ranks; ++r) { counts[r] *= sizeof(ElementDynamic); displs[r] *= sizeof(ElementDynamic); }
    MPI_Gatherv(world.elements_dynamic.data(), local_count * sizeof(ElementDynamic), MPI_BYTE,
                rank == 0 ? full_state.data() : nullptr, counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const uint64_t hash = computeHash(full_state);
        printf("  Result hash: %016lX\n\n", hash);
    }
    
    // Print results for external validation
    if (printResults) {
        if (rank == 0) print_results(global_energy, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            World full;
            full.elements_dynamic.resize(n_elems);
            for (int i=0; i<n_elems; ++i) full.elements_dynamic[i]={global_energy[i],global_flux[i]};
            if (!validateResults(full)) { MPI_Finalize(); return 1; }
        }
    }
    MPI_Finalize();
    return 0;
}
