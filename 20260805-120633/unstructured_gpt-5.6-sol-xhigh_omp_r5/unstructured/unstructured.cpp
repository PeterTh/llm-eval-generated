#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections for the generated 2D grid.
constexpr int MAX_CONNECTIONS = 4;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element
struct ElementStatic {
    // Every generated edge has unit connection flux, so storing that invariant
    // per edge would add bandwidth without changing the calculation.
    uint32_t connected_idx[MAX_CONNECTIONS];  // Indices of connected elements
    uint8_t material_idx;
    uint8_t num_connections;

    // All fields used by the simulation are initialized in parallel.  Keeping
    // this constructor non-initializing avoids a serial first-touch pass in
    // std::vector::resize, which is important on NUMA machines.
    ElementStatic() noexcept {}
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;

    ElementDynamic() noexcept {}
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
    
    // Initialize state and build connectivity in the same parallel pass.  The
    // static schedule also establishes the NUMA placement used by simulation.
    constexpr int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < n_elems; ++idx) {
        const int x = idx / n_elems_root;
        const int y = idx - x * n_elems_root;
        ElementStatic& elem = world.elements_static[idx];

        elem.material_idx = static_cast<uint8_t>(DEFAULT_MAT_ID);
        elem.num_connections = 0;
        world.elements_dynamic[idx].current_energy = 0.0;
        world.elements_dynamic[idx].total_flux = 0.0;
        world.elements_dynamic_swap[idx].current_energy = 0.0;
        world.elements_dynamic_swap[idx].total_flux = 0.0;

        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];

            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const int neighbor_idx = nx * n_elems_root + ny;
                elem.connected_idx[elem.num_connections] =
                    static_cast<uint32_t>(neighbor_idx);
                ++elem.num_connections;
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    world.elements_static[0 * n_elems_root + 0].material_idx =
        static_cast<uint8_t>(INFLOW_MAT_ID);
    world.elements_static[0 * n_elems_root + last].material_idx =
        static_cast<uint8_t>(OUTFLOW_MAT_ID);
    world.elements_static[last * n_elems_root + 0].material_idx =
        static_cast<uint8_t>(OUTFLOW_MAT_ID);
    world.elements_static[last * n_elems_root + last].material_idx =
        static_cast<uint8_t>(INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    const ElementStatic* const static_elements = world.elements_static.data();
    const Material* const materials = world.materials.data();
    ElementDynamic* const primary_elements = world.elements_dynamic.data();
    ElementDynamic* const swap_elements = world.elements_dynamic_swap.data();

    // Keep one team alive across all timesteps and compute its contiguous
    // partition once.  Alternating the two fixed buffer pointers by iteration
    // leaves only the single barrier required by the Jacobi dependency.
#pragma omp parallel default(none) \
    shared(n_iters, n_elems, static_elements, materials, primary_elements, swap_elements)
    {
        const size_t thread_id = static_cast<size_t>(omp_get_thread_num());
        const size_t thread_count = static_cast<size_t>(omp_get_num_threads());
        const size_t elements_per_thread = n_elems / thread_count;
        const size_t extra_elements = n_elems % thread_count;
        const size_t begin = thread_id * elements_per_thread +
                             std::min(thread_id, extra_elements);
        const size_t end = begin + elements_per_thread +
                           (thread_id < extra_elements ? 1 : 0);

        for (int iter = 0; iter < n_iters; ++iter) {
            const ElementDynamic* const read_elements =
                (iter & 1) == 0 ? primary_elements : swap_elements;
            ElementDynamic* const write_elements =
                (iter & 1) == 0 ? swap_elements : primary_elements;

            for (size_t i = begin; i < end; ++i) {
                const ElementStatic& elem_static = static_elements[i];
                const ElementDynamic& elem_dyn = read_elements[i];
                const Material& mat = materials[elem_static.material_idx];

                // Start with external flow
                val_t total_flux = mat.external_flow;

                // Add flux from all connected elements
                for (uint8_t j = 0; j < elem_static.num_connections; ++j) {
                    const uint32_t neighbor_idx = elem_static.connected_idx[j];
                    const ElementDynamic& neighbor_dyn = read_elements[neighbor_idx];
                    total_flux += computeFlux(mat, elem_dyn, neighbor_dyn);
                }

                // Update element state
                ElementDynamic& elem_write = write_elements[i];
                elem_write.current_energy = elem_dyn.current_energy + total_flux;
                elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }

#pragma omp barrier
        }
    }

    // Make the vector owning the final read buffer the externally visible one.
    if (n_iters > 0 && (n_iters & 1) != 0) {
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    // Dynamic teams can silently serialize small regions.  A fixed team makes
    // the OpenMP execution model unconditional and the timestep cost stable.
    omp_set_dynamic(0);
    
    const int n_elems = n_elems_root * n_elems_root;
    
    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");
    
    // Build the unstructured mesh
    printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
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
    const uint64_t hash = computeHash(world.elements_dynamic);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> energyData;
        energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (!valid) {
            return 1;
        }
    }
    
    return 0;
}
