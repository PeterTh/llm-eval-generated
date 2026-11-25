#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint32_t;
using val_t = double;

// The square mesh builder supplies at most four neighbors per element.
constexpr int MAX_CONNECTIONS = 4;

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

// World state
struct World {
    std::vector<Material> materials;
    size_t n_elems;
    int num_threads;
    std::unique_ptr<ElementStatic[]> elements_static;
    std::unique_ptr<val_t[]> energy;
    std::unique_ptr<val_t[]> energy_swap;
    std::unique_ptr<val_t[]> total_flux;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const size_t width) {
    world.n_elems = width * width;
    // Keep enough work per worker to amortize the barrier in each step.
    constexpr size_t elements_per_thread = 8192;
    world.num_threads = static_cast<int>(std::min<size_t>(
        omp_get_max_threads(), std::max<size_t>(1, world.n_elems / elements_per_thread)));
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // First-touch the arrays in the same contiguous partitions used by the solver.
    world.elements_static = std::make_unique_for_overwrite<ElementStatic[]>(world.n_elems);
    world.energy = std::make_unique_for_overwrite<val_t[]>(world.n_elems);
    world.energy_swap = std::make_unique_for_overwrite<val_t[]>(world.n_elems);
    world.total_flux = std::make_unique_for_overwrite<val_t[]>(world.n_elems);

    const size_t last = width - 1;
    #pragma omp parallel for collapse(2) schedule(static) num_threads(world.num_threads)
    for (size_t x = 0; x < width; ++x) {
        for (size_t y = 0; y < width; ++y) {
            const size_t idx = x * width + y;
            ElementStatic& elem = world.elements_static[idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            world.energy[idx] = 0.0;
            world.energy_swap[idx] = 0.0;
            world.total_flux[idx] = 0.0;

            if ((x == 0 || x == last) && (y == 0 || y == last)) {
                elem.material_idx = x == y ? INFLOW_MAT_ID : OUTFLOW_MAT_ID;
            }

            // Keep the original neighbor order for the floating-point sum.
            const auto connect = [&elem](size_t neighbor) {
                const idx_t connection = elem.num_connections++;
                elem.connected_idx[connection] = static_cast<idx_t>(neighbor);
                elem.connected_flux[connection] = 1.0;
            };
            if (x < last) connect(idx + width);
            if (x > 0) connect(idx - width);
            if (y < last) connect(idx + 1);
            if (y > 0) connect(idx - 1);
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, val_t energy, val_t connection_flux,
                        val_t neighbor_energy) {
    return (neighbor_energy - energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.n_elems;
    const ElementStatic* elements = world.elements_static.get();
    const Material* materials = world.materials.data();
    val_t* accumulated_flux = world.total_flux.get();
    val_t* energy = world.energy.get();
    val_t* next_energy = world.energy_swap.get();

    #pragma omp parallel num_threads(world.num_threads) firstprivate(energy, next_energy)
    {
        for (int iter = 0; iter < n_iters; ++iter) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n_elems; ++i) {
                const ElementStatic& elem = elements[i];
                const Material& mat = materials[elem.material_idx];
                const val_t current_energy = energy[i];
                val_t total_flux = mat.external_flow;
                for (idx_t j = 0; j < elem.num_connections; ++j) {
                    total_flux += computeFlux(mat, current_energy, elem.connected_flux[j],
                                              energy[elem.connected_idx[j]]);
                }
                next_energy[i] = current_energy + total_flux;
                accumulated_flux[i] += std::abs(total_flux);
            }

            // The loop barrier publishes the complete next state before it is read.
            std::swap(energy, next_energy);
        }
    }
    if (n_iters > 0 && n_iters % 2 != 0) {
        world.energy.swap(world.energy_swap);
    }
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    // Keep the serial summation order so the diagnostics are reproducible.
    for (size_t i = 0; i < world.n_elems; ++i) {
        const val_t energy = world.energy[i];
        energy_sum += energy;
        flux_sum += world.total_flux[i];
        energy_max = std::max(energy, energy_max);
        energy_min = std::min(energy, energy_min);
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
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    const val_t* energy = world.energy.get();
    const val_t* flux = world.total_flux.get();
    #pragma omp parallel for simd schedule(static) num_threads(world.num_threads) reduction(^:hash)
    for (size_t i = 0; i < world.n_elems; ++i) {
        hash ^= (std::bit_cast<uint64_t>(energy[i]) + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (std::bit_cast<uint64_t>(flux[i]) + i) * 0xbf58476d1ce4e5b9ULL;
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
    
    if (n_elems_root <= 0 ||
        static_cast<uint64_t>(n_elems_root) * n_elems_root > std::numeric_limits<idx_t>::max()) {
        fprintf(stderr, "Grid size must be positive and fit in 32-bit element indices.\n");
        return 1;
    }
    const size_t width = static_cast<size_t>(n_elems_root);
    const size_t n_elems = width * width;
    
    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("Grid size: %d x %d = %zu elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");
    
    // Build the unstructured mesh
    printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, width);
    
    // Calculate memory usage
    const size_t static_mem = n_elems * sizeof(ElementStatic);
    const size_t dynamic_mem = n_elems * sizeof(val_t) * 3;
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
    const int n_measured_iters = std::max(n_iters, 2) - 1;
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double time_per_iter = elapsed_ms / n_measured_iters;
    const double giga_elems_per_sec = n_elems / time_per_iter / 1e6;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults) {
        const val_t* energy = world.energy.get();
        std::vector<val_t> energyData(energy, energy + world.n_elems);
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
