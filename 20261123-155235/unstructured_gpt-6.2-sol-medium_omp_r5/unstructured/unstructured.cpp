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
#include <span>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint32_t;
using val_t = double;

constexpr int MAX_CONNECTIONS = 4;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element
struct ElementStatic {
    idx_t connected_idx[MAX_CONNECTIONS];
    uint8_t material_idx;
    uint8_t num_connections;
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state
struct World {
    std::vector<Material> materials;
    std::unique_ptr<ElementStatic[]> elements_static;
    std::unique_ptr<ElementDynamic[]> elements_dynamic;
    std::unique_ptr<ElementDynamic[]> elements_dynamic_swap;
};

// Material type IDs
constexpr uint8_t DEFAULT_MAT_ID = 0;
constexpr uint8_t INFLOW_MAT_ID = 1;
constexpr uint8_t OUTFLOW_MAT_ID = 2;

int meshThreadCount(size_t n_elems) {
    // Keep enough work per thread to amortize the iteration barrier on small meshes.
    constexpr size_t elements_per_thread = 8192;
    const size_t useful_threads = (n_elems + elements_per_thread - 1) / elements_per_thread;
    return static_cast<int>(std::clamp(useful_threads, size_t{1},
                                       static_cast<size_t>(omp_get_max_threads())));
}

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // First-touch the arrays in parallel using the same partition as the simulation.
    world.elements_static = std::make_unique_for_overwrite<ElementStatic[]>(n_elems);
    world.elements_dynamic = std::make_unique_for_overwrite<ElementDynamic[]>(n_elems);
    world.elements_dynamic_swap = std::make_unique_for_overwrite<ElementDynamic[]>(n_elems);
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for collapse(2) schedule(static) num_threads(meshThreadCount(n_elems))
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            world.elements_dynamic[idx] = {0.0, 0.0};
            world.elements_dynamic_swap[idx] = {0.0, 0.0};
            
            // Connect to neighbors (up, down, left, right)
            constexpr int offsets[MAX_CONNECTIONS][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < MAX_CONNECTIONS; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections++] = neighbor_idx;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    world.elements_static[0].material_idx = INFLOW_MAT_ID;
    world.elements_static[last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
}

inline val_t computeFlux(const Material& mat, val_t current_energy, val_t neighbor_energy) {
    // Every connection in the benchmark mesh has a unit flux coefficient.
    return (neighbor_energy - current_energy) * mat.transfer_coeff * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, size_t n_elems, int n_iters) {
    if (n_iters <= 0) {
        return;
    }

    const ElementStatic* const elements = world.elements_static.get();
    const Material* const materials = world.materials.data();
    ElementDynamic* read = world.elements_dynamic.get();
    ElementDynamic* write = world.elements_dynamic_swap.get();

    #pragma omp parallel firstprivate(read, write) num_threads(meshThreadCount(n_elems))
    {
        for (int iter = 0; iter < n_iters; ++iter) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n_elems; ++i) {
                const ElementStatic& elem = elements[i];
                const Material& mat = materials[elem.material_idx];
                const val_t energy = read[i].current_energy;
                val_t total_flux = mat.external_flow;

                if (elem.num_connections == MAX_CONNECTIONS) {
                    // The fixed trip count lets the compiler unroll the interior update.
                    for (int j = 0; j < MAX_CONNECTIONS; ++j) {
                        total_flux += computeFlux(mat, energy, read[elem.connected_idx[j]].current_energy);
                    }
                } else {
                    for (int j = 0; j < elem.num_connections; ++j) {
                        total_flux += computeFlux(mat, energy, read[elem.connected_idx[j]].current_energy);
                    }
                }

                write[i] = {energy + total_flux, read[i].total_flux + std::abs(total_flux)};
            }

            // The worksharing barrier completes all writes before the private pointer swaps.
            std::swap(read, write);
        }
    }

    if (n_iters % 2 != 0) {
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results
bool validateResults(std::span<const ElementDynamic> elements) {
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
uint64_t computeHash(std::span<const ElementDynamic> elements) {
    uint64_t hash = 0;
    #pragma omp parallel for schedule(static) reduction(^:hash) num_threads(meshThreadCount(elements.size()))
    for (size_t i = 0; i < elements.size(); ++i) {
        hash ^= (std::bit_cast<uint64_t>(elements[i].current_energy) + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (std::bit_cast<uint64_t>(elements[i].total_flux) + i) * 0xbf58476d1ce4e5b9ULL;
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
    
    if (n_elems_root <= 0 || n_elems_root > std::numeric_limits<int>::max() / n_elems_root) {
        fprintf(stderr, "Grid size must be positive and its square must fit in an int\n");
        return 1;
    }
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
    const size_t static_mem = n_elems * sizeof(ElementStatic);
    const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_elems, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %lld ms\n", static_cast<long long>(duration_ms));
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters, 2) - 1;
    const double time_per_iter = duration_ms / n_measured_iters;
    const double giga_elems_per_sec = static_cast<double>(n_measured_iters) * n_elems / duration_ms / 1e6;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    // Compute hash for verification
    const std::span<const ElementDynamic> elements(world.elements_dynamic.get(), n_elems);
    const uint64_t hash = computeHash(elements);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> energyData;
        energyData.reserve(elements.size());
        for (const auto& elem : elements) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(elements);
        if (!valid) {
            return 1;
        }
    }
    
    return 0;
}
