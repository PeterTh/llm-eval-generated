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
#include <utility>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

using val_t = double;

// Store each neighbor slot contiguously for SIMD across elements.
constexpr int CONNECTIONS_PER_ELEMENT = 4;
constexpr val_t TRANSFER_COEFF = 0.8;

struct World {
    size_t size;
    int threads;
    std::unique_ptr<int[]> neighbors[CONNECTIONS_PER_ELEMENT];
    std::unique_ptr<val_t[]> external_flow;
    std::unique_ptr<val_t[]> energy;
    std::unique_ptr<val_t[]> energy_swap;
    std::unique_ptr<val_t[]> total_flux;
};

// Build indexed connectivity for the square mesh.
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    const int last = n_elems_root - 1;
    world.size = n_elems;

    // Reserve enough work per thread to amortize the iteration barrier.
    constexpr int elements_per_thread = 16384;
    world.threads = std::min(omp_get_max_threads(), 1 + (n_elems - 1) / elements_per_thread);

    for (auto& neighbors : world.neighbors) {
        neighbors = std::make_unique_for_overwrite<int[]>(n_elems);
    }
    world.external_flow = std::make_unique_for_overwrite<val_t[]>(n_elems);
    world.energy = std::make_unique_for_overwrite<val_t[]>(n_elems);
    world.energy_swap = std::make_unique_for_overwrite<val_t[]>(n_elems);
    world.total_flux = std::make_unique_for_overwrite<val_t[]>(n_elems);

    // First-touch the arrays using the same element partition as the simulation.
    #pragma omp parallel for schedule(static) num_threads(world.threads)
    for (int i = 0; i < n_elems; ++i) {
        const int x = i / n_elems_root;
        const int y = i % n_elems_root;

        // Missing neighbors contribute zero flux through self-connections.
        // Keep real connections in the original order.
        world.neighbors[0][i] = x < last ? i + n_elems_root : i;
        world.neighbors[1][i] = x > 0 ? i - n_elems_root : i;
        world.neighbors[2][i] = y < last ? i + 1 : i;
        world.neighbors[3][i] = y > 0 ? i - 1 : i;
        world.energy[i] = 0.0;
        world.total_flux[i] = 0.0;

        val_t flow = 0.0;
        if ((x == 0 || x == last) && (y == 0 || y == last)) {
            flow = x == y ? 0.5 : -0.5;
        }
        world.external_flow[i] = flow;
    }
}

inline val_t computeFlux(val_t current, val_t neighbor) {
    return (neighbor - current) * TRANSFER_COEFF * 0.25;
}

// Run simulation for n_iters iterations.
void runSimulation(World& world, const int n_iters) {
    if (n_iters <= 0) {
        return;
    }

    const size_t n_elems = world.size;
    const int* const neighbor0 = world.neighbors[0].get();
    const int* const neighbor1 = world.neighbors[1].get();
    const int* const neighbor2 = world.neighbors[2].get();
    const int* const neighbor3 = world.neighbors[3].get();
    const val_t* const external_flow = world.external_flow.get();
    val_t* const accumulated_flux = world.total_flux.get();

    #pragma omp parallel num_threads(world.threads)
    {
        val_t* energy = world.energy.get();
        val_t* next_energy = world.energy_swap.get();

        for (int iter = 0; iter < n_iters; ++iter) {
            #pragma omp for simd schedule(static)
            for (size_t i = 0; i < n_elems; ++i) {
                const val_t current = energy[i];
                val_t total_flux = external_flow[i];

                // Accumulate each flux in order to preserve floating-point results.
                total_flux += computeFlux(current, energy[neighbor0[i]]);
                total_flux += computeFlux(current, energy[neighbor1[i]]);
                total_flux += computeFlux(current, energy[neighbor2[i]]);
                total_flux += computeFlux(current, energy[neighbor3[i]]);

                next_energy[i] = current + total_flux;
                accumulated_flux[i] += std::abs(total_flux);
            }

            // The loop barrier publishes the output before the private pointers flip.
            std::swap(energy, next_energy);
        }
    }

    if (n_iters % 2 != 0) {
        world.energy.swap(world.energy_swap);
    }
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = 0; i < world.size; ++i) {
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
    const val_t* const energy = world.energy.get();
    const val_t* const flux = world.total_flux.get();

    #pragma omp parallel for simd schedule(static) num_threads(world.threads) reduction(^ : hash)
    for (size_t i = 0; i < world.size; ++i) {
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
    
    if (n_elems_root <= 0 || n_elems_root > std::numeric_limits<int>::max() / n_elems_root) {
        fprintf(stderr, "Grid size must be positive and its square must fit in an int.\n");
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
    const size_t static_mem = world.size * (CONNECTIONS_PER_ELEMENT * sizeof(int) + sizeof(val_t));
    const size_t dynamic_mem = world.size * 3 * sizeof(val_t);
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
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;
    
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
        const val_t* const energy = world.energy.get();
        const std::vector<val_t> energyData(energy, energy + world.size);
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
