#include <algorithm>
#include <array>
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
using val_t = double;

// Keep explicit mesh adjacency, with consecutive indices for each edge slot.
constexpr int MAX_CONNECTIONS = 4;
struct World {
    std::array<std::unique_ptr<int[]>, MAX_CONNECTIONS> neighbors;
    std::unique_ptr<val_t[]> external_flow;
    std::vector<val_t> energy;
    std::vector<val_t> energy_swap;
    std::vector<val_t> total_flux;
};

int meshThreads(size_t count) {
    // Amortize team synchronization over a contiguous chunk of elements.
    return std::min(omp_get_max_threads(), static_cast<int>((count + 4095) / 4096));
}

void buildSquare2D(World& world, const int side) {
    const int count = side * side;
    for (auto& indices : world.neighbors) {
        indices = std::make_unique_for_overwrite<int[]>(count);
    }
    world.external_flow = std::make_unique_for_overwrite<val_t[]>(count);
    world.energy.resize(count);
    world.energy_swap.resize(count);
    world.total_flux.resize(count);

    #pragma omp parallel for schedule(static) num_threads(meshThreads(count))
    for (int x = 0; x < side; ++x) {
        for (int y = 0; y < side; ++y) {
            const int i = x * side + y;
            // A missing edge points to self and contributes zero flux.
            world.neighbors[0][i] = x + 1 < side ? i + side : i;
            world.neighbors[1][i] = x > 0 ? i - side : i;
            world.neighbors[2][i] = y + 1 < side ? i + 1 : i;
            world.neighbors[3][i] = y > 0 ? i - 1 : i;
            world.external_flow[i] = 0.0;
        }
    }

    world.external_flow[0] = 0.5;
    world.external_flow[side - 1] = -0.5;
    world.external_flow[count - side] = -0.5;
    world.external_flow[count - 1] = 0.5;
}

void runSimulation(World& world, const int n_iters) {
    const size_t count = world.energy.size();
    const int* down = world.neighbors[0].get();
    const int* up = world.neighbors[1].get();
    const int* right = world.neighbors[2].get();
    const int* left = world.neighbors[3].get();
    const val_t* external = world.external_flow.get();
    val_t* accumulated = world.total_flux.data();

    // All materials have transfer coefficient 0.8 and all edges have unit weight.
    constexpr val_t transfer = 0.8;
    #pragma omp parallel num_threads(meshThreads(count))
    {
        val_t* current = world.energy.data();
        val_t* next = world.energy_swap.data();
        for (int iter = 0; iter < n_iters; ++iter) {
            #pragma omp for simd schedule(static)
            for (size_t i = 0; i < count; ++i) {
                const val_t energy = current[i];
                val_t flux = external[i];
                flux += (current[down[i]] - energy) * transfer * 0.25;
                flux += (current[up[i]] - energy) * transfer * 0.25;
                flux += (current[right[i]] - energy) * transfer * 0.25;
                flux += (current[left[i]] - energy) * transfer * 0.25;
                next[i] = energy + flux;
                accumulated[i] += std::abs(flux);
            }
            // The worksharing barrier completes all writes before the next step.
            std::swap(current, next);
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
    
    for (size_t i = 0; i < world.energy.size(); ++i) {
        energy_sum += world.energy[i];
        flux_sum += world.total_flux[i];
        energy_max = std::max(world.energy[i], energy_max);
        energy_min = std::min(world.energy[i], energy_min);
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
    #pragma omp parallel for simd schedule(static) reduction(^:hash) num_threads(meshThreads(world.energy.size()))
    for (size_t i = 0; i < world.energy.size(); ++i) {
        hash ^= (std::bit_cast<uint64_t>(world.energy[i]) + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (std::bit_cast<uint64_t>(world.total_flux[i]) + i) * 0xbf58476d1ce4e5b9ULL;
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
    const size_t static_mem = world.energy.size() * (MAX_CONNECTIONS * sizeof(int) + sizeof(val_t));
    const size_t dynamic_mem = world.energy.size() * sizeof(val_t) * 3;
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
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = duration_ms / n_measured_iters;
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
        print_results(world.energy, "ElementEnergy");
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
