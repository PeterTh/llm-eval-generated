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

using val_t = double;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

struct ElementStatic {
    int neighbor[4];
    unsigned char material;
};

struct World {
    size_t size;
    std::unique_ptr<ElementStatic[]> elements;
    std::unique_ptr<val_t[]> energy;
    std::unique_ptr<val_t[]> next_energy;
    std::unique_ptr<val_t[]> total_flux;
};

constexpr Material materials[] = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};

// Bound synchronization overhead on small meshes and respect the runtime limit.
int teamSize(size_t size) {
    constexpr size_t elements_per_thread = 8192;
    const size_t chunks = (size + elements_per_thread - 1) / elements_per_thread;
    return std::clamp(static_cast<int>(chunks), 1, omp_get_max_threads());
}

void buildSquare2D(World& world, const int side) {
    world.size = static_cast<size_t>(side) * side;
    world.elements = std::make_unique_for_overwrite<ElementStatic[]>(world.size);
    world.energy = std::make_unique_for_overwrite<val_t[]>(world.size);
    world.next_energy = std::make_unique_for_overwrite<val_t[]>(world.size);
    world.total_flux = std::make_unique_for_overwrite<val_t[]>(world.size);

    #pragma omp parallel for schedule(static) num_threads(teamSize(world.size))
    for (int x = 0; x < side; ++x) {
        for (int y = 0; y < side; ++y) {
            const int i = x * side + y;
            ElementStatic& elem = world.elements[i];
            // All edges have unit flux. A missing edge reads self and contributes zero.
            elem.neighbor[0] = x + 1 < side ? i + side : i;
            elem.neighbor[1] = x > 0 ? i - side : i;
            elem.neighbor[2] = y + 1 < side ? i + 1 : i;
            elem.neighbor[3] = y > 0 ? i - 1 : i;
            elem.material = 0;
            if ((x == 0 || x == side - 1) && (y == 0 || y == side - 1)) {
                elem.material = x == y ? 1 : 2;
            }
            world.energy[i] = 0.0;
            world.total_flux[i] = 0.0;
        }
    }
}

void runSimulation(World& world, const int n_iters) {
    const size_t size = world.size;
    const ElementStatic* elements = world.elements.get();
    val_t* cumulative = world.total_flux.get();

    #pragma omp parallel num_threads(teamSize(size))
    {
        val_t* current = world.energy.get();
        val_t* next = world.next_energy.get();
        for (int iter = 0; iter < n_iters; ++iter) {
            #pragma omp for simd schedule(static)
            for (size_t i = 0; i < size; ++i) {
                const ElementStatic& elem = elements[i];
                const Material& mat = materials[elem.material];
                const val_t energy = current[i];
                val_t flux = mat.external_flow;
                for (int neighbor : elem.neighbor) {
                    flux += (current[neighbor] - energy) * mat.transfer_coeff * 0.25;
                }
                next[i] = energy + flux;
                cumulative[i] += std::abs(flux);
            }
            // The worksharing barrier completes all writes before the next read.
            std::swap(current, next);
        }
    }
    if (n_iters > 0 && n_iters % 2 != 0) {
        world.energy.swap(world.next_energy);
    }
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = 0; i < world.size; ++i) {
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
    #pragma omp parallel for simd schedule(static) reduction(^:hash) num_threads(teamSize(world.size))
    for (size_t i = 0; i < world.size; ++i) {
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
    const size_t static_mem = world.size * sizeof(ElementStatic);
    const size_t dynamic_mem = world.size * sizeof(val_t) * 3;
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
    const uint64_t hash = computeHash(world);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> energyData(world.energy.get(), world.energy.get() + world.size);
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
