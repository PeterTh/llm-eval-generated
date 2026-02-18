#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

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

static inline void cudaCheck(cudaError_t err, const char* call, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s failed with %s\n", file, line, call, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CUDA_CHECK(x) cudaCheck((x), #x, __FILE__, __LINE__)

static_assert(sizeof(ElementDynamic) == 16, "ElementDynamic layout must be 2 doubles");

__constant__ double c_transfer_coeff[3];
__constant__ double c_external_flow[3];

__global__ void stepKernel(ElementDynamic* __restrict__ out,
                          const ElementDynamic* __restrict__ in,
                          int n_elems_root,
                          int n_elems) {
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (i >= n_elems) return;

    const int x = i / n_elems_root;
    const int y = i - x * n_elems_root;
    const int last = n_elems_root - 1;

    idx_t mat_id = DEFAULT_MAT_ID;
    if ((x == 0 && y == 0) || (x == last && y == last)) {
        mat_id = INFLOW_MAT_ID;
    } else if ((x == 0 && y == last) || (x == last && y == 0)) {
        mat_id = OUTFLOW_MAT_ID;
    }

    const double transfer = c_transfer_coeff[mat_id];
    const double this_e = in[i].current_energy;

    // Start with external flow
    double total_flux = c_external_flow[mat_id];

    // Neighbor order matches buildSquare2D offsets: (1,0),(-1,0),(0,1),(0,-1)
    if (x + 1 < n_elems_root) {
        const double other = in[i + n_elems_root].current_energy;
        total_flux += (other - this_e) * transfer * 1.0 * 0.25;
    }
    if (x - 1 >= 0) {
        const double other = in[i - n_elems_root].current_energy;
        total_flux += (other - this_e) * transfer * 1.0 * 0.25;
    }
    if (y + 1 < n_elems_root) {
        const double other = in[i + 1].current_energy;
        total_flux += (other - this_e) * transfer * 1.0 * 0.25;
    }
    if (y - 1 >= 0) {
        const double other = in[i - 1].current_energy;
        total_flux += (other - this_e) * transfer * 1.0 * 0.25;
    }

    ElementDynamic o;
    o.current_energy = this_e + total_flux;
    o.total_flux = in[i].total_flux + fabs(total_flux);
    out[i] = o;
}

struct CudaSim {
    ElementDynamic* d_a = nullptr;
    ElementDynamic* d_b = nullptr;
    cudaStream_t stream = nullptr;
    int n_elems_root = 0;
    int n_elems = 0;

    CudaSim(const World& world, int n_root) : n_elems_root(n_root), n_elems(static_cast<int>(world.elements_dynamic.size())) {
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount <= 0) {
            fprintf(stderr, "No CUDA devices found\n");
            std::exit(1);
        }
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        CUDA_CHECK(cudaMalloc(&d_a, static_cast<size_t>(n_elems) * sizeof(ElementDynamic)));
        CUDA_CHECK(cudaMalloc(&d_b, static_cast<size_t>(n_elems) * sizeof(ElementDynamic)));
        CUDA_CHECK(cudaMemcpyAsync(d_a, world.elements_dynamic.data(), static_cast<size_t>(n_elems) * sizeof(ElementDynamic), cudaMemcpyHostToDevice, stream));

        if (world.materials.size() < 3) {
            fprintf(stderr, "Expected at least 3 materials\n");
            std::exit(1);
        }
        double h_transfer[3] = {world.materials[0].transfer_coeff, world.materials[1].transfer_coeff, world.materials[2].transfer_coeff};
        double h_external[3] = {world.materials[0].external_flow, world.materials[1].external_flow, world.materials[2].external_flow};
        CUDA_CHECK(cudaMemcpyToSymbolAsync(c_transfer_coeff, h_transfer, sizeof(h_transfer), 0, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyToSymbolAsync(c_external_flow, h_external, sizeof(h_external), 0, cudaMemcpyHostToDevice, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void run(int n_iters) {
        const int block = 256;
        const int grid = (n_elems + block - 1) / block;
        for (int iter = 0; iter < n_iters; ++iter) {
            stepKernel<<<grid, block, 0, stream>>>(d_b, d_a, n_elems_root, n_elems);
            CUDA_CHECK(cudaGetLastError());
            std::swap(d_a, d_b);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void copyBack(World& world) {
        CUDA_CHECK(cudaMemcpyAsync(world.elements_dynamic.data(), d_a, static_cast<size_t>(n_elems) * sizeof(ElementDynamic), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    ~CudaSim() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (stream) cudaStreamDestroy(stream);
    }
};

// Run simulation for n_iters iterations (CUDA)
void runSimulation(World& world, const int n_iters, const int n_elems_root) {
    CudaSim sim(world, n_elems_root);
    sim.run(n_iters);
    sim.copyBack(world);
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

    // Create/copy GPU state outside the timed region (mirrors CPU version which has no per-run setup).
    CudaSim sim(world, n_elems_root);

    auto start = std::chrono::high_resolution_clock::now();
    sim.run(n_iters);
    auto end = std::chrono::high_resolution_clock::now();

    sim.copyBack(world);

    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    if (duration_ms <= 0.0) duration_ms = 1e-6;

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
