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

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

__device__ inline val_t computeFluxDevice(const Material& mat,
                                          const ElementDynamic& this_elem,
                                          val_t connection_flux,
                                          const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Helper: atomicMax / atomicMin for double (not natively supported)
__device__ double atomicMaxCustom(double* address, double val) {
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(__longlong_as_double(assumed), val)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double atomicMinCustom(double* address, double val) {
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmin(__longlong_as_double(assumed), val)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Kernel: build 2D grid connectivity on device
__global__ void buildSquare2DKernel(ElementStatic* __restrict__ elements_static,
                                    ElementDynamic* __restrict__ elements_dynamic,
                                    const int n_elems_root,
                                    const int n_elems) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= n_elems_root || y >= n_elems_root) return;

    const int idx = x * n_elems_root + y;
    ElementStatic& elem = elements_static[idx];
    elem.material_idx = DEFAULT_MAT_ID;
    elem.num_connections = 0;
    elements_dynamic[idx].current_energy = 0.0;
    elements_dynamic[idx].total_flux = 0.0;

    // Connect to neighbors (right, left, up, down)
    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};

    for (int n = 0; n < 4; ++n) {
        const int nx = x + dx[n];
        const int ny = y + dy[n];
        if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
            const int neighbor_idx = nx * n_elems_root + ny;
            elem.connected_idx[elem.num_connections] = static_cast<idx_t>(neighbor_idx);
            elem.connected_flux[elem.num_connections] = 1.0;
            elem.num_connections++;
        }
    }

    // Set corner materials
    const int last = n_elems_root - 1;
    if (x == 0 && y == 0)                        elem.material_idx = INFLOW_MAT_ID;
    if (x == 0 && y == last)                     elem.material_idx = OUTFLOW_MAT_ID;
    if (x == last && y == 0)                     elem.material_idx = OUTFLOW_MAT_ID;
    if (x == last && y == last)                  elem.material_idx = INFLOW_MAT_ID;
}

// Kernel: single simulation step — one thread per element
__global__ void simulationStepKernel(
    const ElementStatic* __restrict__ elements_static,
    const Material* __restrict__ materials,
    const ElementDynamic* __restrict__ elements_dynamic,
    ElementDynamic* __restrict__ elements_dynamic_swap,
    const size_t n_elems)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const ElementStatic& elem_static = elements_static[i];
    const ElementDynamic& elem_dyn   = elements_dynamic[i];
    const Material& mat              = materials[elem_static.material_idx];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const idx_t neighbor_idx = elem_static.connected_idx[j];
        const ElementDynamic& neighbor_dyn = elements_dynamic[neighbor_idx];
        total_flux += computeFluxDevice(mat, elem_dyn,
                                        elem_static.connected_flux[j], neighbor_dyn);
    }

    // Update element state
    elements_dynamic_swap[i].current_energy = elem_dyn.current_energy + total_flux;
    elements_dynamic_swap[i].total_flux     = elem_dyn.total_flux + fabs(total_flux);
}

// Kernel: reduce — sum energy (per-block partial sums)
__global__ void reduceSumKernel(const ElementDynamic* __restrict__ elements,
                                const size_t n_elems,
                                double* d_energy_sum,
                                double* d_flux_sum) {
    extern __shared__ double sdata[];
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    double e_sum = 0.0;
    double f_sum = 0.0;
    for (size_t i = tid; i < n_elems; i += stride) {
        e_sum += elements[i].current_energy;
        f_sum += elements[i].total_flux;
    }

    // Block-level reduction
    sdata[threadIdx.x] = e_sum;
    sdata[threadIdx.x + blockDim.x] = f_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
            sdata[threadIdx.x + blockDim.x] += sdata[threadIdx.x + s + blockDim.x];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(d_energy_sum, sdata[0]);
        atomicAdd(d_flux_sum, sdata[blockDim.x]);
    }
}

// Kernel: reduce — min/max energy
__global__ void reduceMinMaxKernel(const ElementDynamic* __restrict__ elements,
                                   const size_t n_elems,
                                   double* d_energy_max,
                                   double* d_energy_min) {
    extern __shared__ double sdata[];
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    double e_max = -HUGE_VAL;
    double e_min =  HUGE_VAL;
    for (size_t i = tid; i < n_elems; i += stride) {
        e_max = fmax(e_max, elements[i].current_energy);
        e_min = fmin(e_min, elements[i].current_energy);
    }

    sdata[threadIdx.x] = e_max;
    sdata[threadIdx.x + blockDim.x] = e_min;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x]         = fmax(sdata[threadIdx.x],         sdata[threadIdx.x + s]);
            sdata[threadIdx.x + blockDim.x] = fmin(sdata[threadIdx.x + blockDim.x],
                                                   sdata[threadIdx.x + s + blockDim.x]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMaxCustom(d_energy_max, sdata[0]);
        atomicMinCustom(d_energy_min, sdata[blockDim.x]);
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers (unchanged semantics)
// ---------------------------------------------------------------------------

// Build a 2D square grid as an unstructured mesh (host-side copy for
// compatibility; actual work is done on GPU in buildSquare2DKernel).
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;

    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});

    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);

    // Initialise on host (values will be overwritten by GPU kernel)
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
}

// Run simulation for n_iters iterations — fully on GPU
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    const int n_elems_root = static_cast<int>(std::sqrt(static_cast<double>(n_elems)));

    // ---- allocate device memory ----
    Material*         d_materials        = nullptr;
    ElementStatic*    d_elements_static  = nullptr;
    ElementDynamic*   d_elements_dynamic = nullptr;
    ElementDynamic*   d_elements_swap    = nullptr;

    cudaMalloc(&d_materials,        world.materials.size() * sizeof(Material));
    cudaMalloc(&d_elements_static,  n_elems * sizeof(ElementStatic));
    cudaMalloc(&d_elements_dynamic, n_elems * sizeof(ElementDynamic));
    cudaMalloc(&d_elements_swap,    n_elems * sizeof(ElementDynamic));

    // ---- copy materials (read-only during simulation) ----
    cudaMemcpy(d_materials, world.materials.data(),
               world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    // ---- build mesh on GPU ----
    {
        dim3 block(32, 32);
        dim3 grid((n_elems_root + block.x - 1) / block.x,
                  (n_elems_root + block.y - 1) / block.y);

        // Initialise dynamic memory to zero
        cudaMemset(d_elements_dynamic, 0, n_elems * sizeof(ElementDynamic));

        buildSquare2DKernel<<<grid, block>>>(d_elements_static,
                                             d_elements_dynamic,
                                             n_elems_root,
                                             static_cast<int>(n_elems));
    }
    cudaDeviceSynchronize();

    // ---- simulation loop ----
    const int threads = 256;
    const int blocks  = (static_cast<int>(n_elems) + threads - 1) / threads;

    for (int iter = 0; iter < n_iters; ++iter) {
        simulationStepKernel<<<blocks, threads>>>(
            d_elements_static, d_materials,
            d_elements_dynamic, d_elements_swap, n_elems);

        // Swap pointers (zero-copy swap on device)
        std::swap(d_elements_dynamic, d_elements_swap);
    }
    cudaDeviceSynchronize();

    // ---- copy results back ----
    cudaMemcpy(world.elements_dynamic.data(), d_elements_dynamic,
               n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost);

    // ---- free device memory ----
    cudaFree(d_materials);
    cudaFree(d_elements_static);
    cudaFree(d_elements_dynamic);
    cudaFree(d_elements_swap);
}

// Validate simulation results (host-side, same as original)
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
