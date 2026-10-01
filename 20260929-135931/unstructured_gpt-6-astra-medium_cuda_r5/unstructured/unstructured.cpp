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

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation)

// The square builder needs at most four edges. Connectivity is stored by edge
// slot, so adjacent GPU threads read adjacent entries even on irregular meshes.
constexpr int GRID_CONNECTIONS = 4;
__constant__ Material device_materials[3] = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};

struct DeviceMesh {
    idx_t* neighbors = nullptr;
    val_t* weights = nullptr;
    unsigned char* counts = nullptr;
    unsigned char* materials = nullptr;
    val_t* energy[2] = {nullptr, nullptr};
    val_t* flux = nullptr;
    ElementDynamic* result = nullptr;
};

struct World {
    std::vector<ElementDynamic> elements_dynamic;
    DeviceMesh device;
    size_t size = 0;
    World() = default;
    World(const World&) = delete;
    World& operator=(const World&) = delete;
    ~World() {
        cudaFree(device.neighbors);
        cudaFree(device.weights);
        cudaFree(device.counts);
        cudaFree(device.materials);
        cudaFree(device.energy[0]);
        cudaFree(device.energy[1]);
        cudaFree(device.flux);
        cudaFree(device.result);
    }
};

template<class T>
void allocateDevice(T*& pointer, size_t count) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
}

__global__ void buildMesh(DeviceMesh mesh, size_t size, int root) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const int x = i / root;
    const int y = i % root;
    int count = 0;
    // Preserve the original up, down, left, right accumulation order.
    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};
    #pragma unroll
    for (int j = 0; j < GRID_CONNECTIONS; ++j) {
        const int nx = x + dx[j], ny = y + dy[j];
        if (nx >= 0 && nx < root && ny >= 0 && ny < root) {
            mesh.neighbors[size_t(count) * size + i] = size_t(nx) * root + ny;
            mesh.weights[size_t(count) * size + i] = 1.0;
            ++count;
        }
    }
    mesh.counts[i] = count;
    unsigned char material = 0;
    if (i == 0) material = 1;
    if (i == size_t(root - 1)) material = 2;
    if (i == size - root) material = 2;
    if (i == size - 1) material = 1; // Last assignment also wins for a 1x1 grid.
    mesh.materials[i] = material;
    mesh.energy[0][i] = 0.0;
    mesh.energy[1][i] = 0.0;
    mesh.flux[i] = 0.0;
}

void buildSquare2D(World& world, const int root) {
    world.size = size_t(root) * root;
    world.elements_dynamic.resize(world.size);
    auto& mesh = world.device;
    allocateDevice(mesh.neighbors, world.size * GRID_CONNECTIONS);
    allocateDevice(mesh.weights, world.size * GRID_CONNECTIONS);
    allocateDevice(mesh.counts, world.size);
    allocateDevice(mesh.materials, world.size);
    allocateDevice(mesh.energy[0], world.size);
    allocateDevice(mesh.energy[1], world.size);
    allocateDevice(mesh.flux, world.size);
    allocateDevice(mesh.result, world.size);
    buildMesh<<<(world.size + 255) / 256, 256>>>(mesh, world.size, root);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void updateElements(DeviceMesh mesh, size_t size,
                               const val_t* __restrict__ input,
                               val_t* __restrict__ output) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const Material mat = device_materials[mesh.materials[i]];
    const val_t energy = input[i];
    val_t total = mat.external_flow;
    const int count = mesh.counts[i];
    #pragma unroll
    for (int j = 0; j < GRID_CONNECTIONS; ++j) {
        if (j < count) {
            const size_t edge = size_t(j) * size + i;
            total += (input[mesh.neighbors[edge]] - energy) *
                     mat.transfer_coeff * mesh.weights[edge] * 0.25;
        }
    }
    output[i] = energy + total;
    // Only the owning thread accesses its accumulated flux, so no second
    // flux buffer or atomics are necessary.
    mesh.flux[i] += fabs(total);
}

__global__ void packResults(DeviceMesh mesh, size_t size, const val_t* energy) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) mesh.result[i] = {energy[i], mesh.flux[i]};
}

void runSimulation(World& world, const int n_iters) {
    auto& mesh = world.device;
    const unsigned blocks = (world.size + 255) / 256;
    int iter = 0;
    // An even-sized graph can be replayed without changing ping-pong pointers.
    // This amortizes CPU submission overhead for long simulations.
    constexpr int batch = 32;
    if (n_iters >= 2 * batch) {
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int j = 0; j < batch; ++j) {
            updateElements<<<blocks, 256, 0, stream>>>(world.device, world.size,
                                                      mesh.energy[j & 1], mesh.energy[(j + 1) & 1]);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (; iter <= n_iters - batch; iter += batch)
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    for (; iter < n_iters; ++iter) {
        updateElements<<<blocks, 256>>>(mesh, world.size,
                                       mesh.energy[iter & 1], mesh.energy[(iter + 1) & 1]);
    }
    packResults<<<blocks, 256>>>(mesh, world.size, mesh.energy[std::max(n_iters, 0) & 1]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), mesh.result,
                          world.size * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    if (n_iters > 0 && (n_iters & 1)) std::swap(mesh.energy[0], mesh.energy[1]);
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
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        hash ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
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
    
    if (n_elems_root <= 0 || n_elems_root > 46340) {
        fprintf(stderr, "Grid size must be between 1 and 46340.\n");
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
    const size_t static_mem = world.size * (GRID_CONNECTIONS * (sizeof(idx_t) + sizeof(val_t)) + 2 * sizeof(unsigned char));
    const size_t dynamic_mem = world.size * (3 * sizeof(val_t) + sizeof(ElementDynamic));
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
    const double giga_elems_per_sec = (double(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;
    
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
