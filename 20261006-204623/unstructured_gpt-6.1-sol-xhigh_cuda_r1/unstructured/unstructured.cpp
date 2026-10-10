#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Connectivity is stored slot-major (ELL layout), so neighboring CUDA threads
// read consecutive indices and coefficients even for an unstructured mesh.
using idx_t = uint32_t;
using val_t = double;
constexpr int GRID_CONNECTIONS = 4;
constexpr int BLOCK_SIZE = 256;
constexpr int GRAPH_ITERATIONS = 64;
constexpr size_t MATERIAL_STORAGE_BYTES = 256;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

constexpr uint8_t DEFAULT_MAT_ID = 0;
constexpr uint8_t INFLOW_MAT_ID = 1;
constexpr uint8_t OUTFLOW_MAT_ID = 2;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation)

struct World {
    size_t n_elems = 0;
    size_t pitch = 0;
    size_t static_mem = 0;
    size_t dynamic_mem = 0;
    int blocks = 0;
    int graph_iters = 0;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable_graph = nullptr;
    void* device_memory = nullptr;
    Material* materials = nullptr;
    uint8_t* material_idx = nullptr;
    uint8_t* num_connections = nullptr;
    idx_t* connected_idx = nullptr;
    val_t* connected_flux = nullptr;
    val_t* energy[2] = {nullptr, nullptr};
    val_t* total_flux = nullptr;
    const val_t* final_energy = nullptr;
    std::vector<val_t> host_energy;
    std::vector<val_t> host_flux;

    World() = default;
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    ~World() {
        // Also release partially constructed worlds after a CUDA error.
        if (stream) cudaStreamSynchronize(stream);
        if (executable_graph) cudaGraphExecDestroy(executable_graph);
        if (graph) cudaGraphDestroy(graph);
        if (device_memory) cudaFree(device_memory);
        if (stream) cudaStreamDestroy(stream);
    }
};

__global__ void initializeMesh(idx_t n_elems, int root, size_t pitch,
                               Material* materials, uint8_t* material_idx,
                               uint8_t* num_connections, idx_t* connected_idx,
                               val_t* connected_flux, val_t* energy,
                               val_t* total_flux) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        materials[DEFAULT_MAT_ID] = Material{0.8, 0.0};
        materials[INFLOW_MAT_ID] = Material{0.8, 0.5};
        materials[OUTFLOW_MAT_ID] = Material{0.8, -0.5};
    }
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         i < n_elems; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int x = static_cast<int>(i / root);
        const int y = static_cast<int>(i % root);
        uint8_t count = 0;
        // Preserve the original +x, -x, +y, -y connection order.
        if (x + 1 < root) {
            connected_idx[count * pitch + i] = static_cast<idx_t>(i + root);
            connected_flux[count++ * pitch + i] = 1.0;
        }
        if (x > 0) {
            connected_idx[count * pitch + i] = static_cast<idx_t>(i - root);
            connected_flux[count++ * pitch + i] = 1.0;
        }
        if (y + 1 < root) {
            connected_idx[count * pitch + i] = static_cast<idx_t>(i + 1);
            connected_flux[count++ * pitch + i] = 1.0;
        }
        if (y > 0) {
            connected_idx[count * pitch + i] = static_cast<idx_t>(i - 1);
            connected_flux[count++ * pitch + i] = 1.0;
        }
        num_connections[i] = count;
        uint8_t material = DEFAULT_MAT_ID;
        // Ordered assignments also preserve the overlapping corners for root=1.
        if (i == 0) material = INFLOW_MAT_ID;
        if (i == static_cast<size_t>(root - 1)) material = OUTFLOW_MAT_ID;
        if (i == static_cast<size_t>(root - 1) * root) material = OUTFLOW_MAT_ID;
        if (i == static_cast<size_t>(n_elems - 1)) material = INFLOW_MAT_ID;
        material_idx[i] = material;
        energy[i] = 0.0;
        total_flux[i] = 0.0;
    }
}

__global__ void updateElements(idx_t n_elems, size_t pitch,
                               const Material* __restrict__ materials,
                               const uint8_t* __restrict__ material_idx,
                               const uint8_t* __restrict__ num_connections,
                               const idx_t* __restrict__ connected_idx,
                               const val_t* __restrict__ connected_flux,
                               const val_t* __restrict__ energy,
                               val_t* __restrict__ next_energy,
                               val_t* __restrict__ total_flux) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         i < n_elems; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const val_t current = energy[i];
        const Material mat = materials[material_idx[i]];
        val_t flux = mat.external_flow;
        const int count = num_connections[i];
        #pragma unroll
        for (int j = 0; j < GRID_CONNECTIONS; ++j) {
            if (j < count) {
                const size_t slot = j * pitch + i;
                flux += (energy[connected_idx[slot]] - current) *
                        mat.transfer_coeff * connected_flux[slot] * 0.25;
            }
        }
        next_energy[i] = current + flux;
        // No neighbor reads accumulated flux, so it needs only one buffer.
        total_flux[i] += fabs(flux);
    }
}

void launchUpdate(World& world, const val_t* energy, val_t* next_energy) {
    updateElements<<<world.blocks, BLOCK_SIZE, 0, world.stream>>>(
        static_cast<idx_t>(world.n_elems), world.pitch, world.materials,
        world.material_idx, world.num_connections, world.connected_idx,
        world.connected_flux, energy, next_energy, world.total_flux);
}

// Build the original square mesh directly on the GPU, without a host copy of
// its connectivity or a serial preprocessing pass.
void buildSquare2D(World& world, const int n_elems_root) {
    world.n_elems = static_cast<size_t>(n_elems_root) * n_elems_root;
    world.pitch = (world.n_elems + 31) & ~size_t{31};
    world.static_mem = MATERIAL_STORAGE_BYTES + world.pitch *
        (GRID_CONNECTIONS * (sizeof(idx_t) + sizeof(val_t)) + 2 * sizeof(uint8_t));
    world.dynamic_mem = 3 * world.pitch * sizeof(val_t);
    CUDA_CHECK(cudaStreamCreateWithFlags(&world.stream, cudaStreamNonBlocking));
    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    world.blocks = static_cast<int>(std::min(
        (world.n_elems + BLOCK_SIZE - 1) / BLOCK_SIZE,
        static_cast<size_t>(properties.multiProcessorCount) * 32));
    CUDA_CHECK(cudaMalloc(&world.device_memory, world.static_mem + world.dynamic_mem));
    auto* cursor = static_cast<unsigned char*>(world.device_memory);
    world.materials = reinterpret_cast<Material*>(cursor);
    // Align all slot and state arrays to full memory-transaction boundaries.
    cursor += MATERIAL_STORAGE_BYTES;
    for (auto& energy : world.energy) {
        energy = reinterpret_cast<val_t*>(cursor);
        cursor += world.pitch * sizeof(val_t);
    }
    world.total_flux = reinterpret_cast<val_t*>(cursor);
    cursor += world.pitch * sizeof(val_t);
    world.connected_flux = reinterpret_cast<val_t*>(cursor);
    cursor += GRID_CONNECTIONS * world.pitch * sizeof(val_t);
    world.connected_idx = reinterpret_cast<idx_t*>(cursor);
    cursor += GRID_CONNECTIONS * world.pitch * sizeof(idx_t);
    world.material_idx = cursor;
    cursor += world.pitch;
    world.num_connections = cursor;
    world.final_energy = world.energy[0];
    initializeMesh<<<world.blocks, BLOCK_SIZE, 0, world.stream>>>(
        static_cast<idx_t>(world.n_elems), n_elems_root, world.pitch,
        world.materials, world.material_idx, world.num_connections,
        world.connected_idx, world.connected_flux, world.energy[0], world.total_flux);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(world.stream));
}

void prepareSimulation(World& world, int n_iters) {
    world.graph_iters = std::min(std::max(n_iters, 0), GRAPH_ITERATIONS);
    if (world.graph_iters == 0) return;
    // An even batch ends in the same energy buffer it starts in, so the graph
    // can be replayed without changing kernel arguments. Capture also provides
    // the global dependency between consecutive Jacobi iterations.
    CUDA_CHECK(cudaStreamBeginCapture(world.stream, cudaStreamCaptureModeThreadLocal));
    for (int iter = 0; iter < world.graph_iters; ++iter) {
        launchUpdate(world, world.energy[iter & 1], world.energy[(iter + 1) & 1]);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(world.stream, &world.graph));
    CUDA_CHECK(cudaGraphInstantiate(&world.executable_graph, world.graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphUpload(world.executable_graph, world.stream));
    CUDA_CHECK(cudaStreamSynchronize(world.stream));
}

void runSimulation(World& world, const int n_iters) {
    if (world.graph_iters == 0) return;
    val_t* energy = world.energy[0];
    val_t* next_energy = world.energy[1];
    const int batches = n_iters / world.graph_iters;
    for (int batch = 0; batch < batches; ++batch) {
        CUDA_CHECK(cudaGraphLaunch(world.executable_graph, world.stream));
    }
    // Odd-sized graphs are used only when the entire run fits in one batch.
    if (world.graph_iters & 1) std::swap(energy, next_energy);
    for (int iter = 0; iter < n_iters % world.graph_iters; ++iter) {
        launchUpdate(world, energy, next_energy);
        std::swap(energy, next_energy);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(world.stream));
    world.final_energy = energy;
}

void downloadResults(World& world) {
    world.host_energy.resize(world.n_elems);
    world.host_flux.resize(world.n_elems);
    CUDA_CHECK(cudaMemcpyAsync(world.host_energy.data(), world.final_energy,
                               world.n_elems * sizeof(val_t), cudaMemcpyDeviceToHost,
                               world.stream));
    CUDA_CHECK(cudaMemcpyAsync(world.host_flux.data(), world.total_flux,
                               world.n_elems * sizeof(val_t), cudaMemcpyDeviceToHost,
                               world.stream));
    CUDA_CHECK(cudaStreamSynchronize(world.stream));
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < world.n_elems; ++i) {
        energy_sum += world.host_energy[i];
        flux_sum += world.host_flux[i];
        energy_max = std::max(world.host_energy[i], energy_max);
        energy_min = std::min(world.host_energy[i], energy_min);
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
    for (size_t i = 0; i < world.n_elems; ++i) {
        uint64_t energy_bits, flux_bits;
        // memcpy preserves the bitwise hash without violating strict aliasing.
        std::memcpy(&energy_bits, &world.host_energy[i], sizeof(energy_bits));
        std::memcpy(&flux_bits, &world.host_flux[i], sizeof(flux_bits));
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

    if (n_elems_root <= 0 ||
        static_cast<uint64_t>(n_elems_root) * n_elems_root >
            static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "Grid size must be positive and NxN must fit in a signed int.\n");
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;

    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    try {
        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
        World world;
        buildSquare2D(world, n_elems_root);

        // Calculate memory usage
        const size_t static_mem = world.static_mem;
        const size_t dynamic_mem = world.dynamic_mem;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        prepareSimulation(world, n_iters);

        // Run simulation
        printf("Running simulation...\n");
        auto start = std::chrono::high_resolution_clock::now();

        runSimulation(world, n_iters);

        auto end = std::chrono::high_resolution_clock::now();
        const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

        printf("Computation time: %.3f ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters, 0);
        const double time_per_iter = duration_ms / std::max(n_measured_iters, 1);
        const double giga_elems_per_sec = duration_ms > 0.0 ?
            (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms * 1e6) : 0.0;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Compute hash for verification
        downloadResults(world);
        const uint64_t hash = computeHash(world);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            print_results(world.host_energy, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(world);
            if (!valid) {
                return 1;
            }
        }

        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
