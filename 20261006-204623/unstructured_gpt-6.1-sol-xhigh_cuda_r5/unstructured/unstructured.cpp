#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint32_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

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

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

// ELLPACK connectivity: consecutive threads access consecutive elements in
// each connection slot. Indices fit in 32 bits for every supported grid.
struct MeshView {
    idx_t* num_connections;
    idx_t* connected_idx;
    val_t* connected_flux;
    val_t* transfer_coeff;
    val_t* external_flow;
    size_t pitch;
    idx_t n_elems;
};

constexpr unsigned int BLOCK_SIZE = 256;
// Beyond four warps, the graph path hides double-precision instruction
// latency better than a single shared-memory block on throughput GPUs.
constexpr idx_t SMALL_MESH_LIMIT = 128;
constexpr int GRAPH_ITERS = 32; // Even, so each replay starts with the same buffer.

struct World {
    MeshView mesh{};
    idx_t* indices = nullptr;
    val_t* values = nullptr;
    val_t* energy[2]{};
    val_t* flux[2]{};
    int current = 0;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    std::vector<ElementDynamic> elements_dynamic;

    World() = default;
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    ~World() {
        // Also wait for outstanding work on an error path before releasing it.
        if (stream) cudaStreamSynchronize(stream);
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        if (values) cudaFree(values);
        if (indices) cudaFree(indices);
        if (stream) cudaStreamDestroy(stream);
    }
};

// Build precisely the original mesh on the GPU, including the order of
// neighbors and of corner material assignments (important for a 1x1 grid).
__global__ void initializeMesh(MeshView mesh, idx_t root, Material default_mat,
                               Material inflow_mat, Material outflow_mat,
                               val_t* energy0, val_t* flux0,
                               val_t* energy1, val_t* flux1) {
    const idx_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mesh.n_elems) return;
    const idx_t x = i / root;
    const idx_t y = i % root;
    idx_t count = 0;
    if (x + 1 < root) mesh.connected_idx[count++ * mesh.pitch + i] = i + root;
    if (x > 0)        mesh.connected_idx[count++ * mesh.pitch + i] = i - root;
    if (y + 1 < root) mesh.connected_idx[count++ * mesh.pitch + i] = i + 1;
    if (y > 0)        mesh.connected_idx[count++ * mesh.pitch + i] = i - 1;
    for (idx_t j = 0; j < count; ++j) {
        mesh.connected_flux[j * mesh.pitch + i] = 1.0;
    }
    mesh.num_connections[i] = count;

    Material mat = default_mat;
    if (i == 0) mat = inflow_mat;
    if (i == root - 1) mat = outflow_mat;
    if (i == mesh.n_elems - root) mat = outflow_mat;
    if (i == mesh.n_elems - 1) mat = inflow_mat;
    mesh.transfer_coeff[i] = mat.transfer_coeff;
    mesh.external_flow[i] = mat.external_flow;
    energy0[i] = energy1[i] = 0.0;
    flux0[i] = flux1[i] = 0.0;
}

void buildSquare2D(World& world, const int n_elems_root) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&world.stream, cudaStreamNonBlocking));
    world.mesh.n_elems = static_cast<idx_t>(n_elems_root) * n_elems_root;
    // Align every array, including small or non-power-of-two grids.
    world.mesh.pitch = (static_cast<size_t>(world.mesh.n_elems) + 31) & ~size_t{31};
    const size_t pitch = world.mesh.pitch;
    CUDA_CHECK(cudaMalloc(&world.indices, (MAX_CONNECTIONS + 1) * pitch * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&world.values, (MAX_CONNECTIONS + 6) * pitch * sizeof(val_t)));
    world.mesh.num_connections = world.indices;
    world.mesh.connected_idx = world.indices + pitch;
    world.mesh.connected_flux = world.values;
    world.mesh.transfer_coeff = world.values + MAX_CONNECTIONS * pitch;
    world.mesh.external_flow = world.mesh.transfer_coeff + pitch;
    world.energy[0] = world.mesh.external_flow + pitch;
    world.flux[0] = world.energy[0] + pitch;
    world.energy[1] = world.flux[0] + pitch;
    world.flux[1] = world.energy[1] + pitch;

    const unsigned int blocks = (world.mesh.n_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeMesh<<<blocks, BLOCK_SIZE, 0, world.stream>>>(
        world.mesh, n_elems_root, Material{0.8, 0.0}, Material{0.8, 0.5},
        Material{0.8, -0.5}, world.energy[0], world.flux[0], world.energy[1], world.flux[1]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(world.stream));
}

// No atomics or in-place updates: each thread owns one destination element,
// and all neighbors are read from the preceding iteration's energy buffer.
__global__ void updateElements(MeshView mesh,
                               const val_t* __restrict__ energy,
                               const val_t* __restrict__ flux,
                               val_t* __restrict__ next_energy,
                               val_t* __restrict__ next_flux) {
    const idx_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mesh.n_elems) return;
    const val_t current_energy = energy[i];
    const val_t transfer_coeff = mesh.transfer_coeff[i];
    val_t total_flux = mesh.external_flow[i];
    const idx_t count = mesh.num_connections[i];
#pragma unroll
    for (idx_t j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j < count) {
            const size_t connection = j * mesh.pitch + i;
            total_flux += (energy[mesh.connected_idx[connection]] - current_energy) *
                          transfer_coeff * mesh.connected_flux[connection] * 0.25;
        }
    }
    next_energy[i] = current_energy + total_flux;
    next_flux[i] = flux[i] + fabs(total_flux);
}

// When the entire mesh fits in one block, shared memory provides the global
// iteration barrier. Keep connectivity and accumulated flux in registers and
// run all iterations in one launch rather than paying for many tiny kernels.
__global__ void updateSmallMesh(MeshView mesh, int n_iters,
                                const val_t* energy, const val_t* flux,
                                val_t* next_energy, val_t* next_flux) {
    extern __shared__ val_t shared_energy[];
    const idx_t i = threadIdx.x;
    const bool active = i < mesh.n_elems;
    const idx_t count = active ? mesh.num_connections[i] : 0;
    const val_t transfer_coeff = active ? mesh.transfer_coeff[i] : 0.0;
    const val_t external_flow = active ? mesh.external_flow[i] : 0.0;
    idx_t neighbors[MAX_CONNECTIONS]{};
    val_t weights[MAX_CONNECTIONS]{};
#pragma unroll
    for (idx_t j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j < count) {
            neighbors[j] = mesh.connected_idx[j * mesh.pitch + i];
            weights[j] = mesh.connected_flux[j * mesh.pitch + i];
        }
    }
    val_t current_energy = active ? energy[i] : 0.0;
    val_t accumulated_flux = active ? flux[i] : 0.0;
    if (active) shared_energy[i] = current_energy;
    __syncthreads();

    val_t* read_energy = shared_energy;
    val_t* write_energy = shared_energy + mesh.n_elems;
    for (int iter = 0; iter < n_iters; ++iter) {
        if (active) {
            val_t total_flux = external_flow;
#pragma unroll
            for (idx_t j = 0; j < MAX_CONNECTIONS; ++j) {
                if (j < count) {
                    total_flux += (read_energy[neighbors[j]] - current_energy) *
                                  transfer_coeff * weights[j] * 0.25;
                }
            }
            current_energy += total_flux;
            accumulated_flux += fabs(total_flux);
            write_energy[i] = current_energy;
        }
        __syncthreads();
        val_t* old_read = read_energy;
        read_energy = write_energy;
        write_energy = old_read;
    }
    if (active) {
        next_energy[i] = current_energy;
        next_flux[i] = accumulated_flux;
    }
}

void launchIteration(World& world, int current) {
    const unsigned int blocks = (world.mesh.n_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
    updateElements<<<blocks, BLOCK_SIZE, 0, world.stream>>>(
        world.mesh, world.energy[current], world.flux[current],
        world.energy[current ^ 1], world.flux[current ^ 1]);
}

// Amortize CPU launch overhead in long runs. Capture a bounded batch instead
// of constructing a graph whose memory usage grows with the iteration count.
void prepareSimulation(World& world, const int n_iters) {
    if (n_iters < 256 || world.mesh.n_elems <= SMALL_MESH_LIMIT) return;
    CUDA_CHECK(cudaStreamBeginCapture(world.stream, cudaStreamCaptureModeThreadLocal));
    for (int iter = 0; iter < GRAPH_ITERS; ++iter) {
        launchIteration(world, world.current ^ (iter & 1));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(world.stream, &world.graph));
    CUDA_CHECK(cudaGraphInstantiate(&world.graph_exec, world.graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphUpload(world.graph_exec, world.stream));
    CUDA_CHECK(cudaStreamSynchronize(world.stream));
}

void runSimulation(World& world, const int n_iters) {
    if (world.mesh.n_elems <= SMALL_MESH_LIMIT && n_iters > 1) {
        const int next = world.current ^ (n_iters & 1);
        const unsigned int threads = ((world.mesh.n_elems + 31) / 32) * 32;
        updateSmallMesh<<<1, threads, 2 * world.mesh.n_elems * sizeof(val_t), world.stream>>>(
            world.mesh, n_iters, world.energy[world.current], world.flux[world.current],
            world.energy[next], world.flux[next]);
        world.current = next;
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(world.stream));
        return;
    }
    const int batches = world.graph_exec ? std::max(n_iters, 0) / GRAPH_ITERS : 0;
    for (int batch = 0; batch < batches; ++batch) {
        CUDA_CHECK(cudaGraphLaunch(world.graph_exec, world.stream));
    }
    for (int iter = batches * GRAPH_ITERS; iter < n_iters; ++iter) {
        launchIteration(world, world.current);
        world.current ^= 1;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(world.stream));
}

__global__ void packResults(idx_t n_elems, const val_t* energy, const val_t* flux,
                            ElementDynamic* results) {
    const idx_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) results[i] = ElementDynamic{energy[i], flux[i]};
}

void collectResults(World& world) {
    world.elements_dynamic.resize(world.mesh.n_elems);
    // The inactive state contains two contiguous padded arrays, enough room
    // for the packed output. Reuse it instead of allocating another buffer.
    auto* results = reinterpret_cast<ElementDynamic*>(world.energy[world.current ^ 1]);
    const unsigned int blocks = (world.mesh.n_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
    packResults<<<blocks, BLOCK_SIZE, 0, world.stream>>>(
        world.mesh.n_elems, world.energy[world.current], world.flux[world.current], results);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(world.elements_dynamic.data(), results,
        world.elements_dynamic.size() * sizeof(ElementDynamic), cudaMemcpyDeviceToHost, world.stream));
    CUDA_CHECK(cudaStreamSynchronize(world.stream));
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
        memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
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

int main(int argc, char** argv) try {
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
        static_cast<uint64_t>(n_elems_root) * n_elems_root > std::numeric_limits<int>::max()) {
        fprintf(stderr, "Grid size must be positive and NxN must fit in a 32-bit signed integer.\n");
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
    const size_t static_mem = world.mesh.pitch *
        ((MAX_CONNECTIONS + 1) * sizeof(idx_t) + (MAX_CONNECTIONS + 2) * sizeof(val_t));
    const size_t dynamic_mem = world.mesh.pitch * sizeof(val_t) * 4;
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");
    
    prepareSimulation(world, n_iters);

    // Time completed GPU computation. Setup and the single result transfer
    // are outside the simulation interval, as are hashing and validation.
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters, 0);
    const double time_per_iter = n_measured_iters ? duration_ms / n_measured_iters : 0.0;
    const double giga_elems_per_sec = duration_ms > 0.0 ?
        (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9 : 0.0;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    collectResults(world);

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
} catch (const std::exception& error) {
    fprintf(stderr, "Error: %s\n", error.what());
    return 1;
}
