#include <algorithm>
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

// The square mesh has at most four connections. Store each connection slot
// separately so neighboring GPU threads access consecutive locations.
constexpr int MAX_CONNECTIONS = 4;
constexpr int BLOCK_SIZE = 256;

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
static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t));

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data) cudaFree(data); }

    void allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            fprintf(stderr, "Device allocation size overflow\n");
            std::exit(EXIT_FAILURE);
        }
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)),
                  "cudaMalloc");
    }
};

// All simulation state and connectivity remain on the GPU until reporting.
struct World {
    size_t n_elems = 0;
    unsigned int n_blocks = 0;
    int current = 0;
    DeviceBuffer<uint32_t> connected_idx;
    // buildSquare2D assigns the same flux coefficient to every edge. Store it
    // once, retaining the multiplication without reading redundant edge data.
    val_t connection_flux = 1.0;
    DeviceBuffer<uint8_t> num_connections;
    DeviceBuffer<uint8_t> material_idx;
    // Each allocation holds an energy array followed by a flux array.
    DeviceBuffer<val_t> state[2];
    std::vector<ElementDynamic> elements_dynamic;

    val_t* energy(int buffer) const { return state[buffer].data; }
    val_t* flux(int buffer) const { return state[buffer].data + n_elems; }
};

__constant__ Material device_materials[3];

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

__global__ void buildMeshKernel(size_t n_elems, uint32_t root,
                               uint32_t* __restrict__ connected_idx,
                               uint8_t* __restrict__ num_connections,
                               uint8_t* __restrict__ material_idx,
                               val_t* __restrict__ energy,
                               val_t* __restrict__ flux) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elems; i += size_t(blockDim.x) * gridDim.x) {
        const uint32_t x = i / root;
        const uint32_t y = i % root;
        uint8_t count = 0;
        // Append valid neighbors in exactly the reference's order.
        if (x + 1 < root) {
            connected_idx[size_t(count++) * n_elems + i] = i + root;
        }
        if (x > 0) {
            connected_idx[size_t(count++) * n_elems + i] = i - root;
        }
        if (y + 1 < root) {
            connected_idx[size_t(count++) * n_elems + i] = i + 1;
        }
        if (y > 0) {
            connected_idx[size_t(count++) * n_elems + i] = i - 1;
        }
        num_connections[i] = count;
        uint8_t material = DEFAULT_MAT_ID;
        if (i == 0) material = INFLOW_MAT_ID;
        if (i == root - 1) material = OUTFLOW_MAT_ID;
        if (i == n_elems - root) material = OUTFLOW_MAT_ID;
        if (i == n_elems - 1) material = INFLOW_MAT_ID;
        material_idx[i] = material;
        energy[i] = 0.0;
        flux[i] = 0.0;
    }
}

// Build connectivity and initialize the starting state in parallel on CUDA.
void buildSquare2D(World& world, const int n_elems_root) {
    world.n_elems = size_t(n_elems_root) * n_elems_root;
    int device = 0;
    cudaDeviceProp properties{};
    checkCuda(cudaGetDevice(&device), "cudaGetDevice");
    checkCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    // Enough resident work to hide memory latency, without oversized grids.
    world.n_blocks = static_cast<unsigned int>(std::min(
        (world.n_elems + BLOCK_SIZE - 1) / BLOCK_SIZE,
        size_t(properties.multiProcessorCount) * 64));

    const Material materials[] = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    checkCuda(cudaMemcpyToSymbol(device_materials, materials, sizeof(materials)),
              "copy materials");
    world.connected_idx.allocate(world.n_elems * MAX_CONNECTIONS);
    world.num_connections.allocate(world.n_elems);
    world.material_idx.allocate(world.n_elems);
    for (int i = 0; i < 2; ++i)
        world.state[i].allocate(2 * world.n_elems);

    buildMeshKernel<<<world.n_blocks, BLOCK_SIZE>>>(
        world.n_elems, n_elems_root, world.connected_idx.data,
        world.num_connections.data,
        world.material_idx.data, world.energy(0), world.flux(0));
    checkCuda(cudaGetLastError(), "build mesh kernel");
    checkCuda(cudaDeviceSynchronize(), "initialize mesh");
}

__global__ void updateKernel(size_t n_elems,
                             const uint32_t* __restrict__ connected_idx,
                             val_t connection_flux,
                             const uint8_t* __restrict__ num_connections,
                             const uint8_t* __restrict__ material_idx,
                             const val_t* __restrict__ energy,
                             const val_t* __restrict__ flux,
                             val_t* __restrict__ next_energy,
                             val_t* __restrict__ next_flux) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elems; i += size_t(blockDim.x) * gridDim.x) {
        const val_t current_energy = energy[i];
        const Material mat = device_materials[material_idx[i]];
        val_t total_flux = mat.external_flow;
        const unsigned int count = num_connections[i];
        #pragma unroll
        for (unsigned int j = 0; j < MAX_CONNECTIONS; ++j) {
            if (j < count) {
                const size_t connection = size_t(j) * n_elems + i;
                const val_t other_energy = energy[connected_idx[connection]];
                total_flux += (other_energy - current_energy) *
                              mat.transfer_coeff * connection_flux * 0.25;
            }
        }
        next_energy[i] = current_energy + total_flux;
        next_flux[i] = flux[i] + fabs(total_flux);
    }
}

void launchUpdate(const World& world, int current, cudaStream_t stream = nullptr) {
    updateKernel<<<world.n_blocks, BLOCK_SIZE, 0, stream>>>(
        world.n_elems, world.connected_idx.data, world.connection_flux,
        world.num_connections.data, world.material_idx.data,
        world.energy(current), world.flux(current),
        world.energy(current ^ 1), world.flux(current ^ 1));
}

// Ordered launches provide a global iteration boundary. CUDA graphs amortize
// launch overhead for long runs, while an even batch preserves buffer parity.
double runSimulation(World& world, const int n_iters) {
    constexpr int GRAPH_BATCH = 32;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaStream_t stream = nullptr;
    if (n_iters >= GRAPH_BATCH) {
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                  "create capture stream");
        checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                  "begin graph capture");
        for (int iter = 0; iter < GRAPH_BATCH; ++iter) {
            const int current = world.current ^ (iter & 1);
            launchUpdate(world, current, stream);
        }
        checkCuda(cudaStreamEndCapture(stream, &graph), "end graph capture");
        checkCuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                  "instantiate graph");
        checkCuda(cudaGraphUpload(executable, nullptr), "upload graph");
        checkCuda(cudaStreamSynchronize(nullptr), "finish graph upload");
    }

    cudaEvent_t start = nullptr, stop = nullptr;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");
    checkCuda(cudaEventRecord(start), "record start event");
    int iter = 0;
    if (executable) {
        for (; iter <= n_iters - GRAPH_BATCH; iter += GRAPH_BATCH)
            checkCuda(cudaGraphLaunch(executable, nullptr), "launch graph");
    }
    for (; iter < n_iters; ++iter) {
        launchUpdate(world, world.current);
        world.current ^= 1;
    }
    checkCuda(cudaGetLastError(), "update kernel");
    checkCuda(cudaEventRecord(stop), "record stop event");
    checkCuda(cudaEventSynchronize(stop), "finish simulation");
    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed time");
    checkCuda(cudaEventDestroy(start), "destroy start event");
    checkCuda(cudaEventDestroy(stop), "destroy stop event");
    if (executable) {
        checkCuda(cudaGraphExecDestroy(executable), "destroy executable graph");
        checkCuda(cudaGraphDestroy(graph), "destroy graph");
        checkCuda(cudaStreamDestroy(stream), "destroy capture stream");
    }
    return milliseconds;
}

__global__ void collectResultsKernel(size_t n_elems,
                                    const val_t* __restrict__ energy,
                                    const val_t* __restrict__ flux,
                                    ElementDynamic* __restrict__ results) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elems; i += size_t(blockDim.x) * gridDim.x) {
        results[i] = {energy[i], flux[i]};
    }
}

void collectResults(World& world) {
    world.elements_dynamic.resize(world.n_elems);
    // Reuse the inactive state buffer for packing; no extra GPU allocation or
    // host-side interleaving is needed for the final contiguous download.
    auto* results = reinterpret_cast<ElementDynamic*>(world.state[world.current ^ 1].data);
    collectResultsKernel<<<world.n_blocks, BLOCK_SIZE>>>(
        world.n_elems, world.energy(world.current), world.flux(world.current), results);
    checkCuda(cudaGetLastError(), "collect results kernel");
    checkCuda(cudaMemcpy(world.elements_dynamic.data(), results,
                         world.n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost),
              "copy final results");
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
    
    if (n_elems_root <= 0 ||
        int64_t(n_elems_root) * n_elems_root > std::numeric_limits<int>::max()) {
        fprintf(stderr, "Grid size must be positive with at most INT_MAX elements.\n");
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
    const size_t static_mem = world.n_elems *
        (MAX_CONNECTIONS * sizeof(uint32_t) + 2 * sizeof(uint8_t)) + sizeof(val_t);
    const size_t dynamic_mem = world.n_elems * 4 * sizeof(val_t);
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    // CUDA events measure the completed computation, excluding mesh setup and
    // the single final download used only for reporting and validation.
    const double duration_ms = runSimulation(world, n_iters);
    collectResults(world);
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const double time_per_iter = n_iters > 0 ? duration_ms / n_iters : 0.0;
    const double giga_elems_per_sec = duration_ms > 0.0 ?
        (double(std::max(n_iters, 0)) * n_elems) / (duration_ms / 1000.0) / 1e9 : 0.0;
    
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
