#include <algorithm>
#include <array>
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
// Grid element counts are bounded by INT_MAX, so 32-bit indices are sufficient
// and reduce both connectivity bandwidth and GPU register pressure.
using idx_t = uint32_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Slot-major connectivity: neighboring GPU lanes read neighboring addresses.
// Only the slots needed by the mesh are allocated (at most MAX_CONNECTIONS).
struct ElementStatic {
    std::vector<idx_t> material_idx;
    std::vector<unsigned> num_connections;
    std::array<std::vector<idx_t>, MAX_CONNECTIONS> connected_idx;
    std::array<std::vector<val_t>, MAX_CONNECTIONS> connected_flux;
    unsigned slots = 4;
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state
struct World {
    std::vector<Material> materials;
    ElementStatic elements_static;
    std::vector<ElementDynamic> elements_dynamic;
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
    auto& mesh = world.elements_static;
    mesh.material_idx.resize(n_elems, DEFAULT_MAT_ID);
    mesh.num_connections.resize(n_elems, 0);
    for (unsigned slot = 0; slot < mesh.slots; ++slot) {
        mesh.connected_idx[slot].resize(n_elems);
        mesh.connected_flux[slot].resize(n_elems);
    }
    world.elements_dynamic.resize(n_elems);
    
    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_elems; ++i) {
        mesh.material_idx[i] = DEFAULT_MAT_ID;
        mesh.num_connections[i] = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    const unsigned slot = mesh.num_connections[idx]++;
                    mesh.connected_idx[slot][idx] = neighbor_idx;
                    mesh.connected_flux[slot][idx] = 1.0;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    mesh.material_idx[0 * n_elems_root + 0] = INFLOW_MAT_ID;
    mesh.material_idx[0 * n_elems_root + last] = OUTFLOW_MAT_ID;
    mesh.material_idx[last * n_elems_root + 0] = OUTFLOW_MAT_ID;
    mesh.material_idx[last * n_elems_root + last] = INFLOW_MAT_ID;
}

static void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

struct DeviceMesh {
    const Material* materials;
    const idx_t* material_idx;
    const unsigned* num_connections;
    const idx_t* connected_idx;
    const val_t* connected_flux;
};

__global__ void unpackState(const ElementDynamic* state, val_t* energy,
                            val_t* flux, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        energy[i] = state[i].current_energy;
        flux[i] = state[i].total_flux;
    }
}

__global__ void packState(const val_t* energy, const val_t* flux,
                          ElementDynamic* state, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        state[i] = ElementDynamic{energy[i], flux[i]};
    }
}

__global__ void updateElements(DeviceMesh mesh, const val_t* __restrict__ energy,
                               const val_t* __restrict__ flux,
                               val_t* __restrict__ next_energy,
                               val_t* __restrict__ next_flux, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const Material mat = mesh.materials[mesh.material_idx[i]];
        const val_t current = energy[i];
        val_t total = mat.external_flow;
        // Keep the original per-element accumulation order, including borders.
        for (unsigned slot = 0; slot < mesh.num_connections[i]; ++slot) {
            const size_t offset = size_t(slot) * count + i;
            total += (energy[mesh.connected_idx[offset]] - current) *
                     mat.transfer_coeff * mesh.connected_flux[offset] * 0.25;
        }
        next_energy[i] = current + total;
        next_flux[i] = flux[i] + fabs(total);
    }
}

// All iterations run on CUDA, with no host synchronization between updates.
void runSimulation(World& world, const int n_iters) {
    const size_t count = world.elements_dynamic.size();
    const auto& mesh = world.elements_static;
    const size_t values_bytes = count * sizeof(val_t);
    const size_t indices_bytes = count * sizeof(idx_t);
    const size_t materials_bytes = world.materials.size() * sizeof(Material);
    const size_t mesh_bytes = materials_bytes + indices_bytes +
        mesh.slots * (indices_bytes + values_bytes) + count * sizeof(unsigned);

    // A single allocation for immutable connectivity and one for four state
    // arrays avoid allocation overhead and keep the live GPU working set small.
    unsigned char* storage = nullptr;
    val_t* states = nullptr;
    checkCuda(cudaMalloc(&storage, mesh_bytes));
    checkCuda(cudaMalloc(&states, 4 * values_bytes));
    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto* materials = reinterpret_cast<Material*>(storage);
    // Place all doubles first to preserve alignment even for odd element counts.
    auto* connected_flux = reinterpret_cast<val_t*>(storage + materials_bytes);
    auto* material_idx = reinterpret_cast<idx_t*>(connected_flux + mesh.slots * count);
    auto* connected_idx = material_idx + count;
    auto* num_connections = reinterpret_cast<unsigned*>(connected_idx + mesh.slots * count);
    checkCuda(cudaMemcpyAsync(materials, world.materials.data(), materials_bytes,
                              cudaMemcpyHostToDevice, stream));
    checkCuda(cudaMemcpyAsync(material_idx, mesh.material_idx.data(), indices_bytes,
                              cudaMemcpyHostToDevice, stream));
    checkCuda(cudaMemcpyAsync(num_connections, mesh.num_connections.data(), count * sizeof(unsigned),
                              cudaMemcpyHostToDevice, stream));
    for (unsigned slot = 0; slot < mesh.slots; ++slot) {
        checkCuda(cudaMemcpyAsync(connected_idx + slot * count, mesh.connected_idx[slot].data(),
                                  indices_bytes, cudaMemcpyHostToDevice, stream));
        checkCuda(cudaMemcpyAsync(connected_flux + slot * count, mesh.connected_flux[slot].data(),
                                  values_bytes, cudaMemcpyHostToDevice, stream));
    }

    val_t* energy = states;
    val_t* flux = states + count;
    val_t* next_energy = states + 2 * count;
    val_t* next_flux = states + 3 * count;
    // The inactive pair doubles as interleaved transfer staging storage.
    auto* staging = reinterpret_cast<ElementDynamic*>(next_energy);
    checkCuda(cudaMemcpyAsync(staging, world.elements_dynamic.data(), 2 * values_bytes,
                              cudaMemcpyHostToDevice, stream));
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
    unpackState<<<blocks, threads, 0, stream>>>(staging, energy, flux, count);
    const DeviceMesh device_mesh{materials, material_idx, num_connections,
                                 connected_idx, connected_flux};
    auto enqueue_update = [&]() {
        updateElements<<<blocks, threads, 0, stream>>>(device_mesh, energy, flux,
                                                      next_energy, next_flux, count);
        std::swap(energy, next_energy);
        std::swap(flux, next_flux);
    };
    // An even-length reusable graph keeps both ping-pong pointers unchanged
    // between replays and amortizes CPU launch costs for long simulations.
    constexpr int graph_iters = 64;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    int remaining = n_iters;
    if (remaining >= 2 * graph_iters) {
        checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int iter = 0; iter < graph_iters; ++iter) {
            enqueue_update();
        }
        checkCuda(cudaStreamEndCapture(stream, &graph));
        checkCuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        while (remaining >= graph_iters) {
            checkCuda(cudaGraphLaunch(executable, stream));
            remaining -= graph_iters;
        }
    }
    for (int iter = 0; iter < remaining; ++iter) {
        enqueue_update();
    }
    staging = reinterpret_cast<ElementDynamic*>(next_energy);
    packState<<<blocks, threads, 0, stream>>>(energy, flux, staging, count);
    checkCuda(cudaGetLastError());
    checkCuda(cudaMemcpyAsync(world.elements_dynamic.data(), staging, 2 * values_bytes,
                              cudaMemcpyDeviceToHost, stream));
    checkCuda(cudaStreamSynchronize(stream));
    if (executable) {
        checkCuda(cudaGraphExecDestroy(executable));
        checkCuda(cudaGraphDestroy(graph));
    }
    checkCuda(cudaStreamDestroy(stream));
    checkCuda(cudaFree(states));
    checkCuda(cudaFree(storage));
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
    
    if (n_elems_root <= 0 || n_elems_root > 46340 || n_iters < 0) {
        fprintf(stderr, "Grid size must be in [1, 46340] and iterations must be nonnegative.\n");
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
    const size_t static_mem = size_t(n_elems) * (sizeof(idx_t) + sizeof(unsigned) +
        world.elements_static.slots * (sizeof(idx_t) + sizeof(val_t)));
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic);
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    // Initialize the CUDA context before measuring simulation work.
    checkCuda(cudaFree(nullptr));
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = n_iters;
    const double time_per_iter = duration_ms / std::max(n_measured_iters, 1);
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
