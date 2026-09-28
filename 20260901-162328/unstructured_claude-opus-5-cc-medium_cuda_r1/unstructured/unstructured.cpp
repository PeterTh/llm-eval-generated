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

// Abort on any CUDA error
#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        const cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(err_));                                       \
            exit(EXIT_FAILURE);                                                      \
        }                                                                            \
    } while (0)

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

// Dynamic state for all elements, kept as structure-of-arrays so that the host
// side mirrors the GPU layout and results can be transferred without repacking.
// The second (swap) energy buffer lives on the device only.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<val_t> current_energy;
    std::vector<val_t> total_flux;
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
    world.current_energy.resize(n_elems);
    world.total_flux.resize(n_elems);

    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.current_energy[i] = 0.0;
        world.total_flux[i] = 0.0;
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
__host__ __device__ inline val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

// Device-side structure-of-arrays mirror of the mesh.
// The per-connection arrays are stored transposed (connection-major, element-minor)
// so that neighbouring threads access consecutive addresses.
struct DeviceMesh {
    int n_elems = 0;
    Material* materials = nullptr;      // [n_materials]
    uint32_t* material_idx = nullptr;   // [n_elems]
    uint8_t* num_connections = nullptr; // [n_elems]
    uint32_t* connected_idx = nullptr;  // [MAX_CONNECTIONS * n_elems]
    val_t* connected_flux = nullptr;    // [MAX_CONNECTIONS * n_elems]
    val_t* energy[2] = {nullptr, nullptr};
    val_t* total_flux = nullptr;        // accumulated in place (element-local)
};

// Thread block size used for all kernel launches
constexpr int BLOCK_SIZE = 256;

// One thread per mesh element; grid-stride loop so the launch shape is independent
// of the mesh size. The per-element reduction is performed in the original order,
// which keeps the floating-point result bit-identical to the serial version.
__global__ __launch_bounds__(BLOCK_SIZE) void simulationKernel(
    const int n_elems,
    const val_t* __restrict__ energy_in,
    val_t* __restrict__ energy_out,
    val_t* __restrict__ total_flux,
    const uint32_t* __restrict__ material_idx,
    const uint8_t* __restrict__ num_connections,
    const uint32_t* __restrict__ connected_idx,
    const val_t* __restrict__ connected_flux,
    const Material* __restrict__ materials) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elems; i += stride) {
        const Material mat = materials[material_idx[i]];
        const val_t this_energy = energy_in[i];
        const int n_conn = num_connections[i];

        // Start with external flow
        val_t total = mat.external_flow;

        // Add flux from all connected elements
        #pragma unroll 4
        for (int j = 0; j < n_conn; ++j) {
            const uint32_t neighbor_idx = connected_idx[(size_t)j * n_elems + i];
            const val_t conn_flux = connected_flux[(size_t)j * n_elems + i];
            total += computeFlux(mat.transfer_coeff, this_energy, conn_flux,
                                 __ldg(&energy_in[neighbor_idx]));
        }

        // Update element state
        energy_out[i] = this_energy + total;
        total_flux[i] += fabs(total);
    }
}

// Upload the mesh to the GPU in a layout suited for coalesced access
static void uploadMesh(World& world, DeviceMesh& mesh) {
    const size_t n_elems = world.elements_static.size();
    mesh.n_elems = static_cast<int>(n_elems);

    // Flatten the array-of-structs mesh into structure-of-arrays staging buffers
    std::vector<uint32_t> h_material_idx(n_elems);
    std::vector<uint8_t> h_num_connections(n_elems);
    std::vector<uint32_t> h_connected_idx(MAX_CONNECTIONS * n_elems, 0);
    std::vector<val_t> h_connected_flux(MAX_CONNECTIONS * n_elems, 0.0);

    for (size_t i = 0; i < n_elems; ++i) {
        const ElementStatic& es = world.elements_static[i];
        h_material_idx[i] = static_cast<uint32_t>(es.material_idx);
        h_num_connections[i] = static_cast<uint8_t>(es.num_connections);
        for (idx_t j = 0; j < es.num_connections; ++j) {
            h_connected_idx[j * n_elems + i] = static_cast<uint32_t>(es.connected_idx[j]);
            h_connected_flux[j * n_elems + i] = es.connected_flux[j];
        }
    }

    CUDA_CHECK(cudaMalloc(&mesh.materials, world.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&mesh.material_idx, n_elems * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&mesh.num_connections, n_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&mesh.connected_idx, h_connected_idx.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&mesh.connected_flux, h_connected_flux.size() * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.energy[0], n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.energy[1], n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.total_flux, n_elems * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(mesh.materials, world.materials.data(),
                          world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.material_idx, h_material_idx.data(),
                          n_elems * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.num_connections, h_num_connections.data(),
                          n_elems * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.connected_idx, h_connected_idx.data(),
                          h_connected_idx.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.connected_flux, h_connected_flux.data(),
                          h_connected_flux.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.energy[0], world.current_energy.data(),
                          n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.total_flux, world.total_flux.data(),
                          n_elems * sizeof(val_t), cudaMemcpyHostToDevice));

    // Page-lock the result buffers so the final transfer runs at full PCIe speed
    CUDA_CHECK(cudaHostRegister(world.current_energy.data(), n_elems * sizeof(val_t),
                                cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(world.total_flux.data(), n_elems * sizeof(val_t),
                                cudaHostRegisterDefault));

    // Empty launch to force module loading before the timed region
    simulationKernel<<<1, BLOCK_SIZE>>>(0, mesh.energy[0], mesh.energy[1], mesh.total_flux,
                                        mesh.material_idx, mesh.num_connections,
                                        mesh.connected_idx, mesh.connected_flux, mesh.materials);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Copy the final state back into the host world representation
static void downloadState(World& world, const DeviceMesh& mesh, const int current_buffer) {
    const size_t n_elems = world.current_energy.size();
    CUDA_CHECK(cudaMemcpy(world.current_energy.data(), mesh.energy[current_buffer],
                          n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(world.total_flux.data(), mesh.total_flux,
                          n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
}

static void freeMesh(World& world, DeviceMesh& mesh) {
    CUDA_CHECK(cudaHostUnregister(world.current_energy.data()));
    CUDA_CHECK(cudaHostUnregister(world.total_flux.data()));
    CUDA_CHECK(cudaFree(mesh.materials));
    CUDA_CHECK(cudaFree(mesh.material_idx));
    CUDA_CHECK(cudaFree(mesh.num_connections));
    CUDA_CHECK(cudaFree(mesh.connected_idx));
    CUDA_CHECK(cudaFree(mesh.connected_flux));
    CUDA_CHECK(cudaFree(mesh.energy[0]));
    CUDA_CHECK(cudaFree(mesh.energy[1]));
    CUDA_CHECK(cudaFree(mesh.total_flux));
}

// Run simulation for n_iters iterations on the GPU.
// Note: total_flux is purely element-local (read and written at the same index), so a
// single accumulator array replaces the double buffering of the serial version.
void runSimulation(World& world, DeviceMesh& mesh, const int n_iters) {
    const int n_elems = mesh.n_elems;
    if (n_elems == 0) {
        return;
    }

    // One block per BLOCK_SIZE elements; the grid-stride loop in the kernel then
    // executes exactly one element per thread.
    const int grid_size = (n_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        simulationKernel<<<grid_size, BLOCK_SIZE>>>(
            n_elems, mesh.energy[cur], mesh.energy[cur ^ 1], mesh.total_flux,
            mesh.material_idx, mesh.num_connections, mesh.connected_idx,
            mesh.connected_flux, mesh.materials);
        cur ^= 1;
    }
    CUDA_CHECK(cudaGetLastError());

    downloadState(world, mesh, cur);
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = 0; i < world.current_energy.size(); ++i) {
        energy_sum += world.current_energy[i];
        flux_sum += world.total_flux[i];
        energy_max = std::max(world.current_energy[i], energy_max);
        energy_min = std::min(world.current_energy[i], energy_min);
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
    for (size_t i = 0; i < world.current_energy.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.current_energy[i]);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.total_flux[i]);
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
    // energy + accumulated flux per element, double buffered
    const size_t dynamic_mem = world.current_energy.size() * (2 * sizeof(val_t)) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Transfer the mesh to the GPU and warm up the CUDA context
    DeviceMesh mesh;
    CUDA_CHECK(cudaFree(nullptr));
    uploadMesh(world, mesh);

    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, mesh, n_iters);

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
        print_results(world.current_energy, "ElementEnergy");
    }
    
    // Validation
    bool valid = true;
    if (validate) {
        valid = validateResults(world);
    }

    freeMesh(world, mesh);

    return valid ? 0 : 1;
}
