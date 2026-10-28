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

// CUDA error checking helper
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Device-side index type (element counts are checked to fit in 32 bits)
using dev_idx_t = uint32_t;

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

// Device-resident mesh: static connectivity in SoA layout (allocated once) and
// double-buffered dynamic state (SoA: energy and total flux)
struct DeviceMesh {
    size_t n_elems = 0;
    size_t max_conn = 0;
    dev_idx_t* mat_idx = nullptr;     // [n_elems]
    dev_idx_t* num_conn = nullptr;    // [n_elems]
    dev_idx_t* conn_idx = nullptr;    // [max_conn][n_elems]
    val_t* conn_flux = nullptr;       // [max_conn][n_elems]
    Material* materials = nullptr;    // [n_materials]
    val_t* energy[2] = {nullptr, nullptr};
    val_t* flux[2] = {nullptr, nullptr};
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

// One Jacobi-style update step: each thread owns one element and reads only
// the previous-step state (e_in/f_in), writing to (e_out/f_out). The per-element
// arithmetic order matches the original serial implementation exactly:
//   total = external_flow; for each neighbor: total += (other - this) * coeff * flux * 0.25
// Connectivity is laid out so that threads sharing a connection slot access
// consecutive addresses (coalesced).
__global__ void stepKernel(const dev_idx_t* __restrict__ mat_idx,
                           const dev_idx_t* __restrict__ num_conn,
                           const dev_idx_t* __restrict__ conn_idx,
                           const val_t* __restrict__ conn_flux,
                           const Material* __restrict__ materials,
                           const val_t* __restrict__ e_in,
                           const val_t* __restrict__ f_in,
                           val_t* __restrict__ e_out,
                           val_t* __restrict__ f_out,
                           size_t n_elems) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) {
        return;
    }

    const Material mat = materials[mat_idx[i]];
    const val_t e_self = e_in[i];
    const val_t f_self = f_in[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    const size_t n_conn = num_conn[i];
    for (size_t j = 0; j < n_conn; ++j) {
        const size_t slot = j * n_elems + i;
        const size_t neighbor = conn_idx[slot];
        const val_t e_other = e_in[neighbor];
        total_flux += (e_other - e_self) * mat.transfer_coeff * conn_flux[slot] * 0.25;
    }

    e_out[i] = e_self + total_flux;
    f_out[i] = f_self + fabs(total_flux);
}

// Build the device-resident mesh from the host world (static data only; dynamic
// buffers are allocated here and initialized in runSimulation)
DeviceMesh uploadMesh(const World& world) {
    DeviceMesh mesh;
    mesh.n_elems = world.elements_static.size();
    if (mesh.n_elems == 0) {
        return mesh;
    }
    const size_t n_elems = mesh.n_elems;

    if (n_elems > std::numeric_limits<dev_idx_t>::max()) {
        fprintf(stderr, "Mesh too large for 32-bit device indices (%zu elements)\n", n_elems);
        exit(EXIT_FAILURE);
    }

    // Repack static data into SoA host arrays, sized to the actual maximum
    // connection count (rather than MAX_CONNECTIONS) to minimize transfer volume
    size_t max_conn = 1;
    for (const auto& es : world.elements_static) {
        max_conn = std::max(max_conn, static_cast<size_t>(es.num_connections));
    }
    mesh.max_conn = max_conn;

    std::vector<dev_idx_t> h_mat_idx(n_elems);
    std::vector<dev_idx_t> h_num_conn(n_elems);
    std::vector<dev_idx_t> h_conn_idx(n_elems * max_conn, 0);
    std::vector<val_t> h_conn_flux(n_elems * max_conn, 0.0);
    for (size_t i = 0; i < n_elems; ++i) {
        const ElementStatic& es = world.elements_static[i];
        h_mat_idx[i] = static_cast<dev_idx_t>(es.material_idx);
        h_num_conn[i] = static_cast<dev_idx_t>(es.num_connections);
        for (size_t j = 0; j < es.num_connections; ++j) {
            h_conn_idx[j * n_elems + i] = static_cast<dev_idx_t>(es.connected_idx[j]);
            h_conn_flux[j * n_elems + i] = es.connected_flux[j];
        }
    }

    // Device allocations
    CUDA_CHECK(cudaMalloc(&mesh.mat_idx, n_elems * sizeof(dev_idx_t)));
    CUDA_CHECK(cudaMalloc(&mesh.num_conn, n_elems * sizeof(dev_idx_t)));
    CUDA_CHECK(cudaMalloc(&mesh.conn_idx, n_elems * max_conn * sizeof(dev_idx_t)));
    CUDA_CHECK(cudaMalloc(&mesh.conn_flux, n_elems * max_conn * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.materials, world.materials.size() * sizeof(Material)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&mesh.energy[b], n_elems * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&mesh.flux[b], n_elems * sizeof(val_t)));
    }

    // Host -> device transfers (static data)
    CUDA_CHECK(cudaMemcpy(mesh.mat_idx, h_mat_idx.data(), n_elems * sizeof(dev_idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.num_conn, h_num_conn.data(), n_elems * sizeof(dev_idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.conn_idx, h_conn_idx.data(), n_elems * max_conn * sizeof(dev_idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.conn_flux, h_conn_flux.data(), n_elems * max_conn * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.materials, world.materials.data(), world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

    return mesh;
}

void freeMesh(DeviceMesh& mesh) {
    CUDA_CHECK(cudaFree(mesh.mat_idx));
    CUDA_CHECK(cudaFree(mesh.num_conn));
    CUDA_CHECK(cudaFree(mesh.conn_idx));
    CUDA_CHECK(cudaFree(mesh.conn_flux));
    CUDA_CHECK(cudaFree(mesh.materials));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(mesh.energy[b]));
        CUDA_CHECK(cudaFree(mesh.flux[b]));
    }
    mesh = DeviceMesh{};
}

// Run simulation for n_iters iterations on the GPU
void runSimulation(World& world, DeviceMesh& mesh, const int n_iters) {
    const size_t n_elems = mesh.n_elems;
    if (n_elems == 0 || n_iters <= 0) {
        return;
    }

    // Host -> device: current dynamic state
    std::vector<val_t> h_energy(n_elems);
    std::vector<val_t> h_flux(n_elems);
    for (size_t i = 0; i < n_elems; ++i) {
        h_energy[i] = world.elements_dynamic[i].current_energy;
        h_flux[i] = world.elements_dynamic[i].total_flux;
    }
    CUDA_CHECK(cudaMemcpy(mesh.energy[0], h_energy.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.flux[0], h_flux.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice));

    // Iterate: kernels are queued asynchronously on the same stream, so
    // successive iterations execute back-to-back without host synchronization.
    constexpr unsigned int BLOCK_SIZE = 256;
    const unsigned int grid_size = static_cast<unsigned int>((n_elems + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int nxt = cur ^ 1;
        stepKernel<<<grid_size, BLOCK_SIZE>>>(mesh.mat_idx, mesh.num_conn, mesh.conn_idx, mesh.conn_flux,
                                              mesh.materials, mesh.energy[cur], mesh.flux[cur],
                                              mesh.energy[nxt], mesh.flux[nxt], n_elems);
        CUDA_CHECK(cudaGetLastError());
        cur = nxt;
    }

    // Device -> host: copy the latest state back into the world
    CUDA_CHECK(cudaMemcpy(h_energy.data(), mesh.energy[cur], n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flux.data(), mesh.flux[cur], n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = h_energy[i];
        world.elements_dynamic[i].total_flux = h_flux[i];
    }
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
    
    // Upload the static mesh to the GPU. Like buildSquare2D, this is one-time
    // setup and is excluded from the timed region (it also initializes the CUDA context).
    DeviceMesh mesh = uploadMesh(world);
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, mesh, n_iters);
    CUDA_CHECK(cudaDeviceSynchronize());
    
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
            freeMesh(mesh);
            return 1;
        }
    }
    
    freeMesh(mesh);
    return 0;
}
