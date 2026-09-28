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
// CUDA infrastructure
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        const cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,        \
                    __LINE__, cudaGetErrorString(err_));                            \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)

// Device-side mesh representation.
//
// The host mesh is stored as an array of fat structs (ElementStatic), which is a
// poor fit for a GPU: neighbouring threads would read strided, mostly unused
// bytes.  On the device the same information is kept as a structure of arrays,
// with the per-connection data transposed (connection-major) so that all
// threads of a warp read consecutive addresses when they process connection j.
//
// Element indices are stored as 32-bit values: the element count is computed as
// an `int` (n_elems_root * n_elems_root), so it always fits.
struct DeviceMesh {
    uint32_t* connected_idx = nullptr;   // [max_connections][n_elems]
    val_t* connected_flux = nullptr;     // [max_connections][n_elems]
    uint8_t* num_connections = nullptr;  // [n_elems]
    uint32_t* material_idx = nullptr;    // [n_elems]
    val_t* mat_transfer = nullptr;       // [n_materials]
    val_t* mat_external = nullptr;       // [n_materials]

    // Double-buffered dynamic state, kept as separate arrays so that the energy
    // gather (the only randomly-accessed field) does not pull in flux data.
    val_t* energy[2] = {nullptr, nullptr};
    val_t* flux[2] = {nullptr, nullptr};

    // Pinned staging buffers for the initial/final state transfer.  Allocated
    // (and first-touched) during setup so that neither the allocation nor the
    // page faults land in the measured region.
    val_t* pinned_energy = nullptr;
    val_t* pinned_flux = nullptr;

    uint32_t n_elems = 0;
    int max_connections = 0;

    // Launch configuration, determined once during setup
    void* kernel = nullptr;
    int grid = 0;
    int block = 0;
};

// One thread per element; the grid-stride loop keeps the kernel correct for any
// launch configuration (and for meshes larger than the launched grid).
//
// MAXC is the mesh-wide maximum connection count; making it a template
// parameter lets the compiler fully unroll the gather loop and keep the
// per-connection state in registers.  The per-element count is still honoured,
// so irregular meshes (boundary elements with fewer connections) stay correct.
template <int MAXC>
__global__ __launch_bounds__(256) void simulationStepKernel(
    const uint32_t* __restrict__ connected_idx, const val_t* __restrict__ connected_flux,
    const uint8_t* __restrict__ num_connections, const uint32_t* __restrict__ material_idx,
    const val_t* __restrict__ mat_transfer, const val_t* __restrict__ mat_external,
    const val_t* __restrict__ energy_in, const val_t* __restrict__ flux_in,
    val_t* __restrict__ energy_out, val_t* __restrict__ flux_out, const uint32_t n_elems) {
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elems; i += stride) {
        const uint32_t mat = material_idx[i];
        const val_t transfer_coeff = mat_transfer[mat];
        const val_t this_energy = energy_in[i];

        // Start with external flow
        val_t total_flux = mat_external[mat];

        const int n_conn = num_connections[i];

        // Add flux from all connected elements (same order as the reference)
#pragma unroll
        for (int j = 0; j < MAXC; ++j) {
            if (j >= n_conn) break;
            const uint32_t neighbor_idx = connected_idx[static_cast<size_t>(j) * n_elems + i];
            const val_t conn_flux = connected_flux[static_cast<size_t>(j) * n_elems + i];
            const val_t neighbor_energy = energy_in[neighbor_idx];
            total_flux += (neighbor_energy - this_energy) * transfer_coeff * conn_flux * 0.25;
        }

        // Update element state
        energy_out[i] = this_energy + total_flux;
        flux_out[i] = flux_in[i] + fabs(total_flux);
    }
}

using StepKernel = void (*)(const uint32_t*, const val_t*, const uint8_t*, const uint32_t*,
                            const val_t*, const val_t*, const val_t*, const val_t*, val_t*,
                            val_t*, uint32_t);

static StepKernel selectKernel(int max_connections) {
    // Specialise for the connection counts that actually occur; the generic
    // MAX_CONNECTIONS instantiation handles everything else.
    switch (max_connections) {
        case 0:
        case 1: return simulationStepKernel<1>;
        case 2: return simulationStepKernel<2>;
        case 3: return simulationStepKernel<3>;
        case 4: return simulationStepKernel<4>;
        default: return simulationStepKernel<MAX_CONNECTIONS>;
    }
}

// Transpose the host mesh into the device layout and upload it.
void uploadMesh(const World& world, DeviceMesh& mesh) {
    const size_t n_elems = world.elements_static.size();
    mesh.n_elems = static_cast<uint32_t>(n_elems);

    int max_conn = 0;
    for (size_t i = 0; i < n_elems; ++i) {
        max_conn = std::max(max_conn, static_cast<int>(world.elements_static[i].num_connections));
    }
    mesh.max_connections = std::max(max_conn, 1);
    const size_t mc = static_cast<size_t>(mesh.max_connections);

    // Connection-major staging buffers
    std::vector<uint32_t> h_idx(mc * n_elems, 0);
    std::vector<val_t> h_flux(mc * n_elems, 0.0);
    std::vector<uint8_t> h_nconn(n_elems);
    std::vector<uint32_t> h_mat(n_elems);
    for (size_t i = 0; i < n_elems; ++i) {
        const ElementStatic& e = world.elements_static[i];
        h_nconn[i] = static_cast<uint8_t>(e.num_connections);
        h_mat[i] = static_cast<uint32_t>(e.material_idx);
        for (idx_t j = 0; j < e.num_connections; ++j) {
            h_idx[static_cast<size_t>(j) * n_elems + i] = static_cast<uint32_t>(e.connected_idx[j]);
            h_flux[static_cast<size_t>(j) * n_elems + i] = e.connected_flux[j];
        }
    }

    const size_t n_mat = world.materials.size();
    std::vector<val_t> h_transfer(n_mat), h_external(n_mat);
    for (size_t m = 0; m < n_mat; ++m) {
        h_transfer[m] = world.materials[m].transfer_coeff;
        h_external[m] = world.materials[m].external_flow;
    }

    CUDA_CHECK(cudaMalloc(&mesh.connected_idx, h_idx.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&mesh.connected_flux, h_flux.size() * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.num_connections, n_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&mesh.material_idx, n_elems * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&mesh.mat_transfer, n_mat * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.mat_external, n_mat * sizeof(val_t)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&mesh.energy[b], n_elems * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&mesh.flux[b], n_elems * sizeof(val_t)));
    }

    CUDA_CHECK(cudaMemcpy(mesh.connected_idx, h_idx.data(), h_idx.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.connected_flux, h_flux.data(), h_flux.size() * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.num_connections, h_nconn.data(), n_elems * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.material_idx, h_mat.data(), n_elems * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.mat_transfer, h_transfer.data(), n_mat * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.mat_external, h_external.data(), n_mat * sizeof(val_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaHostAlloc(&mesh.pinned_energy, n_elems * sizeof(val_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&mesh.pinned_flux, n_elems * sizeof(val_t), cudaHostAllocDefault));
    memset(mesh.pinned_energy, 0, n_elems * sizeof(val_t));
    memset(mesh.pinned_flux, 0, n_elems * sizeof(val_t));

    // Launch configuration: one thread per element.  A persistent (SM-sized)
    // grid measured slower here -- with one block per 256 consecutive elements
    // the neighbour gathers of concurrently running blocks overlap, so the
    // ±1 / ±row_length accesses hit in cache.
    const StepKernel kernel = selectKernel(mesh.max_connections);
    mesh.kernel = reinterpret_cast<void*>(kernel);
    mesh.block = 256;
    mesh.grid = static_cast<int>(std::max<size_t>((n_elems + mesh.block - 1) / mesh.block, 1));

    // Empty launch: forces the kernel module to be loaded here rather than on
    // the first timed iteration.
    kernel<<<1, mesh.block>>>(mesh.connected_idx, mesh.connected_flux, mesh.num_connections,
                              mesh.material_idx, mesh.mat_transfer, mesh.mat_external,
                              mesh.energy[0], mesh.flux[0], mesh.energy[1], mesh.flux[1], 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void freeMesh(DeviceMesh& mesh) {
    cudaFree(mesh.connected_idx);
    cudaFree(mesh.connected_flux);
    cudaFree(mesh.num_connections);
    cudaFree(mesh.material_idx);
    cudaFree(mesh.mat_transfer);
    cudaFree(mesh.mat_external);
    cudaFreeHost(mesh.pinned_energy);
    cudaFreeHost(mesh.pinned_flux);
    for (int b = 0; b < 2; ++b) {
        cudaFree(mesh.energy[b]);
        cudaFree(mesh.flux[b]);
    }
    mesh = DeviceMesh{};
}

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

// Run simulation for n_iters iterations on the GPU.
//
// The dynamic state is uploaded once, the iteration loop runs entirely on the
// device (buffers are swapped by exchanging pointers, no host synchronisation
// in between), and the final state is copied back into world.elements_dynamic.
void runSimulation(World& world, DeviceMesh& mesh, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    if (n_elems == 0) {
        return;
    }

    // Upload initial dynamic state in the device (de-interleaved) layout
    for (size_t i = 0; i < n_elems; ++i) {
        mesh.pinned_energy[i] = world.elements_dynamic[i].current_energy;
        mesh.pinned_flux[i] = world.elements_dynamic[i].total_flux;
    }
    CUDA_CHECK(cudaMemcpyAsync(mesh.energy[0], mesh.pinned_energy, n_elems * sizeof(val_t),
                               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(mesh.flux[0], mesh.pinned_flux, n_elems * sizeof(val_t),
                               cudaMemcpyHostToDevice));

    const StepKernel kernel = reinterpret_cast<StepKernel>(mesh.kernel);
    const int grid = mesh.grid;
    const int block = mesh.block;

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int next = cur ^ 1;
        kernel<<<grid, block>>>(mesh.connected_idx, mesh.connected_flux, mesh.num_connections,
                                mesh.material_idx, mesh.mat_transfer, mesh.mat_external,
                                mesh.energy[cur], mesh.flux[cur], mesh.energy[next],
                                mesh.flux[next], mesh.n_elems);
        // Swap buffers
        cur = next;
    }
    CUDA_CHECK(cudaGetLastError());

    // Copy the final state back into the host representation
    CUDA_CHECK(cudaMemcpyAsync(mesh.pinned_energy, mesh.energy[cur], n_elems * sizeof(val_t),
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(mesh.pinned_flux, mesh.flux[cur], n_elems * sizeof(val_t),
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = mesh.pinned_energy[i];
        world.elements_dynamic[i].total_flux = mesh.pinned_flux[i];
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

    // Initialize the CUDA device (context creation is setup, not compute)
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("ERROR: no CUDA-capable device found\n");
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("CUDA device: %s (%d SMs, compute %d.%d)\n", prop.name, prop.multiProcessorCount,
           prop.major, prop.minor);
    CUDA_CHECK(cudaFree(nullptr));  // force context initialization

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

    // Transfer the (immutable) mesh connectivity to the device
    printf("Uploading mesh to GPU...\n");
    DeviceMesh mesh;
    uploadMesh(world, mesh);
    CUDA_CHECK(cudaDeviceSynchronize());

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
    bool ok = true;
    if (validate) {
        ok = validateResults(world);
    }

    freeMesh(mesh);

    return ok ? 0 : 1;
}
