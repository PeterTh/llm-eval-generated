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

// Maximum number of distinct materials (kept in __constant__ memory)
constexpr int MAX_MATERIALS = 8;

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

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_),        \
                    __FILE__, __LINE__, cudaGetErrorString(err_));                        \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

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

// Material properties are read-only during the simulation and there are only a
// handful of them, so they live in constant memory.
__constant__ double c_mat_transfer_coeff[MAX_MATERIALS];
__constant__ double c_mat_external_flow[MAX_MATERIALS];

// GPU-resident mesh in structure-of-arrays layout.
//
// The connectivity is stored transposed (connection-major) so that all threads
// of a warp read consecutive addresses for a given connection slot, which turns
// the gather over the mesh into fully coalesced loads. The dynamic state is
// split into separate energy and flux arrays so that the neighbour gather only
// touches the energies; the interleaving expected by the host is done on the
// device around the transfers.
struct GpuWorld {
    uint32_t* conn_idx = nullptr;    // [max_conn][conn_stride]
    double* conn_flux = nullptr;     // [max_conn][conn_stride]
    uint8_t* num_conn = nullptr;     // [n_elems]
    uint8_t* mat_idx = nullptr;      // [n_elems]
    double* energy[2] = {nullptr, nullptr};
    double* flux[2] = {nullptr, nullptr};
    double2* state_aos = nullptr;    // staging buffer in the host's layout
    bool host_registered = false;    // elements_dynamic pinned for fast transfers
    uint32_t n_elems = 0;
    uint32_t conn_stride = 0;
    int max_conn = 0;
    int block_size = 256;
    int grid_size = 0;
};

static_assert(sizeof(ElementDynamic) == sizeof(double2),
              "device state layout must match ElementDynamic");

// Compute energy flux between two elements
__device__ __forceinline__ val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

// One thread per element: gather the flux from all connected elements and write
// the updated state to the output buffers.
__global__ __launch_bounds__(256) void stepKernel(
    const uint32_t* __restrict__ conn_idx, const double* __restrict__ conn_flux,
    const uint8_t* __restrict__ num_conn, const uint8_t* __restrict__ mat_idx,
    const double* __restrict__ energy_in, const double* __restrict__ flux_in,
    double* __restrict__ energy_out, double* __restrict__ flux_out,
    const uint32_t n_elems, const uint32_t conn_stride) {
    const uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elems; i += stride) {
        const int mat = mat_idx[i];
        const double transfer_coeff = c_mat_transfer_coeff[mat];
        const double energy = energy_in[i];

        // Start with external flow
        double total_flux = c_mat_external_flow[mat];

        // Add flux from all connected elements
        const int n_conn = num_conn[i];
        for (int j = 0; j < n_conn; ++j) {
            const size_t slot = static_cast<size_t>(j) * conn_stride + i;
            const uint32_t neighbor_idx = conn_idx[slot];
            total_flux += computeFlux(transfer_coeff, energy, conn_flux[slot],
                                      __ldg(&energy_in[neighbor_idx]));
        }

        // Update element state
        energy_out[i] = energy + total_flux;
        flux_out[i] = flux_in[i] + fabs(total_flux);
    }
}

// Convert between the host's array-of-structs dynamic state and the device's
// split arrays. Both directions are pure bandwidth and run in a few hundred
// microseconds even for the largest meshes.
__global__ void scatterStateKernel(const double2* __restrict__ state, double* __restrict__ energy,
                                   double* __restrict__ flux, const uint32_t n_elems) {
    const uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elems; i += stride) {
        const double2 s = state[i];
        energy[i] = s.x;
        flux[i] = s.y;
    }
}

__global__ void gatherStateKernel(const double* __restrict__ energy, const double* __restrict__ flux,
                                  double2* __restrict__ state, const uint32_t n_elems) {
    const uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elems; i += stride) {
        state[i] = make_double2(energy[i], flux[i]);
    }
}

// Upload the (immutable) mesh topology and material table to the device.
void initGpuWorld(GpuWorld& gpu, World& world) {
    const size_t n_elems = world.elements_static.size();
    if (n_elems == 0 || n_elems > 0xffffffffull) {
        fprintf(stderr, "Unsupported element count: %zu\n", n_elems);
        exit(1);
    }
    if (world.materials.size() > MAX_MATERIALS) {
        fprintf(stderr, "Too many materials: %zu\n", world.materials.size());
        exit(1);
    }

    gpu.n_elems = static_cast<uint32_t>(n_elems);
    // Pad the per-connection rows so every row start is 128-byte aligned.
    gpu.conn_stride = static_cast<uint32_t>((n_elems + 15) & ~static_cast<size_t>(15));

    size_t max_conn = 1;
    for (const auto& elem : world.elements_static) {
        max_conn = std::max(max_conn, static_cast<size_t>(elem.num_connections));
    }
    gpu.max_conn = static_cast<int>(max_conn);  // rows actually allocated

    // Flatten the array-of-structs mesh into transposed SoA staging buffers.
    std::vector<uint32_t> h_conn_idx(max_conn * gpu.conn_stride, 0);
    std::vector<double> h_conn_flux(max_conn * gpu.conn_stride, 0.0);
    std::vector<uint8_t> h_num_conn(n_elems);
    std::vector<uint8_t> h_mat_idx(n_elems);
    for (size_t i = 0; i < n_elems; ++i) {
        const ElementStatic& elem = world.elements_static[i];
        h_num_conn[i] = static_cast<uint8_t>(elem.num_connections);
        h_mat_idx[i] = static_cast<uint8_t>(elem.material_idx);
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            h_conn_idx[j * gpu.conn_stride + i] = static_cast<uint32_t>(elem.connected_idx[j]);
            h_conn_flux[j * gpu.conn_stride + i] = elem.connected_flux[j];
        }
    }

    CUDA_CHECK(cudaMalloc(&gpu.conn_idx, h_conn_idx.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&gpu.conn_flux, h_conn_flux.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.num_conn, n_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&gpu.mat_idx, n_elems * sizeof(uint8_t)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&gpu.energy[b], n_elems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&gpu.flux[b], n_elems * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu.energy[b], 0, n_elems * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu.flux[b], 0, n_elems * sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc(&gpu.state_aos, n_elems * sizeof(double2)));
    CUDA_CHECK(cudaMemset(gpu.state_aos, 0, n_elems * sizeof(double2)));

    // Page-lock the host state array so the transfers in and out of the timed
    // region run at full PCIe bandwidth without an extra staging copy.
    gpu.host_registered =
        cudaHostRegister(world.elements_dynamic.data(), n_elems * sizeof(ElementDynamic),
                         cudaHostRegisterDefault) == cudaSuccess;
    cudaGetLastError();  // discard the error if the memory could not be pinned

    CUDA_CHECK(cudaMemcpy(gpu.conn_idx, h_conn_idx.data(), h_conn_idx.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.conn_flux, h_conn_flux.data(), h_conn_flux.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.num_conn, h_num_conn.data(), n_elems * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.mat_idx, h_mat_idx.data(), n_elems * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));

    double mat_coeff[MAX_MATERIALS] = {};
    double mat_flow[MAX_MATERIALS] = {};
    for (size_t m = 0; m < world.materials.size(); ++m) {
        mat_coeff[m] = world.materials[m].transfer_coeff;
        mat_flow[m] = world.materials[m].external_flow;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_mat_transfer_coeff, mat_coeff, sizeof(mat_coeff)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_mat_external_flow, mat_flow, sizeof(mat_flow)));

    // Size the grid so that the whole device is filled exactly once; each thread
    // then walks its elements with a grid-stride loop.
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int num_sm = 0, blocks_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, stepKernel,
                                                             gpu.block_size, 0));
    const int max_blocks = std::max(1, num_sm * blocks_per_sm);
    const int needed_blocks =
        static_cast<int>((n_elems + gpu.block_size - 1) / gpu.block_size);
    gpu.grid_size = std::max(1, std::min(max_blocks, needed_blocks));

    // Absorb lazy-init costs (module load, first launch) before timing starts.
    stepKernel<<<gpu.grid_size, gpu.block_size>>>(gpu.conn_idx, gpu.conn_flux, gpu.num_conn,
                                                  gpu.mat_idx, gpu.energy[0], gpu.flux[0],
                                                  gpu.energy[1], gpu.flux[1], gpu.n_elems,
                                                  gpu.conn_stride);
    scatterStateKernel<<<gpu.grid_size, gpu.block_size>>>(gpu.state_aos, gpu.energy[0],
                                                          gpu.flux[0], gpu.n_elems);
    gatherStateKernel<<<gpu.grid_size, gpu.block_size>>>(gpu.energy[0], gpu.flux[0],
                                                         gpu.state_aos, gpu.n_elems);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void freeGpuWorld(GpuWorld& gpu, World& world) {
    if (gpu.host_registered) {
        cudaHostUnregister(world.elements_dynamic.data());
    }
    cudaFree(gpu.conn_idx);
    cudaFree(gpu.conn_flux);
    cudaFree(gpu.num_conn);
    cudaFree(gpu.mat_idx);
    for (int b = 0; b < 2; ++b) {
        cudaFree(gpu.energy[b]);
        cudaFree(gpu.flux[b]);
    }
    cudaFree(gpu.state_aos);
}

// Run simulation for n_iters iterations
void runSimulation(World& world, GpuWorld& gpu, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    const size_t state_bytes = n_elems * sizeof(ElementDynamic);

    // Upload the initial dynamic state and split it into the device layout
    CUDA_CHECK(cudaMemcpyAsync(gpu.state_aos, world.elements_dynamic.data(), state_bytes,
                               cudaMemcpyHostToDevice));
    scatterStateKernel<<<gpu.grid_size, gpu.block_size>>>(gpu.state_aos, gpu.energy[0],
                                                          gpu.flux[0], gpu.n_elems);

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        // Update all elements
        stepKernel<<<gpu.grid_size, gpu.block_size>>>(gpu.conn_idx, gpu.conn_flux, gpu.num_conn,
                                                      gpu.mat_idx, gpu.energy[cur], gpu.flux[cur],
                                                      gpu.energy[cur ^ 1], gpu.flux[cur ^ 1],
                                                      gpu.n_elems, gpu.conn_stride);

        // Swap buffers
        cur ^= 1;
    }
    CUDA_CHECK(cudaGetLastError());

    // Bring the final state back into the host-side element array.
    gatherStateKernel<<<gpu.grid_size, gpu.block_size>>>(gpu.energy[cur], gpu.flux[cur],
                                                         gpu.state_aos, gpu.n_elems);
    CUDA_CHECK(cudaMemcpyAsync(world.elements_dynamic.data(), gpu.state_aos, state_bytes,
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
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

    // Mirror the mesh onto the GPU (part of the setup, like mesh construction)
    GpuWorld gpu;
    initGpuWorld(gpu, world);

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

    runSimulation(world, gpu, n_iters);

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
    bool valid = true;
    if (validate) {
        valid = validateResults(world);
    }

    freeGpuWorld(gpu, world);

    return valid ? 0 : 1;
}
