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

static void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

// Host array used to stage simulation results.  Pinned (page-locked) memory is
// used when available so the device-to-host transfer is a direct DMA instead
// of going through the driver's pageable bounce buffer.
struct HostResultArray {
    val_t* ptr = nullptr;
    size_t count = 0;
    bool pinned = false;

    HostResultArray() = default;
    HostResultArray(const HostResultArray&) = delete;
    HostResultArray& operator=(const HostResultArray&) = delete;

    ~HostResultArray() {
        if (ptr != nullptr) {
            if (pinned) {
                cudaFreeHost(ptr);
            } else {
                std::free(ptr);
            }
        }
    }

    void allocate(size_t n) {
        count = n;
        if (n == 0) {
            return;
        }
        cudaError_t err =
            cudaHostAlloc(reinterpret_cast<void**>(&ptr), n * sizeof(val_t),
                          cudaHostAllocDefault);
        if (err == cudaSuccess) {
            pinned = true;
        } else {
            (void)cudaGetLastError();
            ptr = static_cast<val_t*>(std::malloc(n * sizeof(val_t)));
            if (ptr == nullptr) {
                fprintf(stderr, "Unable to allocate host result buffer\n");
                std::exit(1);
            }
            pinned = false;
        }
        std::memset(ptr, 0, n * sizeof(val_t));
    }

    val_t* data() { return ptr; }
    const val_t* data() const { return ptr; }
    val_t& operator[](size_t i) { return ptr[i]; }
    const val_t& operator[](size_t i) const { return ptr[i]; }
    size_t size() const { return count; }
};

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element (host-side description,
// kept for memory accounting / semantic reference)
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

// World state.  Static connectivity is stored on the device in CSR form:
//   offsets[e] .. offsets[e+1] indexes into connected_idx / connected_flux.
struct World {
    std::vector<Material> materials;

    // Host mirror of the dynamic state (only populated when requested)
    HostResultArray energy;
    HostResultArray total_flux;

    // Device buffers
    uint8_t* d_material_idx = nullptr;
    uint32_t* d_offsets = nullptr;
    uint32_t* d_connected_idx = nullptr;
    val_t* d_connected_flux = nullptr;
    val_t* d_energy[2] = {nullptr, nullptr};
    val_t* d_flux[2] = {nullptr, nullptr};
    unsigned long long* d_hash = nullptr;
    uint64_t result_hash = 0;
    size_t n_elems = 0;
    bool unit_connection_flux = true;

    ~World() {
        if (d_material_idx) cudaFree(d_material_idx);
        if (d_offsets) cudaFree(d_offsets);
        if (d_connected_idx) cudaFree(d_connected_idx);
        if (d_connected_flux) cudaFree(d_connected_flux);
        for (int i = 0; i < 2; ++i) {
            if (d_energy[i]) cudaFree(d_energy[i]);
            if (d_flux[i]) cudaFree(d_flux[i]);
        }
        if (d_hash) cudaFree(d_hash);
    }

    World() = default;
    World(const World&) = delete;
    World& operator=(const World&) = delete;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Material properties kept in constant memory for fast broadcast access
__constant__ val_t c_transfer_coeff[4];
__constant__ val_t c_external_flow[4];

// Compute the energy flux for every element, writing into the swap buffers.
// One thread per element, so the whole mesh is processed in parallel.
// UNIT_FLUX specializes the common case where every connection coefficient is
// exactly 1.0, which lets the kernel skip those loads without changing any
// floating-point result (multiplication by 1.0 is exact).
template <bool UNIT_FLUX>
__global__ void energyTransferKernel(
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ connected_idx,
    const val_t* __restrict__ connected_flux,
    const uint8_t* __restrict__ material_idx,
    const val_t* __restrict__ in_energy,
    const val_t* __restrict__ in_flux,
    val_t* __restrict__ out_energy,
    val_t* __restrict__ out_flux,
    size_t n_elems) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) {
        return;
    }

    const uint8_t mat_idx = material_idx[i];
    const val_t transfer_coeff = c_transfer_coeff[mat_idx];
    const val_t self_energy = in_energy[i];

    // Start with external flow
    val_t total_flux = c_external_flow[mat_idx];

    // Add flux from all connected elements
    const uint32_t begin = offsets[i];
    const uint32_t end = offsets[i + 1];
    if constexpr (UNIT_FLUX) {
        for (uint32_t j = begin; j < end; ++j) {
            const val_t other_energy = in_energy[connected_idx[j]];
            total_flux += (other_energy - self_energy) * transfer_coeff * 0.25;
        }
    } else {
        for (uint32_t j = begin; j < end; ++j) {
            const val_t other_energy = in_energy[connected_idx[j]];
            total_flux += (other_energy - self_energy) * transfer_coeff *
                          connected_flux[j] * 0.25;
        }
    }

    // Update element state
    out_energy[i] = self_energy + total_flux;
    out_flux[i] = in_flux[i] + fabs(total_flux);
}

// Compute the result hash on the device.  The hash is a XOR of per-element
// terms, so a tree/atomic reduction produces exactly the same value as the
// sequential host computation regardless of ordering.
__global__ void resultHashKernel(
    const val_t* __restrict__ energy,
    const val_t* __restrict__ flux,
    size_t n_elems,
    unsigned long long* __restrict__ out) {
    const unsigned int tid = threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    unsigned long long h = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
         i < n_elems; i += stride) {
        const unsigned long long e = __double_as_longlong(energy[i]);
        const unsigned long long f = __double_as_longlong(flux[i]);
        h ^= (e + i) * 0x9e3779b97f4a7c15ULL;
        h ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
    }

    extern __shared__ unsigned long long smem[];
    smem[tid] = h;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] ^= smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicXor(out, smem[0]);
    }
}

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const size_t n_elems = static_cast<size_t>(n_elems_root) *
                           static_cast<size_t>(n_elems_root);
    world.n_elems = n_elems;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Host-side dynamic state initialized to zero energy
    world.energy.allocate(n_elems);
    world.total_flux.allocate(n_elems);

    // Build CSR connectivity on the host
    std::vector<uint8_t> h_material_idx(n_elems, static_cast<uint8_t>(DEFAULT_MAT_ID));
    std::vector<uint32_t> h_offsets(n_elems + 1, 0);
    std::vector<uint32_t> h_connected_idx;
    std::vector<val_t> h_connected_flux;
    h_connected_idx.reserve(n_elems * 4);
    h_connected_flux.reserve(n_elems * 4);

    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t idx = static_cast<size_t>(x) * n_elems_root + y;

            uint32_t count = 0;
            for (int k = 0; k < 4; ++k) {
                const int nx = x + offsets[k][0];
                const int ny = y + offsets[k][1];

                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const uint32_t neighbor_idx =
                        static_cast<uint32_t>(static_cast<size_t>(nx) * n_elems_root + ny);
                    h_connected_idx.push_back(neighbor_idx);
                    h_connected_flux.push_back(1.0);
                    ++count;
                }
            }
            h_offsets[idx + 1] = h_offsets[idx] + count;
        }
    }

    world.unit_connection_flux = true;
    for (val_t f : h_connected_flux) {
        if (f != static_cast<val_t>(1.0)) {
            world.unit_connection_flux = false;
            break;
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    if (n_elems_root > 0) {
        const int last = n_elems_root - 1;
        h_material_idx[static_cast<size_t>(0) * n_elems_root + 0] =
            static_cast<uint8_t>(INFLOW_MAT_ID);
        h_material_idx[static_cast<size_t>(0) * n_elems_root + last] =
            static_cast<uint8_t>(OUTFLOW_MAT_ID);
        h_material_idx[static_cast<size_t>(last) * n_elems_root + 0] =
            static_cast<uint8_t>(OUTFLOW_MAT_ID);
        h_material_idx[static_cast<size_t>(last) * n_elems_root + last] =
            static_cast<uint8_t>(INFLOW_MAT_ID);
    }

    // Upload material properties to constant memory
    val_t transfer_coeff[4] = {0.0, 0.0, 0.0, 0.0};
    val_t external_flow[4] = {0.0, 0.0, 0.0, 0.0};
    for (size_t m = 0; m < world.materials.size(); ++m) {
        transfer_coeff[m] = world.materials[m].transfer_coeff;
        external_flow[m] = world.materials[m].external_flow;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_transfer_coeff, transfer_coeff, sizeof(transfer_coeff)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_external_flow, external_flow, sizeof(external_flow)));

    if (n_elems == 0) {
        return;
    }

    // Allocate device buffers
    CUDA_CHECK(cudaMalloc(&world.d_material_idx, n_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&world.d_offsets, (n_elems + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&world.d_connected_idx,
                          h_connected_idx.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&world.d_connected_flux,
                          h_connected_flux.size() * sizeof(val_t)));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc(&world.d_energy[i], n_elems * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&world.d_flux[i], n_elems * sizeof(val_t)));
    }
    CUDA_CHECK(cudaMalloc(&world.d_hash, sizeof(unsigned long long)));

    // Copy the static mesh to the device
    CUDA_CHECK(cudaMemcpy(world.d_material_idx, h_material_idx.data(),
                          n_elems * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_offsets, h_offsets.data(),
                          (n_elems + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    if (!h_connected_idx.empty()) {
        CUDA_CHECK(cudaMemcpy(world.d_connected_idx, h_connected_idx.data(),
                              h_connected_idx.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(world.d_connected_flux, h_connected_flux.data(),
                              h_connected_flux.size() * sizeof(val_t),
                              cudaMemcpyHostToDevice));
    }

    // Initialize the dynamic state on the device
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMemset(world.d_energy[i], 0, n_elems * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(world.d_flux[i], 0, n_elems * sizeof(val_t)));
    }

    // Release the host-side static copy now that the device owns it
    std::vector<uint8_t>().swap(h_material_idx);
    std::vector<uint32_t>().swap(h_offsets);
    std::vector<uint32_t>().swap(h_connected_idx);
    std::vector<val_t>().swap(h_connected_flux);

    // Warm up the kernel/module so the first timed launch is not penalized
    energyTransferKernel<true><<<1, 32>>>(
        world.d_offsets, world.d_connected_idx, world.d_connected_flux,
        world.d_material_idx, world.d_energy[0], world.d_flux[0],
        world.d_energy[1], world.d_flux[1], 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Run simulation for n_iters iterations.  The full dynamic state is only
// copied back to the host when the caller needs it (validation/results);
// otherwise just the result hash is retrieved.
void runSimulation(World& world, const int n_iters, const bool need_host_state) {
    const size_t n_elems = world.n_elems;

    if (n_elems == 0) {
        world.result_hash = 0;
        return;
    }

    int cur = 0;
    int nxt = 1;

    constexpr int threads = 256;
    const unsigned int blocks =
        static_cast<unsigned int>((n_elems + threads - 1) / threads);

    if (n_iters > 0) {
        for (int iter = 0; iter < n_iters; ++iter) {
            if (world.unit_connection_flux) {
                energyTransferKernel<true><<<blocks, threads>>>(
                    world.d_offsets, world.d_connected_idx, world.d_connected_flux,
                    world.d_material_idx, world.d_energy[cur], world.d_flux[cur],
                    world.d_energy[nxt], world.d_flux[nxt], n_elems);
            } else {
                energyTransferKernel<false><<<blocks, threads>>>(
                    world.d_offsets, world.d_connected_idx, world.d_connected_flux,
                    world.d_material_idx, world.d_energy[cur], world.d_flux[cur],
                    world.d_energy[nxt], world.d_flux[nxt], n_elems);
            }
            cur ^= 1;
            nxt ^= 1;
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // Hash the final state on the device
    CUDA_CHECK(cudaMemset(world.d_hash, 0, sizeof(unsigned long long)));
    resultHashKernel<<<blocks, threads, threads * sizeof(unsigned long long)>>>(
        world.d_energy[cur], world.d_flux[cur], n_elems, world.d_hash);

    unsigned long long host_hash = 0;
    CUDA_CHECK(cudaMemcpy(&host_hash, world.d_hash, sizeof(host_hash),
                          cudaMemcpyDeviceToHost));
    world.result_hash = static_cast<uint64_t>(host_hash);

    // Bring the final state back to the host only if it is needed
    if (need_host_state) {
        CUDA_CHECK(cudaMemcpy(world.energy.data(), world.d_energy[cur],
                              n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(world.total_flux.data(), world.d_flux[cur],
                              n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < world.energy.size(); ++i) {
        energy_sum += world.energy[i];
        flux_sum += world.total_flux[i];
        energy_max = std::max(world.energy[i], energy_max);
        energy_min = std::min(world.energy[i], energy_min);
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
    const size_t static_mem = world.n_elems * sizeof(ElementStatic);
    const size_t dynamic_mem = world.n_elems * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");

    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, validate || printResults);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const long duration_ms = static_cast<long>(duration_us / 1000);

    printf("Computation time: %ld ms\n", duration_ms);

    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double measured_us = static_cast<double>(std::max<long long>(duration_us, 1));
    const double time_per_iter = (measured_us / 1000.0) / n_measured_iters;
    const double giga_elems_per_sec =
        (n_measured_iters * static_cast<double>(n_elems)) / (measured_us / 1e6) / 1e9;

    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;

    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);

    // Compute hash for verification
    const uint64_t hash = world.result_hash;
    printf("  Result hash: %016lX\n", hash);
    printf("\n");

    // Print results for external validation
    if (printResults) {
        std::vector<double> energyData;
        energyData.reserve(world.energy.size());
        for (size_t i = 0; i < world.energy.size(); ++i) {
            energyData.push_back(world.energy[i]);
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
