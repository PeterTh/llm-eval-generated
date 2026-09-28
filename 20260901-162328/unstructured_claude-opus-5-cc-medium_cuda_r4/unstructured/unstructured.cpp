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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Threads per block for the update kernel
constexpr int BLOCK_SIZE = 256;

// Device-side mesh in structure-of-arrays form.
//
// The host representation (array of ElementStatic) is convenient to build but
// gives strided, cache-hostile access on a GPU: every thread would touch a
// 144-byte record. On the device the same information is stored as separate
// arrays, with the per-connection data transposed so that connection j of all
// elements is contiguous (conn[j * n + i]). Consecutive threads therefore read
// consecutive addresses for every load, which is what the memory system wants.
struct DeviceMesh {
    uint32_t n_elems = 0;
    uint32_t max_conn = 0;              // Largest num_connections in the mesh

    Material* materials = nullptr;      // [n_materials]
    void* material_idx = nullptr;       // [n_elems], uint8_t or uint32_t
    bool narrow_material_idx = true;    // true if material_idx holds uint8_t
    uint8_t* num_conn = nullptr;        // [n_elems]
    uint32_t* conn_idx = nullptr;       // [max_conn * n_elems], transposed
    val_t* conn_flux = nullptr;         // [max_conn * n_elems], transposed

    val_t* energy[2] = {nullptr, nullptr};  // Ping-pong energy buffers
    val_t* flux = nullptr;                  // Accumulated |flux| per element

    val_t* stage = nullptr;             // Pinned host staging buffer
    uint32_t stage_elems = 0;           // Elements per staging chunk
};

// Compute energy flux between two elements
__host__ __device__ inline val_t computeFlux(const Material& mat, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// One simulation step: one thread per element, grid-stride so that any mesh
// size maps onto any launch configuration.
//
// The accumulated flux is updated in place: a thread only ever touches its own
// element for that quantity, so no second buffer (and no extra traffic) is
// needed. Energy is double-buffered because neighbours read it.
template <typename mat_idx_t>
__global__ __launch_bounds__(BLOCK_SIZE) void updateElementsKernel(
    const uint32_t n_elems,
    const val_t* __restrict__ energy_in,
    val_t* __restrict__ energy_out,
    val_t* __restrict__ flux_acc,
    const uint32_t* __restrict__ conn_idx,
    const val_t* __restrict__ conn_flux,
    const uint8_t* __restrict__ num_conn,
    const mat_idx_t* __restrict__ material_idx,
    const Material* __restrict__ materials) {
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elems; i += stride) {
        const Material mat = materials[material_idx[i]];
        const val_t energy = energy_in[i];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements, in the same order as the
        // reference implementation so the result is bit-identical.
        //
        // The connectivity streams are read exactly once per step and are
        // therefore loaded with the streaming (evict-first) hint: that keeps
        // them from pushing the neighbour energies - which are reused by every
        // adjacent element - out of L2.
        const uint32_t nc = num_conn[i];
        for (uint32_t j = 0; j < nc; ++j) {
            const uint32_t base = j * n_elems + i;
            const uint32_t neighbor_idx = __ldcs(&conn_idx[base]);
            total_flux += computeFlux(mat, energy, __ldcs(&conn_flux[base]),
                                      __ldg(&energy_in[neighbor_idx]));
        }

        // Update element state
        __stcs(&energy_out[i], energy + total_flux);
        flux_acc[i] += fabs(total_flux);
    }
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

// Elements handled per host<->device staging chunk. Repacking is done chunk by
// chunk through a pinned buffer: a single pass over the host mesh, fast
// (DMA-able) transfers, and no large temporary host allocation.
constexpr uint32_t STAGE_ELEMS = 1u << 18;

// The pinned staging buffer holds MAX_CONNECTIONS index columns followed by
// MAX_CONNECTIONS value columns, which also covers the two columns
// (energy, flux) needed for the dynamic state.
inline uint32_t* stageIdx(const DeviceMesh& mesh, uint32_t column) {
    return reinterpret_cast<uint32_t*>(mesh.stage) + column * mesh.stage_elems;
}
inline val_t* stageVal(const DeviceMesh& mesh, uint32_t column) {
    return reinterpret_cast<val_t*>(reinterpret_cast<uint32_t*>(mesh.stage) +
                                    MAX_CONNECTIONS * mesh.stage_elems) +
           column * mesh.stage_elems;
}

// Launch the update kernel with the material-index width chosen for this mesh
void launchUpdate(const DeviceMesh& mesh, int blocks, const val_t* energy_in,
                  val_t* energy_out) {
    if (mesh.narrow_material_idx) {
        updateElementsKernel<uint8_t><<<blocks, BLOCK_SIZE>>>(
            mesh.n_elems, energy_in, energy_out, mesh.flux, mesh.conn_idx, mesh.conn_flux,
            mesh.num_conn, static_cast<const uint8_t*>(mesh.material_idx), mesh.materials);
    } else {
        updateElementsKernel<uint32_t><<<blocks, BLOCK_SIZE>>>(
            mesh.n_elems, energy_in, energy_out, mesh.flux, mesh.conn_idx, mesh.conn_flux,
            mesh.num_conn, static_cast<const uint32_t*>(mesh.material_idx), mesh.materials);
    }
}

// Transfer the static mesh (connectivity, materials) to the device.
// This is part of problem setup, like building the mesh itself, and is
// therefore reported separately from the simulation time.
void uploadMesh(const World& world, DeviceMesh& mesh) {
    const size_t n_elems = world.elements_static.size();
    if (n_elems > std::numeric_limits<uint32_t>::max()) {
        fprintf(stderr, "Mesh too large for 32-bit element indices\n");
        exit(EXIT_FAILURE);
    }
    mesh.n_elems = static_cast<uint32_t>(n_elems);
    mesh.stage_elems = static_cast<uint32_t>(std::min<size_t>(STAGE_ELEMS, n_elems));
    mesh.narrow_material_idx = world.materials.size() <= 256;

    size_t max_conn = 1;
    for (size_t i = 0; i < n_elems; ++i) {
        max_conn = std::max<size_t>(max_conn, world.elements_static[i].num_connections);
    }
    mesh.max_conn = static_cast<uint32_t>(max_conn);

    const size_t mat_idx_size = mesh.narrow_material_idx ? sizeof(uint8_t) : sizeof(uint32_t);
    CUDA_CHECK(cudaHostAlloc(&mesh.stage,
                             MAX_CONNECTIONS * static_cast<size_t>(mesh.stage_elems) *
                                 (sizeof(uint32_t) + sizeof(val_t)),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&mesh.materials, world.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&mesh.material_idx, n_elems * mat_idx_size));
    CUDA_CHECK(cudaMalloc(&mesh.num_conn, n_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&mesh.conn_idx, max_conn * n_elems * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&mesh.conn_flux, max_conn * n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.energy[0], n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.energy[1], n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.flux, n_elems * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(mesh.materials, world.materials.data(),
                          world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

    // Repack connectivity into transposed structure-of-arrays form, chunkwise
    std::vector<uint8_t> h_num_conn(n_elems);
    std::vector<uint8_t> h_mat_narrow(mesh.narrow_material_idx ? n_elems : 0);
    std::vector<uint32_t> h_mat_wide(mesh.narrow_material_idx ? 0 : n_elems);

    for (size_t off = 0; off < n_elems; off += mesh.stage_elems) {
        const size_t k = std::min<size_t>(mesh.stage_elems, n_elems - off);

        for (size_t t = 0; t < k; ++t) {
            const ElementStatic& elem = world.elements_static[off + t];
            if (mesh.narrow_material_idx) {
                h_mat_narrow[off + t] = static_cast<uint8_t>(elem.material_idx);
            } else {
                h_mat_wide[off + t] = static_cast<uint32_t>(elem.material_idx);
            }
            h_num_conn[off + t] = static_cast<uint8_t>(elem.num_connections);
            for (size_t j = 0; j < max_conn; ++j) {
                const bool present = j < elem.num_connections;
                stageIdx(mesh, j)[t] = present ? static_cast<uint32_t>(elem.connected_idx[j]) : 0u;
                stageVal(mesh, j)[t] = present ? elem.connected_flux[j] : 0.0;
            }
        }

        for (size_t j = 0; j < max_conn; ++j) {
            CUDA_CHECK(cudaMemcpy(mesh.conn_idx + j * n_elems + off, stageIdx(mesh, j),
                                  k * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(mesh.conn_flux + j * n_elems + off, stageVal(mesh, j),
                                  k * sizeof(val_t), cudaMemcpyHostToDevice));
        }
    }

    CUDA_CHECK(cudaMemcpy(mesh.num_conn, h_num_conn.data(), n_elems * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.material_idx,
                          mesh.narrow_material_idx ? static_cast<const void*>(h_mat_narrow.data())
                                                   : static_cast<const void*>(h_mat_wide.data()),
                          n_elems * mat_idx_size, cudaMemcpyHostToDevice));

    // Warm up: force the kernel module to be loaded now, so that one-time
    // runtime initialization is not attributed to the simulation. The launch
    // itself is a no-op (zero elements).
    DeviceMesh warmup = mesh;
    warmup.n_elems = 0;
    launchUpdate(warmup, 1, mesh.energy[0], mesh.energy[1]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void freeMesh(DeviceMesh& mesh) {
    CUDA_CHECK(cudaFreeHost(mesh.stage));
    CUDA_CHECK(cudaFree(mesh.materials));
    CUDA_CHECK(cudaFree(mesh.material_idx));
    CUDA_CHECK(cudaFree(mesh.num_conn));
    CUDA_CHECK(cudaFree(mesh.conn_idx));
    CUDA_CHECK(cudaFree(mesh.conn_flux));
    CUDA_CHECK(cudaFree(mesh.energy[0]));
    CUDA_CHECK(cudaFree(mesh.energy[1]));
    CUDA_CHECK(cudaFree(mesh.flux));
    mesh = DeviceMesh{};
}

// Run simulation for n_iters iterations on the GPU
void runSimulation(World& world, DeviceMesh& mesh, const int n_iters) {
    const size_t n_elems = mesh.n_elems;

    // Upload the initial dynamic state (energy / accumulated flux)
    for (size_t off = 0; off < n_elems; off += mesh.stage_elems) {
        const size_t k = std::min<size_t>(mesh.stage_elems, n_elems - off);
        val_t* s_energy = stageVal(mesh, 0);
        val_t* s_flux = stageVal(mesh, 1);
        for (size_t t = 0; t < k; ++t) {
            s_energy[t] = world.elements_dynamic[off + t].current_energy;
            s_flux[t] = world.elements_dynamic[off + t].total_flux;
        }
        CUDA_CHECK(cudaMemcpy(mesh.energy[0] + off, s_energy, k * sizeof(val_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mesh.flux + off, s_flux, k * sizeof(val_t),
                              cudaMemcpyHostToDevice));
    }

    const int blocks = static_cast<int>((n_elems + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        launchUpdate(mesh, blocks, mesh.energy[cur], mesh.energy[cur ^ 1]);
        cur ^= 1;
    }
    CUDA_CHECK(cudaGetLastError());

    // Bring the final state back into the host representation
    for (size_t off = 0; off < n_elems; off += mesh.stage_elems) {
        const size_t k = std::min<size_t>(mesh.stage_elems, n_elems - off);
        val_t* s_energy = stageVal(mesh, 0);
        val_t* s_flux = stageVal(mesh, 1);
        CUDA_CHECK(cudaMemcpy(s_energy, mesh.energy[cur] + off, k * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(s_flux, mesh.flux + off, k * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
        for (size_t t = 0; t < k; ++t) {
            world.elements_dynamic[off + t].current_energy = s_energy[t];
            world.elements_dynamic[off + t].total_flux = s_flux[t];
        }
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

    // Select and report the GPU used for the simulation
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("CUDA device: %s (%d SMs, sm_%d%d)\n", prop.name, prop.multiProcessorCount,
           prop.major, prop.minor);
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

    // Move the static mesh to the GPU (setup, not part of the timed loop)
    DeviceMesh mesh;
    auto upload_start = std::chrono::high_resolution_clock::now();
    uploadMesh(world, mesh);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto upload_end = std::chrono::high_resolution_clock::now();
    printf("Mesh upload time: %ld ms\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(upload_end - upload_start).count());
    printf("\n");

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
    bool valid = true;
    if (validate) {
        valid = validateResults(world);
    }

    freeMesh(mesh);

    return valid ? 0 : 1;
}
