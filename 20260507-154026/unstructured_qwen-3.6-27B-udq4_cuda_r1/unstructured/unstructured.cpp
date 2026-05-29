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

// ============================================================
// CUDA Kernels
// ============================================================

// Simulation kernel: one thread per element
__global__ void simulationKernel(
    const int n_elems,
    const double* d_mat_transfer_coeff,
    const double* d_mat_external_flow,
    const int* d_element_material_idx,
    const int* d_element_num_connections,
    const int* d_element_connected_idx,
    const double* d_element_connected_flux,
    const double* d_current_energy,
    const double* d_total_flux,
    double* d_next_energy,
    double* d_next_total_flux)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    int mat_id = d_element_material_idx[i];
    int num_conn = d_element_num_connections[i];
    double this_energy = d_current_energy[i];
    double this_flux = d_total_flux[i];

    double total_flux = d_mat_external_flow[mat_id];
    double tc = d_mat_transfer_coeff[mat_id];

    int base = i * MAX_CONNECTIONS;
    for (int j = 0; j < num_conn; ++j) {
        int neighbor = d_element_connected_idx[base + j];
        double neighbor_energy = d_current_energy[neighbor];
        double conn_flux = d_element_connected_flux[base + j];
        total_flux += (neighbor_energy - this_energy) * tc * conn_flux * 0.25;
    }

    d_next_energy[i] = this_energy + total_flux;
    d_next_total_flux[i] = this_flux + fabs(total_flux);
}

// Reduction kernel for sum (block-level reduction + atomic add)
__global__ void sumKernel(const int n, const double* data, double* result) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

    double val = 0.0;
    if (i < n) val = data[i];
    if (i + blockDim.x < n) val += data[i + blockDim.x];

    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(result, sdata[0]);
}

// Helper atomics for double min/max
__device__ void atomicMinAsDouble(double* addr, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)addr;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (val >= __longlong_as_double(assumed)) break;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (old != assumed);
}

__device__ void atomicMaxAsDouble(double* addr, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)addr;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (val <= __longlong_as_double(assumed)) break;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (old != assumed);
}

// Reduction kernel for min/max
__global__ void minMaxKernel(const int n, const double* data, double* d_min, double* d_max) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

    double vmin = 0.0, vmax = 0.0;
    bool has_val = false;
    if (i < n) { vmin = data[i]; vmax = data[i]; has_val = true; }
    if (i + blockDim.x < n) {
        double v = data[i + blockDim.x];
        if (!has_val) { vmin = v; vmax = v; has_val = true; }
        else { vmin = fmin(vmin, v); vmax = fmax(vmax, v); }
    }

    sdata[tid * 2] = has_val ? vmin : HUGE_VAL;
    sdata[tid * 2 + 1] = has_val ? vmax : -HUGE_VAL;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid * 2] = fmin(sdata[tid * 2], sdata[(tid + s) * 2]);
            sdata[tid * 2 + 1] = fmax(sdata[tid * 2 + 1], sdata[(tid + s) * 2 + 1]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMinAsDouble(d_min, sdata[0]);
        atomicMaxAsDouble(d_max, sdata[1]);
    }
}

// Hash kernel: each thread computes per-element hash contribution
__global__ void hashKernel(
    const int n_elems,
    const double* d_current_energy,
    const double* d_total_flux,
    uint64_t* d_hash_parts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&d_current_energy[i]);
    const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&d_total_flux[i]);
    uint64_t h = 0;
    h ^= (*e_ptr + (uint64_t)i) * 0x9e3779b97f4a7c15ULL;
    h ^= (*f_ptr + (uint64_t)i) * 0xbf58476d1ce4e5b9ULL;
    d_hash_parts[i] = h;
}

// ============================================================
// Build a 2D square grid as an unstructured mesh
// ============================================================
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

// ============================================================
// CUDA data management
// ============================================================

struct CudaData {
    int n_elems;
    int n_materials;

    // Material data (separate arrays for coalesced access)
    double* d_mat_transfer_coeff;
    double* d_mat_external_flow;

    // Static element data (flattened SoA)
    int* d_element_material_idx;
    int* d_element_num_connections;
    int* d_element_connected_idx;
    double* d_element_connected_flux;

    // Dynamic data (two ping-pong buffers)
    double* d_energy[2];
    double* d_flux[2];
    int active;  // 0 or 1
};

static void uploadToCuda(CudaData& cd, const World& world) {
    cd.n_elems = (int)world.elements_static.size();
    cd.n_materials = (int)world.materials.size();
    cd.active = 0;

    const int n = cd.n_elems;

    // Materials: separate arrays
    cudaMalloc(&cd.d_mat_transfer_coeff, cd.n_materials * sizeof(double));
    cudaMalloc(&cd.d_mat_external_flow, cd.n_materials * sizeof(double));
    std::vector<double> h_tc(cd.n_materials), h_ef(cd.n_materials);
    for (int i = 0; i < cd.n_materials; ++i) {
        h_tc[i] = world.materials[i].transfer_coeff;
        h_ef[i] = world.materials[i].external_flow;
    }
    cudaMemcpy(cd.d_mat_transfer_coeff, h_tc.data(), cd.n_materials * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cd.d_mat_external_flow, h_ef.data(), cd.n_materials * sizeof(double), cudaMemcpyHostToDevice);

    // Static element data: flattened SoA
    cudaMalloc(&cd.d_element_material_idx, n * sizeof(int));
    cudaMalloc(&cd.d_element_num_connections, n * sizeof(int));
    cudaMalloc(&cd.d_element_connected_idx, n * MAX_CONNECTIONS * sizeof(int));
    cudaMalloc(&cd.d_element_connected_flux, n * MAX_CONNECTIONS * sizeof(double));

    std::vector<int> h_mat_idx(n), h_num_conn(n);
    std::vector<int> h_conn_idx(n * MAX_CONNECTIONS, 0);
    std::vector<double> h_conn_flux(n * MAX_CONNECTIONS, 0.0);
    for (int i = 0; i < n; ++i) {
        h_mat_idx[i] = (int)world.elements_static[i].material_idx;
        h_num_conn[i] = (int)world.elements_static[i].num_connections;
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            h_conn_idx[i * MAX_CONNECTIONS + j] = (int)world.elements_static[i].connected_idx[j];
            h_conn_flux[i * MAX_CONNECTIONS + j] = world.elements_static[i].connected_flux[j];
        }
    }
    cudaMemcpy(cd.d_element_material_idx, h_mat_idx.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cd.d_element_num_connections, h_num_conn.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cd.d_element_connected_idx, h_conn_idx.data(), n * MAX_CONNECTIONS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cd.d_element_connected_flux, h_conn_flux.data(), n * MAX_CONNECTIONS * sizeof(double), cudaMemcpyHostToDevice);

    // Dynamic data: two ping-pong buffers, initialized to 0
    cudaMalloc(&cd.d_energy[0], n * sizeof(double));
    cudaMalloc(&cd.d_energy[1], n * sizeof(double));
    cudaMalloc(&cd.d_flux[0], n * sizeof(double));
    cudaMalloc(&cd.d_flux[1], n * sizeof(double));
    cudaMemset(cd.d_energy[0], 0, n * sizeof(double));
    cudaMemset(cd.d_energy[1], 0, n * sizeof(double));
    cudaMemset(cd.d_flux[0], 0, n * sizeof(double));
    cudaMemset(cd.d_flux[1], 0, n * sizeof(double));

    // Initial dynamic state is all zeros (already set by cudaMemset above)
}

static void downloadFromCuda(CudaData& cd, World& world) {
    const int n = cd.n_elems;
    // Copy energy
    std::vector<double> h_energy(n);
    cudaMemcpy(h_energy.data(), cd.d_energy[cd.active], n * sizeof(double), cudaMemcpyDeviceToHost);
    // Copy flux
    std::vector<double> h_flux(n);
    cudaMemcpy(h_flux.data(), cd.d_flux[cd.active], n * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        world.elements_dynamic[i].current_energy = h_energy[i];
        world.elements_dynamic[i].total_flux = h_flux[i];
    }
}

static void freeCuda(CudaData& cd) {
    cudaFree(cd.d_mat_transfer_coeff);
    cudaFree(cd.d_mat_external_flow);
    cudaFree(cd.d_element_material_idx);
    cudaFree(cd.d_element_num_connections);
    cudaFree(cd.d_element_connected_idx);
    cudaFree(cd.d_element_connected_flux);
    cudaFree(cd.d_energy[0]);
    cudaFree(cd.d_energy[1]);
    cudaFree(cd.d_flux[0]);
    cudaFree(cd.d_flux[1]);
}

// ============================================================
// CUDA simulation
// ============================================================
void runSimulation(World& world, const int n_iters) {
    CudaData cd;
    uploadToCuda(cd, world);

    const int n = cd.n_elems;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    for (int iter = 0; iter < n_iters; ++iter) {
        int read_buf = cd.active;
        int write_buf = 1 - cd.active;

        simulationKernel<<<grid_size, block_size>>>(
            n,
            cd.d_mat_transfer_coeff,
            cd.d_mat_external_flow,
            cd.d_element_material_idx,
            cd.d_element_num_connections,
            cd.d_element_connected_idx,
            cd.d_element_connected_flux,
            cd.d_energy[read_buf],
            cd.d_flux[read_buf],
            cd.d_energy[write_buf],
            cd.d_flux[write_buf]);

        cd.active = write_buf;
    }

    // Download results back to host
    downloadFromCuda(cd, world);
    freeCuda(cd);
}

// ============================================================
// Validate simulation results
// ============================================================
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

// ============================================================
// Compute hash of results
// ============================================================
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
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

    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

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
            return 1;
        }
    }

    return 0;
}
