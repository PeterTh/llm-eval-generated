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

// SoA (Struct of Arrays) layout for GPU-friendly memory access
struct SoAStatic {
    uint64_t* material_idx;     // [n_elems]
    uint64_t* num_connections;  // [n_elems]
    uint64_t* connected_idx;    // [n_elems * MAX_CONNECTIONS]
    double* connected_flux;     // [n_elems * MAX_CONNECTIONS]
};

struct SoADynamic {
    double* current_energy;  // [n_elems]
    double* total_flux;      // [n_elems]
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Number of materials
constexpr int NUM_MATERIALS = 3;

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// GPU kernel: simulate one iteration of energy transfer
__global__ void simulation_kernel(
    const Material* __restrict__ materials,
    const uint64_t* __restrict__ mat_idx,
    const uint64_t* __restrict__ num_conn,
    const uint64_t* __restrict__ conn_idx,
    const double* __restrict__ conn_flux,
    const double* __restrict__ energy_in,
    const double* __restrict__ flux_in,
    double* __restrict__ energy_out,
    double* __restrict__ flux_out,
    const size_t n_elems)
{
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    // Load material properties (small, will be cached in L1/L2)
    const uint64_t midx = __ldg(&mat_idx[i]);
    const double transfer_coeff = __ldg(&materials[midx].transfer_coeff);
    const double external_flow = __ldg(&materials[midx].external_flow);

    const double this_energy = __ldg(&energy_in[i]);
    const uint64_t nc = __ldg(&num_conn[i]);

    double total_flux = external_flow;

    const size_t base = i * MAX_CONNECTIONS;
    for (uint64_t j = 0; j < nc; ++j) {
        const uint64_t neighbor = __ldg(&conn_idx[base + j]);
        const double neighbor_energy = __ldg(&energy_in[neighbor]);
        const double cflux = __ldg(&conn_flux[base + j]);
        total_flux += (neighbor_energy - this_energy) * transfer_coeff * cflux * 0.25;
    }

    energy_out[i] = this_energy + total_flux;
    flux_out[i] = __ldg(&flux_in[i]) + fabs(total_flux);
}

// Allocate SoA on device
void allocSoAStatic(SoAStatic& s, size_t n_elems) {
    CUDA_CHECK(cudaMalloc(&s.material_idx, n_elems * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&s.num_connections, n_elems * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&s.connected_idx, n_elems * MAX_CONNECTIONS * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&s.connected_flux, n_elems * MAX_CONNECTIONS * sizeof(double)));
}

void allocSoADynamic(SoADynamic& d, size_t n_elems) {
    CUDA_CHECK(cudaMalloc(&d.current_energy, n_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.total_flux, n_elems * sizeof(double)));
}

void freeSoAStatic(SoAStatic& s) {
    cudaFree(s.material_idx);
    cudaFree(s.num_connections);
    cudaFree(s.connected_idx);
    cudaFree(s.connected_flux);
}

void freeSoADynamic(SoADynamic& d) {
    cudaFree(d.current_energy);
    cudaFree(d.total_flux);
}

// Build mesh on host, then upload to GPU
void buildAndUpload(SoAStatic& d_static, SoADynamic& d_dyn, SoADynamic& d_dyn_swap,
                    const std::vector<ElementStatic>& h_static,
                    const std::vector<ElementDynamic>& h_dyn,
                    size_t n_elems) {
    // Flatten host AoS to SoA arrays
    std::vector<uint64_t> mat_idx(n_elems);
    std::vector<uint64_t> num_conn(n_elems);
    std::vector<uint64_t> conn_idx(n_elems * MAX_CONNECTIONS, 0);
    std::vector<double> conn_flux(n_elems * MAX_CONNECTIONS, 0.0);
    std::vector<double> energy(n_elems);
    std::vector<double> flux(n_elems);

    for (size_t i = 0; i < n_elems; ++i) {
        mat_idx[i] = h_static[i].material_idx;
        num_conn[i] = h_static[i].num_connections;
        for (idx_t j = 0; j < h_static[i].num_connections; ++j) {
            conn_idx[i * MAX_CONNECTIONS + j] = h_static[i].connected_idx[j];
            conn_flux[i * MAX_CONNECTIONS + j] = h_static[i].connected_flux[j];
        }
        energy[i] = h_dyn[i].current_energy;
        flux[i] = h_dyn[i].total_flux;
    }

    // Allocate device memory
    allocSoAStatic(d_static, n_elems);
    allocSoADynamic(d_dyn, n_elems);
    allocSoADynamic(d_dyn_swap, n_elems);

    // Upload static data
    CUDA_CHECK(cudaMemcpy(d_static.material_idx, mat_idx.data(), n_elems * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_static.num_connections, num_conn.data(), n_elems * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_static.connected_idx, conn_idx.data(), n_elems * MAX_CONNECTIONS * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_static.connected_flux, conn_flux.data(), n_elems * MAX_CONNECTIONS * sizeof(double), cudaMemcpyHostToDevice));

    // Upload dynamic data
    CUDA_CHECK(cudaMemcpy(d_dyn.current_energy, energy.data(), n_elems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dyn.total_flux, flux.data(), n_elems * sizeof(double), cudaMemcpyHostToDevice));

    // Zero swap buffer
    CUDA_CHECK(cudaMemset(d_dyn_swap.current_energy, 0, n_elems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dyn_swap.total_flux, 0, n_elems * sizeof(double)));
}

// Download results from GPU to host vectors
void downloadResults(const SoADynamic& d_dyn, std::vector<ElementDynamic>& h_dyn, size_t n_elems) {
    std::vector<double> energy(n_elems);
    std::vector<double> flux(n_elems);
    CUDA_CHECK(cudaMemcpy(energy.data(), d_dyn.current_energy, n_elems * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flux.data(), d_dyn.total_flux, n_elems * sizeof(double), cudaMemcpyDeviceToHost));
    h_dyn.resize(n_elems);
    for (size_t i = 0; i < n_elems; ++i) {
        h_dyn[i].current_energy = energy[i];
        h_dyn[i].total_flux = flux[i];
    }
}

// Build a 2D square grid as an unstructured mesh (host)
void buildSquare2D(std::vector<ElementStatic>& elements_static,
                   std::vector<ElementDynamic>& elements_dynamic,
                   const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;

    elements_static.resize(n_elems);
    elements_dynamic.resize(n_elems);

    for (int i = 0; i < n_elems; ++i) {
        elements_static[i].material_idx = DEFAULT_MAT_ID;
        elements_static[i].num_connections = 0;
        elements_dynamic[i].current_energy = 0.0;
        elements_dynamic[i].total_flux = 0.0;
    }

    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = elements_static[idx];

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    const int last = n_elems_root - 1;
    elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
}

// Run simulation on GPU
void runSimulation(SoAStatic& d_static, SoADynamic& d_dyn, SoADynamic& d_dyn_swap,
                   const Material* d_materials, size_t n_elems, int n_iters) {
    const int blockSize = 256;
    const int gridSize = (int)((n_elems + blockSize - 1) / blockSize);

    SoADynamic* current = &d_dyn;
    SoADynamic* next = &d_dyn_swap;

    for (int iter = 0; iter < n_iters; ++iter) {
        simulation_kernel<<<gridSize, blockSize>>>(
            d_materials,
            d_static.material_idx,
            d_static.num_connections,
            d_static.connected_idx,
            d_static.connected_flux,
            current->current_energy,
            current->total_flux,
            next->current_energy,
            next->total_flux,
            n_elems);

        // Swap pointers
        SoADynamic* tmp = current;
        current = next;
        next = tmp;
    }

    // After loop, result is in 'current'. If n_iters is odd, current == d_dyn_swap.
    // If n_iters is even, current == d_dyn.
    // We need result in d_dyn for download. If current != &d_dyn, copy.
    if (current != &d_dyn) {
        CUDA_CHECK(cudaMemcpy(d_dyn.current_energy, d_dyn_swap.current_energy,
                              n_elems * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_dyn.total_flux, d_dyn_swap.total_flux,
                              n_elems * sizeof(double), cudaMemcpyDeviceToDevice));
    }
}

// Validate simulation results
bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    constexpr val_t energy_epsilon = 1e-8;

    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }

    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
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
    printf("Parallelization: CUDA GPU\n");
    printf("\n");

    // Build mesh on host
    printf("Building unstructured mesh...\n");
    std::vector<ElementStatic> h_static;
    std::vector<ElementDynamic> h_dyn;
    buildSquare2D(h_static, h_dyn, n_elems_root);

    // Materials
    Material h_materials[NUM_MATERIALS];
    h_materials[0] = {0.8, 0.0};
    h_materials[1] = {0.8, 0.5};
    h_materials[2] = {0.8, -0.5};

    // Calculate memory usage
    const size_t static_mem = n_elems * sizeof(ElementStatic);
    const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");

    // Upload to GPU
    printf("Uploading data to GPU...\n");
    SoAStatic d_static;
    SoADynamic d_dyn, d_dyn_swap;
    buildAndUpload(d_static, d_dyn, d_dyn_swap, h_static, h_dyn, n_elems);

    // Upload materials
    Material* d_materials;
    CUDA_CHECK(cudaMalloc(&d_materials, NUM_MATERIALS * sizeof(Material)));
    CUDA_CHECK(cudaMemcpy(d_materials, h_materials, NUM_MATERIALS * sizeof(Material), cudaMemcpyHostToDevice));

    // Free host mesh data (no longer needed)
    h_static.clear();
    h_static.shrink_to_fit();
    h_dyn.clear();
    h_dyn.shrink_to_fit();

    // Run simulation
    printf("Running simulation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(d_static, d_dyn, d_dyn_swap, d_materials, n_elems, n_iters);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const long duration_ms = duration_us / 1000;

    printf("Computation time: %ld ms\n", duration_ms);

    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter_ms = (static_cast<double>(duration_us) / 1000.0) / n_measured_iters;
    const double time_per_iter_s = static_cast<double>(duration_us) / 1e6 / n_measured_iters;
    const double giga_elems_per_sec = (static_cast<double>(n_elems) / time_per_iter_s) / 1e9;
    const double gflops = giga_elems_per_sec * 22.0;

    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter_ms);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);

    // Download results
    std::vector<ElementDynamic> h_results;
    downloadResults(d_dyn, h_results, n_elems);

    const uint64_t hash_val = computeHash(h_results);
    printf("  Result hash: %016lX\n", hash_val);
    printf("\n");

    // Print results for external validation
    if (printResults) {
        std::vector<double> energyData;
        energyData.reserve(h_results.size());
        for (const auto& elem : h_results) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }

    // Validation
    if (validate) {
        bool valid = validateResults(h_results);
        if (!valid) {
            return 1;
        }
    }

    // Cleanup
    freeSoAStatic(d_static);
    freeSoADynamic(d_dyn);
    freeSoADynamic(d_dyn_swap);
    cudaFree(d_materials);

    return 0;
}
