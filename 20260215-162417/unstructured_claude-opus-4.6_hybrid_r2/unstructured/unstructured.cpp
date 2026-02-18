#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element
// SoA layout for GPU-friendly access
struct ElementStaticSoA {
    idx_t* material_idx;
    idx_t* num_connections;
    idx_t* connected_idx;   // [n_elems * MAX_CONNECTIONS] flattened
    val_t* connected_flux;  // [n_elems * MAX_CONNECTIONS] flattened
};

// Dynamic state SoA
struct ElementDynamicSoA {
    val_t* current_energy;
    val_t* total_flux;
};

// AoS versions for host-side building
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state (host)
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

// Build a 2D square grid as an unstructured mesh
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});

    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);

    // Initialize all elements with OpenMP
    #pragma omp parallel for
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity with OpenMP
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];

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

    // Set corner elements as inflow/outflow
    const int last = n_elems_root - 1;
    world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
}

// CUDA kernel: each thread processes one local element
__global__ void simulationKernel(
    const idx_t* __restrict__ material_idx,
    const idx_t* __restrict__ num_connections,
    const idx_t* __restrict__ connected_idx,
    const val_t* __restrict__ connected_flux,
    const val_t* __restrict__ mat_transfer_coeff,
    const val_t* __restrict__ mat_external_flow,
    const val_t* __restrict__ cur_energy,
    const val_t* __restrict__ cur_total_flux,
    val_t* __restrict__ new_energy,
    val_t* __restrict__ new_total_flux,
    const int n_local,
    const int global_offset)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_local) return;

    idx_t mat_id = material_idx[i];
    val_t tc = mat_transfer_coeff[mat_id];
    val_t ext = mat_external_flow[mat_id];
    idx_t nc = num_connections[i];

    val_t my_energy = cur_energy[i + global_offset];
    val_t flux = ext;

    for (idx_t j = 0; j < nc; ++j) {
        idx_t nidx = connected_idx[i * MAX_CONNECTIONS + j];
        val_t cf = connected_flux[i * MAX_CONNECTIONS + j];
        val_t neighbor_energy = cur_energy[nidx];
        flux += (neighbor_energy - my_energy) * tc * cf * 0.25;
    }

    new_energy[i] = my_energy + flux;
    new_total_flux[i] = cur_total_flux[i + global_offset] + fabs(flux);
}

// Run simulation with hybrid MPI+OpenMP+CUDA
void runSimulation(World& world, const int n_iters, int rank, int nprocs) {
    const int n_elems = (int)world.elements_static.size();
    const int n_materials = (int)world.materials.size();

    // Domain decomposition: contiguous element ranges per rank
    int base_count = n_elems / nprocs;
    int remainder = n_elems % nprocs;
    int local_start = rank * base_count + std::min(rank, remainder);
    int local_count = base_count + (rank < remainder ? 1 : 0);

    // Convert local static data to SoA for GPU
    std::vector<idx_t> h_material_idx(local_count);
    std::vector<idx_t> h_num_connections(local_count);
    std::vector<idx_t> h_connected_idx(local_count * MAX_CONNECTIONS);
    std::vector<val_t> h_connected_flux(local_count * MAX_CONNECTIONS);

    #pragma omp parallel for
    for (int i = 0; i < local_count; ++i) {
        const ElementStatic& es = world.elements_static[local_start + i];
        h_material_idx[i] = es.material_idx;
        h_num_connections[i] = es.num_connections;
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            h_connected_idx[i * MAX_CONNECTIONS + j] = es.connected_idx[j];
            h_connected_flux[i * MAX_CONNECTIONS + j] = es.connected_flux[j];
        }
    }

    // Material arrays
    std::vector<val_t> h_mat_tc(n_materials), h_mat_ef(n_materials);
    for (int m = 0; m < n_materials; ++m) {
        h_mat_tc[m] = world.materials[m].transfer_coeff;
        h_mat_ef[m] = world.materials[m].external_flow;
    }

    // Full energy/flux arrays on host
    std::vector<val_t> h_energy(n_elems, 0.0);
    std::vector<val_t> h_flux(n_elems, 0.0);

    // Allocate device memory
    idx_t *d_material_idx, *d_num_connections, *d_connected_idx;
    val_t *d_connected_flux, *d_mat_tc, *d_mat_ef;
    val_t *d_energy, *d_flux;          // full arrays (for reading neighbor data)
    val_t *d_new_energy, *d_new_flux;  // local output

    CUDA_CHECK(cudaMalloc(&d_material_idx, local_count * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_num_connections, local_count * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_connected_idx, local_count * MAX_CONNECTIONS * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_connected_flux, local_count * MAX_CONNECTIONS * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_mat_tc, n_materials * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_mat_ef, n_materials * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_energy, n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_flux, n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_new_energy, local_count * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_new_flux, local_count * sizeof(val_t)));

    // Copy static data to device
    CUDA_CHECK(cudaMemcpy(d_material_idx, h_material_idx.data(), local_count * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_connections, h_num_connections.data(), local_count * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_connected_idx, h_connected_idx.data(), local_count * MAX_CONNECTIONS * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_connected_flux, h_connected_flux.data(), local_count * MAX_CONNECTIONS * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mat_tc, h_mat_tc.data(), n_materials * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mat_ef, h_mat_ef.data(), n_materials * sizeof(val_t), cudaMemcpyHostToDevice));

    // Copy initial state to device (all zeros)
    CUDA_CHECK(cudaMemcpy(d_energy, h_energy.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flux, h_flux.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice));

    // Kernel launch config
    const int blockSize = 256;
    const int gridSize = (local_count + blockSize - 1) / blockSize;

    // Compute send/recv counts for MPI_Allgatherv
    std::vector<int> recvcounts(nprocs), displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        int rc = base_count + (r < remainder ? 1 : 0);
        int rd = r * base_count + std::min(r, remainder);
        recvcounts[r] = rc;
        displs[r] = rd;
    }

    for (int iter = 0; iter < n_iters; ++iter) {
        // Launch CUDA kernel
        simulationKernel<<<gridSize, blockSize>>>(
            d_material_idx, d_num_connections, d_connected_idx, d_connected_flux,
            d_mat_tc, d_mat_ef,
            d_energy, d_flux,
            d_new_energy, d_new_flux,
            local_count, local_start
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy local results back to host
        CUDA_CHECK(cudaMemcpy(h_energy.data() + local_start, d_new_energy,
                              local_count * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_flux.data() + local_start, d_new_flux,
                              local_count * sizeof(val_t), cudaMemcpyDeviceToHost));

        // MPI_Allgatherv to distribute updated energy/flux to all ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       h_energy.data(), recvcounts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       h_flux.data(), recvcounts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Update full arrays on device for next iteration
        CUDA_CHECK(cudaMemcpy(d_energy, h_energy.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_flux, h_flux.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    }

    // Write results back to world on all ranks (needed for validation/output on rank 0)
    #pragma omp parallel for
    for (int i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = h_energy[i];
        world.elements_dynamic[i].total_flux = h_flux[i];
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_material_idx));
    CUDA_CHECK(cudaFree(d_num_connections));
    CUDA_CHECK(cudaFree(d_connected_idx));
    CUDA_CHECK(cudaFree(d_connected_flux));
    CUDA_CHECK(cudaFree(d_mat_tc));
    CUDA_CHECK(cudaFree(d_mat_ef));
    CUDA_CHECK(cudaFree(d_energy));
    CUDA_CHECK(cudaFree(d_flux));
    CUDA_CHECK(cudaFree(d_new_energy));
    CUDA_CHECK(cudaFree(d_new_flux));
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
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Assign GPU: each rank uses GPU (rank % num_devices)
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices > 0) {
        CUDA_CHECK(cudaSetDevice(rank % num_devices));
    }

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d, CUDA devices: %d\n",
               nprocs, omp_get_max_threads(), num_devices);
        printf("\n");
    }

    // Build the unstructured mesh (all ranks build the full mesh for simplicity)
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);

    if (rank == 0) {
        const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
        const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, rank, nprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        const uint64_t hash = computeHash(world.elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(world.elements_dynamic.size());
            for (const auto& elem : world.elements_dynamic) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            bool valid = validateResults(world);
            if (!valid) {
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
