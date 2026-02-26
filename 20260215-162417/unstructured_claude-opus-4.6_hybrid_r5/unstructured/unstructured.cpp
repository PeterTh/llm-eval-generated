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
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

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
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for collapse(2) schedule(static)
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

// CUDA kernel: each thread updates one element
__global__ void updateElementsKernel(
    const int n_local,
    const int n_ghost_below,
    const double* __restrict__ cur_energy,
    const double* __restrict__ cur_flux,
    double* __restrict__ new_energy,
    double* __restrict__ new_flux,
    const idx_t* __restrict__ mat_idx,
    const idx_t* __restrict__ num_conn,
    const idx_t* __restrict__ conn_idx,
    const double* __restrict__ conn_flux,
    const double* __restrict__ transfer_coeff,
    const double* __restrict__ ext_flow)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_local) return;

    const int gi = n_ghost_below + i;
    const idx_t mat = mat_idx[i];
    const double my_e = cur_energy[gi];
    const double tc = transfer_coeff[mat];
    double flux = ext_flow[mat];

    const idx_t nc = num_conn[i];
    const idx_t base = (idx_t)i * MAX_CONNECTIONS;
    for (idx_t j = 0; j < nc; j++) {
        const idx_t ni = conn_idx[base + j];
        const double cf = conn_flux[base + j];
        flux += (cur_energy[ni] - my_e) * tc * cf * 0.25;
    }

    new_energy[gi] = my_e + flux;
    new_flux[gi] = cur_flux[gi] + fabs(flux);
}

// Validate simulation results
bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    #pragma omp parallel for reduction(+:energy_sum,flux_sum) reduction(max:energy_max) reduction(min:energy_min)
    for (size_t i = 0; i < elements.size(); i++) {
        energy_sum += elements[i].current_energy;
        flux_sum += elements[i].total_flux;
        if (elements[i].current_energy > energy_max) energy_max = elements[i].current_energy;
        if (elements[i].current_energy < energy_min) energy_min = elements[i].current_energy;
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
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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
        printf("\n");
    }

    // Assign GPU based on local rank
    int n_devices;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    CUDA_CHECK(cudaSetDevice(rank % n_devices));
    
    // Build the unstructured mesh (all ranks build full mesh for setup)
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);

    // Row-based domain decomposition for MPI
    const int rows_per_rank = n_elems_root / nranks;
    const int rem = n_elems_root % nranks;
    const int row_start = rank * rows_per_rank + std::min(rank, rem);
    const int row_end = (rank + 1) * rows_per_rank + std::min(rank + 1, rem);
    const int n_local = (row_end - row_start) * n_elems_root;
    const int g_start = row_start * n_elems_root;

    const bool has_below = (rank > 0);
    const bool has_above = (rank < nranks - 1);
    const int n_ghost_below = has_below ? n_elems_root : 0;
    const int n_ghost_above = has_above ? n_elems_root : 0;
    const int n_total = n_ghost_below + n_local + n_ghost_above;

    const int ghost_below_g = has_below ? (row_start - 1) * n_elems_root : 0;
    const int ghost_above_g = has_above ? row_end * n_elems_root : 0;

    // Convert AoS to SoA with remapped local indices for GPU
    std::vector<idx_t> h_mat_idx(n_local);
    std::vector<idx_t> h_num_conn(n_local);
    std::vector<idx_t> h_conn_idx((size_t)n_local * MAX_CONNECTIONS, 0);
    std::vector<double> h_conn_flux((size_t)n_local * MAX_CONNECTIONS, 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; i++) {
        const int gi = g_start + i;
        const ElementStatic& es = world.elements_static[gi];
        h_mat_idx[i] = es.material_idx;
        h_num_conn[i] = es.num_connections;
        for (idx_t j = 0; j < es.num_connections; j++) {
            const idx_t gnb = es.connected_idx[j];
            idx_t lnb;
            if (gnb >= (idx_t)g_start && gnb < (idx_t)(g_start + n_local)) {
                lnb = n_ghost_below + (gnb - g_start);
            } else if (has_below && gnb >= (idx_t)ghost_below_g &&
                       gnb < (idx_t)(ghost_below_g + n_elems_root)) {
                lnb = gnb - ghost_below_g;
            } else {
                lnb = n_ghost_below + n_local + (gnb - ghost_above_g);
            }
            h_conn_idx[(size_t)i * MAX_CONNECTIONS + j] = lnb;
            h_conn_flux[(size_t)i * MAX_CONNECTIONS + j] = es.connected_flux[j];
        }
    }

    // Initialize energy/flux arrays (local + ghost regions)
    std::vector<double> h_energy(n_total, 0.0);
    std::vector<double> h_flux(n_total, 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; i++) {
        h_energy[n_ghost_below + i] = world.elements_dynamic[g_start + i].current_energy;
        h_flux[n_ghost_below + i] = world.elements_dynamic[g_start + i].total_flux;
    }

    // Material SoA
    const int n_mats = (int)world.materials.size();
    std::vector<double> h_tc(n_mats), h_ef(n_mats);
    for (int i = 0; i < n_mats; i++) {
        h_tc[i] = world.materials[i].transfer_coeff;
        h_ef[i] = world.materials[i].external_flow;
    }

    // Calculate memory usage
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

    // Allocate GPU memory
    double *d_energy, *d_flux, *d_new_energy, *d_new_flux;
    idx_t *d_mat_idx, *d_num_conn, *d_conn_idx;
    double *d_conn_flux, *d_tc, *d_ef;

    CUDA_CHECK(cudaMalloc(&d_energy, n_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flux, n_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_new_energy, n_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_new_flux, n_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mat_idx, std::max(n_local, 1) * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_num_conn, std::max(n_local, 1) * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_conn_idx, std::max((size_t)n_local * MAX_CONNECTIONS, (size_t)1) * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_conn_flux, std::max((size_t)n_local * MAX_CONNECTIONS, (size_t)1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tc, n_mats * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ef, n_mats * sizeof(double)));

    CUDA_CHECK(cudaMemset(d_new_energy, 0, n_total * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_new_flux, 0, n_total * sizeof(double)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_energy, h_energy.data(), n_total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flux, h_flux.data(), n_total * sizeof(double), cudaMemcpyHostToDevice));
    if (n_local > 0) {
        CUDA_CHECK(cudaMemcpy(d_mat_idx, h_mat_idx.data(), n_local * sizeof(idx_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_num_conn, h_num_conn.data(), n_local * sizeof(idx_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conn_idx, h_conn_idx.data(), (size_t)n_local * MAX_CONNECTIONS * sizeof(idx_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conn_flux, h_conn_flux.data(), (size_t)n_local * MAX_CONNECTIONS * sizeof(double), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_tc, h_tc.data(), n_mats * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ef, h_ef.data(), n_mats * sizeof(double), cudaMemcpyHostToDevice));

    // Pinned host memory for efficient halo exchange
    double *halo_send_lo, *halo_send_hi, *halo_recv_lo, *halo_recv_hi;
    CUDA_CHECK(cudaMallocHost(&halo_send_lo, n_elems_root * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&halo_send_hi, n_elems_root * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&halo_recv_lo, n_elems_root * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&halo_recv_hi, n_elems_root * sizeof(double)));

    // Run simulation
    if (rank == 0) printf("Running simulation...\n");

    const int block_size = 256;
    const int grid_size = (n_local + block_size - 1) / block_size;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < n_iters; iter++) {
        // Copy boundary rows from GPU for MPI halo exchange
        if (has_below) {
            CUDA_CHECK(cudaMemcpy(halo_send_lo, d_energy + n_ghost_below,
                                  n_elems_root * sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (has_above) {
            CUDA_CHECK(cudaMemcpy(halo_send_hi,
                                  d_energy + n_ghost_below + n_local - n_elems_root,
                                  n_elems_root * sizeof(double), cudaMemcpyDeviceToHost));
        }

        // MPI halo exchange with neighbors
        if (has_below) {
            MPI_Sendrecv(halo_send_lo, n_elems_root, MPI_DOUBLE, rank - 1, 0,
                         halo_recv_lo, n_elems_root, MPI_DOUBLE, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (has_above) {
            MPI_Sendrecv(halo_send_hi, n_elems_root, MPI_DOUBLE, rank + 1, 1,
                         halo_recv_hi, n_elems_root, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Copy received ghost data to GPU
        if (has_below) {
            CUDA_CHECK(cudaMemcpy(d_energy, halo_recv_lo,
                                  n_elems_root * sizeof(double), cudaMemcpyHostToDevice));
        }
        if (has_above) {
            CUDA_CHECK(cudaMemcpy(d_energy + n_ghost_below + n_local, halo_recv_hi,
                                  n_elems_root * sizeof(double), cudaMemcpyHostToDevice));
        }

        // Launch CUDA kernel for element update
        if (n_local > 0) {
            updateElementsKernel<<<grid_size, block_size>>>(
                n_local, n_ghost_below,
                d_energy, d_flux, d_new_energy, d_new_flux,
                d_mat_idx, d_num_conn, d_conn_idx, d_conn_flux,
                d_tc, d_ef);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Swap buffers (pointer swap, no data movement)
        std::swap(d_energy, d_new_energy);
        std::swap(d_flux, d_new_flux);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Copy results back from GPU
    CUDA_CHECK(cudaMemcpy(h_energy.data(), d_energy, n_total * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flux.data(), d_flux, n_total * sizeof(double), cudaMemcpyDeviceToHost));

    // Extract local results
    std::vector<double> local_e(n_local), local_f(n_local);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; i++) {
        local_e[i] = h_energy[n_ghost_below + i];
        local_f[i] = h_flux[n_ghost_below + i];
    }

    // Gather all results on rank 0 via MPI
    std::vector<int> counts(nranks), offsets(nranks);
    MPI_Gather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        offsets[0] = 0;
        for (int r = 1; r < nranks; r++) offsets[r] = offsets[r-1] + counts[r-1];
    }

    std::vector<double> all_e, all_f;
    if (rank == 0) {
        all_e.resize(n_elems);
        all_f.resize(n_elems);
    }

    MPI_Gatherv(local_e.data(), n_local, MPI_DOUBLE,
                all_e.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_f.data(), n_local, MPI_DOUBLE,
                all_f.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Reconstruct ElementDynamic array for hash and validation
        std::vector<ElementDynamic> result(n_elems);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_elems; i++) {
            result[i].current_energy = all_e[i];
            result[i].total_flux = all_f[i];
        }

        // Compute hash for verification
        const uint64_t hash = computeHash(result);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData(n_elems);
            for (int i = 0; i < n_elems; i++) {
                energyData[i] = all_e[i];
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(result);
            if (!valid) {
                cudaFree(d_energy); cudaFree(d_flux);
                cudaFree(d_new_energy); cudaFree(d_new_flux);
                cudaFree(d_mat_idx); cudaFree(d_num_conn);
                cudaFree(d_conn_idx); cudaFree(d_conn_flux);
                cudaFree(d_tc); cudaFree(d_ef);
                cudaFreeHost(halo_send_lo); cudaFreeHost(halo_send_hi);
                cudaFreeHost(halo_recv_lo); cudaFreeHost(halo_recv_hi);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Cleanup GPU resources
    cudaFree(d_energy); cudaFree(d_flux);
    cudaFree(d_new_energy); cudaFree(d_new_flux);
    cudaFree(d_mat_idx); cudaFree(d_num_conn);
    cudaFree(d_conn_idx); cudaFree(d_conn_flux);
    cudaFree(d_tc); cudaFree(d_ef);
    cudaFreeHost(halo_send_lo); cudaFreeHost(halo_send_hi);
    cudaFreeHost(halo_recv_lo); cudaFreeHost(halo_recv_hi);

    MPI_Finalize();
    return 0;
}
