#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: compute flux and update energy for each local element
__global__ void computeFluxKernel(
    const double* __restrict__ current_energy,
    const double* __restrict__ total_flux_in,
    const int*    __restrict__ material_idx,
    const int*    __restrict__ num_connections,
    const int*    __restrict__ connected_idx,
    const double* __restrict__ connected_flux,
    const double* __restrict__ materials_transfer,
    const double* __restrict__ materials_flow,
    double* __restrict__ out_energy,
    double* __restrict__ out_total_flux,
    int n_local_start,
    int n_local_end)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + n_local_start;
    if (i >= n_local_end) return;

    int mat = material_idx[i];
    double transfer = materials_transfer[mat];
    double flow = materials_flow[mat];
    double energy = current_energy[i];

    double total = flow;
    int nc = num_connections[i];
    const int* nbr_idx = &connected_idx[i * MAX_CONNECTIONS];
    const double* nbr_flux = &connected_flux[i * MAX_CONNECTIONS];

    #pragma unroll
    for (int j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j >= nc) break;
        total += (current_energy[nbr_idx[j]] - energy) * transfer * nbr_flux[j] * 0.25;
    }

    out_energy[i] = energy + total;
    out_total_flux[i] = total_flux_in[i] + fabs(total);
}

// Distributed world state — each MPI rank owns a row slice
struct DistributedWorld {
    // Materials (host)
    std::vector<Material> materials;

    // Host arrays (size n_total = ghost_top + local + ghost_bottom)
    std::vector<int>    h_material_idx;
    std::vector<int>    h_num_connections;
    std::vector<int>    h_connected_idx;       // [n_total][MAX_CONNECTIONS]
    std::vector<double> h_connected_flux;      // [n_total][MAX_CONNECTIONS]
    std::vector<double> h_current_energy;
    std::vector<double> h_total_flux;
    std::vector<double> h_swap_energy;
    std::vector<double> h_swap_flux;

    // Device arrays (same layout)
    int*    d_material_idx;
    int*    d_num_connections;
    int*    d_connected_idx;
    double* d_connected_flux;
    double* d_current_energy;
    double* d_total_flux;
    double* d_swap_energy;
    double* d_swap_flux;
    double* d_materials_transfer;
    double* d_materials_flow;

    // Pinned host buffers for MPI ghost exchange
    double *h_send_top, *h_send_bot, *h_recv_top, *h_recv_bot;

    // Dimensions
    int n_elems_root;
    int n_local_rows;
    int n_local;          // local elements (no ghosts)
    int n_ghost;          // ghost elements per side (= n_elems_root)
    int n_total;          // n_ghost + n_local + n_ghost
    int x_start, x_end;   // global row range for this rank
};

// Build the local portion of the grid (row slice) with OpenMP parallelism
void buildLocalGrid(DistributedWorld& w, int n_elems_root,
                    int x_start, int x_end, int rank, int n_ranks) {
    w.n_elems_root = n_elems_root;
    w.x_start = x_start;
    w.x_end = x_end;
    w.n_local_rows = x_end - x_start;
    w.n_ghost = n_elems_root;
    w.n_local = w.n_local_rows * n_elems_root;
    w.n_total = w.n_ghost + w.n_local + w.n_ghost;

    // Materials
    w.materials.clear();
    w.materials.emplace_back(Material{0.8, 0.0});
    w.materials.emplace_back(Material{0.8, 0.5});
    w.materials.emplace_back(Material{0.8, -0.5});

    // Allocate host arrays
    w.h_material_idx.assign(w.n_total, 0);
    w.h_num_connections.assign(w.n_total, 0);
    w.h_connected_idx.assign(w.n_total * MAX_CONNECTIONS, -1);
    w.h_connected_flux.assign(w.n_total * MAX_CONNECTIONS, 0.0);
    w.h_current_energy.assign(w.n_total, 0.0);
    w.h_total_flux.assign(w.n_total, 0.0);
    w.h_swap_energy.assign(w.n_total, 0.0);
    w.h_swap_flux.assign(w.n_total, 0.0);

    // Build connectivity for local elements (excluding ghosts)
    #pragma omp parallel for collapse(2)
    for (int rx = 0; rx < w.n_local_rows; ++rx) {
        for (int y = 0; y < n_elems_root; ++y) {
            int x = x_start + rx;
            int local_idx = w.n_ghost + rx * n_elems_root + y;

            w.h_material_idx[local_idx] = static_cast<int>(DEFAULT_MAT_ID);
            int& nc = w.h_num_connections[local_idx];

            static const int dx[4] = {1, -1, 0, 0};
            static const int dy[4] = {0, 0, 1, -1};

            for (int n = 0; n < 4; ++n) {
                int nx = x + dx[n];
                int ny = y + dy[n];
                if (nx < 0 || nx >= n_elems_root || ny < 0 || ny >= n_elems_root)
                    continue;

                int nbr_local;
                if (nx < x_start) {
                    nbr_local = ny;   // top ghost
                } else if (nx >= x_end) {
                    nbr_local = w.n_ghost + w.n_local + ny;   // bottom ghost
                } else {
                    int nrx = nx - x_start;
                    nbr_local = w.n_ghost + nrx * n_elems_root + ny;
                }

                int c = nc++;
                w.h_connected_idx[local_idx * MAX_CONNECTIONS + c] = nbr_local;
                w.h_connected_flux[local_idx * MAX_CONNECTIONS + c] = 1.0;
            }
        }
    }

    // Set corner elements as inflow/outflow
    const int last = n_elems_root - 1;
    if (x_start == 0) {
        w.h_material_idx[w.n_ghost + 0] = static_cast<int>(INFLOW_MAT_ID);
        w.h_material_idx[w.n_ghost + last] = static_cast<int>(OUTFLOW_MAT_ID);
    }
    if (x_end == n_elems_root) {
        int last_row_start = w.n_ghost + (w.n_local_rows - 1) * n_elems_root;
        w.h_material_idx[last_row_start + 0] = static_cast<int>(OUTFLOW_MAT_ID);
        w.h_material_idx[last_row_start + last] = static_cast<int>(INFLOW_MAT_ID);
    }
}

// Allocate GPU memory and upload static data
void initGPU(DistributedWorld& w) {
    size_t mat_sz = w.materials.size();
    size_t n_total = static_cast<size_t>(w.n_total);
    size_t conn_sz = n_total * MAX_CONNECTIONS;

    CUDA_CHECK(cudaMalloc(&w.d_material_idx,     n_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&w.d_num_connections,  n_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&w.d_connected_idx,     conn_sz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&w.d_connected_flux,    conn_sz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.d_current_energy,   n_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.d_total_flux,       n_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.d_swap_energy,      n_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.d_swap_flux,        n_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.d_materials_transfer, mat_sz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.d_materials_flow,    mat_sz * sizeof(double)));

    // Copy static data to device
    CUDA_CHECK(cudaMemcpy(w.d_material_idx,    w.h_material_idx.data(),   n_total * sizeof(int),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w.d_num_connections, w.h_num_connections.data(), n_total * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w.d_connected_idx,   w.h_connected_idx.data(),  conn_sz * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w.d_connected_flux,  w.h_connected_flux.data(), conn_sz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w.d_current_energy,  w.h_current_energy.data(), n_total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w.d_total_flux,      w.h_total_flux.data(),     n_total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w.d_swap_energy,     w.h_swap_energy.data(),    n_total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w.d_swap_flux,       w.h_swap_flux.data(),      n_total * sizeof(double), cudaMemcpyHostToDevice));

    // Upload material arrays
    std::vector<double> h_transfer(w.materials.size());
    std::vector<double> h_flow(w.materials.size());
    for (size_t i = 0; i < w.materials.size(); ++i) {
        h_transfer[i] = w.materials[i].transfer_coeff;
        h_flow[i] = w.materials[i].external_flow;
    }
    CUDA_CHECK(cudaMemcpy(w.d_materials_transfer, h_transfer.data(), mat_sz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w.d_materials_flow,     h_flow.data(),     mat_sz * sizeof(double), cudaMemcpyHostToDevice));

    // Pinned host buffers for MPI ghost exchange
    size_t row_sz = static_cast<size_t>(w.n_ghost) * sizeof(double);
    CUDA_CHECK(cudaMallocHost(&w.h_send_top, row_sz));
    CUDA_CHECK(cudaMallocHost(&w.h_send_bot, row_sz));
    CUDA_CHECK(cudaMallocHost(&w.h_recv_top, row_sz));
    CUDA_CHECK(cudaMallocHost(&w.h_recv_bot, row_sz));
}

// Exchange ghost rows between MPI ranks
void exchangeGhosts(DistributedWorld& w, int rank, int n_ranks) {
    int n_elems_root = w.n_ghost;
    size_t row_bytes = static_cast<size_t>(n_elems_root) * sizeof(double);

    // Copy boundary rows from GPU to pinned host
    CUDA_CHECK(cudaMemcpy(w.h_send_top,
        w.d_current_energy + w.n_ghost,
        row_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(w.h_send_bot,
        w.d_current_energy + w.n_ghost + w.n_local - n_elems_root,
        row_bytes, cudaMemcpyDeviceToHost));

    // Non-blocking MPI exchange
    MPI_Request reqs[4];
    int nreq = 0;

    if (rank > 0) {
        // Send first local row to rank-1 (their bottom ghost)
        // Receive from rank-1 (their last row -> our top ghost)
        MPI_Isend(w.h_send_top, n_elems_root, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(w.h_recv_top, n_elems_root, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    if (rank + 1 < n_ranks) {
        // Send last local row to rank+1 (their top ghost)
        // Receive from rank+1 (their first row -> our bottom ghost)
        MPI_Isend(w.h_send_bot, n_elems_root, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(w.h_recv_bot, n_elems_root, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
    }

    if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // Copy received ghost data to GPU
    if (rank > 0) {
        CUDA_CHECK(cudaMemcpy(w.d_current_energy,
            w.h_recv_top, row_bytes, cudaMemcpyHostToDevice));
    }
    if (rank + 1 < n_ranks) {
        CUDA_CHECK(cudaMemcpy(w.d_current_energy + w.n_ghost + w.n_local,
            w.h_recv_bot, row_bytes, cudaMemcpyHostToDevice));
    }
}

// Free GPU memory and pinned buffers
void cleanupGPU(DistributedWorld& w) {
    CUDA_CHECK(cudaFree(w.d_material_idx));
    CUDA_CHECK(cudaFree(w.d_num_connections));
    CUDA_CHECK(cudaFree(w.d_connected_idx));
    CUDA_CHECK(cudaFree(w.d_connected_flux));
    CUDA_CHECK(cudaFree(w.d_current_energy));
    CUDA_CHECK(cudaFree(w.d_total_flux));
    CUDA_CHECK(cudaFree(w.d_swap_energy));
    CUDA_CHECK(cudaFree(w.d_swap_flux));
    CUDA_CHECK(cudaFree(w.d_materials_transfer));
    CUDA_CHECK(cudaFree(w.d_materials_flow));
    CUDA_CHECK(cudaFreeHost(w.h_send_top));
    CUDA_CHECK(cudaFreeHost(w.h_send_bot));
    CUDA_CHECK(cudaFreeHost(w.h_recv_top));
    CUDA_CHECK(cudaFreeHost(w.h_recv_bot));
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
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // Pin MPI processes to GPUs in round-robin
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    if (n_devices > 0) {
        int dev_id = rank % n_devices;
        cudaSetDevice(dev_id);
    }

    // Parse command line arguments (all ranks parse)
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

    // Row decomposition across MPI ranks
    int base = n_elems_root / n_ranks;
    int rem  = n_elems_root % n_ranks;
    int n_local_rows = base + (rank < rem ? 1 : 0);
    int x_start = rank * base + (rank < rem ? rank : rem);
    int x_end   = x_start + n_local_rows;

    // Ensure at least one row per rank
    if (n_local_rows == 0) {
        if (rank == 0)
            fprintf(stderr, "Error: too many MPI ranks (%d) for grid size %d\n",
                    n_ranks, n_elems_root);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int n_elems_global = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("=====================================================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems_global);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", n_ranks);
        printf("GPU devices available: %d\n", n_devices);
        int nthreads = 1;
        #pragma omp parallel
        { nthreads = omp_get_num_threads(); }
        printf("OpenMP threads per rank: %d\n", nthreads);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Build local mesh
    if (rank == 0) printf("Building unstructured mesh (distributed)...\n");
    DistributedWorld world;
    buildLocalGrid(world, n_elems_root, x_start, x_end, rank, n_ranks);

    // Allocate GPU memory and upload data
    initGPU(world);

    // Calculate memory usage (rank 0 only)
    if (rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems_global) *
            (sizeof(Material) * 3 + sizeof(int) * 2 + sizeof(int) * MAX_CONNECTIONS + sizeof(double) * MAX_CONNECTIONS);
        const size_t dynamic_mem = static_cast<size_t>(n_elems_global) * sizeof(double) * 4;
        const size_t total_mem = static_mem + dynamic_mem;
        // per-rank GPU memory
        size_t gpu_mem = static_cast<size_t>(world.n_total) *
            (sizeof(int) + sizeof(int) + sizeof(int) * MAX_CONNECTIONS + sizeof(double) * (4 + MAX_CONNECTIONS))
            + sizeof(double) * 2 * world.materials.size();
        printf("Memory: %.2f MB (host-global), ~%.2f MB (GPU per rank)\n",
               total_mem / (1024.0 * 1024.0), gpu_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Warm-up iteration
    {
        exchangeGhosts(world, rank, n_ranks);
        constexpr int THREADS = 256;
        int blocks = (world.n_local + THREADS - 1) / THREADS;
        computeFluxKernel<<<blocks, THREADS>>>(
            world.d_current_energy, world.d_total_flux,
            world.d_material_idx, world.d_num_connections,
            world.d_connected_idx, world.d_connected_flux,
            world.d_materials_transfer, world.d_materials_flow,
            world.d_swap_energy, world.d_swap_flux,
            world.n_ghost, world.n_ghost + world.n_local);
        cudaDeviceSynchronize();
        std::swap(world.d_current_energy, world.d_swap_energy);
        std::swap(world.d_total_flux, world.d_swap_flux);
    }

    // Reset state for timing
    CUDA_CHECK(cudaMemcpy(world.d_current_energy, world.h_current_energy.data(),
        static_cast<size_t>(world.n_total) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_total_flux, world.h_total_flux.data(),
        static_cast<size_t>(world.n_total) * sizeof(double), cudaMemcpyHostToDevice));

    // Timed simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < n_iters; ++iter) {
        exchangeGhosts(world, rank, n_ranks);

        constexpr int THREADS = 256;
        int blocks = (world.n_local + THREADS - 1) / THREADS;
        computeFluxKernel<<<blocks, THREADS>>>(
            world.d_current_energy, world.d_total_flux,
            world.d_material_idx, world.d_num_connections,
            world.d_connected_idx, world.d_connected_flux,
            world.d_materials_transfer, world.d_materials_flow,
            world.d_swap_energy, world.d_swap_flux,
            world.n_ghost, world.n_ghost + world.n_local);
        cudaDeviceSynchronize();

        std::swap(world.d_current_energy, world.d_swap_energy);
        std::swap(world.d_total_flux, world.d_swap_flux);
    }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long global_duration_ms = 0;
    MPI_Reduce(&duration_ms, &global_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(world.h_current_energy.data(), world.d_current_energy,
        static_cast<size_t>(world.n_total) * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(world.h_total_flux.data(), world.d_total_flux,
        static_cast<size_t>(world.n_total) * sizeof(double), cudaMemcpyDeviceToHost));

    // Distributed validation
    if (validate) {
        // Compute local reduction
        val_t local_energy_sum = 0.0;
        val_t local_flux_sum = 0.0;
        val_t local_energy_max = std::numeric_limits<val_t>::lowest();
        val_t local_energy_min = std::numeric_limits<val_t>::max();

        for (int i = world.n_ghost; i < world.n_ghost + world.n_local; ++i) {
            val_t e = world.h_current_energy[i];
            val_t f = world.h_total_flux[i];
            local_energy_sum += e;
            local_flux_sum += f;
            local_energy_max = std::max(local_energy_max, e);
            local_energy_min = std::min(local_energy_min, e);
        }

        val_t global_energy_sum = 0.0, global_flux_sum = 0.0;
        val_t global_energy_max = 0.0, global_energy_min = 0.0;
        MPI_Reduce(&local_energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_flux_sum,   &global_flux_sum,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validation results:\n");
            printf("  Energy sum: %.12f\n", global_energy_sum);
            printf("  Flux sum: %.2f\n", global_flux_sum);
            printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);

            constexpr val_t energy_epsilon = 1e-8;
            bool valid = true;
            if (!std::isfinite(global_energy_sum)) {
                printf("  ERROR: Energy sum is not finite\n");
                valid = false;
            }
            if (std::abs(global_energy_sum) > energy_epsilon)
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            if (!std::isfinite(global_flux_sum)) {
                printf("  ERROR: Flux sum is not finite\n");
                valid = false;
            }
            if (!std::isfinite(global_energy_max) || !std::isfinite(global_energy_min)) {
                printf("  ERROR: Energy extrema are not finite\n");
                valid = false;
            }
            if (valid)
                printf("  Validation: PASSED\n");
            else {
                printf("  Validation: FAILED\n");
                cleanupGPU(world);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Compute hash using global indices (XOR across ranks is associative)
    uint64_t local_hash = 0;
    for (int rx = 0; rx < world.n_local_rows; ++rx) {
        for (int y = 0; y < n_elems_root; ++y) {
            int global_idx = (world.x_start + rx) * n_elems_root + y;
            int local_idx = world.n_ghost + rx * n_elems_root + y;
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.h_current_energy[local_idx]);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.h_total_flux[local_idx]);
            local_hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
        }
    }

    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int n_measured_iters = std::max(n_iters - 1, 1);
        double tpi = static_cast<double>(global_duration_ms) / n_measured_iters;
        double gelem_s = (n_measured_iters * n_elems_global) / (global_duration_ms / 1000.0) / 1e9;
        double gflops = gelem_s * 22.0;

        printf("Computation time: %lld ms (max across ranks)\n", global_duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", tpi);
        printf("  Elements/sec: %.4f GigaElements/s\n", gelem_s);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", global_hash);
        printf("\n");
    }

    // Print results for external validation (gather energy data to rank 0)
    if (printResults) {
        std::vector<int> recv_counts(n_ranks, 0);
        std::vector<int> displs(n_ranks, 0);
        int local_n = world.n_local;
        MPI_Gather(&local_n, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            int total = 0;
            for (int r = 0; r < n_ranks; ++r) {
                displs[r] = total;
                total += recv_counts[r];
            }
            std::vector<double> all_energy(total);

            std::vector<double> own_data(world.n_local);
            #pragma omp parallel for
            for (int i = 0; i < world.n_local; ++i)
                own_data[i] = world.h_current_energy[world.n_ghost + i];

            MPI_Gatherv(own_data.data(), world.n_local, MPI_DOUBLE,
                        all_energy.data(), recv_counts.data(), displs.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);

            print_results(all_energy, "ElementEnergy");
        } else {
            std::vector<double> own_data(world.n_local);
            #pragma omp parallel for
            for (int i = 0; i < world.n_local; ++i)
                own_data[i] = world.h_current_energy[world.n_ghost + i];
            MPI_Gatherv(own_data.data(), world.n_local, MPI_DOUBLE,
                        nullptr, nullptr, nullptr,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    // Cleanup
    cleanupGPU(world);

    MPI_Finalize();
    return 0;
}
