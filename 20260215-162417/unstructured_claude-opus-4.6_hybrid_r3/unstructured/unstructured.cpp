#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
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

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: each thread updates one local element
__global__ void simulationKernel(
    const int n_local,
    const idx_t* __restrict__ d_mat_idx,
    const idx_t* __restrict__ d_nconn,
    const idx_t* __restrict__ d_conn_idx,
    const val_t* __restrict__ d_conn_flux,
    const val_t* __restrict__ d_mat_tc,
    const val_t* __restrict__ d_mat_ef,
    const val_t* __restrict__ d_cur_energy,
    const val_t* __restrict__ d_cur_flux,
    val_t* __restrict__ d_new_energy,
    val_t* __restrict__ d_new_flux)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_local) return;

    idx_t mat = d_mat_idx[i];
    val_t tc = d_mat_tc[mat];
    val_t ef = d_mat_ef[mat];
    val_t my_energy = d_cur_energy[i];
    idx_t nc = d_nconn[i];

    val_t flux = ef;
    size_t base = (size_t)i * MAX_CONNECTIONS;
    for (idx_t j = 0; j < nc; ++j) {
        idx_t ni = d_conn_idx[base + j];
        val_t cf = d_conn_flux[base + j];
        flux += (__ldg(&d_cur_energy[ni]) - my_energy) * tc * cf * 0.25;
    }

    d_new_energy[i] = my_energy + flux;
    d_new_flux[i] = d_cur_flux[i] + fabs(flux);
}

// GPU gather: pack boundary elements for halo send
__global__ void gatherKernel(const val_t* __restrict__ src,
    const int* __restrict__ indices, val_t* __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[indices[i]];
}

// GPU scatter: unpack received ghost data
__global__ void scatterKernel(const val_t* __restrict__ src,
    const int* __restrict__ positions, val_t* __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[positions[i]] = src[i];
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

    // Assign GPU based on node-local rank
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &node_comm);
    int local_rank;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
    CUDA_CHECK(cudaSetDevice(local_rank % num_devices));

    // Build the full mesh on all ranks (with OpenMP)
    if (rank == 0) printf("Building unstructured mesh...\n");

    std::vector<Material> materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    int n_mat = (int)materials.size();

    std::vector<ElementStatic> all_static(n_elems);
    std::vector<ElementDynamic> all_dynamic(n_elems);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_elems; ++i) {
        all_static[i].material_idx = DEFAULT_MAT_ID;
        all_static[i].num_connections = 0;
        all_dynamic[i].current_energy = 0.0;
        all_dynamic[i].total_flux = 0.0;
    }

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < n_elems; ++idx) {
        int x = idx / n_elems_root, y = idx % n_elems_root;
        ElementStatic& elem = all_static[idx];
        const int dx[] = {1, -1, 0, 0};
        const int dy[] = {0, 0, 1, -1};
        for (int n = 0; n < 4; ++n) {
            int nx = x + dx[n], ny = y + dy[n];
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                elem.connected_idx[elem.num_connections] = nx * n_elems_root + ny;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }

    const int last = n_elems_root - 1;
    all_static[0].material_idx = INFLOW_MAT_ID;
    all_static[last].material_idx = OUTFLOW_MAT_ID;
    all_static[(size_t)last * n_elems_root].material_idx = OUTFLOW_MAT_ID;
    all_static[(size_t)last * n_elems_root + last].material_idx = INFLOW_MAT_ID;

    // Domain decomposition: contiguous chunks across MPI ranks
    int local_start = (int)((long long)rank * n_elems / nprocs);
    int local_end   = (int)((long long)(rank + 1) * n_elems / nprocs);
    int n_local = local_end - local_start;

    // Identify ghost elements (neighbors owned by other ranks)
    std::set<int> ghost_set;
    for (int i = local_start; i < local_end; ++i) {
        for (idx_t j = 0; j < all_static[i].num_connections; ++j) {
            int gidx = (int)all_static[i].connected_idx[j];
            if (gidx < local_start || gidx >= local_end)
                ghost_set.insert(gidx);
        }
    }
    std::vector<int> ghost_global(ghost_set.begin(), ghost_set.end());
    int n_ghost = (int)ghost_global.size();
    int n_total_local = n_local + n_ghost;

    // Global-to-local index mapping (dense array for O(1) lookup)
    std::vector<int> g2l(n_elems, -1);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i)
        g2l[local_start + i] = i;
    for (int i = 0; i < n_ghost; ++i)
        g2l[ghost_global[i]] = n_local + i;

    // Build SoA arrays for GPU
    std::vector<idx_t> h_mat_idx(n_local), h_nconn(n_local);
    std::vector<idx_t> h_conn_idx((size_t)n_local * MAX_CONNECTIONS, 0);
    std::vector<val_t> h_conn_flux((size_t)n_local * MAX_CONNECTIONS, 0.0);
    std::vector<val_t> h_energy(n_total_local, 0.0);
    std::vector<val_t> h_flux(n_local, 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i) {
        int gi = local_start + i;
        h_mat_idx[i] = all_static[gi].material_idx;
        h_nconn[i] = all_static[gi].num_connections;
        for (idx_t j = 0; j < all_static[gi].num_connections; ++j) {
            size_t base = (size_t)i * MAX_CONNECTIONS + j;
            h_conn_idx[base] = g2l[(int)all_static[gi].connected_idx[j]];
            h_conn_flux[base] = all_static[gi].connected_flux[j];
        }
        h_energy[i] = all_dynamic[gi].current_energy;
        h_flux[i] = all_dynamic[gi].total_flux;
    }
    for (int i = 0; i < n_ghost; ++i)
        h_energy[n_local + i] = all_dynamic[ghost_global[i]].current_energy;

    // Build MPI communication schedule
    // Determine owning rank for each ghost element
    auto ownerOf = [&](int gidx) -> int {
        int r = (int)((long long)gidx * nprocs / n_elems);
        while (r > 0 && gidx < (int)((long long)r * n_elems / nprocs)) r--;
        while (r < nprocs - 1 && gidx >= (int)((long long)(r + 1) * n_elems / nprocs)) r++;
        return r;
    };

    // recv_ghost_pos[r]: local positions where ghost data from rank r goes
    // req_owner_idx[r]: owner-local indices of elements we need from rank r
    std::vector<std::vector<int>> recv_ghost_pos(nprocs);
    std::vector<std::vector<int>> req_owner_idx(nprocs);
    for (int i = 0; i < n_ghost; ++i) {
        int gidx = ghost_global[i];
        int owner = ownerOf(gidx);
        int owner_start = (int)((long long)owner * n_elems / nprocs);
        recv_ghost_pos[owner].push_back(n_local + i);
        req_owner_idx[owner].push_back(gidx - owner_start);
    }

    // Exchange request counts
    std::vector<int> n_recv(nprocs), n_send(nprocs);
    for (int r = 0; r < nprocs; ++r)
        n_recv[r] = (int)req_owner_idx[r].size();
    MPI_Alltoall(n_recv.data(), 1, MPI_INT, n_send.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    // Compute displacements
    std::vector<int> recv_disp(nprocs), send_disp(nprocs);
    int total_recv = 0, total_send = 0;
    for (int r = 0; r < nprocs; ++r) {
        recv_disp[r] = total_recv; total_recv += n_recv[r];
        send_disp[r] = total_send; total_send += n_send[r];
    }

    // Exchange requested indices via Alltoallv
    // I send my requests (owner-local indices I need) to each owner rank
    // I receive from each rank: my local indices they need
    std::vector<int> req_send_buf(total_recv);
    for (int r = 0; r < nprocs; ++r)
        for (int i = 0; i < n_recv[r]; ++i)
            req_send_buf[recv_disp[r] + i] = req_owner_idx[r][i];

    std::vector<int> send_indices(total_send);
    MPI_Alltoallv(req_send_buf.data(), n_recv.data(), recv_disp.data(), MPI_INT,
                  send_indices.data(), n_send.data(), send_disp.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // Flatten recv ghost positions for scatter
    std::vector<int> recv_pos_flat(total_recv);
    for (int r = 0; r < nprocs; ++r)
        for (int i = 0; i < n_recv[r]; ++i)
            recv_pos_flat[recv_disp[r] + i] = recv_ghost_pos[r][i];

    // Calculate memory usage
    if (rank == 0) {
        const size_t static_mem = (size_t)n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = (size_t)n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Free full mesh (no longer needed)
    { std::vector<ElementStatic>().swap(all_static); }
    { std::vector<ElementDynamic>().swap(all_dynamic); }
    { std::vector<int>().swap(g2l); }

    // Allocate GPU memory
    idx_t *d_mat_idx, *d_nconn, *d_conn_idx;
    val_t *d_conn_flux, *d_mat_tc, *d_mat_ef;
    val_t *d_energy_a, *d_flux_a, *d_energy_b, *d_flux_b;

    CUDA_CHECK(cudaMalloc(&d_mat_idx, n_local * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_nconn, n_local * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_conn_idx, (size_t)n_local * MAX_CONNECTIONS * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_conn_flux, (size_t)n_local * MAX_CONNECTIONS * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_mat_tc, n_mat * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_mat_ef, n_mat * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_energy_a, n_total_local * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_flux_a, n_local * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_energy_b, n_total_local * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_flux_b, n_local * sizeof(val_t)));

    // Copy static data to GPU
    CUDA_CHECK(cudaMemcpy(d_mat_idx, h_mat_idx.data(),
        n_local * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nconn, h_nconn.data(),
        n_local * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conn_idx, h_conn_idx.data(),
        (size_t)n_local * MAX_CONNECTIONS * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conn_flux, h_conn_flux.data(),
        (size_t)n_local * MAX_CONNECTIONS * sizeof(val_t), cudaMemcpyHostToDevice));

    std::vector<val_t> h_mat_tc(n_mat), h_mat_ef(n_mat);
    for (int i = 0; i < n_mat; ++i) {
        h_mat_tc[i] = materials[i].transfer_coeff;
        h_mat_ef[i] = materials[i].external_flow;
    }
    CUDA_CHECK(cudaMemcpy(d_mat_tc, h_mat_tc.data(),
        n_mat * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mat_ef, h_mat_ef.data(),
        n_mat * sizeof(val_t), cudaMemcpyHostToDevice));

    // Copy initial dynamic data
    CUDA_CHECK(cudaMemcpy(d_energy_a, h_energy.data(),
        n_total_local * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flux_a, h_flux.data(),
        n_local * sizeof(val_t), cudaMemcpyHostToDevice));

    // Allocate GPU halo buffers and index arrays
    int *d_send_indices = nullptr, *d_recv_pos = nullptr;
    val_t *d_halo_send = nullptr, *d_halo_recv = nullptr;
    if (nprocs > 1) {
        if (total_send > 0) {
            CUDA_CHECK(cudaMalloc(&d_send_indices, total_send * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_send_indices, send_indices.data(),
                total_send * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_halo_send, total_send * sizeof(val_t)));
        }
        if (total_recv > 0) {
            CUDA_CHECK(cudaMalloc(&d_recv_pos, total_recv * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_recv_pos, recv_pos_flat.data(),
                total_recv * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_halo_recv, total_recv * sizeof(val_t)));
        }
    }

    std::vector<val_t> halo_send_buf(total_send);
    std::vector<val_t> halo_recv_buf(total_recv);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int block_size = 256;
    const int grid_size = (n_local + block_size - 1) / block_size;

    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    val_t *d_ce = d_energy_a, *d_cf = d_flux_a;
    val_t *d_ne = d_energy_b, *d_nf = d_flux_b;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Launch simulation kernel on GPU
        if (grid_size > 0) {
            simulationKernel<<<grid_size, block_size, 0, stream>>>(
                n_local, d_mat_idx, d_nconn, d_conn_idx, d_conn_flux,
                d_mat_tc, d_mat_ef, d_ce, d_cf, d_ne, d_nf);
        }

        // Halo exchange (skip on last iteration - no next iteration needs it)
        if (nprocs > 1 && iter < n_iters - 1) {
            // GPU gather: pack boundary elements into send buffer
            if (total_send > 0) {
                int gs = (total_send + block_size - 1) / block_size;
                gatherKernel<<<gs, block_size, 0, stream>>>(
                    d_ne, d_send_indices, d_halo_send, total_send);
                CUDA_CHECK(cudaMemcpyAsync(halo_send_buf.data(), d_halo_send,
                    total_send * sizeof(val_t), cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // MPI exchange
            MPI_Alltoallv(halo_send_buf.data(), n_send.data(), send_disp.data(),
                          MPI_DOUBLE, halo_recv_buf.data(), n_recv.data(),
                          recv_disp.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            // GPU scatter: place ghost data into new energy buffer
            if (total_recv > 0) {
                CUDA_CHECK(cudaMemcpyAsync(d_halo_recv, halo_recv_buf.data(),
                    total_recv * sizeof(val_t), cudaMemcpyHostToDevice, stream));
                int gs = (total_recv + block_size - 1) / block_size;
                scatterKernel<<<gs, block_size, 0, stream>>>(
                    d_halo_recv, d_recv_pos, d_ne, total_recv);
            }
        }

        // Swap buffers
        std::swap(d_ce, d_ne);
        std::swap(d_cf, d_nf);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long long local_duration_ms = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    long long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0,
               MPI_COMM_WORLD);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_energy.data(), d_ce,
        n_local * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flux.data(), d_cf,
        n_local * sizeof(val_t), cudaMemcpyDeviceToHost));

    // Gather all results to rank 0
    std::vector<int> all_counts(nprocs), all_disps(nprocs);
    MPI_Gather(&n_local, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        all_disps[0] = 0;
        for (int r = 1; r < nprocs; ++r)
            all_disps[r] = all_disps[r-1] + all_counts[r-1];
    }

    std::vector<val_t> all_energy, all_flux;
    if (rank == 0) { all_energy.resize(n_elems); all_flux.resize(n_elems); }

    MPI_Gatherv(h_energy.data(), n_local, MPI_DOUBLE,
                rank == 0 ? all_energy.data() : nullptr,
                rank == 0 ? all_counts.data() : nullptr,
                rank == 0 ? all_disps.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(h_flux.data(), n_local, MPI_DOUBLE,
                rank == 0 ? all_flux.data() : nullptr,
                rank == 0 ? all_counts.data() : nullptr,
                rank == 0 ? all_disps.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int ret = 0;

    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (double)n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Compute hash for verification (same as original)
        uint64_t hash = 0;
        for (int i = 0; i < n_elems; ++i) {
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&all_energy[i]);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&all_flux[i]);
            hash ^= (*e_ptr + (uint64_t)i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (*f_ptr + (uint64_t)i) * 0xbf58476d1ce4e5b9ULL;
        }
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData(all_energy.begin(), all_energy.end());
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            val_t energy_sum = 0.0;
            val_t flux_sum = 0.0;
            val_t energy_max = std::numeric_limits<val_t>::lowest();
            val_t energy_min = std::numeric_limits<val_t>::max();

            for (int i = 0; i < n_elems; ++i) {
                energy_sum += all_energy[i];
                flux_sum += all_flux[i];
                energy_max = std::max(all_energy[i], energy_max);
                energy_min = std::min(all_energy[i], energy_min);
            }

            printf("Validation results:\n");
            printf("  Energy sum: %.12f\n", energy_sum);
            printf("  Flux sum: %.2f\n", flux_sum);
            printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

            constexpr val_t energy_epsilon = 1e-8;
            bool valid = true;

            if (!std::isfinite(energy_sum)) {
                printf("  ERROR: Energy sum is not finite\n");
                valid = false;
            }
            if (std::abs(energy_sum) > energy_epsilon) {
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            }
            if (!std::isfinite(flux_sum)) {
                printf("  ERROR: Flux sum is not finite\n");
                valid = false;
            }
            if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
                printf("  ERROR: Energy extrema are not finite\n");
                valid = false;
            }

            if (valid) printf("  Validation: PASSED\n");
            else ret = 1;
        }
    }

    // Broadcast return code so all ranks exit consistently
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_mat_idx));
    CUDA_CHECK(cudaFree(d_nconn));
    CUDA_CHECK(cudaFree(d_conn_idx));
    CUDA_CHECK(cudaFree(d_conn_flux));
    CUDA_CHECK(cudaFree(d_mat_tc));
    CUDA_CHECK(cudaFree(d_mat_ef));
    CUDA_CHECK(cudaFree(d_energy_a));
    CUDA_CHECK(cudaFree(d_flux_a));
    CUDA_CHECK(cudaFree(d_energy_b));
    CUDA_CHECK(cudaFree(d_flux_b));
    if (d_send_indices) CUDA_CHECK(cudaFree(d_send_indices));
    if (d_halo_send) CUDA_CHECK(cudaFree(d_halo_send));
    if (d_recv_pos) CUDA_CHECK(cudaFree(d_recv_pos));
    if (d_halo_recv) CUDA_CHECK(cudaFree(d_halo_recv));

    MPI_Finalize();
    return ret;
}
