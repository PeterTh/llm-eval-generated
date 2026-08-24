#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID  = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// ---------------------------------------------------------------------------
// CUDA kernel: one thread per owned element.
//
// Local energy buffer layout (n_local = (x_count+2)*grid_size):
//   [0 .. grid_size-1]                    : top halo row
//   [grid_size .. grid_size+n_owned-1]    : owned rows
//   [grid_size+n_owned .. n_local-1]      : bottom halo row
//
// Kernel reads energy[li] (local index including halo offset) and writes
// energy_swap[li] for owned elements.  Flux arrays are flat (n_owned, no halo).
// ---------------------------------------------------------------------------
__global__ void simulationKernel(
    const double*  __restrict__ energy,
    const double*  __restrict__ flux,
    double*        __restrict__ energy_swap,
    double*        __restrict__ flux_swap,
    const uint64_t* __restrict__ mat_idx,
    const uint64_t* __restrict__ n_conn,
    const uint64_t* __restrict__ conn_idx,
    const double*  __restrict__ conn_flux,
    const double*  __restrict__ mat_transfer,
    const double*  __restrict__ mat_external,
    int grid_size,
    int x_start,
    size_t n_owned)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_owned) return;

    int    gx = x_start + static_cast<int>(i / grid_size);
    int    gy = static_cast<int>(i % grid_size);
    size_t li = static_cast<size_t>(gx - x_start + 1) * grid_size + gy;

    uint64_t mat_id = mat_idx[i];
    double   tc     = mat_transfer[mat_id];
    double   ef     = mat_external[mat_id];

    double this_e     = energy[li];
    double this_f     = flux[i];
    double total_flux = ef;

    uint64_t nc   = n_conn[i];
    size_t   base = i * MAX_CONNECTIONS;
    for (uint64_t j = 0; j < nc; j++) {
        uint64_t ng   = conn_idx[base + j];
        double   cf   = conn_flux[base + j];
        int      ngx  = static_cast<int>(ng / grid_size);
        int      ngy  = static_cast<int>(ng % grid_size);
        size_t   nl   = static_cast<size_t>(ngx - x_start + 1) * grid_size + ngy;
        total_flux += (energy[nl] - this_e) * tc * cf * 0.25;
    }

    energy_swap[li] = this_e + total_flux;
    flux_swap[i]    = this_f + fabs(total_flux);
}

// ---------------------------------------------------------------------------
// Helpers: find which MPI rank owns a given global row / boundary.
// ---------------------------------------------------------------------------
static int findRankForRow(int row, int N, int P) {
    int bx = N / P, rem = N % P;
    for (int r = 0; r < P; r++) {
        int rs = r * bx + std::min(r, rem);
        int rc = bx + (r < rem ? 1 : 0);
        if (row >= rs && row < rs + rc) return r;
    }
    return -1;
}

static int findRankNeedingBottomHalo(int row, int N, int P) {
    int bx = N / P, rem = N % P;
    for (int r = 0; r < P; r++) {
        int rs = r * bx + std::min(r, rem);
        int rc = bx + (r < rem ? 1 : 0);
        if (rs + rc == row) return r;
    }
    return -1;
}

static int findRankNeedingTopHalo(int row, int N, int P) {
    int bx = N / P, rem = N % P;
    for (int r = 0; r < P; r++) {
        int rs = r * bx + std::min(r, rem);
        if (rs == row) return r;
    }
    return -1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank_id, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // ---- parse arguments (rank 0 only, then broadcast) ------------------
    int grid_size = 512, n_iters = 10, validate_flag = 0, printResults_flag = 0;
    if (rank_id == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
                grid_size = atoi(argv[++i]);
            else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
                n_iters = atoi(argv[++i]);
            else if (strcmp(argv[i], "-v") == 0)
                validate_flag = 1;
            else if (strcmp(argv[i], "-r") == 0)
                printResults_flag = 1;
            else if (strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
                printf("  -i <num>     Number of simulation iterations (default: 10)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
        }
    }
    MPI_Bcast(&grid_size,         1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters,           1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_flag,     1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    bool validate     = validate_flag     != 0;
    bool printResults = printResults_flag != 0;
    const int n_elems = grid_size * grid_size;

    // ---- domain decomposition (1-D along x / rows) ---------------------
    int bx      = grid_size / n_ranks;
    int rem     = grid_size % n_ranks;
    int x_start = rank_id * bx + std::min(rank_id, rem);
    int x_count = bx + (rank_id < rem ? 1 : 0);
    int n_owned = x_count * grid_size;
    int n_local = (x_count + 2) * grid_size;  // owned + 2 halo rows

    // ---- build local connectivity (OpenMP) ------------------------------
    std::vector<uint64_t> h_mat_idx(n_owned);
    std::vector<uint64_t> h_n_conn(n_owned);
    std::vector<uint64_t> h_conn_idx(n_owned * MAX_CONNECTIONS);
    std::vector<double>   h_conn_flux(n_owned * MAX_CONNECTIONS);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_owned; i++) {
        int gx = x_start + i / grid_size;
        int gy = i % grid_size;
        h_mat_idx[i] = DEFAULT_MAT_ID;
        int nc = 0;
        const int off[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
        for (int n = 0; n < 4; n++) {
            int nx = gx + off[n][0], ny = gy + off[n][1];
            if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size) {
                h_conn_idx[i * MAX_CONNECTIONS + nc] = static_cast<uint64_t>(nx * grid_size + ny);
                h_conn_flux[i * MAX_CONNECTIONS + nc] = 1.0;
                nc++;
            }
        }
        h_n_conn[i] = static_cast<uint64_t>(nc);
    }

    // corner materials
    int last = grid_size - 1;
    const int corners[4][2] = {{0,0},{0,last},{last,0},{last,last}};
    const idx_t cmat[4]     = {INFLOW_MAT_ID,OUTFLOW_MAT_ID,OUTFLOW_MAT_ID,INFLOW_MAT_ID};
    for (int c = 0; c < 4; c++) {
        if (corners[c][0] >= x_start && corners[c][0] < x_start + x_count) {
            h_mat_idx[(corners[c][0] - x_start) * grid_size + corners[c][1]] = cmat[c];
        }
    }

    constexpr double mat_tr[3] = {0.8, 0.8, 0.8};
    constexpr double mat_ex[3] = {0.0, 0.5, -0.5};

    // ---- device memory --------------------------------------------------
    double   *d_energy[2] = {nullptr, nullptr};
    double   *d_flux[2]   = {nullptr, nullptr};
    uint64_t *d_mat_idx   = nullptr;
    uint64_t *d_n_conn    = nullptr;
    uint64_t *d_conn_idx  = nullptr;
    double   *d_conn_flux = nullptr;
    double   *d_mat_tr    = nullptr;
    double   *d_mat_ex    = nullptr;
    int d_buf = 0;

    if (n_owned > 0) {
        cudaMalloc(&d_energy[0], n_local * sizeof(double));
        cudaMalloc(&d_energy[1], n_local * sizeof(double));
        cudaMalloc(&d_flux[0],   n_owned * sizeof(double));
        cudaMalloc(&d_flux[1],   n_owned * sizeof(double));
        cudaMalloc(&d_mat_idx,   n_owned * sizeof(uint64_t));
        cudaMalloc(&d_n_conn,    n_owned * sizeof(uint64_t));
        cudaMalloc(&d_conn_idx,  static_cast<size_t>(n_owned) * MAX_CONNECTIONS * sizeof(uint64_t));
        cudaMalloc(&d_conn_flux, static_cast<size_t>(n_owned) * MAX_CONNECTIONS * sizeof(double));
        cudaMalloc(&d_mat_tr,    3 * sizeof(double));
        cudaMalloc(&d_mat_ex,    3 * sizeof(double));

        cudaMemcpy(d_mat_idx,   h_mat_idx.data(),   n_owned * sizeof(uint64_t),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_n_conn,    h_n_conn.data(),    n_owned * sizeof(uint64_t),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_conn_idx,  h_conn_idx.data(),  static_cast<size_t>(n_owned) * MAX_CONNECTIONS * sizeof(uint64_t),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_conn_flux, h_conn_flux.data(), static_cast<size_t>(n_owned) * MAX_CONNECTIONS * sizeof(double),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_mat_tr,    mat_tr,             3 * sizeof(double),           cudaMemcpyHostToDevice);
        cudaMemcpy(d_mat_ex,    mat_ex,             3 * sizeof(double),           cudaMemcpyHostToDevice);

        cudaMemset(d_energy[0], 0, n_local * sizeof(double));
        cudaMemset(d_energy[1], 0, n_local * sizeof(double));
        cudaMemset(d_flux[0],   0, n_owned * sizeof(double));
        cudaMemset(d_flux[1],   0, n_owned * sizeof(double));
    }

    // ---- pre-compute MPI communication partners --------------------------
    // top_src: rank whose bottom boundary row feeds into my top halo
    // bot_src: rank whose top boundary row feeds into my bottom halo
    int top_src   = (x_start > 0)                    ? findRankForRow(x_start - 1, grid_size, n_ranks) : -1;
    int bot_src   = (x_start + x_count < grid_size)  ? findRankForRow(x_start + x_count, grid_size, n_ranks) : -1;
    // snd_first: rank that needs my first owned row as its bottom halo
    // snd_last:  rank that needs my last owned row as its top halo
    int snd_first = (x_start > 0)                    ? findRankNeedingBottomHalo(x_start, grid_size, n_ranks) : -1;
    int snd_last  = (x_start + x_count < grid_size)  ? findRankNeedingTopHalo(x_start + x_count, grid_size, n_ranks) : -1;

    // small host buffers for boundary / halo exchange
    std::vector<double> h_bnd_first(grid_size);
    std::vector<double> h_bnd_last(grid_size);
    std::vector<double> h_halo_top(grid_size);
    std::vector<double> h_halo_bot(grid_size);

    // ---- print banner (rank 0) ------------------------------------------
    if (rank_id == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", grid_size, grid_size, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallel: MPI ranks=%d, OpenMP threads=%d, CUDA\n",
               n_ranks, omp_get_max_threads());
        printf("\n");
        printf("Building unstructured mesh...\n");
        const size_t est_static = static_cast<size_t>(n_elems) *
            (sizeof(uint64_t) + sizeof(uint64_t) +
             MAX_CONNECTIONS * (sizeof(uint64_t) + sizeof(double)));
        const size_t est_dynamic = static_cast<size_t>(n_elems) * sizeof(double) * 4;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               (est_static + est_dynamic) / (1024.0 * 1024.0),
               est_static / (1024.0 * 1024.0),
               est_dynamic / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // ---- simulation loop ------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < n_iters; ++iter) {
        // --- 1. Copy boundary rows from device to host -------------------
        if (snd_first >= 0 && n_owned > 0) {
            cudaMemcpyAsync(h_bnd_first.data(),
                            d_energy[d_buf] + grid_size,
                            grid_size * sizeof(double),
                            cudaMemcpyDeviceToHost);
        }
        if (snd_last >= 0 && n_owned > 0) {
            cudaMemcpyAsync(h_bnd_last.data(),
                            d_energy[d_buf] + static_cast<size_t>(x_count) * grid_size,
                            grid_size * sizeof(double),
                            cudaMemcpyDeviceToHost);
        }
        if (n_owned > 0) cudaDeviceSynchronize();

        // --- 2. Non-blocking MPI halo exchange ---------------------------
        // Tags are chosen from the RECEIVER'S perspective:
        //   tag 10 = top halo data  (received from rank above)
        //   tag 11 = bottom halo data (received from rank below)
        MPI_Request reqs[4];
        int nreq = 0;

        if (snd_first >= 0)
            MPI_Isend(h_bnd_first.data(), grid_size, MPI_DOUBLE,
                      snd_first, 11, MPI_COMM_WORLD, &reqs[nreq++]);
        if (snd_last >= 0)
            MPI_Isend(h_bnd_last.data(), grid_size, MPI_DOUBLE,
                      snd_last, 10, MPI_COMM_WORLD, &reqs[nreq++]);
        if (top_src >= 0)
            MPI_Irecv(h_halo_top.data(), grid_size, MPI_DOUBLE,
                      top_src, 10, MPI_COMM_WORLD, &reqs[nreq++]);
        if (bot_src >= 0)
            MPI_Irecv(h_halo_bot.data(), grid_size, MPI_DOUBLE,
                      bot_src, 11, MPI_COMM_WORLD, &reqs[nreq++]);

        // --- 3. Wait for halo data to arrive -----------------------------
        if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        // --- 4. Copy received halos to device ----------------------------
        if (top_src >= 0) {
            cudaMemcpyAsync(d_energy[d_buf],
                            h_halo_top.data(),
                            grid_size * sizeof(double),
                            cudaMemcpyHostToDevice);
        }
        if (bot_src >= 0) {
            cudaMemcpyAsync(d_energy[d_buf] + static_cast<size_t>(x_count + 1) * grid_size,
                            h_halo_bot.data(),
                            grid_size * sizeof(double),
                            cudaMemcpyHostToDevice);
        }
        if (n_owned > 0) cudaDeviceSynchronize();

        // --- 5. CUDA kernel (reads d_buf, writes 1-d_buf) ---------------
        if (n_owned > 0) {
            int block = 256;
            int grid  = (n_owned + block - 1) / block;
            simulationKernel<<<grid, block>>>(
                d_energy[d_buf], d_flux[d_buf],
                d_energy[1 - d_buf], d_flux[1 - d_buf],
                d_mat_idx, d_n_conn, d_conn_idx, d_conn_flux,
                d_mat_tr, d_mat_ex,
                grid_size, x_start, n_owned);
        }

        // --- 6. Swap buffers ---------------------------------------------
        d_buf = 1 - d_buf;
    }

    if (n_owned > 0) cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    long long duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long max_duration_ms = 0;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX,
               0, MPI_COMM_WORLD);

    // ---- gather results to rank 0 for validation / output ---------------
    std::vector<double> h_energy(n_owned);
    std::vector<double> h_flux(n_owned);
    if (n_owned > 0) {
        cudaMemcpy(h_energy.data(),
                   d_energy[d_buf] + grid_size,
                   n_owned * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_flux.data(), d_flux[d_buf],
                   n_owned * sizeof(double), cudaMemcpyDeviceToHost);
    }

    std::vector<int> recvcounts(n_ranks);
    std::vector<int> displs(n_ranks);
    for (int r = 0; r < n_ranks; r++) {
        int rbx  = grid_size / n_ranks;
        int rrem = grid_size % n_ranks;
        int rc   = rbx + (r < rrem ? 1 : 0);
        recvcounts[r] = rc * grid_size;
        displs[r]     = 0;
        for (int rr = 0; rr < r; rr++) {
            int rrc = rbx + (rr < rrem ? 1 : 0);
            displs[r] += rrc * grid_size;
        }
    }

    std::vector<double> all_energy(n_elems);
    std::vector<double> all_flux(n_elems);
    MPI_Gatherv(h_energy.data(), n_owned, MPI_DOUBLE,
                all_energy.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(h_flux.data(), n_owned, MPI_DOUBLE,
                all_flux.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // ---- timing / performance (rank 0) ----------------------------------
    if (rank_id == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(max_duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (n_measured_iters * n_elems) / (max_duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        uint64_t hash = 0;
        for (int i = 0; i < n_elems; ++i) {
            const uint64_t* e_ptr =
                reinterpret_cast<const uint64_t*>(&all_energy[i]);
            const uint64_t* f_ptr =
                reinterpret_cast<const uint64_t*>(&all_flux[i]);
            hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
        }
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData(all_energy.begin(), all_energy.end());
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            val_t energy_sum = 0.0;
            val_t flux_sum   = 0.0;
            val_t energy_max = std::numeric_limits<val_t>::lowest();
            val_t energy_min = std::numeric_limits<val_t>::max();

            for (int i = 0; i < n_elems; i++) {
                energy_sum += all_energy[i];
                flux_sum   += all_flux[i];
                energy_max  = std::max(all_energy[i], energy_max);
                energy_min  = std::min(all_energy[i], energy_min);
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
                printf("  WARNING: Energy sum diverged from 0 "
                       "(expected conservation)\n");
            }
            if (!std::isfinite(flux_sum)) {
                printf("  ERROR: Flux sum is not finite\n");
                valid = false;
            }
            if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
                printf("  ERROR: Energy extrema are not finite\n");
                valid = false;
            }

            if (valid)
                printf("  Validation: PASSED\n");
            if (!valid) {
                MPI_Finalize();
                return 1;
            }
        }
    }

    // ---- cleanup --------------------------------------------------------
    if (n_owned > 0) {
        cudaFree(d_energy[0]);
        cudaFree(d_energy[1]);
        cudaFree(d_flux[0]);
        cudaFree(d_flux[1]);
        cudaFree(d_mat_idx);
        cudaFree(d_n_conn);
        cudaFree(d_conn_idx);
        cudaFree(d_conn_flux);
        cudaFree(d_mat_tr);
        cudaFree(d_mat_ex);
    }

    MPI_Finalize();
    return 0;
}
