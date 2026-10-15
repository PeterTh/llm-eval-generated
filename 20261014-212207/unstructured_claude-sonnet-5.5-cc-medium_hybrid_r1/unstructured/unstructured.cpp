#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA implementation.
//  - MPI: the mesh is partitioned into contiguous blocks of grid rows; each rank
//    owns one block and exchanges one halo row of energies with each neighbour.
//  - CUDA: each rank builds its mesh partition on its GPU and runs every update
//    there (SoA layout, gather over explicit connectivity). Boundary rows are
//    computed first so the halo exchange overlaps with the interior kernel.
//  - OpenMP: host-side work (halo buffer staging, result gathering, reductions).

using idx_t = uint64_t;
using val_t = double;

// Connections per element actually used by the 2D grid (up, down, left, right)
constexpr int MAX_CONNECTIONS = 4;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Material type IDs
constexpr int DEFAULT_MAT_ID = 0;
constexpr int INFLOW_MAT_ID = 1;
constexpr int OUTFLOW_MAT_ID = 2;

__constant__ Material d_materials[3];

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Partition of the mesh owned by this rank. Local element index of global
// (x, y) is (x - row_begin + 1) * n + y; rows 0 and nloc+1 are halo rows.
struct Partition {
    int n = 0;          // grid size (n x n)
    int row_begin = 0;  // first owned global row
    int nloc = 0;       // number of owned rows
    __host__ __device__ size_t owned() const { return (size_t)nloc * n; }
    size_t extended() const { return (size_t)(nloc + 2) * n; }
};

// Device-side world state (SoA)
struct DeviceWorld {
    uint8_t* material_idx = nullptr;
    uint8_t* num_connections = nullptr;
    int* connected_idx = nullptr;       // [MAX_CONNECTIONS][owned], local extended indices
    val_t* connected_flux = nullptr;    // [MAX_CONNECTIONS][owned]
    val_t* energy[2] = {nullptr, nullptr};  // extended (with halo rows)
    val_t* total_flux = nullptr;        // owned
};

// Build connectivity for the owned elements directly on the device
__global__ void buildKernel(const Partition p, int n_global, uint8_t* mat, uint8_t* ncon,
                            int* cidx, val_t* cflux) {
    const size_t owned = p.owned();
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < owned;
         i += (size_t)gridDim.x * blockDim.x) {
        const int lx = (int)(i / p.n);
        const int y = (int)(i % p.n);
        const int x = p.row_begin + lx;
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        int c = 0;
        for (int k = 0; k < 4; ++k) {
            const int nx = x + offsets[k][0];
            const int ny = y + offsets[k][1];
            if (nx >= 0 && nx < n_global && ny >= 0 && ny < n_global) {
                cidx[(size_t)c * owned + i] = (nx - p.row_begin + 1) * p.n + ny;
                cflux[(size_t)c * owned + i] = 1.0;
                ++c;
            }
        }
        ncon[i] = (uint8_t)c;
        int m = DEFAULT_MAT_ID;
        const int last = n_global - 1;
        if ((x == 0 && y == 0) || (x == last && y == last)) m = INFLOW_MAT_ID;
        else if ((x == 0 && y == last) || (x == last && y == 0)) m = OUTFLOW_MAT_ID;
        mat[i] = (uint8_t)m;
    }
}

// Update owned rows [lrow_begin, lrow_end)
__global__ void updateKernel(const Partition p, int lrow_begin, int lrow_end,
                             const uint8_t* __restrict__ mat_idx,
                             const uint8_t* __restrict__ ncon,
                             const int* __restrict__ cidx,
                             const val_t* __restrict__ cflux,
                             const val_t* __restrict__ e_in, val_t* __restrict__ e_out,
                             val_t* __restrict__ flux_acc) {
    const size_t owned = p.owned();
    const size_t first = (size_t)lrow_begin * p.n;
    const size_t last = (size_t)lrow_end * p.n;
    for (size_t i = first + blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < last;
         i += (size_t)gridDim.x * blockDim.x) {
        const size_t ei = i + p.n;  // extended index
        const Material mat = d_materials[mat_idx[i]];
        const val_t e = e_in[ei];
        val_t total = mat.external_flow;
        const int nc = ncon[i];
#pragma unroll
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            if (j < nc) {
                const val_t other = __ldg(&e_in[cidx[(size_t)j * owned + i]]);
                total += (other - e) * mat.transfer_coeff * cflux[(size_t)j * owned + i] * 0.25;
            }
        }
        e_out[ei] = e + total;
        flux_acc[i] += fabs(total);
    }
}

__global__ void hashKernel(const Partition p, const val_t* e, const val_t* f,
                           unsigned long long* out) {
    unsigned long long h = 0;
    const size_t owned = p.owned();
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < owned;
         i += (size_t)gridDim.x * blockDim.x) {
        const uint64_t gi = (uint64_t)p.row_begin * p.n + i;
        const uint64_t eb = __double_as_longlong(e[i + p.n]);
        const uint64_t fb = __double_as_longlong(f[i]);
        h ^= (eb + gi) * 0x9e3779b97f4a7c15ULL;
        h ^= (fb + gi) * 0xbf58476d1ce4e5b9ULL;
    }
    for (int o = 16; o > 0; o >>= 1) h ^= __shfl_xor_sync(0xffffffffu, h, o);
    if ((threadIdx.x & 31) == 0) atomicXor(out, h);
}

constexpr int STAT_BLOCK = 256;
constexpr int STAT_BLOCKS = 256;

// Per-block partial sums/min/max: partial[0..3][block] = esum, fsum, emin, emax
__global__ void statsKernel(const Partition p, const val_t* e, const val_t* f, val_t* partial) {
    __shared__ val_t sh[4][STAT_BLOCK];
    val_t es = 0, fs = 0, mn = DBL_MAX, mx = -DBL_MAX;
    const size_t owned = p.owned();
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < owned;
         i += (size_t)gridDim.x * blockDim.x) {
        const val_t v = e[i + p.n];
        es += v;
        fs += f[i];
        mn = fmin(mn, v);
        mx = fmax(mx, v);
    }
    sh[0][threadIdx.x] = es; sh[1][threadIdx.x] = fs;
    sh[2][threadIdx.x] = mn; sh[3][threadIdx.x] = mx;
    __syncthreads();
    for (int s = STAT_BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh[0][threadIdx.x] += sh[0][threadIdx.x + s];
            sh[1][threadIdx.x] += sh[1][threadIdx.x + s];
            sh[2][threadIdx.x] = fmin(sh[2][threadIdx.x], sh[2][threadIdx.x + s]);
            sh[3][threadIdx.x] = fmax(sh[3][threadIdx.x], sh[3][threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int k = 0; k < 4; ++k) partial[(size_t)k * gridDim.x + blockIdx.x] = sh[k][0];
}

static int gridFor(size_t count, int block = 256) {
    size_t b = (count + block - 1) / block;
    return (int)std::max<size_t>(1, std::min<size_t>(b, 1 << 20));
}

// Run simulation for n_iters iterations. Result ends up in dw.energy[cur] (returned).
int runSimulation(DeviceWorld& dw, const Partition& p, int n_iters, MPI_Comm comm, int rank,
                  int nranks, int n_active) {
    const int n = p.n;
    int cur = 0;
    if (p.nloc == 0) return cur;

    const int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int down = (rank < n_active - 1) ? rank + 1 : MPI_PROC_NULL;
    (void)nranks;

    cudaStream_t s_bnd, s_int;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_bnd, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_int, cudaStreamNonBlocking));
    cudaEvent_t ev_done;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_done, cudaEventDisableTiming));

    val_t *send_up, *send_dn, *recv_up, *recv_dn;
    CUDA_CHECK(cudaMallocHost(&send_up, n * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&send_dn, n * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&recv_up, n * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&recv_dn, n * sizeof(val_t)));

    const int bs = 256;
    auto launch = [&](int r0, int r1, cudaStream_t s, int c) {
        if (r1 <= r0) return;
        const size_t cnt = (size_t)(r1 - r0) * n;
        updateKernel<<<gridFor(cnt, bs), bs, 0, s>>>(p, r0, r1, dw.material_idx, dw.num_connections,
                                                    dw.connected_idx, dw.connected_flux,
                                                    dw.energy[c], dw.energy[c ^ 1], dw.total_flux);
    };

    for (int iter = 0; iter < n_iters; ++iter) {
        const int nxt = cur ^ 1;
        const bool exchange = (iter + 1 < n_iters);
        const bool two_rows = p.nloc > 1;

        // Boundary rows first (top row 0, bottom row nloc-1)
        launch(0, 1, s_bnd, cur);
        if (two_rows) launch(p.nloc - 1, p.nloc, s_bnd, cur);
        // Interior rows overlap with the halo exchange
        launch(1, p.nloc - 1, s_int, cur);

        if (exchange) {
            val_t* en = dw.energy[nxt];
            CUDA_CHECK(cudaMemcpyAsync(send_up, en + n, n * sizeof(val_t), cudaMemcpyDeviceToHost, s_bnd));
            CUDA_CHECK(cudaMemcpyAsync(send_dn, en + (size_t)p.nloc * n, n * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, s_bnd));
            CUDA_CHECK(cudaStreamSynchronize(s_bnd));

            MPI_Request reqs[4];
            MPI_Irecv(recv_up, n, MPI_DOUBLE, up, 0, comm, &reqs[0]);
            MPI_Irecv(recv_dn, n, MPI_DOUBLE, down, 1, comm, &reqs[1]);
            MPI_Isend(send_up, n, MPI_DOUBLE, up, 1, comm, &reqs[2]);
            MPI_Isend(send_dn, n, MPI_DOUBLE, down, 0, comm, &reqs[3]);
            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

            if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(en, recv_up, n * sizeof(val_t), cudaMemcpyHostToDevice, s_bnd));
            if (down != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(en + (size_t)(p.nloc + 1) * n, recv_dn, n * sizeof(val_t),
                                           cudaMemcpyHostToDevice, s_bnd));
        }
        CUDA_CHECK(cudaEventRecord(ev_done, s_int));
        CUDA_CHECK(cudaStreamWaitEvent(s_bnd, ev_done, 0));
        CUDA_CHECK(cudaStreamSynchronize(s_bnd));
        cur = nxt;
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeHost(send_up));
    CUDA_CHECK(cudaFreeHost(send_dn));
    CUDA_CHECK(cudaFreeHost(recv_up));
    CUDA_CHECK(cudaFreeHost(recv_dn));
    CUDA_CHECK(cudaEventDestroy(ev_done));
    CUDA_CHECK(cudaStreamDestroy(s_bnd));
    CUDA_CHECK(cudaStreamDestroy(s_int));
    return cur;
}

// Validate simulation results (global stats already reduced)
bool validateResults(val_t energy_sum, val_t flux_sum, val_t energy_min, val_t energy_max) {
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
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

    // Select GPU by node-local rank
    int local_rank = 0;
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        MPI_Comm_rank(local, &local_rank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(local_rank % ndev));
    }

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", nranks, omp_get_max_threads());
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    // Row partition: only ranks that receive at least one row are active
    const int n_active = std::max(1, std::min(nranks, n_elems_root));
    Partition p;
    p.n = n_elems_root;
    {
        const int base = n_elems_root / n_active, rem = n_elems_root % n_active;
        if (rank < n_active) {
            p.nloc = base + (rank < rem ? 1 : 0);
            p.row_begin = rank * base + std::min(rank, rem);
        }
    }

    // Build mesh on the device
    const size_t owned = p.owned();
    DeviceWorld dw;
    if (owned > 0) {
        const Material h_mats[3] = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
        CUDA_CHECK(cudaMemcpyToSymbol(d_materials, h_mats, sizeof(h_mats)));
        CUDA_CHECK(cudaMalloc(&dw.material_idx, owned));
        CUDA_CHECK(cudaMalloc(&dw.num_connections, owned));
        CUDA_CHECK(cudaMalloc(&dw.connected_idx, MAX_CONNECTIONS * owned * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dw.connected_flux, MAX_CONNECTIONS * owned * sizeof(val_t)));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMalloc(&dw.energy[b], p.extended() * sizeof(val_t)));
            CUDA_CHECK(cudaMemset(dw.energy[b], 0, p.extended() * sizeof(val_t)));
        }
        CUDA_CHECK(cudaMalloc(&dw.total_flux, owned * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(dw.total_flux, 0, owned * sizeof(val_t)));
        buildKernel<<<gridFor(owned), 256>>>(p, n_elems_root, dw.material_idx, dw.num_connections,
                                            dw.connected_idx, dw.connected_flux);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Memory usage (global)
    {
        const size_t per_static = 8 + 8 + 8 * 8 + 8 * 8;  // as in the reference layout
        const size_t static_mem = (size_t)n_elems * per_static;
        const size_t dynamic_mem = (size_t)n_elems * 16 * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        if (rank == 0) {
            printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
                   total_mem / (1024.0 * 1024.0), static_mem / (1024.0 * 1024.0),
                   dynamic_mem / (1024.0 * 1024.0));
            printf("\n");
            printf("Running simulation...\n");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int cur = runSimulation(dw, p, n_iters, MPI_COMM_WORLD, rank, nranks, n_active);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (double)n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Hash of the results (XOR is order independent, so per-rank hashes combine exactly)
    unsigned long long local_hash = 0, global_hash = 0;
    if (owned > 0) {
        unsigned long long* d_hash;
        CUDA_CHECK(cudaMalloc(&d_hash, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_hash, 0, sizeof(unsigned long long)));
        hashKernel<<<gridFor(owned), 256>>>(p, dw.energy[cur], dw.total_flux, d_hash);
        CUDA_CHECK(cudaMemcpy(&local_hash, d_hash, sizeof(local_hash), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_hash));
    }
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", (unsigned long)global_hash);
        printf("\n");
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> local_energy(owned);
        if (owned > 0)
            CUDA_CHECK(cudaMemcpy(local_energy.data(), dw.energy[cur] + p.n, owned * sizeof(val_t),
                                  cudaMemcpyDeviceToHost));
        std::vector<long long> counts_ll(nranks);
        long long mine = (long long)owned;
        MPI_Gather(&mine, 1, MPI_LONG_LONG, counts_ll.data(), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        std::vector<double> energyData;
        if (rank == 0) {
            energyData.resize((size_t)n_elems);
            // Gather row blocks in chunks of rows to avoid int overflow of counts
            size_t off = 0;
            for (int r = 0; r < nranks; ++r) {
                const size_t cnt = (size_t)counts_ll[r];
                if (cnt == 0) continue;
                if (r == 0) {
                    std::copy(local_energy.begin(), local_energy.end(), energyData.begin());
                } else {
                    // received in n-sized row pieces
                    for (size_t o = 0; o < cnt; o += (size_t)1 << 28) {
                        const int c = (int)std::min<size_t>((size_t)1 << 28, cnt - o);
                        MPI_Recv(energyData.data() + off + o, c, MPI_DOUBLE, r, 100, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                    }
                }
                off += cnt;
            }
        } else {
            for (size_t o = 0; o < owned; o += (size_t)1 << 28) {
                const int c = (int)std::min<size_t>((size_t)1 << 28, owned - o);
                MPI_Send(local_energy.data() + o, c, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
            }
        }
        if (rank == 0) print_results(energyData, "ElementEnergy");
    }

    // Validation
    int valid_flag = 1;
    if (validate) {
        val_t part[4] = {0.0, 0.0, std::numeric_limits<val_t>::max(),
                         std::numeric_limits<val_t>::lowest()};
        if (owned > 0) {
            val_t* d_part;
            CUDA_CHECK(cudaMalloc(&d_part, 4 * STAT_BLOCKS * sizeof(val_t)));
            statsKernel<<<STAT_BLOCKS, STAT_BLOCK>>>(p, dw.energy[cur], dw.total_flux, d_part);
            std::vector<val_t> h(4 * STAT_BLOCKS);
            CUDA_CHECK(cudaMemcpy(h.data(), d_part, h.size() * sizeof(val_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_part));
            double es = 0, fs = 0, mn = part[2], mx = part[3];
#pragma omp parallel for reduction(+ : es, fs) reduction(min : mn) reduction(max : mx)
            for (int b = 0; b < STAT_BLOCKS; ++b) {
                es += h[b];
                fs += h[STAT_BLOCKS + b];
                mn = std::min(mn, h[2 * STAT_BLOCKS + b]);
                mx = std::max(mx, h[3 * STAT_BLOCKS + b]);
            }
            part[0] = es; part[1] = fs; part[2] = mn; part[3] = mx;
        }
        val_t g[4];
        MPI_Reduce(&part[0], &g[0], 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&part[2], &g[2], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&part[3], &g[3], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) valid_flag = validateResults(g[0], g[1], g[2], g[3]) ? 1 : 0;
        MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (owned > 0) {
        cudaFree(dw.material_idx);
        cudaFree(dw.num_connections);
        cudaFree(dw.connected_idx);
        cudaFree(dw.connected_flux);
        cudaFree(dw.energy[0]);
        cudaFree(dw.energy[1]);
        cudaFree(dw.total_flux);
    }
    MPI_Finalize();
    return valid_flag ? 0 : 1;
}
