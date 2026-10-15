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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

__constant__ Material d_materials[3];

// Rank-local part of the mesh: a contiguous block of grid rows. Element indices
// in the connectivity are local, in a space [halo_lo | owned | halo_hi] where
// each halo is one grid row (n_root elements).
struct LocalMesh {
    int n_root = 0;
    int rank = 0, size = 1;
    idx_t row_begin = 0;          // first owned grid row
    idx_t n_rows = 0;             // owned grid rows
    idx_t n_owned = 0;            // owned elements
    int max_conn = 0;
    std::vector<uint8_t> material;     // [n_owned]
    std::vector<uint8_t> nconn;        // [n_owned]
    std::vector<uint32_t> cidx;        // [max_conn][n_owned] local indices
    std::vector<val_t> cflux;          // [max_conn][n_owned]
    std::vector<idx_t> row_counts;     // rows per rank
};

static void partitionRows(LocalMesh& m, int n_root, int rank, int size) {
    m.n_root = n_root;
    m.rank = rank;
    m.size = size;
    m.row_counts.resize(size);
    const idx_t base = n_root / size, rem = n_root % size;
    idx_t begin = 0;
    for (int r = 0; r < size; ++r) {
        m.row_counts[r] = base + (static_cast<idx_t>(r) < rem ? 1 : 0);
        if (r < rank) begin += m.row_counts[r];
    }
    m.row_begin = begin;
    m.n_rows = m.row_counts[rank];
    m.n_owned = m.n_rows * n_root;
}

// Build the rank-local part of the 2D square grid as an unstructured mesh
static void buildSquare2D(LocalMesh& m) {
    const long n = m.n_root;
    const long last = n - 1;
    const idx_t N = m.n_owned;
    m.max_conn = 4;
    m.material.assign(N, DEFAULT_MAT_ID);
    m.nconn.assign(N, 0);
    m.cidx.assign(N * m.max_conn, 0);
    m.cflux.assign(N * m.max_conn, 0.0);

    const long halo_origin = static_cast<long>(m.row_begin) * n - n;  // global idx of local 0
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

#pragma omp parallel for schedule(static)
    for (long k = 0; k < static_cast<long>(N); ++k) {
        const long gi = static_cast<long>(m.row_begin) * n + k;
        const long x = gi / n, y = gi % n;
        int nc = 0;
        for (int o = 0; o < 4; ++o) {
            const long nx = x + offsets[o][0], ny = y + offsets[o][1];
            if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                m.cidx[nc * N + k] = static_cast<uint32_t>(nx * n + ny - halo_origin);
                m.cflux[nc * N + k] = 1.0;
                nc++;
            }
        }
        m.nconn[k] = static_cast<uint8_t>(nc);
        idx_t mat = DEFAULT_MAT_ID;
        if ((x == 0 && y == 0) || (x == last && y == last)) mat = INFLOW_MAT_ID;
        else if ((x == 0 && y == last) || (x == last && y == 0)) mat = OUTFLOW_MAT_ID;
        m.material[k] = static_cast<uint8_t>(mat);
    }
}

// One simulation step for owned elements [begin, end). Arrays e_* are indexed
// in the halo-extended local space; owned element k lives at e[k + halo].
__global__ void stepKernel(const uint8_t* __restrict__ material,
                           const uint8_t* __restrict__ nconn,
                           const uint32_t* __restrict__ cidx,
                           const val_t* __restrict__ cflux,
                           const val_t* __restrict__ e_in,
                           val_t* __restrict__ e_out,
                           const val_t* __restrict__ f_in,
                           val_t* __restrict__ f_out,
                           idx_t N, idx_t halo, idx_t begin, idx_t end) {
    const idx_t stride = static_cast<idx_t>(gridDim.x) * blockDim.x;
    for (idx_t k = begin + static_cast<idx_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < end;
         k += stride) {
        const Material mat = d_materials[material[k]];
        const val_t self = e_in[k + halo];
        val_t total_flux = mat.external_flow;
        const int nc = nconn[k];
        for (int j = 0; j < nc; ++j) {
            const uint32_t nb = cidx[j * N + k];
            const val_t other = e_in[nb];
            total_flux += (other - self) * mat.transfer_coeff * cflux[j * N + k] * 0.25;
        }
        e_out[k + halo] = self + total_flux;
        f_out[k] = f_in[k] + fabs(total_flux);
    }
}

struct DeviceState {
    uint8_t *material = nullptr, *nconn = nullptr;
    uint32_t* cidx = nullptr;
    val_t* cflux = nullptr;
    val_t* e[2] = {nullptr, nullptr};
    val_t* f[2] = {nullptr, nullptr};
};

static void launchStep(const LocalMesh& m, const DeviceState& d, int cur, idx_t begin, idx_t end,
                       cudaStream_t s) {
    if (end <= begin) return;
    const int threads = 256;
    idx_t blocks = (end - begin + threads - 1) / threads;
    const idx_t max_blocks = 1 << 20;
    if (blocks > max_blocks) blocks = max_blocks;
    stepKernel<<<static_cast<unsigned>(blocks), threads, 0, s>>>(
        d.material, d.nconn, d.cidx, d.cflux, d.e[cur], d.e[1 - cur], d.f[cur], d.f[1 - cur],
        m.n_owned, m.n_root, begin, end);
    CUDA_CHECK(cudaGetLastError());
}

// Run simulation for n_iters iterations. Results are copied back to host.
// Returns the buffer index holding the final state in host_energy/host_flux.
static void runSimulation(const LocalMesh& m, int n_iters, std::vector<val_t>& host_energy,
                          std::vector<val_t>& host_flux, double& elapsed_s) {
    const idx_t n = m.n_root;
    const idx_t N = m.n_owned;
    const idx_t E = N + 2 * n;

    DeviceState d;
    CUDA_CHECK(cudaMalloc(&d.material, N + 1));
    CUDA_CHECK(cudaMalloc(&d.nconn, N + 1));
    CUDA_CHECK(cudaMalloc(&d.cidx, (N * m.max_conn + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d.cflux, (N * m.max_conn + 1) * sizeof(val_t)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&d.e[b], E * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d.f[b], (N + 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(d.e[b], 0, E * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(d.f[b], 0, (N + 1) * sizeof(val_t)));
    }
    const Material h_mats[3] = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    CUDA_CHECK(cudaMemcpyToSymbol(d_materials, h_mats, sizeof(h_mats)));
    if (N) {
        CUDA_CHECK(cudaMemcpy(d.material, m.material.data(), N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.nconn, m.nconn.data(), N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.cidx, m.cidx.data(), N * m.max_conn * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.cflux, m.cflux.data(), N * m.max_conn * sizeof(val_t),
                              cudaMemcpyHostToDevice));
    }

    // Neighbours (ranks with zero rows are only at the tail and are skipped)
    const int up = (m.rank > 0 && m.n_rows > 0) ? m.rank - 1 : MPI_PROC_NULL;
    const int down = (m.rank + 1 < m.size && m.row_counts[m.rank + 1] > 0 && m.n_rows > 0)
                         ? m.rank + 1 : MPI_PROC_NULL;

    val_t *send_up, *send_down, *recv_up, *recv_down;
    CUDA_CHECK(cudaMallocHost(&send_up, (n + 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&send_down, (n + 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&recv_up, (n + 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&recv_down, (n + 1) * sizeof(val_t)));

    cudaStream_t s_bnd, s_int;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_bnd, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_int, cudaStreamNonBlocking));

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::high_resolution_clock::now();

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int nxt = 1 - cur;
        const bool exchange = (iter + 1 < n_iters);
        // Boundary rows first so the halo exchange overlaps interior compute
        if (N > 0) {
            launchStep(m, d, cur, 0, n, s_bnd);
            if (m.n_rows > 1) launchStep(m, d, cur, N - n, N, s_bnd);
            if (m.n_rows > 2) launchStep(m, d, cur, n, N - n, s_int);
        }
        if (exchange && N > 0) {
            if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(send_up, d.e[nxt] + n, n * sizeof(val_t),
                                           cudaMemcpyDeviceToHost, s_bnd));
            if (down != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(send_down, d.e[nxt] + N, n * sizeof(val_t),
                                           cudaMemcpyDeviceToHost, s_bnd));
            CUDA_CHECK(cudaStreamSynchronize(s_bnd));
            MPI_Request reqs[4];
            int nr = 0;
            if (up != MPI_PROC_NULL) {
                MPI_Irecv(recv_up, static_cast<int>(n), MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &reqs[nr++]);
                MPI_Isend(send_up, static_cast<int>(n), MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &reqs[nr++]);
            }
            if (down != MPI_PROC_NULL) {
                MPI_Irecv(recv_down, static_cast<int>(n), MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &reqs[nr++]);
                MPI_Isend(send_down, static_cast<int>(n), MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &reqs[nr++]);
            }
            MPI_Waitall(nr, reqs, MPI_STATUSES_IGNORE);
            if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(d.e[nxt], recv_up, n * sizeof(val_t),
                                           cudaMemcpyHostToDevice, s_bnd));
            if (down != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(d.e[nxt] + n + N, recv_down, n * sizeof(val_t),
                                           cudaMemcpyHostToDevice, s_bnd));
        }
        CUDA_CHECK(cudaStreamSynchronize(s_bnd));
        CUDA_CHECK(cudaStreamSynchronize(s_int));
        cur = nxt;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::high_resolution_clock::now();
    elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    host_energy.resize(N);
    host_flux.resize(N);
    if (N) {
        CUDA_CHECK(cudaMemcpy(host_energy.data(), d.e[cur] + n, N * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_flux.data(), d.f[cur], N * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaStreamDestroy(s_bnd));
    CUDA_CHECK(cudaStreamDestroy(s_int));
    cudaFreeHost(send_up); cudaFreeHost(send_down); cudaFreeHost(recv_up); cudaFreeHost(recv_down);
    cudaFree(d.material); cudaFree(d.nconn); cudaFree(d.cidx); cudaFree(d.cflux);
    for (int b = 0; b < 2; ++b) { cudaFree(d.e[b]); cudaFree(d.f[b]); }
}

// Validate simulation results (collective; prints on rank 0)
static bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux,
                            int rank) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    const long N = static_cast<long>(energy.size());
#pragma omp parallel for reduction(+ : energy_sum, flux_sum) reduction(max : energy_max) \
    reduction(min : energy_min)
    for (long i = 0; i < N; ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
    }
    val_t v[2] = {energy_sum, flux_sum}, r[2];
    MPI_Allreduce(v, r, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    energy_sum = r[0];
    flux_sum = r[1];
    MPI_Allreduce(MPI_IN_PLACE, &energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    bool ok = true;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        constexpr val_t energy_epsilon = 1e-8;
        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            ok = false;
        } else {
            if (std::abs(energy_sum) > energy_epsilon)
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            if (!std::isfinite(flux_sum)) {
                printf("  ERROR: Flux sum is not finite\n");
                ok = false;
            } else if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
                printf("  ERROR: Energy extrema are not finite\n");
                ok = false;
            }
        }
        if (ok) printf("  Validation: PASSED\n");
    }
    return ok;
}

// Hash of local results; global element index i = first_idx + local index.
// Per-element terms are XOR-combined, so the result is order independent.
static uint64_t computeHash(const std::vector<val_t>& energy, const std::vector<val_t>& flux,
                            uint64_t first_idx) {
    uint64_t hash = 0;
    const long N = static_cast<long>(energy.size());
#pragma omp parallel for reduction(^ : hash)
    for (long k = 0; k < N; ++k) {
        const uint64_t i = first_idx + k;
        uint64_t e, f;
        memcpy(&e, &energy[k], sizeof(e));
        memcpy(&f, &flux[k], sizeof(f));
        hash ^= (e + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    {
        // One GPU per rank, round-robin over the GPUs of the node
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank, ndev = 0;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
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
    
    const size_t n_elems = static_cast<size_t>(n_elems_root) * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", size, omp_get_max_threads());
        printf("\n");
        printf("Building unstructured mesh...\n");
    }
    LocalMesh mesh;
    partitionRows(mesh, n_elems_root, rank, size);
    buildSquare2D(mesh);

    if (rank == 0) {
        const size_t static_mem = n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    std::vector<val_t> energy, flux;
    double elapsed_s = 0.0;
    runSimulation(mesh, n_iters, energy, flux, elapsed_s);

    const long duration_ms = static_cast<long>(elapsed_s * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    uint64_t hash = computeHash(energy, flux, mesh.row_begin * n_elems_root);
    MPI_Allreduce(MPI_IN_PLACE, &hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    if (printResults) {
        std::vector<int> counts(size), displs(size);
        int off = 0;
        for (int r = 0; r < size; ++r) {
            counts[r] = static_cast<int>(mesh.row_counts[r] * n_elems_root);
            displs[r] = off;
            off += counts[r];
        }
        std::vector<double> energyData;
        if (rank == 0) energyData.resize(n_elems);
        MPI_Gatherv(energy.data(), static_cast<int>(energy.size()), MPI_DOUBLE, energyData.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(energyData, "ElementEnergy");
    }

    int ret = 0;
    if (validate) {
        if (!validateResults(energy, flux, rank)) ret = 1;
    }

    MPI_Finalize();
    return ret;
}
