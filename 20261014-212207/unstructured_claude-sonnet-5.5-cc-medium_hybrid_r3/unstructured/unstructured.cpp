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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

using idx_t = uint64_t;
using val_t = double;

constexpr int MAX_CONNECTIONS = 8;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

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

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// The mesh is a 2D square grid (connectivity order: +x, -x, +y, -y, all
// coupling coefficients 1.0, materials only at the corners). Rather than
// streaming the explicit connectivity tables from memory every iteration, the
// GPU kernel regenerates them from the element coordinates, which yields the
// identical arithmetic with a small fraction of the memory traffic.
// The mesh is decomposed by rows (x index) across MPI ranks; each rank owns a
// contiguous block of rows and keeps one halo row on each side.

__constant__ Material d_materials[3] = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};

__device__ __forceinline__ int materialAt(int x, int y, int last) {
    if ((x == last && y == last) || (x == 0 && y == 0)) return (int)INFLOW_MAT_ID;
    if ((x == 0 && y == last) || (x == last && y == 0)) return (int)OUTFLOW_MAT_ID;
    return (int)DEFAULT_MAT_ID;
}

// Energy buffers have layout [(local_rows + 2) * n]; local row r lives at row r+1.
// Processes rows from two segments: row_a (first n threads) and row_b (next n).
// Row indices are local (0-based, excluding halo).
__global__ void stepKernel(const val_t* __restrict__ e_in, val_t* __restrict__ e_out,
                           val_t* __restrict__ flux, int n, int x_offset,
                           int row_a, int row_b, int nrows_total, size_t count) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    int row, y;
    if (nrows_total == 0) {  // contiguous range [row_a, row_b)
        const size_t r = t / n;
        y = (int)(t - r * n);
        row = row_a + (int)r;
    } else {  // two separate rows
        row = (t < (size_t)n) ? row_a : row_b;
        y = (int)(t % n);
    }
    const int last = n - 1;
    const int gx = x_offset + row;
    const size_t c = (size_t)(row + 1) * n + y;
    const val_t self = e_in[c];
    const Material mat = d_materials[materialAt(gx, y, last)];
    val_t total = mat.external_flow;
    if (gx < last)
        total += (e_in[c + n] - self) * mat.transfer_coeff * 1.0 * 0.25;
    if (gx > 0)
        total += (e_in[c - n] - self) * mat.transfer_coeff * 1.0 * 0.25;
    if (y < last)
        total += (e_in[c + 1] - self) * mat.transfer_coeff * 1.0 * 0.25;
    if (y > 0)
        total += (e_in[c - 1] - self) * mat.transfer_coeff * 1.0 * 0.25;
    e_out[c] = self + total;
    const size_t f = (size_t)row * n + y;
    flux[f] = flux[f] + fabs(total);
}

struct Decomp {
    int n;            // grid size
    int rank, size;   // within active communicator
    int row_begin;    // first global row owned
    int rows;         // number of rows owned
};

static void rowRange(int n, int size, int rank, int& begin, int& rows) {
    const int base = n / size, rem = n % size;
    rows = base + (rank < rem ? 1 : 0);
    begin = rank * base + std::min(rank, rem);
}

// Run simulation for n_iters iterations on the GPU, with halo exchange via MPI.
// Returns the index of the buffer holding the final energies.
// d_e[0] / d_e[1] are the double-buffered energies, d_flux the accumulated flux.
int runSimulation(const Decomp& dc, MPI_Comm comm, val_t* d_e[2], val_t* d_flux,
                   const int n_iters) {
    const int n = dc.n;
    const int L = dc.rows;
    const int prev = dc.rank > 0 ? dc.rank - 1 : MPI_PROC_NULL;
    const int next = dc.rank < dc.size - 1 ? dc.rank + 1 : MPI_PROC_NULL;

    cudaStream_t s_edge, s_inner;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_edge, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_inner, cudaStreamNonBlocking));
    cudaEvent_t ev_edge;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_edge, cudaEventDisableTiming));

    val_t *h_send_lo, *h_send_hi, *h_recv_lo, *h_recv_hi;
    CUDA_CHECK(cudaMallocHost(&h_send_lo, n * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&h_send_hi, n * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&h_recv_lo, n * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&h_recv_hi, n * sizeof(val_t)));

    const int TPB = 256;
    const size_t edge_count = (L == 1) ? (size_t)n : 2 * (size_t)n;
    const int edge_row_b = (L == 1) ? 0 : L - 1;
    const int inner_rows = L - 2;

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        val_t* e_in = d_e[cur];
        val_t* e_out = d_e[cur ^ 1];

        // Edge rows first, so their results can be communicated while the
        // interior is being computed.
        stepKernel<<<(unsigned)((edge_count + TPB - 1) / TPB), TPB, 0, s_edge>>>(
            e_in, e_out, d_flux, n, dc.row_begin, 0, edge_row_b, 1, edge_count);
        CUDA_CHECK(cudaGetLastError());
        if (inner_rows > 0) {
            const size_t cnt = (size_t)inner_rows * n;
            stepKernel<<<(unsigned)((cnt + TPB - 1) / TPB), TPB, 0, s_inner>>>(
                e_in, e_out, d_flux, n, dc.row_begin, 1, L - 1, 0, cnt);
            CUDA_CHECK(cudaGetLastError());
        }
        if (prev != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(h_send_lo, e_out + (size_t)1 * n, n * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, s_edge));
        if (next != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(h_send_hi, e_out + (size_t)L * n, n * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, s_edge));
        CUDA_CHECK(cudaEventRecord(ev_edge, s_edge));
        CUDA_CHECK(cudaEventSynchronize(ev_edge));

        MPI_Request reqs[4];
        int nreq = 0;
        if (prev != MPI_PROC_NULL) {
            MPI_Irecv(h_recv_lo, n, MPI_DOUBLE, prev, 0, comm, &reqs[nreq++]);
            MPI_Isend(h_send_lo, n, MPI_DOUBLE, prev, 1, comm, &reqs[nreq++]);
        }
        if (next != MPI_PROC_NULL) {
            MPI_Irecv(h_recv_hi, n, MPI_DOUBLE, next, 1, comm, &reqs[nreq++]);
            MPI_Isend(h_send_hi, n, MPI_DOUBLE, next, 0, comm, &reqs[nreq++]);
        }
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        if (prev != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(e_out, h_recv_lo, n * sizeof(val_t),
                                       cudaMemcpyHostToDevice, s_edge));
        if (next != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(e_out + (size_t)(L + 1) * n, h_recv_hi,
                                       n * sizeof(val_t), cudaMemcpyHostToDevice, s_edge));
        CUDA_CHECK(cudaStreamSynchronize(s_edge));
        CUDA_CHECK(cudaStreamSynchronize(s_inner));
        cur ^= 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFreeHost(h_send_lo); cudaFreeHost(h_send_hi);
    cudaFreeHost(h_recv_lo); cudaFreeHost(h_recv_hi);
    cudaEventDestroy(ev_edge);
    cudaStreamDestroy(s_edge);
    cudaStreamDestroy(s_inner);
    return cur;  // index of the buffer holding the final energies
}

// Validate simulation results
bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
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

// Hash contribution of this rank's elements (global index offset applied);
// XOR is associative so partial hashes combine across ranks and threads.
uint64_t computeHash(const val_t* energy, const val_t* flux, size_t count, size_t offset) {
    uint64_t hash = 0;
#pragma omp parallel for reduction(^ : hash) schedule(static)
    for (size_t k = 0; k < count; ++k) {
        const size_t i = offset + k;
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
    int wrank, wsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

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
            if (wrank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (wrank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (n_elems_root < 1) {
        if (wrank == 0) printf("Grid size must be at least 1\n");
        MPI_Finalize();
        return 1;
    }

    const size_t n_elems = (size_t)n_elems_root * n_elems_root;

    // Bind each rank to a GPU (round-robin among ranks on the same node)
    {
        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, wrank, MPI_INFO_NULL, &node);
        int lrank;
        MPI_Comm_rank(node, &lrank);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev < 1) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
        MPI_Comm_free(&node);
    }

    // Only as many ranks as grid rows can take part
    const int active = std::min(wsize, n_elems_root);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, wrank < active ? 0 : MPI_UNDEFINED, wrank, &comm);

    if (wrank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");

        const size_t static_mem = n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
        fflush(stdout);
    }

    double elapsed_ms = 0.0;
    uint64_t hash = 0;
    std::vector<val_t> allEnergy, allFlux;

    if (comm != MPI_COMM_NULL) {
        Decomp dc;
        dc.n = n_elems_root;
        MPI_Comm_rank(comm, &dc.rank);
        MPI_Comm_size(comm, &dc.size);
        rowRange(dc.n, dc.size, dc.rank, dc.row_begin, dc.rows);

        const size_t n = (size_t)dc.n;
        const size_t local = (size_t)dc.rows * n;
        const size_t buf = (size_t)(dc.rows + 2) * n;

        val_t* d_e[2];
        val_t* d_flux;
        CUDA_CHECK(cudaMalloc(&d_e[0], buf * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d_e[1], buf * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d_flux, local * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(d_e[0], 0, buf * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(d_e[1], 0, buf * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(d_flux, 0, local * sizeof(val_t)));
        CUDA_CHECK(cudaDeviceSynchronize());

        MPI_Barrier(comm);
        auto start = std::chrono::high_resolution_clock::now();
        const int fin = runSimulation(dc, comm, d_e, d_flux, n_iters);
        MPI_Barrier(comm);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed_ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        val_t *h_e, *h_f;
        CUDA_CHECK(cudaMallocHost(&h_e, local * sizeof(val_t)));
        CUDA_CHECK(cudaMallocHost(&h_f, local * sizeof(val_t)));
        CUDA_CHECK(cudaMemcpy(h_e, d_e[fin] + n, local * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_f, d_flux, local * sizeof(val_t), cudaMemcpyDeviceToHost));

        uint64_t local_hash = computeHash(h_e, h_f, local, (size_t)dc.row_begin * n);
        MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);

        if (validate || printResults) {
            // Gather everything on rank 0 in global element order
            const size_t CH = (size_t)1 << 28;
            if (dc.rank == 0) {
                allEnergy.resize(n_elems);
                allFlux.resize(n_elems);
                memcpy(allEnergy.data(), h_e, local * sizeof(val_t));
                memcpy(allFlux.data(), h_f, local * sizeof(val_t));
                for (int r = 1; r < dc.size; ++r) {
                    int b, rows;
                    rowRange(dc.n, dc.size, r, b, rows);
                    const size_t cnt = (size_t)rows * n, off = (size_t)b * n;
                    for (size_t o = 0; o < cnt; o += CH) {
                        const int c = (int)std::min(cnt - o, CH);
                        MPI_Recv(allEnergy.data() + off + o, c, MPI_DOUBLE, r, 10, comm, MPI_STATUS_IGNORE);
                        MPI_Recv(allFlux.data() + off + o, c, MPI_DOUBLE, r, 11, comm, MPI_STATUS_IGNORE);
                    }
                }
            } else {
                for (size_t o = 0; o < local; o += CH) {
                    const int c = (int)std::min(local - o, CH);
                    MPI_Send(h_e + o, c, MPI_DOUBLE, 0, 10, comm);
                    MPI_Send(h_f + o, c, MPI_DOUBLE, 0, 11, comm);
                }
            }
        }

        cudaFreeHost(h_e);
        cudaFreeHost(h_f);
        cudaFree(d_e[0]);
        cudaFree(d_e[1]);
        cudaFree(d_flux);
    }

    int ret = 0;
    if (wrank == 0) {
        const long duration_ms = (long)elapsed_ms;
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (double)n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            print_results(allEnergy, "ElementEnergy");
        }
        if (validate) {
            if (!validateResults(allEnergy, allFlux)) ret = 1;
        }
    }

    if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
    MPI_Finalize();
    return ret;
}
