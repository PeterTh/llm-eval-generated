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

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static inline void cuda_check(cudaError_t e, const char* expr, const char* file, int line) {
    if (e == cudaSuccess) return;
    fprintf(stderr, "CUDA error %s:%d: %s failed: %s\n", file, line, expr, cudaGetErrorString(e));
    MPI_Abort(MPI_COMM_WORLD, 2);
}
#define CUDA_CHECK(x) cuda_check((x), #x, __FILE__, __LINE__)

struct Decomp {
    int N;
    int rank;
    int size;
    int start_row;
    int local_rows;
};

static inline Decomp make_decomp(int N, int rank, int size) {
    const int base = N / size;
    const int rem = N % size;
    const int local_rows = base + (rank < rem ? 1 : 0);
    const int start_row = rank * base + (rank < rem ? rank : rem);
    return Decomp{N, rank, size, start_row, local_rows};
}

__device__ __forceinline__ double external_flow_for_cell(int global_r, int c, int N) {
    // Match buildSquare2D(): inflow at (0,0) and (N-1,N-1), outflow at (0,N-1) and (N-1,0).
    if ((global_r == 0 && c == 0) || (global_r == N - 1 && c == N - 1)) return 0.5;
    if ((global_r == 0 && c == N - 1) || (global_r == N - 1 && c == 0)) return -0.5;
    return 0.0;
}

__global__ void step_rows_kernel(const double* __restrict__ energy, const double* __restrict__ flux,
                                double* __restrict__ energy_out, double* __restrict__ flux_out,
                                int N, int local_rows, int start_row,
                                int row_begin, int row_end) {
    const int c = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int r = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y) + row_begin;
    if (c >= N || r >= row_end) return;

    const int global_r = start_row + r;
    const int owned_idx = r * N + c;
    const int e_idx = (r + 1) * N + c;  // +1 for top halo row

    const double transfer = 0.8;
    const double k = transfer * 0.25;

    const double center = energy[e_idx];
    double total = external_flow_for_cell(global_r, c, N);

    // Up/down use halo rows at subdomain boundaries; skip global boundaries.
    if (global_r > 0) total += (energy[e_idx - N] - center) * k;
    if (global_r + 1 < N) total += (energy[e_idx + N] - center) * k;

    // Left/right are always local within the row.
    if (c > 0) total += (energy[e_idx - 1] - center) * k;
    if (c + 1 < N) total += (energy[e_idx + 1] - center) * k;

    energy_out[e_idx] = center + total;
    flux_out[owned_idx] = flux[owned_idx] + fabs(total);
}

static inline void exchange_halos(const Decomp& d, cudaStream_t stream_comm,
                                 double* d_energy, int up, int down,
                                 double* h_send_top, double* h_send_bottom,
                                 double* h_recv_top, double* h_recv_bottom) {
    const size_t row_bytes = static_cast<size_t>(d.N) * sizeof(double);

    // D2H boundary rows (owned rows 0 and local_rows-1).
    CUDA_CHECK(cudaMemcpyAsync(h_send_top, d_energy + d.N, row_bytes, cudaMemcpyDeviceToHost, stream_comm));
    CUDA_CHECK(cudaMemcpyAsync(h_send_bottom, d_energy + static_cast<size_t>(d.local_rows) * d.N, row_bytes,
                               cudaMemcpyDeviceToHost, stream_comm));
    CUDA_CHECK(cudaStreamSynchronize(stream_comm));

    MPI_Status st;

    // Exchange with up neighbor (receive into top halo row).
    MPI_Sendrecv(h_send_top, d.N, MPI_DOUBLE, up, 0,
                 h_recv_top, d.N, MPI_DOUBLE, up, 1,
                 MPI_COMM_WORLD, &st);

    // Exchange with down neighbor (receive into bottom halo row).
    MPI_Sendrecv(h_send_bottom, d.N, MPI_DOUBLE, down, 1,
                 h_recv_bottom, d.N, MPI_DOUBLE, down, 0,
                 MPI_COMM_WORLD, &st);

    if (up != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_energy, h_recv_top, row_bytes, cudaMemcpyHostToDevice, stream_comm));
    }
    if (down != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_energy + static_cast<size_t>(d.local_rows + 1) * d.N, h_recv_bottom, row_bytes,
                                   cudaMemcpyHostToDevice, stream_comm));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_comm));
}

static inline void runSimulationHybrid(const Decomp& d, int n_iters,
                                       double*& d_energy, double*& d_flux,
                                       double*& d_energy_swap, double*& d_flux_swap,
                                       double* h_send_top, double* h_send_bottom,
                                       double* h_recv_top, double* h_recv_bottom) {
    const int up = (d.rank > 0) ? (d.rank - 1) : MPI_PROC_NULL;
    const int down = (d.rank + 1 < d.size) ? (d.rank + 1) : MPI_PROC_NULL;

    cudaStream_t stream_compute{};
    cudaStream_t stream_comm{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_comm, cudaStreamNonBlocking));

    dim3 block(32, 8, 1);
    dim3 grid((d.N + block.x - 1) / block.x,
              (d.local_rows + block.y - 1) / block.y,
              1);

    for (int iter = 0; iter < n_iters; ++iter) {
        // Start halo exchange (D2H copies can overlap with interior compute).
        // While communication is in flight, compute interior rows that don't depend on halos.
        int interior_begin = 1;
        int interior_end = d.local_rows - 1;

        // Launch interior kernel first (stream_compute). It reads d_energy and writes d_energy_swap.
        if (interior_end > interior_begin) {
            step_rows_kernel<<<grid, block, 0, stream_compute>>>(
                d_energy, d_flux, d_energy_swap, d_flux_swap,
                d.N, d.local_rows, d.start_row,
                interior_begin, interior_end);
            CUDA_CHECK(cudaGetLastError());
        }

        // Exchange halos on stream_comm.
        exchange_halos(d, stream_comm, d_energy, up, down,
                       h_send_top, h_send_bottom, h_recv_top, h_recv_bottom);

        // Boundary rows (need halos when neighbors exist).
        if (d.local_rows > 0) {
            step_rows_kernel<<<grid, block, 0, stream_compute>>>(
                d_energy, d_flux, d_energy_swap, d_flux_swap,
                d.N, d.local_rows, d.start_row,
                0, 1);
            CUDA_CHECK(cudaGetLastError());

            if (d.local_rows > 1) {
                step_rows_kernel<<<grid, block, 0, stream_compute>>>(
                    d_energy, d_flux, d_energy_swap, d_flux_swap,
                    d.N, d.local_rows, d.start_row,
                    d.local_rows - 1, d.local_rows);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_compute));

        std::swap(d_energy, d_energy_swap);
        std::swap(d_flux, d_flux_swap);
    }

    CUDA_CHECK(cudaStreamDestroy(stream_comm));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
}


// Validate simulation results
static bool validateResults(const std::vector<ElementDynamic>& elements_dynamic) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : elements_dynamic) {
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

// Compute a simple hash of the results for verification
static uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static void printUsage(const char* progName) {
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_elems_root = 512;
    int n_iters = 10;
    int validate = 0;
    int printResults = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                n_elems_root = -1;
                n_iters = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                n_elems_root = -1;
                n_iters = 0;
                break;
            }
        }
    }

    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n_elems_root <= 0) {
        MPI_Finalize();
        return (rank == 0) ? 0 : 0;
    }

    const int N = n_elems_root;
    const int n_elems = N * N;

    // Bind each rank to a GPU (simple round-robin).
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (rank == 0) fprintf(stderr, "ERROR: No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    const int device = rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));

    const Decomp d = make_decomp(N, rank, size);

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("==============================================================\n");
        printf("Grid size: %d x %d = %d elements\n", N, N, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
        printf("CUDA devices visible: %d\n", device_count);
        printf("\n");
    }

    // Device allocations: energy has 2 halo rows.
    const size_t energy_elems = static_cast<size_t>(d.N) * static_cast<size_t>(d.local_rows + 2);
    const size_t flux_elems = static_cast<size_t>(d.N) * static_cast<size_t>(d.local_rows);

    double* d_energy = nullptr;
    double* d_energy_swap = nullptr;
    double* d_flux = nullptr;
    double* d_flux_swap = nullptr;

    CUDA_CHECK(cudaMalloc(&d_energy, energy_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy_swap, energy_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flux, flux_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flux_swap, flux_elems * sizeof(double)));

    CUDA_CHECK(cudaMemset(d_energy, 0, energy_elems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_energy_swap, 0, energy_elems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_flux, 0, flux_elems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_flux_swap, 0, flux_elems * sizeof(double)));

    // Host pinned buffers for halo exchange.
    double* h_send_top = nullptr;
    double* h_send_bottom = nullptr;
    double* h_recv_top = nullptr;
    double* h_recv_bottom = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_send_top, static_cast<size_t>(d.N) * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_send_bottom, static_cast<size_t>(d.N) * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recv_top, static_cast<size_t>(d.N) * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recv_bottom, static_cast<size_t>(d.N) * sizeof(double), cudaHostAllocDefault));

    // Memory usage reporting (aggregate across ranks).
    const double local_mem =
        (energy_elems * 2 * sizeof(double) + flux_elems * 2 * sizeof(double)) / (1024.0 * 1024.0);
    double global_mem = 0.0;
    MPI_Reduce(&local_mem, &global_mem, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Memory usage (approx): %.2f MB aggregate across ranks (%.2f MB per rank avg)\n\n",
               global_mem, global_mem / size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    runSimulationHybrid(d, n_iters,
                        d_energy, d_flux,
                        d_energy_swap, d_flux_swap,
                        h_send_top, h_send_bottom,
                        h_recv_top, h_recv_bottom);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double local_s = t1 - t0;
    double max_s = 0.0;
    MPI_Reduce(&local_s, &max_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy local owned results back to host.
    std::vector<double> h_energy_local(static_cast<size_t>(d.local_rows) * d.N);
    std::vector<double> h_flux_local(static_cast<size_t>(d.local_rows) * d.N);

    if (d.local_rows > 0) {
        CUDA_CHECK(cudaMemcpy(h_energy_local.data(), d_energy + d.N,
                              static_cast<size_t>(d.local_rows) * d.N * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_flux_local.data(), d_flux,
                              static_cast<size_t>(d.local_rows) * d.N * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // Gather to rank 0 for output/hash/validation.
    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<double> h_energy_global;
    std::vector<double> h_flux_global;

    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            Decomp dr = make_decomp(N, r, size);
            counts[r] = dr.local_rows * N;
            displs[r] = dr.start_row * N;
        }
        h_energy_global.resize(static_cast<size_t>(N) * N);
        h_flux_global.resize(static_cast<size_t>(N) * N);
    }

    MPI_Gatherv(h_energy_local.data(), d.local_rows * N, MPI_DOUBLE,
                rank == 0 ? h_energy_global.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Gatherv(h_flux_local.data(), d.local_rows * N, MPI_DOUBLE,
                rank == 0 ? h_flux_global.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double duration_ms = max_s * 1000.0;
        printf("Computation time (max rank): %.3f ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter_ms = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / (max_s) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter_ms);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        std::vector<ElementDynamic> elements_dynamic(static_cast<size_t>(N) * N);
#pragma omp parallel for
        for (int i = 0; i < n_elems; ++i) {
            elements_dynamic[static_cast<size_t>(i)].current_energy = h_energy_global[static_cast<size_t>(i)];
            elements_dynamic[static_cast<size_t>(i)].total_flux = h_flux_global[static_cast<size_t>(i)];
        }

        const uint64_t hash = computeHash(elements_dynamic);
        printf("  Result hash: %016lX\n\n", hash);

        if (printResults) {
            print_results(h_energy_global, "ElementEnergy");
        }

        if (validate) {
            bool ok = validateResults(elements_dynamic);
            if (!ok) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    CUDA_CHECK(cudaFreeHost(h_recv_bottom));
    CUDA_CHECK(cudaFreeHost(h_recv_top));
    CUDA_CHECK(cudaFreeHost(h_send_bottom));
    CUDA_CHECK(cudaFreeHost(h_send_top));

    CUDA_CHECK(cudaFree(d_flux_swap));
    CUDA_CHECK(cudaFree(d_flux));
    CUDA_CHECK(cudaFree(d_energy_swap));
    CUDA_CHECK(cudaFree(d_energy));

    MPI_Finalize();
    return 0;
}
