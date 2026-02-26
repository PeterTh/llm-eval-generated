#include <algorithm>
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

using idx_t = uint64_t;
using val_t = double;

namespace {

constexpr val_t TRANSFER_COEFF = 0.8;
constexpr val_t CORNER_INFLOW = 0.5;
constexpr val_t CORNER_OUTFLOW = -0.5;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t _e = (call);                                                         \
        if (_e != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            MPI_Abort(MPI_COMM_WORLD, 2);                                                \
        }                                                                                \
    } while (0)

static inline void decompose_rows(const int n_rows, const int size, const int rank, int& x0, int& nx) {
    const int base = n_rows / size;
    const int rem = n_rows % size;
    if (rank < rem) {
        nx = base + 1;
        x0 = rank * nx;
    } else {
        nx = base;
        x0 = rem * (base + 1) + (rank - rem) * base;
    }
}

__global__ void update_kernel(const val_t* __restrict__ energy_in,
                             const val_t* __restrict__ flux_in,
                             val_t* __restrict__ energy_out,
                             val_t* __restrict__ flux_out,
                             const val_t* __restrict__ external_flow,
                             const val_t* __restrict__ halo_top,
                             const val_t* __restrict__ halo_bottom,
                             int n,
                             int local_nx,
                             int global_x0) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t local_elems = static_cast<size_t>(local_nx) * static_cast<size_t>(n);
    if (tid >= local_elems) return;

    const int y = static_cast<int>(tid % static_cast<size_t>(n));
    const int r = static_cast<int>(tid / static_cast<size_t>(n));
    const int gx = global_x0 + r;

    const val_t this_energy = energy_in[tid];
    val_t total_flux = external_flow[tid];

    // Match neighbor accumulation order from the original connectivity builder: (x+1), (x-1), (y+1), (y-1).
    // (x+1)
    if (gx + 1 < n) {
        const val_t other = (r + 1 < local_nx) ? energy_in[tid + n] : halo_bottom[y];
        total_flux += (other - this_energy) * TRANSFER_COEFF * 0.25;
    }
    // (x-1)
    if (gx > 0) {
        const val_t other = (r > 0) ? energy_in[tid - n] : halo_top[y];
        total_flux += (other - this_energy) * TRANSFER_COEFF * 0.25;
    }
    // (y+1)
    if (y + 1 < n) {
        const val_t other = energy_in[tid + 1];
        total_flux += (other - this_energy) * TRANSFER_COEFF * 0.25;
    }
    // (y-1)
    if (y > 0) {
        const val_t other = energy_in[tid - 1];
        total_flux += (other - this_energy) * TRANSFER_COEFF * 0.25;
    }

    energy_out[tid] = this_energy + total_flux;
    flux_out[tid] = flux_in[tid] + fabs(total_flux);
}

static inline uint64_t u64_from_f64(val_t v) {
    uint64_t u;
    static_assert(sizeof(u) == sizeof(v), "size mismatch");
    memcpy(&u, &v, sizeof(u));
    return u;
}

static uint64_t computeHashLocal(const std::vector<val_t>& energy,
                                const std::vector<val_t>& flux,
                                const idx_t global_start_idx) {
    uint64_t hash = 0;
#pragma omp parallel for reduction(^ : hash)
    for (size_t i = 0; i < energy.size(); ++i) {
        const idx_t gi = global_start_idx + static_cast<idx_t>(i);
        const uint64_t e_bits = u64_from_f64(energy[i]);
        const uint64_t f_bits = u64_from_f64(flux[i]);
        uint64_t h = 0;
        h ^= (e_bits + static_cast<uint64_t>(gi)) * 0x9e3779b97f4a7c15ULL;
        h ^= (f_bits + static_cast<uint64_t>(gi)) * 0xbf58476d1ce4e5b9ULL;
        hash ^= h;
    }
    return hash;
}

static bool validateDistributed(const std::vector<val_t>& energy,
                               const std::vector<val_t>& flux,
                               MPI_Comm comm,
                               int rank) {
    val_t energy_sum_local = 0.0;
    val_t flux_sum_local = 0.0;
    val_t energy_max_local = std::numeric_limits<val_t>::lowest();
    val_t energy_min_local = std::numeric_limits<val_t>::max();

#pragma omp parallel for reduction(+ : energy_sum_local, flux_sum_local) reduction(max : energy_max_local) reduction(min : energy_min_local)
    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum_local += energy[i];
        flux_sum_local += flux[i];
        energy_max_local = std::max(energy_max_local, energy[i]);
        energy_min_local = std::min(energy_min_local, energy[i]);
    }

    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = 0.0, energy_min = 0.0;
    MPI_Reduce(&energy_sum_local, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&flux_sum_local, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&energy_max_local, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&energy_min_local, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);

    int valid = 1;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        constexpr val_t energy_epsilon = 1e-8;
        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        }
        if (std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = 0;
        }
        if (valid) {
            printf("  Validation: PASSED\n");
        }
    }

    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    return valid != 0;
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

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_elems_root = 512;
    int n_iters = 10;
    int validate = 0;
    int printResults = 0;

    int exit_now = 0;
    int exit_code = 0;

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
                exit_now = 1;
                exit_code = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exit_now = 1;
                exit_code = 1;
                break;
            }
        }
    }

    MPI_Bcast(&exit_now, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exit_now) {
        MPI_Finalize();
        return exit_code;
    }

    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (size > n_elems_root) {
        if (rank == 0) {
            fprintf(stderr, "ERROR: MPI ranks (%d) must be <= grid size (%d) for row decomposition.\n", size, n_elems_root);
        }
        MPI_Finalize();
        return 1;
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    const int device = (device_count > 0) ? (rank % device_count) : 0;
    CUDA_CHECK(cudaSetDevice(device));

    const int n = n_elems_root;
    const int n_elems = n * n;

    int global_x0 = 0, local_nx = 0;
    decompose_rows(n, size, rank, global_x0, local_nx);

    const size_t local_elems = static_cast<size_t>(local_nx) * static_cast<size_t>(n);
    const idx_t global_start_idx = static_cast<idx_t>(global_x0) * static_cast<idx_t>(n);

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads (rank 0): %d\n", omp_get_max_threads());
        printf("CUDA devices visible: %d\n", device_count);
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    // External flow for this rank's rows (corners only, matching original semantics)
    std::vector<val_t> h_external(local_elems, 0.0);
#pragma omp parallel for
    for (int r = 0; r < local_nx; ++r) {
        const int gx = global_x0 + r;
        for (int y = 0; y < n; ++y) {
            const size_t i = static_cast<size_t>(r) * static_cast<size_t>(n) + static_cast<size_t>(y);
            if (gx == 0 && y == 0) h_external[i] = CORNER_INFLOW;
            if (gx == 0 && y == n - 1) h_external[i] = CORNER_OUTFLOW;
            if (gx == n - 1 && y == 0) h_external[i] = CORNER_OUTFLOW;
            if (gx == n - 1 && y == n - 1) h_external[i] = CORNER_INFLOW;
        }
    }

    val_t *d_energy_a = nullptr, *d_energy_b = nullptr;
    val_t *d_flux_a = nullptr, *d_flux_b = nullptr;
    val_t *d_external = nullptr;
    val_t *d_halo_top = nullptr, *d_halo_bottom = nullptr;

    CUDA_CHECK(cudaMalloc(&d_energy_a, local_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_energy_b, local_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_flux_a, local_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_flux_b, local_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_external, local_elems * sizeof(val_t)));

    CUDA_CHECK(cudaMalloc(&d_halo_top, static_cast<size_t>(n) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_halo_bottom, static_cast<size_t>(n) * sizeof(val_t)));

    CUDA_CHECK(cudaMemset(d_energy_a, 0, local_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(d_energy_b, 0, local_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(d_flux_a, 0, local_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(d_flux_b, 0, local_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(d_external, h_external.data(), local_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_halo_top, 0, static_cast<size_t>(n) * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(d_halo_bottom, 0, static_cast<size_t>(n) * sizeof(val_t)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Host pinned buffers for halo exchange
    val_t *h_top_send = nullptr, *h_top_recv = nullptr;
    val_t *h_bot_send = nullptr, *h_bot_recv = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_top_send, static_cast<size_t>(n) * sizeof(val_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_top_recv, static_cast<size_t>(n) * sizeof(val_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_bot_send, static_cast<size_t>(n) * sizeof(val_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_bot_recv, static_cast<size_t>(n) * sizeof(val_t), cudaHostAllocDefault));

    // Memory usage (device-side dominant); report global totals.
    const unsigned long long local_dynamic_mem =
        static_cast<unsigned long long>((6 * local_elems + 2ull * static_cast<size_t>(n)) * sizeof(val_t));
    unsigned long long global_dynamic_mem = 0;
    MPI_Reduce(&local_dynamic_mem, &global_dynamic_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Memory usage: %.2f MB (dynamic, device): %.2f MB\n\n",
               global_dynamic_mem / (1024.0 * 1024.0),
               global_dynamic_mem / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    val_t* d_energy_in = d_energy_a;
    val_t* d_energy_out = d_energy_b;
    val_t* d_flux_in = d_flux_a;
    val_t* d_flux_out = d_flux_b;

    const int threads = 256;
    const int blocks = static_cast<int>((local_elems + threads - 1) / threads);

    for (int iter = 0; iter < n_iters; ++iter) {
        if (size > 1) {
            if (rank > 0) {
                CUDA_CHECK(cudaMemcpyAsync(h_top_send, d_energy_in, static_cast<size_t>(n) * sizeof(val_t), cudaMemcpyDeviceToHost, stream));
            }
            if (rank < size - 1) {
                const val_t* bottom_row = d_energy_in + static_cast<size_t>(local_nx - 1) * static_cast<size_t>(n);
                CUDA_CHECK(cudaMemcpyAsync(h_bot_send, bottom_row, static_cast<size_t>(n) * sizeof(val_t), cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            if (rank > 0) {
                MPI_Sendrecv(h_top_send, n, MPI_DOUBLE, rank - 1, 0,
                             h_top_recv, n, MPI_DOUBLE, rank - 1, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (rank < size - 1) {
                MPI_Sendrecv(h_bot_send, n, MPI_DOUBLE, rank + 1, 1,
                             h_bot_recv, n, MPI_DOUBLE, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            if (rank > 0) {
                CUDA_CHECK(cudaMemcpyAsync(d_halo_top, h_top_recv, static_cast<size_t>(n) * sizeof(val_t), cudaMemcpyHostToDevice, stream));
            }
            if (rank < size - 1) {
                CUDA_CHECK(cudaMemcpyAsync(d_halo_bottom, h_bot_recv, static_cast<size_t>(n) * sizeof(val_t), cudaMemcpyHostToDevice, stream));
            }
        }

        update_kernel<<<blocks, threads, 0, stream>>>(d_energy_in, d_flux_in, d_energy_out, d_flux_out,
                                                     d_external, d_halo_top, d_halo_bottom,
                                                     n, local_nx, global_x0);
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_energy_in, d_energy_out);
        std::swap(d_flux_in, d_flux_out);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double local_ms = (t1 - t0) * 1000.0;
    double duration_ms = 0.0;
    MPI_Reduce(&local_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<val_t> h_energy(local_elems);
    std::vector<val_t> h_flux(local_elems);
    CUDA_CHECK(cudaMemcpy(h_energy.data(), d_energy_in, local_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flux.data(), d_flux_in, local_elems * sizeof(val_t), cudaMemcpyDeviceToHost));

    const unsigned long long local_hash = static_cast<unsigned long long>(computeHashLocal(h_energy, h_flux, global_start_idx));
    unsigned long long global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(global_hash));
    }

    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> all_energy;
        if (rank == 0) {
            counts.resize(size);
            displs.resize(size);
            int disp = 0;
            for (int r = 0; r < size; ++r) {
                int rx0 = 0, rnx = 0;
                decompose_rows(n, size, r, rx0, rnx);
                counts[r] = rnx * n;
                displs[r] = disp;
                disp += counts[r];
            }
            all_energy.resize(static_cast<size_t>(n_elems));
        }
        MPI_Gatherv(h_energy.data(), static_cast<int>(local_elems), MPI_DOUBLE,
                    rank == 0 ? all_energy.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(all_energy, "ElementEnergy");
        }
    }

    if (validate) {
        const bool ok = validateDistributed(h_energy, h_flux, MPI_COMM_WORLD, rank);
        if (!ok) {
            CUDA_CHECK(cudaFree(d_energy_a));
            CUDA_CHECK(cudaFree(d_energy_b));
            CUDA_CHECK(cudaFree(d_flux_a));
            CUDA_CHECK(cudaFree(d_flux_b));
            CUDA_CHECK(cudaFree(d_external));
            CUDA_CHECK(cudaFree(d_halo_top));
            CUDA_CHECK(cudaFree(d_halo_bottom));
            CUDA_CHECK(cudaFreeHost(h_top_send));
            CUDA_CHECK(cudaFreeHost(h_top_recv));
            CUDA_CHECK(cudaFreeHost(h_bot_send));
            CUDA_CHECK(cudaFreeHost(h_bot_recv));
            CUDA_CHECK(cudaStreamDestroy(stream));
            MPI_Finalize();
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(d_energy_a));
    CUDA_CHECK(cudaFree(d_energy_b));
    CUDA_CHECK(cudaFree(d_flux_a));
    CUDA_CHECK(cudaFree(d_flux_b));
    CUDA_CHECK(cudaFree(d_external));
    CUDA_CHECK(cudaFree(d_halo_top));
    CUDA_CHECK(cudaFree(d_halo_bottom));
    CUDA_CHECK(cudaFreeHost(h_top_send));
    CUDA_CHECK(cudaFreeHost(h_top_recv));
    CUDA_CHECK(cudaFreeHost(h_bot_send));
    CUDA_CHECK(cudaFreeHost(h_bot_recv));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Finalize();
    return 0;
}
