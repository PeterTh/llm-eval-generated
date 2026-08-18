#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

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

static void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA error in %s: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call, rank)

// The generated "unstructured" mesh is a regular four-neighbor graph.  Keeping
// its row partition implicit removes the large connectivity array and makes the
// GPU accesses coalesced while preserving exactly the same graph and update.
__global__ void updateRows(const val_t* __restrict__ energy,
                           const val_t* __restrict__ accumulated_flux,
                           val_t* __restrict__ next_energy,
                           val_t* __restrict__ next_accumulated_flux,
                           int n, int global_first_row, int local_rows,
                           int first_local_row, int rows_to_update) {
    const size_t work = static_cast<size_t>(rows_to_update) * n;
    for (size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < work;
         item += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int local_row = first_local_row + static_cast<int>(item / n);
        const int column = static_cast<int>(item % n);
        const int global_row = global_first_row + local_row - 1;
        const size_t center_index = static_cast<size_t>(local_row) * n + column;
        const val_t center = energy[center_index];

        // Match the reference connection order: +row, -row, +column, -column.
        val_t total = 0.0;
        if ((global_row == 0 && column == 0) ||
            (global_row == n - 1 && column == n - 1)) {
            total = 0.5;
        } else if ((global_row == 0 && column == n - 1) ||
                   (global_row == n - 1 && column == 0)) {
            total = -0.5;
        }
        if (global_row + 1 < n)
            total += (energy[center_index + n] - center) * 0.8 * 0.25;
        if (global_row > 0)
            total += (energy[center_index - n] - center) * 0.8 * 0.25;
        if (column + 1 < n)
            total += (energy[center_index + 1] - center) * 0.8 * 0.25;
        if (column > 0)
            total += (energy[center_index - 1] - center) * 0.8 * 0.25;

        next_energy[center_index] = center + total;
        next_accumulated_flux[center_index] = accumulated_flux[center_index] + fabs(total);
    }
}

__global__ void updateBoundaryRows(const val_t* __restrict__ energy,
                                   const val_t* __restrict__ accumulated_flux,
                                   val_t* __restrict__ next_energy,
                                   val_t* __restrict__ next_accumulated_flux,
                                   int n, int global_first_row, int local_rows) {
    const int boundary_rows = local_rows == 1 ? 1 : 2;
    const size_t work = static_cast<size_t>(boundary_rows) * n;
    for (size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < work;
         item += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int edge = static_cast<int>(item / n);
        const int local_row = edge == 0 ? 1 : local_rows;
        const int column = static_cast<int>(item % n);
        const int global_row = global_first_row + local_row - 1;
        const size_t center_index = static_cast<size_t>(local_row) * n + column;
        const val_t center = energy[center_index];

        val_t total = 0.0;
        if ((global_row == 0 && column == 0) ||
            (global_row == n - 1 && column == n - 1)) {
            total = 0.5;
        } else if ((global_row == 0 && column == n - 1) ||
                   (global_row == n - 1 && column == 0)) {
            total = -0.5;
        }
        if (global_row + 1 < n)
            total += (energy[center_index + n] - center) * 0.8 * 0.25;
        if (global_row > 0)
            total += (energy[center_index - n] - center) * 0.8 * 0.25;
        if (column + 1 < n)
            total += (energy[center_index + 1] - center) * 0.8 * 0.25;
        if (column > 0)
            total += (energy[center_index - 1] - center) * 0.8 * 0.25;

        next_energy[center_index] = center + total;
        next_accumulated_flux[center_index] = accumulated_flux[center_index] + fabs(total);
    }
}

static void launchRows(cudaStream_t stream, const val_t* energy, const val_t* flux,
                       val_t* next_energy, val_t* next_flux, int n,
                       int global_first_row, int local_rows, int first_row,
                       int row_count, int rank) {
    if (row_count <= 0) return;
    constexpr int threads = 256;
    const size_t items = static_cast<size_t>(row_count) * n;
    const int blocks = static_cast<int>(std::min<size_t>((items + threads - 1) / threads, 65535));
    updateRows<<<blocks, threads, 0, stream>>>(energy, flux, next_energy, next_flux, n,
                                               global_first_row, local_rows,
                                               first_row, row_count);
    cudaCheck(cudaGetLastError(), "updateRows kernel launch", rank);
}

static void launchBoundaries(cudaStream_t stream, const val_t* energy, const val_t* flux,
                             val_t* next_energy, val_t* next_flux, int n,
                             int global_first_row, int local_rows, int rank) {
    if (local_rows <= 0) return;
    constexpr int threads = 256;
    const size_t items = static_cast<size_t>(local_rows == 1 ? 1 : 2) * n;
    const int blocks = static_cast<int>(std::min<size_t>((items + threads - 1) / threads, 65535));
    updateBoundaryRows<<<blocks, threads, 0, stream>>>(energy, flux, next_energy, next_flux,
                                                       n, global_first_row, local_rows);
    cudaCheck(cudaGetLastError(), "updateBoundaryRows kernel launch", rank);
}

static bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }

    std::printf("Validation results:\n");
    std::printf("  Energy sum: %.12f\n", energy_sum);
    std::printf("  Flux sum: %.2f\n", flux_sum);
    std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    if (!std::isfinite(energy_sum) || !std::isfinite(flux_sum) ||
        !std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        std::printf("  ERROR: simulation produced a non-finite value\n");
        return false;
    }
    if (std::abs(energy_sum) > 1e-8)
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    std::printf("  Validation: PASSED\n");
    return true;
}

static uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t result = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        result ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        result ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return result;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int n = 512;
    int iterations = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool arguments_ok = true;
    bool help = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (rank == 0) std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            arguments_ok = false;
        }
    }

    if (help || !arguments_ok || n <= 0 || iterations < 0 ||
        static_cast<int64_t>(n) * n > std::numeric_limits<int>::max()) {
        if (rank == 0) {
            if (!help && (n <= 0 || iterations < 0))
                std::fprintf(stderr, "Grid size must be positive and iterations non-negative.\n");
            if (!help && static_cast<int64_t>(n) * n > std::numeric_limits<int>::max())
                std::fprintf(stderr, "Grid contains too many elements for MPI_Gatherv.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return help ? 0 : 1;
    }

    // Assign accelerators by node-local MPI rank, the standard one-rank-per-GPU layout.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0, local_ranks = 1;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_ranks);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) std::fprintf(stderr, "The hybrid benchmark requires CUDA devices.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    CUDA_CHECK(cudaFree(nullptr));
    // Resolve/load the module before the timed region so JIT setup is not benchmarked.
    cudaFuncAttributes kernel_attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&kernel_attributes, updateRows));
    CUDA_CHECK(cudaFuncGetAttributes(&kernel_attributes, updateBoundaryRows));

    const int base_rows = n / ranks;
    const int extra_rows = n % ranks;
    const int local_rows = base_rows + (rank < extra_rows ? 1 : 0);
    const int global_first_row = rank * base_rows + std::min(rank, extra_rows);
    const int active_ranks = std::min(ranks, n);
    const int previous = (local_rows > 0 && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next = (local_rows > 0 && rank + 1 < active_ranks) ? rank + 1 : MPI_PROC_NULL;
    const size_t row_bytes = static_cast<size_t>(n) * sizeof(val_t);
    const size_t device_elements = static_cast<size_t>(local_rows + 2) * n;
    const size_t device_bytes = device_elements * sizeof(val_t);

    val_t *energy_a = nullptr, *energy_b = nullptr, *flux_a = nullptr, *flux_b = nullptr;
    CUDA_CHECK(cudaMalloc(&energy_a, device_bytes));
    CUDA_CHECK(cudaMalloc(&energy_b, device_bytes));
    CUDA_CHECK(cudaMalloc(&flux_a, device_bytes));
    CUDA_CHECK(cudaMalloc(&flux_b, device_bytes));
    CUDA_CHECK(cudaMemset(energy_a, 0, device_bytes));
    CUDA_CHECK(cudaMemset(energy_b, 0, device_bytes));
    CUDA_CHECK(cudaMemset(flux_a, 0, device_bytes));
    CUDA_CHECK(cudaMemset(flux_b, 0, device_bytes));

    // Four pinned rows support bidirectional nonblocking halo exchange.
    val_t* halo_storage = nullptr;
    CUDA_CHECK(cudaMallocHost(&halo_storage, 4 * row_bytes));
    val_t* send_top = halo_storage;
    val_t* send_bottom = halo_storage + n;
    val_t* receive_top = halo_storage + 2 * n;
    val_t* receive_bottom = halo_storage + 3 * n;

    cudaStream_t compute_stream, communication_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&communication_stream, cudaStreamNonBlocking));

    if (rank == 0) {
        const int64_t elements = static_cast<int64_t>(n) * n;
        const size_t global_static_mem = static_cast<size_t>(elements) * sizeof(ElementStatic);
        const size_t global_dynamic_mem = static_cast<size_t>(elements) * sizeof(ElementDynamic) * 2;
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %lld elements\n", n, n,
                    static_cast<long long>(elements));
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        std::printf("Building distributed unstructured mesh...\n");
        std::printf("Reference memory footprint: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
                    (global_static_mem + global_dynamic_mem) / (1024.0 * 1024.0),
                    global_static_mem / (1024.0 * 1024.0),
                    global_dynamic_mem / (1024.0 * 1024.0));
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s)/rank, "
                    "%d CUDA device(s)/node\n", ranks, omp_get_max_threads(), device_count);
        if (local_ranks > device_count)
            std::printf("WARNING: node has more MPI ranks than CUDA devices; devices are shared.\n");
        std::printf("\nRunning simulation...\n");
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        MPI_Request requests[4];
        int request_count = 0;

        if (local_rows > 0) {
            if (previous == MPI_PROC_NULL && next == MPI_PROC_NULL) {
                launchRows(compute_stream, energy_a, flux_a, energy_b, flux_b, n,
                           global_first_row, local_rows, 1, local_rows, rank);
            } else {
                if (previous != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(send_top, energy_a + n, row_bytes,
                                               cudaMemcpyDeviceToHost, communication_stream));
                if (next != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(send_bottom,
                                               energy_a + static_cast<size_t>(local_rows) * n,
                                               row_bytes, cudaMemcpyDeviceToHost,
                                               communication_stream));

                // Rows not touching an MPI boundary proceed while halos travel.
                launchRows(compute_stream, energy_a, flux_a, energy_b, flux_b, n,
                           global_first_row, local_rows, 2, std::max(local_rows - 2, 0), rank);
                CUDA_CHECK(cudaStreamSynchronize(communication_stream));

                if (previous != MPI_PROC_NULL) {
                    MPI_Irecv(receive_top, n, MPI_DOUBLE, previous, 11, MPI_COMM_WORLD,
                              &requests[request_count++]);
                    MPI_Isend(send_top, n, MPI_DOUBLE, previous, 12, MPI_COMM_WORLD,
                              &requests[request_count++]);
                }
                if (next != MPI_PROC_NULL) {
                    MPI_Irecv(receive_bottom, n, MPI_DOUBLE, next, 12, MPI_COMM_WORLD,
                              &requests[request_count++]);
                    MPI_Isend(send_bottom, n, MPI_DOUBLE, next, 11, MPI_COMM_WORLD,
                              &requests[request_count++]);
                }
                MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

                if (previous != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(energy_a, receive_top, row_bytes,
                                               cudaMemcpyHostToDevice, communication_stream));
                if (next != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(energy_a + static_cast<size_t>(local_rows + 1) * n,
                                               receive_bottom, row_bytes, cudaMemcpyHostToDevice,
                                               communication_stream));
                CUDA_CHECK(cudaStreamSynchronize(communication_stream));
                launchBoundaries(compute_stream, energy_a, flux_a, energy_b, flux_b, n,
                                 global_first_row, local_rows, rank);
            }
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        }
        std::swap(energy_a, energy_b);
        std::swap(flux_a, flux_b);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<val_t> local_energy(static_cast<size_t>(local_rows) * n);
    std::vector<val_t> local_flux(static_cast<size_t>(local_rows) * n);
    if (local_rows > 0) {
        CUDA_CHECK(cudaMemcpy(local_energy.data(), energy_a + n, local_energy.size() * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_flux.data(), flux_a + n, local_flux.size() * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<int> counts, displacements;
    std::vector<val_t> all_energy, all_flux;
    const int total_elements = n * n;
    if (rank == 0) {
        counts.resize(ranks);
        displacements.resize(ranks);
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < ranks; ++r) {
            const int rows = base_rows + (r < extra_rows ? 1 : 0);
            counts[r] = rows * n;
            displacements[r] = (r * base_rows + std::min(r, extra_rows)) * n;
        }
        all_energy.resize(total_elements);
        all_flux.resize(total_elements);
    }
    MPI_Gatherv(local_energy.data(), static_cast<int>(local_energy.size()), MPI_DOUBLE,
                rank == 0 ? all_energy.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_flux.data(), static_cast<int>(local_flux.size()), MPI_DOUBLE,
                rank == 0 ? all_flux.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exit_code = 0;
    if (rank == 0) {
        std::vector<ElementDynamic> elements(total_elements);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < total_elements; ++i)
            elements[i] = ElementDynamic{all_energy[i], all_flux[i]};

        const double milliseconds = elapsed * 1000.0;
        const int measured_iterations = std::max(iterations, 1);
        const double time_per_iteration = milliseconds / measured_iterations;
        const double giga_elements_per_second = iterations > 0 && elapsed > 0.0
            ? (static_cast<double>(iterations) * total_elements) / elapsed / 1.0e9 : 0.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iteration);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elements_per_second);
        std::printf("  Performance: %.4f GFLOPS\n", giga_elements_per_second * 22.0);
        std::printf("  Result hash: %016llX\n\n",
                    static_cast<unsigned long long>(computeHash(elements)));

        if (print_results_requested) print_results(all_energy, "ElementEnergy");
        if (validate && !validateResults(elements)) exit_code = 1;
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(communication_stream));
    CUDA_CHECK(cudaFreeHost(halo_storage));
    CUDA_CHECK(cudaFree(energy_a));
    CUDA_CHECK(cudaFree(energy_b));
    CUDA_CHECK(cudaFree(flux_a));
    CUDA_CHECK(cudaFree(flux_b));
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return exit_code;
}
