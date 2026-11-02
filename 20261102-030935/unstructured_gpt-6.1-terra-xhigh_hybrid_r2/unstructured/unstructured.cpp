#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "../common/results_output.hpp"

using val_t = double;
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)
#define MPI_CHECK(call) do { \
    int error = (call); \
    if (error != MPI_SUCCESS) { \
        char message[MPI_MAX_ERROR_STRING]; int len = 0; \
        MPI_Error_string(error, message, &len); \
        fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, len, message); \
        MPI_Abort(MPI_COMM_WORLD, error); \
    } \
} while (0)

// Local rows are stored between two ghost rows. The neighbor order matches the
// original mesh construction: x+1, x-1, y+1, y-1.
__global__ void update(const ElementDynamic* in, ElementDynamic* out,
                       int n, int global_start, int first_row, int row_count) {
    size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = static_cast<size_t>(row_count) * n;
    if (k >= count) return;
    int local_row = first_row + static_cast<int>(k / n);
    int y = static_cast<int>(k % n);
    int x = global_start + local_row - 1;
    size_t i = static_cast<size_t>(local_row) * n + y;
    double energy = in[i].current_energy;
    // Assignments in the original mesh builder leave the only cell of a 1x1
    // grid as inflow after all four corner assignments.
    double flux = 0.0;
    if ((x == 0 && y == 0) || (x == n - 1 && y == n - 1)) flux = 0.5;
    else if ((x == 0 && y == n - 1) || (x == n - 1 && y == 0)) flux = -0.5;
    if (x + 1 < n) flux += (in[i + n].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (x > 0)     flux += (in[i - n].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (y + 1 < n) flux += (in[i + 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (y > 0)     flux += (in[i - 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
    out[i].current_energy = energy + flux;
    out[i].total_flux = in[i].total_flux + fabs(flux);
}

static void launch_update(const ElementDynamic* in, ElementDynamic* out, int n,
                          int start, int first, int rows, cudaStream_t stream) {
    if (rows <= 0) return;
    constexpr int block = 256;
    size_t count = static_cast<size_t>(rows) * n;
    update<<<static_cast<unsigned>((count + block - 1) / block), block, 0, stream>>>(
        in, out, n, start, first, rows);
    CUDA_CHECK(cudaGetLastError());
}

static bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    if (!std::isfinite(energy_sum)) { printf("  ERROR: Energy sum is not finite\n"); return false; }
    if (std::abs(energy_sum) > 1e-8)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) { printf("  ERROR: Flux sum is not finite\n"); return false; }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n"); return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n = 512, n_iters = 10;
    bool validate = false, printResults = false;
    int parse_result = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); parse_result = 1; break; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } parse_result = -1; break; }
    }
    if (parse_result || n <= 0 || static_cast<long long>(n) * n > INT_MAX) {
        if (!parse_result && rank == 0) fprintf(stderr, "Grid size must be positive and contain at most INT_MAX elements\n");
        MPI_Finalize();
        return parse_result == 1 ? 0 : 1;
    }

    // Ranks without rows do not participate in simulation collectives.
    int active_size = std::min(size, n);
    MPI_Comm active;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < active_size ? 0 : MPI_UNDEFINED, rank, &active));
    int status = 0;
    if (rank < active_size) {
        MPI_Comm local;
        MPI_CHECK(MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
        int local_rank;
        MPI_CHECK(MPI_Comm_rank(local, &local_rank));
        MPI_CHECK(MPI_Comm_free(&local));
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(local_rank % device_count));
        const int rows = n / active_size + (rank < n % active_size);
        const int start_row = rank * (n / active_size) + std::min(rank, n % active_size);
        const size_t row_bytes = static_cast<size_t>(n) * sizeof(ElementDynamic);
        const size_t allocation_bytes = static_cast<size_t>(rows + 2) * row_bytes;
        ElementDynamic *current, *next, *staging = nullptr;
        CUDA_CHECK(cudaMalloc(&current, allocation_bytes));
        CUDA_CHECK(cudaMalloc(&next, allocation_bytes));
        CUDA_CHECK(cudaMemset(current, 0, allocation_bytes));
        CUDA_CHECK(cudaMemset(next, 0, allocation_bytes));
        cudaStream_t interior, boundary;
        CUDA_CHECK(cudaStreamCreateWithFlags(&interior, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&boundary, cudaStreamNonBlocking));
        if (active_size > 1) CUDA_CHECK(cudaMallocHost(&staging, 4 * row_bytes));
        if (rank == 0) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n");
            printf("============================================\n");
            printf("Grid size: %d x %d = %d elements\n", n, n, n * n);
            printf("Iterations: %d\n", n_iters);
            printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
            printf("Building unstructured mesh...\n");
            printf("MPI ranks: %d (active: %d)\n", size, active_size);
            printf("GPU state memory: %.2f MB across active ranks\n\n",
                   (2.0 * (static_cast<double>(n) * n + 2.0 * active_size * n) * sizeof(ElementDynamic)) / (1024.0 * 1024.0));
            printf("Running simulation...\n");
        }
        MPI_CHECK(MPI_Barrier(active));
        auto begin = std::chrono::steady_clock::now();
        for (int iter = 0; iter < n_iters; ++iter) {
            if (active_size == 1) {
                launch_update(current, next, n, start_row, 1, rows, interior);
            } else {
                MPI_Request requests[4];
                int request_count = 0;
                const int previous = rank > 0 ? rank - 1 : MPI_PROC_NULL;
                const int following = rank + 1 < active_size ? rank + 1 : MPI_PROC_NULL;
                // Staging rows: send first, send last, receive first halo, receive last halo.
                if (previous != MPI_PROC_NULL) {
                    MPI_CHECK(MPI_Irecv(staging + 2 * n, static_cast<int>(row_bytes), MPI_BYTE,
                                        previous, 1, active, &requests[request_count++]));
                    CUDA_CHECK(cudaMemcpyAsync(staging, current + n, row_bytes,
                                               cudaMemcpyDeviceToHost, boundary));
                }
                if (following != MPI_PROC_NULL) {
                    MPI_CHECK(MPI_Irecv(staging + 3 * n, static_cast<int>(row_bytes), MPI_BYTE,
                                        following, 0, active, &requests[request_count++]));
                    CUDA_CHECK(cudaMemcpyAsync(staging + n, current + static_cast<size_t>(rows) * n,
                                               row_bytes, cudaMemcpyDeviceToHost, boundary));
                }
                launch_update(current, next, n, start_row, 2, rows - 2, interior);
                CUDA_CHECK(cudaStreamSynchronize(boundary));
                if (previous != MPI_PROC_NULL)
                    MPI_CHECK(MPI_Isend(staging, static_cast<int>(row_bytes), MPI_BYTE,
                                        previous, 0, active, &requests[request_count++]));
                if (following != MPI_PROC_NULL)
                    MPI_CHECK(MPI_Isend(staging + n, static_cast<int>(row_bytes), MPI_BYTE,
                                        following, 1, active, &requests[request_count++]));
                MPI_CHECK(MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE));
                if (previous != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(current, staging + 2 * n, row_bytes,
                                               cudaMemcpyHostToDevice, boundary));
                if (following != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(current + static_cast<size_t>(rows + 1) * n,
                                               staging + 3 * n, row_bytes,
                                               cudaMemcpyHostToDevice, boundary));
                launch_update(current, next, n, start_row, 1, 1, boundary);
                if (rows > 1) launch_update(current, next, n, start_row, rows, 1, boundary);
                CUDA_CHECK(cudaStreamSynchronize(interior));
                CUDA_CHECK(cudaStreamSynchronize(boundary));
            }
            std::swap(current, next);
        }
        CUDA_CHECK(cudaStreamSynchronize(interior));
        double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        double max_elapsed_ms = 0;
        MPI_CHECK(MPI_Reduce(&elapsed_ms, &max_elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, active));

        const int local_count = rows * n;
        std::vector<ElementDynamic> local_data(local_count);
        CUDA_CHECK(cudaMemcpy(local_data.data(), current + n, static_cast<size_t>(local_count) * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
        unsigned long long local_hash = 0;
        const size_t offset = static_cast<size_t>(start_row) * n;
        #pragma omp parallel for reduction(^:local_hash) schedule(static)
        for (int i = 0; i < local_count; ++i) {
            uint64_t energy_bits, flux_bits;
            memcpy(&energy_bits, &local_data[i].current_energy, sizeof(energy_bits));
            memcpy(&flux_bits, &local_data[i].total_flux, sizeof(flux_bits));
            uint64_t global_index = offset + static_cast<size_t>(i);
            local_hash ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
        }
        unsigned long long global_hash = 0;
        MPI_CHECK(MPI_Reduce(&local_hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, active));
        std::vector<ElementDynamic> all_data;
        if (validate || printResults) {
            std::vector<int> counts, displacements;
            if (rank == 0) {
                counts.resize(active_size);
                displacements.resize(active_size);
                for (int r = 0; r < active_size; ++r) {
                    counts[r] = (n / active_size + (r < n % active_size)) * n;
                    displacements[r] = (r * (n / active_size) + std::min(r, n % active_size)) * n;
                }
                all_data.resize(static_cast<size_t>(n) * n);
            }
            MPI_Datatype element_type;
            MPI_CHECK(MPI_Type_contiguous(2, MPI_DOUBLE, &element_type));
            MPI_CHECK(MPI_Type_commit(&element_type));
            MPI_CHECK(MPI_Gatherv(local_data.data(), local_count, element_type, all_data.data(),
                                  counts.data(), displacements.data(), element_type, 0, active));
            MPI_CHECK(MPI_Type_free(&element_type));
        }
        if (rank == 0) {
            printf("Computation time: %.3f ms\n", max_elapsed_ms);
            const int measured = std::max(n_iters, 0);
            double time_per_iter = measured ? max_elapsed_ms / measured : 0.0;
            double giga_elems_per_sec = max_elapsed_ms > 0.0
                ? (static_cast<double>(measured) * n * n) / (max_elapsed_ms * 1e6) : 0.0;
            printf("Performance:\n");
            printf("  Time per iteration: %.4f ms\n", time_per_iter);
            printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
            printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
            printf("  Result hash: %016llX\n\n", global_hash);
            if (printResults) {
                std::vector<double> energies(all_data.size());
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < all_data.size(); ++i) energies[i] = all_data[i].current_energy;
                print_results(energies, "ElementEnergy");
            }
            if (validate && !validateResults(all_data)) status = 1;
        }
        if (staging) CUDA_CHECK(cudaFreeHost(staging));
        CUDA_CHECK(cudaStreamDestroy(interior));
        CUDA_CHECK(cudaStreamDestroy(boundary));
        CUDA_CHECK(cudaFree(current));
        CUDA_CHECK(cudaFree(next));
        MPI_CHECK(MPI_Comm_free(&active));
    }
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return status;
}
