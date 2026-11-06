#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using val_t = double;
struct ElementDynamic { val_t current_energy, total_flux; };

#define CUDA_CHECK(call) do { cudaError_t err_ = (call); if (err_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { int err_ = (call); if (err_ != MPI_SUCCESS) { \
    char msg_[MPI_MAX_ERROR_STRING]; int len_; MPI_Error_string(err_, msg_, &len_); \
    fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, len_, msg_); \
    MPI_Abort(MPI_COMM_WORLD, err_); } } while (0)

// The halo rows occupy rows 0 and rows+1; owned rows start at 1.
// A second invocation handles the two boundary rows after halo exchange.
__global__ void update(const double* __restrict__ old_energy,
                       double* __restrict__ new_energy, double* __restrict__ flux,
                       int n, int rows, int first_global_row, int first_row,
                       int row_count, bool boundaries) {
    size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = static_cast<size_t>(row_count) * n;
    if (k >= count) return;
    int local_row = static_cast<int>(k / n);
    if (boundaries) local_row = (local_row == 0 ? 0 : rows - 1);
    else local_row += first_row;
    int y = static_cast<int>(k % n);
    int x = first_global_row + local_row;
    size_t p = (static_cast<size_t>(local_row) + 1) * n + y;
    double e = old_energy[p];
    double f = 0.0;
    // Corner assignment order in the original makes the single-cell grid an inflow.
    if ((x == 0 && y == 0) || (x == n - 1 && y == n - 1)) f = 0.5;
    else if ((x == 0 && y == n - 1) || (x == n - 1 && y == 0)) f = -0.5;
    if (x + 1 < n) f += (old_energy[p + n] - e) * 0.8 * 1.0 * 0.25;
    if (x > 0)     f += (old_energy[p - n] - e) * 0.8 * 1.0 * 0.25;
    if (y + 1 < n) f += (old_energy[p + 1] - e) * 0.8 * 1.0 * 0.25;
    if (y > 0)     f += (old_energy[p - 1] - e) * 0.8 * 1.0 * 0.25;
    new_energy[p] = e + f;
    size_t q = static_cast<size_t>(local_row) * n + y;
    flux[q] += fabs(f);
}

bool validateResults(const std::vector<ElementDynamic>& elements) {
    double energy_sum = 0.0, flux_sum = 0.0;
    double energy_max = std::numeric_limits<double>::lowest();
    double energy_min = std::numeric_limits<double>::max();
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
    if (std::abs(energy_sum) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) { printf("  ERROR: Flux sum is not finite\n"); return false; }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n"); return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank, size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    int n = 512, n_iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (n <= 0 || static_cast<int64_t>(n) * n > std::numeric_limits<int>::max()) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and its square must fit in an int.\n");
        MPI_Finalize(); return 1;
    }
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int active = std::min(n, size);
    const int rows = rank < active ? n / active + (rank < n % active) : 0;
    const int first = rank < active ? rank * (n / active) + std::min(rank, n % active) : n;
    const size_t local_count = static_cast<size_t>(rows) * n;
    const size_t total_count = static_cast<size_t>(n) * n;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n, n, total_count);
        printf("Iterations: %d\nValidation: %s\n\n", n_iters, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
        // Connectivity is implicit. Each rank stores two energy buffers with
        // two halo rows apiece, plus one accumulated-flux array.
        const size_t static_mem = 0;
        const size_t dynamic_mem = (3 * total_count + static_cast<size_t>(4) * active * n) * sizeof(double);
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (static_mem + dynamic_mem) / 1048576.0, static_mem / 1048576.0, dynamic_mem / 1048576.0);
        printf("Running simulation...\n");
    }
    double *energy[2] = {nullptr, nullptr}, *flux = nullptr, *host_halos = nullptr;
    cudaStream_t compute = nullptr, transfer = nullptr;
    cudaEvent_t ready = nullptr, halos_ready = nullptr;
    if (rows) {
        MPI_Comm local_comm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
        int local_rank, devices;
        MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
        MPI_CHECK(MPI_Comm_free(&local_comm));
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices == 0) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
        CUDA_CHECK(cudaSetDevice(local_rank % devices));
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&halos_ready, cudaEventDisableTiming));
        size_t energy_bytes = static_cast<size_t>(rows + 2) * n * sizeof(double);
        CUDA_CHECK(cudaMalloc(&energy[0], energy_bytes));
        CUDA_CHECK(cudaMalloc(&energy[1], energy_bytes));
        CUDA_CHECK(cudaMalloc(&flux, local_count * sizeof(double)));
        CUDA_CHECK(cudaMemsetAsync(energy[0], 0, energy_bytes, compute));
        CUDA_CHECK(cudaMemsetAsync(energy[1], 0, energy_bytes, compute));
        CUDA_CHECK(cudaMemsetAsync(flux, 0, local_count * sizeof(double), compute));
        if (active > 1) CUDA_CHECK(cudaMallocHost(&host_halos, static_cast<size_t>(4) * n * sizeof(double)));
        CUDA_CHECK(cudaEventRecord(ready, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
    } else {
        // MPI_Comm_split_type is collective on the whole world, including idle ranks.
        MPI_Comm local_comm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
        MPI_CHECK(MPI_Comm_free(&local_comm));
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    int current = 0;
    constexpr int block = 256;
    if (rows) for (int iter = 0; iter < n_iters; ++iter) {
        if (active == 1) {
            size_t count = local_count;
            update<<<static_cast<unsigned>((count + block - 1) / block), block, 0, compute>>>(
                energy[current], energy[1-current], flux, n, rows, first, 0, rows, false);
            CUDA_CHECK(cudaGetLastError());
        } else {
            // The previous host-to-device halo copies must finish before MPI
            // can reuse the pinned receive buffers in this iteration.
            if (iter > 0) CUDA_CHECK(cudaEventSynchronize(halos_ready));
            MPI_Request requests[4]; int nr = 0;
            int up = rank > 0 ? rank - 1 : MPI_PROC_NULL;
            int down = rank + 1 < active ? rank + 1 : MPI_PROC_NULL;
            if (up != MPI_PROC_NULL) MPI_CHECK(MPI_Irecv(host_halos + 2*n, n, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &requests[nr++]));
            if (down != MPI_PROC_NULL) MPI_CHECK(MPI_Irecv(host_halos + 3*n, n, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &requests[nr++]));
            CUDA_CHECK(cudaStreamWaitEvent(transfer, ready, 0));
            if (up != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(host_halos, energy[current] + n, static_cast<size_t>(n)*sizeof(double), cudaMemcpyDeviceToHost, transfer));
            if (down != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(host_halos + n, energy[current] + static_cast<size_t>(rows)*n, static_cast<size_t>(n)*sizeof(double), cudaMemcpyDeviceToHost, transfer));
            if (rows > 2) {
                size_t count = static_cast<size_t>(rows - 2) * n;
                update<<<static_cast<unsigned>((count + block - 1) / block), block, 0, compute>>>(
                    energy[current], energy[1-current], flux, n, rows, first, 1, rows - 2, false);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaStreamSynchronize(transfer));
            if (up != MPI_PROC_NULL) MPI_CHECK(MPI_Isend(host_halos, n, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &requests[nr++]));
            if (down != MPI_PROC_NULL) MPI_CHECK(MPI_Isend(host_halos + n, n, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &requests[nr++]));
            MPI_CHECK(MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE));
            if (up != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(energy[current], host_halos + 2*n, static_cast<size_t>(n)*sizeof(double), cudaMemcpyHostToDevice, transfer));
            if (down != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(energy[current] + static_cast<size_t>(rows+1)*n, host_halos + 3*n, static_cast<size_t>(n)*sizeof(double), cudaMemcpyHostToDevice, transfer));
            CUDA_CHECK(cudaEventRecord(halos_ready, transfer));
            CUDA_CHECK(cudaStreamWaitEvent(compute, halos_ready, 0));
            int boundary_rows = rows == 1 ? 1 : 2;
            size_t count = static_cast<size_t>(boundary_rows) * n;
            update<<<static_cast<unsigned>((count + block - 1) / block), block, 0, compute>>>(
                energy[current], energy[1-current], flux, n, rows, first, 0, boundary_rows, true);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(ready, compute));
        }
        current = 1 - current;
    }
    if (rows) CUDA_CHECK(cudaStreamSynchronize(compute));
    double elapsed = MPI_Wtime() - start, max_elapsed = 0;
    MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    std::vector<double> local_energy(local_count), local_flux(local_count);
    if (rows) {
        CUDA_CHECK(cudaMemcpy(local_energy.data(), energy[current] + n, local_count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_flux.data(), flux, local_count * sizeof(double), cudaMemcpyDeviceToHost));
    }
    uint64_t local_hash = 0;
    #pragma omp parallel for reduction(^:local_hash) schedule(static) if(local_count >= 16384)
    for (size_t i = 0; i < local_count; ++i) {
        uint64_t e, f;
        memcpy(&e, &local_energy[i], sizeof(e)); memcpy(&f, &local_flux[i], sizeof(f));
        uint64_t global_i = static_cast<uint64_t>(first) * n + i;
        local_hash ^= (e + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (f + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t hash = 0;
    MPI_CHECK(MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD));
    std::vector<ElementDynamic> all;
    if (validate || printResults) {
        // Gather in row order, in chunks to avoid MPI's int count/displacement limit.
        if (rank == 0) all.resize(total_count);
        std::vector<ElementDynamic> local(local_count);
        #pragma omp parallel for schedule(static) if(local_count >= 16384)
        for (size_t i = 0; i < local_count; ++i) local[i] = {local_energy[i], local_flux[i]};
        if (rank == 0) std::copy(local.begin(), local.end(), all.begin());
        constexpr size_t chunk = static_cast<size_t>(1) << 26;
        for (int r = 1; r < active; ++r) {
            size_t rrows = n / active + (r < n % active);
            size_t offset = static_cast<size_t>(r * (n / active) + std::min(r, n % active)) * n;
            size_t bytes = rrows * n * sizeof(ElementDynamic);
            for (size_t pos = 0; pos < bytes; pos += chunk) {
                int count = static_cast<int>(std::min(chunk, bytes - pos));
                if (rank == r) MPI_CHECK(MPI_Send(reinterpret_cast<const char*>(local.data()) + pos, count, MPI_BYTE, 0, 2, MPI_COMM_WORLD));
                if (rank == 0) MPI_CHECK(MPI_Recv(reinterpret_cast<char*>(all.data() + offset) + pos, count, MPI_BYTE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            }
        }
    }
    int valid = 1;
    if (rank == 0) {
        double duration_ms = max_elapsed * 1000.0;
        printf("Computation time: %.4f ms\n", duration_ms);
        int measured = std::max(n_iters - 1, 1);
        double per_iter = duration_ms / measured;
        double rate = (static_cast<double>(measured) * total_count) / (duration_ms / 1000.0) / 1e9;
        printf("Performance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", per_iter, rate, rate * 22.0);
        printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
        if (printResults) {
            std::vector<double> data(total_count);
            #pragma omp parallel for schedule(static) if(total_count >= 16384)
            for (size_t i = 0; i < total_count; ++i) data[i] = all[i].current_energy;
            print_results(data, "ElementEnergy");
        }
        if (validate && !validateResults(all)) valid = 0;
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (rows) {
        if (host_halos) CUDA_CHECK(cudaFreeHost(host_halos));
        CUDA_CHECK(cudaFree(energy[0])); CUDA_CHECK(cudaFree(energy[1])); CUDA_CHECK(cudaFree(flux));
        CUDA_CHECK(cudaEventDestroy(ready)); CUDA_CHECK(cudaEventDestroy(halos_ready));
        CUDA_CHECK(cudaStreamDestroy(compute)); CUDA_CHECK(cudaStreamDestroy(transfer));
    }
    MPI_CHECK(MPI_Finalize());
    return valid ? 0 : 1;
}
