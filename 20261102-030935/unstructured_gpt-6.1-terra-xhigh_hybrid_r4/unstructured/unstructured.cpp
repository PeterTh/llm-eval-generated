#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

using val_t = double;
struct ElementDynamic { val_t current_energy, total_flux; };

#define CUDA(call) do { cudaError_t err_ = (call); if (err_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { int err_ = (call); if (err_ != MPI_SUCCESS) { \
    char msg_[MPI_MAX_ERROR_STRING]; int len_; MPI_Error_string(err_, msg_, &len_); \
    fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, len_, msg_); \
    MPI_Abort(MPI_COMM_WORLD, err_); } } while (0)

// The generated mesh has unit-flux links in the order x+1, x-1, y+1, y-1.
// Energy has one halo row on either side; accumulated flux is local only.
__global__ void update(const double* energy, const double* flux, double* next_energy,
                       double* next_flux, int n, int start_row, int rows,
                       int first_row, int row_count, bool edges_only) {
    size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = size_t(row_count) * n;
    if (t >= count) return;
    int row = first_row + int(t / n);
    if (edges_only) row = (row == 0 ? 1 : rows);
    int col = int(t % n);
    int x = start_row + row - 1;
    size_t p = size_t(row) * n + col;
    size_t q = size_t(row - 1) * n + col;
    double e = energy[p];
    double f = 0.0;
    if (x == 0 && col == 0) f = 0.5;
    if (x == 0 && col == n - 1) f = -0.5;
    if (x == n - 1 && col == 0) f = -0.5;
    if (x == n - 1 && col == n - 1) f = 0.5;
    if (x + 1 < n) f += (energy[p + n] - e) * 0.8 * 0.25;
    if (x > 0)     f += (energy[p - n] - e) * 0.8 * 0.25;
    if (col + 1 < n) f += (energy[p + 1] - e) * 0.8 * 0.25;
    if (col > 0)     f += (energy[p - 1] - e) * 0.8 * 0.25;
    next_energy[p] = e + f;
    next_flux[q] = flux[q] + fabs(f);
}

// Keep verification distributed and transfer only one hash per CUDA block.
__global__ void hashElements(const double* energy, const double* flux, size_t count,
                             uint64_t offset, uint64_t* block_hashes) {
    __shared__ uint64_t partial[256];
    uint64_t h = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(blockDim.x) * gridDim.x) {
        uint64_t e = static_cast<uint64_t>(__double_as_longlong(energy[i]));
        uint64_t f = static_cast<uint64_t>(__double_as_longlong(flux[i]));
        uint64_t index = offset + i;
        h ^= (e + index) * 0x9e3779b97f4a7c15ULL;
        h ^= (f + index) * 0xbf58476d1ce4e5b9ULL;
    }
    partial[threadIdx.x] = h;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] ^= partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) block_hashes[blockIdx.x] = partial[0];
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

bool validateResults(const std::vector<ElementDynamic>& elements) {
    double energy_sum = 0, flux_sum = 0;
    double energy_max = std::numeric_limits<double>::lowest();
    double energy_min = std::numeric_limits<double>::max();
    for (const auto& e : elements) {
        energy_sum += e.current_energy; flux_sum += e.total_flux;
        energy_max = std::max(e.current_energy, energy_max);
        energy_min = std::min(e.current_energy, energy_min);
    }
    printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", energy_sum, flux_sum, energy_min, energy_max);
    if (!std::isfinite(energy_sum)) { printf("  ERROR: Energy sum is not finite\n"); return false; }
    if (std::abs(energy_sum) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) { printf("  ERROR: Flux sum is not finite\n"); return false; }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) { printf("  ERROR: Energy extrema are not finite\n"); return false; }
    printf("  Validation: PASSED\n"); return true;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_FUNNELED) { if (!rank) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    int n = 512, iters = 10; bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    // MPI_Gatherv uses int counts and displacements. Original code uses an int element count.
    if (n <= 0 || int64_t(n) * n > INT_MAX) {
        if (!rank) fprintf(stderr, "Grid size must be positive and its square must fit in an int\n");
        MPI_Finalize(); return 1;
    }
    int active = std::min(size, n);
    MPI_Comm comm;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm));
    int status = 0;
    if (rank < active) {
        MPI_Comm local_comm;
        MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
        int local_rank; MPI_Comm_rank(local_comm, &local_rank);
        int device_count = 0; CUDA(cudaGetDeviceCount(&device_count));
        if (!device_count) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
        CUDA(cudaSetDevice(local_rank % device_count));
        MPI_Comm_free(&local_comm);
        int rows = n / active + (rank < n % active);
        int start = rank * (n / active) + std::min(rank, n % active);
        size_t local_count = size_t(rows) * n, halo_count = size_t(rows + 2) * n;
        if (!rank) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
            printf("Grid size: %d x %d = %d elements\nIterations: %d\nValidation: %s\n\n", n, n, n * n, iters, validate ? "enabled" : "disabled");
            printf("Building unstructured mesh...\n");
            printf("MPI ranks: %d, row-decomposed CUDA mesh\n\nRunning simulation...\n", active);
        }
        double *energy[2], *flux[2], *halo;
        CUDA(cudaMalloc(&energy[0], halo_count * sizeof(double))); CUDA(cudaMalloc(&energy[1], halo_count * sizeof(double)));
        CUDA(cudaMalloc(&flux[0], local_count * sizeof(double))); CUDA(cudaMalloc(&flux[1], local_count * sizeof(double)));
        CUDA(cudaMallocHost(&halo, size_t(4) * n * sizeof(double)));
        cudaStream_t compute, transfer;
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        cudaEvent_t ready, copied;
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming)); CUDA(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
        CUDA(cudaMemsetAsync(energy[0], 0, halo_count * sizeof(double), compute));
        CUDA(cudaMemsetAsync(flux[0], 0, local_count * sizeof(double), compute));
        CUDA(cudaEventRecord(ready, compute));
        CUDA(cudaEventSynchronize(ready));
        MPI_CHECK(MPI_Barrier(comm));
        double begin = MPI_Wtime();
        constexpr int block = 256;
        int cur = 0;
        for (int iter = 0; iter < iters; ++iter) {
            int next = cur ^ 1;
            if (active == 1) {
                update<<<(local_count + block - 1) / block, block, 0, compute>>>(energy[cur], flux[cur], energy[next], flux[next], n, start, rows, 1, rows, false);
            } else {
                MPI_Request requests[4]; int nr = 0;
                int prev = rank > 0 ? rank - 1 : MPI_PROC_NULL;
                int next_rank = rank + 1 < active ? rank + 1 : MPI_PROC_NULL;
                MPI_CHECK(MPI_Irecv(halo + 2 * n, n, MPI_DOUBLE, prev, 1, comm, &requests[nr++]));
                MPI_CHECK(MPI_Irecv(halo + 3 * n, n, MPI_DOUBLE, next_rank, 0, comm, &requests[nr++]));
                CUDA(cudaStreamWaitEvent(transfer, ready, 0));
                if (prev != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(halo, energy[cur] + n, size_t(n) * sizeof(double), cudaMemcpyDeviceToHost, transfer));
                if (next_rank != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(halo + n, energy[cur] + size_t(rows) * n, size_t(n) * sizeof(double), cudaMemcpyDeviceToHost, transfer));
                CUDA(cudaEventRecord(copied, transfer));
                if (rows > 2) {
                    size_t count = size_t(rows - 2) * n;
                    update<<<(count + block - 1) / block, block, 0, compute>>>(energy[cur], flux[cur], energy[next], flux[next], n, start, rows, 2, rows - 2, false);
                }
                CUDA(cudaEventSynchronize(copied));
                MPI_CHECK(MPI_Isend(halo, n, MPI_DOUBLE, prev, 0, comm, &requests[nr++]));
                MPI_CHECK(MPI_Isend(halo + n, n, MPI_DOUBLE, next_rank, 1, comm, &requests[nr++]));
                MPI_CHECK(MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE));
                if (prev != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(energy[cur], halo + 2 * n, size_t(n) * sizeof(double), cudaMemcpyHostToDevice, transfer));
                if (next_rank != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(energy[cur] + size_t(rows + 1) * n, halo + 3 * n, size_t(n) * sizeof(double), cudaMemcpyHostToDevice, transfer));
                CUDA(cudaEventRecord(copied, transfer));
                CUDA(cudaStreamWaitEvent(compute, copied, 0));
                int edge_rows = rows > 1 ? 2 : 1;
                size_t count = size_t(edge_rows) * n;
                update<<<(count + block - 1) / block, block, 0, compute>>>(energy[cur], flux[cur], energy[next], flux[next], n, start, rows, 0, edge_rows, true);
            }
            CUDA(cudaGetLastError());
            CUDA(cudaEventRecord(ready, compute));
            cur = next;
        }
        CUDA(cudaStreamSynchronize(compute));
        double elapsed = MPI_Wtime() - begin, max_elapsed = 0;
        MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
        int hash_blocks = int(std::min<size_t>(4096, (local_count + block - 1) / block));
        uint64_t* device_hash;
        CUDA(cudaMalloc(&device_hash, size_t(hash_blocks) * sizeof(uint64_t)));
        hashElements<<<hash_blocks, block>>>(energy[cur] + n, flux[cur], local_count,
                                             uint64_t(start) * n, device_hash);
        CUDA(cudaGetLastError());
        std::vector<uint64_t> hashes(hash_blocks);
        CUDA(cudaMemcpy(hashes.data(), device_hash, size_t(hash_blocks) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA(cudaFree(device_hash));
        uint64_t local_hash = 0;
        #pragma omp parallel for reduction(^:local_hash) schedule(static) if(hash_blocks >= 1024) num_threads(std::min(omp_get_max_threads(), 8))
        for (int i = 0; i < hash_blocks; ++i) local_hash ^= hashes[i];
        uint64_t hash = 0;
        MPI_CHECK(MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm));
        if (!rank) {
            int measured = std::max(iters, 0);
            double ms = max_elapsed * 1000.0;
            double rate = measured && max_elapsed > 0 ? measured * double(n) * n / max_elapsed / 1e9 : 0.0;
            printf("Computation time: %.3f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n", ms, measured ? ms / measured : 0.0, rate, rate * 22, (unsigned long long)hash);
        }
        if (validate || print_results_flag) {
            std::vector<double> host_energy(local_count), host_flux;
            CUDA(cudaMemcpy(host_energy.data(), energy[cur] + n, local_count * sizeof(double), cudaMemcpyDeviceToHost));
            if (validate) {
                host_flux.resize(local_count);
                CUDA(cudaMemcpy(host_flux.data(), flux[cur], local_count * sizeof(double), cudaMemcpyDeviceToHost));
            }
            std::vector<int> counts(active), displs(active);
            for (int r = 0; r < active; ++r) { counts[r] = (n / active + (r < n % active)) * n; displs[r] = (r * (n / active) + std::min(r, n % active)) * n; }
            std::vector<double> all_energy, all_flux;
            if (!rank) { all_energy.resize(size_t(n) * n); if (validate) all_flux.resize(size_t(n) * n); }
            MPI_CHECK(MPI_Gatherv(host_energy.data(), int(local_count), MPI_DOUBLE, all_energy.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm));
            if (validate) MPI_CHECK(MPI_Gatherv(host_flux.data(), int(local_count), MPI_DOUBLE, all_flux.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm));
            if (!rank) {
                if (print_results_flag) print_results(all_energy, "ElementEnergy");
                if (validate) {
                    std::vector<ElementDynamic> elements(all_energy.size());
                    #pragma omp parallel for schedule(static) if(elements.size() >= 65536)
                    for (size_t i = 0; i < elements.size(); ++i) elements[i] = {all_energy[i], all_flux[i]};
                    status = validateResults(elements) ? 0 : 1;
                }
            }
        }
        CUDA(cudaEventDestroy(copied)); CUDA(cudaEventDestroy(ready));
        CUDA(cudaStreamDestroy(transfer)); CUDA(cudaStreamDestroy(compute)); CUDA(cudaFreeHost(halo));
        CUDA(cudaFree(energy[0])); CUDA(cudaFree(energy[1])); CUDA(cudaFree(flux[0])); CUDA(cudaFree(flux[1]));
        MPI_Comm_free(&comm);
    }
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize(); return status;
}
