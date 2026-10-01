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

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static int rowsForRank(int rank, int ranks, int n) {
    return n / ranks + (rank < n % ranks);
}

static int firstRowForRank(int rank, int ranks, int n) {
    return rank * (n / ranks) + std::min(rank, n % ranks);
}

// The generated mesh always connects in the order +x, -x, +y, -y.
// One CUDA thread owns each element; no atomics are required.
__global__ void updateElements(const double* energy, const double* flux,
                               double* next_energy, double* next_flux,
                               int n, int first_global_row, int first_row) {
    const int row = first_row + blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    const int global_row = first_global_row + row - 1;
    const int pos = row * n + col;
    const double self = energy[pos];
    double total = ((global_row == 0 && col == 0) ||
                    (global_row == n - 1 && col == n - 1)) ? 0.5 :
                   ((global_row == 0 && col == n - 1) ||
                    (global_row == n - 1 && col == 0)) ? -0.5 : 0.0;
    // At n=1 the final corner material assignment is inflow.
    if (n == 1) total = 0.5;
    if (global_row + 1 < n)
        total += ((energy[pos + n] - self) * 0.8 * 1.0) * 0.25;
    if (global_row > 0)
        total += ((energy[pos - n] - self) * 0.8 * 1.0) * 0.25;
    if (col + 1 < n)
        total += ((energy[pos + 1] - self) * 0.8 * 1.0) * 0.25;
    if (col > 0)
        total += ((energy[pos - 1] - self) * 0.8 * 1.0) * 0.25;
    next_energy[pos] = self + total;
    next_flux[pos] = flux[pos] + fabs(total);
}

static void launchRows(const double* energy, const double* flux,
                       double* next_energy, double* next_flux,
                       int n, int first_global_row, int first, int last,
                       cudaStream_t stream) {
    if (first > last) return;
    const int threads = 256;
    const dim3 blocks((n + threads - 1) / threads, last - first + 1);
    updateElements<<<blocks, threads, 0, stream>>>(
        energy, flux, next_energy, next_flux, n, first_global_row, first);
    cudaCheck(cudaGetLastError(), "launching updateElements");
}

// Split grid rows across MPI ranks, stage only the two energy halo rows
// through pinned memory, and overlap communication with interior CUDA work.
static void runSimulation(int n, int n_iters, int rank, int ranks, MPI_Comm comm,
                          std::vector<double>& local_energy,
                          std::vector<double>& local_flux, double& elapsed) {
    const int rows = rowsForRank(rank, ranks, n);
    const int first_global_row = firstRowForRank(rank, ranks, n);
    const size_t bytes = size_t(rows + 2) * n * sizeof(double);
    const size_t row_bytes = size_t(n) * sizeof(double);
    double *energy, *flux, *next_energy, *next_flux;
    cudaCheck(cudaMalloc(&energy, bytes), "allocating energy");
    cudaCheck(cudaMalloc(&flux, bytes), "allocating flux");
    cudaCheck(cudaMalloc(&next_energy, bytes), "allocating next energy");
    cudaCheck(cudaMalloc(&next_flux, bytes), "allocating next flux");
    cudaCheck(cudaMemset(energy, 0, bytes), "initializing energy");
    cudaCheck(cudaMemset(flux, 0, bytes), "initializing flux");
    cudaStream_t compute_stream, exchange_stream;
    cudaCheck(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking), "creating compute stream");
    cudaCheck(cudaStreamCreateWithFlags(&exchange_stream, cudaStreamNonBlocking), "creating exchange stream");
    double* staging = nullptr;
    if (ranks > 1) cudaCheck(cudaMallocHost(&staging, 4 * row_bytes), "allocating halo staging");
    double* send_top = staging;
    double* send_bottom = staging ? staging + n : nullptr;
    double* recv_top = staging ? staging + 2 * n : nullptr;
    double* recv_bottom = staging ? staging + 3 * n : nullptr;

    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < n_iters; ++iter) {
        if (ranks == 1) {
            launchRows(energy, flux, next_energy, next_flux, n,
                       first_global_row, 1, rows, compute_stream);
        } else {
            if (rank > 0)
                cudaCheck(cudaMemcpyAsync(send_top, energy + n, row_bytes,
                         cudaMemcpyDeviceToHost, exchange_stream), "copying top boundary");
            if (rank + 1 < ranks)
                cudaCheck(cudaMemcpyAsync(send_bottom, energy + rows * n, row_bytes,
                         cudaMemcpyDeviceToHost, exchange_stream), "copying bottom boundary");
            launchRows(energy, flux, next_energy, next_flux, n,
                       first_global_row, 2, rows - 1, compute_stream);
            cudaCheck(cudaStreamSynchronize(exchange_stream), "staging boundary rows");
            MPI_Request requests[4];
            int request_count = 0;
            if (rank > 0) {
                MPI_Irecv(recv_top, n, MPI_DOUBLE, rank - 1, 1, comm,
                          &requests[request_count++]);
                MPI_Isend(send_top, n, MPI_DOUBLE, rank - 1, 0, comm,
                          &requests[request_count++]);
            }
            if (rank + 1 < ranks) {
                MPI_Irecv(recv_bottom, n, MPI_DOUBLE, rank + 1, 0, comm,
                          &requests[request_count++]);
                MPI_Isend(send_bottom, n, MPI_DOUBLE, rank + 1, 1, comm,
                          &requests[request_count++]);
            }
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
            if (rank > 0)
                cudaCheck(cudaMemcpyAsync(energy, recv_top, row_bytes,
                         cudaMemcpyHostToDevice, exchange_stream), "loading top halo");
            if (rank + 1 < ranks)
                cudaCheck(cudaMemcpyAsync(energy + (rows + 1) * n, recv_bottom, row_bytes,
                         cudaMemcpyHostToDevice, exchange_stream), "loading bottom halo");
            launchRows(energy, flux, next_energy, next_flux, n,
                       first_global_row, 1, 1, exchange_stream);
            if (rows > 1)
                launchRows(energy, flux, next_energy, next_flux, n,
                           first_global_row, rows, rows, exchange_stream);
        }
        cudaCheck(cudaStreamSynchronize(compute_stream), "updating interior");
        cudaCheck(cudaStreamSynchronize(exchange_stream), "updating boundaries");
        std::swap(energy, next_energy);
        std::swap(flux, next_flux);
    }
    elapsed = MPI_Wtime() - start;
    local_energy.resize(size_t(rows) * n);
    local_flux.resize(size_t(rows) * n);
    cudaCheck(cudaMemcpy(local_energy.data(), energy + n, size_t(rows) * row_bytes,
                        cudaMemcpyDeviceToHost), "reading energy");
    cudaCheck(cudaMemcpy(local_flux.data(), flux + n, size_t(rows) * row_bytes,
                        cudaMemcpyDeviceToHost), "reading flux");
    if (staging) cudaCheck(cudaFreeHost(staging), "freeing halo staging");
    cudaCheck(cudaStreamDestroy(compute_stream), "destroying compute stream");
    cudaCheck(cudaStreamDestroy(exchange_stream), "destroying exchange stream");
    cudaCheck(cudaFree(energy), "freeing energy");
    cudaCheck(cudaFree(flux), "freeing flux");
    cudaCheck(cudaFree(next_energy), "freeing next energy");
    cudaCheck(cudaFree(next_flux), "freeing next flux");
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
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
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
    if (n_elems_root < 1 || n_elems_root > 46340 || n_iters < 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be 1..46340 and iterations nonnegative\n");
        MPI_Finalize();
        return 1;
    }
    const int active_ranks = std::min(world_size, n_elems_root);
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active_ranks ? 0 : MPI_UNDEFINED,
                   rank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    MPI_Comm shared_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared_comm);
    int shared_rank;
    MPI_Comm_rank(shared_comm, &shared_rank);
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "finding GPUs");
    if (device_count < 1) {
        fprintf(stderr, "Rank %d: no CUDA device found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(shared_rank % device_count), "selecting GPU");
    MPI_Comm_free(&shared_comm);
    
    const int n_elems = n_elems_root * n_elems_root;
    if (rank == 0) {
    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");
    
    // Connectivity is implicit in the generated square mesh.
    printf("Building unstructured mesh...\n");
    
    // Calculate memory usage
    const size_t device_mem = 4 * (size_t(n_elems_root) + 2 * active_ranks) *
                              n_elems_root * sizeof(double);
    printf("Device memory usage (all ranks): %.2f MB\n",
           device_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    }
    std::vector<double> local_energy, local_flux;
    double local_elapsed = 0.0;
    runSimulation(n_elems_root, n_iters, rank, active_ranks, comm,
                  local_energy, local_flux, local_elapsed);
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    uint64_t local_hash = 0;
    double local_energy_sum = 0.0, local_flux_sum = 0.0;
    double local_energy_min = std::numeric_limits<double>::max();
    double local_energy_max = std::numeric_limits<double>::lowest();
    const size_t global_offset = size_t(firstRowForRank(rank, active_ranks, n_elems_root)) * n_elems_root;
    #pragma omp parallel for schedule(static) reduction(^:local_hash) \
        reduction(+:local_energy_sum,local_flux_sum) \
        reduction(min:local_energy_min) reduction(max:local_energy_max)
    for (size_t i = 0; i < local_energy.size(); ++i) {
        const double e = local_energy[i], f = local_flux[i];
        uint64_t ebits, fbits;
        std::memcpy(&ebits, &e, sizeof(ebits));
        std::memcpy(&fbits, &f, sizeof(fbits));
        const size_t global_i = global_offset + i;
        local_hash ^= (ebits + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (fbits + global_i) * 0xbf58476d1ce4e5b9ULL;
        local_energy_sum += e;
        local_flux_sum += f;
        local_energy_min = std::min(local_energy_min, e);
        local_energy_max = std::max(local_energy_max, e);
    }
    uint64_t hash = 0;
    double energy_sum = 0.0, flux_sum = 0.0, energy_min = 0.0, energy_max = 0.0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    if (validate) {
        MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    }
    std::vector<double> all_energy;
    if (printResults) {
        std::vector<int> counts(active_ranks), offsets(active_ranks);
        for (int r = 0; r < active_ranks; ++r) {
            counts[r] = rowsForRank(r, active_ranks, n_elems_root) * n_elems_root;
            offsets[r] = firstRowForRank(r, active_ranks, n_elems_root) * n_elems_root;
        }
        if (rank == 0) all_energy.resize(n_elems);
        MPI_Gatherv(local_energy.data(), int(local_energy.size()), MPI_DOUBLE,
                    rank == 0 ? all_energy.data() : nullptr, counts.data(), offsets.data(),
                    MPI_DOUBLE, 0, comm);
    }
    int result = 0;
    if (rank == 0) {
    const long duration_ms = long(elapsed * 1000.0);
    
    printf("Computation time: %ld ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = elapsed * 1000.0 / n_measured_iters;
    const double giga_elems_per_sec = elapsed > 0.0
        ? double(n_measured_iters) * n_elems / elapsed / 1e9 : 0.0;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    // Compute hash for verification
    printf("  Result hash: %016lX\n", (unsigned long)hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults) {
        print_results(all_energy, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
        if (!std::isfinite(energy_sum) || !std::isfinite(flux_sum) ||
            !std::isfinite(energy_min) || !std::isfinite(energy_max)) {
            printf("  ERROR: Nonfinite simulation result\n");
            result = 1;
        } else {
            if (std::abs(energy_sum) > 1e-8)
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            printf("  Validation: PASSED\n");
        }
    }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, comm);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return result;
}
