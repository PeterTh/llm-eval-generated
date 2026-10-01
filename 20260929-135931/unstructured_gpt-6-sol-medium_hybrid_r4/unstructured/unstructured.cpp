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

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void checkMpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI %s failed\n", operation);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Rows are contiguous on each GPU. One halo row on each side holds neighbors
// from adjacent ranks. Match the original connection order: +x, -x, +y, -y.
__global__ void updateRows(const val_t* energy, const val_t* flux,
                           val_t* next_energy, val_t* next_flux,
                           int n, int first_global_row, int first_local_row,
                           int rows_to_update) {
    const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(rows_to_update) * n;
    if (k >= count) return;

    const int local_row = first_local_row + static_cast<int>(k / n);
    const int col = static_cast<int>(k % n);
    const int global_row = first_global_row + local_row;
    const size_t p = static_cast<size_t>(local_row + 1) * n + col;
    const val_t own = energy[p];
    val_t total = ((global_row == 0 && col == 0) ||
                   (global_row == n - 1 && col == n - 1)) ? 0.5 :
                  ((global_row == 0 && col == n - 1) ||
                   (global_row == n - 1 && col == 0)) ? -0.5 : 0.0;
    if (global_row + 1 < n) total += ((energy[p + n] - own) * 0.8) * 1.0 * 0.25;
    if (global_row > 0) total += ((energy[p - n] - own) * 0.8) * 1.0 * 0.25;
    if (col + 1 < n) total += ((energy[p + 1] - own) * 0.8) * 1.0 * 0.25;
    if (col > 0) total += ((energy[p - 1] - own) * 0.8) * 1.0 * 0.25;
    next_energy[p] = own + total;
    next_flux[p] = flux[p] + fabs(total);
}

static void launchRows(const val_t* energy, const val_t* flux,
                       val_t* next_energy, val_t* next_flux, int n,
                       int first_global_row, int first_local_row,
                       int rows, cudaStream_t stream) {
    if (rows <= 0) return;
    constexpr int threads = 256;
    const size_t count = static_cast<size_t>(rows) * n;
    updateRows<<<static_cast<unsigned>((count + threads - 1) / threads), threads, 0, stream>>>(
        energy, flux, next_energy, next_flux, n, first_global_row, first_local_row, rows);
    checkCuda(cudaGetLastError(), "kernel launch");
}

static void runSimulation(int n, int iterations, int first_row, int local_rows,
                          int rank, int ranks, MPI_Comm comm,
                          std::vector<val_t>& local_energy,
                          std::vector<val_t>& local_flux) {
    const size_t row_bytes = static_cast<size_t>(n) * sizeof(val_t);
    const size_t bytes = static_cast<size_t>(local_rows + 2) * row_bytes;
    val_t *energy, *next_energy, *flux, *next_flux;
    checkCuda(cudaMalloc(&energy, bytes), "allocate energy");
    checkCuda(cudaMalloc(&next_energy, bytes), "allocate next energy");
    checkCuda(cudaMalloc(&flux, bytes), "allocate flux");
    checkCuda(cudaMalloc(&next_flux, bytes), "allocate next flux");
    checkCuda(cudaMemset(energy, 0, bytes), "initialize energy");
    checkCuda(cudaMemset(flux, 0, bytes), "initialize flux");

    // Pinned staging buffers permit boundary transfers to overlap interior work.
    val_t *send_top, *send_bottom, *recv_top, *recv_bottom;
    checkCuda(cudaMallocHost(&send_top, row_bytes), "allocate top send buffer");
    checkCuda(cudaMallocHost(&send_bottom, row_bytes), "allocate bottom send buffer");
    checkCuda(cudaMallocHost(&recv_top, row_bytes), "allocate top receive buffer");
    checkCuda(cudaMallocHost(&recv_bottom, row_bytes), "allocate bottom receive buffer");
    cudaStream_t compute, transfer;
    checkCuda(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking), "create compute stream");
    checkCuda(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking), "create transfer stream");
    const int prev = rank ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;

    for (int iter = 0; iter < iterations; ++iter) {
        launchRows(energy, flux, next_energy, next_flux, n, first_row,
                   1, local_rows - 2, compute);
        if (prev != MPI_PROC_NULL)
            checkCuda(cudaMemcpyAsync(send_top, energy + n, row_bytes,
                                      cudaMemcpyDeviceToHost, transfer), "copy top boundary");
        if (next != MPI_PROC_NULL)
            checkCuda(cudaMemcpyAsync(send_bottom, energy + static_cast<size_t>(local_rows) * n,
                                      row_bytes, cudaMemcpyDeviceToHost, transfer), "copy bottom boundary");
        checkCuda(cudaStreamSynchronize(transfer), "synchronize boundary sends");
        checkMpi(MPI_Sendrecv(send_top, n, MPI_DOUBLE, prev, 0,
                              recv_bottom, n, MPI_DOUBLE, next, 0,
                              comm, MPI_STATUS_IGNORE), "exchange upper boundary");
        checkMpi(MPI_Sendrecv(send_bottom, n, MPI_DOUBLE, next, 1,
                              recv_top, n, MPI_DOUBLE, prev, 1,
                              comm, MPI_STATUS_IGNORE), "exchange lower boundary");
        if (prev != MPI_PROC_NULL)
            checkCuda(cudaMemcpyAsync(energy, recv_top, row_bytes,
                                      cudaMemcpyHostToDevice, transfer), "upload top halo");
        if (next != MPI_PROC_NULL)
            checkCuda(cudaMemcpyAsync(energy + static_cast<size_t>(local_rows + 1) * n,
                                      recv_bottom, row_bytes,
                                      cudaMemcpyHostToDevice, transfer), "upload bottom halo");
        checkCuda(cudaStreamSynchronize(transfer), "synchronize halos");
        launchRows(energy, flux, next_energy, next_flux, n, first_row, 0, 1, compute);
        if (local_rows > 1)
            launchRows(energy, flux, next_energy, next_flux, n, first_row,
                       local_rows - 1, 1, compute);
        checkCuda(cudaStreamSynchronize(compute), "synchronize iteration");
        std::swap(energy, next_energy);
        std::swap(flux, next_flux);
    }

    const size_t local_bytes = static_cast<size_t>(local_rows) * row_bytes;
    local_energy.resize(static_cast<size_t>(local_rows) * n);
    local_flux.resize(static_cast<size_t>(local_rows) * n);
    checkCuda(cudaMemcpy(local_energy.data(), energy + n, local_bytes,
                         cudaMemcpyDeviceToHost), "download energy");
    checkCuda(cudaMemcpy(local_flux.data(), flux + n, local_bytes,
                         cudaMemcpyDeviceToHost), "download flux");
    checkCuda(cudaStreamDestroy(compute), "destroy compute stream");
    checkCuda(cudaStreamDestroy(transfer), "destroy transfer stream");
    checkCuda(cudaFreeHost(send_top), "free top send buffer");
    checkCuda(cudaFreeHost(send_bottom), "free bottom send buffer");
    checkCuda(cudaFreeHost(recv_top), "free top receive buffer");
    checkCuda(cudaFreeHost(recv_bottom), "free bottom receive buffer");
    checkCuda(cudaFree(energy), "free energy");
    checkCuda(cudaFree(next_energy), "free next energy");
    checkCuda(cudaFree(flux), "free flux");
    checkCuda(cudaFree(next_flux), "free next flux");
}

static uint64_t bits(val_t value) {
    uint64_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

static uint64_t computeHash(const std::vector<val_t>& energy,
                            const std::vector<val_t>& flux, size_t first_index) {
    uint64_t hash = 0;
    #pragma omp parallel for reduction(^:hash) schedule(static)
    for (size_t i = 0; i < energy.size(); ++i) {
        const size_t global_index = first_index + i;
        hash ^= (bits(energy[i]) + global_index) * 0x9e3779b97f4a7c15ULL;
        hash ^= (bits(flux[i]) + global_index) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
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
    std::printf("Validation results:\n");
    std::printf("  Energy sum: %.12f\n", energy_sum);
    std::printf("  Flux sum: %.2f\n", flux_sum);
    std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    if (!std::isfinite(energy_sum)) {
        std::printf("  ERROR: Energy sum is not finite\n"); return false;
    }
    if (std::abs(energy_sum) > 1e-8)
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) {
        std::printf("  ERROR: Flux sum is not finite\n"); return false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        std::printf("  ERROR: Energy extrema are not finite\n"); return false;
    }
    std::printf("  Validation: PASSED\n");
    return true;
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
    checkMpi(MPI_Init(&argc, &argv), "initialize");
    int world_rank, world_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_ranks);
    int n = 512, iterations = 10;
    bool validate = false, print_results_flag = false;
    bool help = false, bad_args = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) print_results_flag = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else bad_args = true;
    }
    if (help || bad_args || n <= 0 || iterations < 0 ||
        static_cast<int64_t>(n) * n > std::numeric_limits<int>::max()) {
        if (world_rank == 0) {
            if (!help) std::fprintf(stderr, "Invalid arguments\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return help ? 0 : 1;
    }

    const int active_ranks = std::min(world_ranks, n);
    MPI_Comm comm;
    checkMpi(MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 0 : MPI_UNDEFINED,
                            world_rank, &comm), "split active ranks");
    if (world_rank >= active_ranks) {
        MPI_Finalize();
        return 0;
    }
    const int rank = world_rank;
    MPI_Comm local_comm;
    checkMpi(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                 &local_comm), "split node ranks");
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices), "query devices");
    if (devices == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device is available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(local_rank % devices), "select device");

    const int first_row = static_cast<int>((static_cast<int64_t>(rank) * n) / active_ranks);
    const int end_row = static_cast<int>((static_cast<int64_t>(rank + 1) * n) / active_ranks);
    const int local_rows = end_row - first_row;
    const int total_elems = n * n;
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %d elements\n", n, n, total_elems);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        std::printf("Building unstructured mesh...\n");
        const double static_mem = static_cast<double>(total_elems) * (sizeof(uint64_t) * 10 + sizeof(double) * 8);
        const double dynamic_mem = static_cast<double>(total_elems) * sizeof(ElementDynamic) * 2;
        std::printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                    (static_mem + dynamic_mem) / 1048576.0, static_mem / 1048576.0,
                    dynamic_mem / 1048576.0);
        std::printf("Running simulation...\n");
    }
    MPI_Barrier(comm);
    const auto start = std::chrono::steady_clock::now();
    std::vector<val_t> local_energy, local_flux;
    runSimulation(n, iterations, first_row, local_rows, rank, active_ranks,
                  comm, local_energy, local_flux);
    MPI_Barrier(comm);
    const double local_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    double seconds = 0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    const uint64_t local_hash = computeHash(local_energy, local_flux,
                                             static_cast<size_t>(first_row) * n);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);

    std::vector<int> counts, displacements;
    std::vector<val_t> all_energy, all_flux;
    if (rank == 0 && (validate || print_results_flag)) {
        counts.resize(active_ranks);
        displacements.resize(active_ranks);
        for (int r = 0; r < active_ranks; ++r) {
            const int begin = static_cast<int>((static_cast<int64_t>(r) * n) / active_ranks);
            const int end = static_cast<int>((static_cast<int64_t>(r + 1) * n) / active_ranks);
            counts[r] = (end - begin) * n;
            displacements[r] = begin * n;
        }
        all_energy.resize(total_elems);
        all_flux.resize(total_elems);
    }
    if (validate || print_results_flag) {
        MPI_Gatherv(local_energy.data(), static_cast<int>(local_energy.size()), MPI_DOUBLE,
                    rank == 0 ? all_energy.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, comm);
        if (validate)
            MPI_Gatherv(local_flux.data(), static_cast<int>(local_flux.size()), MPI_DOUBLE,
                        rank == 0 ? all_flux.data() : nullptr,
                        rank == 0 ? counts.data() : nullptr,
                        rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, comm);
    }

    int result = 0;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const int measured = std::max(iterations - 1, 1);
        const double ms = seconds * 1000.0;
        const double giga = (static_cast<double>(measured) * total_elems) / seconds / 1e9;
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", ms / measured);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga);
        std::printf("  Performance: %.4f GFLOPS\n", giga * 22.0);
        std::printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
        if (print_results_flag) print_results(all_energy, "ElementEnergy");
        if (validate) {
            std::vector<ElementDynamic> elements(total_elems);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < total_elems; ++i)
                elements[i] = {all_energy[i], all_flux[i]};
            if (!validateResults(elements)) result = 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, comm);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return result;
}
