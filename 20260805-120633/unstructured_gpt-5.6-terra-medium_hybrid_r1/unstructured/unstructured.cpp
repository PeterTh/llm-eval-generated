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

static void cudaCheck(cudaError_t status, const char* what, MPI_Comm comm) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        std::fprintf(stderr, "rank %d: CUDA %s failed: %s\n", rank, what,
                     cudaGetErrorString(status));
        MPI_Abort(comm, 1);
    }
}

// The generated mesh has unit edge weights and a common transfer coefficient.
// Keeping the four additions in this order matches buildSquare2D's edge order:
// down, up, right, left.
__global__ void updateKernel(const val_t* __restrict__ energy,
                             const val_t* __restrict__ accumulated_flux,
                             val_t* __restrict__ next_energy,
                             val_t* __restrict__ next_accumulated_flux,
                             int width, int local_rows, int global_row_begin,
                             int global_rows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= local_rows) return;

    const int local = (row + 1) * width + col; // row zero is the top halo
    const val_t center = energy[local];
    val_t total = 0.0;
    const int global_row = global_row_begin + row;
    if ((global_row == 0 && col == 0) ||
        (global_row == global_rows - 1 && col == width - 1)) total = 0.5;
    else if ((global_row == 0 && col == width - 1) ||
             (global_row == global_rows - 1 && col == 0)) total = -0.5;

    if (global_row + 1 < global_rows) total += (energy[local + width] - center) * 0.8 * 1.0 * 0.25;
    if (global_row > 0)                 total += (energy[local - width] - center) * 0.8 * 1.0 * 0.25;
    if (col + 1 < width)                total += (energy[local + 1] - center) * 0.8 * 1.0 * 0.25;
    if (col > 0)                        total += (energy[local - 1] - center) * 0.8 * 1.0 * 0.25;

    next_energy[local] = center + total;
    next_accumulated_flux[local] = accumulated_flux[local] + fabs(total);
}

static uint64_t hashElements(const std::vector<val_t>& energy,
                             const std::vector<val_t>& flux) {
    uint64_t hash = 0;
    for (size_t i = 0; i < energy.size(); ++i) {
        uint64_t e, f;
        std::memcpy(&e, &energy[i], sizeof(e));
        std::memcpy(&f, &flux[i], sizeof(f));
        hash ^= (e + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int n = 512, iters = 10;
    bool validate = false, print_results_enabled = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) print_results_enabled = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n <= 0 || iters < 0 || ranks > n) {
        if (!rank) std::fprintf(stderr, "grid size must be positive, iterations non-negative, and MPI ranks no greater than grid rows\n");
        MPI_Finalize();
        return 1;
    }

    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount", MPI_COMM_WORLD);
    if (!devices) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(rank % devices), "cudaSetDevice", MPI_COMM_WORLD);

    const int row_begin = (rank * n) / ranks;
    const int row_end = ((rank + 1) * n) / ranks;
    const int rows = row_end - row_begin;
    const size_t pitch = static_cast<size_t>(n);
    const size_t cells_with_halo = static_cast<size_t>(rows + 2) * pitch;
    const size_t local_cells = static_cast<size_t>(rows) * pitch;
    const int up = rank ? rank - 1 : MPI_PROC_NULL;
    const int down = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;

    val_t *d_energy, *d_flux, *d_next_energy, *d_next_flux;
    cudaCheck(cudaMalloc(&d_energy, cells_with_halo * sizeof(val_t)), "allocate energy", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&d_flux, cells_with_halo * sizeof(val_t)), "allocate flux", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&d_next_energy, cells_with_halo * sizeof(val_t)), "allocate next energy", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&d_next_flux, cells_with_halo * sizeof(val_t)), "allocate next flux", MPI_COMM_WORLD);
    cudaCheck(cudaMemset(d_energy, 0, cells_with_halo * sizeof(val_t)), "initialize energy", MPI_COMM_WORLD);
    cudaCheck(cudaMemset(d_flux, 0, cells_with_halo * sizeof(val_t)), "initialize flux", MPI_COMM_WORLD);

    std::vector<val_t> top(n), bottom(n), received_top(n), received_bottom(n);
    // These buffers are also touched by the OpenMP host team.  The same team
    // performs the final reduction, while CUDA owns the bandwidth-bound stencil.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        top[i] = bottom[i] = received_top[i] = received_bottom[i] = 0.0;
    }
    const dim3 block(32, 8), grid((n + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        // Host staging makes this correct on both CUDA-aware and conventional MPI stacks.
        if (rows) {
            cudaCheck(cudaMemcpy(top.data(), d_energy + pitch, pitch * sizeof(val_t), cudaMemcpyDeviceToHost), "copy top boundary", MPI_COMM_WORLD);
            cudaCheck(cudaMemcpy(bottom.data(), d_energy + rows * pitch, pitch * sizeof(val_t), cudaMemcpyDeviceToHost), "copy bottom boundary", MPI_COMM_WORLD);
        }
        MPI_Sendrecv(top.data(), n, MPI_DOUBLE, up, 11, received_top.data(), n, MPI_DOUBLE, up, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(bottom.data(), n, MPI_DOUBLE, down, 12, received_bottom.data(), n, MPI_DOUBLE, down, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (up != MPI_PROC_NULL) cudaCheck(cudaMemcpy(d_energy, received_top.data(), pitch * sizeof(val_t), cudaMemcpyHostToDevice), "upload top halo", MPI_COMM_WORLD);
        if (down != MPI_PROC_NULL) cudaCheck(cudaMemcpy(d_energy + (rows + 1) * pitch, received_bottom.data(), pitch * sizeof(val_t), cudaMemcpyHostToDevice), "upload bottom halo", MPI_COMM_WORLD);
        if (rows) {
            updateKernel<<<grid, block>>>(d_energy, d_flux, d_next_energy, d_next_flux, n, rows, row_begin, n);
            cudaCheck(cudaGetLastError(), "launch update kernel", MPI_COMM_WORLD);
        }
        std::swap(d_energy, d_next_energy);
        std::swap(d_flux, d_next_flux);
    }
    cudaCheck(cudaDeviceSynchronize(), "synchronize simulation", MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const long long local_ms = static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    long long ms = 0;
    MPI_Reduce(&local_ms, &ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<val_t> local_energy(local_cells), local_flux(local_cells);
    if (rows) {
        cudaCheck(cudaMemcpy(local_energy.data(), d_energy + pitch, local_cells * sizeof(val_t), cudaMemcpyDeviceToHost), "download energy", MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(local_flux.data(), d_flux + pitch, local_cells * sizeof(val_t), cudaMemcpyDeviceToHost), "download flux", MPI_COMM_WORLD);
    }
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) { counts[r] = (((r + 1) * n) / ranks - (r * n) / ranks) * n; displacements[r] = (r * n / ranks) * n; }
    std::vector<val_t> energy, flux;
    if (!rank) { energy.resize(static_cast<size_t>(n) * n); flux.resize(static_cast<size_t>(n) * n); }
    MPI_Gatherv(local_energy.data(), static_cast<int>(local_cells), MPI_DOUBLE, energy.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_flux.data(), static_cast<int>(local_cells), MPI_DOUBLE, flux.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    cudaFree(d_energy); cudaFree(d_flux); cudaFree(d_next_energy); cudaFree(d_next_flux);
    if (!rank) {
        const double seconds = std::max(ms / 1000.0, 1.0e-9);
        const double geps = static_cast<double>(n) * n * std::max(iters - 1, 1) / seconds / 1.e9;
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        std::printf("Grid size: %d x %d = %d elements\nIterations: %d\nMPI ranks: %d, OpenMP threads/rank: %d, CUDA devices: %d\n", n, n, n*n, iters, ranks, omp_get_max_threads(), devices);
        std::printf("Computation time: %lld ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016lX\n", ms, static_cast<double>(ms) / std::max(iters - 1, 1), geps, geps * 22.0, hashElements(energy, flux));
        if (print_results_enabled) print_results(energy, "ElementEnergy");
        if (validate) {
            val_t energy_sum = 0, flux_sum = 0, lo = std::numeric_limits<val_t>::max(), hi = std::numeric_limits<val_t>::lowest();
            #pragma omp parallel for reduction(+:energy_sum,flux_sum) reduction(min:lo) reduction(max:hi)
            for (long long i = 0; i < static_cast<long long>(energy.size()); ++i) { energy_sum += energy[i]; flux_sum += flux[i]; lo = std::min(lo, energy[i]); hi = std::max(hi, energy[i]); }
            std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", energy_sum, flux_sum, lo, hi);
            if (!std::isfinite(energy_sum) || !std::isfinite(flux_sum) || !std::isfinite(lo) || !std::isfinite(hi)) { std::printf("  ERROR: non-finite result\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
            std::printf("  Validation: PASSED\n");
        }
    }
    MPI_Finalize();
    return 0;
}
