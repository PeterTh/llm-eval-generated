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

#include "../common/results_output.hpp"

using val_t = double;

// The generated mesh is an unstructured representation of this regular grid.  A
// one-dimensional row decomposition keeps the only non-local edges in two halos.
struct DeviceState {
    val_t* energy;
    val_t* flux;
    val_t* next_energy;
    val_t* next_flux;
    val_t* top_halo;
    val_t* bottom_halo;
};

static void checkCuda(cudaError_t status, const char* operation, MPI_Comm comm) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        std::fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(comm, 1);
    }
}

__global__ void updateInterior(const val_t* energy, const val_t* flux,
                               val_t* next_energy, val_t* next_flux,
                               int width, int local_rows, int first_global_row,
                               int first_row, int last_row) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row < first_row || row >= last_row) return;

    const int pos = row * width + col;
    const int global_row = first_global_row + row;
    const val_t current = energy[pos];
    val_t total = 0.0;
    // Preserve the original connection order: down, up, right, left.
    if (global_row + 1 < width) total += (energy[pos + width] - current) * 0.2;
    if (global_row > 0) total += (energy[pos - width] - current) * 0.2;
    if (col + 1 < width) total += (energy[pos + 1] - current) * 0.2;
    if (col > 0) total += (energy[pos - 1] - current) * 0.2;

    if ((global_row == 0 && col == 0) ||
        (global_row == width - 1 && col == width - 1)) total += 0.5;
    if ((global_row == 0 && col == width - 1) ||
        (global_row == width - 1 && col == 0)) total -= 0.5;
    next_energy[pos] = current + total;
    next_flux[pos] = flux[pos] + fabs(total);
}

__global__ void updateBoundary(const val_t* energy, const val_t* flux,
                               val_t* next_energy, val_t* next_flux,
                               const val_t* top_halo, const val_t* bottom_halo,
                               int width, int local_rows, int first_global_row) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int which = blockIdx.y;
    if (col >= width || local_rows == 0) return;
    const int row = which == 0 ? 0 : local_rows - 1;
    if (which == 1 && local_rows == 1) return;
    const int pos = row * width + col;
    const int global_row = first_global_row + row;
    const val_t current = energy[pos];
    val_t total = 0.0;
    if (global_row + 1 < width) {
        const val_t down = row + 1 < local_rows ? energy[pos + width] : bottom_halo[col];
        total += (down - current) * 0.2;
    }
    if (global_row > 0) {
        const val_t up = row > 0 ? energy[pos - width] : top_halo[col];
        total += (up - current) * 0.2;
    }
    if (col + 1 < width) total += (energy[pos + 1] - current) * 0.2;
    if (col > 0) total += (energy[pos - 1] - current) * 0.2;
    if ((global_row == 0 && col == 0) ||
        (global_row == width - 1 && col == width - 1)) total += 0.5;
    if ((global_row == 0 && col == width - 1) ||
        (global_row == width - 1 && col == 0)) total -= 0.5;
    next_energy[pos] = current + total;
    next_flux[pos] = flux[pos] + fabs(total);
}

static uint64_t computeHash(const std::vector<val_t>& energy, const std::vector<val_t>& flux,
                            size_t global_offset) {
    uint64_t hash = 0;
    #pragma omp parallel for reduction(^:hash) schedule(static)
    for (size_t i = 0; i < energy.size(); ++i) {
        uint64_t e_bits, f_bits;
        std::memcpy(&e_bits, &energy[i], sizeof(e_bits));
        std::memcpy(&f_bits, &flux[i], sizeof(f_bits));
        const uint64_t global_i = global_offset + i;
        hash ^= (e_bits + global_i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -n <num>     Grid size (NxN elements) (default: 512)\n"
                "  -i <num>     Number of simulation iterations (default: 10)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512, iterations = 10;
    bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n <= 0 || iterations < 0 || ranks > n) {
        if (!rank) std::fprintf(stderr, "Grid size must be positive, iterations non-negative, and MPI ranks no more than grid rows.\n");
        MPI_Finalize(); return 1;
    }
    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", MPI_COMM_WORLD);
    if (!device_count) { if (!rank) std::fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    checkCuda(cudaSetDevice(rank % device_count), "cudaSetDevice", MPI_COMM_WORLD);

    const int base_rows = n / ranks, remainder = n % ranks;
    const int local_rows = base_rows + (rank < remainder ? 1 : 0);
    const int first_row = rank * base_rows + std::min(rank, remainder);
    const size_t local_elements = static_cast<size_t>(local_rows) * n;
    std::vector<val_t> host_energy(local_elements), host_flux(local_elements);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_elements; ++i) { host_energy[i] = 0.0; host_flux[i] = 0.0; }

    DeviceState d{};
    const size_t bytes = local_elements * sizeof(val_t), row_bytes = static_cast<size_t>(n) * sizeof(val_t);
    checkCuda(cudaMalloc(&d.energy, bytes), "allocate energy", MPI_COMM_WORLD);
    checkCuda(cudaMalloc(&d.flux, bytes), "allocate flux", MPI_COMM_WORLD);
    checkCuda(cudaMalloc(&d.next_energy, bytes), "allocate next energy", MPI_COMM_WORLD);
    checkCuda(cudaMalloc(&d.next_flux, bytes), "allocate next flux", MPI_COMM_WORLD);
    checkCuda(cudaMalloc(&d.top_halo, row_bytes), "allocate top halo", MPI_COMM_WORLD);
    checkCuda(cudaMalloc(&d.bottom_halo, row_bytes), "allocate bottom halo", MPI_COMM_WORLD);
    checkCuda(cudaMemcpy(d.energy, host_energy.data(), bytes, cudaMemcpyHostToDevice), "copy energy", MPI_COMM_WORLD);
    checkCuda(cudaMemcpy(d.flux, host_flux.data(), bytes, cudaMemcpyHostToDevice), "copy flux", MPI_COMM_WORLD);
    std::vector<val_t> send_top(n), send_bottom(n), recv_top(n), recv_bottom(n);
    const dim3 block(32, 8), grid((n + block.x - 1) / block.x, (local_rows + block.y - 1) / block.y);
    const dim3 boundary_grid((n + block.x - 1) / block.x, 2);

    if (!rank) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\nIterations: %d\nMPI ranks: %d, CUDA devices/node: %d\nValidation: %s\n\n",
                    n, n, static_cast<size_t>(n) * n, iterations, ranks, device_count, validate ? "enabled" : "disabled");
        std::printf("Running distributed CUDA simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        checkCuda(cudaMemcpyAsync(send_top.data(), d.energy, row_bytes, cudaMemcpyDeviceToHost), "copy top halo", MPI_COMM_WORLD);
        checkCuda(cudaMemcpyAsync(send_bottom.data(), d.energy + (local_rows - 1) * n, row_bytes, cudaMemcpyDeviceToHost), "copy bottom halo", MPI_COMM_WORLD);
        if (local_rows > 2) updateInterior<<<grid, block>>>(d.energy, d.flux, d.next_energy, d.next_flux, n, local_rows, first_row, 1, local_rows - 1);
        checkCuda(cudaStreamSynchronize(0), "synchronize halo copies", MPI_COMM_WORLD);
        MPI_Request requests[4]; int count = 0;
        if (rank > 0) { MPI_Irecv(recv_top.data(), n, MPI_DOUBLE, rank - 1, 17, MPI_COMM_WORLD, &requests[count++]); MPI_Isend(send_top.data(), n, MPI_DOUBLE, rank - 1, 18, MPI_COMM_WORLD, &requests[count++]); }
        if (rank + 1 < ranks) { MPI_Irecv(recv_bottom.data(), n, MPI_DOUBLE, rank + 1, 18, MPI_COMM_WORLD, &requests[count++]); MPI_Isend(send_bottom.data(), n, MPI_DOUBLE, rank + 1, 17, MPI_COMM_WORLD, &requests[count++]); }
        if (count) MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        if (rank > 0) checkCuda(cudaMemcpyAsync(d.top_halo, recv_top.data(), row_bytes, cudaMemcpyHostToDevice), "copy top receive halo", MPI_COMM_WORLD);
        if (rank + 1 < ranks) checkCuda(cudaMemcpyAsync(d.bottom_halo, recv_bottom.data(), row_bytes, cudaMemcpyHostToDevice), "copy bottom receive halo", MPI_COMM_WORLD);
        updateBoundary<<<boundary_grid, block>>>(d.energy, d.flux, d.next_energy, d.next_flux, d.top_halo, d.bottom_halo, n, local_rows, first_row);
        checkCuda(cudaGetLastError(), "launch update kernels", MPI_COMM_WORLD);
        std::swap(d.energy, d.next_energy); std::swap(d.flux, d.next_flux);
    }
    checkCuda(cudaDeviceSynchronize(), "finish simulation", MPI_COMM_WORLD);
    const double local_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double seconds = 0.0; MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    checkCuda(cudaMemcpy(host_energy.data(), d.energy, bytes, cudaMemcpyDeviceToHost), "retrieve energy", MPI_COMM_WORLD);
    checkCuda(cudaMemcpy(host_flux.data(), d.flux, bytes, cudaMemcpyDeviceToHost), "retrieve flux", MPI_COMM_WORLD);

    val_t local_energy = 0, local_flux = 0, local_min = std::numeric_limits<val_t>::max(), local_max = std::numeric_limits<val_t>::lowest();
    #pragma omp parallel for reduction(+:local_energy,local_flux) reduction(min:local_min) reduction(max:local_max) schedule(static)
    for (size_t i = 0; i < local_elements; ++i) { local_energy += host_energy[i]; local_flux += host_flux[i]; local_min = std::min(local_min, host_energy[i]); local_max = std::max(local_max, host_energy[i]); }
    val_t energy_sum, flux_sum, energy_min, energy_max;
    MPI_Reduce(&local_energy, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    uint64_t local_hash = computeHash(host_energy, host_flux, static_cast<size_t>(first_row) * n), hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    std::vector<val_t> all_energy;
    std::vector<int> counts, offsets;
    if (!rank && print_results_flag) { all_energy.resize(static_cast<size_t>(n) * n); counts.resize(ranks); offsets.resize(ranks); for (int r = 0; r < ranks; ++r) { const int rows = base_rows + (r < remainder); counts[r] = rows * n; offsets[r] = (r * base_rows + std::min(r, remainder)) * n; } }
    if (print_results_flag) MPI_Gatherv(host_energy.data(), static_cast<int>(local_elements), MPI_DOUBLE, all_energy.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (!rank) {
        const double safe_seconds = std::max(seconds, 1.0e-12);
        const double geps = static_cast<double>(n) * n * iterations / safe_seconds / 1.e9;
        std::printf("Computation time: %.3f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016lX\n\n", seconds * 1000., seconds * 1000. / std::max(iterations, 1), geps, geps * 22., static_cast<unsigned long>(hash));
        if (print_results_flag) print_results(all_energy, "ElementEnergy");
        if (validate) { std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n  Validation: %s\n", energy_sum, flux_sum, energy_min, energy_max, (std::isfinite(energy_sum) && std::isfinite(flux_sum) && std::isfinite(energy_min) && std::isfinite(energy_max)) ? "PASSED" : "FAILED"); }
    }
    cudaFree(d.energy); cudaFree(d.flux); cudaFree(d.next_energy); cudaFree(d.next_flux); cudaFree(d.top_halo); cudaFree(d.bottom_halo);
    MPI_Finalize();
    return 0;
}
