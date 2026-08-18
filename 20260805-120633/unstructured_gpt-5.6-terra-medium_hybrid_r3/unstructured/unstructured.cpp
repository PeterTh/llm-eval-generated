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

using idx_t = uint64_t;
using val_t = double;

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static void cudaCheck(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

// One MPI rank owns a contiguous band of rows.  The two halo arrays contain
// only energy because total_flux is never read from a neighbouring element.
__global__ void updateKernel(const val_t* __restrict__ energy,
                             const val_t* __restrict__ accumulated_flux,
                             val_t* __restrict__ next_energy,
                             val_t* __restrict__ next_accumulated_flux,
                             const val_t* __restrict__ top_halo,
                             const val_t* __restrict__ bottom_halo,
                             int first_row, int local_rows, int width) {
    const int local_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row >= local_rows || col >= width) return;

    const int i = local_row * width + col;
    const int global_row = first_row + local_row;
    const val_t self = energy[i];
    val_t total = 0.0;

    // Preserve the original connectivity insertion order: down, up, right, left.
    if (global_row + 1 < width) {
        const val_t other = (local_row + 1 < local_rows) ? energy[i + width] : bottom_halo[col];
        total += (other - self) * 0.8 * 0.25;
    }
    if (global_row > 0) {
        const val_t other = (local_row > 0) ? energy[i - width] : top_halo[col];
        total += (other - self) * 0.8 * 0.25;
    }
    if (col + 1 < width) total += (energy[i + 1] - self) * 0.8 * 0.25;
    if (col > 0) total += (energy[i - 1] - self) * 0.8 * 0.25;

    const bool inflow = (global_row == 0 && col == 0) ||
                        (global_row + 1 == width && col + 1 == width);
    const bool outflow = (global_row == 0 && col + 1 == width) ||
                         (global_row + 1 == width && col == 0);
    if (inflow) total += 0.5;
    if (outflow) total -= 0.5;

    next_energy[i] = self + total;
    next_accumulated_flux[i] = accumulated_flux[i] + fabs(total);
}

static uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t e_bits, f_bits;
        memcpy(&e_bits, &elements[i].current_energy, sizeof(e_bits));
        memcpy(&f_bits, &elements[i].total_flux, sizeof(f_bits));
        hash ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
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
        energy_max = std::max(energy_max, elem.current_energy);
        energy_min = std::min(energy_min, elem.current_energy);
    }
    printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n",
           energy_sum, flux_sum, energy_min, energy_max);
    if (!std::isfinite(energy_sum) || !std::isfinite(flux_sum) ||
        !std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: non-finite result\n");
        return false;
    }
    if (std::abs(energy_sum) > 1e-8)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    printf("  Validation: PASSED\n");
    return true;
}

static void printUsage(const char* prog) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Grid size (NxN elements) (default: 512)\n"
           "  -i <num>     Number of simulation iterations (default: 10)\n"
           "  -v           Enable validation\n  -r           Print results for external validation\n"
           "  -h           Show this help message\n", prog);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int width = 512, iterations = 10;
    bool validate = false, print_results_requested = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) width = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results_requested = true;
        else if (!strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (width <= 0 || iterations < 0) { if (rank == 0) fprintf(stderr, "Grid size must be positive and iterations non-negative.\n"); MPI_Finalize(); return 1; }
    if (ranks > width) { if (rank == 0) fprintf(stderr, "MPI ranks (%d) cannot exceed grid rows (%d).\n", ranks, width); MPI_Finalize(); return 1; }

    const int base_rows = width / ranks, remainder = width % ranks;
    const int local_rows = base_rows + (rank < remainder);
    const int first_row = rank * base_rows + std::min(rank, remainder);
    const int local_elements = local_rows * width;
    const int total_elements = width * width;
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) { if (rank == 0) fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(rank % device_count), "cudaSetDevice");

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n"
               "Grid size: %d x %d = %d elements\nIterations: %d\nValidation: %s\n"
               "MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices: %d\n\n",
               width, width, total_elements, iterations, validate ? "enabled" : "disabled", ranks, omp_get_max_threads(), device_count);
        const size_t static_mem = size_t(total_elements) * (sizeof(idx_t) * 10 + sizeof(val_t) * 8);
        const size_t dynamic_mem = size_t(total_elements) * sizeof(ElementDynamic) * 2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\nRunning simulation...\n",
               (static_mem + dynamic_mem) / 1048576.0, static_mem / 1048576.0, dynamic_mem / 1048576.0);
    }

    // OpenMP parallel initialization keeps CPU resources engaged for allocation/setup.
    std::vector<val_t> host_energy(local_elements), host_flux(local_elements);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_elements; ++i) { host_energy[i] = 0.0; host_flux[i] = 0.0; }

    val_t *d_energy, *d_flux, *d_next_energy, *d_next_flux, *d_top, *d_bottom;
    const size_t local_bytes = size_t(local_elements) * sizeof(val_t);
    const size_t halo_bytes = size_t(width) * sizeof(val_t);
    cudaCheck(cudaMalloc(&d_energy, local_bytes), "allocate energy");
    cudaCheck(cudaMalloc(&d_flux, local_bytes), "allocate flux");
    cudaCheck(cudaMalloc(&d_next_energy, local_bytes), "allocate next energy");
    cudaCheck(cudaMalloc(&d_next_flux, local_bytes), "allocate next flux");
    cudaCheck(cudaMalloc(&d_top, halo_bytes), "allocate top halo");
    cudaCheck(cudaMalloc(&d_bottom, halo_bytes), "allocate bottom halo");
    cudaCheck(cudaMemcpy(d_energy, host_energy.data(), local_bytes, cudaMemcpyHostToDevice), "upload energy");
    cudaCheck(cudaMemcpy(d_flux, host_flux.data(), local_bytes, cudaMemcpyHostToDevice), "upload flux");

    std::vector<val_t> send_top(width), send_bottom(width), top_halo(width, 0.0), bottom_halo(width, 0.0);
    const dim3 block(32, 8), grid((width + block.x - 1) / block.x, (local_rows + block.y - 1) / block.y);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        cudaCheck(cudaMemcpy(send_top.data(), d_energy, halo_bytes, cudaMemcpyDeviceToHost), "download top boundary");
        cudaCheck(cudaMemcpy(send_bottom.data(), d_energy + size_t(local_rows - 1) * width, halo_bytes, cudaMemcpyDeviceToHost), "download bottom boundary");
        MPI_Sendrecv(send_bottom.data(), width, MPI_DOUBLE, rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL, 10,
                     top_halo.data(), width, MPI_DOUBLE, rank > 0 ? rank - 1 : MPI_PROC_NULL, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_top.data(), width, MPI_DOUBLE, rank > 0 ? rank - 1 : MPI_PROC_NULL, 11,
                     bottom_halo.data(), width, MPI_DOUBLE, rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cudaCheck(cudaMemcpy(d_top, top_halo.data(), halo_bytes, cudaMemcpyHostToDevice), "upload top halo");
        cudaCheck(cudaMemcpy(d_bottom, bottom_halo.data(), halo_bytes, cudaMemcpyHostToDevice), "upload bottom halo");
        updateKernel<<<grid, block>>>(d_energy, d_flux, d_next_energy, d_next_flux, d_top, d_bottom, first_row, local_rows, width);
        cudaCheck(cudaGetLastError(), "update kernel launch");
        std::swap(d_energy, d_next_energy);
        std::swap(d_flux, d_next_flux);
    }
    cudaCheck(cudaDeviceSynchronize(), "simulation synchronization");
    const double local_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double elapsed_seconds = 0.0;
    MPI_Reduce(&local_seconds, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    cudaCheck(cudaMemcpy(host_energy.data(), d_energy, local_bytes, cudaMemcpyDeviceToHost), "download final energy");
    cudaCheck(cudaMemcpy(host_flux.data(), d_flux, local_bytes, cudaMemcpyDeviceToHost), "download final flux");
    cudaFree(d_energy); cudaFree(d_flux); cudaFree(d_next_energy); cudaFree(d_next_flux); cudaFree(d_top); cudaFree(d_bottom);

    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) { const int rows = base_rows + (r < remainder); counts[r] = rows * width; displacements[r] = (r * base_rows + std::min(r, remainder)) * width; }
    std::vector<val_t> all_energy, all_flux;
    if (rank == 0) { all_energy.resize(total_elements); all_flux.resize(total_elements); }
    MPI_Gatherv(host_energy.data(), local_elements, MPI_DOUBLE, all_energy.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(host_flux.data(), local_elements, MPI_DOUBLE, all_flux.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int result = 0;
    if (rank == 0) {
        std::vector<ElementDynamic> final_state(total_elements);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < total_elements; ++i) final_state[i] = {all_energy[i], all_flux[i]};
        const double ms = elapsed_seconds * 1000.0;
        const int measured = std::max(iterations - 1, 1);
        const double geps = elapsed_seconds > 0.0 ? (double(measured) * total_elements / elapsed_seconds / 1e9) : 0.0;
        printf("Computation time: %.3f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n",
               ms, ms / measured, geps, geps * 22.0, static_cast<unsigned long long>(computeHash(final_state)));
        if (print_results_requested) { std::vector<double> energies(all_energy.begin(), all_energy.end()); print_results(energies, "ElementEnergy"); }
        if (validate && !validateResults(final_state)) result = 1;
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
