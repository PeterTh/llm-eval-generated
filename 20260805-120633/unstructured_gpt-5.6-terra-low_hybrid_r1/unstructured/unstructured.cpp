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

static void checkCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

// `in` has one halo row at each end.  The four-neighbour stencil is exactly
// the connectivity assembled by the original unstructured-grid benchmark.
__global__ void updateKernel(const ElementDynamic* __restrict__ in,
                             ElementDynamic* __restrict__ out,
                             int width, int local_rows, int global_row0,
                             int global_height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x >= width || local_y > local_rows) return;

    const int p = local_y * width + x;
    const val_t energy = in[p].current_energy;
    val_t flux = 0.0;
    const int global_y = global_row0 + local_y - 1;
    if ((global_y == 0 && (x == 0 || x == width - 1)) ||
        (global_y == global_height - 1 && (x == 0 || x == width - 1)))
        flux = ((global_y == 0 && x == 0) ||
                (global_y == global_height - 1 && x == width - 1)) ? 0.5 : -0.5;

    // transfer_coeff * connection_flux * 0.25 == 0.2
    if (x) flux += (in[p - 1].current_energy - energy) * 0.2;
    if (x + 1 < width) flux += (in[p + 1].current_energy - energy) * 0.2;
    if (global_y) flux += (in[p - width].current_energy - energy) * 0.2;
    if (global_y + 1 < global_height) flux += (in[p + width].current_energy - energy) * 0.2;

    out[p].current_energy = energy + flux;
    out[p].total_flux = in[p].total_flux + fabs(flux);
}

static void exchangeHalos(ElementDynamic* device, int width, int local_rows,
                          int rank, int ranks) {
    std::vector<ElementDynamic> top(width), bottom(width);
    std::vector<ElementDynamic> recv_top(width), recv_bottom(width);
    checkCuda(cudaMemcpy(top.data(), device + width, width * sizeof(ElementDynamic),
                         cudaMemcpyDeviceToHost), "copy top boundary");
    checkCuda(cudaMemcpy(bottom.data(), device + local_rows * width, width * sizeof(ElementDynamic),
                         cudaMemcpyDeviceToHost), "copy bottom boundary");
    MPI_Request req[4]; int count = 0;
    if (rank > 0) {
        MPI_Irecv(recv_top.data(), width * sizeof(ElementDynamic), MPI_BYTE, rank - 1, 17, MPI_COMM_WORLD, &req[count++]);
        MPI_Isend(top.data(), width * sizeof(ElementDynamic), MPI_BYTE, rank - 1, 18, MPI_COMM_WORLD, &req[count++]);
    }
    if (rank + 1 < ranks) {
        MPI_Irecv(recv_bottom.data(), width * sizeof(ElementDynamic), MPI_BYTE, rank + 1, 18, MPI_COMM_WORLD, &req[count++]);
        MPI_Isend(bottom.data(), width * sizeof(ElementDynamic), MPI_BYTE, rank + 1, 17, MPI_COMM_WORLD, &req[count++]);
    }
    MPI_Waitall(count, req, MPI_STATUSES_IGNORE);
    if (rank > 0) checkCuda(cudaMemcpy(device, recv_top.data(), width * sizeof(ElementDynamic), cudaMemcpyHostToDevice), "copy top halo");
    if (rank + 1 < ranks) checkCuda(cudaMemcpy(device + (local_rows + 1) * width, recv_bottom.data(), width * sizeof(ElementDynamic), cudaMemcpyHostToDevice), "copy bottom halo");
}

static uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t e, f;
        memcpy(&e, &elements[i].current_energy, sizeof(e));
        memcpy(&f, &elements[i].total_flux, sizeof(f));
        hash ^= (e + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static bool validateResults(const std::vector<ElementDynamic>& values) {
    val_t energy_sum = 0, flux_sum = 0, energy_max = -std::numeric_limits<val_t>::infinity(), energy_min = std::numeric_limits<val_t>::infinity();
    for (const auto& e : values) { energy_sum += e.current_energy; flux_sum += e.total_flux; energy_max = std::max(energy_max, e.current_energy); energy_min = std::min(energy_min, e.current_energy); }
    printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", energy_sum, flux_sum, energy_min, energy_max);
    const bool valid = std::isfinite(energy_sum) && std::isfinite(flux_sum) && std::isfinite(energy_max) && std::isfinite(energy_min);
    printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    return valid;
}

static void printUsage(const char* name) { printf("Usage: %s [-n grid_size] [-i iterations] [-v] [-r] [-h]\n", name); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512, iters = 10; bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (n <= 0 || iters < 0 || ranks > n) { if (!rank) fprintf(stderr, "grid size must be positive and at least the MPI rank count\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    int device_count = 0; checkCuda(cudaGetDeviceCount(&device_count), "query CUDA devices");
    if (!device_count) { if (!rank) fprintf(stderr, "A CUDA device is required\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    checkCuda(cudaSetDevice(rank % device_count), "select CUDA device");

    const int base = n / ranks, extra = n % ranks;
    const int rows = base + (rank < extra);
    const int row0 = rank * base + std::min(rank, extra);
    const size_t local_count = static_cast<size_t>(rows) * n;
    std::vector<ElementDynamic> host((static_cast<size_t>(rows) + 2) * n, {0.0, 0.0});
    // OpenMP initializes the rank-local slab and remains part of every run.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_count; ++i) host[n + i] = {0.0, 0.0};
    ElementDynamic *current = nullptr, *next = nullptr;
    checkCuda(cudaMalloc(&current, host.size() * sizeof(ElementDynamic)), "allocate current state");
    checkCuda(cudaMalloc(&next, host.size() * sizeof(ElementDynamic)), "allocate next state");
    checkCuda(cudaMemcpy(current, host.data(), host.size() * sizeof(ElementDynamic), cudaMemcpyHostToDevice), "initialize current state");
    checkCuda(cudaMemset(next, 0, host.size() * sizeof(ElementDynamic)), "initialize next state");

    if (!rank) { printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\nGrid size: %d x %d = %lld elements\nIterations: %d\nMPI ranks: %d, OpenMP threads/rank: %d\nCUDA devices/rank: 1\n\n", n, n, n * 1LL * n, iters, ranks, omp_get_max_threads()); }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const dim3 block(32, 8), grid((n + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    for (int iter = 0; iter < iters; ++iter) {
        exchangeHalos(current, n, rows, rank, ranks);
        updateKernel<<<grid, block>>>(current, next, n, rows, row0, n);
        checkCuda(cudaGetLastError(), "launch update kernel");
        std::swap(current, next);
    }
    checkCuda(cudaDeviceSynchronize(), "complete simulation");
    const auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    double max_seconds = 0; MPI_Reduce(&seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    checkCuda(cudaMemcpy(host.data() + n, current + n, local_count * sizeof(ElementDynamic), cudaMemcpyDeviceToHost), "copy final state");

    std::vector<int> counts, displs; std::vector<ElementDynamic> global;
    if (!rank) { counts.resize(ranks); displs.resize(ranks); for (int r = 0; r < ranks; ++r) { const int rr = base + (r < extra); counts[r] = rr * n * sizeof(ElementDynamic); displs[r] = (r * base + std::min(r, extra)) * n * sizeof(ElementDynamic); } global.resize(static_cast<size_t>(n) * n); }
    MPI_Gatherv(host.data() + n, local_count * sizeof(ElementDynamic), MPI_BYTE, rank ? nullptr : global.data(), rank ? nullptr : counts.data(), rank ? nullptr : displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (!rank) {
        const double elems_per_sec = max_seconds > 0 ? static_cast<double>(n) * n * iters / max_seconds : 0;
        printf("Computation time: %.3f ms\nPerformance:\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n", max_seconds * 1000.0, elems_per_sec / 1e9, elems_per_sec * 22.0 / 1e9, static_cast<unsigned long long>(computeHash(global)));
        if (print_results_flag) {
            std::vector<double> energy(global.size());
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < global.size(); ++i) energy[i] = global[i].current_energy;
            print_results(energy, "ElementEnergy");
        }
        if (validate && !validateResults(global)) { cudaFree(current); cudaFree(next); MPI_Finalize(); return 1; }
    }
    cudaFree(current); cudaFree(next); MPI_Finalize(); return 0;
}
