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

struct ElementDynamic { val_t current_energy; val_t total_flux; };
struct World {
    int grid_size = 0, first_row = 0, local_rows = 0;
    std::vector<ElementDynamic> elements_dynamic;
};

static void cudaCheck(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA failure at %s: %s\n", where, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

// Each rank owns a contiguous slab.  The two extra rows are MPI halo cells.
__global__ void updateKernel(const ElementDynamic* current, ElementDynamic* next,
                             int width, int local_rows, int first_row) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= local_rows || col >= width) return;

    const int p = (row + 1) * width + col;
    const val_t energy = current[p].current_energy;
    val_t flux = 0.0;
    const int global_row = first_row + row;
    if ((global_row == 0 && col == 0) ||
        (global_row == width - 1 && col == width - 1)) flux = 0.5;
    else if ((global_row == 0 && col == width - 1) ||
             (global_row == width - 1 && col == 0)) flux = -0.5;

    // transfer_coeff * connection_flux * 0.25 == 0.2 for every material.
    if (global_row > 0)         flux += 0.2 * (current[p - width].current_energy - energy);
    if (global_row + 1 < width) flux += 0.2 * (current[p + width].current_energy - energy);
    if (col > 0)                flux += 0.2 * (current[p - 1].current_energy - energy);
    if (col + 1 < width)        flux += 0.2 * (current[p + 1].current_energy - energy);
    next[p].current_energy = energy + flux;
    next[p].total_flux = current[p].total_flux + fabs(flux);
}

static void buildSquare2D(World& world, int n, int first_row, int local_rows) {
    world.grid_size = n; world.first_row = first_row; world.local_rows = local_rows;
    world.elements_dynamic.resize(static_cast<size_t>(n) * local_rows);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i)
        world.elements_dynamic[i] = {0.0, 0.0};
}

static void runSimulation(World& world, int n_iters, int rank, int ranks) {
    const int n = world.grid_size, rows = world.local_rows;
    const size_t pitch = static_cast<size_t>(n) * sizeof(ElementDynamic);
    const size_t padded_count = static_cast<size_t>(rows + 2) * n;
    ElementDynamic *d_current = nullptr, *d_next = nullptr;
    cudaCheck(cudaMalloc(&d_current, padded_count * sizeof(ElementDynamic)), "cudaMalloc current");
    cudaCheck(cudaMalloc(&d_next, padded_count * sizeof(ElementDynamic)), "cudaMalloc next");
    cudaCheck(cudaMemset(d_current, 0, padded_count * sizeof(ElementDynamic)), "cudaMemset current");
    cudaCheck(cudaMemset(d_next, 0, padded_count * sizeof(ElementDynamic)), "cudaMemset next");

    std::vector<ElementDynamic> send_first(n), send_last(n), recv_top(n, {0.0, 0.0}), recv_bottom(n, {0.0, 0.0});
    const int up = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int down = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const dim3 block(32, 8), grid((n + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    for (int iter = 0; iter < n_iters; ++iter) {
        updateKernel<<<grid, block>>>(d_current, d_next, n, rows, world.first_row);
        cudaCheck(cudaGetLastError(), "update kernel launch");
        cudaCheck(cudaDeviceSynchronize(), "update kernel");
        std::swap(d_current, d_next);
        if (iter + 1 == n_iters) break;

        cudaCheck(cudaMemcpy(send_first.data(), d_current + n, pitch, cudaMemcpyDeviceToHost), "copy first halo row");
        cudaCheck(cudaMemcpy(send_last.data(), d_current + static_cast<size_t>(rows) * n, pitch, cudaMemcpyDeviceToHost), "copy last halo row");
        MPI_Sendrecv(send_first.data(), n * sizeof(ElementDynamic), MPI_BYTE, up, 0,
                     recv_bottom.data(), n * sizeof(ElementDynamic), MPI_BYTE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_last.data(), n * sizeof(ElementDynamic), MPI_BYTE, down, 1,
                     recv_top.data(), n * sizeof(ElementDynamic), MPI_BYTE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (up != MPI_PROC_NULL)
            cudaCheck(cudaMemcpy(d_current, recv_top.data(), pitch, cudaMemcpyHostToDevice), "upload top halo");
        if (down != MPI_PROC_NULL)
            cudaCheck(cudaMemcpy(d_current + static_cast<size_t>(rows + 1) * n, recv_bottom.data(), pitch, cudaMemcpyHostToDevice), "upload bottom halo");
    }
    cudaCheck(cudaMemcpy2D(world.elements_dynamic.data(), pitch, d_current + n, pitch,
                           pitch, rows, cudaMemcpyDeviceToHost), "copy final state");
    cudaFree(d_current); cudaFree(d_next);
}

static uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    #pragma omp parallel for reduction(^:hash) schedule(static)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const auto& e = world.elements_dynamic[i];
        const uint64_t global_i = static_cast<uint64_t>(world.first_row) * world.grid_size + i;
        uint64_t eb, fb; std::memcpy(&eb, &e.current_energy, sizeof eb); std::memcpy(&fb, &e.total_flux, sizeof fb);
        hash ^= (eb + global_i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (fb + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static bool validateResults(const World& world, int rank) {
    val_t local_energy = 0.0, local_flux = 0.0, local_min = std::numeric_limits<val_t>::max(), local_max = std::numeric_limits<val_t>::lowest();
    #pragma omp parallel for reduction(+:local_energy,local_flux) reduction(min:local_min) reduction(max:local_max)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const auto& e = world.elements_dynamic[i]; local_energy += e.current_energy; local_flux += e.total_flux;
        local_min = std::min(local_min, e.current_energy); local_max = std::max(local_max, e.current_energy);
    }
    val_t energy, flux, minimum, maximum;
    MPI_Reduce(&local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux, &flux, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return true;
    printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", energy, flux, minimum, maximum);
    const bool valid = std::isfinite(energy) && std::isfinite(flux) && std::isfinite(minimum) && std::isfinite(maximum);
    printf("  Validation: %s\n", valid ? "PASSED" : "FAILED"); return valid;
}

static void printUsage(const char* p) { printf("Usage: %s [-n grid] [-i iterations] [-v] [-r] [-h]\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    // Map ranks sharing a node round-robin over the node's accelerators.
    MPI_Comm local_comm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank; MPI_Comm_rank(local_comm, &local_rank);
    int device_count = 0; cudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) { if (!rank) fprintf(stderr, "No CUDA accelerator is available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(local_rank % device_count), "cudaSetDevice");
    MPI_Comm_free(&local_comm);
    int n = 512, iters = 10; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true; else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (n <= 0 || iters < 0 || ranks > n) { if (!rank) fprintf(stderr, "grid size must be positive and at least the MPI rank count\n"); MPI_Finalize(); return 1; }
    const int base = n / ranks, extra = n % ranks;
    const int rows = base + (rank < extra), first = rank * base + std::min(rank, extra);
    if (!rank) printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\nGrid size: %d x %d = %lld elements\nIterations: %d\nMPI ranks: %d, OpenMP threads/rank: %d, CUDA: enabled\n\n", n, n, static_cast<long long>(n) * n, iters, ranks, omp_get_max_threads());
    World world; buildSquare2D(world, n, first, rows);
    MPI_Barrier(MPI_COMM_WORLD); const auto start = std::chrono::high_resolution_clock::now();
    runSimulation(world, iters, rank, ranks);
    const double local_seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    double seconds; MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    uint64_t local_hash = computeHash(world), hash; MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (!rank) { const double geps = seconds > 0 ? static_cast<double>(n) * n * iters / seconds / 1e9 : 0; printf("Computation time: %.3f ms\nPerformance:\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n", seconds * 1000, geps, geps * 22, static_cast<unsigned long long>(hash)); }
    if (results) {
        std::vector<int> counts, displs; std::vector<double> all;
        if (!rank) { counts.resize(ranks); displs.resize(ranks); for (int r = 0; r < ranks; ++r) { int rr = base + (r < extra); counts[r] = rr * n; displs[r] = (r * base + std::min(r, extra)) * n; } all.resize(static_cast<size_t>(n) * n); }
        std::vector<double> local(world.elements_dynamic.size());
        #pragma omp parallel for
        for (size_t i = 0; i < local.size(); ++i) local[i] = world.elements_dynamic[i].current_energy;
        MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(all, "ElementEnergy");
    }
    const bool valid = !validate || validateResults(world, rank); MPI_Finalize(); return valid ? 0 : 1;
}
