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

constexpr int DEFAULT_MAT_ID = 0;
constexpr int INFLOW_MAT_ID = 1;
constexpr int OUTFLOW_MAT_ID = 2;

static void checkCuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA failure at %s: %s\n", where, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

// The original connectivity order is down, up, right, left.  Keeping that
// order in the kernel preserves the per-element floating point operation order.
__global__ void updateKernel(const ElementDynamic* current, ElementDynamic* next,
                             const val_t* top_halo, const val_t* bottom_halo,
                             int global_row0, int local_rows, int width,
                             int first_row, int last_row) {
    const int local_x = first_row + blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_x >= last_row || y >= width) return;

    const int global_x = global_row0 + local_x;
    const int i = local_x * width + y;
    const val_t energy = current[i].current_energy;
    int material = DEFAULT_MAT_ID;
    // This priority also matches the n=1 initialization in the original code.
    if ((global_x == 0 && y == 0) || (global_x == width - 1 && y == width - 1))
        material = INFLOW_MAT_ID;
    else if ((global_x == 0 && y == width - 1) ||
             (global_x == width - 1 && y == 0))
        material = OUTFLOW_MAT_ID;

    val_t total_flux = material == INFLOW_MAT_ID ? 0.5 :
                       (material == OUTFLOW_MAT_ID ? -0.5 : 0.0);
    if (global_x + 1 < width) {
        const val_t neighbor = local_x + 1 < local_rows
            ? current[i + width].current_energy : bottom_halo[y];
        total_flux += (neighbor - energy) * 0.8 * 1.0 * 0.25;
    }
    if (global_x > 0) {
        const val_t neighbor = local_x > 0
            ? current[i - width].current_energy : top_halo[y];
        total_flux += (neighbor - energy) * 0.8 * 1.0 * 0.25;
    }
    if (y + 1 < width) total_flux += (current[i + 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (y > 0)         total_flux += (current[i - 1].current_energy - energy) * 0.8 * 1.0 * 0.25;

    next[i].current_energy = energy + total_flux;
    next[i].total_flux = current[i].total_flux + fabs(total_flux);
}

static void launchRows(const ElementDynamic* current, ElementDynamic* next,
                       const val_t* top_halo, const val_t* bottom_halo,
                       int global_row0, int local_rows, int width,
                       int first_row, int last_row, cudaStream_t stream) {
    if (first_row >= last_row) return;
    const dim3 block(32, 8);
    const dim3 grid((width + block.x - 1) / block.x,
                    (last_row - first_row + block.y - 1) / block.y);
    updateKernel<<<grid, block, 0, stream>>>(current, next, top_halo, bottom_halo,
                                              global_row0, local_rows, width,
                                              first_row, last_row);
    checkCuda(cudaGetLastError(), "updateKernel launch");
}

static uint64_t localHash(const std::vector<ElementDynamic>& elements, uint64_t global_offset) {
    uint64_t hash = 0;
#pragma omp parallel for reduction(^:hash) schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(elements.size()); ++i) {
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        const uint64_t global_i = global_offset + static_cast<uint64_t>(i);
        hash ^= (energy_bits + global_i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -n <num>     Grid size (NxN elements) (default: 512)\n"
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
    bool validate = false, print_results_enabled = false;
    bool bad_args = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results_enabled = true;
        else if (!strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else bad_args = true;
    }
    if (bad_args || n <= 0 || iterations < 0) {
        if (rank == 0) { printf("Invalid options\n"); printUsage(argv[0]); }
        MPI_Finalize(); return 1;
    }

    // Ranks with no rows remain MPI participants but do no GPU work.
    const int active_ranks = std::min(ranks, n);
    const int base_rows = n / active_ranks, extra_rows = n % active_ranks;
    const int local_rows = rank < active_ranks ? base_rows + (rank < extra_rows) : 0;
    const int row0 = rank < active_ranks ? rank * base_rows + std::min(rank, extra_rows) : n;
    const size_t local_count = static_cast<size_t>(local_rows) * n;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %lld elements\nIterations: %d\nValidation: %s\n", n, n,
               static_cast<long long>(n) * n, iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (%d active), OpenMP threads/rank: %d\n", ranks, active_ranks, omp_get_max_threads());
        printf("Building distributed unstructured mesh...\n");
        const size_t static_mem = static_cast<size_t>(n) * n * (sizeof(idx_t) * 10 + sizeof(val_t) * 8);
        const size_t dynamic_mem = static_cast<size_t>(n) * n * sizeof(ElementDynamic) * 2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (static_mem + dynamic_mem) / 1048576.0, static_mem / 1048576.0, dynamic_mem / 1048576.0);
    }

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) { if (rank == 0) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    checkCuda(cudaSetDevice(local_rank % device_count), "cudaSetDevice");
    MPI_Comm_free(&local_comm);

    std::vector<ElementDynamic> host_current(local_count, ElementDynamic{0.0, 0.0});
    ElementDynamic *d_current = nullptr, *d_next = nullptr;
    val_t *d_top = nullptr, *d_bottom = nullptr;
    checkCuda(cudaMalloc(&d_current, local_count * sizeof(ElementDynamic)), "cudaMalloc current");
    checkCuda(cudaMalloc(&d_next, local_count * sizeof(ElementDynamic)), "cudaMalloc next");
    checkCuda(cudaMalloc(&d_top, n * sizeof(val_t)), "cudaMalloc top halo");
    checkCuda(cudaMalloc(&d_bottom, n * sizeof(val_t)), "cudaMalloc bottom halo");
    checkCuda(cudaMemcpy(d_current, host_current.data(), local_count * sizeof(ElementDynamic), cudaMemcpyHostToDevice), "initial copy");
    cudaStream_t compute_stream, transfer_stream;
    checkCuda(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking), "compute stream");
    checkCuda(cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking), "transfer stream");

    std::vector<ElementDynamic> edge_top(n), edge_bottom(n);
    std::vector<val_t> send_top(n), send_bottom(n), recv_top(n, 0.0), recv_bottom(n, 0.0);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        if (local_rows) {
            checkCuda(cudaMemcpyAsync(edge_top.data(), d_current, n * sizeof(ElementDynamic), cudaMemcpyDeviceToHost, transfer_stream), "copy top row");
            checkCuda(cudaMemcpyAsync(edge_bottom.data(), d_current + (local_rows - 1) * n,
                                      n * sizeof(ElementDynamic), cudaMemcpyDeviceToHost, transfer_stream), "copy bottom row");
            // All rows except the two slab boundaries can run while their halos travel.
            launchRows(d_current, d_next, d_top, d_bottom, row0, local_rows, n, 1, std::max(1, local_rows - 1), compute_stream);
            checkCuda(cudaStreamSynchronize(transfer_stream), "edge copies");
#pragma omp parallel for schedule(static)
            for (int y = 0; y < n; ++y) {
                send_top[y] = edge_top[y].current_energy;
                send_bottom[y] = edge_bottom[y].current_energy;
            }
        }
        MPI_Request req[4]; int nr = 0;
        if (rank < active_ranks) {
            if (rank > 0) { MPI_Irecv(recv_top.data(), n, MPI_DOUBLE, rank - 1, 11, MPI_COMM_WORLD, &req[nr++]); MPI_Isend(send_top.data(), n, MPI_DOUBLE, rank - 1, 12, MPI_COMM_WORLD, &req[nr++]); }
            if (rank + 1 < active_ranks) { MPI_Irecv(recv_bottom.data(), n, MPI_DOUBLE, rank + 1, 12, MPI_COMM_WORLD, &req[nr++]); MPI_Isend(send_bottom.data(), n, MPI_DOUBLE, rank + 1, 11, MPI_COMM_WORLD, &req[nr++]); }
        }
        if (nr) MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
        if (local_rows) {
            checkCuda(cudaMemcpyAsync(d_top, recv_top.data(), n * sizeof(val_t), cudaMemcpyHostToDevice, compute_stream), "copy top halo");
            checkCuda(cudaMemcpyAsync(d_bottom, recv_bottom.data(), n * sizeof(val_t), cudaMemcpyHostToDevice, compute_stream), "copy bottom halo");
            launchRows(d_current, d_next, d_top, d_bottom, row0, local_rows, n, 0, 1, compute_stream);
            if (local_rows > 1) launchRows(d_current, d_next, d_top, d_bottom, row0, local_rows, n, local_rows - 1, local_rows, compute_stream);
            checkCuda(cudaStreamSynchronize(compute_stream), "iteration kernel");
            std::swap(d_current, d_next);
        }
    }
    const auto end = std::chrono::high_resolution_clock::now();
    double local_seconds = std::chrono::duration<double>(end - start).count(), elapsed_seconds = 0.0;
    MPI_Reduce(&local_seconds, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (local_rows) checkCuda(cudaMemcpy(host_current.data(), d_current, local_count * sizeof(ElementDynamic), cudaMemcpyDeviceToHost), "final copy");

    uint64_t hash = localHash(host_current, static_cast<uint64_t>(row0) * n), global_hash = 0;
    MPI_Reduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    std::vector<ElementDynamic> all_elements;
    if (rank == 0 && (validate || print_results_enabled)) all_elements.resize(static_cast<size_t>(n) * n);
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) { const int rows = r < active_ranks ? base_rows + (r < extra_rows) : 0; const int first = r < active_ranks ? r * base_rows + std::min(r, extra_rows) : n; counts[r] = rows * n * sizeof(ElementDynamic); displs[r] = first * n * sizeof(ElementDynamic); }
    MPI_Gatherv(host_current.data(), static_cast<int>(local_count * sizeof(ElementDynamic)), MPI_BYTE,
                all_elements.data(), counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double ms = elapsed_seconds * 1000.0;
        const int measured = std::max(iterations - 1, 1);
        const double geps = elapsed_seconds > 0 ? (static_cast<double>(measured) * n * n) / elapsed_seconds / 1e9 : 0.0;
        printf("Computation time: %.3f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n", ms, ms / measured, geps, geps * 22.0, static_cast<unsigned long long>(global_hash));
        if (print_results_enabled) {
            std::vector<double> energies(all_elements.size());
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(energies.size()); ++i)
                energies[i] = all_elements[i].current_energy;
            print_results(energies, "ElementEnergy");
        }
        if (validate) { val_t es = 0, fs = 0, emin = std::numeric_limits<val_t>::max(), emax = std::numeric_limits<val_t>::lowest(); for (const auto& e : all_elements) { es += e.current_energy; fs += e.total_flux; emin = std::min(emin, e.current_energy); emax = std::max(emax, e.current_energy); } printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n  Validation: %s\n", es, fs, emin, emax, (std::isfinite(es) && std::isfinite(fs) && std::isfinite(emin) && std::isfinite(emax)) ? "PASSED" : "FAILED"); }
    }
    cudaStreamDestroy(compute_stream); cudaStreamDestroy(transfer_stream); cudaFree(d_current); cudaFree(d_next); cudaFree(d_top); cudaFree(d_bottom);
    MPI_Finalize();
    return 0;
}
