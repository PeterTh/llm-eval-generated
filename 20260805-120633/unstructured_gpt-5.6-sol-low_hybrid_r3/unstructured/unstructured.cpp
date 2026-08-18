#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using val_t = double;

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static void mpiCheck(int error, const char* operation) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "MPI error in %s: %.*s\n", operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
}

// local_x includes one ghost row on either side. Only owned rows are written.
__global__ void updateRows(const ElementDynamic* __restrict__ input,
                           ElementDynamic* __restrict__ output,
                           int n, int global_x0, int local_begin, int local_end) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int lx = local_begin + blockIdx.y;
    if (y >= n || lx >= local_end) return;

    const int gx = global_x0 + lx - 1;
    const int pos = lx * n + y;
    const val_t center = input[pos].current_energy;
    val_t flux = 0.0;

    // transfer_coeff (0.8) * connection_flux (1.0) * 0.25
    if (gx > 0)     flux += (input[pos - n].current_energy - center) * 0.2;
    if (gx + 1 < n) flux += (input[pos + n].current_energy - center) * 0.2;
    if (y > 0)      flux += (input[pos - 1].current_energy - center) * 0.2;
    if (y + 1 < n)  flux += (input[pos + 1].current_energy - center) * 0.2;

    if ((gx == 0 || gx == n - 1) && (y == 0 || y == n - 1))
        flux += (gx == y) ? 0.5 : -0.5;

    output[pos].current_energy = center + flux;
    output[pos].total_flux = input[pos].total_flux + fabs(flux);
}

static void launchRows(const ElementDynamic* in, ElementDynamic* out, int n,
                       int x0, int begin, int end, cudaStream_t stream) {
    if (begin >= end) return;
    constexpr int threads = 256;
    const dim3 grid((n + threads - 1) / threads, end - begin);
    updateRows<<<grid, threads, 0, stream>>>(in, out, n, x0, begin, end);
    cudaCheck(cudaGetLastError(), "updateRows launch");
}

static uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        hash ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
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
    std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n"
                "  Energy range: [%.6f, %.6f]\n", energy_sum, flux_sum,
                energy_min, energy_max);
    const bool valid = std::isfinite(energy_sum) && std::isfinite(flux_sum) &&
                       std::isfinite(energy_max) && std::isfinite(energy_min);
    if (std::abs(energy_sum) > 1e-8)
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    std::printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    return valid;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>  Grid size (default: 512)\n"
                "  -i <num>  Iterations (default: 10)\n  -v        Validate\n"
                "  -r        Print results\n  -h        Help\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 2);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int n = 512, iterations = 10;
    bool validate = false, printResults = false, help = false, bad_args = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else bad_args = true;
    }
    if (help || bad_args || n <= 0 || iterations < 0) {
        if (world_rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return bad_args || n <= 0 || iterations < 0;
    }
    if (world_size > n) {
        if (world_rank == 0) std::fprintf(stderr, "Grid must have at least one row per MPI rank\n");
        MPI_Finalize();
        return 1;
    }

    // Bind ranks sharing a node round-robin to that node's accelerators.
    MPI_Comm local_comm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                                 MPI_INFO_NULL, &local_comm), "MPI_Comm_split_type");
    int local_rank = 0, device_count = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    cudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (!device_count) MPI_Abort(MPI_COMM_WORLD, 3);
    cudaCheck(cudaSetDevice(local_rank % device_count), "cudaSetDevice");
    MPI_Comm_free(&local_comm);

    const int base = n / world_size, remainder = n % world_size;
    const int rows = base + (world_rank < remainder);
    const int x0 = world_rank * base + std::min(world_rank, remainder);
    const size_t row_bytes = static_cast<size_t>(n) * sizeof(ElementDynamic);
    const size_t local_bytes = static_cast<size_t>(rows + 2) * row_bytes;

    ElementDynamic *d_current = nullptr, *d_next = nullptr, *halo = nullptr;
    cudaCheck(cudaMalloc(&d_current, local_bytes), "cudaMalloc current");
    cudaCheck(cudaMalloc(&d_next, local_bytes), "cudaMalloc next");
    cudaCheck(cudaMemset(d_current, 0, local_bytes), "cudaMemset current");
    cudaCheck(cudaMemset(d_next, 0, local_bytes), "cudaMemset next");
    cudaCheck(cudaMallocHost(&halo, 4 * row_bytes), "cudaMallocHost halo staging");
    ElementDynamic* send_top = halo;
    ElementDynamic* send_bottom = halo + n;
    ElementDynamic* recv_top = halo + 2 * n;
    ElementDynamic* recv_bottom = halo + 3 * n;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(4ULL * n); ++i) halo[i] = {0.0, 0.0};

    cudaStream_t compute_stream, copy_stream;
    cudaCheck(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking), "create compute stream");
    cudaCheck(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking), "create copy stream");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        cudaCheck(cudaMemcpyAsync(send_top, d_current + n, row_bytes,
                                  cudaMemcpyDeviceToHost, copy_stream), "copy top boundary");
        cudaCheck(cudaMemcpyAsync(send_bottom, d_current + rows * n, row_bytes,
                                  cudaMemcpyDeviceToHost, copy_stream), "copy bottom boundary");
        cudaCheck(cudaStreamSynchronize(copy_stream), "boundary D2H sync");

        MPI_Request requests[4]; int count = 0;
        if (world_rank > 0) {
            MPI_Irecv(recv_top, static_cast<int>(row_bytes), MPI_BYTE, world_rank - 1, 11,
                      MPI_COMM_WORLD, &requests[count++]);
            MPI_Isend(send_top, static_cast<int>(row_bytes), MPI_BYTE, world_rank - 1, 12,
                      MPI_COMM_WORLD, &requests[count++]);
        }
        if (world_rank + 1 < world_size) {
            MPI_Irecv(recv_bottom, static_cast<int>(row_bytes), MPI_BYTE, world_rank + 1, 12,
                      MPI_COMM_WORLD, &requests[count++]);
            MPI_Isend(send_bottom, static_cast<int>(row_bytes), MPI_BYTE, world_rank + 1, 11,
                      MPI_COMM_WORLD, &requests[count++]);
        }

        // Interior rows do not depend on incoming halo data.
        launchRows(d_current, d_next, n, x0, 2, rows, compute_stream);
        if (count) mpiCheck(MPI_Waitall(count, requests, MPI_STATUSES_IGNORE), "MPI_Waitall");
        if (world_rank > 0)
            cudaCheck(cudaMemcpyAsync(d_current, recv_top, row_bytes,
                                      cudaMemcpyHostToDevice, copy_stream), "copy top halo");
        if (world_rank + 1 < world_size)
            cudaCheck(cudaMemcpyAsync(d_current + (rows + 1) * n, recv_bottom, row_bytes,
                                      cudaMemcpyHostToDevice, copy_stream), "copy bottom halo");
        cudaCheck(cudaStreamSynchronize(copy_stream), "halo H2D sync");
        launchRows(d_current, d_next, n, x0, 1, std::min(2, rows + 1), compute_stream);
        if (rows > 1) launchRows(d_current, d_next, n, x0, rows, rows + 1, compute_stream);
        cudaCheck(cudaStreamSynchronize(compute_stream), "iteration compute sync");
        std::swap(d_current, d_next);
    }
    cudaCheck(cudaDeviceSynchronize(), "final synchronize");
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<ElementDynamic> local(rows);
    cudaCheck(cudaMemcpy(local.data(), d_current + n, rows * row_bytes,
                         cudaMemcpyDeviceToHost), "copy final state");
    std::vector<int> counts(world_size), displacements(world_size);
    for (int r = 0; r < world_size; ++r) {
        const int rr = base + (r < remainder);
        counts[r] = rr * n * static_cast<int>(sizeof(ElementDynamic));
        displacements[r] = (r * base + std::min(r, remainder)) * n * static_cast<int>(sizeof(ElementDynamic));
    }
    std::vector<ElementDynamic> global;
    if (world_rank == 0) global.resize(static_cast<size_t>(n) * n);
    mpiCheck(MPI_Gatherv(local.data(), counts[world_rank], MPI_BYTE, global.data(), counts.data(),
                         displacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD), "MPI_Gatherv");

    int result = 0;
    if (world_rank == 0) {
        const double ms = elapsed * 1000.0;
        const double giga = elapsed > 0.0 ? (static_cast<double>(iterations) * n * n / elapsed / 1e9) : 0.0;
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        std::printf("Grid size: %d x %d = %lld elements\nIterations: %d\n", n, n, 1LL*n*n, iterations);
        std::printf("Parallel configuration: %d MPI ranks, %d OpenMP threads/rank, CUDA GPUs\n",
                    world_size, omp_get_max_threads());
        std::printf("Computation time: %.3f ms\nPerformance:\n  Time per iteration: %.4f ms\n"
                    "  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n",
                    ms, iterations ? ms / iterations : 0.0, giga, giga * 22.0);
        std::printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(computeHash(global)));
        if (printResults) {
            std::vector<double> energies(global.size());
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(global.size()); ++i)
                energies[i] = global[i].current_energy;
            print_results(energies, "ElementEnergy");
        }
        if (validate && !validateResults(global)) result = 1;
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaStreamDestroy(compute_stream); cudaStreamDestroy(copy_stream);
    cudaFreeHost(halo); cudaFree(d_current); cudaFree(d_next);
    MPI_Finalize();
    return result;
}
