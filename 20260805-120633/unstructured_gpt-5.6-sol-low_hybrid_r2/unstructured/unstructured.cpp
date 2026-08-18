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

// local_row includes two halo rows. global_row is its corresponding global row.
__global__ void updateKernel(const ElementDynamic* __restrict__ current,
                             ElementDynamic* __restrict__ next,
                             int columns, int local_rows, int global_first_row,
                             int global_rows) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = local_rows * columns;
    if (tid >= count) return;

    const int local_row = tid / columns + 1;
    const int column = tid - (local_row - 1) * columns;
    const int global_row = global_first_row + local_row - 1;
    const int index = local_row * columns + column;
    const val_t energy = current[index].current_energy;

    val_t external = 0.0;
    if ((global_row == 0 || global_row == global_rows - 1) &&
        (column == 0 || column == columns - 1)) {
        external = (global_row == column) ? 0.5 : -0.5;
    }

    val_t total_flux = external;
    // Match the reference connectivity order: down, up, right, left.
    if (global_row + 1 < global_rows) {
        total_flux += (current[index + columns].current_energy - energy) * 0.8 * 1.0 * 0.25;
    }
    if (global_row > 0) {
        total_flux += (current[index - columns].current_energy - energy) * 0.8 * 1.0 * 0.25;
    }
    if (column + 1 < columns) {
        total_flux += (current[index + 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
    }
    if (column > 0) {
        total_flux += (current[index - 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
    }

    next[index].current_energy = energy + total_flux;
    next[index].total_flux = current[index].total_flux + fabs(total_flux);
}

static void decomposition(int n, int ranks, int rank, int& first_row, int& rows) {
    const int base = n / ranks;
    const int remainder = n % ranks;
    rows = base + (rank < remainder);
    first_row = rank * base + std::min(rank, remainder);
}

static void exchangeHalos(ElementDynamic* device, ElementDynamic* top,
                          ElementDynamic* bottom, int rows, int columns,
                          int rank, int ranks, cudaStream_t stream) {
    const size_t bytes = static_cast<size_t>(columns) * sizeof(ElementDynamic);
    if (rank > 0)
        cudaCheck(cudaMemcpyAsync(top, device + columns, bytes,
                                  cudaMemcpyDeviceToHost, stream), "copy top boundary");
    if (rank + 1 < ranks)
        cudaCheck(cudaMemcpyAsync(bottom, device + static_cast<size_t>(rows) * columns,
                                  bytes, cudaMemcpyDeviceToHost, stream), "copy bottom boundary");
    cudaCheck(cudaStreamSynchronize(stream), "stage halo sends");

    MPI_Request requests[4];
    int count = 0;
    if (rank > 0) {
        mpiCheck(MPI_Irecv(top + columns, static_cast<int>(bytes), MPI_BYTE, rank - 1,
                           1, MPI_COMM_WORLD, &requests[count++]), "receive top halo");
        mpiCheck(MPI_Isend(top, static_cast<int>(bytes), MPI_BYTE, rank - 1,
                           0, MPI_COMM_WORLD, &requests[count++]), "send top row");
    }
    if (rank + 1 < ranks) {
        mpiCheck(MPI_Irecv(bottom + columns, static_cast<int>(bytes), MPI_BYTE, rank + 1,
                           0, MPI_COMM_WORLD, &requests[count++]), "receive bottom halo");
        mpiCheck(MPI_Isend(bottom, static_cast<int>(bytes), MPI_BYTE, rank + 1,
                           1, MPI_COMM_WORLD, &requests[count++]), "send bottom row");
    }
    if (count) mpiCheck(MPI_Waitall(count, requests, MPI_STATUSES_IGNORE), "halo exchange");

    if (rank > 0)
        cudaCheck(cudaMemcpyAsync(device, top + columns, bytes,
                                  cudaMemcpyHostToDevice, stream), "upload top halo");
    if (rank + 1 < ranks)
        cudaCheck(cudaMemcpyAsync(device + static_cast<size_t>(rows + 1) * columns,
                                  bottom + columns, bytes, cudaMemcpyHostToDevice, stream),
                  "upload bottom halo");
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -n <num>  Grid size (default: 512)\n");
    std::printf("  -i <num>  Iterations (default: 10)\n");
    std::printf("  -v        Validate results\n");
    std::printf("  -r        Print results for external validation\n");
    std::printf("  -h        Show help\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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
    if (help || bad_args || n <= 0 || iterations < 0 || ranks > n) {
        if (rank == 0) {
            if (ranks > n) std::fprintf(stderr, "Grid must have at least one row per MPI rank.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return (help && !bad_args) ? 0 : 1;
    }

    MPI_Comm local_comm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &local_comm), "local communicator");
    int local_rank = 0, device_count = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    cudaCheck(cudaGetDeviceCount(&device_count), "discover CUDA devices");
    if (device_count == 0) {
        if (rank == 0) std::fprintf(stderr, "This benchmark requires CUDA devices.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(local_rank % device_count), "select CUDA device");

    int first_row = 0, local_rows = 0;
    decomposition(n, ranks, rank, first_row, local_rows);
    const size_t local_with_halos = static_cast<size_t>(local_rows + 2) * n;
    std::vector<ElementDynamic> host(local_with_halos);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(local_with_halos); ++i) host[i] = {0.0, 0.0};

    ElementDynamic *device_a = nullptr, *device_b = nullptr;
    cudaCheck(cudaMalloc(&device_a, local_with_halos * sizeof(ElementDynamic)), "allocate state A");
    cudaCheck(cudaMalloc(&device_b, local_with_halos * sizeof(ElementDynamic)), "allocate state B");
    cudaCheck(cudaMemcpy(device_a, host.data(), local_with_halos * sizeof(ElementDynamic),
                         cudaMemcpyHostToDevice), "initialize device state");
    cudaCheck(cudaMemset(device_b, 0, local_with_halos * sizeof(ElementDynamic)), "initialize swap state");
    ElementDynamic *top = nullptr, *bottom = nullptr;
    cudaCheck(cudaMallocHost(&top, static_cast<size_t>(2) * n * sizeof(ElementDynamic)), "allocate top staging");
    cudaCheck(cudaMallocHost(&bottom, static_cast<size_t>(2) * n * sizeof(ElementDynamic)), "allocate bottom staging");
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create CUDA stream");

    if (rank == 0) {
        const size_t global_elements = static_cast<size_t>(n) * n;
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\nIterations: %d\nValidation: %s\n", n, n,
                    global_elements, iterations, validate ? "enabled" : "disabled");
        std::printf("Parallelism: %d MPI ranks, up to %d OpenMP threads/rank, CUDA device/rank\n\n",
                    ranks, omp_get_max_threads());
    }

    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "pre-simulation barrier");
    const double start = MPI_Wtime();
    const int threads = 256;
    const int blocks = (local_rows * n + threads - 1) / threads;
    for (int iter = 0; iter < iterations; ++iter) {
        exchangeHalos(device_a, top, bottom, local_rows, n, rank, ranks, stream);
        updateKernel<<<blocks, threads, 0, stream>>>(device_a, device_b, n, local_rows, first_row, n);
        cudaCheck(cudaGetLastError(), "launch update kernel");
        cudaCheck(cudaStreamSynchronize(stream), "execute update kernel");
        std::swap(device_a, device_b);
    }
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "post-simulation barrier");
    const double elapsed = MPI_Wtime() - start;

    cudaCheck(cudaMemcpy(host.data() + n, device_a + n,
                         static_cast<size_t>(local_rows) * n * sizeof(ElementDynamic),
                         cudaMemcpyDeviceToHost), "download result");

    const int local_count = local_rows * n;
    std::vector<int> counts, displacements;
    std::vector<ElementDynamic> global;
    if (rank == 0) {
        counts.resize(ranks); displacements.resize(ranks);
        for (int r = 0; r < ranks; ++r) {
            int begin, rows;
            decomposition(n, ranks, r, begin, rows);
            counts[r] = rows * n * static_cast<int>(sizeof(ElementDynamic));
            displacements[r] = begin * n * static_cast<int>(sizeof(ElementDynamic));
        }
        global.resize(static_cast<size_t>(n) * n);
    }
    mpiCheck(MPI_Gatherv(host.data() + n, local_count * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE,
                         global.data(), counts.data(), displacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD),
             "gather results");

    if (rank == 0) {
        const double giga = iterations > 0 && elapsed > 0.0
            ? (static_cast<double>(iterations) * n * n) / elapsed / 1e9 : 0.0;
        std::printf("Computation time: %.3f ms\nPerformance:\n", elapsed * 1000.0);
        std::printf("  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n",
                    iterations ? elapsed * 1000.0 / iterations : 0.0, giga, giga * 22.0);

        uint64_t hash = 0;
        for (size_t i = 0; i < global.size(); ++i) {
            uint64_t energy_bits, flux_bits;
            std::memcpy(&energy_bits, &global[i].current_energy, sizeof(energy_bits));
            std::memcpy(&flux_bits, &global[i].total_flux, sizeof(flux_bits));
            hash ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
        }
        std::printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));

        if (printResults) {
            std::vector<double> energies(global.size());
#pragma omp parallel for schedule(static)
            for (long long i = 0; i < static_cast<long long>(global.size()); ++i)
                energies[i] = global[i].current_energy;
            print_results(energies, "ElementEnergy");
        }
        if (validate) {
            double energy_sum = 0.0, flux_sum = 0.0;
            double energy_min = std::numeric_limits<double>::max();
            double energy_max = std::numeric_limits<double>::lowest();
#pragma omp parallel for reduction(+:energy_sum,flux_sum) reduction(min:energy_min) reduction(max:energy_max)
            for (long long i = 0; i < static_cast<long long>(global.size()); ++i) {
                energy_sum += global[i].current_energy;
                flux_sum += global[i].total_flux;
                energy_min = std::min(energy_min, global[i].current_energy);
                energy_max = std::max(energy_max, global[i].current_energy);
            }
            std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n", energy_sum, flux_sum);
            std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
            const bool valid = std::isfinite(energy_sum) && std::isfinite(flux_sum) &&
                               std::isfinite(energy_min) && std::isfinite(energy_max);
            std::printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    cudaStreamDestroy(stream);
    cudaFreeHost(top); cudaFreeHost(bottom);
    cudaFree(device_a); cudaFree(device_b);
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return 0;
}
