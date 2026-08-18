#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// The domain is decomposed in contiguous z slabs.  Each GPU allocation has
// one ghost plane at either end; x and y retain the original clamped boundary.
static inline size_t plane_size(size_t nx, size_t ny) { return nx * ny; }

static void cuda_check(cudaError_t status, const char* operation, MPI_Comm comm) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        std::fprintf(stderr, "rank %d: CUDA %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(comm, 2);
    }
}
#define CUDA_CHECK(call, comm) cuda_check((call), #call, (comm))

__global__ void chemical_kernel(const double* __restrict__ c, double* __restrict__ mu,
                                int nx, int ny, int nz_local, double gamma,
                                double e_aa, double e_bb, double e_ab) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > nz_local) return;
    const size_t plane = static_cast<size_t>(nx) * ny;
    const size_t i = static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx + x;
    const size_t xp = i + (x + 1 < nx ? 1 : 0);
    const size_t xm = i - (x > 0 ? 1 : 0);
    const size_t yp = i + (y + 1 < ny ? nx : 0);
    const size_t ym = i - (y > 0 ? nx : 0);
    const double cv = c[i];
    const double lap = c[xp] + c[xm] + c[yp] + c[ym] + c[i + plane] + c[i - plane] - 6.0 * cv;
    mu[i] = 4.5 * ((cv + 1.0) * e_aa + (cv - 1.0) * e_bb - 2.0 * cv * e_ab)
          + 3.0 * cv + cv * cv * cv - gamma * lap;
}

__global__ void update_kernel(double* __restrict__ next, const double* __restrict__ old,
                              const double* __restrict__ mu, int nx, int ny, int nz_local,
                              double dt_d) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > nz_local) return;
    const size_t plane = static_cast<size_t>(nx) * ny;
    const size_t i = static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx + x;
    const size_t xp = i + (x + 1 < nx ? 1 : 0);
    const size_t xm = i - (x > 0 ? 1 : 0);
    const size_t yp = i + (y + 1 < ny ? nx : 0);
    const size_t ym = i - (y > 0 ? nx : 0);
    const double mv = mu[i];
    next[i] = old[i] + dt_d * (mu[xp] + mu[xm] + mu[yp] + mu[ym] +
                               mu[i + plane] + mu[i - plane] - 6.0 * mv);
}

static void exchange_halos(double* device, double* send_lo, double* send_hi,
                           double* recv_lo, double* recv_hi, size_t plane, int nz_local,
                           int rank, int ranks, cudaStream_t stream, MPI_Comm comm) {
    const size_t bytes = plane * sizeof(double);
    CUDA_CHECK(cudaMemcpyAsync(send_lo, device + plane, bytes, cudaMemcpyDeviceToHost, stream), comm);
    CUDA_CHECK(cudaMemcpyAsync(send_hi, device + static_cast<size_t>(nz_local) * plane, bytes,
                               cudaMemcpyDeviceToHost, stream), comm);
    CUDA_CHECK(cudaStreamSynchronize(stream), comm);

    MPI_Request requests[4];
    int nreq = 0;
    if (rank > 0) {
        MPI_Irecv(recv_lo, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 71, comm, &requests[nreq++]);
        MPI_Isend(send_lo, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 72, comm, &requests[nreq++]);
    } else {
        std::memcpy(recv_lo, send_lo, bytes);
    }
    if (rank + 1 < ranks) {
        MPI_Irecv(recv_hi, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 72, comm, &requests[nreq++]);
        MPI_Isend(send_hi, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 71, comm, &requests[nreq++]);
    } else {
        std::memcpy(recv_hi, send_hi, bytes);
    }
    MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
    CUDA_CHECK(cudaMemcpyAsync(device, recv_lo, bytes, cudaMemcpyHostToDevice, stream), comm);
    CUDA_CHECK(cudaMemcpyAsync(device + static_cast<size_t>(nz_local + 1) * plane, recv_hi, bytes,
                               cudaMemcpyHostToDevice, stream), comm);
}

static void print_usage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("  -x <num>  Grid X size (default: 64)\n  -y <num>  Grid Y size (default: X)\n");
    std::printf("  -z <num>  Grid Z size (default: X)\n  -i <num>  Time steps (default: 20)\n");
    std::printf("  -v        Enable validation\n  -r        Print results for external validation\n  -h        Show help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    MPI_Comm node_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, print_results_requested = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) print_results_requested = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) print_usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); print_usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || iterations < 0 || nz < static_cast<size_t>(ranks) ||
        nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (!rank) std::fprintf(stderr, "Invalid dimensions or more MPI ranks than z planes.\n");
        MPI_Abort(comm, 1);
    }

    const int base = static_cast<int>(nz / ranks), remainder = static_cast<int>(nz % ranks);
    const int nz_local = base + (rank < remainder);
    const int z_start = rank * base + std::min(rank, remainder);
    const size_t plane = plane_size(nx, ny), local_cells = plane * nz_local;
    const size_t alloc_cells = plane * static_cast<size_t>(nz_local + 2);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count), comm);
    if (!device_count) { if (!rank) std::fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(comm, 2); }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count), comm);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), comm);

    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    double *send_lo = nullptr, *send_hi = nullptr, *recv_lo = nullptr, *recv_hi = nullptr;
    CUDA_CHECK(cudaMalloc(&cold, alloc_cells * sizeof(double)), comm);
    CUDA_CHECK(cudaMalloc(&cnew, alloc_cells * sizeof(double)), comm);
    CUDA_CHECK(cudaMalloc(&mu, alloc_cells * sizeof(double)), comm);
    CUDA_CHECK(cudaMallocHost(&send_lo, plane * sizeof(double)), comm);
    CUDA_CHECK(cudaMallocHost(&send_hi, plane * sizeof(double)), comm);
    CUDA_CHECK(cudaMallocHost(&recv_lo, plane * sizeof(double)), comm);
    CUDA_CHECK(cudaMallocHost(&recv_hi, plane * sizeof(double)), comm);

    std::vector<double> host(local_cells);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(local_cells); ++i) {
        const size_t global_id = static_cast<size_t>(z_start) * plane + static_cast<size_t>(i);
        const size_t volume = nx * ny * nz;
        host[static_cast<size_t>(i)] = -1.0 + 2.0 * (((global_id + 1) * 1299709 % volume) /
                                                      static_cast<double>(volume));
    }
    CUDA_CHECK(cudaMemcpyAsync(cold + plane, host.data(), local_cells * sizeof(double),
                               cudaMemcpyHostToDevice, stream), comm);
    CUDA_CHECK(cudaStreamSynchronize(stream), comm);

    if (!rank) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark (MPI + OpenMP + CUDA)\n");
        std::printf("Grid size: %zu x %zu x %zu, MPI ranks: %d, OpenMP threads/rank: %d\n",
                    nx, ny, nz, ranks, omp_get_max_threads());
        std::printf("Time steps: %d\n", iterations);
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    const dim3 block(8, 8, 4);
    const dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y,
                    (nz_local + block.z - 1) / block.z);
    for (int step = 0; step < iterations; ++step) {
        exchange_halos(cold, send_lo, send_hi, recv_lo, recv_hi, plane, nz_local, rank, ranks, stream, comm);
        chemical_kernel<<<grid, block, 0, stream>>>(cold, mu, static_cast<int>(nx), static_cast<int>(ny), nz_local,
                                                     0.5, -(2.0 / 9.0), -(2.0 / 9.0), 2.0 / 9.0);
        CUDA_CHECK(cudaGetLastError(), comm);
        CUDA_CHECK(cudaStreamSynchronize(stream), comm);
        exchange_halos(mu, send_lo, send_hi, recv_lo, recv_hi, plane, nz_local, rank, ranks, stream, comm);
        update_kernel<<<grid, block, 0, stream>>>(cnew, cold, mu, static_cast<int>(nx), static_cast<int>(ny), nz_local, 0.01);
        CUDA_CHECK(cudaGetLastError(), comm);
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream), comm);
    double elapsed = MPI_Wtime() - start, max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, local_cells * sizeof(double), cudaMemcpyDeviceToHost), comm);

    if (!rank) {
        std::printf("Computation time: %.3f ms\n", max_elapsed * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(nx * ny * nz) * iterations / max_elapsed / 1e6);
    }
    if (print_results_requested) {
        std::vector<int> counts, displacements;
        std::vector<double> global;
        if (!rank) { counts.resize(ranks); displacements.resize(ranks); for (int r = 0; r < ranks; ++r) { const int n = base + (r < remainder); counts[r] = static_cast<int>(plane) * n; displacements[r] = static_cast<int>(plane) * (r * base + std::min(r, remainder)); } global.resize(nx * ny * nz); }
        MPI_Gatherv(host.data(), static_cast<int>(local_cells), MPI_DOUBLE, rank ? nullptr : global.data(),
                    rank ? nullptr : counts.data(), rank ? nullptr : displacements.data(), MPI_DOUBLE, 0, comm);
        if (!rank) print_results(global, "Concentration");
    }
    int exit_code = 0;
    if (validate) {
        int local_bad = 0;
        double local_min = std::numeric_limits<double>::infinity(), local_max = -std::numeric_limits<double>::infinity();
        #pragma omp parallel for reduction(|:local_bad) reduction(min:local_min) reduction(max:local_max)
        for (long long i = 0; i < static_cast<long long>(local_cells); ++i) { const double v = host[static_cast<size_t>(i)]; local_bad |= !std::isfinite(v); local_min = std::min(local_min, v); local_max = std::max(local_max, v); }
        int bad; double min_value, max_value;
        MPI_Reduce(&local_bad, &bad, 1, MPI_INT, MPI_LOR, 0, comm);
        MPI_Reduce(&local_min, &min_value, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&local_max, &max_value, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!rank) {
            const bool valid = !bad && max_value <= 10.0 && min_value >= -10.0;
            std::printf("Concentration range: [%.6f, %.6f]\n", min_value, max_value);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exit_code = valid ? 0 : 1;
        }
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, comm);
    }
    CUDA_CHECK(cudaFree(cold), comm); CUDA_CHECK(cudaFree(cnew), comm); CUDA_CHECK(cudaFree(mu), comm);
    CUDA_CHECK(cudaFreeHost(send_lo), comm); CUDA_CHECK(cudaFreeHost(send_hi), comm); CUDA_CHECK(cudaFreeHost(recv_lo), comm); CUDA_CHECK(cudaFreeHost(recv_hi), comm);
    CUDA_CHECK(cudaStreamDestroy(stream), comm);
    MPI_Comm_free(&node_comm);
    MPI_Finalize();
    return exit_code;
}
