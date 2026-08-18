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

using Real = double;

#define CUDA_CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__global__ void stencil_kernel(const Real* __restrict__ in, Real* __restrict__ out,
                               int nx, int ny, int local_z, int global_z0,
                               int global_nz, int first_z, int last_z) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int lz = first_z + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || lz > last_z) return;
    const size_t plane = static_cast<size_t>(nx) * ny;
    const size_t p = static_cast<size_t>(lz) * plane + static_cast<size_t>(y) * nx + x;
    const int gz = global_z0 + lz - 1;
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || gz == 0 || gz == global_nz - 1) {
        out[p] = in[p];
    } else {
        out[p] = (in[p] + in[p - 1] + in[p + 1] + in[p - nx] + in[p + nx] +
                  in[p - plane] + in[p + plane]) * (1.0 / 7.0);
    }
}

static void launch_stencil(const Real* in, Real* out, int nx, int ny, int local_z,
                           int global_z0, int global_nz, int first_z, int last_z,
                           cudaStream_t stream) {
    if (first_z > last_z) return;
    const dim3 block(32, 4, 2);
    const dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y,
                    (last_z - first_z + 1 + block.z - 1) / block.z);
    stencil_kernel<<<grid, block, 0, stream>>>(in, out, nx, ny, local_z, global_z0,
                                                 global_nz, first_z, last_z);
    CUDA_CHECK(cudaGetLastError());
}

static bool validate_result(const std::vector<Real>& grid, int rank) {
    int bad = 0;
    Real local_min = std::numeric_limits<Real>::infinity();
    Real local_max = -std::numeric_limits<Real>::infinity();
    #pragma omp parallel for reduction(|:bad) reduction(min:local_min) reduction(max:local_max)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        const Real v = grid[static_cast<size_t>(i)];
        if (!std::isfinite(v)) bad = 1;
        local_min = std::min(local_min, v);
        local_max = std::max(local_max, v);
    }
    int global_bad;
    Real global_min, global_max;
    MPI_Allreduce(&bad, &global_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
        if (global_bad) printf("Validation failed: found NaN or Inf value\n");
        if (global_max > 1e6 || global_min < -1e6) printf("Validation failed: values out of expected range\n");
    }
    return !global_bad && global_max <= 1e6 && global_min >= -1e6;
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  -x <num>  Grid X size (default: 128)\n  -y <num>  Grid Y size (default: X)\n");
    printf("  -z <num>  Grid Z size (default: X)\n  -i <num>  Iterations (default: 10)\n");
    printf("  -v        Enable validation\n  -r        Print results\n  -h        Show help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, print_results_requested = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results_requested = true;
        else if (!strcmp(argv[i], "-h")) { if (rank == 0) print_usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); print_usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (nx < 3 || ny < 3 || nz < static_cast<size_t>(ranks) || iterations < 0 ||
        nx > INT_MAX || ny > INT_MAX || nz > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be at least 3 (and Z at least MPI ranks).\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) MPI_Abort(MPI_COMM_WORLD, 2);
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);

    const int inx = static_cast<int>(nx), iny = static_cast<int>(ny), inz = static_cast<int>(nz);
    const int base = inz / ranks, rem = inz % ranks;
    const int local_z = base + (rank < rem ? 1 : 0);
    const int z0 = rank * base + std::min(rank, rem);
    const size_t plane = nx * ny, local_cells = static_cast<size_t>(local_z) * plane;
    std::vector<Real> host(local_cells);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int lz = 0; lz < local_z; ++lz)
        for (long long q = 0; q < static_cast<long long>(plane); ++q)
            host[static_cast<size_t>(lz) * plane + q] =
                (static_cast<Real>((static_cast<size_t>(z0 + lz) * plane + q) % 19));

    Real *d_a = nullptr, *d_b = nullptr, *send_lo = nullptr, *send_hi = nullptr, *recv_lo = nullptr, *recv_hi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, (local_cells + 2 * plane) * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_b, (local_cells + 2 * plane) * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&send_lo, plane * sizeof(Real))); CUDA_CHECK(cudaMallocHost(&send_hi, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recv_lo, plane * sizeof(Real))); CUDA_CHECK(cudaMallocHost(&recv_hi, plane * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_a, 0, (local_cells + 2 * plane) * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_a + plane, host.data(), local_cells * sizeof(Real), cudaMemcpyHostToDevice));
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n", nx, ny, nz,
               iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA GPUs/rank: 1\n", ranks, omp_get_max_threads());
        printf("Running stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        const int prev = rank ? rank - 1 : MPI_PROC_NULL, next = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
        CUDA_CHECK(cudaMemcpyAsync(send_lo, d_a + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(send_hi, d_a + static_cast<size_t>(local_z) * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Request req[4];
        MPI_Irecv(recv_lo, static_cast<int>(plane), MPI_DOUBLE, prev, 11, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(recv_hi, static_cast<int>(plane), MPI_DOUBLE, next, 10, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(send_lo, static_cast<int>(plane), MPI_DOUBLE, prev, 10, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(send_hi, static_cast<int>(plane), MPI_DOUBLE, next, 11, MPI_COMM_WORLD, &req[3]);
        launch_stencil(d_a, d_b, inx, iny, local_z, z0, inz, 2, local_z - 1, stream);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        if (prev != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(d_a, recv_lo, plane * sizeof(Real), cudaMemcpyHostToDevice, stream));
        if (next != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(d_a + static_cast<size_t>(local_z + 1) * plane, recv_hi, plane * sizeof(Real), cudaMemcpyHostToDevice, stream));
        launch_stencil(d_a, d_b, inx, iny, local_z, z0, inz, 1, 1, stream);
        if (local_z > 1) launch_stencil(d_a, d_b, inx, iny, local_z, z0, inz, local_z, local_z, stream);
        std::swap(d_a, d_b);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    const double local_seconds = std::chrono::duration<double>(end - start).count();
    double elapsed = 0; MPI_Reduce(&local_seconds, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(host.data(), d_a + plane, local_cells * sizeof(Real), cudaMemcpyDeviceToHost));
    if (rank == 0) {
        const double updates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", elapsed * 1000.0, updates / elapsed / 1e6);
    }
    const bool need_global = validate || print_results_requested;
    std::vector<Real> global;
    std::vector<int> counts, offsets;
    if (rank == 0 && need_global) { global.resize(nx * ny * nz); counts.resize(ranks); offsets.resize(ranks); }
    if (need_global) {
        const int count = static_cast<int>(local_cells);
        if (rank == 0) for (int r = 0; r < ranks; ++r) { const int rz = base + (r < rem); const int rz0 = r * base + std::min(r, rem); counts[r] = static_cast<int>(rz * plane); offsets[r] = static_cast<int>(rz0 * plane); }
        MPI_Gatherv(host.data(), count, MPI_DOUBLE, rank == 0 ? global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr, rank == 0 ? offsets.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    bool valid = true;
    if (validate) valid = validate_result(host, rank);
    if (print_results_requested && rank == 0) print_results(global, "Grid");
    CUDA_CHECK(cudaStreamDestroy(stream)); CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFreeHost(send_lo)); CUDA_CHECK(cudaFreeHost(send_hi)); CUDA_CHECK(cudaFreeHost(recv_lo)); CUDA_CHECK(cudaFreeHost(recv_hi));
    MPI_Finalize();
    return valid ? 0 : 1;
}
