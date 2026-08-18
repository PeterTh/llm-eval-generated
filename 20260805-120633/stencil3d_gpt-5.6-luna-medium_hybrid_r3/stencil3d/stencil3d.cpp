#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

__host__ __device__ inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

static void cuda_check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// One CUDA thread updates one cell; MPI supplies the z-neighbor halo planes.
__global__ void stencil_kernel(const Real* __restrict__ input, Real* __restrict__ output,
                               size_t nx, size_t ny, size_t local_nz, size_t global_z0,
                               size_t global_nz) {
    constexpr int BX = 8, BY = 8, BZ = 8;
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const size_t x = blockIdx.x * BX + tx;
    const size_t y = blockIdx.y * BY + ty;
    const size_t z = blockIdx.z * BZ + tz + 1; // local actual planes are 1..local_nz

    if (x == 0 || y == 0 || x + 1 >= nx || y + 1 >= ny || z > local_nz) {
        if (x < nx && y < ny && z <= local_nz) output[idx3(x, y, z, nx, ny)] = input[idx3(x, y, z, nx, ny)];
        return;
    }
    const size_t global_z = global_z0 + z - 1;
    if (global_z == 0 || global_z + 1 >= global_nz) {
        output[idx3(x, y, z, nx, ny)] = input[idx3(x, y, z, nx, ny)];
        return;
    }
    const Real center = input[idx3(x, y, z, nx, ny)];
    const Real left = input[idx3(x - 1, y, z, nx, ny)];
    const Real right = input[idx3(x + 1, y, z, nx, ny)];
    const Real front = input[idx3(x, y - 1, z, nx, ny)];
    const Real back = input[idx3(x, y + 1, z, nx, ny)];
    const Real bottom = input[idx3(x, y, z - 1, nx, ny)];
    const Real top = input[idx3(x, y, z + 1, nx, ny)];
    output[idx3(x, y, z, nx, ny)] = (center + left + right + front + back + bottom + top) / 7.0;
}

static void initialize_local(std::vector<Real>& grid, size_t nx, size_t ny,
                             size_t local_nz, size_t global_z0) {
    const size_t plane = nx * ny;
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long z = 0; z < static_cast<long long>(local_nz); ++z)
        for (long long i = 0; i < static_cast<long long>(plane); ++i)
            grid[static_cast<size_t>(z) * plane + static_cast<size_t>(i)] =
                (idx3(static_cast<size_t>(i) % nx, static_cast<size_t>(i) / nx,
                      global_z0 + static_cast<size_t>(z) - 1, nx, ny) % 19) * 1.0;
}

static bool validateResult(const std::vector<Real>& grid) {
    if (grid.empty()) return true;
    Real minVal = grid[0], maxVal = grid[0];
    int bad = 0;
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(+:bad)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        const Real v = grid[static_cast<size_t>(i)];
        if (std::isnan(v) || std::isinf(v)) ++bad;
        minVal = std::min(minVal, v); maxVal = std::max(maxVal, v);
    }
    if (bad) { std::printf("Validation failed: found NaN or Inf value\n"); return false; }
    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        std::printf("Validation failed: values out of expected range\n"); return false;
    }
    return true;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -x <num>  X dimension (default 128)\n  -y <num>  Y dimension\n  -z <num>  Z dimension\n  -i <num>  Iterations (default 10)\n  -v        Validate\n  -r        Print results\n  -h        Help\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);

    size_t nx = 128, ny = 0, nz = 0; int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0 || static_cast<size_t>(ranks) > nz - 2) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions must be at least 3 and support all MPI ranks\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny;
    const size_t interior_z = nz - 2;
    const size_t base = interior_z / static_cast<size_t>(ranks);
    const size_t extra = interior_z % static_cast<size_t>(ranks);
    const size_t local_interior = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t global_z0 = 1 + static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const size_t local_nz = local_interior + 2;

    int devices = 0; cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    const char* local_rank_env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    const int local_rank = local_rank_env ? std::atoi(local_rank_env) : rank;
    cuda_check(cudaSetDevice(local_rank % devices), "cudaSetDevice");

    std::vector<Real> host(local_nz * plane);
    initialize_local(host, nx, ny, local_nz, global_z0);
    Real *d_a = nullptr, *d_b = nullptr;
    const size_t bytes = host.size() * sizeof(Real);
    cuda_check(cudaMalloc(&d_a, bytes), "cudaMalloc d_a"); cuda_check(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    cuda_check(cudaMemcpy(d_a, host.data(), bytes, cudaMemcpyHostToDevice), "initial copy");
    // Halo planes are not stencil outputs; seed both buffers so they remain
    // valid when double buffering changes the current input allocation.
    cuda_check(cudaMemcpy(d_b, host.data(), bytes, cudaMemcpyHostToDevice), "output buffer seed");
    Real *send_low = nullptr, *send_high = nullptr, *recv_low = nullptr, *recv_high = nullptr;
    cuda_check(cudaHostAlloc(&send_low, plane * sizeof(Real), cudaHostAllocDefault), "cudaHostAlloc");
    cuda_check(cudaHostAlloc(&send_high, plane * sizeof(Real), cudaHostAllocDefault), "cudaHostAlloc");
    cuda_check(cudaHostAlloc(&recv_low, plane * sizeof(Real), cudaHostAllocDefault), "cudaHostAlloc");
    cuda_check(cudaHostAlloc(&recv_high, plane * sizeof(Real), cudaHostAllocDefault), "cudaHostAlloc");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    const dim3 block(8, 8, 8), grid((nx + 7) / 8, (ny + 7) / 8, (local_interior + 7) / 8);
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[4]; int nreq = 0;
        if (rank > 0) {
            cuda_check(cudaMemcpy(send_low, d_a + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost), "low halo copy");
            MPI_Irecv(recv_low, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 101, MPI_COMM_WORLD, &requests[nreq++]);
            MPI_Isend(send_low, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 102, MPI_COMM_WORLD, &requests[nreq++]);
        }
        if (rank + 1 < ranks) {
            cuda_check(cudaMemcpy(send_high, d_a + local_interior * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost), "high halo copy");
            MPI_Irecv(recv_high, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 102, MPI_COMM_WORLD, &requests[nreq++]);
            MPI_Isend(send_high, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 101, MPI_COMM_WORLD, &requests[nreq++]);
        }
        if (nreq) MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
        if (rank > 0) cuda_check(cudaMemcpy(d_a, recv_low, plane * sizeof(Real), cudaMemcpyHostToDevice), "low halo upload");
        if (rank + 1 < ranks) cuda_check(cudaMemcpy(d_a + (local_interior + 1) * plane, recv_high, plane * sizeof(Real), cudaMemcpyHostToDevice), "high halo upload");
        stencil_kernel<<<grid, block>>>(d_a, d_b, nx, ny, local_interior, global_z0, nz);
        cuda_check(cudaGetLastError(), "stencil launch");
        std::swap(d_a, d_b);
    }
    cuda_check(cudaDeviceSynchronize(), "stencil completion");
    const auto end = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpy(host.data(), d_a, bytes, cudaMemcpyDeviceToHost), "final copy");

    long long local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), elapsed_ms = 0;
    MPI_Reduce(&local_ms, &elapsed_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<int> counts, displs; std::vector<Real> global;
    if (rank == 0) {
        counts.resize(ranks); displs.resize(ranks); global.resize(nx * ny * nz);
        // MPI owns only the interior z slabs; retain the original initialized
        // values for the two global boundary planes.
        #pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(global.size()); ++i)
            global[static_cast<size_t>(i)] = (static_cast<size_t>(i) % 19) * 1.0;
    }
    for (int r = 0; r < ranks; ++r) {
        const size_t rz = base + (static_cast<size_t>(r) < extra ? 1 : 0);
        if (rank == 0) { counts[r] = static_cast<int>(rz * plane); displs[r] = static_cast<int>((1 + (static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra))) * plane); }
    }
    MPI_Gatherv(host.data() + plane, static_cast<int>(local_interior * plane), MPI_DOUBLE,
                rank == 0 ? global.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        // Reassert the immutable global boundary planes after the gather.
        #pragma omp parallel for collapse(2)
        for (long long z = 0; z < static_cast<long long>(nz); ++z)
            for (long long i = 0; i < static_cast<long long>(plane); ++i) {
                const size_t x = static_cast<size_t>(i) % nx, y = static_cast<size_t>(i) / nx;
                if (z == 0 || static_cast<size_t>(z) + 1 == nz || x == 0 || x + 1 == nx || y == 0 || y + 1 == ny)
                    global[idx3(x, y, static_cast<size_t>(z), nx, ny)] = (idx3(x, y, static_cast<size_t>(z), nx, ny) % 19) * 1.0;
            }
        const double seconds = elapsed_ms / 1000.0;
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("Computation time: %lld ms\nPerformance: %.3f MCellUpdates/s\n", elapsed_ms,
                    seconds > 0.0 ? (static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations) / seconds / 1e6 : 0.0);
        if (printResults) print_results(global, "Grid");
        if (validate) { std::printf("Validating result...\nValidation: %s\n", validateResult(global) ? "PASSED" : "FAILED"); }
    }
    cudaFreeHost(send_low); cudaFreeHost(send_high); cudaFreeHost(recv_low); cudaFreeHost(recv_high);
    cudaFree(d_a); cudaFree(d_b); MPI_Finalize();
    return 0;
}
