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

static void checkCuda(cudaError_t status, const char* what, MPI_Comm comm) {
    if (status != cudaSuccess) {
        int rank = 0; MPI_Comm_rank(comm, &rank);
        std::fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, what, cudaGetErrorString(status));
        MPI_Abort(comm, 2);
    }
}

__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              size_t nx, size_t ny, size_t localNz, size_t globalZ0, size_t globalNz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1; // skip lower halo
    if (x >= nx || y >= ny || z > localNz) return;
    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    const size_t gz = globalZ0 + z - 1;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || gz == 0 || gz + 1 == globalNz)
        output[i] = input[i];
    else
        output[i] = (input[i] + input[i - 1] + input[i + 1] + input[i - nx] + input[i + nx] +
                     input[i - plane] + input[i + plane]) * (1.0 / 7.0);
}

static void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny, size_t localNz, size_t globalZ0) {
    const size_t plane = nx * ny;
    #pragma omp parallel for schedule(static)
    for (long long lz = 0; lz < static_cast<long long>(localNz); ++lz) {
        const size_t globalBase = (globalZ0 + static_cast<size_t>(lz)) * plane;
        const size_t localBase = (static_cast<size_t>(lz) + 1) * plane;
        for (size_t p = 0; p < plane; ++p) grid[localBase + p] = static_cast<Real>((globalBase + p) % 19);
    }
}

static bool validateResult(const std::vector<Real>& grid) {
    Real lo = std::numeric_limits<Real>::max(), hi = std::numeric_limits<Real>::lowest();
    int bad = 0;
    #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:bad)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        const Real v = grid[static_cast<size_t>(i)];
        if (!std::isfinite(v)) bad = 1;
        lo = std::min(lo, v); hi = std::max(hi, v);
    }
    if (bad) { std::printf("Validation failed: found NaN or Inf value\n"); return false; }
    std::printf("Value range: [%.6f, %.6f]\n", lo, hi);
    if (hi > 1e6 || lo < -1e6) { std::printf("Validation failed: values out of expected range\n"); return false; }
    return true;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -x <num> Grid X (default 128)\n  -y <num> Grid Y (default X)\n"
                "  -z <num> Grid Z (default X)\n  -i <num> Iterations (default 10)\n  -v Validate\n  -r Print results\n  -h Help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, ranks; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &ranks);
    size_t nx = 128, ny = 0, nz = 0; int iterations = 10; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0 || static_cast<size_t>(ranks) > nz) {
        if (!rank) std::fprintf(stderr, "Grid dimensions must be at least 3, iterations non-negative, and ranks <= Z dimension.\n");
        MPI_Abort(comm, 1);
    }
    int devices = 0; checkCuda(cudaGetDeviceCount(&devices), "device discovery", comm);
    if (!devices) { if (!rank) std::fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(comm, 2); }
    checkCuda(cudaSetDevice(rank % devices), "device selection", comm);

    const size_t base = nz / ranks, rem = nz % ranks;
    const size_t localNz = base + (static_cast<size_t>(rank) < rem);
    const size_t globalZ0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t plane = nx * ny, localCount = localNz * plane, allocCount = (localNz + 2) * plane;
    if (localCount > static_cast<size_t>(std::numeric_limits<int>::max())) { if (!rank) std::fprintf(stderr, "Local MPI message exceeds INT_MAX.\n"); MPI_Abort(comm, 1); }
    std::vector<Real> host(allocCount, 0.0); initializeGrid(host, nx, ny, localNz, globalZ0);
    Real *a = nullptr, *b = nullptr;
    checkCuda(cudaMalloc(&a, allocCount * sizeof(Real)), "allocation", comm);
    checkCuda(cudaMalloc(&b, allocCount * sizeof(Real)), "allocation", comm);
    // Pinned buffers make the portable MPI staging path fast and work with MPI stacks
    // that are not built with CUDA-aware transport support.
    Real *sendLower = nullptr, *sendUpper = nullptr, *recvLower = nullptr, *recvUpper = nullptr;
    checkCuda(cudaMallocHost(&sendLower, plane * sizeof(Real)), "pinned allocation", comm);
    checkCuda(cudaMallocHost(&sendUpper, plane * sizeof(Real)), "pinned allocation", comm);
    checkCuda(cudaMallocHost(&recvLower, plane * sizeof(Real)), "pinned allocation", comm);
    checkCuda(cudaMallocHost(&recvUpper, plane * sizeof(Real)), "pinned allocation", comm);
    checkCuda(cudaMemcpy(a, host.data(), allocCount * sizeof(Real), cudaMemcpyHostToDevice), "initial copy", comm);
    checkCuda(cudaMemcpy(b, a, allocCount * sizeof(Real), cudaMemcpyDeviceToDevice), "initial copy", comm);

    if (!rank) { std::printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\nInitializing grid...\nRunning stencil computation...\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled"); }
    const int below = rank ? rank - 1 : MPI_PROC_NULL, above = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    const dim3 threads(32, 4, 2), blocks((nx + threads.x - 1) / threads.x, (ny + threads.y - 1) / threads.y, (localNz + threads.z - 1) / threads.z);
    MPI_Barrier(comm); auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        checkCuda(cudaMemcpy(sendLower, a + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost), "lower halo copy", comm);
        checkCuda(cudaMemcpy(sendUpper, a + localNz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost), "upper halo copy", comm);
        MPI_Sendrecv(sendLower, static_cast<int>(plane), MPI_DOUBLE, below, 10, recvUpper, static_cast<int>(plane), MPI_DOUBLE, above, 10, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(sendUpper, static_cast<int>(plane), MPI_DOUBLE, above, 11, recvLower, static_cast<int>(plane), MPI_DOUBLE, below, 11, comm, MPI_STATUS_IGNORE);
        if (below != MPI_PROC_NULL) checkCuda(cudaMemcpy(a, recvLower, plane * sizeof(Real), cudaMemcpyHostToDevice), "lower halo upload", comm);
        if (above != MPI_PROC_NULL) checkCuda(cudaMemcpy(a + (localNz + 1) * plane, recvUpper, plane * sizeof(Real), cudaMemcpyHostToDevice), "upper halo upload", comm);
        stencilKernel<<<blocks, threads>>>(a, b, nx, ny, localNz, globalZ0, nz);
        checkCuda(cudaGetLastError(), "kernel launch", comm);
        std::swap(a, b);
    }
    checkCuda(cudaDeviceSynchronize(), "kernel synchronization", comm);
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count(), maxSeconds = 0;
    MPI_Reduce(&seconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    std::vector<Real> local(localCount); checkCuda(cudaMemcpy(local.data(), a + plane, localCount * sizeof(Real), cudaMemcpyDeviceToHost), "result copy", comm);
    std::vector<int> counts, displs; std::vector<Real> finalGrid;
    if (!rank) { counts.resize(ranks); displs.resize(ranks); for (int r = 0; r < ranks; ++r) { size_t n = base + (static_cast<size_t>(r) < rem); counts[r] = static_cast<int>(n * plane); displs[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem)) * plane); } finalGrid.resize(nx * ny * nz); }
    MPI_Gatherv(local.data(), static_cast<int>(localCount), MPI_DOUBLE, rank ? nullptr : finalGrid.data(), rank ? nullptr : counts.data(), rank ? nullptr : displs.data(), MPI_DOUBLE, 0, comm);
    int exitCode = 0;
    if (!rank) {
        std::printf("Computation time: %.3f ms\n", maxSeconds * 1000.0);
        const double updates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", updates / maxSeconds / 1e6);
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(finalGrid);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, comm);
    checkCuda(cudaFreeHost(sendLower), "pinned free", comm); checkCuda(cudaFreeHost(sendUpper), "pinned free", comm);
    checkCuda(cudaFreeHost(recvLower), "pinned free", comm); checkCuda(cudaFreeHost(recvUpper), "pinned free", comm);
    checkCuda(cudaFree(a), "free", comm); checkCuda(cudaFree(b), "free", comm);
    MPI_Finalize();
    return exitCode;
}
