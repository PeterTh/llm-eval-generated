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

static void cudaCheck(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void initializeGrid(Real* grid, size_t plane, size_t nzLocal,
                               size_t globalZ0) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t n = plane * nzLocal;
    if (i < n) {
        const size_t localZ = i / plane;
        const size_t globalIndex = (globalZ0 + localZ) * plane + i % plane;
        grid[i + plane] = static_cast<Real>(globalIndex % 19);
    }
}

// Each MPI rank owns nzLocal global z-planes, with one halo plane on each side.
__global__ void stencilIteration(const Real* input, Real* output, size_t nx,
                                 size_t ny, size_t nzLocal, size_t globalZ0,
                                 size_t globalNz) {
    const size_t plane = nx * ny;
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t n = plane * nzLocal;
    if (i >= n) return;

    const size_t z = i / plane;
    const size_t rem = i - z * plane;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;
    const size_t p = i + plane;
    const size_t gz = globalZ0 + z;

    // This also preserves every physical boundary exactly as in the original.
    Real value = input[p];
    if (x != 0 && x + 1 != nx && y != 0 && y + 1 != ny && gz != 0 && gz + 1 != globalNz)
        value = (input[p] + input[p - 1] + input[p + 1] + input[p - nx] +
                 input[p + nx] + input[p - plane] + input[p + plane]) / 7.0;
    output[p] = value;
}

static void exchangeHalos(Real* d_grid, size_t plane, size_t nzLocal, int rank,
                          int ranks, std::vector<Real>& sendLo, std::vector<Real>& sendHi,
                          std::vector<Real>& recvLo, std::vector<Real>& recvHi) {
    const int lo = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int hi = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    cudaCheck(cudaMemcpy(sendLo.data(), d_grid + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost), "copy lower halo send");
    cudaCheck(cudaMemcpy(sendHi.data(), d_grid + nzLocal * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost), "copy upper halo send");
    MPI_Request requests[4];
    MPI_Irecv(recvLo.data(), static_cast<int>(plane), MPI_DOUBLE, lo, 11, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(recvHi.data(), static_cast<int>(plane), MPI_DOUBLE, hi, 10, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(sendLo.data(), static_cast<int>(plane), MPI_DOUBLE, lo, 10, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(sendHi.data(), static_cast<int>(plane), MPI_DOUBLE, hi, 11, MPI_COMM_WORLD, &requests[3]);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    if (lo != MPI_PROC_NULL) cudaCheck(cudaMemcpy(d_grid, recvLo.data(), plane * sizeof(Real), cudaMemcpyHostToDevice), "copy lower halo receive");
    if (hi != MPI_PROC_NULL) cudaCheck(cudaMemcpy(d_grid + (nzLocal + 1) * plane, recvHi.data(), plane * sizeof(Real), cudaMemcpyHostToDevice), "copy upper halo receive");
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -x <num>     Grid size in X dimension (default: 128)\n  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n  -i <num>     Number of iterations (default: 10)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
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
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (nx < 3 || ny < 3 || nz < static_cast<size_t>(ranks) || iterations < 0) {
        if (!rank) std::fprintf(stderr, "Grid dimensions must be at least 3 and z must be at least the MPI rank count.\n");
        MPI_Finalize(); return 1;
    }
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "query device count");
    if (deviceCount == 0) {
        if (!rank) std::fprintf(stderr, "No CUDA device is available.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaCheck(cudaSetDevice(rank % deviceCount), "select device");

    const size_t plane = nx * ny;
    const size_t base = nz / ranks, extra = nz % ranks;
    const size_t nzLocal = base + (static_cast<size_t>(rank) < extra);
    const size_t globalZ0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const size_t localElements = (nzLocal + 2) * plane;
    Real *d_a = nullptr, *d_b = nullptr;
    cudaCheck(cudaMalloc(&d_a, localElements * sizeof(Real)), "allocate first grid");
    cudaCheck(cudaMalloc(&d_b, localElements * sizeof(Real)), "allocate second grid");
    cudaCheck(cudaMemset(d_a, 0, localElements * sizeof(Real)), "clear first grid");
    cudaCheck(cudaMemset(d_b, 0, localElements * sizeof(Real)), "clear second grid");
    constexpr int threads = 256;
    const int blocks = static_cast<int>((plane * nzLocal + threads - 1) / threads);
    initializeGrid<<<blocks, threads>>>(d_a, plane, nzLocal, globalZ0);
    cudaCheck(cudaGetLastError(), "initialize kernel");

    std::vector<Real> sendLo(plane), sendHi(plane), recvLo(plane), recvHi(plane);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    Real *input = d_a, *output = d_b;
    for (int iter = 0; iter < iterations; ++iter) {
        exchangeHalos(input, plane, nzLocal, rank, ranks, sendLo, sendHi, recvLo, recvHi);
        stencilIteration<<<blocks, threads>>>(input, output, nx, ny, nzLocal, globalZ0, nz);
        cudaCheck(cudaGetLastError(), "stencil kernel");
        std::swap(input, output);
    }
    cudaCheck(cudaDeviceSynchronize(), "complete stencil kernels");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<Real> local(nzLocal * plane);
    if (validate || printResults)
        cudaCheck(cudaMemcpy(local.data(), input + plane, local.size() * sizeof(Real), cudaMemcpyDeviceToHost), "copy final grid");
    int validationFailed = 0;
    if (validate) {
        Real localMin = std::numeric_limits<Real>::infinity(), localMax = -localMin;
        int localBad = 0;
        #pragma omp parallel for reduction(min:localMin) reduction(max:localMax) reduction(|:localBad)
        for (size_t i = 0; i < local.size(); ++i) {
            const Real v = local[i];
            if (!std::isfinite(v)) localBad = 1;
            localMin = std::min(localMin, v); localMax = std::max(localMax, v);
        }
        Real minVal, maxVal; int bad;
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localBad, &bad, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
        if (!rank) {
            validationFailed = bad || maxVal > 1e6 || minVal < -1e6;
            std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
            std::printf("Validation: %s\n", validationFailed ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&validationFailed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (printResults) {
        std::vector<int> counts, displs;
        std::vector<Real> global;
        if (!rank) { counts.resize(ranks); displs.resize(ranks); for (int r = 0; r < ranks; ++r) { const size_t n = base + (static_cast<size_t>(r) < extra); counts[r] = static_cast<int>(n * plane); displs[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra)) * plane); } global.resize(nx * ny * nz); }
        MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, rank ? nullptr : global.data(), rank ? nullptr : counts.data(), rank ? nullptr : displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(global, "Grid");
    }
    if (!rank) {
        const double updates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        std::printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\nGrid size: %zu x %zu x %zu\nIterations: %d\nComputation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", nx, ny, nz, iterations, maxElapsed * 1000.0, maxElapsed > 0 ? updates / maxElapsed / 1e6 : 0.0);
    }
    cudaFree(d_a); cudaFree(d_b);
    MPI_Finalize();
    return validationFailed ? 1 : 0;
}
