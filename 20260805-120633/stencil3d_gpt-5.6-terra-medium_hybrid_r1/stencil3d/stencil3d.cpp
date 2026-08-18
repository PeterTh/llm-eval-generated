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

#define CUDA_CHECK(call) do { \
    const cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error_)); \
    } \
} while (0)

__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              int nx, int ny, int localNz, int globalZ0, int globalNz,
                              int localZBegin, int localZEnd) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int localZ = localZBegin + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || localZ >= localZEnd) return;

    const size_t plane = static_cast<size_t>(nx) * ny;
    const size_t index = static_cast<size_t>(localZ + 1) * plane + static_cast<size_t>(y) * nx + x;
    const int globalZ = globalZ0 + localZ;
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || globalZ == 0 || globalZ == globalNz - 1) {
        output[index] = input[index];
    } else {
        output[index] = (input[index] + input[index - 1] + input[index + 1] +
                         input[index - nx] + input[index + nx] + input[index - plane] + input[index + plane]) / 7.0;
    }
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -x <num>     Grid size in X dimension (default: 128)\n"
                "  -y <num>     Grid size in Y dimension (default: same as X)\n"
                "  -z <num>     Grid size in Z dimension (default: same as X)\n"
                "  -i <num>     Number of iterations (default: 10)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    long long nxArg = 128, nyArg = 0, nzArg = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nxArg = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) nyArg = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nzArg = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (nyArg == 0) nyArg = nxArg;
    if (nzArg == 0) nzArg = nxArg;
    if (nxArg < 1 || nyArg < 1 || nzArg < 1 || iterations < 0 || nxArg > std::numeric_limits<int>::max() || nyArg > std::numeric_limits<int>::max() || nzArg > std::numeric_limits<int>::max()) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions must be positive 32-bit integers and iterations non-negative.\n");
        MPI_Finalize(); return 1;
    }
    const int nx = static_cast<int>(nxArg), ny = static_cast<int>(nyArg), nz = static_cast<int>(nzArg);
    const size_t plane = static_cast<size_t>(nx) * ny;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "A grid plane is too large for MPI message counts.\n");
        MPI_Finalize(); return 1;
    }

    if (provided < MPI_THREAD_FUNNELED) { if (rank == 0) std::fprintf(stderr, "MPI lacks required thread support.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    int localRank = 0;
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    const int activeRanks = std::min(ranks, nz);
    const int localNz = rank < activeRanks ? nz / activeRanks + (rank < nz % activeRanks ? 1 : 0) : 0;
    const int globalZ0 = rank < activeRanks ? rank * (nz / activeRanks) + std::min(rank, nz % activeRanks) : nz;
    const int previous = (rank > 0 && rank < activeRanks) ? rank - 1 : MPI_PROC_NULL;
    const int next = (rank + 1 < activeRanks) ? rank + 1 : MPI_PROC_NULL;
    const size_t localElements = plane * static_cast<size_t>(localNz + 2);
    std::vector<Real> hostA(localElements), hostB(localElements), sendLower(plane), sendUpper(plane), recvLower(plane), recvUpper(plane);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < localNz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const size_t localIndex = static_cast<size_t>(z + 1) * plane + static_cast<size_t>(y) * nx + x;
                const size_t globalIndex = static_cast<size_t>(globalZ0 + z) * plane + static_cast<size_t>(y) * nx + x;
                hostA[localIndex] = static_cast<Real>(globalIndex % 19);
            }

    Real *deviceA = nullptr, *deviceB = nullptr;
    if (localNz > 0) {
        CUDA_CHECK(cudaMalloc(&deviceA, localElements * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&deviceB, localElements * sizeof(Real)));
        CUDA_CHECK(cudaMemcpy(deviceA, hostA.data(), localElements * sizeof(Real), cudaMemcpyHostToDevice));
    }
    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %d x %d x %d\nIterations: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, CUDA devices/node: %d, OpenMP threads/rank: %d\nInitializing grid...\nRunning stencil computation...\n", ranks, deviceCount, omp_get_max_threads());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const dim3 block(8, 8, 4);
    const auto launchStencil = [&](int begin, int end) {
        const dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y,
                        (end - begin + block.z - 1) / block.z);
        stencilKernel<<<grid, block>>>(deviceA, deviceB, nx, ny, localNz, globalZ0, nz, begin, end);
        CUDA_CHECK(cudaGetLastError());
    };
    for (int iter = 0; iter < iterations; ++iter) {
        if (localNz == 0) continue;
        CUDA_CHECK(cudaMemcpy(sendLower.data(), deviceA + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sendUpper.data(), deviceA + static_cast<size_t>(localNz) * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
        MPI_Request requests[4]; int requestCount = 0;
        if (previous != MPI_PROC_NULL) MPI_Irecv(recvLower.data(), static_cast<int>(plane), MPI_DOUBLE, previous, 2, MPI_COMM_WORLD, &requests[requestCount++]);
        if (next != MPI_PROC_NULL) MPI_Irecv(recvUpper.data(), static_cast<int>(plane), MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &requests[requestCount++]);
        if (previous != MPI_PROC_NULL) MPI_Isend(sendLower.data(), static_cast<int>(plane), MPI_DOUBLE, previous, 1, MPI_COMM_WORLD, &requests[requestCount++]);
        if (next != MPI_PROC_NULL) MPI_Isend(sendUpper.data(), static_cast<int>(plane), MPI_DOUBLE, next, 2, MPI_COMM_WORLD, &requests[requestCount++]);
        // These planes only depend on locally owned data, so the GPU can work while MPI moves halos.
        if (localNz > 2) {
            launchStencil(1, localNz - 1);
        }
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
        if (previous != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(deviceA, recvLower.data(), plane * sizeof(Real), cudaMemcpyHostToDevice));
        if (next != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(deviceA + static_cast<size_t>(localNz + 1) * plane, recvUpper.data(), plane * sizeof(Real), cudaMemcpyHostToDevice));
        launchStencil(0, 1);
        if (localNz > 1) launchStencil(localNz - 1, localNz);
        std::swap(deviceA, deviceB);
    }
    if (localNz > 0) CUDA_CHECK(cudaMemcpy(hostA.data() + plane, deviceA + plane, static_cast<size_t>(localNz) * plane * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        const double updates = static_cast<double>(std::max(0, nx - 2)) * std::max(0, ny - 2) * std::max(0, nz - 2) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", seconds > 0.0 ? updates / seconds / 1e6 : 0.0);
    }

    std::vector<int> counts, displacements;
    std::vector<Real> finalGrid;
    // MPI_Gather requires a valid root receive buffer even when no full-grid output is requested.
    if (rank == 0) counts.resize(ranks);
    const int localCount = static_cast<int>(static_cast<size_t>(localNz) * plane);
    MPI_Gather(&localCount, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0 && (printResults || validate)) {
        displacements.resize(ranks);
        int offset = 0; for (int r = 0; r < ranks; ++r) { displacements[r] = offset; offset += counts[r]; }
        finalGrid.resize(static_cast<size_t>(nx) * ny * nz);
    }
    if (printResults || validate) MPI_Gatherv(hostA.data() + plane, localCount, MPI_DOUBLE, rank == 0 ? finalGrid.data() : nullptr,
                                               rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int exitCode = 0;
    if (rank == 0 && printResults) print_results(finalGrid, "Grid");
    if (rank == 0 && validate) {
        bool valid = true; Real minVal = finalGrid[0], maxVal = finalGrid[0];
        #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(&:valid)
        for (size_t i = 0; i < finalGrid.size(); ++i) { valid = valid && std::isfinite(finalGrid[i]); minVal = std::min(minVal, finalGrid[i]); maxVal = std::max(maxVal, finalGrid[i]); }
        std::printf("Validating result...\nValue range: [%.6f, %.6f]\n", minVal, maxVal);
        valid = valid && maxVal <= 1e6 && minVal >= -1e6;
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); exitCode = valid ? 0 : 1;
    }
    if (deviceA) cudaFree(deviceA);
    if (deviceB) cudaFree(deviceB);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
