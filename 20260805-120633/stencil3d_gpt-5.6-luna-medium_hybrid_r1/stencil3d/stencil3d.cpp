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

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

#define CUDA_CHECK(call) do { \
    const cudaError_t error__ = (call); \
    if (error__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error__)); \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error__)); \
    } \
} while (false)

// One MPI rank owns a contiguous set of Z planes. The two extra planes are
// host/device halos; the physical boundary planes remain locally owned.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t ownedNz,
                              size_t globalStart, size_t globalNz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t localZ = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || localZ > ownedNz) return;

    const size_t globalZ = globalStart + localZ - 1;
    const size_t out = idx3(x, y, localZ, nx, ny);
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
        globalZ == 0 || globalZ + 1 == globalNz) {
        output[out] = input[out];
        return;
    }

    const size_t plane = nx * ny;
    output[out] = (input[out] + input[out - 1] + input[out + 1] +
                   input[out - nx] + input[out + nx] +
                   input[out - plane] + input[out + plane]) / 7.0;
}

void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny,
                    size_t ownedNz, size_t globalStart) {
    const size_t plane = nx * ny;
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long z = 0; z < static_cast<long long>(ownedNz + 2); ++z) {
        for (long long y = 0; y < static_cast<long long>(ny); ++y) {
            const size_t globalZ = globalStart + static_cast<size_t>(z) - 1;
            const size_t row = static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx;
            for (size_t x = 0; x < nx; ++x)
                grid[row + x] = static_cast<Real>((globalZ * plane + static_cast<size_t>(y) * nx + x) % 19);
        }
    }
}

bool validateResult(const std::vector<Real>& grid) {
    if (grid.empty()) return true;
    Real minVal = grid[0], maxVal = grid[0];
    bool finite = true;
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(&:finite)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        const Real value = grid[static_cast<size_t>(i)];
        finite = finite && std::isfinite(value);
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }
    if (!finite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    return maxVal <= 1e6 && minVal >= -1e6;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("  -x <num> Grid size in X (default: 128)\n");
    printf("  -y <num> Grid size in Y (default: X)\n");
    printf("  -z <num> Grid size in Z (default: X)\n");
    printf("  -i <num> Number of iterations (default: 10)\n");
    printf("  -v       Enable validation\n  -r       Print results\n  -h       Show this help\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);

    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (worldRank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (worldRank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0) MPI_Abort(MPI_COMM_WORLD, 2);

    // Extra ranks are safely made idle while the active communicator remains
    // a contiguous rank set, which keeps the slab layout and gather simple.
    const int activeRanks = std::min(worldSize, static_cast<int>(nz));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (worldRank >= activeRanks) { MPI_Finalize(); return 0; }
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const size_t base = nz / static_cast<size_t>(size);
    const size_t remainder = nz % static_cast<size_t>(size);
    const size_t ownedNz = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t globalStart = static_cast<size_t>(rank) * base +
                               std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny;
    const size_t localSize = (ownedNz + 2) * plane;
    std::vector<Real> host1(localSize), host2(localSize);
    initializeGrid(host1, nx, ny, ownedNz, globalStart);
    initializeGrid(host2, nx, ny, ownedNz, globalStart);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { fprintf(stderr, "No CUDA device available on MPI rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 3); }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));
    Real *device1 = nullptr, *device2 = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device1), localSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device2), localSize * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(device1, host1.data(), localSize * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device2, host2.data(), localSize * sizeof(Real), cudaMemcpyHostToDevice));

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI %d ranks, OpenMP %d threads, CUDA)\n", size, omp_get_max_threads());
        printf("Grid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }
    MPI_Barrier(comm);
    const auto startTime = std::chrono::high_resolution_clock::now();
    const dim3 block(32, 4, 1);
    const dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y,
                    (ownedNz + block.z - 1) / block.z);
    for (int iter = 0; iter < iterations; ++iter) {
        Real* input = (iter % 2 == 0) ? device1 : device2;
        Real* output = (iter % 2 == 0) ? device2 : device1;
        std::vector<Real>& hostInput = (iter % 2 == 0) ? host1 : host2;

        // Only the two faces participate in MPI. Keep the interior resident
        // on the GPU; this avoids a full-volume device/host transfer every
        // iteration.
        CUDA_CHECK(cudaMemcpy(hostInput.data() + plane, input + plane,
                              plane * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostInput.data() + ownedNz * plane,
                              input + ownedNz * plane,
                              plane * sizeof(Real), cudaMemcpyDeviceToHost));
        MPI_Request requests[4];
        int requestCount = 0;
        if (rank > 0) {
            MPI_Irecv(hostInput.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 11, comm, &requests[requestCount++]);
            MPI_Isend(hostInput.data() + plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 22, comm, &requests[requestCount++]);
        }
        if (rank + 1 < size) {
            MPI_Irecv(hostInput.data() + (ownedNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 22, comm, &requests[requestCount++]);
            MPI_Isend(hostInput.data() + ownedNz * plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 11, comm, &requests[requestCount++]);
        }
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
        if (rank > 0)
            CUDA_CHECK(cudaMemcpy(input, hostInput.data(), plane * sizeof(Real), cudaMemcpyHostToDevice));
        if (rank + 1 < size)
            CUDA_CHECK(cudaMemcpy(input + (ownedNz + 1) * plane,
                                  hostInput.data() + (ownedNz + 1) * plane,
                                  plane * sizeof(Real), cudaMemcpyHostToDevice));
        stencilKernel<<<grid, block>>>(input, output, nx, ny, ownedNz, globalStart, nz);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto endTime = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(endTime - startTime).count();
    double elapsed = 0.0;
    MPI_Reduce(&localSeconds, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    const int ownedCount = static_cast<int>(ownedNz * plane);
    std::vector<Real> finalGrid;
    std::vector<int> counts, displacements;
    if (rank == 0) {
        finalGrid.resize(nx * ny * nz);
        counts.resize(size); displacements.resize(size);
        for (int r = 0; r < size; ++r) {
            const size_t rOwned = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
            const size_t rStart = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder);
            counts[r] = static_cast<int>(rOwned * plane);
            displacements[r] = static_cast<int>(rStart * plane);
        }
    }
    Real* finalDevice = (iterations % 2 == 0) ? device1 : device2;
    std::vector<Real>& finalHost = (iterations % 2 == 0) ? host1 : host2;
    CUDA_CHECK(cudaMemcpy(finalHost.data() + plane, finalDevice + plane,
                          ownedNz * plane * sizeof(Real), cudaMemcpyDeviceToHost));
    MPI_Gatherv(finalHost.data() + plane, ownedCount, MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, comm);
    if (rank == 0) {
        printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", elapsed * 1000.0,
               ((nx - 2.0) * (ny - 2.0) * (nz - 2.0) * iterations) / elapsed / 1e6);
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(finalGrid);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            CUDA_CHECK(cudaFree(device1)); CUDA_CHECK(cudaFree(device2)); MPI_Comm_free(&comm); MPI_Finalize(); return valid ? 0 : 1;
        }
    }
    CUDA_CHECK(cudaFree(device1)); CUDA_CHECK(cudaFree(device2));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return 0;
}
