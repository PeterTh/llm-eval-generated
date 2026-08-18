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

static void checkCuda(cudaError_t status, const char* what, MPI_Comm comm) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        std::fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, what,
                     cudaGetErrorString(status));
        MPI_Abort(comm, 2);
    }
}

struct DeviceGrid {
    Real* data = nullptr;
    size_t elements = 0;

    DeviceGrid() = default;
    DeviceGrid(const DeviceGrid&) = delete;
    DeviceGrid& operator=(const DeviceGrid&) = delete;
    ~DeviceGrid() { if (data != nullptr) cudaFree(data); }
};

// Reused pinned buffers avoid pageable-memory copies and per-timestep allocation
// on the critical halo-exchange path.
struct HaloBuffers {
    Real *sendLow = nullptr, *sendHigh = nullptr, *recvLow = nullptr, *recvHigh = nullptr;

    HaloBuffers(size_t plane, MPI_Comm comm) {
        const size_t bytes = plane * sizeof(Real);
        checkCuda(cudaHostAlloc(&sendLow, bytes, cudaHostAllocDefault), "allocate lower send halo", comm);
        checkCuda(cudaHostAlloc(&sendHigh, bytes, cudaHostAllocDefault), "allocate upper send halo", comm);
        checkCuda(cudaHostAlloc(&recvLow, bytes, cudaHostAllocDefault), "allocate lower receive halo", comm);
        checkCuda(cudaHostAlloc(&recvHigh, bytes, cudaHostAllocDefault), "allocate upper receive halo", comm);
    }
    HaloBuffers(const HaloBuffers&) = delete;
    HaloBuffers& operator=(const HaloBuffers&) = delete;
    ~HaloBuffers() {
        if (sendLow != nullptr) cudaFreeHost(sendLow);
        if (sendHigh != nullptr) cudaFreeHost(sendHigh);
        if (recvLow != nullptr) cudaFreeHost(recvLow);
        if (recvHigh != nullptr) cudaFreeHost(recvHigh);
    }
};

// localNz physical planes are bracketed by one halo plane at each end.
// globalZ0 is the global z coordinate of local plane 1.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t localNz,
                              size_t globalZ0, size_t globalNz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t localZ = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || localZ > localNz) return;

    const size_t plane = nx * ny;
    const size_t i = localZ * plane + y * nx + x;
    const size_t globalZ = globalZ0 + localZ - 1;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
        globalZ == 0 || globalZ + 1 == globalNz) {
        output[i] = input[i];
    } else {
        output[i] = (input[i] + input[i - 1] + input[i + 1] +
                     input[i - nx] + input[i + nx] +
                     input[i - plane] + input[i + plane]) / 7.0;
    }
}

static void initializeLocalGrid(std::vector<Real>& host, size_t nx, size_t ny,
                                size_t localNz, size_t globalZ0) {
    const size_t plane = nx * ny;
    #pragma omp parallel for schedule(static)
    for (long long localZ = 0; localZ < static_cast<long long>(localNz); ++localZ) {
        const size_t z = static_cast<size_t>(localZ);
        const size_t globalZ = globalZ0 + z;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIndex = idx3(x, y, globalZ, nx, ny);
                host[(z + 1) * plane + y * nx + x] = static_cast<Real>(globalIndex % 19);
            }
        }
    }
}

static void exchangeHalos(DeviceGrid& grid, size_t nx, size_t ny, size_t localNz,
                          int previous, int next, HaloBuffers& buffers, MPI_Comm comm) {
    const size_t plane = nx * ny;
    const size_t bytes = plane * sizeof(Real);
    MPI_Request requests[4];
    int count = 0;

    // Staging keeps this correct on both CUDA-aware and conventional MPI stacks.
    if (previous != MPI_PROC_NULL) {
        checkCuda(cudaMemcpy(buffers.sendLow, grid.data + plane, bytes,
                             cudaMemcpyDeviceToHost), "copy lower halo send", comm);
        MPI_Irecv(buffers.recvLow, static_cast<int>(plane), MPI_DOUBLE, previous, 1, comm,
                  &requests[count++]);
        MPI_Isend(buffers.sendLow, static_cast<int>(plane), MPI_DOUBLE, previous, 0, comm,
                  &requests[count++]);
    }
    if (next != MPI_PROC_NULL) {
        checkCuda(cudaMemcpy(buffers.sendHigh, grid.data + localNz * plane, bytes,
                             cudaMemcpyDeviceToHost), "copy upper halo send", comm);
        MPI_Irecv(buffers.recvHigh, static_cast<int>(plane), MPI_DOUBLE, next, 0, comm,
                  &requests[count++]);
        MPI_Isend(buffers.sendHigh, static_cast<int>(plane), MPI_DOUBLE, next, 1, comm,
                  &requests[count++]);
    }
    if (count != 0) MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
    if (previous != MPI_PROC_NULL)
        checkCuda(cudaMemcpy(grid.data, buffers.recvLow, bytes, cudaMemcpyHostToDevice),
                  "copy lower halo receive", comm);
    if (next != MPI_PROC_NULL)
        checkCuda(cudaMemcpy(grid.data + (localNz + 1) * plane, buffers.recvHigh, bytes,
                             cudaMemcpyHostToDevice), "copy upper halo receive", comm);
}

static bool validateResult(const std::vector<Real>& grid) {
    Real minVal = std::numeric_limits<Real>::max();
    Real maxVal = std::numeric_limits<Real>::lowest();
    int invalid = 0;
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(|:invalid)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        const Real value = grid[static_cast<size_t>(i)];
        if (!std::isfinite(value)) invalid = 1;
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }
    if (invalid) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false, badArgs = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else badArgs = true;
    }
    if (help || badArgs) {
        if (rank == 0) {
            if (badArgs) std::printf("Invalid option or missing option value\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return badArgs ? 1 : 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || ranks > static_cast<int>(nz)) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions must be positive, iterations non-negative, and MPI ranks no greater than z.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    checkCuda(cudaGetDeviceCount(&deviceCount), "get device count", MPI_COMM_WORLD);
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device is available.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), "select device", MPI_COMM_WORLD);
    MPI_Comm_free(&nodeComm);

    const size_t baseNz = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t globalZ0 = static_cast<size_t>(rank) * baseNz +
                            std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny;
    const size_t localElements = (localNz + 2) * plane;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        localNz > static_cast<size_t>(std::numeric_limits<int>::max()) / plane ||
        nz > static_cast<size_t>(std::numeric_limits<int>::max()) / plane) {
        if (rank == 0) std::fprintf(stderr, "A local grid or gathered grid is too large for this MPI implementation.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, CUDA devices assigned per node, OpenMP host workers: %d\n",
                    ranks, omp_get_max_threads());
        std::printf("Initializing grid...\nRunning stencil computation...\n");
    }

    std::vector<Real> hostLocal(localElements, 0.0);
    initializeLocalGrid(hostLocal, nx, ny, localNz, globalZ0);
    DeviceGrid grids[2];
    for (auto& grid : grids) {
        grid.elements = localElements;
        checkCuda(cudaMalloc(&grid.data, localElements * sizeof(Real)), "allocate grid", MPI_COMM_WORLD);
    }
    checkCuda(cudaMemcpy(grids[0].data, hostLocal.data(), localElements * sizeof(Real),
                         cudaMemcpyHostToDevice), "upload initial grid", MPI_COMM_WORLD);
    const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    HaloBuffers halos(plane, MPI_COMM_WORLD);
    exchangeHalos(grids[0], nx, ny, localNz, previous, next, halos, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    const dim3 block(32, 4, 2);
    const dim3 launch((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y,
                      (localNz + block.z - 1) / block.z);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int in = iteration & 1;
        const int out = in ^ 1;
        stencilKernel<<<launch, block>>>(grids[in].data, grids[out].data, nx, ny, localNz, globalZ0, nz);
        checkCuda(cudaGetLastError(), "launch stencil kernel", MPI_COMM_WORLD);
        exchangeHalos(grids[out], nx, ny, localNz, previous, next, halos, MPI_COMM_WORLD);
    }
    checkCuda(cudaDeviceSynchronize(), "synchronize stencil computation", MPI_COMM_WORLD);
    const double localSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const int finalIndex = iterations & 1;
    std::vector<Real> localPhysical(localNz * plane);
    checkCuda(cudaMemcpy(localPhysical.data(), grids[finalIndex].data + plane,
                         localPhysical.size() * sizeof(Real), cudaMemcpyDeviceToHost),
              "download final grid", MPI_COMM_WORLD);
    std::vector<int> recvCounts, displacements;
    std::vector<Real> finalGrid;
    if (rank == 0) {
        recvCounts.resize(ranks);
        displacements.resize(ranks);
        for (int r = 0; r < ranks; ++r) {
            const size_t countZ = baseNz + (static_cast<size_t>(r) < remainder ? 1 : 0);
            recvCounts[r] = static_cast<int>(countZ * plane);
            displacements[r] = static_cast<int>((static_cast<size_t>(r) * baseNz +
                                std::min(static_cast<size_t>(r), remainder)) * plane);
        }
        finalGrid.resize(nx * ny * nz);
    }
    MPI_Gatherv(localPhysical.data(), static_cast<int>(localPhysical.size()), MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr,
                rank == 0 ? recvCounts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int result = 0;
    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsedSeconds * 1000.0);
        std::printf("Computation time: %ld ms\n", milliseconds);
        const double updates = static_cast<double>((nx > 2 ? nx - 2 : 0) * (ny > 2 ? ny - 2 : 0) *
                                                   (nz > 2 ? nz - 2 : 0)) * iterations;
        const double mcups = elapsedSeconds > 0.0 ? updates / elapsedSeconds / 1.0e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) {
            std::printf("Validating result...\n");
            result = validateResult(finalGrid) ? 0 : 1;
            std::printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
