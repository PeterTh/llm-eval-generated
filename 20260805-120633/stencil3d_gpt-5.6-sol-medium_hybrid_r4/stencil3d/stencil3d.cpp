#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error_ = (call);                                      \
        if (error_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(error_));                  \
            MPI_Abort(MPI_COMM_WORLD, 2);                                       \
        }                                                                       \
    } while (false)

// zFirst and zLast are inclusive local slab indices.  Index zero and
// localNz+1 are the MPI ghost planes.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output, size_t nx, size_t ny,
                              size_t zFirst, size_t zLast) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z + zFirst;
    if (x >= nx - 1 || y >= ny - 1 || z > zLast) return;

    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    output[i] = (input[i] + input[i - 1] + input[i + 1] + input[i - nx] +
                 input[i + nx] + input[i - plane] + input[i + plane]) /
                Real(7.0);
}

static void launchStencil(const Real* input, Real* output, size_t nx, size_t ny,
                          size_t zFirst, size_t zLast, cudaStream_t stream) {
    if (zFirst > zLast) return;
    const dim3 block(32, 4, 2);
    const dim3 grid(static_cast<unsigned>((nx - 2 + block.x - 1) / block.x),
                    static_cast<unsigned>((ny - 2 + block.y - 1) / block.y),
                    static_cast<unsigned>((zLast - zFirst + 1 + block.z - 1) /
                                          block.z));
    stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, zFirst, zLast);
    CUDA_CHECK(cudaGetLastError());
}

static void initializeLocal(std::vector<Real>& grid, size_t nx, size_t ny,
                            size_t localNz, size_t globalZ0) {
    const size_t plane = nx * ny;
    const size_t count = (localNz + 2) * plane;
#pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(count); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        const size_t zLocal = i / plane;
        const size_t inPlane = i - zLocal * plane;
        const size_t globalZ = globalZ0 + zLocal - 1;
        const size_t globalIndex = globalZ * plane + inPlane;
        grid[i] = static_cast<Real>(globalIndex % 19);
    }
}

static bool validateResult(const std::vector<Real>& grid) {
    int invalid = 0;
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for reduction(| : invalid) reduction(min : minVal) reduction(max : maxVal)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        const Real value = grid[static_cast<size_t>(i)];
        invalid |= !std::isfinite(value);
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

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide required threaded-process support.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

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
        else { if (rank == 0) std::printf("Unknown option: %s\n", argv[i]); badArgs = true; }
    }
    if (help || badArgs) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return badArgs ? 1 : 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0 ||
        static_cast<size_t>(ranks) > nz - 2) {
        if (rank == 0)
            std::fprintf(stderr, "Grid dimensions must be at least 3, iterations nonnegative, "
                                 "and MPI ranks no greater than z-2.\n");
        MPI_Finalize();
        return 1;
    }

    // Assign ranks on the same node round-robin to the GPUs on that node.
    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank = 0;
    MPI_Comm_rank(shared, &localRank);
    MPI_Comm_free(&shared);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    bool cudaAwareMpi = false;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    cudaAwareMpi = MPIX_Query_cuda_support() != 0;
#endif

    const size_t interiorNz = nz - 2;
    const size_t base = interiorNz / static_cast<size_t>(ranks);
    const size_t remainder = interiorNz % static_cast<size_t>(ranks);
    const size_t localNz = base + (static_cast<size_t>(rank) < remainder);
    const size_t globalZ0 = 1 + static_cast<size_t>(rank) * base +
                            std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny;
    if (plane > static_cast<size_t>(INT_MAX) || localNz * plane > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) std::fprintf(stderr, "MPI message exceeds the implementation's count limit.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    const size_t localCount = (localNz + 2) * plane;
    std::vector<Real> hostLocal(localCount);
    initializeLocal(hostLocal, nx, ny, localNz, globalZ0);

    Real *grid1 = nullptr, *grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&grid1, localCount * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&grid2, localCount * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(grid1, hostLocal.data(), localCount * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grid2, hostLocal.data(), localCount * sizeof(Real), cudaMemcpyHostToDevice));
    cudaStream_t compute;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    Real *sendPrevious = nullptr, *sendNext = nullptr;
    Real *recvPrevious = nullptr, *recvNext = nullptr;
    if (!cudaAwareMpi) {
        CUDA_CHECK(cudaMallocHost(&sendPrevious, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&sendNext, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&recvPrevious, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&recvNext, plane * sizeof(Real)));
    }

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\n", nx, ny, nz, iterations);
        std::printf("Validation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n", validate ? "enabled" : "disabled", ranks, omp_get_max_threads());
        std::printf("Halo transport: %s\n", cudaAwareMpi ? "CUDA-aware MPI" : "pinned host staging");
        std::printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    Real* input = grid1;
    Real* output = grid2;
    const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        CUDA_CHECK(cudaStreamSynchronize(compute));
        MPI_Request requests[4];
        if (cudaAwareMpi) {
            // CUDA-aware MPI transfers contiguous device-resident Z planes directly.
            MPI_Irecv(input, static_cast<int>(plane), MPI_DOUBLE, previous, 101, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(input + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 102, MPI_COMM_WORLD, &requests[1]);
            MPI_Isend(input + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, next, 101, MPI_COMM_WORLD, &requests[2]);
            MPI_Isend(input + plane, static_cast<int>(plane), MPI_DOUBLE, previous, 102, MPI_COMM_WORLD, &requests[3]);
        } else {
            // Reusable page-locked buffers retain portability with non-CUDA-aware MPI.
            if (previous != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpy(sendPrevious, input + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
            if (next != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpy(sendNext, input + localNz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
            MPI_Irecv(recvPrevious, static_cast<int>(plane), MPI_DOUBLE, previous, 101, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(recvNext, static_cast<int>(plane), MPI_DOUBLE, next, 102, MPI_COMM_WORLD, &requests[1]);
            MPI_Isend(sendNext, static_cast<int>(plane), MPI_DOUBLE, next, 101, MPI_COMM_WORLD, &requests[2]);
            MPI_Isend(sendPrevious, static_cast<int>(plane), MPI_DOUBLE, previous, 102, MPI_COMM_WORLD, &requests[3]);
        }

        // The inner planes do not depend on incoming halos and overlap communication.
        launchStencil(input, output, nx, ny, 2, localNz > 1 ? localNz - 1 : 0, compute);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        if (!cudaAwareMpi) {
            if (previous != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(input, recvPrevious, plane * sizeof(Real), cudaMemcpyHostToDevice, compute));
            if (next != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(input + (localNz + 1) * plane, recvNext, plane * sizeof(Real), cudaMemcpyHostToDevice, compute));
        }
        launchStencil(input, output, nx, ny, 1, 1, compute);
        if (localNz > 1) launchStencil(input, output, nx, ny, localNz, localNz, compute);
        std::swap(input, output);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double updates = static_cast<double>(nx - 2) * static_cast<double>(ny - 2) *
                               static_cast<double>(nz - 2) * iterations;
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", seconds > 0.0 ? updates / seconds / 1e6 : 0.0);
    }

    int status = 0;
    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(hostLocal.data(), input, localCount * sizeof(Real), cudaMemcpyDeviceToHost));
        std::vector<int> counts, displacements;
        std::vector<Real> finalGrid;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            finalGrid.resize(nx * ny * nz);
#pragma omp parallel for schedule(static)
            for (long long i = 0; i < static_cast<long long>(finalGrid.size()); ++i)
                finalGrid[static_cast<size_t>(i)] = static_cast<Real>(static_cast<size_t>(i) % 19);
            for (int r = 0; r < ranks; ++r) {
                const size_t rn = base + (static_cast<size_t>(r) < remainder);
                const size_t rz = 1 + static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder);
                counts[r] = static_cast<int>(rn * plane);
                displacements[r] = static_cast<int>(rz * plane);
            }
        }
        MPI_Gatherv(hostLocal.data() + plane, static_cast<int>(localNz * plane), MPI_DOUBLE,
                    rank == 0 ? finalGrid.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (printResults) print_results(finalGrid, "Grid");
            if (validate) {
                std::printf("Validating result...\n");
                const bool valid = validateResult(finalGrid);
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!cudaAwareMpi) {
        CUDA_CHECK(cudaFreeHost(sendPrevious));
        CUDA_CHECK(cudaFreeHost(sendNext));
        CUDA_CHECK(cudaFreeHost(recvPrevious));
        CUDA_CHECK(cudaFreeHost(recvNext));
    }
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaFree(grid1));
    CUDA_CHECK(cudaFree(grid2));
    MPI_Finalize();
    return status;
}
