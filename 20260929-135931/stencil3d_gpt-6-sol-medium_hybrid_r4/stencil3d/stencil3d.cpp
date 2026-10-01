#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
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
    cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)
#define MPI_CHECK(call) do { \
    int error_ = (call); \
    if (error_ != MPI_SUCCESS) { \
        char message_[MPI_MAX_ERROR_STRING]; int length_; \
        MPI_Error_string(error_, message_, &length_); \
        fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, length_, message_); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

inline size_t slabStart(int rank, size_t nz, int ranks) {
    return static_cast<size_t>(rank) * (nz / ranks) + std::min(static_cast<size_t>(rank), nz % ranks);
}
inline size_t slabLength(int rank, size_t nz, int ranks) {
    return nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
}

// Plane indices 1..localZ hold the owned cells; 0 and localZ+1 are halos.
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              size_t nx, size_t ny, size_t nz, size_t firstZ,
                              size_t localZ, size_t beginZ, size_t planes) {
    const size_t plane = nx * ny;
    const size_t total = plane * planes;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         t < total; t += stride) {
        const size_t local = beginZ + t / plane;
        const size_t offset = t % plane;
        const size_t x = offset % nx;
        const size_t y = offset / nx;
        const size_t p = local * plane + offset;
        const size_t globalZ = firstZ + local - 1;
        if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
            globalZ == 0 || globalZ + 1 == nz) {
            output[p] = input[p];
        } else {
            const Real center = input[p];
            const Real left = input[p - 1];
            const Real right = input[p + 1];
            const Real front = input[p - nx];
            const Real back = input[p + nx];
            const Real bottom = input[p - plane];
            const Real top = input[p + plane];
            output[p] = (center + left + right + front + back + bottom + top) / 7.0;
        }
    }
}

void launchStencil(const Real* input, Real* output, size_t nx, size_t ny, size_t nz,
                   size_t firstZ, size_t localZ, size_t beginZ, size_t planes,
                   cudaStream_t stream) {
    if (!planes) return;
    constexpr unsigned threads = 256;
    const size_t blocks = (nx * ny * planes + threads - 1) / threads;
    const unsigned grid = static_cast<unsigned>(std::min<size_t>(blocks, 65535));
    stencilKernel<<<grid, threads, 0, stream>>>(input, output, nx, ny, nz,
                                                firstZ, localZ, beginZ, planes);
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<Real>& grid) {
    Real minVal = std::numeric_limits<Real>::infinity();
    Real maxVal = -std::numeric_limits<Real>::infinity();
    int finite = 1;
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(&:finite)
    for (size_t i = 0; i < grid.size(); ++i) {
        finite &= std::isfinite(grid[i]) ? 1 : 0;
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
    }
    if (!finite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int worldRank, worldSize;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_CHECK(MPI_Finalize());
            return 0;
        } else {
            if (worldRank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_CHECK(MPI_Finalize());
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny > static_cast<size_t>(INT_MAX)) {
        if (worldRank == 0) fprintf(stderr, "Invalid grid size or iteration count\n");
        MPI_CHECK(MPI_Finalize());
        return 1;
    }
    const size_t plane = nx * ny;
    const int ranks = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm active;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED,
                             worldRank, &active));
    if (worldRank >= ranks) {
        MPI_CHECK(MPI_Finalize());
        return 0;
    }
    const int rank = worldRank;
    const size_t firstZ = slabStart(rank, nz, ranks);
    const size_t localZ = slabLength(rank, nz, ranks);
    if (localZ > SIZE_MAX / (plane * sizeof(Real)) - 2) {
        fprintf(stderr, "Local slab is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm shared;
    MPI_CHECK(MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared));
    int localRank;
    MPI_CHECK(MPI_Comm_rank(shared, &localRank));
    int gpuCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpuCount));
    if (gpuCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % gpuCount));
    MPI_CHECK(MPI_Comm_free(&shared));

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    const size_t elements = (localZ + 2) * plane;
    std::vector<Real> host(elements);
    #pragma omp parallel for schedule(static)
    for (size_t z = 1; z <= localZ; ++z) {
        const size_t globalOffset = (firstZ + z - 1) * plane;
        for (size_t p = 0; p < plane; ++p) host[z * plane + p] = static_cast<Real>((globalOffset + p) % 19);
    }
    Real *grid1, *grid2;
    CUDA_CHECK(cudaMalloc(&grid1, elements * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&grid2, elements * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(grid1, host.data(), elements * sizeof(Real), cudaMemcpyHostToDevice));
    cudaStream_t computeStream, commStream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&commStream, cudaStreamNonBlocking));
    Real *sendLow, *sendHigh, *recvLow, *recvHigh;
    const size_t planeBytes = plane * sizeof(Real);
    CUDA_CHECK(cudaHostAlloc(&sendLow, planeBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&sendHigh, planeBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&recvLow, planeBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&recvHigh, planeBytes, cudaHostAllocDefault));

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_CHECK(MPI_Barrier(active));
    const double start = MPI_Wtime();
    Real* input = grid1;
    Real* output = grid2;
    for (int iter = 0; iter < iterations; ++iter) {
        if (ranks == 1) {
            launchStencil(input, output, nx, ny, nz, firstZ, localZ, 1, localZ, computeStream);
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
        } else {
            // Work on planes that do not need a remote halo while MPI progresses.
            if (localZ > 2)
                launchStencil(input, output, nx, ny, nz, firstZ, localZ,
                              2, localZ - 2, computeStream);
            if (rank > 0)
                CUDA_CHECK(cudaMemcpyAsync(sendLow, input + plane, planeBytes,
                                           cudaMemcpyDeviceToHost, commStream));
            if (rank + 1 < ranks)
                CUDA_CHECK(cudaMemcpyAsync(sendHigh, input + localZ * plane, planeBytes,
                                           cudaMemcpyDeviceToHost, commStream));
            CUDA_CHECK(cudaStreamSynchronize(commStream));
            MPI_Request requests[4];
            int count = 0;
            if (rank > 0) {
                MPI_CHECK(MPI_Irecv(recvLow, static_cast<int>(plane), MPI_DOUBLE,
                                    rank - 1, 1, active, &requests[count++]));
                MPI_CHECK(MPI_Isend(sendLow, static_cast<int>(plane), MPI_DOUBLE,
                                    rank - 1, 0, active, &requests[count++]));
            }
            if (rank + 1 < ranks) {
                MPI_CHECK(MPI_Irecv(recvHigh, static_cast<int>(plane), MPI_DOUBLE,
                                    rank + 1, 0, active, &requests[count++]));
                MPI_CHECK(MPI_Isend(sendHigh, static_cast<int>(plane), MPI_DOUBLE,
                                    rank + 1, 1, active, &requests[count++]));
            }
            MPI_CHECK(MPI_Waitall(count, requests, MPI_STATUSES_IGNORE));
            if (rank > 0)
                CUDA_CHECK(cudaMemcpyAsync(input, recvLow, planeBytes,
                                           cudaMemcpyHostToDevice, commStream));
            if (rank + 1 < ranks)
                CUDA_CHECK(cudaMemcpyAsync(input + (localZ + 1) * plane, recvHigh, planeBytes,
                                           cudaMemcpyHostToDevice, commStream));
            // This stream orders the halo uploads before boundary computation.
            launchStencil(input, output, nx, ny, nz, firstZ, localZ, 1, 1, commStream);
            if (localZ > 1)
                launchStencil(input, output, nx, ny, nz, firstZ, localZ,
                              localZ, 1, commStream);
            CUDA_CHECK(cudaStreamSynchronize(commStream));
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
        }
        std::swap(input, output);
    }
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, active));
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        const double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                               static_cast<double>(ny > 2 ? ny - 2 : 0) *
                               static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0.0);
    }

    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(host.data() + plane, input + plane,
                              localZ * planeBytes, cudaMemcpyDeviceToHost));
        // Chunking avoids MPI's int count limit for large local slabs.
        constexpr size_t maxChunk = static_cast<size_t>(INT_MAX);
        if (rank == 0) {
            std::vector<Real> full(nx * ny * nz);
            std::copy_n(host.data() + plane, localZ * plane, full.data());
            for (int sender = 1; sender < ranks; ++sender) {
                size_t offset = slabStart(sender, nz, ranks) * plane;
                size_t remaining = slabLength(sender, nz, ranks) * plane;
                while (remaining) {
                    const int n = static_cast<int>(std::min(remaining, maxChunk));
                    MPI_CHECK(MPI_Recv(full.data() + offset, n, MPI_DOUBLE,
                                       sender, 2, active, MPI_STATUS_IGNORE));
                    offset += n;
                    remaining -= n;
                }
            }
            if (printResults) print_results(full, "Grid");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(full);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                if (!valid) exitCode = 1;
            }
        } else {
            size_t offset = plane;
            size_t remaining = localZ * plane;
            while (remaining) {
                const int n = static_cast<int>(std::min(remaining, maxChunk));
                MPI_CHECK(MPI_Send(host.data() + offset, n, MPI_DOUBLE, 0, 2, active));
                offset += n;
                remaining -= n;
            }
        }
    }
    CUDA_CHECK(cudaFreeHost(sendLow));
    CUDA_CHECK(cudaFreeHost(sendHigh));
    CUDA_CHECK(cudaFreeHost(recvLow));
    CUDA_CHECK(cudaFreeHost(recvHigh));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(commStream));
    CUDA_CHECK(cudaFree(grid1));
    CUDA_CHECK(cudaFree(grid2));
    MPI_CHECK(MPI_Comm_free(&active));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
