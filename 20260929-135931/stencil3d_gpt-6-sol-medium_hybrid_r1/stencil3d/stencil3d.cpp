#include <algorithm>
#include <chrono>
#include <climits>
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

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

// Each rank owns a contiguous range of Z planes; planes 0 and localZ+1 are halos.
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              size_t nx, size_t ny, size_t nz, size_t firstGlobalZ,
                              size_t firstLocalZ, size_t planeCount) {
    const size_t plane = nx * ny;
    const size_t count = plane * planeCount;
    for (size_t p = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         p < count; p += size_t(blockDim.x) * gridDim.x) {
        const size_t localZ = firstLocalZ + p / plane;
        const size_t offset = p % plane;
        const size_t y = offset / nx;
        const size_t x = offset % nx;
        const size_t i = localZ * plane + offset;
        const size_t globalZ = firstGlobalZ + localZ - 1;
        if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
            globalZ == 0 || globalZ + 1 == nz) {
            output[i] = input[i];
        } else {
            output[i] = (input[i] + input[i - 1] + input[i + 1] +
                         input[i - nx] + input[i + nx] +
                         input[i - plane] + input[i + plane]) / 7.0;
        }
    }
}

static void launchStencil(const Real* input, Real* output, size_t nx, size_t ny,
                          size_t nz, size_t firstZ, size_t firstLocal, size_t count,
                          cudaStream_t stream) {
    if (count == 0) return;
    const size_t cells = nx * ny * count;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((cells + 255) / 256, 65535));
    stencilKernel<<<blocks, 256, 0, stream>>>(input, output, nx, ny, nz, firstZ, firstLocal, count);
    CUDA_CHECK(cudaGetLastError());
}

static bool validateResult(const std::vector<Real>& grid) {
    Real minVal = std::numeric_limits<Real>::infinity();
    Real maxVal = -std::numeric_limits<Real>::infinity();
    int invalid = 0;
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(|:invalid)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real v = grid[i];
        if (!std::isfinite(v)) invalid = 1;
        else {
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }
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

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int worldRank, worldSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-x") == 0 || std::strcmp(argv[i], "-y") == 0 ||
             std::strcmp(argv[i], "-z") == 0 || std::strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (*end || argv[i][0] == '-' || value > std::numeric_limits<size_t>::max() ||
                (option == 'i' && value > INT_MAX)) bad = true;
            else if (option == 'x') nx = value;
            else if (option == 'y') ny = value;
            else if (option == 'z') nz = value;
            else iterations = static_cast<int>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else bad = true;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (help || bad || nx < 2 || ny < 2 || nz < 2 ||
        nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz) {
        if (worldRank == 0) {
            if (!help) std::fprintf(stderr, "Invalid grid size or option\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return help ? 0 : 1;
    }

    // Ranks beyond the number of planes have no work and do not join the active communicator.
    const int activeSize = static_cast<int>(std::min<size_t>(nz, worldSize));
    MPI_Comm active = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED, worldRank, &active);
    if (active == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const int rank = worldRank;
    const size_t firstZ = nz * size_t(rank) / activeSize;
    const size_t nextZ = nz * size_t(rank + 1) / activeSize;
    const size_t localZ = nextZ - firstZ;
    const size_t plane = nx * ny;
    if (plane > SIZE_MAX / (localZ + 2) || plane > SIZE_MAX / sizeof(Real) || plane > INT_MAX) {
        if (rank == 0) std::fprintf(stderr, "Local grid too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm shared;
    MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank;
    MPI_Comm_rank(shared, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        std::fprintf(stderr, "Rank %d has no CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\nInitializing grid...\n", activeSize);
    }
    std::vector<Real> host((localZ + 2) * plane, 0.0);
    #pragma omp parallel for schedule(static)
    for (size_t z = 0; z < localZ; ++z) {
        const size_t globalBase = (firstZ + z) * plane;
        const size_t localBase = (z + 1) * plane;
        for (size_t j = 0; j < plane; ++j) host[localBase + j] = Real((globalBase + j) % 19);
    }
    Real *a = nullptr, *b = nullptr;
    CUDA_CHECK(cudaMalloc(&a, host.size() * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&b, host.size() * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(a, host.data(), host.size() * sizeof(Real), cudaMemcpyHostToDevice));

    Real *sendLow = nullptr, *sendHigh = nullptr, *recvLow = nullptr, *recvHigh = nullptr;
    const size_t planeBytes = plane * sizeof(Real);
    CUDA_CHECK(cudaMallocHost(&sendLow, planeBytes));
    CUDA_CHECK(cudaMallocHost(&sendHigh, planeBytes));
    CUDA_CHECK(cudaMallocHost(&recvLow, planeBytes));
    CUDA_CHECK(cudaMallocHost(&recvHigh, planeBytes));
    cudaStream_t compute, transfer;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    const int below = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int above = rank + 1 == activeSize ? MPI_PROC_NULL : rank + 1;

    MPI_Barrier(active);
    if (rank == 0) std::printf("Running stencil computation...\n");
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (activeSize == 1) {
            launchStencil(a, b, nx, ny, nz, firstZ, 1, localZ, compute);
            CUDA_CHECK(cudaStreamSynchronize(compute));
            std::swap(a, b);
            continue;
        }
        // Copy outgoing planes while the GPU computes planes that need no remote halo.
        if (below != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(sendLow, a + plane, planeBytes, cudaMemcpyDeviceToHost, transfer));
        if (above != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(sendHigh, a + localZ * plane, planeBytes, cudaMemcpyDeviceToHost, transfer));
        if (localZ > 2) launchStencil(a, b, nx, ny, nz, firstZ, 2, localZ - 2, compute);
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        // Tag 0 travels upward; tag 1 travels downward.
        MPI_Request requests[4];
        int requestCount = 0;
        if (below != MPI_PROC_NULL) {
            MPI_Irecv(recvLow, static_cast<int>(plane), MPI_DOUBLE, below, 0, active, &requests[requestCount++]);
            MPI_Isend(sendLow, static_cast<int>(plane), MPI_DOUBLE, below, 1, active, &requests[requestCount++]);
        }
        if (above != MPI_PROC_NULL) {
            MPI_Irecv(recvHigh, static_cast<int>(plane), MPI_DOUBLE, above, 1, active, &requests[requestCount++]);
            MPI_Isend(sendHigh, static_cast<int>(plane), MPI_DOUBLE, above, 0, active, &requests[requestCount++]);
        }
        if (requestCount) MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
        if (below != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(a, recvLow, planeBytes, cudaMemcpyHostToDevice, transfer));
        if (above != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(a + (localZ + 1) * plane, recvHigh, planeBytes, cudaMemcpyHostToDevice, transfer));
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        launchStencil(a, b, nx, ny, nz, firstZ, 1, 1, compute);
        if (localZ > 1) launchStencil(a, b, nx, ny, nz, firstZ, localZ, 1, compute);
        CUDA_CHECK(cudaStreamSynchronize(compute));
        std::swap(a, b);
    }
    double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, active);
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        const double updates = double(nx - 2) * double(ny - 2) * double(nz - 2) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", maxElapsed > 0 ? updates / maxElapsed / 1e6 : 0.0);
    }

    int result = 0;
    if (printResults || validate) {
        std::vector<Real> local(localZ * plane);
        CUDA_CHECK(cudaMemcpy(local.data(), a + plane, local.size() * sizeof(Real), cudaMemcpyDeviceToHost));
        std::vector<Real> full;
        if (rank == 0) full.resize(nx * ny * nz);
        if (rank == 0) std::copy(local.begin(), local.end(), full.begin());
        for (int source = 1; source < activeSize; ++source) {
            const size_t begin = nz * size_t(source) / activeSize * plane;
            const size_t end = nz * size_t(source + 1) / activeSize * plane;
            const size_t count = end - begin;
            for (size_t off = 0; off < count; off += INT_MAX / sizeof(Real)) {
                const int chunk = static_cast<int>(std::min<size_t>(count - off, INT_MAX / sizeof(Real)));
                if (rank == source) MPI_Send(local.data() + off, chunk, MPI_DOUBLE, 0, 2, active);
                if (rank == 0) MPI_Recv(full.data() + begin + off, chunk, MPI_DOUBLE, source, 2, active, MPI_STATUS_IGNORE);
            }
        }
        if (rank == 0) {
            if (printResults) print_results(full, "Grid");
            if (validate) {
                std::printf("Validating result...\n");
                result = validateResult(full) ? 0 : 1;
                std::printf("Validation: %s\n", result ? "FAILED" : "PASSED");
            }
        }
        MPI_Bcast(&result, 1, MPI_INT, 0, active);
    }
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaStreamDestroy(transfer));
    CUDA_CHECK(cudaFreeHost(sendLow));
    CUDA_CHECK(cudaFreeHost(sendHigh));
    CUDA_CHECK(cudaFreeHost(recvLow));
    CUDA_CHECK(cudaFreeHost(recvHigh));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    MPI_Comm_free(&shared);
    MPI_Comm_free(&active);
    MPI_Finalize();
    return result;
}
