#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

static void cudaCheck(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

// Each GPU holds a contiguous z slab and one ghost plane on either side.
// The two buffers are initialized identically so physical boundaries never
// need to be copied again as the buffers alternate.
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
                              size_t nx, size_t ny, size_t zBegin, size_t zEnd) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t firstY = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t z = static_cast<size_t>(blockIdx.z) + zBegin;
    if (x >= nx - 1 || z >= zEnd) return;
    const size_t plane = nx * ny;
    for (size_t y = firstY; y < ny - 1; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
        const size_t idx = z * plane + y * nx + x;
        out[idx] = (in[idx] + in[idx - 1] + in[idx + 1] + in[idx - nx] +
                    in[idx + nx] + in[idx - plane] + in[idx + plane]) / 7.0;
    }
}

static void launchStencil(const Real* in, Real* out, size_t nx, size_t ny,
                          size_t begin, size_t end, cudaStream_t stream) {
    if (begin >= end || nx < 3 || ny < 3) return;
    const dim3 block(32, 4);
    // Limit grid z so very deep domains work with CUDA's grid dimension limit.
    for (size_t z = begin; z < end;) {
        const size_t count = std::min(end - z, size_t(65535));
        const dim3 grid(static_cast<unsigned>((nx - 2 + block.x - 1) / block.x),
                        static_cast<unsigned>(std::min(size_t(65535), (ny - 2 + block.y - 1) / block.y)),
                        static_cast<unsigned>(count));
        stencilKernel<<<grid, block, 0, stream>>>(in, out, nx, ny, z, z + count);
        CUDA_CHECK(cudaGetLastError());
        z += count;
    }
}

static void printUsage(const char* progName) {
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

// MPI's count argument is an int, so large planes are sent in ordered chunks.
static void postPlane(Real* data, size_t count, int peer, int tag, bool receive,
                      MPI_Comm comm, std::vector<MPI_Request>& requests) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        MPI_Request request;
        if (receive) MPI_Irecv(data, chunk, MPI_DOUBLE, peer, tag, comm, &request);
        else MPI_Isend(data, chunk, MPI_DOUBLE, peer, tag, comm, &request);
        requests.push_back(request);
        data += chunk;
        count -= chunk;
    }
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny * nz > SIZE_MAX / sizeof(Real) || iterations < 0) {
        if (worldRank == 0) fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        MPI_Finalize();
        return 1;
    }

    if (worldRank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    // Ranks beyond the number of planes have no work and need no GPU.
    const int activeCount = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeCount ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (worldRank >= activeCount) {
        MPI_Finalize();
        return 0;
    }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    const size_t plane = nx * ny;
    const size_t base = nz / ranks, extra = nz % ranks;
    const size_t localZ = base + (static_cast<size_t>(rank) < extra);
    const size_t firstZ = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const size_t maxPlanes = SIZE_MAX / sizeof(Real) / plane;
    if (maxPlanes < 2 || localZ > maxPlanes - 2) {
        fprintf(stderr, "Rank %d: local grid is too large\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t bytes = (localZ + 2) * plane * sizeof(Real);
    std::vector<Real> host((localZ + 2) * plane, 0.0);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localZ * plane; ++i)
        host[plane + i] = static_cast<Real>((firstZ * plane + i) % 19);

    Real *grid[2], *sendLow = nullptr, *sendHigh = nullptr, *recvLow = nullptr, *recvHigh = nullptr;
    CUDA_CHECK(cudaMalloc(&grid[0], bytes));
    CUDA_CHECK(cudaMalloc(&grid[1], bytes));
    CUDA_CHECK(cudaMemcpy(grid[0], host.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grid[1], host.data(), bytes, cudaMemcpyHostToDevice));
    const size_t planeBytes = plane * sizeof(Real);
    if (rank > 0) {
        CUDA_CHECK(cudaMallocHost(&sendLow, planeBytes));
        CUDA_CHECK(cudaMallocHost(&recvLow, planeBytes));
    }
    if (rank + 1 < ranks) {
        CUDA_CHECK(cudaMallocHost(&sendHigh, planeBytes));
        CUDA_CHECK(cudaMallocHost(&recvHigh, planeBytes));
    }
    cudaStream_t compute, transfer;
    cudaEvent_t computed, halosReady;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&computed, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&halosReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(computed, compute));

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    const size_t begin = firstZ == 0 ? 2 : 1;
    const size_t end = localZ + 1 - (firstZ + localZ == nz ? 1 : 0);
    for (int iter = 0; iter < iterations; ++iter) {
        if (ranks == 1) {
            launchStencil(grid[iter % 2], grid[(iter + 1) % 2], nx, ny, begin, end, compute);
            continue;
        }
        Real* in = grid[iter % 2];
        Real* out = grid[(iter + 1) % 2];
        CUDA_CHECK(cudaStreamWaitEvent(transfer, computed, 0));
        if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(sendLow, in + plane, planeBytes, cudaMemcpyDeviceToHost, transfer));
        if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(sendHigh, in + localZ * plane, planeBytes, cudaMemcpyDeviceToHost, transfer));

        std::vector<MPI_Request> requests;
        if (rank > 0) postPlane(recvLow, plane, rank - 1, 1, true, comm, requests);
        if (rank + 1 < ranks) postPlane(recvHigh, plane, rank + 1, 0, true, comm, requests);
        // Interior planes do not depend on the halo and overlap communication.
        launchStencil(in, out, nx, ny, std::max(begin, size_t(2)), std::min(end, localZ), compute);
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        if (rank > 0) postPlane(sendLow, plane, rank - 1, 0, false, comm, requests);
        if (rank + 1 < ranks) postPlane(sendHigh, plane, rank + 1, 1, false, comm, requests);
        if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(in, recvLow, planeBytes, cudaMemcpyHostToDevice, transfer));
        if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(in + (localZ + 1) * plane, recvHigh, planeBytes, cudaMemcpyHostToDevice, transfer));
        CUDA_CHECK(cudaEventRecord(halosReady, transfer));
        CUDA_CHECK(cudaStreamWaitEvent(compute, halosReady, 0));
        if (begin <= 1 && 1 < end) launchStencil(in, out, nx, ny, 1, 2, compute);
        if (localZ > 1 && begin <= localZ && localZ < end)
            launchStencil(in, out, nx, ny, localZ, localZ + 1, compute);
        CUDA_CHECK(cudaEventRecord(computed, compute));
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const long milliseconds = static_cast<long>(seconds * 1000.0);
        const double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
            static_cast<double>(ny > 2 ? ny - 2 : 0) * static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Computation time: %ld ms\n", milliseconds);
        printf("Performance: %.3f MCellUpdates/s\n", updates / seconds / 1e6);
    }

    if (validate || printResults)
        CUDA_CHECK(cudaMemcpy(host.data() + plane, grid[iterations % 2] + plane, localZ * planeBytes, cudaMemcpyDeviceToHost));
    if (printResults) {
        if (rank == 0) {
            std::vector<Real> full(nx * ny * nz);
            std::copy_n(host.data() + plane, localZ * plane, full.data());
            for (int source = 1; source < ranks; ++source) {
                const size_t n = base + (static_cast<size_t>(source) < extra);
                const size_t offset = static_cast<size_t>(source) * base + std::min(static_cast<size_t>(source), extra);
                size_t remaining = n * plane, pos = offset * plane;
                while (remaining) {
                    const int chunk = static_cast<int>(std::min(remaining, size_t(INT_MAX)));
                    MPI_Recv(full.data() + pos, chunk, MPI_DOUBLE, source, 2, comm, MPI_STATUS_IGNORE);
                    remaining -= chunk; pos += chunk;
                }
            }
            print_results(full, "Grid");
        } else {
            size_t remaining = localZ * plane, pos = plane;
            while (remaining) {
                const int chunk = static_cast<int>(std::min(remaining, size_t(INT_MAX)));
                MPI_Send(host.data() + pos, chunk, MPI_DOUBLE, 0, 2, comm);
                remaining -= chunk; pos += chunk;
            }
        }
    }
    int valid = 1;
    if (validate) {
        int finite = 1;
        Real low = std::numeric_limits<Real>::infinity(), high = -low;
        #pragma omp parallel for reduction(&:finite) reduction(min:low) reduction(max:high) schedule(static)
        for (size_t i = plane; i < (localZ + 1) * plane; ++i) {
            const Real value = host[i];
            finite &= std::isfinite(value);
            low = std::min(low, value);
            high = std::max(high, value);
        }
        int allFinite;
        Real globalLow, globalHigh;
        MPI_Reduce(&finite, &allFinite, 1, MPI_INT, MPI_LAND, 0, comm);
        MPI_Reduce(&low, &globalLow, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&high, &globalHigh, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!allFinite) { printf("Validation failed: found NaN or Inf value\n"); valid = 0; }
            else {
                printf("Value range: [%.6f, %.6f]\n", globalLow, globalHigh);
                if (globalHigh > 1e6 || globalLow < -1e6) {
                    printf("Validation failed: values out of expected range\n"); valid = 0;
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    }
    CUDA_CHECK(cudaEventDestroy(computed));
    CUDA_CHECK(cudaEventDestroy(halosReady));
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaStreamDestroy(transfer));
    if (sendLow) CUDA_CHECK(cudaFreeHost(sendLow));
    if (recvLow) CUDA_CHECK(cudaFreeHost(recvLow));
    if (sendHigh) CUDA_CHECK(cudaFreeHost(sendHigh));
    if (recvHigh) CUDA_CHECK(cudaFreeHost(recvHigh));
    CUDA_CHECK(cudaFree(grid[0]));
    CUDA_CHECK(cudaFree(grid[1]));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
