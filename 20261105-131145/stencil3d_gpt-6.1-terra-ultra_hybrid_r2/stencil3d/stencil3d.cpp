#include <algorithm>
#include <chrono>
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

#include "../common/results_output.hpp"

using Real = double;

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

// The device grids contain the rank's owned planes at z=1..owned, with a
// halo plane on either side.  A launch may cover any interval of owned planes;
// this lets communication of the halos overlap the interior computation.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t first_z,
                              size_t global_nz, size_t begin, size_t end) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t plane = nx * ny;
    for (size_t local_z = begin + blockIdx.z; local_z < end; local_z += gridDim.z) {
        const size_t global_z = first_z + local_z;
        for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             y < ny; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
            const size_t i = (local_z + 1) * plane + y * nx + x;
            if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
                global_z == 0 || global_z == global_nz - 1) {
                output[i] = input[i];
            } else {
                output[i] = (input[i] + input[i - 1] + input[i + 1] +
                             input[i - nx] + input[i + nx] +
                             input[i - plane] + input[i + plane]) / 7.0;
            }
        }
    }
}

void stencil_gpu(const Real* input, Real* output, size_t nx, size_t ny,
                 [[maybe_unused]] size_t owned, size_t first_z, size_t global_nz,
                 size_t begin, size_t end, cudaStream_t stream) {
    if (begin >= end) return;
    const dim3 block(32, 4);
    const size_t row_blocks = ny / block.y + (ny % block.y != 0);
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>(std::min<size_t>(row_blocks, 65535)),
                    static_cast<unsigned int>(std::min<size_t>(end - begin, 65535)));
    stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, first_z,
                                              global_nz, begin, end);
}

// Abort the entire job if a device operation fails; otherwise peers could hang
// waiting for a rank that has already exited.
static void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) check_cuda((call), #call)

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t maxSize = std::numeric_limits<size_t>::max();
    if (nx == 0 || ny == 0 || nz == 0 || nx > maxSize / ny ||
        nx * ny > maxSize / nz || nx * ny * nz > maxSize / sizeof(Real)) {
        if (rank == 0) fprintf(stderr, "Invalid or too large grid size\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny;
    const int activeRanks = static_cast<int>(std::min(static_cast<size_t>(ranks), nz));

    // Shared-memory rank identifies the GPU without requiring a launcher-specific environment variable.
    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank = 0;
    MPI_Comm_rank(shared, &localRank);
    MPI_Comm_free(&shared);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &active);
    if (rank >= activeRanks) {
        MPI_Finalize();
        return 0;
    }
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    const size_t base = nz / activeRanks, extra = nz % activeRanks;
    const size_t owned = base + (static_cast<size_t>(rank) < extra);
    const size_t first = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    if (owned > maxSize - 2 || plane > maxSize / (owned + 2) ||
        plane * (owned + 2) > maxSize / sizeof(Real) || plane > maxSize / (4 * sizeof(Real))) {
        fprintf(stderr, "Rank %d: local grid is too large\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t localElements = plane * (owned + 2);
    const size_t localBytes = localElements * sizeof(Real);
    const size_t planeBytes = plane * sizeof(Real);
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> host(localElements, 0.0);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < owned * plane; ++i)
        host[plane + i] = static_cast<Real>((first * plane + i) % 19);
    Real *input = nullptr, *output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&input), localBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), localBytes));
    CUDA_CHECK(cudaMemcpy(input, host.data(), localBytes, cudaMemcpyHostToDevice));
    // The original selects the initially zero second grid for negative odd
    // iteration counts, even though its iteration loop does not execute.
    if (iterations < 0 && iterations % 2 != 0)
        CUDA_CHECK(cudaMemset(input, 0, localBytes));
    if (!validate && !printResults) std::vector<Real>().swap(host);
    cudaStream_t computeStream, haloStream;
    cudaEvent_t computed;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&haloStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&computed, cudaEventDisableTiming));
    Real* staging = nullptr;
    if (activeRanks > 1) CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&staging), 4 * planeBytes));
    Real* sendLow = staging;
    Real* sendHigh = staging ? staging + plane : nullptr;
    Real* recvLow = staging ? staging + 2 * plane : nullptr;
    Real* recvHigh = staging ? staging + 3 * plane : nullptr;
    const int low = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int high = rank + 1 < activeRanks ? rank + 1 : MPI_PROC_NULL;

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(active);
    double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (activeRanks == 1) {
            stencil_gpu(input, output, nx, ny, owned, first, nz, 0, owned, computeStream);
            CUDA_CHECK(cudaGetLastError());
            std::swap(input, output);
            continue;
        }
        if (iter > 0) CUDA_CHECK(cudaStreamWaitEvent(haloStream, computed, 0));
        if (low != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(sendLow, input + plane, planeBytes, cudaMemcpyDeviceToHost, haloStream));
        if (high != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(sendHigh, input + owned * plane, planeBytes, cudaMemcpyDeviceToHost, haloStream));
        if (owned > 2) {
            stencil_gpu(input, output, nx, ny, owned, first, nz, 1, owned - 1, computeStream);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(haloStream));
        // MPI's count is an int. Exchange large planes in bounded chunks.
        for (size_t offset = 0; offset < plane && activeRanks > 1;) {
            int count = static_cast<int>(std::min(plane - offset, static_cast<size_t>(INT_MAX)));
            MPI_Request requests[4];
            int n = 0;
            if (low != MPI_PROC_NULL) {
                MPI_Irecv(recvLow + offset, count, MPI_DOUBLE, low, 1, active, &requests[n++]);
                MPI_Isend(sendLow + offset, count, MPI_DOUBLE, low, 0, active, &requests[n++]);
            }
            if (high != MPI_PROC_NULL) {
                MPI_Irecv(recvHigh + offset, count, MPI_DOUBLE, high, 0, active, &requests[n++]);
                MPI_Isend(sendHigh + offset, count, MPI_DOUBLE, high, 1, active, &requests[n++]);
            }
            MPI_Waitall(n, requests, MPI_STATUSES_IGNORE);
            offset += static_cast<size_t>(count);
        }
        if (low != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(input, recvLow, planeBytes, cudaMemcpyHostToDevice, haloStream));
        if (high != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(input + (owned + 1) * plane, recvHigh, planeBytes, cudaMemcpyHostToDevice, haloStream));
        CUDA_CHECK(cudaStreamSynchronize(haloStream));
        stencil_gpu(input, output, nx, ny, owned, first, nz, 0, 1, computeStream);
        if (owned > 1) stencil_gpu(input, output, nx, ny, owned, first, nz, owned - 1, owned, computeStream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(computed, computeStream));
        std::swap(input, output);
    }
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    double elapsed = MPI_Wtime() - start, maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, active);
    if (rank == 0) {
        long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                         static_cast<double>(ny > 2 ? ny - 2 : 0) *
                         static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6);
    }

    if (validate || printResults)
        CUDA_CHECK(cudaMemcpy(host.data() + plane, input + plane, owned * planeBytes, cudaMemcpyDeviceToHost));
    if (printResults) {
        std::vector<Real> grid;
        if (rank == 0) {
            grid.resize(plane * nz);
            std::copy_n(host.data() + plane, owned * plane, grid.data());
            for (int src = 1; src < activeRanks; ++src) {
                size_t srcOwned = base + (static_cast<size_t>(src) < extra);
                size_t srcFirst = static_cast<size_t>(src) * base + std::min(static_cast<size_t>(src), extra);
                size_t countTotal = srcOwned * plane;
                for (size_t offset = 0; offset < countTotal;) {
                    int count = static_cast<int>(std::min(countTotal - offset, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(grid.data() + srcFirst * plane + offset, count, MPI_DOUBLE, src, 2, active, MPI_STATUS_IGNORE);
                    offset += static_cast<size_t>(count);
                }
            }
            print_results(grid, "Grid");
        } else {
            size_t countTotal = owned * plane;
            for (size_t offset = 0; offset < countTotal;) {
                int count = static_cast<int>(std::min(countTotal - offset, static_cast<size_t>(INT_MAX)));
                MPI_Send(host.data() + plane + offset, count, MPI_DOUBLE, 0, 2, active);
                offset += static_cast<size_t>(count);
            }
        }
    }
    int valid = 1;
    if (validate) {
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        int finite = 1;
        #pragma omp parallel for reduction(min:localMin) reduction(max:localMax) reduction(&:finite) schedule(static)
        for (size_t i = 0; i < owned * plane; ++i) {
            Real v = host[plane + i];
            if (!std::isfinite(v)) finite = 0;
            localMin = std::min(localMin, v);
            localMax = std::max(localMax, v);
        }
        Real minVal = 0.0, maxVal = 0.0;
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, active);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, active);
        MPI_Reduce(&finite, &valid, 1, MPI_INT, MPI_BAND, 0, active);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!valid) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 1e6 || minVal < -1e6) {
                    printf("Validation failed: values out of expected range\n");
                    valid = 0;
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, active);
    }
    if (staging) CUDA_CHECK(cudaFreeHost(staging));
    CUDA_CHECK(cudaEventDestroy(computed));
    CUDA_CHECK(cudaStreamDestroy(haloStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
    MPI_Comm_free(&active);
    MPI_Finalize();
    return valid ? 0 : 1;
}
