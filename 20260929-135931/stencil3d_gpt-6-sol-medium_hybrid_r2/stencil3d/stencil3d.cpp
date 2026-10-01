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

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) checkCuda((call), #call)

static void checkMpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        fprintf(stderr, "%s failed\n", operation);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define MPI_CHECK(call) checkMpi((call), #call)

__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              size_t nx, size_t ny, size_t firstGlobalZ,
                              size_t nz, size_t firstLocalZ, size_t layers) {
    const size_t plane = nx * ny;
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
         y < ny; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
        for (size_t layer = blockIdx.z; layer < layers; layer += gridDim.z) {
            const size_t z = firstLocalZ + layer;
            const size_t index = z * plane + y * nx + x;
            const size_t globalZ = firstGlobalZ + z - 1;
            if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
                globalZ == 0 || globalZ + 1 == nz) {
                output[index] = input[index];
            } else {
                output[index] = (input[index] + input[index - 1] + input[index + 1] +
                                 input[index - nx] + input[index + nx] +
                                 input[index - plane] + input[index + plane]) / 7.0;
            }
        }
    }
}

static void launchStencil(const Real* input, Real* output, size_t nx, size_t ny,
                          size_t firstGlobalZ, size_t nz, size_t firstLocalZ,
                          size_t layers, cudaStream_t stream) {
    if (layers == 0) return;
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned int>((nx + 31) / 32),
                    static_cast<unsigned int>(std::min<size_t>((ny + 7) / 8, 65535)),
                    static_cast<unsigned int>(std::min<size_t>(layers, 65535)));
    stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, firstGlobalZ,
                                              nz, firstLocalZ, layers);
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<Real>& grid) {
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (Real val : grid) {
        if (!std::isfinite(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
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
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = static_cast<size_t>(atoll(argv[++i]));
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = static_cast<size_t>(atoll(argv[++i]));
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = static_cast<size_t>(atoll(argv[++i]));
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        static_cast<size_t>(ranks) > nz ||
        nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions, iteration count, or MPI rank count\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny;
    const size_t base = nz / ranks;
    const size_t remainder = nz % ranks;
    const size_t localZ = base + (static_cast<size_t>(rank) < remainder);
    const size_t firstZ = static_cast<size_t>(rank) * base + std::min<size_t>(rank, remainder);
    if (localZ > (SIZE_MAX / plane) - 2 || localZ * plane > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Local slab is too large for MPI counts\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&nodeComm));
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    const size_t localCells = localZ * plane;
    std::vector<Real> initial(localCells);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localCells; ++i) {
        initial[i] = static_cast<Real>(((firstZ * plane) + i) % 19);
    }

    Real *grid1, *grid2;
    const size_t bytes = (localZ + 2) * plane * sizeof(Real);
    CUDA_CHECK(cudaMalloc(&grid1, bytes));
    CUDA_CHECK(cudaMalloc(&grid2, bytes));
    cudaStream_t computeStream, commStream;
    cudaEvent_t ready;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&commStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaMemcpyAsync(grid1 + plane, initial.data(), localCells * sizeof(Real),
                               cudaMemcpyHostToDevice, computeStream));
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    initial.clear();
    initial.shrink_to_fit();

    Real *sendLow = nullptr, *sendHigh = nullptr, *recvLow = nullptr, *recvHigh = nullptr;
    if (ranks > 1) {
        CUDA_CHECK(cudaMallocHost(&sendLow, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&sendHigh, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&recvLow, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&recvHigh, plane * sizeof(Real)));
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) printf("Running stencil computation...\n");
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        Real* input = (iter & 1) ? grid2 : grid1;
        Real* output = (iter & 1) ? grid1 : grid2;
        if (ranks == 1) {
            launchStencil(input, output, nx, ny, firstZ, nz, 1, localZ, computeStream);
            continue;
        }
        // The communication stream waits for the preceding iteration, while the
        // compute stream can work on layers independent of incoming halos.
        CUDA_CHECK(cudaEventRecord(ready, computeStream));
        CUDA_CHECK(cudaStreamWaitEvent(commStream, ready, 0));
        if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(sendLow, input + plane, plane * sizeof(Real),
                                                cudaMemcpyDeviceToHost, commStream));
        if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(sendHigh, input + localZ * plane,
                                                         plane * sizeof(Real), cudaMemcpyDeviceToHost, commStream));
        launchStencil(input, output, nx, ny, firstZ, nz, 2, localZ > 2 ? localZ - 2 : 0,
                      computeStream);
        CUDA_CHECK(cudaStreamSynchronize(commStream));
        MPI_Request requests[4];
        int count = 0;
        if (rank > 0) {
            MPI_CHECK(MPI_Irecv(recvLow, static_cast<int>(plane), MPI_DOUBLE,
                                rank - 1, 1, MPI_COMM_WORLD, &requests[count++]));
            MPI_CHECK(MPI_Isend(sendLow, static_cast<int>(plane), MPI_DOUBLE,
                                rank - 1, 0, MPI_COMM_WORLD, &requests[count++]));
        }
        if (rank + 1 < ranks) {
            MPI_CHECK(MPI_Irecv(recvHigh, static_cast<int>(plane), MPI_DOUBLE,
                                rank + 1, 0, MPI_COMM_WORLD, &requests[count++]));
            MPI_CHECK(MPI_Isend(sendHigh, static_cast<int>(plane), MPI_DOUBLE,
                                rank + 1, 1, MPI_COMM_WORLD, &requests[count++]));
        }
        MPI_CHECK(MPI_Waitall(count, requests, MPI_STATUSES_IGNORE));
        if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(input, recvLow, plane * sizeof(Real),
                                                cudaMemcpyHostToDevice, commStream));
        if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(input + (localZ + 1) * plane,
                                                         recvHigh, plane * sizeof(Real),
                                                         cudaMemcpyHostToDevice, commStream));
        CUDA_CHECK(cudaEventRecord(ready, commStream));
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, ready, 0));
        launchStencil(input, output, nx, ny, firstZ, nz, 1, 1, computeStream);
        if (localZ > 1) launchStencil(input, output, nx, ny, firstZ, nz, localZ, 1, computeStream);
    }
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    const double localSeconds = MPI_Wtime() - start;
    double seconds;
    MPI_CHECK(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        const double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                               static_cast<double>(ny > 2 ? ny - 2 : 0) *
                               static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0.0);
    }

    if (printResults || validate) {
        std::vector<Real> localResult(localCells);
        Real* finalGrid = (iterations & 1) ? grid2 : grid1;
        CUDA_CHECK(cudaMemcpy(localResult.data(), finalGrid + plane, localCells * sizeof(Real),
                              cudaMemcpyDeviceToHost));
        std::vector<Real> result;
        if (rank == 0) result.resize(nx * ny * nz);
        // Each rank's slab is contiguous in the original row-major grid.
        if (rank == 0) {
            std::copy(localResult.begin(), localResult.end(), result.begin());
            for (int source = 1; source < ranks; ++source) {
                const size_t sourceZ = base + (static_cast<size_t>(source) < remainder);
                const size_t sourceFirst = static_cast<size_t>(source) * base +
                                           std::min<size_t>(source, remainder);
                MPI_CHECK(MPI_Recv(result.data() + sourceFirst * plane,
                                   static_cast<int>(sourceZ * plane), MPI_DOUBLE,
                                   source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            }
            if (printResults) print_results(result, "Grid");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(result);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                if (!valid) MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else {
            MPI_CHECK(MPI_Send(localResult.data(), static_cast<int>(localCells), MPI_DOUBLE,
                               0, 2, MPI_COMM_WORLD));
        }
    }
    if (sendLow) CUDA_CHECK(cudaFreeHost(sendLow));
    if (sendHigh) CUDA_CHECK(cudaFreeHost(sendHigh));
    if (recvLow) CUDA_CHECK(cudaFreeHost(recvLow));
    if (recvHigh) CUDA_CHECK(cudaFreeHost(recvHigh));
    CUDA_CHECK(cudaEventDestroy(ready));
    CUDA_CHECK(cudaStreamDestroy(commStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaFree(grid1));
    CUDA_CHECK(cudaFree(grid2));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
