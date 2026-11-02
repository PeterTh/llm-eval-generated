#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

static void cudaCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank, file, line,
                cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

// Each block covers 32 columns and four rows. The grid-stride loop permits
// slabs of any height without exceeding CUDA's grid.y limit.
__global__ void stencil(const Real* __restrict__ input, Real* __restrict__ output,
                        size_t nx, size_t ny, size_t plane, size_t zbegin,
                        size_t zend, size_t rowBlocks) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    for (size_t rowBlock = blockIdx.y; rowBlock < rowBlocks * (zend - zbegin);
         rowBlock += gridDim.y) {
        const size_t z = zbegin + rowBlock / rowBlocks;
        const size_t y = (rowBlock % rowBlocks) * blockDim.y + threadIdx.y + 1;
        if (x < nx - 1 && y < ny - 1) {
            const size_t i = z * plane + y * nx + x;
            output[i] = (input[i] + input[i - 1] + input[i + 1] +
                         input[i - nx] + input[i + nx] + input[i - plane] +
                         input[i + plane]) / 7.0;
        }
    }
}

static void launchStencil(const Real* input, Real* output, size_t nx, size_t ny,
                          size_t plane, size_t firstGlobal, size_t nz,
                          size_t begin, size_t end, cudaStream_t stream) {
    // Local owned planes have indices 1..localNz; preserve global Z boundaries.
    if (firstGlobal == 0) begin = std::max(begin, size_t(2));
    const size_t globalEnd = nz - firstGlobal; // local index of global last plane
    end = std::min(end, globalEnd);
    if (begin >= end || nx < 3 || ny < 3 || nz < 3) return;
    const size_t rows = (ny - 2 + 3) / 4;
    const size_t blocksY = std::min(size_t(65535), rows * (end - begin));
    dim3 block(32, 4);
    dim3 grid(static_cast<unsigned>((nx - 2 + 31) / 32), static_cast<unsigned>(blocksY));
    stencil<<<grid, block, 0, stream>>>(input, output, nx, ny, plane, begin, end, rows);
    CUDA_CHECK(cudaGetLastError());
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// MPI counts are int, so split optional result transfers into bounded chunks.
static void transferResults(Real* data, size_t count, int peer, bool send, MPI_Comm comm) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_DOUBLE, peer, 7, comm);
        else MPI_Recv(data, chunk, MPI_DOUBLE, peer, 7, comm, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

int main(int argc, char** argv) {
    int provided = 0, worldRank = 0, worldSize = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const char* value = argv[++i];
            long long n = strtoll(value, &end, 10);
            if (*value == '\0' || *end != '\0' || n < 0 || n > INT_MAX ||
                (option == 'x' && n == 0)) { bad = true; break; }
            if (option == 'x') nx = static_cast<size_t>(n);
            else if (option == 'y') ny = static_cast<size_t>(n);
            else if (option == 'z') nz = static_cast<size_t>(n);
            else iterations = static_cast<int>(n);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else { bad = true; break; }
    }
    if (help || bad) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return bad ? 1 : 0;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(Real) ||
        nx * ny > static_cast<size_t>(INT_MAX)) {
        if (worldRank == 0) fprintf(stderr, "Grid is too large for this implementation\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t plane = nx * ny;
    // At most one rank per Z plane. Surplus ranks do no work but still finalize MPI.
    const int activeSize = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (worldRank >= activeSize) { MPI_Finalize(); return 0; }
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const size_t localNz = nz / size + (static_cast<size_t>(rank) < nz % size);
    const size_t firstGlobal = static_cast<size_t>(rank) * (nz / size) +
                               std::min(static_cast<size_t>(rank), nz % size);
    if (localNz + 2 > std::numeric_limits<size_t>::max() / plane ||
        (localNz + 2) * plane > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        fprintf(stderr, "Rank %d: local grid too large\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    const size_t localCount = localNz * plane;
    std::vector<Real> host(localCount);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localCount; ++i)
        host[i] = static_cast<Real>((firstGlobal * plane + i) % 19);
    Real *a = nullptr, *b = nullptr;
    const size_t bytes = (localNz + 2) * plane * sizeof(Real);
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    // Both buffers start with identical boundaries, which never change.
    CUDA_CHECK(cudaMemcpy(a + plane, host.data(), localCount * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b + plane, host.data(), localCount * sizeof(Real), cudaMemcpyHostToDevice));
    cudaStream_t computeStream, haloStream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&haloStream, cudaStreamNonBlocking));
    Real *sendLow = nullptr, *sendHigh = nullptr, *recvLow = nullptr, *recvHigh = nullptr;
    const size_t planeBytes = plane * sizeof(Real);
    if (size > 1) {
        CUDA_CHECK(cudaMallocHost(&sendLow, planeBytes));
        CUDA_CHECK(cudaMallocHost(&sendHigh, planeBytes));
        CUDA_CHECK(cudaMallocHost(&recvLow, planeBytes));
        CUDA_CHECK(cudaMallocHost(&recvHigh, planeBytes));
    }
    const int low = rank ? rank - 1 : MPI_PROC_NULL;
    const int high = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    Real *input = a, *output = b;
    if (nx >= 3 && ny >= 3 && nz >= 3) {
        for (int iter = 0; iter < iterations; ++iter) {
            if (size == 1) {
                launchStencil(input, output, nx, ny, plane, firstGlobal, nz, 1, localNz + 1, computeStream);
            } else {
                MPI_Request requests[4];
                int nreq = 0;
                if (low != MPI_PROC_NULL) {
                    MPI_Irecv(recvLow, static_cast<int>(plane), MPI_DOUBLE, low, 2, comm, &requests[nreq++]);
                    CUDA_CHECK(cudaMemcpyAsync(sendLow, input + plane, planeBytes, cudaMemcpyDeviceToHost, haloStream));
                }
                if (high != MPI_PROC_NULL) {
                    MPI_Irecv(recvHigh, static_cast<int>(plane), MPI_DOUBLE, high, 1, comm, &requests[nreq++]);
                    CUDA_CHECK(cudaMemcpyAsync(sendHigh, input + localNz * plane, planeBytes, cudaMemcpyDeviceToHost, haloStream));
                }
                launchStencil(input, output, nx, ny, plane, firstGlobal, nz, 2, localNz, computeStream);
                CUDA_CHECK(cudaStreamSynchronize(haloStream));
                if (low != MPI_PROC_NULL) MPI_Isend(sendLow, static_cast<int>(plane), MPI_DOUBLE, low, 1, comm, &requests[nreq++]);
                if (high != MPI_PROC_NULL) MPI_Isend(sendHigh, static_cast<int>(plane), MPI_DOUBLE, high, 2, comm, &requests[nreq++]);
                MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
                if (low != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(input, recvLow, planeBytes, cudaMemcpyHostToDevice, haloStream));
                if (high != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(input + (localNz + 1) * plane, recvHigh, planeBytes, cudaMemcpyHostToDevice, haloStream));
                launchStencil(input, output, nx, ny, plane, firstGlobal, nz, 1, 2, haloStream);
                if (localNz > 1) launchStencil(input, output, nx, ny, plane, firstGlobal, nz, localNz, localNz + 1, haloStream);
                CUDA_CHECK(cudaStreamSynchronize(haloStream));
                CUDA_CHECK(cudaStreamSynchronize(computeStream));
            }
            std::swap(input, output);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        long duration = static_cast<long>(maxElapsed * 1000.0);
        double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                         (ny > 2 ? ny - 2 : 0) * (nz > 2 ? nz - 2 : 0) * iterations;
        printf("Computation time: %ld ms\n", duration);
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6);
    }
    if (printResults || validate)
        CUDA_CHECK(cudaMemcpy(host.data(), input + plane, localCount * sizeof(Real), cudaMemcpyDeviceToHost));
    if (printResults) {
        if (rank == 0) {
            std::vector<Real> grid(nx * ny * nz);
            std::copy(host.begin(), host.end(), grid.begin());
            size_t offset = localCount;
            for (int r = 1; r < size; ++r) {
                size_t count = (nz / size + (static_cast<size_t>(r) < nz % size)) * plane;
                transferResults(grid.data() + offset, count, r, false, comm);
                offset += count;
            }
            print_results(grid, "Grid");
        } else transferResults(host.data(), localCount, 0, true, comm);
    }
    int status = 0;
    if (validate) {
        Real minVal = std::numeric_limits<Real>::infinity(), maxVal = -minVal;
        int invalid = 0;
        #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(|:invalid) schedule(static)
        for (size_t i = 0; i < localCount; ++i) {
            Real value = host[i];
            if (!std::isfinite(value)) invalid = 1;
            minVal = std::min(minVal, value);
            maxVal = std::max(maxVal, value);
        }
        Real globalMin = 0, globalMax = 0;
        int globalInvalid = 0;
        MPI_Reduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_BOR, 0, comm);
        if (rank == 0) {
            printf("Validating result...\n");
            if (globalInvalid) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalMax > 1e6 || globalMin < -1e6)
                    printf("Validation failed: values out of expected range\n");
            }
            status = globalInvalid || globalMax > 1e6 || globalMin < -1e6;
            printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    }
    if (size > 1) {
        CUDA_CHECK(cudaFreeHost(sendLow)); CUDA_CHECK(cudaFreeHost(sendHigh));
        CUDA_CHECK(cudaFreeHost(recvLow)); CUDA_CHECK(cudaFreeHost(recvHigh));
    }
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(haloStream));
    CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return status;
}
