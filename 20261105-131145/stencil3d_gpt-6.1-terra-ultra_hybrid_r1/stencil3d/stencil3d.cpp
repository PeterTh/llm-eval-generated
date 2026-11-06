#include <algorithm>
#include <cerrno>
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

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// z coordinates are local and include one ghost plane on either side.
__global__ void stencil(const Real* __restrict__ in, Real* __restrict__ out,
                        size_t nx, size_t ny, size_t nz, size_t globalStart,
                        size_t zBegin, size_t zEnd) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t firstY = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    if (x >= nx - 1) return;
    const size_t plane = nx * ny;
    for (size_t z = zBegin + blockIdx.z; z < zEnd; z += gridDim.z) {
        const size_t globalZ = globalStart + z - 1;
        if (globalZ == 0 || globalZ + 1 == nz) continue;
        for (size_t y = firstY; y < ny - 1; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
            const size_t p = z * plane + y * nx + x;
            const Real sum = in[p] + in[p - 1] + in[p + 1] + in[p - nx] +
                             in[p + nx] + in[p - plane] + in[p + plane];
            out[p] = __ddiv_rn(sum, 7.0);
        }
    }
}

static void launchStencil(const Real* in, Real* out, size_t nx, size_t ny, size_t nz,
                          size_t globalStart, size_t begin, size_t end, cudaStream_t stream) {
    if (begin >= end) return;
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned>((nx - 2 + block.x - 1) / block.x),
                    static_cast<unsigned>(std::min<size_t>((ny - 2 + block.y - 1) / block.y, 65535)),
                    static_cast<unsigned>(std::min<size_t>(end - begin, 65535)));
    stencil<<<grid, block, 0, stream>>>(in, out, nx, ny, nz, globalStart, begin, end);
    CUDA(cudaGetLastError());
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

static bool parseNumber(const char* text, size_t& result) {
    if (!*text || *text == '-') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (*end || errno == ERANGE || value > std::numeric_limits<size_t>::max()) return false;
    result = static_cast<size_t>(value);
    return true;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t nx = 128, ny = 0, nz = 0, iterations = 10;
    bool validate = false, printResults = false, bad = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h")) help = true;
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
                  !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            size_t value;
            if (!parseNumber(argv[++i], value)) { bad = true; break; }
            if (option == 'x') nx = value;
            else if (option == 'y') ny = value;
            else if (option == 'z') nz = value;
            else iterations = value;
        } else { bad = true; break; }
    }
    if (help || bad) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return bad ? 1 : 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny ||
        nx * ny > SIZE_MAX / nz || nx * ny > static_cast<size_t>(INT_MAX) ||
        nx * ny * nz > SIZE_MAX / sizeof(Real)) {
        if (worldRank == 0) fprintf(stderr, "Invalid or unsupported grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny;
    const int activeSize = static_cast<int>(std::min<size_t>(nz, worldSize));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (worldRank >= activeSize) {
        MPI_Finalize();
        return 0;
    }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);

    MPI_Comm localComm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    const size_t base = nz / ranks, extra = nz % ranks;
    const size_t localNz = base + (static_cast<size_t>(rank) < extra);
    const size_t startZ = static_cast<size_t>(rank) * base + std::min<size_t>(rank, extra);
    if (localNz > SIZE_MAX / plane - 2 || (localNz + 2) * plane > SIZE_MAX / sizeof(Real)) {
        fprintf(stderr, "Rank %d: local grid is too large\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t localElements = localNz * plane;
    const size_t bytes = (localNz + 2) * plane * sizeof(Real);
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %zu\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> host(localElements);
    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < localElements; ++p)
        host[p] = static_cast<Real>((startZ * plane + p) % 19);

    Real *input, *output;
    CUDA(cudaMalloc(&input, bytes));
    CUDA(cudaMalloc(&output, bytes));
    CUDA(cudaMemcpy(input + plane, host.data(), localElements * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(output + plane, host.data(), localElements * sizeof(Real), cudaMemcpyHostToDevice));
    cudaStream_t compute, transfer;
    CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    Real *sendLower = nullptr, *sendUpper = nullptr, *recvLower = nullptr, *recvUpper = nullptr;
    if (ranks > 1 && nx > 2 && ny > 2 && nz > 2 && iterations) {
        CUDA(cudaMallocHost(&sendLower, plane * sizeof(Real)));
        CUDA(cudaMallocHost(&sendUpper, plane * sizeof(Real)));
        CUDA(cudaMallocHost(&recvLower, plane * sizeof(Real)));
        CUDA(cudaMallocHost(&recvUpper, plane * sizeof(Real)));
    }
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    if (nx > 2 && ny > 2 && nz > 2) {
        for (size_t iter = 0; iter < iterations; ++iter) {
            if (ranks == 1) {
                launchStencil(input, output, nx, ny, nz, startZ, 1, localNz + 1, compute);
            } else {
                MPI_Request requests[4];
                int count = 0;
                if (rank > 0)
                    MPI_Irecv(recvLower, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1, comm, &requests[count++]);
                if (rank + 1 < ranks)
                    MPI_Irecv(recvUpper, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0, comm, &requests[count++]);
                if (rank > 0)
                    CUDA(cudaMemcpyAsync(sendLower, input + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, transfer));
                if (rank + 1 < ranks)
                    CUDA(cudaMemcpyAsync(sendUpper, input + localNz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, transfer));
                launchStencil(input, output, nx, ny, nz, startZ, 2, localNz, compute);
                CUDA(cudaStreamSynchronize(transfer));
                if (rank > 0)
                    MPI_Isend(sendLower, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0, comm, &requests[count++]);
                if (rank + 1 < ranks)
                    MPI_Isend(sendUpper, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 1, comm, &requests[count++]);
                MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
                if (rank > 0)
                    CUDA(cudaMemcpyAsync(input, recvLower, plane * sizeof(Real), cudaMemcpyHostToDevice, transfer));
                if (rank + 1 < ranks)
                    CUDA(cudaMemcpyAsync(input + (localNz + 1) * plane, recvUpper, plane * sizeof(Real), cudaMemcpyHostToDevice, transfer));
                launchStencil(input, output, nx, ny, nz, startZ, 1, 2, transfer);
                if (localNz > 1)
                    launchStencil(input, output, nx, ny, nz, startZ, localNz, localNz + 1, transfer);
            }
            if (ranks > 1) {
                CUDA(cudaStreamSynchronize(compute));
                CUDA(cudaStreamSynchronize(transfer));
            }
            std::swap(input, output);
        }
        if (ranks == 1) CUDA(cudaStreamSynchronize(compute));
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const long ms = static_cast<long>(maxElapsed * 1000.0);
        const double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                               static_cast<double>(ny > 2 ? ny - 2 : 0) *
                               static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Computation time: %ld ms\n", ms);
        printf("Performance: %.3f MCellUpdates/s\n", maxElapsed > 0 ? updates / maxElapsed / 1e6 : 0.0);
    }
    if (validate || printResults)
        CUDA(cudaMemcpy(host.data(), input + plane, localElements * sizeof(Real), cudaMemcpyDeviceToHost));
    if (printResults) {
        // Chunk transfers to avoid MPI's int count and displacement limits.
        if (rank == 0) {
            std::vector<Real> global(nx * ny * nz);
            std::copy(host.begin(), host.end(), global.begin());
            for (int source = 1; source < ranks; ++source) {
                const size_t sourceNz = base + (static_cast<size_t>(source) < extra);
                const size_t sourceStart = (static_cast<size_t>(source) * base + std::min<size_t>(source, extra)) * plane;
                const size_t elements = sourceNz * plane;
                for (size_t offset = 0; offset < elements;) {
                    const int n = static_cast<int>(std::min<size_t>(elements - offset, INT_MAX));
                    MPI_Recv(global.data() + sourceStart + offset, n, MPI_DOUBLE, source, 2, comm, MPI_STATUS_IGNORE);
                    offset += n;
                }
            }
            print_results(global, "Grid");
        } else {
            for (size_t offset = 0; offset < localElements;) {
                const int n = static_cast<int>(std::min<size_t>(localElements - offset, INT_MAX));
                MPI_Send(host.data() + offset, n, MPI_DOUBLE, 0, 2, comm);
                offset += n;
            }
        }
    }
    int valid = 1;
    if (validate) {
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        int finite = 1;
        #pragma omp parallel for reduction(min:localMin) reduction(max:localMax) reduction(&:finite) schedule(static)
        for (size_t p = 0; p < localElements; ++p) {
            const Real value = host[p];
            finite &= std::isfinite(value) ? 1 : 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        Real globalMin, globalMax;
        int allFinite;
        MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&finite, &allFinite, 1, MPI_INT, MPI_BAND, 0, comm);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!allFinite) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalMax > 1e6 || globalMin < -1e6)
                    printf("Validation failed: values out of expected range\n");
            }
            valid = allFinite && globalMax <= 1e6 && globalMin >= -1e6;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    }
    if (sendLower) {
        CUDA(cudaFreeHost(sendLower)); CUDA(cudaFreeHost(sendUpper));
        CUDA(cudaFreeHost(recvLower)); CUDA(cudaFreeHost(recvUpper));
    }
    CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(transfer));
    CUDA(cudaFree(input)); CUDA(cudaFree(output));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
