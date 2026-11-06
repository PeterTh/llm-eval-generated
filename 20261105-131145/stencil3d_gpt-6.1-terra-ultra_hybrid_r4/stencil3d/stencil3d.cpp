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
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

static int worldRank = 0;

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", worldRank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

static void mpiCheck(int error, const char* operation) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "Rank %d: %s: %.*s\n", worldRank, operation, length, message);
        MPI_Abort(MPI_COMM_WORLD, error);
    }
}
#define MPI_CHECK(call) mpiCheck((call), #call)

// The x direction is contiguous; the other grid dimensions cover y and z.
__global__ void stencil(const Real* __restrict__ in, Real* __restrict__ out,
                        size_t nx, size_t ny, size_t firstZ, size_t zCount) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (x >= nx - 1) return;
    const size_t plane = nx * ny;
    for (size_t z = blockIdx.z; z < zCount; z += gridDim.z) {
        for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
             y < ny - 1; y += size_t(gridDim.y) * blockDim.y) {
            size_t p = (firstZ + z) * plane + y * nx + x;
            out[p] = (in[p] + in[p - 1] + in[p + 1] + in[p - nx] +
                      in[p + nx] + in[p - plane] + in[p + plane]) / 7.0;
        }
    }
}

static void launch(const Real* in, Real* out, size_t nx, size_t ny,
                   size_t firstZ, size_t count, cudaStream_t stream) {
    if (!count || nx <= 2 || ny <= 2) return;
    dim3 block(32, 4);
    dim3 grid(static_cast<unsigned>((nx - 2 + block.x - 1) / block.x),
              static_cast<unsigned>(std::min<size_t>((ny - 2 + block.y - 1) / block.y, 65535)),
              static_cast<unsigned>(std::min<size_t>(count, 65535)));
    stencil<<<grid, block, 0, stream>>>(in, out, nx, ny, firstZ, count);
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

// MPI's traditional point-to-point counts are int, so transfer large planes
// and output slabs in chunks. Messages with the same tag retain their order.
static void postChunks(Real* ptr, size_t count, int peer, int tag, bool send,
                       MPI_Comm comm, std::vector<MPI_Request>& requests) {
    while (count) {
        int n = static_cast<int>(std::min<size_t>(count, INT_MAX));
        MPI_Request request;
        if (send) MPI_CHECK(MPI_Isend(ptr, n, MPI_DOUBLE, peer, tag, comm, &request));
        else MPI_CHECK(MPI_Irecv(ptr, n, MPI_DOUBLE, peer, tag, comm, &request));
        requests.push_back(request);
        ptr += n;
        count -= n;
    }
}

static void transferChunks(Real* ptr, size_t count, int peer, bool send, MPI_Comm comm) {
    while (count) {
        int n = static_cast<int>(std::min<size_t>(count, INT_MAX));
        if (send) MPI_CHECK(MPI_Send(ptr, n, MPI_DOUBLE, peer, 20, comm));
        else MPI_CHECK(MPI_Recv(ptr, n, MPI_DOUBLE, peer, 20, comm, MPI_STATUS_IGNORE));
        ptr += n;
        count -= n;
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!worldRank) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, bad = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const char* value = argv[++i];
            long long n = strtoll(value, &end, 10);
            if (*value == '\0' || *end != '\0' || n < 0 ||
                (option == 'i' && n > INT_MAX)) { bad = true; break; }
            if (option == 'x') nx = static_cast<size_t>(n);
            if (option == 'y') ny = static_cast<size_t>(n);
            if (option == 'z') nz = static_cast<size_t>(n);
            if (option == 'i') iterations = static_cast<int>(n);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { help = true; break; }
        else { bad = true; break; }
    }
    if (help || bad) {
        if (!worldRank) { if (bad) fprintf(stderr, "Invalid command line arguments\n"); printUsage(argv[0]); }
        MPI_Finalize();
        return bad ? 1 : 0;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny * nz > SIZE_MAX / sizeof(Real) ||
        (nx > 2 && (nx - 2 + 31) / 32 > INT_MAX)) {
        if (!worldRank) fprintf(stderr, "Invalid or excessively large grid dimensions\n");
        MPI_Finalize();
        return 1;
    }

    // Ranks beyond the number of z planes are idle, rather than owning empty slabs.
    const int activeSize = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED,
                             worldRank, &comm));
    if (worldRank >= activeSize) { MPI_Finalize(); return 0; }
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm localComm;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    int devices = 0;
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", worldRank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    const size_t plane = nx * ny;
    const size_t localNz = nz / activeSize + (static_cast<size_t>(rank) < nz % activeSize);
    const size_t zStart = size_t(rank) * (nz / activeSize) + std::min<size_t>(rank, nz % activeSize);
    if (localNz > SIZE_MAX / plane - 2 || (localNz + 2) * plane > SIZE_MAX / sizeof(Real)) {
        fprintf(stderr, "Rank %d: local grid is too large\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t localElements = localNz * plane;
    const size_t bytes = (localNz + 2) * plane * sizeof(Real);
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> initial((localNz + 2) * plane, 0.0);
    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < localElements; ++p)
        initial[plane + p] = static_cast<Real>((zStart * plane + p) % 19);
    // The initial values are known globally, so the first step needs no MPI.
    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < plane; ++p) {
        if (rank > 0) initial[p] = static_cast<Real>(((zStart - 1) * plane + p) % 19);
        if (rank + 1 < activeSize)
            initial[(localNz + 1) * plane + p] = static_cast<Real>(((zStart + localNz) * plane + p) % 19);
    }
    Real *a, *b;
    CUDA(cudaMalloc(&a, bytes));
    CUDA(cudaMalloc(&b, bytes));
    CUDA(cudaMemcpy(a, initial.data(), bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(b, initial.data(), bytes, cudaMemcpyHostToDevice));
    std::vector<Real>().swap(initial);
    cudaStream_t compute, exchange;
    CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
    Real* staging = nullptr;
    if (activeSize > 1 && iterations > 1 && nx > 2 && ny > 2 && nz > 2) {
        if (plane > SIZE_MAX / (4 * sizeof(Real))) {
            fprintf(stderr, "Rank %d: halo planes are too large\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA(cudaMallocHost(&staging, 4 * plane * sizeof(Real)));
    }
    if (!rank) printf("Running stencil computation...\n");
    MPI_CHECK(MPI_Barrier(comm));
    double start = MPI_Wtime();
    Real* in = a;
    Real* out = b;
    for (int iter = 0; iter < iterations; ++iter) {
        // Only global interior planes are updated; all fixed faces were
        // initialized in both buffers and never overwritten.
        size_t first = zStart == 0 ? 2 : 1;
        size_t last = localNz - (zStart + localNz == nz ? 1 : 0);
        if (nx > 2 && ny > 2 && nz > 2) {
            if (activeSize == 1 || iter == 0) {
                if (first <= last) launch(in, out, nx, ny, first, last - first + 1, compute);
                // Subsequent multi-rank exchanges read this step's output on
                // another stream, so make it ready before swapping buffers.
                if (activeSize > 1) CUDA(cudaStreamSynchronize(compute));
            } else {
                std::vector<MPI_Request> requests;
                if (rank > 0) postChunks(staging + 2 * plane, plane, rank - 1, 11, false, comm, requests);
                if (rank + 1 < activeSize) postChunks(staging + 3 * plane, plane, rank + 1, 10, false, comm, requests);
                if (rank > 0) CUDA(cudaMemcpyAsync(staging, in + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, exchange));
                if (rank + 1 < activeSize) CUDA(cudaMemcpyAsync(staging + plane, in + localNz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, exchange));
                size_t deepFirst = std::max<size_t>(first, 2);
                size_t deepLast = std::min<size_t>(last, localNz - 1);
                if (deepFirst <= deepLast) launch(in, out, nx, ny, deepFirst, deepLast - deepFirst + 1, compute);
                CUDA(cudaStreamSynchronize(exchange));
                if (rank > 0) postChunks(staging, plane, rank - 1, 10, true, comm, requests);
                if (rank + 1 < activeSize) postChunks(staging + plane, plane, rank + 1, 11, true, comm, requests);
                MPI_CHECK(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
                if (rank > 0) CUDA(cudaMemcpyAsync(in, staging + 2 * plane, plane * sizeof(Real), cudaMemcpyHostToDevice, exchange));
                if (rank + 1 < activeSize) CUDA(cudaMemcpyAsync(in + (localNz + 1) * plane, staging + 3 * plane, plane * sizeof(Real), cudaMemcpyHostToDevice, exchange));
                if (first <= last && first == 1) launch(in, out, nx, ny, 1, 1, exchange);
                if (first <= last && last == localNz && localNz != 1) launch(in, out, nx, ny, localNz, 1, exchange);
                CUDA(cudaStreamSynchronize(exchange));
                CUDA(cudaStreamSynchronize(compute));
            }
        }
        std::swap(in, out);
    }
    CUDA(cudaStreamSynchronize(compute));
    MPI_CHECK(MPI_Barrier(comm));
    double elapsed = MPI_Wtime() - start, maxElapsed;
    MPI_CHECK(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (!rank) {
        long long ms = static_cast<long long>(maxElapsed * 1000.0);
        double updates = double(nx > 2 ? nx - 2 : 0) * double(ny > 2 ? ny - 2 : 0) *
                         double(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Computation time: %lld ms\nPerformance: %.3f MCellUpdates/s\n", ms, updates / maxElapsed / 1e6);
    }
    if (printResults || validate) {
        std::vector<Real> local(localElements);
        CUDA(cudaMemcpy(local.data(), in + plane, localElements * sizeof(Real), cudaMemcpyDeviceToHost));
        if (printResults) {
            std::vector<Real> global;
            if (!rank) {
                global.resize(nx * ny * nz);
                std::copy(local.begin(), local.end(), global.begin());
                for (int r = 1; r < activeSize; ++r) {
                    size_t count = nz / activeSize + (size_t(r) < nz % activeSize);
                    size_t offset = size_t(r) * (nz / activeSize) + std::min<size_t>(r, nz % activeSize);
                    transferChunks(global.data() + offset * plane, count * plane, r, false, comm);
                }
                print_results(global, "Grid");
            } else transferChunks(local.data(), localElements, 0, true, comm);
        }
        if (validate) {
            Real localMin = std::numeric_limits<Real>::infinity();
            Real localMax = -std::numeric_limits<Real>::infinity();
            int invalid = 0;
            #pragma omp parallel for reduction(min:localMin) reduction(max:localMax) reduction(|:invalid) schedule(static)
            for (size_t p = 0; p < localElements; ++p) {
                Real value = local[p];
                if (!std::isfinite(value)) invalid = 1;
                localMin = std::min(localMin, value);
                localMax = std::max(localMax, value);
            }
            Real minValue, maxValue;
            int anyInvalid;
            MPI_CHECK(MPI_Allreduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, comm));
            MPI_CHECK(MPI_Allreduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, comm));
            MPI_CHECK(MPI_Allreduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_MAX, comm));
            bad = anyInvalid || minValue < -1e6 || maxValue > 1e6;
            if (!rank) {
                printf("Validating result...\n");
                if (anyInvalid) printf("Validation failed: found NaN or Inf value\n");
                else {
                    printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
                    if (bad) printf("Validation failed: values out of expected range\n");
                }
                printf("Validation: %s\n", bad ? "FAILED" : "PASSED");
            }
        }
    }
    if (staging) CUDA(cudaFreeHost(staging));
    CUDA(cudaStreamDestroy(exchange));
    CUDA(cudaStreamDestroy(compute));
    CUDA(cudaFree(a));
    CUDA(cudaFree(b));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return bad ? 1 : 0;
}
