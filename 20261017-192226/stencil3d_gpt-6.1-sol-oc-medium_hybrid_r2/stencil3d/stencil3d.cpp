#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

static void fail(const char* message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) fail(cudaGetErrorString(error));
}

static void mpiCheck(int error) {
    if (error != MPI_SUCCESS) fail("MPI operation failed");
}

// Each thread walks a short Z column, retaining the two neighboring Z
// values in registers. X is contiguous across a warp for coalesced accesses.
// Only interior cells are written: both buffers keep fixed physical faces.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t nz,
                              size_t zOffset, size_t begin, size_t end) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t first = begin + size_t(blockIdx.z) * 8;
    const size_t last = first + 8 < end ? first + 8 : end;
    if (x >= nx - 1 || y >= ny - 1 || first >= last) return;
    const size_t plane = nx * ny;
    size_t p = first * plane + y * nx + x;
    Real bottom = input[p - plane];
    Real center = input[p];
    for (size_t z = first; z < last; ++z, p += plane) {
        const Real top = input[p + plane];
        const size_t globalZ = zOffset + z - 1;
        if (globalZ > 0 && globalZ < nz - 1) {
            output[p] = (center + input[p - 1] + input[p + 1]
                         + input[p - nx] + input[p + nx] + bottom + top) / 7.0;
        }
        bottom = center;
        center = top;
    }
}

static void launchStencil(const Real* input, Real* output,
                          size_t nx, size_t ny, size_t nz, size_t offset,
                          size_t begin, size_t end, cudaStream_t stream) {
    if (begin >= end || nx < 3 || ny < 3 || nz < 3) return;
    // Limit each launch's Z dimension to CUDA's grid limit.
    const dim3 block(32, 4);
    while (begin < end) {
        const size_t stop = std::min(end, begin + size_t(65535) * 8);
        const dim3 blocks((nx - 2 + 31) / 32, (ny - 2 + 3) / 4,
                          (stop - begin + 7) / 8);
        stencilKernel<<<blocks, block, 0, stream>>>(input, output, nx, ny, nz,
                                                   offset, begin, stop);
        cudaCheck(cudaGetLastError());
        begin = stop;
    }
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

static size_t parseNumber(const char* text) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (text[0] == '-' || end == text || *end || errno ||
        value > std::numeric_limits<size_t>::max()) fail("Invalid numeric argument");
    return static_cast<size_t>(value);
}

static size_t slabSize(size_t nz, int ranks, int rank) {
    return nz / ranks + (size_t(rank) < nz % ranks);
}

static size_t slabOffset(size_t nz, int ranks, int rank) {
    return size_t(rank) * (nz / ranks) + std::min(size_t(rank), nz % ranks);
}

// Gather only on request, in global Z/Y/X order. Chunking avoids MPI's
// int-count limit for large local volumes without requiring MPI-4.
static void printGlobalResults(const Real* local, size_t nx, size_t ny,
                               size_t nz, int rank, int ranks, MPI_Comm comm) {
    const size_t plane = nx * ny;
    const size_t count = slabSize(nz, ranks, rank) * plane;
    std::vector<Real> global;
    if (rank == 0) {
        global.resize(plane * nz);
        std::copy(local, local + count, global.data());
        for (int peer = 1; peer < ranks; ++peer) {
            const size_t peerCount = slabSize(nz, ranks, peer) * plane;
            Real* destination = global.data() + slabOffset(nz, ranks, peer) * plane;
            for (size_t p = 0; p < peerCount;) {
                const int n = static_cast<int>(std::min(size_t(INT_MAX), peerCount - p));
                mpiCheck(MPI_Recv(destination + p, n, MPI_DOUBLE, peer, 2, comm,
                                  MPI_STATUS_IGNORE));
                p += n;
            }
        }
        print_results(global, "Grid");
    } else {
        for (size_t p = 0; p < count;) {
            const int n = static_cast<int>(std::min(size_t(INT_MAX), count - p));
            mpiCheck(MPI_Send(local + p, n, MPI_DOUBLE, 0, 2, comm));
            p += n;
        }
    }
}

static int run(size_t nx, size_t ny, size_t nz, int iterations,
               bool validate, bool printResults, MPI_Comm comm) {
    int rank, ranks;
    mpiCheck(MPI_Comm_rank(comm, &rank));
    mpiCheck(MPI_Comm_size(comm, &ranks));
    MPI_Comm node;
    mpiCheck(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank;
    mpiCheck(MPI_Comm_rank(node, &localRank));
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) fail("A CUDA device is required");
    cudaCheck(cudaSetDevice(localRank % devices));
    mpiCheck(MPI_Comm_free(&node));

    const size_t plane = nx * ny;
    const size_t owned = slabSize(nz, ranks, rank);
    const size_t offset = slabOffset(nz, ranks, rank);
    if (owned > std::numeric_limits<size_t>::max() / plane - 2 ||
        (owned + 2) * plane > std::numeric_limits<size_t>::max() / sizeof(Real))
        fail("Local grid is too large");
    const size_t cells = (owned + 2) * plane;
    const size_t bytes = cells * sizeof(Real);
    std::vector<Real> host(cells);
    // Initialize by GLOBAL index, including internal ghosts, so decomposition
    // never changes the benchmark's initial conditions.
    #pragma omp parallel for schedule(static)
    for (size_t z = 0; z < owned + 2; ++z) {
        const size_t globalZ = z == 0 ? (offset ? offset - 1 : 0)
                                      : std::min(nz - 1, offset + z - 1);
        #pragma omp simd
        for (size_t p = 0; p < plane; ++p)
            host[z * plane + p] = Real((globalZ * plane + p) % 19);
    }
    Real *input = nullptr, *output = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&input), bytes));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&output), bytes));
    cudaCheck(cudaMemcpy(input, host.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice));

    cudaStream_t compute, halo;
    cudaCheck(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    cudaCheck(cudaStreamCreateWithFlags(&halo, cudaStreamNonBlocking));
    Real* staging = nullptr;
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    if (ranks > 1)
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&staging), 4 * plane * sizeof(Real)));
    const size_t planeBytes = plane * sizeof(Real);

    if (rank == 0) printf("Running stencil computation...\n");
    cudaCheck(cudaDeviceSynchronize());
    mpiCheck(MPI_Barrier(comm));
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (ranks == 1) {
            launchStencil(input, output, nx, ny, nz, offset, 1, owned + 1, compute);
            // Stream ordering suffices; no per-iteration host synchronization.
        } else {
            MPI_Request requests[4];
            mpiCheck(MPI_Irecv(staging + 2 * plane, int(plane), MPI_DOUBLE, lower, 1,
                               comm, &requests[0]));
            mpiCheck(MPI_Irecv(staging + 3 * plane, int(plane), MPI_DOUBLE, upper, 0,
                               comm, &requests[1]));
            if (lower != MPI_PROC_NULL)
                cudaCheck(cudaMemcpyAsync(staging, input + plane, planeBytes,
                                           cudaMemcpyDeviceToHost, halo));
            if (upper != MPI_PROC_NULL)
                cudaCheck(cudaMemcpyAsync(staging + plane, input + owned * plane,
                                           planeBytes, cudaMemcpyDeviceToHost, halo));
            // Interior work overlaps D2H copies, network transfer and H2D copies.
            launchStencil(input, output, nx, ny, nz, offset, 2, owned, compute);
            cudaCheck(cudaStreamSynchronize(halo));
            mpiCheck(MPI_Isend(staging, int(plane), MPI_DOUBLE, lower, 0,
                               comm, &requests[2]));
            mpiCheck(MPI_Isend(staging + plane, int(plane), MPI_DOUBLE, upper, 1,
                               comm, &requests[3]));
            mpiCheck(MPI_Waitall(2, requests, MPI_STATUSES_IGNORE));
            if (lower != MPI_PROC_NULL)
                cudaCheck(cudaMemcpyAsync(input, staging + 2 * plane, planeBytes,
                                           cudaMemcpyHostToDevice, halo));
            if (upper != MPI_PROC_NULL)
                cudaCheck(cudaMemcpyAsync(input + (owned + 1) * plane, staging + 3 * plane,
                                           planeBytes, cudaMemcpyHostToDevice, halo));
            launchStencil(input, output, nx, ny, nz, offset, 1, 2, halo);
            if (owned > 1)
                launchStencil(input, output, nx, ny, nz, offset, owned, owned + 1, halo);
            mpiCheck(MPI_Waitall(2, requests + 2, MPI_STATUSES_IGNORE));
            cudaCheck(cudaStreamSynchronize(halo));
            cudaCheck(cudaStreamSynchronize(compute));
        }
        std::swap(input, output);
    }
    cudaCheck(cudaDeviceSynchronize());
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0;
    mpiCheck(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
        const double updates = double(nx > 2 ? nx - 2 : 0) *
                               double(ny > 2 ? ny - 2 : 0) *
                               double(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0);
    }
    if (validate || printResults)
        cudaCheck(cudaMemcpy(host.data() + plane, input + plane, owned * planeBytes,
                              cudaMemcpyDeviceToHost));
    if (printResults)
        printGlobalResults(host.data() + plane, nx, ny, nz, rank, ranks, comm);

    int valid = 1;
    if (validate) {
        Real minVal = std::numeric_limits<Real>::infinity();
        Real maxVal = -std::numeric_limits<Real>::infinity();
        int bad = 0;
        #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(|:bad) schedule(static)
        for (size_t p = 0; p < owned * plane; ++p) {
            const Real value = host[plane + p];
            bad |= !std::isfinite(value);
            minVal = std::min(minVal, value);
            maxVal = std::max(maxVal, value);
            const size_t global = offset * plane + p;
            const size_t x = p % nx, y = (p / nx) % ny, z = global / plane;
            if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1)
                bad |= value != Real(global % 19);
        }
        Real globalMin, globalMax;
        int globalBad;
        mpiCheck(MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm));
        mpiCheck(MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm));
        mpiCheck(MPI_Allreduce(&bad, &globalBad, 1, MPI_INT, MPI_BOR, comm));
        valid = !globalBad && globalMin >= -1e6 && globalMax <= 1e6;
        if (rank == 0) {
            printf("Validating result...\n");
            printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    if (staging) cudaCheck(cudaFreeHost(staging));
    cudaCheck(cudaStreamDestroy(halo));
    cudaCheck(cudaStreamDestroy(compute));
    cudaCheck(cudaFree(input));
    cudaCheck(cudaFree(output));
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
    int rank, ranks;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            const size_t value = parseNumber(argv[++i]);
            if (option == 'x') nx = value;
            else if (option == 'y') ny = value;
            else if (option == 'z') nz = value;
            else {
                if (value > INT_MAX) fail("Iteration count exceeds INT_MAX");
                iterations = int(value);
            }
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
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
    if (!nx || !ny || !nz) fail("Grid dimensions must be positive");
    const size_t limit = std::numeric_limits<size_t>::max();
    if (nx > limit / ny || nx * ny > limit / nz || nx * ny * nz > limit / sizeof(Real))
        fail("Grid size overflows address space");
    if (nx * ny > INT_MAX) fail("A halo plane must fit an MPI int count");
    if ((nx > 2 && (nx - 2 + 31) / 32 > INT_MAX) ||
        (ny > 2 && (ny - 2 + 3) / 4 > 65535)) fail("Grid exceeds CUDA launch dimensions");
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    // Surplus ranks have no slab; keep them out of halo collectives.
    MPI_Comm active;
    const int activeRanks = static_cast<int>(std::min(nz, size_t(ranks)));
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED,
                            rank, &active));
    int status = 0;
    if (active != MPI_COMM_NULL) {
        try {
            status = run(nx, ny, nz, iterations, validate, printResults, active);
        } catch (const std::bad_alloc&) {
            fail("Host grid allocation failed");
        }
        mpiCheck(MPI_Comm_free(&active));
    }
    mpiCheck(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    mpiCheck(MPI_Finalize());
    return status;
}
