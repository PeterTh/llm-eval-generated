#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>
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
#define CUDA_CHECK(call) cudaCheck((call), #call)

// Contiguous X accesses, with four Z points per thread and register reuse
// of the center and the two Z neighbors. Both buffers retain fixed boundaries.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output, size_t nx, size_t ny,
                              size_t first, size_t last) {
    const size_t x = 1 + size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = 1 + size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = first + size_t(blockIdx.z) * 4;
    if (x >= nx - 1 || y >= ny - 1 || z > last) return;
    const size_t plane = nx * ny;
    size_t i = z * plane + y * nx + x;
    Real bottom = input[i - plane];
    Real center = input[i];
    const size_t end = (z + 3 < last) ? z + 3 : last;
    for (size_t k = z; k <= end; ++k, i += plane) {
        const Real top = input[i + plane];
        // Preserve the original order of additions and double precision division.
        output[i] = (center + input[i - 1] + input[i + 1] + input[i - nx]
                     + input[i + nx] + bottom + top) / 7.0;
        bottom = center;
        center = top;
    }
}

static void launchStencil(const Real* input, Real* output, size_t nx, size_t ny,
                          size_t first, size_t last, cudaStream_t stream) {
    if (nx < 3 || ny < 3 || first > last) return;
    const dim3 threads(32, 4);
    // Split oversized Z ranges to respect CUDA grid dimension limits.
    for (size_t z = first; z <= last;) {
        const size_t end = std::min(last, z + size_t(65535) * 4 - 1);
        const dim3 blocks(static_cast<unsigned>((nx - 2 + 31) / 32),
                          static_cast<unsigned>((ny - 2 + 3) / 4),
                          static_cast<unsigned>((end - z + 4) / 4));
        stencilKernel<<<blocks, threads, 0, stream>>>(input, output, nx, ny, z, end);
        CUDA_CHECK(cudaGetLastError());
        z = end + 1;
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

static bool parseNumber(const char* text, size_t& value) {
    if (!*text || *text == '-') return false;
    errno = 0;
    char* end = nullptr;
    unsigned long long number = strtoull(text, &end, 10);
    if (errno || *end || number > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(number);
    return true;
}

// Chunked transfers avoid MPI's int count limit for large result arrays.
static void transferResults(Real* data, size_t count, int peer, bool send, MPI_Comm comm) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_DOUBLE, peer, 2, comm);
        else MPI_Recv(data, chunk, MPI_DOUBLE, peer, 2, comm, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI must support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 128, ny = 0, nz = 0, iterationCount = 10;
    bool validate = false, printResults = false, argumentsOK = true, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
                  !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            size_t* target = !strcmp(argv[i], "-x") ? &nx :
                             !strcmp(argv[i], "-y") ? &ny :
                             !strcmp(argv[i], "-z") ? &nz : &iterationCount;
            argumentsOK &= parseNumber(argv[++i], *target);
        } else argumentsOK = false;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t sizeMax = std::numeric_limits<size_t>::max();
    argumentsOK &= nx > 0 && ny > 0 && nz > 0 && iterationCount <= INT_MAX;
    if (argumentsOK) {
        argumentsOK = nx <= sizeMax / ny && nx * ny <= sizeMax / nz &&
                      nx * ny * nz <= sizeMax / sizeof(Real) &&
                      nx * ny <= INT_MAX && (ny - 1) / 4 < 65535;
    }
    if (help || !argumentsOK) {
        if (worldRank == 0) {
            if (!argumentsOK) fprintf(stderr, "Invalid options or unsupported grid size\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return help ? 0 : 1;
    }

    // No empty slabs: surplus ranks participate in finalization only.
    const int activeSize = static_cast<int>(std::min(nz, size_t(worldSize)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm nodeComm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank, devices = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&nodeComm);
    int cudaAware = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    cudaAware = MPIX_Query_cuda_support();
#endif
    // All ranks must use the same transport. Unrecognized MPI implementations
    // use portable pinned-memory staging rather than assuming GPU support.
    MPI_Allreduce(MPI_IN_PLACE, &cudaAware, 1, MPI_INT, MPI_MIN, comm);

    const size_t plane = nx * ny;
    const size_t localZ = nz / activeSize + (size_t(rank) < nz % activeSize);
    const size_t startZ = size_t(rank) * (nz / activeSize) +
                          std::min(size_t(rank), nz % activeSize);
    if (localZ > sizeMax / plane - 2 || (localZ + 2) * plane > sizeMax / sizeof(Real)) {
        fprintf(stderr, "Local grid size overflow\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t cells = (localZ + 2) * plane;
    const size_t bytes = cells * sizeof(Real), planeBytes = plane * sizeof(Real);
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %zu\nValidation: %s\n", iterationCount, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", activeSize, omp_get_max_threads());
        printf("Halo transport: %s\n", cudaAware ? "CUDA-aware MPI" : "pinned host memory");
        printf("Initializing grid...\n");
    }
    Real *input = nullptr, *output = nullptr, *halo = nullptr;
    CUDA_CHECK(cudaMalloc(&input, bytes));
    CUDA_CHECK(cudaMalloc(&output, bytes));
    std::vector<Real> host;
    try { host.resize(cells); }
    catch (...) { MPI_Abort(MPI_COMM_WORLD, 1); }
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localZ * plane; ++i)
        host[plane + i] = Real((startZ * plane + i) % 19);
    CUDA_CHECK(cudaMemcpy(input, host.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice));
    // Retain host storage only when an output/validation copy is needed.
    if (!validate && !printResults) std::vector<Real>().swap(host);
    cudaStream_t compute, exchange;
    int lowPriority, highPriority;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&compute, cudaStreamNonBlocking, lowPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&exchange, cudaStreamNonBlocking, highPriority));
    const int lower = rank ? rank - 1 : MPI_PROC_NULL;
    const int upper = rank + 1 < activeSize ? rank + 1 : MPI_PROC_NULL;
    if (activeSize > 1 && !cudaAware) CUDA_CHECK(cudaMallocHost(&halo, 4 * planeBytes));
    const size_t first = startZ == 0 ? 2 : 1;
    const size_t last = localZ - (startZ + localZ == nz ? 1 : 0);
    const bool hasInterior = nx >= 3 && ny >= 3 && first <= last;
    // Load/JIT the kernel before timing. Restore the initialized second buffer.
    if (hasInterior) launchStencil(input, output, nx, ny, first, last, compute);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice));
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double begin = MPI_Wtime();
    for (size_t iter = 0; iter < iterationCount; ++iter) {
        if (activeSize == 1 || !hasInterior) {
            if (hasInterior) launchStencil(input, output, nx, ny, first, last, compute);
            // Ranks with only fixed boundaries must still exchange with neighbors.
            if (activeSize == 1) { std::swap(input, output); continue; }
        }
        MPI_Request requests[4];
        Real* receiveLower = cudaAware ? input : halo + 2 * plane;
        Real* receiveUpper = cudaAware ? input + (localZ + 1) * plane : halo + 3 * plane;
        Real* sendLower = cudaAware ? input + plane : halo;
        Real* sendUpper = cudaAware ? input + localZ * plane : halo + plane;
        MPI_Irecv(receiveLower, int(plane), MPI_DOUBLE, lower, 1, comm, &requests[0]);
        MPI_Irecv(receiveUpper, int(plane), MPI_DOUBLE, upper, 0, comm, &requests[1]);
        if (!cudaAware && lower != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(halo, input + plane, planeBytes, cudaMemcpyDeviceToHost, exchange));
        if (!cudaAware && upper != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(halo + plane, input + localZ * plane, planeBytes, cudaMemcpyDeviceToHost, exchange));
        // Deep interior needs no remote planes; MPI progresses while this runs.
        if (hasInterior && localZ > 2)
            launchStencil(input, output, nx, ny, std::max(first, size_t(2)),
                          std::min(last, localZ - 1), compute);
        if (!cudaAware) CUDA_CHECK(cudaStreamSynchronize(exchange));
        MPI_Isend(sendLower, int(plane), MPI_DOUBLE, lower, 0, comm, &requests[2]);
        MPI_Isend(sendUpper, int(plane), MPI_DOUBLE, upper, 1, comm, &requests[3]);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        if (!cudaAware && lower != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(input, halo + 2 * plane, planeBytes, cudaMemcpyHostToDevice, exchange));
        if (!cudaAware && upper != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(input + (localZ + 1) * plane, halo + 3 * plane, planeBytes, cudaMemcpyHostToDevice, exchange));
        // This stream orders the edge kernels after the incoming halo copies.
        if (hasInterior) {
            if (first == 1) launchStencil(input, output, nx, ny, 1, 1, exchange);
            if (last == localZ && (localZ != 1 || first != 1))
                launchStencil(input, output, nx, ny, localZ, localZ, exchange);
        }
        CUDA_CHECK(cudaStreamSynchronize(exchange));
        CUDA_CHECK(cudaStreamSynchronize(compute));
        std::swap(input, output);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    double elapsed = MPI_Wtime() - begin, seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const double updates = double(nx > 2 ? nx - 2 : 0) * double(ny > 2 ? ny - 2 : 0)
                             * double(nz > 2 ? nz - 2 : 0) * double(iterationCount);
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0.0);
    }
    if (validate || printResults)
        CUDA_CHECK(cudaMemcpy(host.data() + plane, input + plane, localZ * planeBytes, cudaMemcpyDeviceToHost));
    if (printResults) {
        if (rank == 0) {
            std::vector<Real> result;
            try { result.resize(nx * ny * nz); }
            catch (...) { MPI_Abort(MPI_COMM_WORLD, 1); }
            std::copy_n(host.data() + plane, localZ * plane, result.data());
            for (int peer = 1; peer < activeSize; ++peer) {
                const size_t peerZ = nz / activeSize + (size_t(peer) < nz % activeSize);
                const size_t offset = size_t(peer) * (nz / activeSize) + std::min(size_t(peer), nz % activeSize);
                transferResults(result.data() + offset * plane, peerZ * plane, peer, false, comm);
            }
            print_results(result, "Grid");
        } else transferResults(host.data() + plane, localZ * plane, 0, true, comm);
    }
    int valid = 1;
    if (validate) {
        Real minimum = std::numeric_limits<Real>::infinity();
        Real maximum = -std::numeric_limits<Real>::infinity();
        int finite = 1;
        #pragma omp parallel for reduction(min:minimum) reduction(max:maximum) reduction(&:finite) schedule(static)
        for (size_t i = plane; i < (localZ + 1) * plane; ++i) {
            const Real value = host[i];
            finite &= std::isfinite(value);
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        Real globalMin, globalMax;
        MPI_Allreduce(&finite, &valid, 1, MPI_INT, MPI_MIN, comm);
        MPI_Allreduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);
        valid &= globalMin >= -1e6 && globalMax <= 1e6;
        if (rank == 0) {
            printf("Validating result...\n");
            if (!finite) printf("Validation failed: found NaN or Inf value\n");
            printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    if (halo) CUDA_CHECK(cudaFreeHost(halo));
    CUDA_CHECK(cudaStreamDestroy(exchange));
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
