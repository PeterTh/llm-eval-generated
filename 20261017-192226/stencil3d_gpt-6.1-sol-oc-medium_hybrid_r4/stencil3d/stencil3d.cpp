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
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// Each warp accesses adjacent X coordinates. Rolling Z registers reuse the
// center and Z neighbors while keeping the working set small.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output, size_t nx, size_t ny,
                                 size_t nz, size_t startZ, size_t first, size_t last) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z0 = first + size_t(blockIdx.z) * 8;
    if (x >= nx || y >= ny || z0 > last) return;
    const size_t plane = nx * ny;
    size_t index = z0 * plane + y * nx + x;
    Real bottom = input[index - plane];
    Real center = input[index];
    for (size_t z = z0; z <= last && z < z0 + 8; ++z, index += plane) {
        const Real top = input[index + plane];
        const size_t globalZ = startZ + z - 1;
        if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
            globalZ == 0 || globalZ + 1 == nz) {
            output[index] = center;
        } else {
            output[index] = (center + input[index - 1] + input[index + 1] +
                             input[index - nx] + input[index + nx] + bottom + top) / 7.0;
        }
        bottom = center;
        center = top;
    }
}

static void launch(const Real* input, Real* output, size_t nx, size_t ny,
                   size_t nz, size_t startZ, size_t first, size_t last,
                   cudaStream_t stream) {
    if (first > last) return;
    // Limit grid.z to the CUDA architectural limit, even for very deep slabs.
    for (size_t begin = first; begin <= last;) {
        const size_t end = std::min(last, begin + size_t(65535) * 8 - 1);
        const dim3 block(32, 4);
        const dim3 grid((nx + 31) / 32, (ny + 3) / 4, (end - begin + 8) / 8);
        stencilIteration<<<grid, block, 0, stream>>>(input, output, nx, ny, nz,
                                                    startZ, begin, end);
        CUDA(cudaGetLastError());
        begin = end + 1;
    }
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseNumber(const char* text, size_t& value) {
    if (*text == '-' || *text == '\0') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long number = strtoull(text, &end, 10);
    if (errno || *end || number > std::numeric_limits<size_t>::max()) return false;
    value = size_t(number);
    return true;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    size_t nx = 128, ny = 0, nz = 0, iterationCount = 10;
    bool validate = false, printResults = false, argumentsOK = true;
    for (int i = 1; i < argc; ++i) {
        size_t* target = nullptr;
        if (!strcmp(argv[i], "-x")) target = &nx;
        else if (!strcmp(argv[i], "-y")) target = &ny;
        else if (!strcmp(argv[i], "-z")) target = &nz;
        else if (!strcmp(argv[i], "-i")) target = &iterationCount;
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else argumentsOK = false;
        if (target && (i + 1 >= argc || !parseNumber(argv[++i], *target))) argumentsOK = false;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (!argumentsOK || !nx || !ny || !nz || nx > limit / ny ||
        nx * ny > limit / nz || nx * ny > size_t(INT_MAX) || ny > size_t(65535) * 4 ||
        nx > size_t(INT_MAX) || iterationCount > size_t(INT_MAX)) {
        if (worldRank == 0) {
            fprintf(stderr, "Invalid arguments or grid dimensions exceed supported limits.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Excess ranks do not own empty slabs or participate in halo exchange.
    const int activeSize = int(std::min(nz, size_t(worldSize)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const int rank = worldRank;
    MPI_Comm nodeComm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank, devices;
    MPI_Comm_rank(nodeComm, &localRank);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "A CUDA device is required on every participating node.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&nodeComm);
    // Use device buffers directly when MPI advertises CUDA support. Other MPI
    // implementations retain the portable, asynchronous pinned-host path.
    int cudaAware = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    cudaAware = MPIX_Query_cuda_support();
#endif
    MPI_Allreduce(MPI_IN_PLACE, &cudaAware, 1, MPI_INT, MPI_MIN, comm);

    const size_t localZ = nz / activeSize + (size_t(rank) < nz % activeSize);
    const size_t startZ = size_t(rank) * (nz / activeSize) +
                          std::min(size_t(rank), nz % activeSize);
    const size_t plane = nx * ny, owned = localZ * plane;
    if (localZ > limit / plane || limit / plane - localZ < 2) MPI_Abort(MPI_COMM_WORLD, 1);
    const size_t bytes = (localZ + 2) * plane * sizeof(Real);
    const size_t planeBytes = plane * sizeof(Real);
    const int below = rank ? rank - 1 : MPI_PROC_NULL;
    const int above = rank + 1 < activeSize ? rank + 1 : MPI_PROC_NULL;
    Real *host, *input, *output, *halos;
    CUDA(cudaMallocHost(&host, bytes));
    CUDA(cudaMallocHost(&halos, 4 * planeBytes));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < owned; ++i)
        host[plane + i] = Real((startZ * plane + i) % 19);
    // Physical exterior ghosts are never used in the stencil calculation.
    std::fill(host, host + plane, 0.0);
    std::fill(host + plane + owned, host + (localZ + 2) * plane, 0.0);
    CUDA(cudaMalloc(&input, bytes));
    CUDA(cudaMalloc(&output, bytes));
    CUDA(cudaMemcpy(input, host, bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemset(output, 0, bytes));
    cudaStream_t compute, transfer;
    CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %zu\nValidation: %s\n", iterationCount, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d\n", activeSize, omp_get_max_threads());
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (size_t iter = 0; iter < iterationCount; ++iter) {
        if (activeSize == 1) {
            // Stream ordering alone handles dependencies when no MPI halos are
            // needed: enqueue the complete run without per-step host waits.
            launch(input, output, nx, ny, nz, startZ, 1, localZ, compute);
            std::swap(input, output);
            continue;
        }
        MPI_Request requests[4];
        int count = 0;
        if (below != MPI_PROC_NULL)
            MPI_Irecv(cudaAware ? input : halos + 2 * plane,
                      int(plane), MPI_DOUBLE, below, 1, comm, &requests[count++]);
        if (above != MPI_PROC_NULL)
            MPI_Irecv(cudaAware ? input + (localZ + 1) * plane : halos + 3 * plane,
                      int(plane), MPI_DOUBLE, above, 0, comm, &requests[count++]);
        // Interior work does not depend on received halo planes.
        if (localZ > 2) launch(input, output, nx, ny, nz, startZ, 2, localZ - 1, compute);
        if (!cudaAware && below != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(halos, input + plane, planeBytes, cudaMemcpyDeviceToHost, transfer));
        if (!cudaAware && above != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(halos + plane, input + localZ * plane, planeBytes, cudaMemcpyDeviceToHost, transfer));
        if (!cudaAware) CUDA(cudaStreamSynchronize(transfer));
        if (below != MPI_PROC_NULL)
            MPI_Isend(cudaAware ? input + plane : halos,
                      int(plane), MPI_DOUBLE, below, 0, comm, &requests[count++]);
        if (above != MPI_PROC_NULL)
            MPI_Isend(cudaAware ? input + localZ * plane : halos + plane,
                      int(plane), MPI_DOUBLE, above, 1, comm, &requests[count++]);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        if (!cudaAware && below != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(input, halos + 2 * plane, planeBytes, cudaMemcpyHostToDevice, transfer));
        if (!cudaAware && above != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(input + (localZ + 1) * plane, halos + 3 * plane,
                                 planeBytes, cudaMemcpyHostToDevice, transfer));
        launch(input, output, nx, ny, nz, startZ, 1, 1, transfer);
        if (localZ > 1) launch(input, output, nx, ny, nz, startZ, localZ, localZ, transfer);
        CUDA(cudaStreamSynchronize(compute));
        CUDA(cudaStreamSynchronize(transfer));
        std::swap(input, output);
    }
    CUDA(cudaStreamSynchronize(compute));
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const double updates = double(nx > 2 ? nx - 2 : 0) * double(ny > 2 ? ny - 2 : 0) *
                               double(nz > 2 ? nz - 2 : 0) * double(iterationCount);
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0.0);
    }
    if (validate || printResults)
        CUDA(cudaMemcpy(host + plane, input + plane, owned * sizeof(Real), cudaMemcpyDeviceToHost));

    if (printResults) {
        // Chunked rank-ordered gather preserves the original global linear order
        // without MPI_Gatherv's int displacement/total-size restrictions.
        std::vector<Real> fullGrid;
        if (rank == 0) {
            fullGrid.resize(plane * nz);
            std::copy(host + plane, host + plane + owned, fullGrid.begin());
        }
        for (int source = 1; source < activeSize; ++source) {
            const size_t depth = nz / activeSize + (size_t(source) < nz % activeSize);
            const size_t offset = (size_t(source) * (nz / activeSize) +
                                   std::min(size_t(source), nz % activeSize)) * plane;
            for (size_t i = 0; i < depth * plane;) {
                const int chunk = int(std::min(depth * plane - i, size_t(INT_MAX)));
                if (rank == source) MPI_Send(host + plane + i, chunk, MPI_DOUBLE, 0, 2, comm);
                if (rank == 0) MPI_Recv(fullGrid.data() + offset + i, chunk, MPI_DOUBLE,
                                        source, 2, comm, MPI_STATUS_IGNORE);
                i += size_t(chunk);
            }
        }
        if (rank == 0) print_results(fullGrid, "Grid");
    }

    int invalid = 0;
    if (validate) {
        Real minimum = std::numeric_limits<Real>::infinity();
        Real maximum = -minimum;
        #pragma omp parallel for reduction(min:minimum) reduction(max:maximum) reduction(|:invalid) schedule(static)
        for (size_t i = 0; i < owned; ++i) {
            const Real value = host[plane + i];
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            if (!std::isfinite(value) || value < -1e6 || value > 1e6) invalid = 1;
            const size_t global = startZ * plane + i;
            const size_t x = global % nx, y = (global / nx) % ny, z = global / plane;
            if ((x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz) &&
                value != Real(global % 19)) invalid = 1;
        }
        MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, comm);
        Real globalMin, globalMax;
        MPI_Reduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            printf("Validating result...\nValue range: [%.6f, %.6f]\n", globalMin, globalMax);
            printf("Validation: %s\n", invalid ? "FAILED" : "PASSED");
        }
    }
    CUDA(cudaStreamDestroy(compute));
    CUDA(cudaStreamDestroy(transfer));
    CUDA(cudaFree(input));
    CUDA(cudaFree(output));
    CUDA(cudaFreeHost(host));
    CUDA(cudaFreeHost(halos));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return invalid ? 1 : 0;
}
