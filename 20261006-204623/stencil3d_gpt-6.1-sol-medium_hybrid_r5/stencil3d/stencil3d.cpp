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
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
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
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fail(message);
    }
}

// Each thread advances along Z, retaining the vertical neighbors in registers.
// X-contiguous warps provide coalesced reads and writes. Fixed boundary values
// are initialized in both buffers and never need copying during an iteration.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t nz, size_t zOffset,
                              size_t first, size_t last) {
    const size_t plane = nx * ny;
    constexpr size_t depth = 8;
    for (size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
         y + 1 < ny; y += size_t(gridDim.y) * blockDim.y) {
        for (size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
             x + 1 < nx; x += size_t(gridDim.x) * blockDim.x) {
            for (size_t base = first + size_t(blockIdx.z) * depth;
                 base <= last; base += size_t(gridDim.z) * depth) {
                const size_t end = (base + depth - 1 < last) ? base + depth - 1 : last;
                size_t index = base * plane + y * nx + x;
                Real bottom = input[index - plane];
                Real center = input[index];
                for (size_t z = base; z <= end; ++z, index += plane) {
                    const Real top = input[index + plane];
                    const size_t globalZ = zOffset + z - 1;
                    if (globalZ > 0 && globalZ + 1 < nz) {
                        output[index] = (center + input[index - 1] + input[index + 1]
                            + input[index - nx] + input[index + nx] + bottom + top) / 7.0;
                    }
                    bottom = center;
                    center = top;
                }
            }
        }
    }
}

static void launch(const Real* input, Real* output, size_t nx, size_t ny,
                   size_t nz, size_t offset, size_t first, size_t last,
                   cudaStream_t stream) {
    if (first > last || nx < 3 || ny < 3 || nz < 3) return;
    const dim3 block(32, 4);
    const dim3 grid(static_cast<unsigned>(std::min<size_t>((nx - 2 + 31) / 32, 65535)),
                    static_cast<unsigned>(std::min<size_t>((ny - 2 + 3) / 4, 65535)),
                    static_cast<unsigned>(std::min<size_t>((last - first + 8) / 8, 65535)));
    stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, nz, offset, first, last);
    cudaCheck(cudaGetLastError());
}

static size_t slabSize(size_t nz, int ranks, int rank) {
    return nz / size_t(ranks) + (size_t(rank) < nz % size_t(ranks));
}

static size_t slabOffset(size_t nz, int ranks, int rank) {
    return size_t(rank) * (nz / size_t(ranks)) + std::min(size_t(rank), nz % size_t(ranks));
}

// Chunk MPI messages so grids are not limited by the legacy int count API.
static void postHalo(Real* data, size_t count, int peer, int tag, bool send,
                     MPI_Comm comm, std::vector<MPI_Request>& requests) {
    for (size_t at = 0; at < count;) {
        const int chunk = static_cast<int>(std::min<size_t>(count - at, INT_MAX));
        MPI_Request request;
        if (send) mpiCheck(MPI_Isend(data + at, chunk, MPI_DOUBLE, peer, tag, comm, &request));
        else mpiCheck(MPI_Irecv(data + at, chunk, MPI_DOUBLE, peer, tag, comm, &request));
        requests.push_back(request);
        at += size_t(chunk);
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
    if (!*text || *text == '-') fail("Invalid nonnegative integer argument");
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno || *end || value > std::numeric_limits<size_t>::max())
        fail("Invalid or overflowing integer argument");
    return static_cast<size_t>(value);
}

static int run(int argc, char** argv, int worldRank, int worldSize) {
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
                iterations = static_cast<int>(value);
            }
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!worldRank) printUsage(argv[0]);
            return 0;
        } else {
            if (!worldRank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz) fail("Grid dimensions must be positive");
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx > limit / ny || nx * ny > limit / nz) fail("Grid size overflows address space");
    const size_t plane = nx * ny;

    // Extra ranks stay outside the slab communicator when there are fewer
    // planes than ranks. Every nonempty slab is computed by CUDA.
    const int ranks = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED,
                           worldRank, &comm));
    if (worldRank >= ranks) return 0;
    const int rank = worldRank;
    MPI_Comm local;
    mpiCheck(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
    int localRank = 0, devices = 0;
    mpiCheck(MPI_Comm_rank(local, &localRank));
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) fail("The hybrid stencil requires a CUDA device on every compute node");
    cudaCheck(cudaSetDevice(localRank % devices));
    mpiCheck(MPI_Comm_free(&local));
    // Use device buffers directly when MPI advertises CUDA support, allowing
    // GPUDirect transports. Other MPI implementations use pinned staging.
    int deviceMPI = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    deviceMPI = MPIX_Query_cuda_support();
#endif
    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &deviceMPI, 1, MPI_INT, MPI_MIN, comm));

    const size_t owned = slabSize(nz, ranks, rank);
    const size_t offset = slabOffset(nz, ranks, rank);
    if (owned > limit / plane || limit / plane - owned < 2)
        fail("Local grid including halos overflows address space");
    const size_t cells = (owned + 2) * plane;
    const size_t bytes = cells * sizeof(Real), planeBytes = plane * sizeof(Real);
    const int lower = rank ? rank - 1 : MPI_PROC_NULL;
    const int upper = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d\n", ranks, omp_get_max_threads());
        printf("Initializing grid...\n");
    }
    std::vector<Real> host(cells);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells; ++i) {
        const size_t localZ = i / plane;
        if ((localZ == 0 && offset == 0) || (localZ == owned + 1 && offset + owned == nz))
            host[i] = 0.0; // Unused halos outside the physical domain.
        else host[i] = Real(((offset + localZ - 1) * plane + i % plane) % 19);
    }
    Real *input = nullptr, *output = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&input), bytes));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&output), bytes));
    cudaCheck(cudaMemcpy(input, host.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(output, host.data(), bytes, cudaMemcpyHostToDevice));
    // Portable pinned staging avoids requiring a CUDA-aware MPI installation.
    Real *sendLower = nullptr, *sendUpper = nullptr, *recvLower = nullptr, *recvUpper = nullptr;
    if (!deviceMPI && lower != MPI_PROC_NULL) {
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&sendLower), planeBytes));
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&recvLower), planeBytes));
    }
    if (!deviceMPI && upper != MPI_PROC_NULL) {
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&sendUpper), planeBytes));
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&recvUpper), planeBytes));
    }
    cudaStream_t compute, transfer;
    cudaEvent_t halosReady;
    cudaCheck(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    cudaCheck(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    cudaCheck(cudaEventCreateWithFlags(&halosReady, cudaEventDisableTiming));
    std::vector<MPI_Request> requests;
    requests.reserve(4 * (plane / size_t(INT_MAX) + 1));
    // Load the CUDA kernel before timing. Input remains at iteration zero.
    launch(input, output, nx, ny, nz, offset, 1, owned, compute);
    cudaCheck(cudaStreamSynchronize(compute));
    if (!rank) printf("Running stencil computation...\n");
    mpiCheck(MPI_Barrier(comm));
    const double start = MPI_Wtime();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (ranks == 1) {
            // No halo dependencies: queue all iterations without host waits.
            launch(input, output, nx, ny, nz, offset, 1, owned, compute);
            std::swap(input, output);
            continue;
        }
        requests.clear();
        if (lower != MPI_PROC_NULL)
            postHalo(deviceMPI ? input : recvLower, plane, lower, 1, false, comm, requests);
        if (upper != MPI_PROC_NULL)
            postHalo(deviceMPI ? input + (owned + 1) * plane : recvUpper,
                     plane, upper, 0, false, comm, requests);
        // The slab interior uses only local values and runs while halos travel.
        if (owned > 2) launch(input, output, nx, ny, nz, offset, 2, owned - 1, compute);
        if (!deviceMPI && lower != MPI_PROC_NULL)
            cudaCheck(cudaMemcpyAsync(sendLower, input + plane, planeBytes, cudaMemcpyDeviceToHost, transfer));
        if (!deviceMPI && upper != MPI_PROC_NULL)
            cudaCheck(cudaMemcpyAsync(sendUpper, input + owned * plane, planeBytes, cudaMemcpyDeviceToHost, transfer));
        if (!deviceMPI) cudaCheck(cudaStreamSynchronize(transfer));
        if (lower != MPI_PROC_NULL)
            postHalo(deviceMPI ? input + plane : sendLower, plane, lower, 0, true, comm, requests);
        if (upper != MPI_PROC_NULL)
            postHalo(deviceMPI ? input + owned * plane : sendUpper,
                     plane, upper, 1, true, comm, requests);
        if (!requests.empty())
            mpiCheck(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
        if (!deviceMPI && lower != MPI_PROC_NULL)
            cudaCheck(cudaMemcpyAsync(input, recvLower, planeBytes, cudaMemcpyHostToDevice, transfer));
        if (!deviceMPI && upper != MPI_PROC_NULL)
            cudaCheck(cudaMemcpyAsync(input + (owned + 1) * plane, recvUpper, planeBytes, cudaMemcpyHostToDevice, transfer));
        if (!deviceMPI) {
            cudaCheck(cudaEventRecord(halosReady, transfer));
            cudaCheck(cudaStreamWaitEvent(compute, halosReady, 0));
        }
        launch(input, output, nx, ny, nz, offset, 1, 1, compute);
        if (owned > 1) launch(input, output, nx, ny, nz, offset, owned, owned, compute);
        cudaCheck(cudaStreamSynchronize(compute));
        std::swap(input, output);
    }
    cudaCheck(cudaStreamSynchronize(compute));
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0;
    mpiCheck(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (!rank) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        const double updates = double(nx > 2 ? nx - 2 : 0) * double(ny > 2 ? ny - 2 : 0)
                             * double(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0.0);
    }
    if (printResults || validate)
        cudaCheck(cudaMemcpy(host.data(), input + plane, owned * planeBytes, cudaMemcpyDeviceToHost));
    if (printResults) {
        std::vector<Real> full;
        if (!rank) full.resize(plane * nz);
        // Ordered chunked gathering preserves the original row-major result
        // and permits counts larger than MPI_Gatherv's int displacement range.
        for (int source = 0; source < ranks; ++source) {
            const size_t count = slabSize(nz, ranks, source) * plane;
            const size_t destination = slabOffset(nz, ranks, source) * plane;
            if (!source && !rank) std::copy_n(host.data(), count, full.data());
            else if (source && (rank == source || rank == 0)) {
                for (size_t at = 0; at < count;) {
                    const int chunk = static_cast<int>(std::min<size_t>(count - at, INT_MAX));
                    if (!rank) mpiCheck(MPI_Recv(full.data() + destination + at, chunk, MPI_DOUBLE,
                                               source, 2, comm, MPI_STATUS_IGNORE));
                    else mpiCheck(MPI_Send(host.data() + at, chunk, MPI_DOUBLE, 0, 2, comm));
                    at += size_t(chunk);
                }
            }
        }
        if (!rank) print_results(full, "Grid");
    }
    int valid = 1;
    if (validate) {
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
        #pragma omp parallel for reduction(min:minimum) reduction(max:maximum) reduction(&:valid) schedule(static)
        for (size_t i = 0; i < owned * plane; ++i) {
            const Real value = host[i];
            valid &= std::isfinite(value) ? 1 : 0;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            const size_t z = offset + i / plane, y = (i % plane) / nx, x = i % nx;
            if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz)
                valid &= value == Real(((offset * plane + i) % 19)) ? 1 : 0;
        }
        double globalMinimum, globalMaximum;
        int globalValid;
        mpiCheck(MPI_Allreduce(&minimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN, comm));
        mpiCheck(MPI_Allreduce(&maximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX, comm));
        mpiCheck(MPI_Allreduce(&valid, &globalValid, 1, MPI_INT, MPI_MIN, comm));
        valid = globalValid && globalMinimum >= -1e6 && globalMaximum <= 1e6;
        if (!rank) {
            printf("Validating result...\nValue range: [%.6f, %.6f]\n", globalMinimum, globalMaximum);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    cudaCheck(cudaEventDestroy(halosReady));
    cudaCheck(cudaStreamDestroy(compute));
    cudaCheck(cudaStreamDestroy(transfer));
    cudaCheck(cudaFree(input));
    cudaCheck(cudaFree(output));
    if (sendLower) cudaCheck(cudaFreeHost(sendLower));
    if (sendUpper) cudaCheck(cudaFreeHost(sendUpper));
    if (recvLower) cudaCheck(cudaFreeHost(recvLower));
    if (recvUpper) cudaCheck(cudaFreeHost(recvUpper));
    mpiCheck(MPI_Comm_free(&comm));
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int rank, ranks;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int status = 1;
    try { status = run(argc, argv, rank, ranks); }
    catch (const std::exception& error) { fail(error.what()); }
    int globalStatus = 0;
    mpiCheck(MPI_Allreduce(&status, &globalStatus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    mpiCheck(MPI_Finalize());
    return globalStatus;
}
