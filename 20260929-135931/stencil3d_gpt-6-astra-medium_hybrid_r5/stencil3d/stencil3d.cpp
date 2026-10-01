#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <cuda_runtime.h>
#include <omp.h>
#include "../common/results_output.hpp"

using Real = double;

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)
static void mpiCheck(int error, const char* operation) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "%s: %.*s\n", operation, length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define MPI(call) mpiCheck((call), #call)

// A warp spans contiguous X coordinates. Rolling Z registers reuse two of
// the three Z values; the read-only cache supplies neighboring X/Y values.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz,
                                 size_t globalStart, size_t first, size_t last) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (x >= nx - 1) return;
    const size_t plane = nx * ny;
    for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
         y < ny - 1; y += size_t(gridDim.y) * blockDim.y) {
        // Grid-stride Z tiles avoid CUDA's grid.z limit for long, thin grids.
        for (size_t begin = first + size_t(blockIdx.z) * 8; begin < last;
             begin += size_t(gridDim.z) * 8) {
            const size_t end = begin + 8 < last ? begin + 8 : last;
            size_t index = begin * plane + y * nx + x;
            Real bottom = input[index - plane];
            Real center = input[index];
            for (size_t z = begin; z < end; ++z, index += plane) {
                const Real top = input[index + plane];
                const size_t globalZ = globalStart + z - 1;
                if (globalZ > 0 && globalZ < nz - 1) {
                    output[index] = (center + input[index - 1] + input[index + 1]
                                   + input[index - nx] + input[index + nx]
                                   + bottom + top) / 7.0;
                }
                bottom = center;
                center = top;
            }
        }
    }
}

static void launch(const Real* input, Real* output, size_t nx, size_t ny,
                   size_t nz, size_t start, size_t first, size_t last,
                   cudaStream_t stream) {
    if (first >= last || nx < 3 || ny < 3 || nz < 3) return;
    const dim3 block(32, 4);
    const dim3 grid(unsigned((nx - 2 + 31) / 32), unsigned(std::min<size_t>((ny - 2 + 3) / 4, 65535)),
                    unsigned(std::min<size_t>((last - first + 7) / 8, 65535)));
    stencilIteration<<<grid, block, 0, stream>>>(input, output, nx, ny, nz, start, first, last);
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

static size_t number(const char* value) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long result = strtoull(value, &end, 10);
    if (value[0] == '-' || end == value || *end || errno ||
        result > std::numeric_limits<size_t>::max())
        throw std::runtime_error("Invalid nonnegative integer argument");
    return size_t(result);
}

// Chunk transfers so neither large slabs nor large planes overflow MPI counts.
static constexpr size_t mpiChunk = 1u << 26;
static void postHalo(Real* data, size_t count, int peer, int tag, bool receive,
                     MPI_Comm comm, std::vector<MPI_Request>& requests) {
    if (peer == MPI_PROC_NULL) return;
    for (size_t offset = 0; offset < count; offset += mpiChunk) {
        const int n = int(std::min(mpiChunk, count - offset));
        requests.emplace_back();
        if (receive)
            MPI(MPI_Irecv(data + offset, n, MPI_DOUBLE, peer, tag, comm, &requests.back()));
        else
            MPI(MPI_Isend(data + offset, n, MPI_DOUBLE, peer, tag, comm, &requests.back()));
    }
}

static int benchmark(int argc, char** argv, int worldRank, int worldSize) {
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            const size_t value = number(argv[++i]);
            if (option == 'x') nx = value;
            if (option == 'y') ny = value;
            if (option == 'z') nz = value;
            if (option == 'i') {
                if (value > INT_MAX) throw std::runtime_error("Too many iterations");
                iterations = int(value);
            }
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!worldRank) printUsage(argv[0]);
            return 0;
        } else {
            if (!worldRank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (!nx || !ny || !nz || nx > maxElements / ny || nx * ny > maxElements / nz)
        throw std::runtime_error("Invalid or overflowing grid dimensions");
    const size_t plane = nx * ny;
    const int ranks = int(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    MPI(MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm));
    if (worldRank >= ranks) return 0;
    const int rank = worldRank;
    const size_t localZ = nz / ranks + (size_t(rank) < nz % ranks);
    const size_t startZ = size_t(rank) * (nz / ranks) + std::min<size_t>(rank, nz % ranks);
    if (maxElements / plane < 2 || localZ > maxElements / plane - 2)
        throw std::runtime_error("Local grid including halos is too large");
    const size_t elements = (localZ + 2) * plane;
    const size_t bytes = elements * sizeof(Real), planeBytes = plane * sizeof(Real);
    const int lower = rank ? rank - 1 : MPI_PROC_NULL;
    const int upper = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;

    MPI_Comm node;
    MPI(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int nodeRank, devices;
    MPI(MPI_Comm_rank(node, &nodeRank));
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("A CUDA device is required on every active rank");
    CUDA(cudaSetDevice(nodeRank % devices));
    MPI(MPI_Comm_free(&node));

    // CUDA-aware MPI is used when the implementation advertises support.
    // Portable MPI implementations use pinned host staging instead.
    int direct = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    direct = MPIX_Query_cuda_support();
#endif
    MPI(MPI_Allreduce(MPI_IN_PLACE, &direct, 1, MPI_INT, MPI_MIN, comm));
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\n", nx, ny, nz, iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> host(elements);
    #pragma omp parallel for schedule(static)
    for (size_t z = 0; z < localZ + 2; ++z) {
        const size_t globalZ = z == 0 ? (startZ ? startZ - 1 : 0) : startZ + z - 1;
        for (size_t p = 0; p < plane; ++p)
            host[z * plane + p] = Real(((globalZ % 19) * (plane % 19) + p % 19) % 19);
    }
    Real *input, *output, *staging = nullptr;
    CUDA(cudaMalloc(&input, bytes));
    CUDA(cudaMalloc(&output, bytes));
    CUDA(cudaMemcpy(input, host.data(), bytes, cudaMemcpyHostToDevice));
    // Both buffers retain immutable physical boundaries, eliminating copies
    // of all six faces on every iteration.
    CUDA(cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice));
    if (ranks > 1 && !direct) CUDA(cudaMallocHost(&staging, 4 * planeBytes));
    cudaStream_t compute, exchange;
    CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
    cudaEvent_t edgesDone;
    CUDA(cudaEventCreateWithFlags(&edgesDone, cudaEventDisableTiming));
    std::vector<MPI_Request> requests;
    requests.reserve(4 * ((plane + mpiChunk - 1) / mpiChunk));
    if (!rank) printf("Running stencil computation...\n");
    MPI(MPI_Barrier(comm));
    const double begin = MPI_Wtime();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (ranks == 1) {
            launch(input, output, nx, ny, nz, startZ, 1, localZ + 1, compute);
        } else {
            requests.clear();
            Real* receiveLower = direct ? input : staging;
            Real* receiveUpper = direct ? input + (localZ + 1) * plane : staging + plane;
            Real* sendLower = direct ? input + plane : staging + 2 * plane;
            Real* sendUpper = direct ? input + localZ * plane : staging + 3 * plane;
            postHalo(receiveLower, plane, lower, 1, true, comm, requests);
            postHalo(receiveUpper, plane, upper, 0, true, comm, requests);
            // The bulk neither reads receive halos nor writes send planes.
            launch(input, output, nx, ny, nz, startZ, 2, localZ, compute);
            if (!direct) {
                if (lower != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(sendLower, input + plane, planeBytes, cudaMemcpyDeviceToHost, exchange));
                if (upper != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(sendUpper, input + localZ * plane, planeBytes, cudaMemcpyDeviceToHost, exchange));
                CUDA(cudaStreamSynchronize(exchange));
            }
            postHalo(sendLower, plane, lower, 0, false, comm, requests);
            postHalo(sendUpper, plane, upper, 1, false, comm, requests);
            MPI(MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
            if (!direct) {
                if (lower != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(input, receiveLower, planeBytes, cudaMemcpyHostToDevice, exchange));
                if (upper != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(input + (localZ + 1) * plane, receiveUpper, planeBytes, cudaMemcpyHostToDevice, exchange));
            }
            launch(input, output, nx, ny, nz, startZ, 1, 2, exchange);
            if (localZ > 1) launch(input, output, nx, ny, nz, startZ, localZ, localZ + 1, exchange);
            CUDA(cudaEventRecord(edgesDone, exchange));
            CUDA(cudaStreamWaitEvent(compute, edgesDone, 0));
        }
        // In the single-rank case stream ordering alone handles dependencies.
        if (ranks > 1) CUDA(cudaStreamSynchronize(compute));
        std::swap(input, output);
    }
    CUDA(cudaStreamSynchronize(compute));
    const double elapsed = MPI_Wtime() - begin;
    double seconds;
    MPI(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (!rank) {
        const double updates = double(nx > 2 ? nx - 2 : 0) * double(ny > 2 ? ny - 2 : 0)
                             * double(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0);
    }
    if (validate || printResults)
        CUDA(cudaMemcpy(host.data(), input + plane, localZ * planeBytes, cudaMemcpyDeviceToHost));
    if (printResults) {
        // Gather only for the original ordered external validation/hash output.
        std::vector<Real> full;
        if (!rank) {
            full.resize(nx * ny * nz);
            std::copy_n(host.data(), localZ * plane, full.data());
        }
        for (int source = 1; source < ranks; ++source) {
            const size_t count = (nz / ranks + (size_t(source) < nz % ranks)) * plane;
            const size_t offset = (size_t(source) * (nz / ranks) + std::min<size_t>(source, nz % ranks)) * plane;
            for (size_t p = 0; p < count; p += mpiChunk) {
                const int n = int(std::min(mpiChunk, count - p));
                if (!rank) MPI(MPI_Recv(full.data() + offset + p, n, MPI_DOUBLE, source, 2, comm, MPI_STATUS_IGNORE));
                else if (rank == source) MPI(MPI_Send(host.data() + p, n, MPI_DOUBLE, 0, 2, comm));
            }
        }
        if (!rank) print_results(full, "Grid");
    }
    int invalid = 0;
    if (validate) {
        Real minValue = std::numeric_limits<Real>::infinity();
        Real maxValue = -std::numeric_limits<Real>::infinity();
        #pragma omp parallel for reduction(min:minValue) reduction(max:maxValue) reduction(|:invalid) schedule(static)
        for (size_t i = 0; i < localZ * plane; ++i) {
            const Real value = host[i];
            invalid |= !std::isfinite(value);
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
        }
        MPI(MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, comm));
        MPI(MPI_Allreduce(MPI_IN_PLACE, &minValue, 1, MPI_DOUBLE, MPI_MIN, comm));
        MPI(MPI_Allreduce(MPI_IN_PLACE, &maxValue, 1, MPI_DOUBLE, MPI_MAX, comm));
        invalid |= minValue < -1e6 || maxValue > 1e6;
        if (!rank) {
            printf("Validating result...\nValue range: [%.6f, %.6f]\n", minValue, maxValue);
            printf("Validation: %s\n", invalid ? "FAILED" : "PASSED");
        }
    }
    CUDA(cudaEventDestroy(edgesDone));
    CUDA(cudaStreamDestroy(exchange));
    CUDA(cudaStreamDestroy(compute));
    if (staging) CUDA(cudaFreeHost(staging));
    CUDA(cudaFree(output));
    CUDA(cudaFree(input));
    MPI(MPI_Comm_free(&comm));
    return invalid;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, size, result = 1;
    MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
    try {
        result = benchmark(argc, argv, rank, size);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI(MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    MPI(MPI_Finalize());
    return result;
}
