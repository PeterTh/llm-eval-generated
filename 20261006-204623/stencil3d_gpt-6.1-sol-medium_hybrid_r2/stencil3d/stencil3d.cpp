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
#if defined(OPEN_MPI)
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

// Each warp spans contiguous X coordinates. Register rolling in Z reuses
// the center and vertical neighbors across eight output planes.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t first, size_t last,
                              size_t firstY) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + firstY;
    const size_t begin = first + size_t(blockIdx.z) * 8;
    if (x >= nx - 1 || y >= ny - 1 || begin > last) return;
    const size_t plane = nx * ny;
    const size_t end = (begin + 7 < last) ? begin + 7 : last;
    size_t index = begin * plane + y * nx + x;
    Real bottom = input[index - plane];
    Real center = input[index];
    for (size_t z = begin; z <= end; ++z, index += plane) {
        const Real top = input[index + plane];
        // Keep the original summation order and double precision division.
        output[index] = (center + input[index - 1] + input[index + 1]
                         + input[index - nx] + input[index + nx]
                         + bottom + top) / 7.0;
        bottom = center;
        center = top;
    }
}

static void launchStencil(const Real* input, Real* output, size_t nx,
                          size_t ny, size_t first, size_t last,
                          cudaStream_t stream) {
    if (first > last || nx < 3 || ny < 3) return;
    // Limit the Z grid dimension even for very long, narrow domains.
    const size_t maxPlanes = 8 * size_t(65535);
    for (size_t begin = first; begin <= last;) {
        const size_t count = std::min(maxPlanes, last - begin + 1);
        for (size_t row = 1; row < ny - 1;) {
            const size_t rows = std::min(size_t(8 * 65535), ny - 1 - row);
            dim3 block(32, 8);
            dim3 grid((nx - 2 + 31) / 32, (rows + 7) / 8, (count + 7) / 8);
            stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, begin, begin + count - 1, row);
            CUDA_CHECK(cudaGetLastError());
            row += rows;
        }
        begin += count;
    }
}

// Chunk messages to avoid MPI's int-count limit. Ordering is preserved within
// each neighbor/tag pair, including for the final result collection.
static void postTransfer(Real* buffer, size_t count, int peer, int tag,
                         bool receive, MPI_Comm comm, std::vector<MPI_Request>& requests) {
    if (peer == MPI_PROC_NULL) return;
    while (count) {
        int chunk = int(std::min(count, size_t(INT_MAX)));
        MPI_Request request;
        if (receive)
            MPI_Irecv(buffer, chunk, MPI_DOUBLE, peer, tag, comm, &request);
        else
            MPI_Isend(buffer, chunk, MPI_DOUBLE, peer, tag, comm, &request);
        requests.push_back(request);
        buffer += chunk;
        count -= chunk;
    }
}

static bool parseNumber(const char* text, size_t& value) {
    if (!text[0] || text[0] == '-') return false;
    char* end = nullptr;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || *end || parsed > std::numeric_limits<size_t>::max()) return false;
    value = size_t(parsed);
    return true;
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!worldRank) fprintf(stderr, "MPI_THREAD_FUNNELED is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t nx = 128, ny = 0, nz = 0, iterations = 10;
    bool validate = false, printResults = false, badArgs = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else {
            size_t* value = nullptr;
            if (!strcmp(argv[i], "-x")) value = &nx;
            else if (!strcmp(argv[i], "-y")) value = &ny;
            else if (!strcmp(argv[i], "-z")) value = &nz;
            else if (!strcmp(argv[i], "-i")) value = &iterations;
            if (!value || i + 1 == argc || !parseNumber(argv[++i], *value)) {
                badArgs = true;
                break;
            }
        }
    }
    if (help || badArgs) {
        if (!worldRank) printUsage(argv[0]);
        MPI_Finalize();
        return badArgs ? 1 : 0;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (!nx || !ny || !nz || nx > limit / ny || nx * ny > limit / nz) {
        if (!worldRank) fprintf(stderr, "Invalid or overflowing grid dimensions.\n");
        MPI_Finalize();
        return 1;
    }
    if (!worldRank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %zu\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    // Avoid empty slabs when there are more ranks than Z planes. Idle ranks
    // still participate in the world completion/status collective.
    const int activeSize = int(std::min(nz, size_t(worldSize)));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    int status = 0;
    if (comm != MPI_COMM_NULL) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm nodeComm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int nodeRank = 0, deviceCount = 0;
        MPI_Comm_rank(nodeComm, &nodeRank);
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (!deviceCount) {
            fprintf(stderr, "Rank %d: a CUDA device is required.\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(nodeRank % deviceCount));
        MPI_Comm_free(&nodeComm);
        const size_t base = nz / activeSize, extra = nz % activeSize;
        const size_t owned = base + (size_t(rank) < extra);
        const size_t startZ = size_t(rank) * base + std::min(size_t(rank), extra);
        const size_t plane = nx * ny;
        if (owned > limit / plane || limit / plane - owned < 2) {
            fprintf(stderr, "Rank %d: local grid size overflow.\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        const size_t localSize = (owned + 2) * plane;
        const size_t bytes = localSize * sizeof(Real);
        Real* host = nullptr;
        CUDA_CHECK(cudaMallocHost(&host, bytes));
        // Global indexing is essential: initialization must be independent
        // of the decomposition, including the initially known halo planes.
        #pragma omp parallel for schedule(static)
        for (size_t z = 0; z < owned + 2; ++z) {
            const bool validPlane = !(z == 0 && startZ == 0)
                                    && startZ + z <= nz;
            const size_t globalZ = validPlane ? startZ + z - 1 : 0;
            for (size_t i = 0; i < plane; ++i)
                host[z * plane + i] = validPlane ? Real((globalZ * plane + i) % 19) : 0.0;
        }
        Real *input = nullptr, *output = nullptr;
        CUDA_CHECK(cudaMalloc(&input, bytes));
        CUDA_CHECK(cudaMalloc(&output, bytes));
        CUDA_CHECK(cudaMemcpy(input, host, bytes, cudaMemcpyHostToDevice));
        // Fixed physical boundaries are initialized in both ping-pong grids;
        // subsequent kernels only write interior cells.
        CUDA_CHECK(cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice));
        cudaStream_t computeStream, exchangeStream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&exchangeStream, cudaStreamNonBlocking));
        const int lower = rank ? rank - 1 : MPI_PROC_NULL;
        const int upper = rank + 1 < activeSize ? rank + 1 : MPI_PROC_NULL;
        bool deviceMPI = false;
        #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        deviceMPI = MPIX_Query_cuda_support() != 0;
        #endif
        // Every participant must use the same exchange mode.
        int supported = deviceMPI, allSupported = 0;
        MPI_Allreduce(&supported, &allSupported, 1, MPI_INT, MPI_MIN, comm);
        deviceMPI = allSupported != 0;
        Real* staging = nullptr;
        if (!deviceMPI && activeSize > 1)
            CUDA_CHECK(cudaMallocHost(&staging, 4 * plane * sizeof(Real)));
        const size_t first = startZ == 0 ? 2 : 1;
        const size_t last = startZ + owned == nz ? owned - 1 : owned;
        std::vector<MPI_Request> requests;
        requests.reserve(4 * (plane / INT_MAX + 1));
        if (!rank) printf("Running stencil computation...\n");
        MPI_Barrier(comm);
        const double begin = MPI_Wtime();
        for (size_t iter = 0; iter < iterations; ++iter) {
            // Iteration zero's halos are initialized analytically.
            if (iter == 0 || activeSize == 1 || nx < 3 || ny < 3 || nz < 3) {
                launchStencil(input, output, nx, ny, first, last, computeStream);
                if (activeSize > 1) CUDA_CHECK(cudaStreamSynchronize(computeStream));
            } else {
                requests.clear();
                Real* recvLower = deviceMPI ? input : staging;
                Real* recvUpper = deviceMPI ? input + (owned + 1) * plane : staging + plane;
                Real* sendLower = deviceMPI ? input + plane : staging + 2 * plane;
                Real* sendUpper = deviceMPI ? input + owned * plane : staging + 3 * plane;
                postTransfer(recvLower, plane, lower, 1, true, comm, requests);
                postTransfer(recvUpper, plane, upper, 0, true, comm, requests);
                if (!deviceMPI) {
                    if (lower != MPI_PROC_NULL)
                        CUDA_CHECK(cudaMemcpyAsync(sendLower, input + plane, plane * sizeof(Real),
                                                   cudaMemcpyDeviceToHost, exchangeStream));
                    if (upper != MPI_PROC_NULL)
                        CUDA_CHECK(cudaMemcpyAsync(sendUpper, input + owned * plane, plane * sizeof(Real),
                                                   cudaMemcpyDeviceToHost, exchangeStream));
                }
                // These planes do not depend on an incoming halo. The CPU
                // drives MPI progress while the GPU computes this bulk.
                const size_t bulkFirst = std::max(first, size_t(2));
                const size_t bulkLast = std::min(last, owned - 1);
                launchStencil(input, output, nx, ny, bulkFirst, bulkLast, computeStream);
                if (!deviceMPI) CUDA_CHECK(cudaStreamSynchronize(exchangeStream));
                postTransfer(sendLower, plane, lower, 0, false, comm, requests);
                postTransfer(sendUpper, plane, upper, 1, false, comm, requests);
                MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
                if (!deviceMPI) {
                    if (lower != MPI_PROC_NULL)
                        CUDA_CHECK(cudaMemcpyAsync(input, recvLower, plane * sizeof(Real),
                                                   cudaMemcpyHostToDevice, exchangeStream));
                    if (upper != MPI_PROC_NULL)
                        CUDA_CHECK(cudaMemcpyAsync(input + (owned + 1) * plane, recvUpper, plane * sizeof(Real),
                                                   cudaMemcpyHostToDevice, exchangeStream));
                }
                if (first == 1 && last >= 1)
                    launchStencil(input, output, nx, ny, 1, 1, exchangeStream);
                if (last == owned && owned != 1)
                    launchStencil(input, output, nx, ny, owned, owned, exchangeStream);
                CUDA_CHECK(cudaStreamSynchronize(exchangeStream));
                CUDA_CHECK(cudaStreamSynchronize(computeStream));
            }
            std::swap(input, output);
        }
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        const double localSeconds = MPI_Wtime() - begin;
        double seconds = 0.0;
        MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!rank) {
            printf("Computation time: %lld ms\n", (long long)(seconds * 1000));
            const double updates = double(nx > 2 ? nx - 2 : 0)
                                   * double(ny > 2 ? ny - 2 : 0)
                                   * double(nz > 2 ? nz - 2 : 0) * double(iterations);
            printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0);
        }
        if (validate || printResults)
            CUDA_CHECK(cudaMemcpy(host + plane, input + plane, owned * plane * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
        if (printResults) {
            // Only explicit result output requires assembling the global grid.
            std::vector<Real> finalGrid;
            if (!rank) {
                finalGrid.resize(nx * ny * nz);
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < owned * plane; ++i) finalGrid[i] = host[plane + i];
                for (int peer = 1; peer < activeSize; ++peer) {
                    size_t peerOwned = base + (size_t(peer) < extra);
                    size_t offset = (size_t(peer) * base + std::min(size_t(peer), extra)) * plane;
                    size_t remaining = peerOwned * plane;
                    while (remaining) {
                        int chunk = int(std::min(remaining, size_t(INT_MAX)));
                        MPI_Recv(finalGrid.data() + offset, chunk, MPI_DOUBLE, peer, 2, comm, MPI_STATUS_IGNORE);
                        remaining -= chunk;
                        offset += chunk;
                    }
                }
                print_results(finalGrid, "Grid");
            } else {
                size_t offset = plane, remaining = owned * plane;
                while (remaining) {
                    int chunk = int(std::min(remaining, size_t(INT_MAX)));
                    MPI_Send(host + offset, chunk, MPI_DOUBLE, 0, 2, comm);
                    remaining -= chunk;
                    offset += chunk;
                }
            }
        }
        if (validate) {
            Real minValue = std::numeric_limits<Real>::infinity();
            Real maxValue = -std::numeric_limits<Real>::infinity();
            int invalid = 0;
            #pragma omp parallel for reduction(min:minValue) reduction(max:maxValue) reduction(|:invalid) schedule(static)
            for (size_t i = 0; i < owned * plane; ++i) {
                Real value = host[plane + i];
                invalid |= !std::isfinite(value);
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
            }
            Real globalMin = 0, globalMax = 0;
            int globalInvalid = 0;
            MPI_Allreduce(&minValue, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
            MPI_Allreduce(&maxValue, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_BOR, comm);
            status = globalInvalid || globalMin < -1e6 || globalMax > 1e6;
            if (!rank) {
                printf("Validating result...\n");
                printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalInvalid) printf("Validation failed: found NaN or Inf value\n");
                else if (status) printf("Validation failed: values out of expected range\n");
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
        CUDA_CHECK(cudaStreamDestroy(computeStream));
        CUDA_CHECK(cudaStreamDestroy(exchangeStream));
        CUDA_CHECK(cudaFree(input));
        CUDA_CHECK(cudaFree(output));
        if (staging) CUDA_CHECK(cudaFreeHost(staging));
        CUDA_CHECK(cudaFreeHost(host));
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
