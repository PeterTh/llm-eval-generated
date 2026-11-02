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

// Each rank owns complete Z planes, with one extra plane on either side.
// Both buffers are initialized with the fixed boundary values once.
__global__ void stencil(const Real* __restrict__ input, Real* __restrict__ output,
                        size_t nx, size_t ny, size_t nz, size_t globalStart,
                        size_t first, size_t count) {
    const size_t plane = nx * ny;
    for (size_t z = first + blockIdx.z; z < first + count; z += gridDim.z) {
        const size_t globalZ = globalStart + z - 1;
        if (globalZ == 0 || globalZ + 1 == nz) continue;
        for (size_t y = 1 + blockIdx.y * blockDim.y + threadIdx.y; y + 1 < ny;
             y += static_cast<size_t>(gridDim.y) * blockDim.y) {
            for (size_t x = 1 + blockIdx.x * blockDim.x + threadIdx.x; x + 1 < nx;
                 x += static_cast<size_t>(gridDim.x) * blockDim.x) {
                size_t i = z * plane + y * nx + x;
                output[i] = (input[i] + input[i - 1] + input[i + 1] +
                             input[i - nx] + input[i + nx] +
                             input[i - plane] + input[i + plane]) / 7.0;
            }
        }
    }
}

static void launchStencil(const Real* input, Real* output, size_t nx, size_t ny,
                          size_t nz, size_t globalStart, size_t first, size_t count,
                          cudaStream_t stream) {
    if (count == 0 || nx < 3 || ny < 3 || nz < 3) return;
    dim3 block(32, 8);
    dim3 grid(static_cast<unsigned>(std::min<size_t>((nx - 3) / 32 + 1, 65535)),
              static_cast<unsigned>(std::min<size_t>((ny - 3) / 8 + 1, 65535)),
              static_cast<unsigned>(std::min<size_t>(count, 65535)));
    stencil<<<grid, block, 0, stream>>>(input, output, nx, ny, nz, globalStart, first, count);
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

// Split large planes into MPI messages whose element counts fit in an int.
static void exchangePost(std::vector<MPI_Request>& requests, Real* buffer, size_t count,
                         int peer, int tag, bool receive, MPI_Comm comm) {
    if (peer == MPI_PROC_NULL) return;
    for (size_t offset = 0; offset < count;) {
        int n = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
        MPI_Request request;
        if (receive) MPI_CHECK(MPI_Irecv(buffer + offset, n, MPI_DOUBLE, peer, tag, comm, &request));
        else MPI_CHECK(MPI_Isend(buffer + offset, n, MPI_DOUBLE, peer, tag, comm, &request));
        requests.push_back(request);
        offset += n;
    }
}

static void transferLarge(Real* data, size_t count, int peer, bool receive, MPI_Comm comm) {
    for (size_t offset = 0; offset < count;) {
        int n = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
        if (receive) MPI_CHECK(MPI_Recv(data + offset, n, MPI_DOUBLE, peer, 10, comm, MPI_STATUS_IGNORE));
        else MPI_CHECK(MPI_Send(data + offset, n, MPI_DOUBLE, peer, 10, comm));
        offset += n;
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (provided < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") || !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const char* text = argv[++i];
            if (*text == '-') { parseStatus = 1; break; }
            unsigned long long value = strtoull(text, &end, 10);
            if (!*text || *end || value > std::numeric_limits<size_t>::max() ||
                (option == 'i' && value > INT_MAX)) { parseStatus = 1; break; }
            if (option == 'x') nx = static_cast<size_t>(value);
            else if (option == 'y') ny = static_cast<size_t>(value);
            else if (option == 'z') nz = static_cast<size_t>(value);
            else iterations = static_cast<int>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { parseStatus = 2; break; }
        else { if (worldRank == 0) printf("Unknown option: %s\n", argv[i]); parseStatus = 1; break; }
    }
    if (parseStatus) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (!nx || !ny || !nz || nx > maxElements / ny || nx * ny > maxElements / nz ||
        nz > std::numeric_limits<size_t>::max() - 2) {
        if (worldRank == 0) fprintf(stderr, "Invalid or excessively large grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny;
    int activeCount = static_cast<int>(std::min<size_t>(nz, worldSize));
    MPI_Comm active = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeCount ? 0 : MPI_UNDEFINED, worldRank, &active));
    int status = 0;
    if (active != MPI_COMM_NULL) {
        MPI_Comm nodeComm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm));
        int localRank = 0;
        MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
        MPI_CHECK(MPI_Comm_free(&nodeComm));
        int devices = 0;
        CUDA(cudaGetDeviceCount(&devices));
        if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", worldRank); MPI_Abort(MPI_COMM_WORLD, 1); }
        CUDA(cudaSetDevice(localRank % devices));

        const size_t base = nz / activeCount, remainder = nz % activeCount;
        const size_t owned = base + (static_cast<size_t>(worldRank) < remainder);
        const size_t start = static_cast<size_t>(worldRank) * base + std::min<size_t>(worldRank, remainder);
        if (plane > maxElements / (owned + 2)) {
            fprintf(stderr, "Rank %d: local grid is too large\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        const size_t localElements = plane * owned, bytes = plane * (owned + 2) * sizeof(Real);
        Real *a = nullptr, *b = nullptr;
        CUDA(cudaMalloc(&a, bytes));
        CUDA(cudaMalloc(&b, bytes));
        Real* host = nullptr;
        CUDA(cudaMallocHost(&host, localElements * sizeof(Real)));
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < localElements; ++i) host[i] = static_cast<Real>((start * plane + i) % 19);
        CUDA(cudaMemcpy(a + plane, host, localElements * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(b + plane, host, localElements * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA(cudaFreeHost(host));
        host = nullptr;
        cudaStream_t compute, communication;
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&communication, cudaStreamNonBlocking));
        const int lower = worldRank == 0 ? MPI_PROC_NULL : worldRank - 1;
        const int upper = worldRank + 1 == activeCount ? MPI_PROC_NULL : worldRank + 1;
        Real *sendLower = nullptr, *sendUpper = nullptr, *recvLower = nullptr, *recvUpper = nullptr;
        const bool exchange = activeCount > 1 && iterations > 0 && nx >= 3 && ny >= 3 && nz >= 3;
        if (exchange) {
            if (lower != MPI_PROC_NULL) {
                CUDA(cudaMallocHost(&sendLower, plane * sizeof(Real)));
                CUDA(cudaMallocHost(&recvLower, plane * sizeof(Real)));
            }
            if (upper != MPI_PROC_NULL) {
                CUDA(cudaMallocHost(&sendUpper, plane * sizeof(Real)));
                CUDA(cudaMallocHost(&recvUpper, plane * sizeof(Real)));
            }
        }
        if (worldRank == 0) {
            printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                   nx, ny, nz, iterations, validate ? "enabled" : "disabled");
            printf("Initializing grid...\nRunning stencil computation...\n");
        }
        MPI_CHECK(MPI_Barrier(active));
        double timeStart = MPI_Wtime();
        Real* input = a;
        Real* output = b;
        for (int iter = 0; iter < iterations; ++iter) {
            if (!exchange) {
                launchStencil(input, output, nx, ny, nz, start, 1, owned, compute);
            } else {
                std::vector<MPI_Request> requests;
                exchangePost(requests, recvLower, plane, lower, 1, true, active);
                exchangePost(requests, recvUpper, plane, upper, 0, true, active);
                if (lower != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(sendLower, input + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, communication));
                if (upper != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(sendUpper, input + owned * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, communication));
                if (owned > 2) launchStencil(input, output, nx, ny, nz, start, 2, owned - 2, compute);
                CUDA(cudaStreamSynchronize(communication));
                exchangePost(requests, sendLower, plane, lower, 0, false, active);
                exchangePost(requests, sendUpper, plane, upper, 1, false, active);
                MPI_CHECK(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
                if (lower != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(input, recvLower, plane * sizeof(Real), cudaMemcpyHostToDevice, communication));
                if (upper != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(input + (owned + 1) * plane, recvUpper, plane * sizeof(Real), cudaMemcpyHostToDevice, communication));
                launchStencil(input, output, nx, ny, nz, start, 1, 1, communication);
                if (owned > 1) launchStencil(input, output, nx, ny, nz, start, owned, 1, communication);
                CUDA(cudaStreamSynchronize(communication));
                CUDA(cudaStreamSynchronize(compute));
            }
            std::swap(input, output);
        }
        CUDA(cudaStreamSynchronize(compute));
        double elapsed = MPI_Wtime() - timeStart, maxElapsed = 0;
        MPI_CHECK(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, active));
        if (worldRank == 0) {
            long long milliseconds = static_cast<long long>(maxElapsed * 1000.0);
            double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) * (ny > 2 ? ny - 2 : 0) * (nz > 2 ? nz - 2 : 0) * iterations;
            printf("Computation time: %lld ms\nPerformance: %.3f MCellUpdates/s\n", milliseconds, updates / maxElapsed / 1e6);
        }
        if (printResults || validate) {
            std::vector<Real> local(localElements);
            CUDA(cudaMemcpy(local.data(), input + plane, localElements * sizeof(Real), cudaMemcpyDeviceToHost));
            if (printResults) {
                if (worldRank == 0) {
                    std::vector<Real> global(plane * nz);
                    std::copy(local.begin(), local.end(), global.begin());
                    for (int rank = 1; rank < activeCount; ++rank) {
                        size_t count = base + (static_cast<size_t>(rank) < remainder);
                        size_t offset = static_cast<size_t>(rank) * base + std::min<size_t>(rank, remainder);
                        transferLarge(global.data() + offset * plane, count * plane, rank, true, active);
                    }
                    print_results(global, "Grid");
                } else transferLarge(local.data(), localElements, 0, false, active);
            }
            if (validate) {
                double minValue = std::numeric_limits<double>::infinity();
                double maxValue = -std::numeric_limits<double>::infinity();
                int invalid = 0;
                #pragma omp parallel for reduction(min:minValue) reduction(max:maxValue) reduction(|:invalid) schedule(static)
                for (size_t i = 0; i < localElements; ++i) {
                    double value = local[i];
                    if (!std::isfinite(value)) invalid = 1;
                    else { minValue = std::min(minValue, value); maxValue = std::max(maxValue, value); }
                }
                double globalMin = 0, globalMax = 0;
                int globalInvalid = 0;
                MPI_CHECK(MPI_Reduce(&minValue, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, active));
                MPI_CHECK(MPI_Reduce(&maxValue, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, active));
                MPI_CHECK(MPI_Reduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_LOR, 0, active));
                if (worldRank == 0) {
                    printf("Validating result...\n");
                    if (globalInvalid) { printf("Validation failed: found NaN or Inf value\n"); status = 1; }
                    else {
                        printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
                        if (globalMax > 1e6 || globalMin < -1e6) { printf("Validation failed: values out of expected range\n"); status = 1; }
                    }
                    printf("Validation: %s\n", status ? "FAILED" : "PASSED");
                }
            }
        }
        if (sendLower) CUDA(cudaFreeHost(sendLower));
        if (recvLower) CUDA(cudaFreeHost(recvLower));
        if (sendUpper) CUDA(cudaFreeHost(sendUpper));
        if (recvUpper) CUDA(cudaFreeHost(recvUpper));
        CUDA(cudaStreamDestroy(compute));
        CUDA(cudaStreamDestroy(communication));
        CUDA(cudaFree(a));
        CUDA(cudaFree(b));
        MPI_CHECK(MPI_Comm_free(&active));
    } else {
        // split_type is collective over the world communicator, including idle ranks.
        MPI_Comm nodeComm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm));
        MPI_CHECK(MPI_Comm_free(&nodeComm));
    }
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return status;
}
