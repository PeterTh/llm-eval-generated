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
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

static void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "MPI: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define MPI(call) mpiCheck(call)

struct Grid {
    size_t nx, ny, nz, firstZ, localZ, plane;
};

// The physical boundary is clamped; internal slab edges use MPI ghost planes.
__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                            size_t i, size_t x, size_t y,
                                            size_t z, Grid g) {
    const size_t xp = x + 1 < g.nx ? i + 1 : i;
    const size_t xn = x > 0 ? i - 1 : i;
    const size_t yp = y + 1 < g.ny ? i + g.nx : i;
    const size_t yn = y > 0 ? i - g.nx : i;
    const size_t globalZ = g.firstZ + z - 1;
    const size_t zp = globalZ + 1 < g.nz ? i + g.plane : i;
    const size_t zn = globalZ > 0 ? i - g.plane : i;
    const double center = field[i];
    const double cxx = field[xp] + field[xn] - 2.0 * center;
    const double cyy = field[yp] + field[yn] - 2.0 * center;
    const double czz = field[zp] + field[zn] - 2.0 * center;
    return cxx + cyy + czz;
}

// Contiguous X lanes coalesce all seven loads. A 3-D launch avoids costly
// per-cell integer divisions; grid strides handle CUDA Y/Z grid limits.
template<bool Chemical>
__global__ void stencil(const double* __restrict__ input,
                        const double* __restrict__ cold,
                        double* __restrict__ output, Grid g,
                        size_t first, size_t planes, bool edges) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= g.nx) return;
    for (size_t p = blockIdx.z; p < planes; p += gridDim.z) {
        const size_t z = edges ? (p == 0 ? 1 : g.localZ) : first + p;
        for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
             y < g.ny; y += size_t(blockDim.y) * gridDim.y) {
            const size_t i = z * g.plane + y * g.nx + x;
            const double lap = laplacian(input, i, x, y, z, g);
            if constexpr (Chemical) {
                constexpr double e_AA = -(2.0 / 9.0);
                constexpr double e_BB = -(2.0 / 9.0);
                constexpr double e_AB = 2.0 / 9.0;
                const double cv = input[i];
                output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB
                                  - 2.0 * cv * e_AB)
                          + 3.0 * cv + cv * cv * cv - 0.5 * lap;
            } else {
                output[i] = cold[i] + 0.01 * 1.0 * lap;
            }
        }
    }
}

template<bool Chemical>
static void launch(const double* input, const double* cold, double* output,
                   Grid g, size_t first, size_t planes, bool edges,
                   cudaStream_t stream) {
    if (planes == 0) return;
    unsigned bx = 1, by = 1;
    while (bx < 32 && bx < g.nx) bx *= 2;
    while (by < 8 && by < g.ny) by *= 2;
    const dim3 block(bx, by);
    const dim3 grid(static_cast<unsigned>((g.nx + bx - 1) / bx),
                    static_cast<unsigned>(std::min<size_t>((g.ny + by - 1) / by, 65535)),
                    static_cast<unsigned>(std::min<size_t>(planes, 65535)));
    stencil<Chemical><<<grid, block, 0, stream>>>(input, cold, output, g, first, planes, edges);
    CUDA(cudaGetLastError());
}

// Host staging works with ordinary MPI: CUDA-aware MPI is not required.
// Two streams and completion events overlap interior kernels with transfers
// and MPI progress, while protecting both producers and ping-pong buffers.
struct Halo {
    cudaStream_t interior{}, boundary{};
    cudaEvent_t interiorDone{}, boundaryDone{}, haloCopied{};
    double* send = nullptr;
    double* recv = nullptr;
    int below, above;
    MPI_Comm comm;
    Grid g;

    Halo(Grid grid, int rank, int ranks, MPI_Comm communicator)
        : below(rank > 0 ? rank - 1 : MPI_PROC_NULL),
          above(rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL),
          comm(communicator), g(grid) {
        CUDA(cudaStreamCreateWithFlags(&interior, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&boundary, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&interiorDone, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&boundaryDone, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&haloCopied, cudaEventDisableTiming));
        CUDA(cudaMallocHost(&send, 2 * g.plane * sizeof(double)));
        CUDA(cudaMallocHost(&recv, 2 * g.plane * sizeof(double)));
        CUDA(cudaEventRecord(interiorDone, interior));
        CUDA(cudaEventRecord(boundaryDone, boundary));
        CUDA(cudaEventRecord(haloCopied, boundary));
    }

    template<bool Chemical>
    void phase(double* input, const double* cold, double* output) {
        // MPI must not overwrite pinned receive buffers until the previous
        // asynchronous upload has consumed them. The boundary kernel can
        // continue running while the next receives are posted.
        CUDA(cudaEventSynchronize(haloCopied));
        MPI_Request requests[4];
        int n = 0;
        const int count = static_cast<int>(g.plane);
        if (below != MPI_PROC_NULL)
            MPI(MPI_Irecv(recv, count, MPI_DOUBLE, below, 1, comm, &requests[n++]));
        if (above != MPI_PROC_NULL)
            MPI(MPI_Irecv(recv + g.plane, count, MPI_DOUBLE, above, 0, comm, &requests[n++]));

        // Each stream waits on the other stream's previous phase only.
        CUDA(cudaStreamWaitEvent(interior, boundaryDone, 0));
        CUDA(cudaStreamWaitEvent(boundary, interiorDone, 0));
        if (below != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(send, input + g.plane, g.plane * sizeof(double),
                                 cudaMemcpyDeviceToHost, boundary));
        if (above != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(send + g.plane, input + g.localZ * g.plane,
                                 g.plane * sizeof(double), cudaMemcpyDeviceToHost, boundary));
        launch<Chemical>(input, cold, output, g, 2,
                         g.localZ > 2 ? g.localZ - 2 : 0, false, interior);
        CUDA(cudaEventRecord(interiorDone, interior));
        CUDA(cudaStreamSynchronize(boundary));
        if (below != MPI_PROC_NULL)
            MPI(MPI_Isend(send, count, MPI_DOUBLE, below, 0, comm, &requests[n++]));
        if (above != MPI_PROC_NULL)
            MPI(MPI_Isend(send + g.plane, count, MPI_DOUBLE, above, 1, comm, &requests[n++]));
        MPI(MPI_Waitall(n, requests, MPI_STATUSES_IGNORE));
        if (below != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(input, recv, g.plane * sizeof(double),
                                 cudaMemcpyHostToDevice, boundary));
        if (above != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(input + (g.localZ + 1) * g.plane, recv + g.plane,
                                 g.plane * sizeof(double), cudaMemcpyHostToDevice, boundary));
        CUDA(cudaEventRecord(haloCopied, boundary));
        launch<Chemical>(input, cold, output, g, 1,
                         g.localZ == 1 ? 1 : 2, true, boundary);
        CUDA(cudaEventRecord(boundaryDone, boundary));
    }

    ~Halo() {
        CUDA(cudaStreamSynchronize(interior));
        CUDA(cudaStreamSynchronize(boundary));
        CUDA(cudaFreeHost(send));
        CUDA(cudaFreeHost(recv));
        CUDA(cudaEventDestroy(interiorDone));
        CUDA(cudaEventDestroy(boundaryDone));
        CUDA(cudaEventDestroy(haloCopied));
        CUDA(cudaStreamDestroy(interior));
        CUDA(cudaStreamDestroy(boundary));
    }
};

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static size_t parseNumber(const char* text) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (text[0] == '-' || end == text || *end || errno == ERANGE ||
        value > std::numeric_limits<size_t>::max())
        throw std::runtime_error("Invalid numeric argument");
    return static_cast<size_t>(value);
}

static int run(int argc, char** argv, int worldRank, int worldSize) {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h")) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (i + 1 < argc && (!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
                                  !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i"))) {
            const char option = argv[i][1];
            const size_t value = parseNumber(argv[++i]);
            if (option == 'x') nx = value;
            if (option == 'y') ny = value;
            if (option == 'z') nz = value;
            if (option == 'i') {
                if (value > INT_MAX) throw std::runtime_error("Too many time steps");
                iterations = static_cast<int>(value);
            }
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx == 0 || ny == 0 || nz == 0 || nx > limit / ny || nx * ny > limit / nz)
        throw std::runtime_error("Invalid or overflowing grid size");
    if (nx * ny > INT_MAX) throw std::runtime_error("MPI halo plane exceeds INT_MAX elements");
    const size_t volume = nx * ny * nz;
    // Empty ranks do not allocate a GPU or enter stencil collectives.
    const int ranks = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    MPI(MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm));
    if (worldRank >= ranks) return 0;
    const int rank = worldRank;
    Grid g{nx, ny, nz, 0, 0, nx * ny};
    g.localZ = nz / ranks + (size_t(rank) < nz % ranks ? 1 : 0);
    g.firstZ = size_t(rank) * (nz / ranks) + std::min<size_t>(rank, nz % ranks);
    if (limit / g.plane < 2 || g.localZ > limit / g.plane - 2)
        throw std::runtime_error("Local grid including halos overflows");

    MPI_Comm node;
    MPI(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank = 0, devices = 0;
    MPI(MPI_Comm_rank(node, &localRank));
    CUDA(cudaGetDeviceCount(&devices));
    if (devices == 0) throw std::runtime_error("A CUDA GPU is required on every active rank");
    // Also supports schedulers that expose only the assigned GPU to each rank.
    CUDA(cudaSetDevice(localRank % devices));
    MPI(MPI_Comm_free(&node));

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    const size_t localCount = g.localZ * g.plane;
    double* host = nullptr;
    CUDA(cudaMallocHost(&host, localCount * sizeof(double)));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localCount; ++i) {
        const size_t linear_id = g.firstZ * g.plane + i;
        const double pseudo = (((linear_id + 1) * size_t(1299709)) % volume) /
                              static_cast<double>(volume);
        host[i] = -1.0 + 2.0 * pseudo;
    }
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    const size_t bytes = (g.localZ + 2) * g.plane * sizeof(double);
    CUDA(cudaMalloc(&cold, bytes));
    CUDA(cudaMalloc(&cnew, bytes));
    CUDA(cudaMalloc(&mu, bytes));
    CUDA(cudaMemcpy(cold + g.plane, host, localCount * sizeof(double), cudaMemcpyHostToDevice));

    double elapsed = 0.0;
    {
        Halo halo(g, rank, ranks, comm);
        cudaGraphExec_t graphs[2]{};
        // For one slab, graphs remove repeated host launch/synchronization cost.
        // The two graphs represent the two ping-pong buffer orientations.
        if (ranks == 1 && iterations > 0) {
            for (int parity = 0; parity < 2; ++parity) {
                double* src = parity == 0 ? cold : cnew;
                double* dst = parity == 0 ? cnew : cold;
                cudaGraph_t graph;
                CUDA(cudaStreamBeginCapture(halo.interior, cudaStreamCaptureModeThreadLocal));
                launch<true>(src, src, mu, g, 1, g.localZ, false, halo.interior);
                launch<false>(mu, src, dst, g, 1, g.localZ, false, halo.interior);
                CUDA(cudaStreamEndCapture(halo.interior, &graph));
                CUDA(cudaGraphInstantiate(&graphs[parity], graph, nullptr, nullptr, 0));
                CUDA(cudaGraphUpload(graphs[parity], halo.interior));
                CUDA(cudaGraphDestroy(graph));
            }
        }
        CUDA(cudaDeviceSynchronize());
        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        MPI(MPI_Barrier(comm));
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            if (ranks == 1) CUDA(cudaGraphLaunch(graphs[t % 2], halo.interior));
            else {
                halo.phase<true>(cold, cold, mu);
                halo.phase<false>(mu, cold, cnew);
            }
            std::swap(cold, cnew);
        }
        CUDA(cudaDeviceSynchronize());
        const double localElapsed = MPI_Wtime() - start;
        MPI(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
        for (auto graph : graphs) if (graph) CUDA(cudaGraphExecDestroy(graph));
    }
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
        const double mcups = elapsed > 0 ? double(volume) * iterations / elapsed / 1e6 : 0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    if (validate || printResults)
        CUDA(cudaMemcpy(host, cold + g.plane, localCount * sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) {
        // Gather in original global order, retaining the exact Kahan sum/hash.
        // Chunked transfers avoid MPI's int-count limit on the full volume.
        std::vector<double> result;
        if (rank == 0) {
            result.resize(volume);
            std::copy(host, host + localCount, result.begin());
        }
        for (int source = 1; source < ranks; ++source) {
            const size_t planes = nz / ranks + (size_t(source) < nz % ranks ? 1 : 0);
            const size_t offset = (size_t(source) * (nz / ranks) + std::min<size_t>(source, nz % ranks)) * g.plane;
            const size_t count = planes * g.plane;
            for (size_t i = 0; i < count;) {
                const int chunk = static_cast<int>(std::min<size_t>(count - i, INT_MAX));
                if (rank == source) MPI(MPI_Send(host + i, chunk, MPI_DOUBLE, 0, 2, comm));
                if (rank == 0) MPI(MPI_Recv(result.data() + offset + i, chunk, MPI_DOUBLE, source, 2, comm, MPI_STATUS_IGNORE));
                i += chunk;
            }
        }
        if (rank == 0) print_results(result, "Concentration");
    }
    int status = 0;
    if (validate) {
        double localMin = std::numeric_limits<double>::infinity();
        double localMax = -std::numeric_limits<double>::infinity();
        int invalid = 0;
        #pragma omp parallel for schedule(static) reduction(min:localMin) reduction(max:localMax) reduction(|:invalid)
        for (size_t i = 0; i < localCount; ++i) {
            const double value = host[i];
            invalid |= !std::isfinite(value);
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        double minValue, maxValue;
        int anyInvalid;
        MPI(MPI_Allreduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_MAX, comm));
        MPI(MPI_Allreduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, comm));
        MPI(MPI_Allreduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, comm));
        status = anyInvalid || minValue < -10.0 || maxValue > 10.0;
        if (rank == 0) {
            printf("Validating result...\n");
            if (anyInvalid) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
                if (status) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
    }
    CUDA(cudaFree(cold));
    CUDA(cudaFree(cnew));
    CUDA(cudaFree(mu));
    CUDA(cudaFreeHost(host));
    MPI(MPI_Comm_free(&comm));
    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0, size = 1;
    MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
    MPI(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int status = 1;
    try {
        status = run(argc, argv, rank, size);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int globalStatus = 0;
    MPI(MPI_Allreduce(&status, &globalStatus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    MPI(MPI_Finalize());
    return globalStatus;
}
