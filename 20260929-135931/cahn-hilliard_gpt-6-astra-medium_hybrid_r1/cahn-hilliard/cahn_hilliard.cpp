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

static int worldRank;

static void fail(const char* message) {
    fprintf(stderr, "Rank %d: %s\n", worldRank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void cudaCheck(cudaError_t error, const char* expression) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", worldRank, expression, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}
#define CUDA(call) cudaCheck((call), #call)

static void mpiCheck(int error) {
    if (error != MPI_SUCCESS) fail("MPI operation failed");
}
#define MPI_CALL(call) mpiCheck(call)

// Each slab has one ghost plane on either side. Physical faces are clamped
// directly in the stencil, so they never read uninitialized ghost cells.
struct Grid {
    size_t nx, ny, nz, offset, globalNz, plane;
};

__device__ __forceinline__ double laplacian(const double* __restrict__ a,
                                          size_t i, size_t x, size_t y,
                                          size_t z, const Grid g) {
    const double center = 2.0 * a[i];
    const double xx = (a[i + (x + 1 < g.nx ? 1 : 0)] +
                       a[i - (x > 0 ? 1 : 0)] - center);
    const double yy = (a[i + (y + 1 < g.ny ? g.nx : 0)] +
                       a[i - (y > 0 ? g.nx : 0)] - center);
    const double zz = (a[i + (g.offset + z < g.globalNz ? g.plane : 0)] +
                       a[i - (g.offset + z > 1 ? g.plane : 0)] - center);
    // dx = dy = dz = 1 in the original benchmark.
    return xx + yy + zz;
}

// Contiguous threads access contiguous X-major cells. A single kernel handles
// either all/interior planes or just the two communication boundary planes.
template<bool Chemical, bool Boundary>
__global__ void stencil(const double* __restrict__ input,
                        const double* __restrict__ cold,
                        double* __restrict__ output, Grid g,
                        size_t first, size_t planes) {
    const size_t count = planes * g.plane;
    for (size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         t < count; t += size_t(blockDim.x) * gridDim.x) {
        const size_t p = t / g.plane;
        const size_t xy = t - p * g.plane;
        const size_t y = xy / g.nx;
        const size_t x = xy - y * g.nx;
        const size_t z = Boundary ? (p == 0 ? 1 : g.nz) : first + p;
        const size_t i = z * g.plane + xy;
        const double lap = laplacian(input, i, x, y, z, g);
        if (Chemical) {
            const double cv = input[i];
            constexpr double eAA = -(2.0 / 9.0);
            constexpr double eBB = -(2.0 / 9.0);
            constexpr double eAB = 2.0 / 9.0;
            output[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
                      + 3.0 * cv + cv * cv * cv - 0.5 * lap;
        } else {
            output[i] = cold[i] + 0.01 * lap;
        }
    }
}

template<bool Chemical, bool Boundary = false>
static void launch(const double* input, const double* cold, double* output,
                   Grid g, size_t first, size_t planes, cudaStream_t stream) {
    if (!planes) return;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>(
        (planes * g.plane + 255) / 256, 65535));
    stencil<Chemical, Boundary><<<blocks, 256, 0, stream>>>(input, cold, output, g, first, planes);
    CUDA(cudaGetLastError());
}

// Use direct device buffers when MPI advertises CUDA support. Otherwise,
// pinned staging supports ordinary MPI without transferring the local volume.
struct Halo {
    MPI_Comm comm;
    int lower, upper;
    Grid g;
    bool direct = false;
    double* send = nullptr;
    double* recv = nullptr;
    cudaStream_t transfer;
    cudaEvent_t ready, arrived;

    Halo(MPI_Comm c, int rank, int ranks, Grid grid) : comm(c),
        lower(rank ? rank - 1 : MPI_PROC_NULL),
        upper(rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL), g(grid) {
        #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        direct = MPIX_Query_cuda_support() != 0;
        #endif
        if (!direct && ranks > 1) {
            CUDA(cudaMallocHost(&send, 2 * g.plane * sizeof(double)));
            CUDA(cudaMallocHost(&recv, 2 * g.plane * sizeof(double)));
        }
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&arrived, cudaEventDisableTiming));
    }
    ~Halo() {
        CUDA(cudaEventDestroy(ready));
        CUDA(cudaEventDestroy(arrived));
        CUDA(cudaStreamDestroy(transfer));
        if (send) CUDA(cudaFreeHost(send));
        if (recv) CUDA(cudaFreeHost(recv));
    }

    template<bool Chemical>
    void apply(double* input, const double* cold, double* output, cudaStream_t compute) {
        // The previous phase may still be copying from the receive staging
        // buffer. Complete that DMA before MPI is allowed to reuse it.
        if (!direct) CUDA(cudaStreamSynchronize(transfer));
        CUDA(cudaEventRecord(ready, compute));
        // MPI has no stream argument: device sends must be ready, and previous
        // kernels must have finished reading the ghost planes before receives.
        if (direct) CUDA(cudaEventSynchronize(ready));
        else CUDA(cudaStreamWaitEvent(transfer, ready, 0));
        double* receiveLower = direct ? input : recv;
        double* receiveUpper = direct ? input + (g.nz + 1) * g.plane : recv + g.plane;
        double* sendLower = direct ? input + g.plane : send;
        double* sendUpper = direct ? input + g.nz * g.plane : send + g.plane;
        MPI_Request requests[4];
        int n = 0;
        const int count = static_cast<int>(g.plane);
        const size_t bytes = g.plane * sizeof(double);
        if (lower != MPI_PROC_NULL)
            MPI_CALL(MPI_Irecv(receiveLower, count, MPI_DOUBLE, lower, 1, comm, &requests[n++]));
        if (upper != MPI_PROC_NULL)
            MPI_CALL(MPI_Irecv(receiveUpper, count, MPI_DOUBLE, upper, 0, comm, &requests[n++]));
        if (!direct && lower != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(send, input + g.plane, bytes, cudaMemcpyDeviceToHost, transfer));
        if (!direct && upper != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(send + g.plane, input + g.nz * g.plane, bytes,
                                 cudaMemcpyDeviceToHost, transfer));
        // These planes do not depend on received halos. MPI progresses on the
        // main thread while CUDA executes the interior on a separate stream.
        launch<Chemical>(input, cold, output, g, 2, g.nz > 2 ? g.nz - 2 : 0, compute);
        if (!direct) CUDA(cudaStreamSynchronize(transfer));
        if (lower != MPI_PROC_NULL)
            MPI_CALL(MPI_Isend(sendLower, count, MPI_DOUBLE, lower, 0, comm, &requests[n++]));
        if (upper != MPI_PROC_NULL)
            MPI_CALL(MPI_Isend(sendUpper, count, MPI_DOUBLE, upper, 1, comm, &requests[n++]));
        MPI_CALL(MPI_Waitall(n, requests, MPI_STATUSES_IGNORE));
        if (!direct && lower != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(input, recv, bytes, cudaMemcpyHostToDevice, transfer));
        if (!direct && upper != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(input + (g.nz + 1) * g.plane, recv + g.plane,
                                 bytes, cudaMemcpyHostToDevice, transfer));
        if (!direct) {
            CUDA(cudaEventRecord(arrived, transfer));
            CUDA(cudaStreamWaitEvent(compute, arrived, 0));
        }
        launch<Chemical, true>(input, cold, output, g, 1, std::min<size_t>(g.nz, 2), compute);
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

static size_t number(const char* text) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (*text == '-' || end == text || *end || errno || value > std::numeric_limits<size_t>::max())
        fail("Invalid nonnegative integer argument");
    return static_cast<size_t>(value);
}

static size_t slabSize(size_t nz, int ranks, int rank) {
    return nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
}
static size_t slabOffset(size_t nz, int ranks, int rank) {
    return (nz / ranks) * rank + std::min<size_t>(rank, nz % ranks);
}

static int run(int argc, char** argv) {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = number(argv[++i]);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = number(argv[++i]);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = number(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            const size_t value = number(argv[++i]);
            if (value > INT_MAX) fail("Iteration count exceeds INT_MAX");
            iterations = static_cast<int>(value);
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
    const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(double);
    if (!nx || !ny || !nz || nx > maxElements / ny || nx * ny > maxElements / nz)
        fail("Invalid grid dimensions or grid size overflow");
    const size_t plane = nx * ny, volume = plane * nz;
    if (plane > INT_MAX) fail("A halo plane exceeds the MPI count limit");

    int worldSize;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    const int ranks = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    MPI_CALL(MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm));
    int status = 0;
    if (worldRank < ranks) {
        int rank;
        MPI_CALL(MPI_Comm_rank(comm, &rank));
        MPI_Comm node;
        MPI_CALL(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
        int localRank, devices;
        MPI_CALL(MPI_Comm_rank(node, &localRank));
        CUDA(cudaGetDeviceCount(&devices));
        if (!devices) fail("CUDA GPU required");
        // Also supports launchers which expose only a single GPU per rank.
        CUDA(cudaSetDevice(localRank % devices));
        MPI_CALL(MPI_Comm_free(&node));
        Grid g{nx, ny, slabSize(nz, ranks, rank), slabOffset(nz, ranks, rank), nz, plane};
        if (g.nz > maxElements / plane - 2) fail("Local allocation size overflow");
        const size_t localCount = g.nz * plane;
        const size_t bytes = (g.nz + 2) * plane * sizeof(double);
        double *cold, *cnew, *mu;
        CUDA(cudaMalloc(&cold, bytes));
        CUDA(cudaMalloc(&cnew, bytes));
        CUDA(cudaMalloc(&mu, bytes));
        cudaStream_t compute;
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        std::vector<double> host(localCount);
        if (!rank) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\nTime steps: %d\n", nx, ny, nz, iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing concentration field...\n");
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < localCount; ++i) {
            const size_t linear = g.offset * plane + i;
            const double pseudo = (((linear + 1) * size_t(1299709)) % volume) / static_cast<double>(volume);
            host[i] = -1.0 + 2.0 * pseudo;
        }
        CUDA(cudaMemcpyAsync(cold + plane, host.data(), localCount * sizeof(double),
                             cudaMemcpyHostToDevice, compute));
        CUDA(cudaStreamSynchronize(compute));
        {
            Halo halo(comm, rank, ranks, g);
            // A two-step graph returns to the original buffer addresses and
            // eliminates per-kernel launch overhead when no network is needed.
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            if (ranks == 1 && iterations >= 2) {
                CUDA(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
                launch<true>(cold, cold, mu, g, 1, g.nz, compute);
                launch<false>(mu, cold, cnew, g, 1, g.nz, compute);
                launch<true>(cnew, cnew, mu, g, 1, g.nz, compute);
                launch<false>(mu, cnew, cold, g, 1, g.nz, compute);
                CUDA(cudaStreamEndCapture(compute, &graph));
                CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            }
            if (!rank) printf("Running Cahn-Hilliard simulation...\n");
            MPI_CALL(MPI_Barrier(comm));
            const double start = MPI_Wtime();
            int t = 0;
            if (executable) {
                for (; t + 1 < iterations; t += 2) CUDA(cudaGraphLaunch(executable, compute));
            }
            for (; t < iterations; ++t) {
                if (ranks == 1) {
                    launch<true>(cold, cold, mu, g, 1, g.nz, compute);
                    launch<false>(mu, cold, cnew, g, 1, g.nz, compute);
                } else {
                    halo.apply<true>(cold, cold, mu, compute);
                    halo.apply<false>(mu, cold, cnew, compute);
                }
                std::swap(cold, cnew);
            }
            CUDA(cudaStreamSynchronize(compute));
            const double elapsed = MPI_Wtime() - start;
            double seconds;
            MPI_CALL(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
            if (!rank) {
                printf("Computation time: %.3f ms\n", seconds * 1000.0);
                printf("Performance: %.3f MCellUpdates/s\n", double(volume) * iterations / seconds / 1e6);
            }
            if (executable) CUDA(cudaGraphExecDestroy(executable));
            if (graph) CUDA(cudaGraphDestroy(graph));
        }
        if (printResults || validate) {
            CUDA(cudaMemcpyAsync(host.data(), cold + plane, localCount * sizeof(double),
                                 cudaMemcpyDeviceToHost, compute));
            CUDA(cudaStreamSynchronize(compute));
        }
        if (printResults) {
            // Gather only on request. Chunking avoids MPI's signed-int count
            // and displacement limits while retaining exact global order.
            std::vector<double> result;
            if (!rank) result.resize(volume);
            for (int source = 0; source < ranks; ++source) {
                const size_t count = slabSize(nz, ranks, source) * plane;
                const size_t offset = slabOffset(nz, ranks, source) * plane;
                for (size_t i = 0; i < count;) {
                    const int chunk = static_cast<int>(std::min<size_t>(count - i, INT_MAX));
                    if (!source && !rank) std::copy_n(host.data() + i, chunk, result.data() + offset + i);
                    else if (!rank) MPI_CALL(MPI_Recv(result.data() + offset + i, chunk, MPI_DOUBLE,
                                                     source, 2, comm, MPI_STATUS_IGNORE));
                    else if (rank == source) MPI_CALL(MPI_Send(host.data() + i, chunk, MPI_DOUBLE, 0, 2, comm));
                    i += chunk;
                }
            }
            if (!rank) print_results(result, "Concentration");
        }
        if (validate) {
            double minimum = std::numeric_limits<double>::infinity();
            double maximum = -minimum;
            int invalid = 0;
            #pragma omp parallel for schedule(static) reduction(min:minimum) reduction(max:maximum) reduction(|:invalid)
            for (size_t i = 0; i < localCount; ++i) {
                invalid |= !std::isfinite(host[i]);
                minimum = std::min(minimum, host[i]);
                maximum = std::max(maximum, host[i]);
            }
            MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, comm));
            MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_DOUBLE, MPI_MIN, comm));
            MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_DOUBLE, MPI_MAX, comm));
            status = invalid || maximum > 10.0 || minimum < -10.0;
            if (!rank) {
                printf("Validating result...\n");
                if (invalid) printf("Validation failed: found NaN or Inf value\n");
                else {
                    printf("Concentration range: [%.6f, %.6f]\n", minimum, maximum);
                    if (status) printf("Validation failed: values out of expected range\n");
                }
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
        CUDA(cudaFree(cold));
        CUDA(cudaFree(cnew));
        CUDA(cudaFree(mu));
        CUDA(cudaStreamDestroy(compute));
        MPI_CALL(MPI_Comm_free(&comm));
    }
    // Extra ranks (when nz < worldSize) do not allocate empty GPU slabs.
    MPI_CALL(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return status;
}

int main(int argc, char** argv) {
    int provided;
    MPI_CALL(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED support required");
    int status;
    try { status = run(argc, argv); }
    catch (const std::exception& e) { fail(e.what()); }
    MPI_CALL(MPI_Finalize());
    return status;
}
