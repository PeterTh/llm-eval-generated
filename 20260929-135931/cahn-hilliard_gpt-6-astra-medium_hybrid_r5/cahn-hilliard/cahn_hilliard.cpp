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
#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

struct Domain {
    size_t nx, ny, planes, offset, plane;
    int lower, upper;
};

// One ghost plane on each side; physical boundaries clamp to the center.
// dx = dy = dz = 1, as in the original benchmark.
__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                           size_t i, size_t x, size_t y,
                                           size_t z, const Domain d) {
    const double center = 2.0 * field[i];
    const double xx = field[i + (x + 1 < d.nx ? 1 : 0)]
                    + field[i - (x ? 1 : 0)] - center;
    const double yy = field[i + (y + 1 < d.ny ? d.nx : 0)]
                    + field[i - (y ? d.nx : 0)] - center;
    const double zz = field[i + (z < d.planes || d.upper != MPI_PROC_NULL ? d.plane : 0)]
                    + field[i - (z > 1 || d.lower != MPI_PROC_NULL ? d.plane : 0)] - center;
    return xx + yy + zz;
}

// Contiguous X lanes coalesce all seven stencil loads. A grid-stride loop
// avoids launch-grid limits even for very large local domains.
template<bool chemical>
__global__ void stencil(const double* __restrict__ field,
                        const double* __restrict__ old,
                        double* __restrict__ result, Domain d,
                        size_t first, size_t count, bool edges) {
    const size_t n = count * d.plane;
    for (size_t q = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         q < n; q += size_t(blockDim.x) * gridDim.x) {
        const size_t xy = q % d.plane;
        const size_t z = edges ? (q / d.plane == 0 ? 1 : d.planes)
                               : first + q / d.plane;
        const size_t i = z * d.plane + xy;
        const size_t x = xy % d.nx, y = xy / d.nx;
        const double lap = laplacian(field, i, x, y, z, d);
        if constexpr (chemical) {
            constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
            const double cv = field[i];
            result[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                      + 3.0 * cv + cv * cv * cv - 0.5 * lap;
        } else {
            result[i] = old[i] + 0.01 * lap;
        }
    }
}

template<bool chemical>
static void launch(const double* field, const double* old, double* result,
                   Domain d, size_t first, size_t count, bool edges, cudaStream_t stream) {
    if (!count) return;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((count * d.plane + 255) / 256, 65535));
    stencil<chemical><<<blocks, 256, 0, stream>>>(field, old, result, d, first, count, edges);
    CUDA_CHECK(cudaGetLastError());
}

// Use device buffers with CUDA-aware Open MPI when supported. Otherwise,
// pinned staging works with any MPI. Only surface data is communicated.
struct HaloExchange {
    Domain d;
    MPI_Comm comm;
    cudaStream_t transfer;
    cudaEvent_t ready;
    double* host = nullptr;
    bool direct = false;
    bool neighbors;

    HaloExchange(Domain domain, MPI_Comm communicator) : d(domain), comm(communicator),
        neighbors(d.lower != MPI_PROC_NULL || d.upper != MPI_PROC_NULL) {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        direct = MPIX_Query_cuda_support() != 0;
#endif
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        if (neighbors && !direct)
            CUDA_CHECK(cudaMallocHost(&host, 4 * d.plane * sizeof(double)));
    }
    ~HaloExchange() {
        if (host) CUDA_CHECK(cudaFreeHost(host));
        CUDA_CHECK(cudaEventDestroy(ready));
        CUDA_CHECK(cudaStreamDestroy(transfer));
    }
    void exchange(double* field) {
        if (!neighbors) return;
        MPI_Request requests[4];
        int n = 0;
        const int count = static_cast<int>(d.plane);
        const size_t bytes = d.plane * sizeof(double);
        if (direct) {
            // MPI may not understand CUDA streams. Wait only for boundary
            // production, leaving the interior kernel running concurrently.
            CUDA_CHECK(cudaEventSynchronize(ready));
            if (d.lower != MPI_PROC_NULL) {
                MPI_Irecv(field, count, MPI_DOUBLE, d.lower, 1, comm, &requests[n++]);
                MPI_Isend(field + d.plane, count, MPI_DOUBLE, d.lower, 0, comm, &requests[n++]);
            }
            if (d.upper != MPI_PROC_NULL) {
                MPI_Irecv(field + (d.planes + 1) * d.plane, count, MPI_DOUBLE,
                          d.upper, 0, comm, &requests[n++]);
                MPI_Isend(field + d.planes * d.plane, count, MPI_DOUBLE,
                          d.upper, 1, comm, &requests[n++]);
            }
            MPI_Waitall(n, requests, MPI_STATUSES_IGNORE);
            return;
        }
        if (d.lower != MPI_PROC_NULL)
            MPI_Irecv(host + 2 * d.plane, count, MPI_DOUBLE, d.lower, 1, comm, &requests[n++]);
        if (d.upper != MPI_PROC_NULL)
            MPI_Irecv(host + 3 * d.plane, count, MPI_DOUBLE, d.upper, 0, comm, &requests[n++]);
        CUDA_CHECK(cudaStreamWaitEvent(transfer, ready, 0));
        if (d.lower != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(host, field + d.plane, bytes, cudaMemcpyDeviceToHost, transfer));
        if (d.upper != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(host + d.plane, field + d.planes * d.plane,
                                       bytes, cudaMemcpyDeviceToHost, transfer));
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        if (d.lower != MPI_PROC_NULL)
            MPI_Isend(host, count, MPI_DOUBLE, d.lower, 0, comm, &requests[n++]);
        if (d.upper != MPI_PROC_NULL)
            MPI_Isend(host + d.plane, count, MPI_DOUBLE, d.upper, 1, comm, &requests[n++]);
        // MPI progresses while the GPU computes the interior on its own stream.
        MPI_Waitall(n, requests, MPI_STATUSES_IGNORE);
        if (d.lower != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(field, host + 2 * d.plane, bytes, cudaMemcpyHostToDevice, transfer));
        if (d.upper != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(field + (d.planes + 1) * d.plane, host + 3 * d.plane,
                                       bytes, cudaMemcpyHostToDevice, transfer));
        CUDA_CHECK(cudaStreamSynchronize(transfer));
    }
    template<bool chemical>
    void phase(const double* field, const double* old, double* result, cudaStream_t compute,
               bool communicate = true) {
        if (!communicate || !neighbors) {
            launch<chemical>(field, old, result, d, 1, d.planes, false, compute);
            return;
        }
        launch<chemical>(field, old, result, d, 1, std::min<size_t>(2, d.planes), true, compute);
        CUDA_CHECK(cudaEventRecord(ready, compute));
        if (d.planes > 2)
            launch<chemical>(field, old, result, d, 2, d.planes - 2, false, compute);
        exchange(result);
    }
};

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}


static size_t dimension(const char* value) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long n = strtoull(value, &end, 10);
    if (value[0] == '-' || !value[0] || *end || errno || n > std::numeric_limits<size_t>::max()) {
        fprintf(stderr, "Invalid grid dimension: %s\n", value);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<size_t>(n);
}

static int run(int argc, char** argv, int worldRank, int worldSize) {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = dimension(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = dimension(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = dimension(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
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
    if (!nx || nx > maxElements / ny || nx * ny > maxElements / nz || nx * ny > INT_MAX) {
        if (!worldRank) fprintf(stderr, "Invalid or oversized grid (a halo plane must fit an MPI count).\n");
        return 1;
    }
    const size_t plane = nx * ny, volume = plane * nz;
    const int ranks = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm);
    // Avoid empty slabs when launched with more ranks than Z planes.
    if (comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm local;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank;
    MPI_Comm_rank(local, &localRank);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA device is required.\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&local);
    const size_t base = nz / ranks, extra = nz % ranks;
    Domain d{nx, ny, base + (size_t(rank) < extra),
             size_t(rank) * base + std::min<size_t>(rank, extra), plane,
             rank ? rank - 1 : MPI_PROC_NULL, rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL};
    if (d.planes > maxElements / plane - 2) {
        fprintf(stderr, "Local allocation size overflow.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t cells = d.planes * plane;
    const size_t bytes = (d.planes + 2) * plane * sizeof(double);
    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    double *cold, *cnew, *mu, *initial;
    CUDA_CHECK(cudaMalloc(&cold, bytes));
    CUDA_CHECK(cudaMalloc(&cnew, bytes));
    CUDA_CHECK(cudaMalloc(&mu, bytes));
    CUDA_CHECK(cudaMallocHost(&initial, cells * sizeof(double)));
    // Integer arithmetic and global linear IDs match the serial initializer.
    const int threads = std::min(8, omp_get_max_threads());
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < cells; ++i) {
        const size_t linear = d.offset * plane + i;
        const double pseudo = (((linear + 1) * size_t(1299709)) % volume) / static_cast<double>(volume);
        initial[i] = -1.0 + 2.0 * pseudo;
    }
    cudaStream_t compute;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(cold + plane, initial, cells * sizeof(double), cudaMemcpyHostToDevice, compute));
    int resultCode = 0;
    {
        HaloExchange halo(d, comm);
        CUDA_CHECK(cudaEventRecord(halo.ready, compute));
        halo.exchange(cold);
        CUDA_CHECK(cudaStreamSynchronize(compute));
        CUDA_CHECK(cudaFreeHost(initial));
        // Replaying two steps leaves ping-pong pointers unchanged and amortizes
        // launch overhead on a single GPU without changing arithmetic order.
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        if (ranks == 1 && iterations >= 2) {
            CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
            launch<true>(cold, cold, mu, d, 1, d.planes, false, compute);
            launch<false>(mu, cold, cnew, d, 1, d.planes, false, compute);
            launch<true>(cnew, cnew, mu, d, 1, d.planes, false, compute);
            launch<false>(mu, cnew, cold, d, 1, d.planes, false, compute);
            CUDA_CHECK(cudaStreamEndCapture(compute, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphUpload(executable, compute));
            CUDA_CHECK(cudaStreamSynchronize(compute));
        }
        if (!rank) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        int t = 0;
        if (executable) {
            for (; t + 1 < iterations; t += 2) CUDA_CHECK(cudaGraphLaunch(executable, compute));
        }
        for (; t < iterations; ++t) {
            halo.phase<true>(cold, cold, mu, compute);
            halo.phase<false>(mu, cold, cnew, compute, t + 1 < iterations);
            std::swap(cold, cnew);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute));
        const double elapsed = MPI_Wtime() - start;
        double seconds = 0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!rank) {
            printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
            printf("Performance: %.3f MCellUpdates/s\n", double(volume) * iterations / seconds / 1e6);
        }
        if (executable) CUDA_CHECK(cudaGraphExecDestroy(executable));
        if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
        if (printResults || validate) {
            std::vector<double> host(cells);
            CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, cells * sizeof(double), cudaMemcpyDeviceToHost));
            if (printResults) {
                // Rank-ordered gather preserves the original Kahan sum, samples
                // and bytewise hash. Chunk messages to avoid MPI int overflow.
                std::vector<double> full;
                if (!rank) {
                    full.resize(volume);
                    std::copy(host.begin(), host.end(), full.begin());
                }
                for (int r = 1; r < ranks; ++r) {
                    const size_t n = (base + (size_t(r) < extra)) * plane;
                    const size_t offset = (size_t(r) * base + std::min<size_t>(r, extra)) * plane;
                    for (size_t j = 0; j < n;) {
                        const int count = static_cast<int>(std::min<size_t>(n - j, INT_MAX));
                        if (!rank) MPI_Recv(full.data() + offset + j, count, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                        else if (rank == r) MPI_Send(host.data() + j, count, MPI_DOUBLE, 0, 2, comm);
                        j += count;
                    }
                }
                if (!rank) print_results(full, "Concentration");
            }
            if (validate) {
                double minimum = std::numeric_limits<double>::infinity();
                double maximum = -std::numeric_limits<double>::infinity();
                int bad = 0;
                #pragma omp parallel for schedule(static) num_threads(threads) reduction(min:minimum) reduction(max:maximum) reduction(|:bad)
                for (size_t i = 0; i < cells; ++i) {
                    bad |= !std::isfinite(host[i]);
                    minimum = std::min(minimum, host[i]);
                    maximum = std::max(maximum, host[i]);
                }
                double globalMin, globalMax;
                int globalBad;
                MPI_Allreduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, comm);
                MPI_Allreduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
                MPI_Allreduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);
                resultCode = globalBad || globalMin < -10.0 || globalMax > 10.0;
                if (!rank) {
                    printf("Validating result...\n");
                    if (globalBad) printf("Validation failed: found NaN or Inf value\n");
                    else {
                        printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
                        if (resultCode) printf("Validation failed: values out of expected range\n");
                    }
                    printf("Validation: %s\n", resultCode ? "FAILED" : "PASSED");
                }
            }
        }
    }
    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));
    CUDA_CHECK(cudaStreamDestroy(compute));
    MPI_Comm_free(&comm);
    return resultCode;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int code = 1;
    try {
        code = run(argc, argv, rank, size);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return code;
}
