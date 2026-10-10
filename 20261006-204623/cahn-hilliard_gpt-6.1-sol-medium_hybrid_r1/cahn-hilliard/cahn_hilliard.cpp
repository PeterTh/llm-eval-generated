#include <algorithm>
#include <climits>
#include <cerrno>
#include <exception>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>
#include "../common/results_output.hpp"

static void cudaCheck(cudaError_t error, const char* expression, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at line %d (%s): %s\n", line, expression,
                cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(expression) cudaCheck((expression), #expression, __LINE__)

// Each rank owns planes 1..localZ; planes 0 and localZ+1 are MPI halos.
// At the global faces the neighbor is the cell itself, as in the original.
__device__ __forceinline__ double laplacian(const double* __restrict__ field,
        size_t i, size_t x, size_t y, size_t z, size_t nx, size_t ny,
        size_t localZ, bool lower, bool upper) {
    const size_t plane = nx * ny;
    const size_t xp = x + 1 < nx ? i + 1 : i;
    const size_t xn = x > 0 ? i - 1 : i;
    const size_t yp = y + 1 < ny ? i + nx : i;
    const size_t yn = y > 0 ? i - nx : i;
    const size_t zp = z < localZ || upper ? i + plane : i;
    const size_t zn = z > 1 || lower ? i - plane : i;
    const double center = field[i];
    const double cxx = field[xp] + field[xn] - 2.0 * center;
    const double cyy = field[yp] + field[yn] - 2.0 * center;
    const double czz = field[zp] + field[zn] - 2.0 * center;
    // dx = dy = dz = 1.0 in the original benchmark.
    return cxx + cyy + czz;
}

// The same kernels handle either all planes, interior planes, or the two faces.
// A grid-stride loop also supports slabs larger than CUDA's launch grid limits.
template<bool Chemical>
__global__ void stencil(const double* __restrict__ cold,
        const double* __restrict__ field, double* __restrict__ out,
        size_t nx, size_t ny, size_t localZ, bool lower, bool upper,
        size_t firstZ, size_t planes, bool faces) {
    const size_t plane = nx * ny;
    // Map complete warps to contiguous X cells; use Y/Z launch dimensions
    // instead of costly runtime 64-bit division for every stencil evaluation.
    for (size_t p = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
         p < planes; p += static_cast<size_t>(blockDim.z) * gridDim.z) {
        const size_t z = faces ? (p == 0 ? 1 : localZ) : firstZ + p;
        for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             y < ny; y += static_cast<size_t>(blockDim.y) * gridDim.y) {
            for (size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 x < nx; x += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const size_t i = z * plane + y * nx + x;
                const double lap = laplacian(field, i, x, y, z, nx, ny,
                                             localZ, lower, upper);
                const double cv = cold[i];
                if constexpr (Chemical) {
                    constexpr double e_AA = -(2.0 / 9.0);
                    constexpr double e_BB = -(2.0 / 9.0);
                    constexpr double e_AB = 2.0 / 9.0;
                    out[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB
                               - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv - 0.5 * lap;
                } else {
                    out[i] = cv + 0.01 * lap;
                }
            }
        }
    }
}

template<bool Chemical>
static void launch(const double* cold, const double* field, double* out,
        size_t nx, size_t ny, size_t localZ, bool lower, bool upper,
        size_t firstZ, size_t planes, bool faces, cudaStream_t stream) {
    if (planes == 0) return;
    const dim3 block(32, 4, 2);
    const dim3 grid(static_cast<unsigned>(std::min<size_t>((nx - 1) / block.x + 1, 65535)),
                    static_cast<unsigned>(std::min<size_t>((ny - 1) / block.y + 1, 65535)),
                    static_cast<unsigned>(std::min<size_t>((planes - 1) / block.z + 1, 65535)));
    stencil<Chemical><<<grid, block, 0, stream>>>(cold, field, out, nx, ny,
        localZ, lower, upper, firstZ, planes, faces);
    CUDA_CHECK(cudaGetLastError());
}

// CUDA-aware MPI can exchange device faces directly. Otherwise pinned host
// staging uses a separate transfer stream while interior work runs.
class HaloExchange {
    size_t plane, localZ;
    int lower, upper;
    MPI_Comm comm;
    bool direct;
    double* staging = nullptr;
    cudaStream_t transfer;
    cudaEvent_t received;
    std::vector<MPI_Request> requests;

    void post(double* buffer, int peer, int tag, bool send) {
        // Split planes so MPI's int count never truncates a large grid.
        for (size_t offset = 0; offset < plane;) {
            const int count = static_cast<int>(std::min<size_t>(plane - offset, INT_MAX));
            MPI_Request request;
            if (send)
                MPI_Isend(buffer + offset, count, MPI_DOUBLE, peer, tag, comm, &request);
            else
                MPI_Irecv(buffer + offset, count, MPI_DOUBLE, peer, tag, comm, &request);
            requests.push_back(request);
            offset += static_cast<size_t>(count);
        }
    }
public:
    HaloExchange(size_t plane_, size_t localZ_, int rank, int ranks, MPI_Comm comm_, bool direct_)
        : plane(plane_), localZ(localZ_),
          lower(rank == 0 ? MPI_PROC_NULL : rank - 1),
          upper(rank + 1 == ranks ? MPI_PROC_NULL : rank + 1), comm(comm_), direct(direct_) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&received, cudaEventDisableTiming));
        if (!direct)
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&staging),
                                      4 * plane * sizeof(double)));
        requests.reserve(4 * ((plane - 1) / INT_MAX + 1));
    }
    ~HaloExchange() {
        if (staging) CUDA_CHECK(cudaFreeHost(staging));
        CUDA_CHECK(cudaEventDestroy(received));
        CUDA_CHECK(cudaStreamDestroy(transfer));
    }
    // Enqueue boundary transfers after the producer has completed, but let the
    // host launch interior work before waiting for the outbound copies.
    void begin(double* field, cudaEvent_t ready) {
        requests.clear();
        if (direct) {
            // Ordinary CUDA-aware MPI calls do not understand CUDA streams.
            // Ensure the producer is complete before MPI accesses its faces.
            CUDA_CHECK(cudaEventSynchronize(ready));
            if (lower != MPI_PROC_NULL) post(field, lower, 1, false);
            if (upper != MPI_PROC_NULL) post(field + (localZ + 1) * plane, upper, 0, false);
            if (lower != MPI_PROC_NULL) post(field + plane, lower, 0, true);
            if (upper != MPI_PROC_NULL) post(field + localZ * plane, upper, 1, true);
            return;
        }
        // MPI must not overwrite pinned receive storage until the preceding
        // asynchronous H2D copies have consumed it. Stream ordering alone
        // cannot order host-side MPI writes to that storage.
        CUDA_CHECK(cudaEventSynchronize(received));
        CUDA_CHECK(cudaStreamWaitEvent(transfer, ready, 0));
        if (lower != MPI_PROC_NULL) {
            post(staging + 2 * plane, lower, 1, false);
            CUDA_CHECK(cudaMemcpyAsync(staging, field + plane, plane * sizeof(double),
                                       cudaMemcpyDeviceToHost, transfer));
        }
        if (upper != MPI_PROC_NULL) {
            post(staging + 3 * plane, upper, 0, false);
            CUDA_CHECK(cudaMemcpyAsync(staging + plane, field + localZ * plane,
                                       plane * sizeof(double), cudaMemcpyDeviceToHost, transfer));
        }
    }
    void finish(double* field, cudaStream_t compute) {
        if (direct) {
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            return;
        }
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        if (lower != MPI_PROC_NULL) post(staging, lower, 0, true);
        if (upper != MPI_PROC_NULL) post(staging + plane, upper, 1, true);
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        if (lower != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(field, staging + 2 * plane, plane * sizeof(double),
                                       cudaMemcpyHostToDevice, transfer));
        if (upper != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(field + (localZ + 1) * plane, staging + 3 * plane,
                                       plane * sizeof(double), cudaMemcpyHostToDevice, transfer));
        CUDA_CHECK(cudaEventRecord(received, transfer));
        CUDA_CHECK(cudaStreamWaitEvent(compute, received, 0));
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

static bool parseSize(const char* text, size_t& value) {
    if (*text == '-' || *text == '\0') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || *end || parsed > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(parsed);
    return true;
}

static int run(size_t nx, size_t ny, size_t nz, int iterations,
                bool validate, bool printResults, MPI_Comm comm) {
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    const size_t plane = nx * ny;
    const size_t volume = plane * nz;
    const size_t localZ = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t startZ = (nz / ranks) * rank + std::min<size_t>(rank, nz % ranks);
    const size_t owned = plane * localZ;
    const size_t bytes = (localZ + 2) * plane * sizeof(double);

    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int nodeRank, devices;
    MPI_Comm_rank(node, &nodeRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Each participating MPI rank requires a CUDA device.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Also works when a launcher masks a different single GPU for each rank.
    CUDA_CHECK(cudaSetDevice(nodeRank % devices));
    MPI_Comm_free(&node);

    std::vector<double> host(owned);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < owned; ++i) {
        const size_t global = startZ * plane + i;
        const double pseudo = (((global + 1) * 1299709) % volume) / static_cast<double>(volume);
        host[i] = -1.0 + 2.0 * pseudo;
    }
    double *cold, *cnew, *mu;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cold), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cnew), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mu), bytes));
    CUDA_CHECK(cudaMemcpy(cold + plane, host.data(), owned * sizeof(double), cudaMemcpyHostToDevice));
    cudaStream_t compute;
    cudaEvent_t ready;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ready, compute));

    // Use device-buffer MPI when the implementation can confirm CUDA support.
    // Other MPI installations use the portable asynchronous staging path.
    int cudaAware = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    cudaAware = MPIX_Query_cuda_support() != 0;
#endif
    int allCudaAware = 0;
    MPI_Allreduce(&cudaAware, &allCudaAware, 1, MPI_INT, MPI_MIN, comm);
    const bool lower = rank > 0, upper = rank + 1 < ranks;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    // Capture two time steps to restore the original buffer order. Graph replay
    // removes repeated launch overhead when the entire domain fits on one GPU.
    if (ranks == 1 && iterations >= 2) {
        CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
        launch<true>(cold, cold, mu, nx, ny, localZ, false, false, 1, localZ, false, compute);
        launch<false>(cold, mu, cnew, nx, ny, localZ, false, false, 1, localZ, false, compute);
        launch<true>(cnew, cnew, mu, nx, ny, localZ, false, false, 1, localZ, false, compute);
        launch<false>(cnew, mu, cold, nx, ny, localZ, false, false, 1, localZ, false, compute);
        CUDA_CHECK(cudaStreamEndCapture(compute, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(graphExec, compute));
    }
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    // Allocate halo storage before starting the timed region.
    {
        HaloExchange halos(plane, localZ, rank, ranks, comm, allCudaAware != 0 || ranks == 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        int t = 0;
        if (graphExec) {
            for (; t + 1 < iterations; t += 2) CUDA_CHECK(cudaGraphLaunch(graphExec, compute));
        }
        for (; t < iterations; ++t) {
            if (ranks == 1) {
                launch<true>(cold, cold, mu, nx, ny, localZ, false, false, 1, localZ, false, compute);
                launch<false>(cold, mu, cnew, nx, ny, localZ, false, false, 1, localZ, false, compute);
            } else {
                halos.begin(cold, ready);
                launch<true>(cold, cold, mu, nx, ny, localZ, lower, upper, 2,
                             localZ > 2 ? localZ - 2 : 0, false, compute);
                halos.finish(cold, compute);
                launch<true>(cold, cold, mu, nx, ny, localZ, lower, upper, 1,
                             localZ > 1 ? 2 : 1, true, compute);
                CUDA_CHECK(cudaEventRecord(ready, compute));
                halos.begin(mu, ready);
                launch<false>(cold, mu, cnew, nx, ny, localZ, lower, upper, 2,
                              localZ > 2 ? localZ - 2 : 0, false, compute);
                halos.finish(mu, compute);
                launch<false>(cold, mu, cnew, nx, ny, localZ, lower, upper, 1,
                              localZ > 1 ? 2 : 1, true, compute);
                CUDA_CHECK(cudaEventRecord(ready, compute));
            }
            std::swap(cold, cnew);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute));
        const double elapsed = MPI_Wtime() - start;
        double seconds;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
            printf("Performance: %.3f MCellUpdates/s\n",
                   static_cast<double>(volume) * iterations / seconds / 1e6);
        }
    }
    if (validate || printResults)
        CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, owned * sizeof(double), cudaMemcpyDeviceToHost));

    // Gathering is only needed for the original ordered hash/Kahan sum output.
    // Chunked transfers avoid MPI_Gatherv's int displacement/count limitation.
    if (printResults) {
        std::vector<double> global;
        if (rank == 0) {
            global.resize(volume);
            std::copy(host.begin(), host.end(), global.begin());
        }
        for (int peer = 1; peer < ranks; ++peer) {
            const size_t zCount = nz / ranks + (static_cast<size_t>(peer) < nz % ranks);
            const size_t zStart = (nz / ranks) * peer + std::min<size_t>(peer, nz % ranks);
            const size_t count = plane * zCount;
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
                if (rank == peer)
                    MPI_Send(host.data() + offset, chunk, MPI_DOUBLE, 0, 2, comm);
                if (rank == 0)
                    MPI_Recv(global.data() + zStart * plane + offset, chunk, MPI_DOUBLE,
                             peer, 2, comm, MPI_STATUS_IGNORE);
                offset += static_cast<size_t>(chunk);
            }
        }
        if (rank == 0) print_results(global, "Concentration");
    }
    int valid = 1;
    if (validate) {
        int bad = 0, globalBad;
        double minVal = std::numeric_limits<double>::infinity();
        double maxVal = -std::numeric_limits<double>::infinity();
        #pragma omp parallel for reduction(|:bad) reduction(min:minVal) reduction(max:maxVal) schedule(static)
        for (size_t i = 0; i < owned; ++i) {
            bad |= !std::isfinite(host[i]);
            minVal = std::min(minVal, host[i]);
            maxVal = std::max(maxVal, host[i]);
        }
        double globalMin, globalMax;
        MPI_Allreduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, comm);
        MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);
        valid = !globalBad && globalMin >= -10.0 && globalMax <= 10.0;
        if (rank == 0) {
            printf("Validating result...\n");
            if (globalBad) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (!valid) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    if (graphExec) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(ready));
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));
    return valid;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            size_t value = 0;
            if (!parseSize(argv[++i], value)) bad = true;
            if (option == 'x') nx = value;
            else if (option == 'y') ny = value;
            else if (option == 'z') nz = value;
            else if (value <= INT_MAX) iterations = static_cast<int>(value);
            else bad = true;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else {
            if (worldRank == 0) printf("Unknown option: %s\n", argv[i]);
            bad = true;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    // Guard all global, halo, and pinned-staging allocation arithmetic.
    const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx == 0 || ny == 0 || nz == 0 || nx > maxElements / ny ||
        nx * ny > maxElements / nz || nz > maxElements - 2 ||
        nx * ny > maxElements / (nz + 2) || nx * ny > maxElements / 4) bad = true;
    if (help || bad) {
        if (worldRank == 0) {
            if (bad) fprintf(stderr, "Invalid dimensions, iteration count, or options.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return bad ? 1 : 0;
    }
    if (worldRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    // Never assign empty slabs; extra ranks wait for the final status.
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, static_cast<size_t>(worldRank) < nz ? 0 : MPI_UNDEFINED,
                   worldRank, &active);
    int valid = 1;
    if (active != MPI_COMM_NULL) {
        try {
            valid = run(nx, ny, nz, iterations, validate, printResults, active);
        } catch (const std::exception& error) {
            fprintf(stderr, "Allocation/runtime error: %s\n", error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Comm_free(&active);
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
