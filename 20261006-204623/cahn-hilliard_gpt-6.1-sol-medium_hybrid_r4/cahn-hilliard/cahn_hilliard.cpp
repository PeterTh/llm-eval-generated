#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
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
#define CUDA(call) cudaCheck((call), #call)

// Interior planes are [1, nz]; planes 0 and nz+1 are MPI ghosts.
// At physical faces use the central value, exactly as in the serial solver.
__device__ __forceinline__ double laplacian(const double* __restrict__ c,
        size_t i, size_t x, size_t y, size_t z, size_t nx, size_t ny,
        size_t nz, bool lower, bool upper) {
    const size_t plane = nx * ny;
    const double center = c[i];
    const double cxx = c[i + (x + 1 < nx)] + c[i - (x > 0)] - 2.0 * center;
    const double cyy = c[i + (y + 1 < ny ? nx : 0)] +
                       c[i - (y > 0 ? nx : 0)] - 2.0 * center;
    const double czz = c[i + (z == nz && upper ? 0 : plane)] +
                       c[i - (z == 1 && lower ? 0 : plane)] - 2.0 * center;
    return cxx + cyy + czz; // dx = dy = dz = 1 in the benchmark
}

// A grid-stride kernel keeps x contiguous across warps, including narrow grids.
// Boundary mode processes just the two end planes in a single launch.
template<bool Potential, bool Boundary>
__global__ void stencil(const double* __restrict__ c,
        const double* __restrict__ mu, double* __restrict__ out,
        size_t nx, size_t ny, size_t nz, size_t first, size_t planes,
        bool lower, bool upper) {
    const size_t plane = nx * ny;
    const size_t count = planes * plane;
    for (size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         j < count; j += size_t(blockDim.x) * gridDim.x) {
        const size_t p = j / plane;
        const size_t z = Boundary ? (p == 0 ? 1 : nz) : first + p;
        const size_t xy = j % plane;
        const size_t x = xy % nx, y = xy / nx;
        const size_t i = z * plane + xy;
        if constexpr (Potential) {
            const double cv = c[i];
            constexpr double e_AA = -(2.0 / 9.0);
            constexpr double e_BB = -(2.0 / 9.0);
            constexpr double e_AB = 2.0 / 9.0;
            out[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                   + 3.0 * cv + cv * cv * cv
                   - 0.5 * laplacian(c, i, x, y, z, nx, ny, nz, lower, upper);
        } else {
            out[i] = c[i] + 0.01 * laplacian(mu, i, x, y, z, nx, ny, nz, lower, upper);
        }
    }
}

template<bool Potential, bool Boundary = false>
static void launch(const double* c, const double* mu, double* out,
        size_t nx, size_t ny, size_t nz, size_t first, size_t planes,
        bool lower, bool upper, cudaStream_t stream) {
    if (!planes) return;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>(
        (nx * ny * planes + 255) / 256, 65535));
    stencil<Potential, Boundary><<<blocks, 256, 0, stream>>>(
        c, mu, out, nx, ny, nz, first, planes, lower, upper);
    CUDA(cudaGetLastError());
}

// Communication is funneled through the main thread. Receives are posted
// before sending; interior kernels execute while MPI progresses in Waitall.
// Pinned staging works with ordinary MPI. Open MPI's runtime CUDA query
// enables direct device transfers when that capability is actually available.
struct Halo {
    MPI_Comm comm;
    int prev, next, count;
    size_t nz, plane;
    bool direct;
    cudaStream_t transfer;
    cudaEvent_t ready;
    double* host = nullptr;
    double* staging = nullptr;
    unsigned bank = 0;
    MPI_Request requests[4];
    int nrequests = 0;

    Halo(MPI_Comm communicator, int rank, int ranks, size_t p, size_t z)
        : comm(communicator), prev(rank ? rank - 1 : MPI_PROC_NULL),
          next(rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL),
          count(static_cast<int>(p)), nz(z), plane(p), direct(false) {
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        direct = MPIX_Query_cuda_support() != 0;
#endif
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        if (!direct) CUDA(cudaMallocHost(&host, 8 * plane * sizeof(double)));
    }

    void begin(double* field, cudaStream_t compute) {
        nrequests = 0;
        // Alternate staging buffers for c and mu: MPI must not overwrite a
        // received plane while the previous asynchronous H2D still reads it.
        if (!direct) {
            staging = host + bank * 4 * plane;
            bank ^= 1;
        }
        CUDA(cudaEventRecord(ready, compute));
        CUDA(cudaStreamWaitEvent(transfer, ready, 0));
        if (prev != MPI_PROC_NULL)
            MPI_Irecv(direct ? field : staging + 2 * plane, count, MPI_DOUBLE,
                      prev, 1, comm, &requests[nrequests++]);
        if (next != MPI_PROC_NULL)
            MPI_Irecv(direct ? field + (nz + 1) * plane : staging + 3 * plane,
                      count, MPI_DOUBLE, next, 0, comm, &requests[nrequests++]);
        if (!direct) {
            if (prev != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(staging, field + plane, plane * sizeof(double),
                                     cudaMemcpyDeviceToHost, transfer));
            if (next != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(staging + plane, field + nz * plane, plane * sizeof(double),
                                     cudaMemcpyDeviceToHost, transfer));
        }
    }

    void finish(double* field, cudaStream_t compute) {
        // Only the producer and outbound copies must finish, not the interior kernel.
        CUDA(cudaStreamSynchronize(transfer));
        if (prev != MPI_PROC_NULL)
            MPI_Isend(direct ? field + plane : staging, count, MPI_DOUBLE,
                      prev, 0, comm, &requests[nrequests++]);
        if (next != MPI_PROC_NULL)
            MPI_Isend(direct ? field + nz * plane : staging + plane, count, MPI_DOUBLE,
                      next, 1, comm, &requests[nrequests++]);
        MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE);
        if (!direct) {
            if (prev != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(field, staging + 2 * plane, plane * sizeof(double),
                                     cudaMemcpyHostToDevice, transfer));
            if (next != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(field + (nz + 1) * plane, staging + 3 * plane,
                                     plane * sizeof(double), cudaMemcpyHostToDevice, transfer));
        }
        CUDA(cudaEventRecord(ready, transfer));
        CUDA(cudaStreamWaitEvent(compute, ready, 0));
    }

    ~Halo() {
        if (host) CUDA(cudaFreeHost(host));
        CUDA(cudaEventDestroy(ready));
        CUDA(cudaStreamDestroy(transfer));
    }
};

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n"
           "  -x <num>     Grid size in X dimension (default: 64)\n"
           "  -y <num>     Grid size in Y dimension (default: same as X)\n"
           "  -z <num>     Grid size in Z dimension (default: same as X)\n"
           "  -i <num>     Number of time steps (default: 20)\n"
           "  -v           Enable validation\n"
           "  -r           Print results for external validation\n"
           "  -h           Show this help message\n");
}

static bool number(const char* s, size_t& value) {
    if (!*s || *s == '-') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long n = strtoull(s, &end, 10);
    if (errno || *end || n > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(n);
    return true;
}

static int run(int argc, char** argv, int worldRank, int worldSize) {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h")) {
            if (!worldRank) printUsage(argv[0]);
            return 0;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
                  !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            size_t value;
            if (!number(argv[++i], value) || (option == 'x' && !value) ||
                (option == 'i' && value > INT_MAX)) {
                if (!worldRank) fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
                return 1;
            }
            if (option == 'x') nx = value;
            else if (option == 'y') ny = value;
            else if (option == 'z') nz = value;
            else iterations = static_cast<int>(value);
        } else {
            if (!worldRank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    // Plane MPI messages use int counts; all array offsets use size_t.
    const size_t limit = std::numeric_limits<size_t>::max();
    if (nx > limit / ny || nx * ny > INT_MAX ||
        nz > limit / (nx * ny) || nz > limit / (nx * ny * sizeof(double)) - 2) {
        if (!worldRank) fprintf(stderr, "Grid dimensions exceed supported address or MPI count range\n");
        return 1;
    }
    const size_t plane = nx * ny, volume = plane * nz;
    const int activeSize = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    const size_t localZ = nz / activeSize + (size_t(rank) < nz % activeSize);
    const size_t startZ = size_t(rank) * (nz / activeSize) + std::min<size_t>(rank, nz % activeSize);
    const size_t localCount = plane * localZ;
    const bool lower = rank == 0, upper = rank + 1 == activeSize;

    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA accelerator is required\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // With per-rank CUDA_VISIBLE_DEVICES each rank sees its assigned accelerator.
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&node);
    cudaStream_t compute;
    CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    double *cold, *cnew, *mu;
    const size_t bytes = (localZ + 2) * plane * sizeof(double);
    CUDA(cudaMalloc(&cold, bytes));
    CUDA(cudaMalloc(&cnew, bytes));
    CUDA(cudaMalloc(&mu, bytes));
    std::vector<double> local(localCount);
    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\nTime steps: %d\n", nx, ny, nz, iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallel execution: %d MPI ranks, CUDA, OpenMP\n", activeSize);
        printf("Initializing concentration field...\n");
    }
#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < localCount; ++j) {
        const size_t linear_id = startZ * plane + j;
        const double pseudo = (((linear_id + 1) * 1299709) % volume) / static_cast<double>(volume);
        local[j] = -1.0 + 2.0 * pseudo;
    }
    CUDA(cudaMemcpyAsync(cold + plane, local.data(), localCount * sizeof(double),
                         cudaMemcpyHostToDevice, compute));
    CUDA(cudaStreamSynchronize(compute));
    {
        Halo halo(comm, rank, activeSize, plane, localZ);
        // Capture a pair of steps so repeated graph launches return to the
        // same concentration buffer and avoid CPU launch overhead on one GPU.
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executableGraph = nullptr;
        if (activeSize == 1 && iterations >= 2) {
            CUDA(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
            launch<true>(cold, mu, mu, nx, ny, localZ, 1, localZ, lower, upper, compute);
            launch<false>(cold, mu, cnew, nx, ny, localZ, 1, localZ, lower, upper, compute);
            launch<true>(cnew, mu, mu, nx, ny, localZ, 1, localZ, lower, upper, compute);
            launch<false>(cnew, mu, cold, nx, ny, localZ, 1, localZ, lower, upper, compute);
            CUDA(cudaStreamEndCapture(compute, &graph));
            CUDA(cudaGraphInstantiate(&executableGraph, graph, nullptr, nullptr, 0));
            CUDA(cudaGraphUpload(executableGraph, compute));
            CUDA(cudaStreamSynchronize(compute));
        }
        if (!rank) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            if (executableGraph && t + 1 < iterations) {
                CUDA(cudaGraphLaunch(executableGraph, compute));
                ++t;
                continue;
            }
            if (activeSize == 1) {
                launch<true>(cold, mu, mu, nx, ny, localZ, 1, localZ, lower, upper, compute);
                launch<false>(cold, mu, cnew, nx, ny, localZ, 1, localZ, lower, upper, compute);
            } else {
                halo.begin(cold, compute);
                launch<true>(cold, mu, mu, nx, ny, localZ, 2,
                             localZ > 2 ? localZ - 2 : 0, lower, upper, compute);
                halo.finish(cold, compute);
                launch<true, true>(cold, mu, mu, nx, ny, localZ, 1,
                                  std::min<size_t>(localZ, 2), lower, upper, compute);
                halo.begin(mu, compute);
                launch<false>(cold, mu, cnew, nx, ny, localZ, 2,
                              localZ > 2 ? localZ - 2 : 0, lower, upper, compute);
                halo.finish(mu, compute);
                launch<false, true>(cold, mu, cnew, nx, ny, localZ, 1,
                                   std::min<size_t>(localZ, 2), lower, upper, compute);
            }
            std::swap(cold, cnew);
        }
        CUDA(cudaStreamSynchronize(compute));
        const double elapsed = MPI_Wtime() - start;
        if (executableGraph) CUDA(cudaGraphExecDestroy(executableGraph));
        if (graph) CUDA(cudaGraphDestroy(graph));
        double seconds = 0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!rank) {
            printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
            printf("Performance: %.3f MCellUpdates/s\n",
                   seconds > 0 ? double(volume) * iterations / seconds / 1e6 : 0.0);
        }
    }
    if (validate || printResults)
        CUDA(cudaMemcpy(local.data(), cold + plane, localCount * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA(cudaFree(cold));
    CUDA(cudaFree(cnew));
    CUDA(cudaFree(mu));
    CUDA(cudaStreamDestroy(compute));

    if (printResults) {
        // Gather only on request. Chunking avoids MPI_Gatherv's int displacement
        // limit and preserves the exact global order used by print_results.
        std::vector<double> global;
        if (!rank) {
            global.resize(volume);
            std::copy(local.begin(), local.end(), global.begin());
        }
        constexpr size_t chunk = 1 << 26;
        for (int source = 1; source < activeSize; ++source) {
            const size_t z = nz / activeSize + (size_t(source) < nz % activeSize);
            const size_t offset = (size_t(source) * (nz / activeSize) +
                                  std::min<size_t>(source, nz % activeSize)) * plane;
            for (size_t j = 0; j < z * plane; j += chunk) {
                const int count = static_cast<int>(std::min(chunk, z * plane - j));
                if (rank == source) MPI_Send(local.data() + j, count, MPI_DOUBLE, 0, 2, comm);
                if (!rank) MPI_Recv(global.data() + offset + j, count, MPI_DOUBLE, source, 2, comm, MPI_STATUS_IGNORE);
            }
        }
        if (!rank) print_results(global, "Concentration");
    }
    int status = 0;
    if (validate) {
        double minVal = std::numeric_limits<double>::infinity();
        double maxVal = -std::numeric_limits<double>::infinity();
        int invalid = 0;
#pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal) reduction(|:invalid)
        for (size_t j = 0; j < localCount; ++j) {
            invalid |= !std::isfinite(local[j]);
            minVal = std::min(minVal, local[j]);
            maxVal = std::max(maxVal, local[j]);
        }
        double globalMin, globalMax;
        int globalInvalid;
        MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_MAX, comm);
        status = globalInvalid || globalMin < -10.0 || globalMax > 10.0;
        if (!rank) {
            printf("Validating result...\n");
            if (globalInvalid) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (status) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
    }
    MPI_Comm_free(&comm);
    return status;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Ranks without a slab still receive the same exit status.
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
