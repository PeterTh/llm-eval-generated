#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#if defined(OMPI_MAJOR_VERSION) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

namespace {
int worldRank = 0;

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", worldRank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}
void mpiCheck(int error, const char* operation) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "Rank %d: %s: %.*s\n", worldRank, operation, length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}
#define CUDA(call) cudaCheck((call), #call)
#define MPI_CHECK(call) mpiCheck((call), #call)

// Arrays have one ghost plane on each side. Physical boundaries clamp to the
// center, exactly as in the serial solver; internal boundaries use MPI ghosts.
__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                           size_t i, size_t x, size_t y, size_t z,
                                           size_t nx, size_t ny, size_t plane,
                                           size_t depth, bool lower, bool upper) {
    const double center = field[i];
    const double cxx = field[i + (x + 1 < nx ? 1 : 0)]
                     + field[i - (x > 0 ? 1 : 0)] - 2.0 * center;
    const double cyy = field[i + (y + 1 < ny ? nx : 0)]
                     + field[i - (y > 0 ? nx : 0)] - 2.0 * center;
    const double czz = field[i + (z < depth || upper ? plane : 0)]
                     + field[i - (z > 1 || lower ? plane : 0)] - 2.0 * center;
    // dx = dy = dz = 1 in the benchmark.
    return cxx + cyy + czz;
}

template<bool Chemical, typename Index>
__global__ void stencil(const double* __restrict__ concentration,
                        const double* __restrict__ input,
                        double* __restrict__ output,
                        Index nx, Index ny, Index depth, Index first,
                        Index count, bool lower, bool upper) {
    const Index plane = nx * ny;
    const size_t cells = size_t(count) * plane;
    for (size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         j < cells; j += size_t(blockDim.x) * gridDim.x) {
        const Index index = static_cast<Index>(j);
        const Index x = index % nx;
        const Index y = (index / nx) % ny;
        const Index z = first + index / plane;
        const size_t i = size_t(z) * plane + size_t(y) * nx + x;
        const double cv = concentration[i];
        const double lap = laplacian(input, i, x, y, z, nx, ny, plane,
                                     depth, lower, upper);
        if constexpr (Chemical) {
            // Keep the original expression and operation order in double precision.
            constexpr double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
            output[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
                      + 3.0 * cv + cv * cv * cv - 0.5 * lap;
        } else {
            output[i] = cv + 0.01 * lap;
        }
    }
}

struct Solver {
    size_t nx, ny, depth, plane;
    int lower, upper;
    MPI_Comm comm;
    bool distributed, directMPI;
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    double* staging = nullptr;
    cudaStream_t compute{}, transfer{};
    cudaEvent_t ready{};
    cudaGraphExec_t graph = nullptr;
    int blocks;

    Solver(size_t x, size_t y, size_t d, int rank, int ranks, MPI_Comm communicator,
           int multiprocessors, bool cudaAware)
        : nx(x), ny(y), depth(d), plane(x * y),
          lower(rank > 0 ? rank - 1 : MPI_PROC_NULL),
          upper(rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL), comm(communicator),
          distributed(ranks > 1), directMPI(cudaAware),
          blocks(std::max(1, multiprocessors * 8)) {
        const size_t bytes = (depth + 2) * plane * sizeof(double);
        CUDA(cudaMalloc(&cold, bytes));
        CUDA(cudaMalloc(&cnew, bytes));
        CUDA(cudaMalloc(&mu, bytes));
        if (distributed && !directMPI)
            CUDA(cudaMallocHost(&staging, 4 * plane * sizeof(double)));
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    }
    ~Solver() {
        if (graph) CUDA(cudaGraphExecDestroy(graph));
        CUDA(cudaEventDestroy(ready));
        CUDA(cudaStreamDestroy(transfer));
        CUDA(cudaStreamDestroy(compute));
        if (staging) CUDA(cudaFreeHost(staging));
        CUDA(cudaFree(mu));
        CUDA(cudaFree(cnew));
        CUDA(cudaFree(cold));
    }

    template<bool Chemical>
    void launch(size_t first, size_t count) {
        if (!count) return;
        const size_t needed = (count * plane + 255) / 256;
        const int grid = static_cast<int>(std::min(needed, size_t(blocks)));
        // 32-bit coordinate division is substantially cheaper on the GPU.
        // Retain a 64-bit specialization for genuinely large local domains.
        if ((depth + 2) * plane <= UINT_MAX)
            stencil<Chemical, unsigned int><<<grid, 256, 0, compute>>>(
                cold, Chemical ? cold : mu, Chemical ? mu : cnew,
                nx, ny, depth, first, count, lower != MPI_PROC_NULL, upper != MPI_PROC_NULL);
        else
            stencil<Chemical, size_t><<<grid, 256, 0, compute>>>(
                cold, Chemical ? cold : mu, Chemical ? mu : cnew,
                nx, ny, depth, first, count, lower != MPI_PROC_NULL, upper != MPI_PROC_NULL);
        CUDA(cudaGetLastError());
    }
    template<bool Chemical>
    void boundaries() {
        launch<Chemical>(1, 1);
        if (depth > 1) launch<Chemical>(depth, 1);
    }

    // Interior computation runs on a separate stream while these transfers and
    // MPI requests progress. Only boundary planes cross PCIe or the network.
    void exchange(double* field) {
        if (!distributed) return;
        if (directMPI) {
            // CUDA-aware MPI can use GPUDirect and avoid PCIe staging. Wait
            // only for the send planes, while the interior kernel runs.
            CUDA(cudaEventSynchronize(ready));
            for (size_t offset = 0; offset < plane; offset += size_t(INT_MAX)) {
                const int count = static_cast<int>(std::min(plane - offset, size_t(INT_MAX)));
                MPI_Request requests[4];
                MPI_CHECK(MPI_Irecv(field + offset, count, MPI_DOUBLE,
                                    lower, 1, comm, &requests[0]));
                MPI_CHECK(MPI_Irecv(field + (depth + 1) * plane + offset, count, MPI_DOUBLE,
                                    upper, 0, comm, &requests[1]));
                MPI_CHECK(MPI_Isend(field + plane + offset, count, MPI_DOUBLE,
                                    lower, 0, comm, &requests[2]));
                MPI_CHECK(MPI_Isend(field + depth * plane + offset, count, MPI_DOUBLE,
                                    upper, 1, comm, &requests[3]));
                MPI_CHECK(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE));
            }
            return;
        }
        CUDA(cudaStreamWaitEvent(transfer, ready, 0));
        const size_t bytes = plane * sizeof(double);
        if (lower != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(staging, field + plane, bytes,
                                 cudaMemcpyDeviceToHost, transfer));
        if (upper != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(staging + plane, field + depth * plane, bytes,
                                 cudaMemcpyDeviceToHost, transfer));
        CUDA(cudaStreamSynchronize(transfer));
        // Chunking avoids the MPI int-count limit, including very large planes.
        for (size_t offset = 0; offset < plane; offset += size_t(INT_MAX)) {
            const int count = static_cast<int>(std::min(plane - offset, size_t(INT_MAX)));
            MPI_Request requests[4];
            MPI_CHECK(MPI_Irecv(staging + 2 * plane + offset, count, MPI_DOUBLE,
                                lower, 1, comm, &requests[0]));
            MPI_CHECK(MPI_Irecv(staging + 3 * plane + offset, count, MPI_DOUBLE,
                                upper, 0, comm, &requests[1]));
            MPI_CHECK(MPI_Isend(staging + offset, count, MPI_DOUBLE,
                                lower, 0, comm, &requests[2]));
            MPI_CHECK(MPI_Isend(staging + plane + offset, count, MPI_DOUBLE,
                                upper, 1, comm, &requests[3]));
            MPI_CHECK(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE));
        }
        if (lower != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(field, staging + 2 * plane, bytes,
                                 cudaMemcpyHostToDevice, transfer));
        if (upper != MPI_PROC_NULL)
            CUDA(cudaMemcpyAsync(field + (depth + 1) * plane, staging + 3 * plane, bytes,
                                 cudaMemcpyHostToDevice, transfer));
        // Stream dependency, rather than a device-wide synchronization, permits
        // the interior kernel to continue until the boundary kernel is ready.
        CUDA(cudaEventRecord(ready, transfer));
        CUDA(cudaStreamWaitEvent(compute, ready, 0));
    }

    void step() {
        if (!distributed) {
            launch<true>(1, depth);
            launch<false>(1, depth);
        } else {
            // The preceding update makes concentration send planes available.
            CUDA(cudaEventRecord(ready, compute));
            if (depth > 2) launch<true>(2, depth - 2);
            exchange(cold);
            boundaries<true>();
            CUDA(cudaEventRecord(ready, compute));
            if (depth > 2) launch<false>(2, depth - 2);
            exchange(mu);
            boundaries<false>();
        }
        std::swap(cold, cnew);
    }

    void prepare(int iterations) {
        if (distributed || iterations < 2) return;
        // Capture two steps so pointer swapping returns to the starting layout.
        cudaGraph_t captured;
        CUDA(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
        step();
        step();
        CUDA(cudaStreamEndCapture(compute, &captured));
        CUDA(cudaGraphInstantiate(&graph, captured, nullptr, nullptr, 0));
        CUDA(cudaGraphDestroy(captured));
    }

    void advance(int iterations) {
        if (graph) {
            for (int t = 0; t < iterations / 2; ++t)
                CUDA(cudaGraphLaunch(graph, compute));
            if (iterations % 2) step();
        } else {
            for (int t = 0; t < iterations; ++t) step();
        }
    }
};

size_t slabDepth(size_t nz, int rank, int ranks) {
    return nz / size_t(ranks) + (size_t(rank) < nz % size_t(ranks));
}
size_t slabStart(size_t nz, int rank, int ranks) {
    return size_t(rank) * (nz / size_t(ranks)) + std::min(size_t(rank), nz % size_t(ranks));
}

void printUsage(const char* name) {
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

bool parseNumber(const char* text, size_t& value) {
    if (!*text || *text == '-') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || *end || parsed > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(parsed);
    return true;
}

int run(int argc, char** argv, int worldSize) {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        const char* option = argv[i];
        if (!strcmp(option, "-h")) {
            if (!worldRank) printUsage(argv[0]);
            return 0;
        } else if (!strcmp(option, "-v")) validate = true;
        else if (!strcmp(option, "-r")) printResults = true;
        else if ((!strcmp(option, "-x") || !strcmp(option, "-y") ||
                  !strcmp(option, "-z") || !strcmp(option, "-i")) && i + 1 < argc) {
            size_t value = 0;
            const bool valid = parseNumber(argv[++i], value);
            if (!valid || (!strcmp(option, "-i") ? value > size_t(INT_MAX)
                                                  : (!strcmp(option, "-x") && value == 0))) {
                if (!worldRank) fprintf(stderr, "Invalid value for %s\n", option);
                return 1;
            }
            if (!strcmp(option, "-x")) nx = value;
            else if (!strcmp(option, "-y")) ny = value;
            else if (!strcmp(option, "-z")) nz = value;
            else iterations = static_cast<int>(value);
        } else {
            if (!worldRank) { printf("Unknown option: %s\n", option); printUsage(argv[0]); }
            return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx > limit / ny || nx * ny > limit / nz ||
        nz > std::numeric_limits<size_t>::max() - 2 || nx * ny > limit / (nz + 2) ||
        nx * ny > limit / 4) {
        if (!worldRank) fprintf(stderr, "Grid size overflows allocation limits\n");
        return 1;
    }
    const size_t plane = nx * ny, volume = plane * nz;
    const int ranks = static_cast<int>(std::min(nz, size_t(worldSize)));
    MPI_Comm active;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED,
                             worldRank, &active));
    int status = 0;
    if (active != MPI_COMM_NULL) {
        // One GPU per node-local active rank; respect CUDA_VISIBLE_DEVICES.
        MPI_Comm local;
        MPI_CHECK(MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, worldRank,
                                      MPI_INFO_NULL, &local));
        int localRank = 0, devices = 0;
        MPI_CHECK(MPI_Comm_rank(local, &localRank));
        CUDA(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fprintf(stderr, "Rank %d: a CUDA GPU is required\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        const int device = localRank % devices;
        CUDA(cudaSetDevice(device));
        cudaDeviceProp properties{};
        CUDA(cudaGetDeviceProperties(&properties, device));
        MPI_CHECK(MPI_Comm_free(&local));
        const size_t depth = slabDepth(nz, worldRank, ranks);
        const size_t start = slabStart(nz, worldRank, ranks);
        const size_t cells = plane * depth;
        int cudaAware = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        cudaAware = MPIX_Query_cuda_support() != 0;
#endif
        // All ranks must agree on the memory type used for MPI messages.
        int allCudaAware = 0;
        MPI_CHECK(MPI_Allreduce(&cudaAware, &allCudaAware, 1, MPI_INT, MPI_MIN, active));
        std::vector<double> host(cells);
        if (!worldRank) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Time steps: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Parallel execution: %d MPI ranks, CUDA, OpenMP\n", ranks);
            printf("Initializing concentration field...\n");
        }
        #pragma omp parallel for schedule(static) if(cells >= 32768)
        for (size_t i = 0; i < cells; ++i) {
            const size_t linear = start * plane + i;
            const double pseudo = (((linear + 1) * size_t(1299709)) % volume)
                                / static_cast<double>(volume);
            host[i] = -1.0 + 2.0 * pseudo;
        }
        {
            Solver solver(nx, ny, depth, worldRank, ranks, active,
                          properties.multiProcessorCount, allCudaAware != 0);
            CUDA(cudaMemcpyAsync(solver.cold + plane, host.data(), cells * sizeof(double),
                                 cudaMemcpyHostToDevice, solver.compute));
            CUDA(cudaStreamSynchronize(solver.compute));
            solver.prepare(iterations);
            if (!worldRank) printf("Running Cahn-Hilliard simulation...\n");
            MPI_CHECK(MPI_Barrier(active));
            const double begin = MPI_Wtime();
            solver.advance(iterations);
            CUDA(cudaStreamSynchronize(solver.compute));
            const double elapsed = MPI_Wtime() - begin;
            double seconds = 0;
            MPI_CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, active));
            if (!worldRank) {
                printf("Computation time: %.0f ms\n", seconds * 1000.0);
                const double mcups = seconds > 0 ? double(volume) * iterations / seconds / 1e6 : 0;
                printf("Performance: %.3f MCellUpdates/s\n", mcups);
            }
            if (printResults || validate)
                CUDA(cudaMemcpy(host.data(), solver.cold + plane, cells * sizeof(double),
                                cudaMemcpyDeviceToHost));
        }
        if (printResults) {
            // Gather only when requested, keeping global z/y/x order and the
            // original serial Kahan sum, hash and samples on rank zero.
            std::vector<double> result;
            if (!worldRank) {
                result.resize(volume);
                std::copy(host.begin(), host.end(), result.begin());
            }
            for (int source = 1; source < ranks; ++source) {
                const size_t count = plane * slabDepth(nz, source, ranks);
                const size_t offset = plane * slabStart(nz, source, ranks);
                for (size_t i = 0; i < count; i += size_t(INT_MAX)) {
                    const int chunk = static_cast<int>(std::min(count - i, size_t(INT_MAX)));
                    if (worldRank == source)
                        MPI_CHECK(MPI_Send(host.data() + i, chunk, MPI_DOUBLE, 0, 2, active));
                    else if (!worldRank)
                        MPI_CHECK(MPI_Recv(result.data() + offset + i, chunk, MPI_DOUBLE,
                                           source, 2, active, MPI_STATUS_IGNORE));
                }
            }
            if (!worldRank) print_results(result, "Concentration");
        }
        if (validate) {
            double localMin = std::numeric_limits<double>::infinity();
            double localMax = -std::numeric_limits<double>::infinity();
            int bad = 0;
            #pragma omp parallel for schedule(static) reduction(min:localMin) reduction(max:localMax) reduction(|:bad) if(cells >= 32768)
            for (size_t i = 0; i < cells; ++i) {
                const double value = host[i];
                bad |= !std::isfinite(value);
                localMin = std::min(localMin, value);
                localMax = std::max(localMax, value);
            }
            double minValue = 0, maxValue = 0;
            int anyBad = 0;
            MPI_CHECK(MPI_Reduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, 0, active));
            MPI_CHECK(MPI_Reduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, 0, active));
            MPI_CHECK(MPI_Reduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, 0, active));
            if (!worldRank) {
                printf("Validating result...\n");
                if (anyBad) printf("Validation failed: found NaN or Inf value\n");
                else {
                    printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
                    if (maxValue > 10.0 || minValue < -10.0)
                        printf("Validation failed: values out of expected range\n");
                }
                status = anyBad || maxValue > 10.0 || minValue < -10.0;
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
        MPI_CHECK(MPI_Comm_free(&active));
    }
    // Surplus ranks (nz < worldSize) participate in shutdown, without empty slabs.
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return status;
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    if (provided < MPI_THREAD_FUNNELED) {
        if (!worldRank) fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int worldSize = 0;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    int status = 1;
    try {
        status = run(argc, argv, worldSize);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Finalize());
    return status;
}
