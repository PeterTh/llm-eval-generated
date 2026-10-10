#include <mpi.h>
#if defined(OMPI_MAJOR_VERSION) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <cuda_runtime.h>
#include <omp.h>

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

#include "../common/results_output.hpp"

using Real = double;

static void mpiCheck(int error, const char* operation) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0, rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "Rank %d: %s: %.*s\n", rank, operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define MPI_CHECK(call) mpiCheck((call), #call)
#define CUDA_CHECK(call) cudaCheck((call), #call)

struct Options {
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, results = false;
};

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n"
                "  -x <num>     Grid size in X dimension (default: 128)\n"
                "  -y <num>     Grid size in Y dimension (default: same as X)\n"
                "  -z <num>     Grid size in Z dimension (default: same as X)\n"
                "  -i <num>     Number of iterations (default: 10)\n"
                "  -v           Enable validation\n"
                "  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

// Return 1 for help, -1 for invalid arguments, and 0 to run.
static int parseOptions(int argc, char** argv, Options& o, int rank) {
    for (int i = 1; i < argc; ++i) {
        const char* option = argv[i];
        if (!std::strcmp(option, "-h")) {
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
        if (!std::strcmp(option, "-v")) { o.validate = true; continue; }
        if (!std::strcmp(option, "-r")) { o.results = true; continue; }
        bool x = !std::strcmp(option, "-x"), y = !std::strcmp(option, "-y");
        bool z = !std::strcmp(option, "-z"), it = !std::strcmp(option, "-i");
        if (!(x || y || z || it) || i + 1 == argc) {
            if (rank == 0) { std::printf("Unknown or incomplete option: %s\n", option); printUsage(argv[0]); }
            return -1;
        }
        const char* value = argv[++i];
        char* end = nullptr;
        errno = 0;
        unsigned long long n = std::strtoull(value, &end, 10);
        if (errno || value[0] == '-' || end == value || *end ||
            n > std::numeric_limits<size_t>::max() || (x && n == 0) ||
            (it && n > INT_MAX)) {
            if (rank == 0) std::fprintf(stderr, "Invalid value for %s: %s\n", option, value);
            return -1;
        }
        if (x) o.nx = static_cast<size_t>(n);
        if (y) o.ny = static_cast<size_t>(n);
        if (z) o.nz = static_cast<size_t>(n);
        if (it) o.iterations = static_cast<int>(n);
    }
    if (!o.ny) o.ny = o.nx;
    if (!o.nz) o.nz = o.nx;
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (o.nx > limit / o.ny || o.nx * o.ny > limit / o.nz) {
        if (rank == 0) std::fprintf(stderr, "Grid size exceeds addressable memory.\n");
        return -1;
    }
    return 0;
}

struct Slab {
    size_t first, depth;
};

static Slab slabFor(size_t planes, size_t firstPlane, int ranks, int rank) {
    const size_t base = planes / ranks, remainder = planes % ranks;
    return {firstPlane + static_cast<size_t>(rank) * base +
                std::min(static_cast<size_t>(rank), remainder),
            base + (static_cast<size_t>(rank) < remainder)};
}

// Initialize both buffers once. Fixed boundaries never need copying during
// an iteration; only interior values are overwritten.
static void initializeGrid(std::vector<Real>& grid, size_t plane, size_t nz, Slab slab) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < grid.size(); ++i) {
        const size_t localZ = i / plane;
        if ((localZ == 0 && slab.first == 0) || slab.first + localZ > nz) {
            grid[i] = 0.0; // Unused ghost plane outside the global domain.
        } else {
            const size_t globalZ = slab.first + localZ - 1;
            grid[i] = static_cast<Real>((globalZ * plane + i % plane) % 19);
        }
    }
}

// Threads access contiguous X coordinates. Marching through Z reuses the
// bottom/center/top values in registers; XY neighbors use the device cache.
// The edge specialization updates both slab faces in a single launch.
template<int ZTile, bool Edges>
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t firstZ, size_t lastZ,
                                 size_t yOffset) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = yOffset + static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    if (x >= nx - 1 || y >= ny - 1) return;
    const size_t plane = nx * ny;
    const size_t begin = Edges ? (blockIdx.z == 0 ? firstZ : lastZ)
                              : firstZ + static_cast<size_t>(blockIdx.z) * ZTile;
    const size_t end = Edges ? begin + 1 : (begin + ZTile < lastZ + 1 ? begin + ZTile : lastZ + 1);
    size_t i = begin * plane + y * nx + x;
    Real bottom = input[i - plane], center = input[i];
    #pragma unroll
    for (int step = 0; step < ZTile; ++step) {
        if (begin + step >= end) break;
        const Real top = input[i + plane];
        // Preserve the original addition order and correctly rounded division,
        // including for result hashes. Do not enable fast math.
        const Real sum = center + input[i - 1] + input[i + 1] +
                         input[i - nx] + input[i + nx] + bottom + top;
        output[i] = __ddiv_rn(sum, 7.0);
        bottom = center;
        center = top;
        i += plane;
    }
}

static void launchStencil(const Real* input, Real* output, const Options& o,
                          size_t first, size_t last, bool edges, cudaStream_t stream) {
    if (first > last || o.nx <= 2 || o.ny <= 2 || o.nz <= 2) return;
    unsigned xThreads = 32;
    while (xThreads > 1 && xThreads / 2 >= o.nx - 2) xThreads /= 2;
    const dim3 threads(xThreads, 128 / xThreads);
    const size_t maxY = 65535ULL * threads.y;
    constexpr size_t maxZ = 65535 * 8ULL;
    const size_t xBlocks = (o.nx - 2 + threads.x - 1) / threads.x;
    if (xBlocks > static_cast<size_t>(INT_MAX)) {
        std::fprintf(stderr, "X dimension exceeds CUDA launch limits.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Split long Y/Z dimensions to respect CUDA grid limits.
    for (size_t y = 0; y < o.ny - 2; y += maxY) {
        for (size_t z = first; z <= last;) {
            const size_t end = edges ? last : z + std::min(last - z + 1, maxZ) - 1;
            const dim3 blocks(static_cast<unsigned>(xBlocks),
                              static_cast<unsigned>((std::min(o.ny - 2 - y, maxY) + threads.y - 1) / threads.y),
                              edges ? (first == last ? 1U : 2U)
                                    : static_cast<unsigned>((end - z + 8) / 8));
            if (edges) stencilIteration<1, true><<<blocks, threads, 0, stream>>>(input, output, o.nx, o.ny, first, last, y);
            else stencilIteration<8, false><<<blocks, threads, 0, stream>>>(input, output, o.nx, o.ny, z, end, y);
            CUDA_CHECK(cudaGetLastError());
            z = end + 1;
        }
    }
}

static bool cudaAwareMPI() {
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    return MPIX_Query_cuda_support() != 0;
#else
    // Without a reliable MPI CUDA capability query, use pinned host staging.
    // CUDA computation remains mandatory in both communication paths.
    return false;
#endif
}

// MPI counts are int in MPI-3. Chunk large planes without truncation.
static void postPlane(Real* data, size_t count, int peer, int tag, bool send,
                      MPI_Comm comm, std::vector<MPI_Request>& requests) {
    if (peer == MPI_PROC_NULL) return;
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
        MPI_Request request;
        if (send) MPI_CHECK(MPI_Isend(data + offset, chunk, MPI_DOUBLE, peer, tag, comm, &request));
        else MPI_CHECK(MPI_Irecv(data + offset, chunk, MPI_DOUBLE, peer, tag, comm, &request));
        requests.push_back(request);
        offset += chunk;
    }
}

static bool validateResult(const std::vector<Real>& grid, const Options& o,
                           Slab slab, MPI_Comm comm, int rank) {
    const size_t plane = o.nx * o.ny;
    Real minimum = std::numeric_limits<Real>::infinity();
    Real maximum = -std::numeric_limits<Real>::infinity();
    int bad = 0;
    #pragma omp parallel for schedule(static) reduction(min:minimum) reduction(max:maximum) reduction(|:bad)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real value = grid[i];
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        if (!std::isfinite(value)) bad |= 1;
        const size_t globalIndex = slab.first * plane + i;
        const size_t x = i % o.nx, y = (i / o.nx) % o.ny, z = slab.first + i / plane;
        if ((x == 0 || x + 1 == o.nx || y == 0 || y + 1 == o.ny || z == 0 || z + 1 == o.nz) &&
            value != static_cast<Real>(globalIndex % 19)) bad |= 2;
    }
    // Physical Z faces are fixed ghost planes, omitted from the decomposition.
    if (rank == 0 && o.nz > 2) {
        #pragma omp parallel for schedule(static) reduction(min:minimum) reduction(max:maximum)
        for (size_t i = 0; i < 2 * plane; ++i) {
            const size_t globalIndex = i < plane ? i : (o.nz - 2) * plane + i;
            const Real value = static_cast<Real>(globalIndex % 19);
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
    }
    Real globalMin, globalMax;
    int globalBad;
    MPI_CHECK(MPI_Allreduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm));
    MPI_CHECK(MPI_Allreduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm));
    MPI_CHECK(MPI_Allreduce(&bad, &globalBad, 1, MPI_INT, MPI_BOR, comm));
    if (rank == 0) {
        std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalBad & 1) std::printf("Validation failed: found NaN or Inf value\n");
        if (globalBad & 2) std::printf("Validation failed: boundary values changed\n");
        if (globalMax > 1e6 || globalMin < -1e6) std::printf("Validation failed: values out of expected range\n");
    }
    return !globalBad && globalMax <= 1e6 && globalMin >= -1e6;
}

static void printDistributedResults(const std::vector<Real>& local, const Options& o,
                                    size_t planes, size_t firstPlane, int ranks,
                                    int rank, MPI_Comm comm) {
    const size_t plane = o.nx * o.ny, total = plane * o.nz;
    // Stream slabs in global order. Rank zero needs only a bounded receive
    // buffer, even when the global grid is larger than one node's memory.
    // Keep exactly the serial Kahan summation and FNV hash order.
    constexpr size_t chunkSize = 1 << 20;
    if (rank != 0) {
        for (size_t offset = 0; offset < local.size();) {
            const int count = static_cast<int>(std::min(local.size() - offset, chunkSize));
            MPI_CHECK(MPI_Send(local.data() + offset, count, MPI_DOUBLE, 0, 9, comm));
            offset += count;
        }
        return;
    }
    Real sum = 0, correction = 0, minimum = 0, maximum = 0;
    const size_t indices[5] = {0, total / 4, total / 2, 3 * total / 4, total - 1};
    Real samples[5] = {};
    size_t index = 0;
    hash digest;
    const auto consume = [&](Real value) {
        const Real y = value - correction;
        const Real t = sum + y;
        correction = (t - sum) - y;
        sum = t;
        if (index == 0) minimum = maximum = value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        digest.add(value);
        for (int j = 0; j < 5; ++j) if (index == indices[j]) samples[j] = value;
        ++index;
    };
    if (o.nz > 2)
        for (size_t i = 0; i < plane; ++i) consume(static_cast<Real>(i % 19));
    for (const Real value : local) consume(value);
    std::vector<Real> buffer(ranks > 1 ? std::min(local.size(), chunkSize) : 0);
    for (int r = 1; r < ranks; ++r) {
        const size_t count = slabFor(planes, firstPlane, ranks, r).depth * plane;
        for (size_t offset = 0; offset < count;) {
            const int chunk = static_cast<int>(std::min(count - offset, chunkSize));
            MPI_CHECK(MPI_Recv(buffer.data(), chunk, MPI_DOUBLE, r, 9, comm, MPI_STATUS_IGNORE));
            for (int j = 0; j < chunk; ++j) consume(buffer[j]);
            offset += chunk;
        }
    }
    if (o.nz > 2) {
        const size_t begin = (o.nz - 1) * plane;
        for (size_t i = 0; i < plane; ++i) consume(static_cast<Real>((begin + i) % 19));
    }
    std::printf("=== RESULTS ===\nName: Grid\nElements: %zu\n", total);
    std::printf("Sum: %.17e\nMin: %.17e\nMax: %.17e\n", sum, minimum, maximum);
    for (int j = 0; j < 5; ++j) std::printf("Sample[%zu]: %.17e\n", indices[j], samples[j]);
    std::printf("Hash: %016" PRIx64 "\n=== END RESULTS ===\n", digest.get());
}

static int runBenchmark(const Options& o, MPI_Comm comm, int worldRanks) {
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));
    MPI_CHECK(MPI_Comm_size(comm, &ranks));
    MPI_Comm shared;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared));
    int localRank, localRanks;
    MPI_CHECK(MPI_Comm_rank(shared, &localRank));
    MPI_CHECK(MPI_Comm_size(shared, &localRanks));
    MPI_CHECK(MPI_Comm_free(&shared));
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        std::fprintf(stderr, "Rank %d: the MPI/OpenMP/CUDA benchmark requires a CUDA device.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    // Honor explicit OpenMP settings; otherwise avoid oversubscribing CPUs.
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / localRanks)));
    const size_t plane = o.nx * o.ny;
    const size_t planes = o.nz > 2 ? o.nz - 2 : o.nz;
    const size_t firstPlane = o.nz > 2 ? 1 : 0;
    const Slab slab = slabFor(planes, firstPlane, ranks, rank);
    const size_t maxPlanes = std::numeric_limits<size_t>::max() / sizeof(Real) / plane;
    if (maxPlanes < 2 || slab.depth > maxPlanes - 2) {
        std::fprintf(stderr, "Local grid exceeds addressable memory.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t localSize = (slab.depth + 2) * plane, bytes = localSize * sizeof(Real);
    const size_t planeBytes = plane * sizeof(Real);
    const bool hasInterior = o.nx > 2 && o.ny > 2 && o.nz > 2;
    const bool distributed = ranks > 1 && hasInterior;
    int aware = cudaAwareMPI() ? 1 : 0;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &aware, 1, MPI_INT, MPI_MIN, comm));
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                    o.nx, o.ny, o.nz, o.iterations, o.validate ? "enabled" : "disabled");
        std::printf("Parallelization: MPI (%d active / %d ranks), OpenMP, CUDA\n", ranks, worldRanks);
        if (distributed) std::printf("Halo exchange: %s\n", aware ? "CUDA-aware MPI" : "pinned host staging");
        std::printf("Initializing grid...\n");
    }
    Real *a = nullptr, *b = nullptr, *staging = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b), bytes));
    {
        std::vector<Real> initial(localSize);
        initializeGrid(initial, plane, o.nz, slab);
        CUDA_CHECK(cudaMemcpy(a, initial.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b, initial.data(), bytes, cudaMemcpyHostToDevice));
    }
    cudaStream_t compute, exchange;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
    if (distributed && !aware) CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&staging), 4 * planeBytes));
    std::vector<MPI_Request> requests;
    requests.reserve(4 * ((plane - 1) / INT_MAX + 1));

    // A bounded, even-length graph avoids per-step host launch overhead when
    // there are no halos to exchange. Capture/instantiation is outside timing.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    const int graphSteps = !distributed && hasInterior ? std::min(o.iterations / 2, 32) * 2 : 0;
    if (graphSteps) {
        CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
        for (int step = 0; step < graphSteps; ++step)
            launchStencil(step % 2 ? b : a, step % 2 ? a : b, o, 1, slab.depth, false, compute);
        CUDA_CHECK(cudaStreamEndCapture(compute, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(executable, compute));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(comm));
    if (rank == 0) std::printf("Running stencil computation...\n");
    const double start = MPI_Wtime();
    Real *input = a, *output = b;
    int iter = 0;
    if (graphSteps) {
        for (; o.iterations - iter >= graphSteps; iter += graphSteps)
            CUDA_CHECK(cudaGraphLaunch(executable, compute));
    }
    for (; iter < o.iterations && hasInterior; ++iter) {
        // Initial ghosts are already correct. Exchange each step's output for
        // the next step, and omit communication after the final update.
        if (!distributed || iter + 1 == o.iterations) {
            launchStencil(input, output, o, 1, slab.depth, false, compute);
        } else {
            requests.clear();
            Real* recvLower = aware ? output : staging + 2 * plane;
            Real* recvUpper = aware ? output + (slab.depth + 1) * plane : staging + 3 * plane;
            postPlane(recvLower, plane, lower, 1, false, comm, requests);
            postPlane(recvUpper, plane, upper, 0, false, comm, requests);
            launchStencil(input, output, o, 1, slab.depth, true, exchange);
            launchStencil(input, output, o, 2, slab.depth - 1, false, compute);
            Real* sendLower = aware ? output + plane : staging;
            Real* sendUpper = aware ? output + slab.depth * plane : staging + plane;
            if (!aware) {
                if (lower != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(sendLower, output + plane, planeBytes, cudaMemcpyDeviceToHost, exchange));
                if (upper != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(sendUpper, output + slab.depth * plane, planeBytes, cudaMemcpyDeviceToHost, exchange));
            }
            // MPI can consume the faces while the bulk kernel is still running.
            CUDA_CHECK(cudaStreamSynchronize(exchange));
            postPlane(sendLower, plane, lower, 0, true, comm, requests);
            postPlane(sendUpper, plane, upper, 1, true, comm, requests);
            MPI_CHECK(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
            if (!aware) {
                if (lower != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(output, recvLower, planeBytes, cudaMemcpyHostToDevice, exchange));
                if (upper != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(output + (slab.depth + 1) * plane, recvUpper, planeBytes, cudaMemcpyHostToDevice, exchange));
            }
            // Both streams must finish before the new input is communicated.
            CUDA_CHECK(cudaStreamSynchronize(exchange));
            CUDA_CHECK(cudaStreamSynchronize(compute));
        }
        std::swap(input, output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_CHECK(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (rank == 0) {
        const double updates = hasInterior ? static_cast<double>(o.nx - 2) *
            static_cast<double>(o.ny - 2) * static_cast<double>(o.nz - 2) * o.iterations : 0;
        std::printf("Computation time: %.3f ms\n", duration * 1000);
        std::printf("Performance: %.3f MCellUpdates/s\n", duration > 0 ? updates / duration / 1e6 : 0);
    }

    std::vector<Real> finalGrid;
    if (o.results || o.validate) {
        finalGrid.resize(slab.depth * plane);
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), input + plane, finalGrid.size() * sizeof(Real), cudaMemcpyDeviceToHost));
    }
    if (o.results) printDistributedResults(finalGrid, o, planes, firstPlane, ranks, rank, comm);
    bool valid = true;
    if (o.validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateResult(finalGrid, o, slab, comm, rank);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    if (executable) CUDA_CHECK(cudaGraphExecDestroy(executable));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    if (staging) CUDA_CHECK(cudaFreeHost(staging));
    CUDA_CHECK(cudaStreamDestroy(exchange));
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(a));
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI_THREAD_FUNNELED support is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    Options options;
    const int parse = parseOptions(argc, argv, options, rank);
    if (parse) { MPI_CHECK(MPI_Finalize()); return parse < 0 ? 1 : 0; }
    const size_t planes = options.nz > 2 ? options.nz - 2 : options.nz;
    const int active = static_cast<int>(std::min(planes, static_cast<size_t>(ranks)));
    MPI_Comm comm;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm));
    int status = 0;
    try {
        if (comm != MPI_COMM_NULL) {
            status = runBenchmark(options, comm, ranks);
            MPI_CHECK(MPI_Comm_free(&comm));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return status;
}
