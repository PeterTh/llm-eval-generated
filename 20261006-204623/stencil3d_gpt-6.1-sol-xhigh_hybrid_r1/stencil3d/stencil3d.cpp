#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

static void checkMPI(int error, const char* operation, int line) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0, rank = 0;
    MPI_Error_string(error, message, &length);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d, line %d: %s: %.*s\n", rank, line,
                 operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

static void checkCUDA(cudaError_t error, const char* operation, int line) {
    if (error == cudaSuccess) return;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d, line %d: %s: %s\n", rank, line,
                 operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define MPI_CHECK(operation) checkMPI((operation), #operation, __LINE__)
#define CUDA_CHECK(operation) checkCUDA((operation), #operation, __LINE__)

struct Options {
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
};

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
    std::printf("Launch one MPI rank per GPU; local ranks select visible GPUs in order.\n");
    std::printf("OMP_NUM_THREADS controls host initialization and validation.\n");
    std::printf("STENCIL3D_CUDA_AWARE_MPI=0/1 overrides automatic MPI CUDA detection.\n");
}

static bool parseNumber(const char* text, size_t& value) {
    if (!*text || *text == '-') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || *end || parsed > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(parsed);
    return true;
}

// Balance the updatable z planes; only the end ranks also own fixed z faces.
struct Slab { size_t first, count; };

static Slab partition(size_t nz, int rank, int ranks) {
    if (ranks == 1) return {0, nz};
    const size_t interior = nz - 2;
    const size_t base = interior / static_cast<size_t>(ranks);
    const size_t remainder = interior % static_cast<size_t>(ranks);
    Slab slab{1 + static_cast<size_t>(rank) * base +
                  std::min(static_cast<size_t>(rank), remainder),
              base + (static_cast<size_t>(rank) < remainder)};
    if (rank == 0) { slab.first = 0; ++slab.count; }
    if (rank == ranks - 1) ++slab.count;
    return slab;
}

// Contiguous x accesses, with a rolling register window along z. The fixed
// faces are initialized in both buffers, so no boundary-copy pass is needed.
// A flattened, grid-stride tile index avoids CUDA's y/z grid-size limits.
template<int Z_TILE>
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t plane,
                              size_t tilesX, size_t tilesXY, size_t tiles,
                              size_t zBegin, size_t zEnd, size_t secondFace) {
    for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const size_t xy = tile % tilesXY;
        const size_t x = 1 + (xy % tilesX) * blockDim.x + threadIdx.x;
        const size_t y = 1 + (xy / tilesX) * blockDim.y + threadIdx.y;
        if (x >= nx - 1 || y >= ny - 1) continue;
        const size_t z = secondFace ? (tile / tilesXY == 0 ? zBegin : secondFace)
                                    : zBegin + (tile / tilesXY) * Z_TILE;
        const size_t limit = secondFace ? z + 1 : min(z + Z_TILE, zEnd);
        size_t index = z * plane + y * nx + x;
        Real bottom = input[index - plane];
        Real center = input[index];
        #pragma unroll
        for (int k = 0; k < Z_TILE; ++k) {
            if (z + k >= limit) break;
            const Real top = input[index + plane];
            // Preserve the original addition order and correctly rounded
            // division, including the bytewise external-validation hash.
            output[index] = __ddiv_rn(center + input[index - 1] + input[index + 1]
                                  + input[index - nx] + input[index + nx]
                                  + bottom + top, 7.0);
            bottom = center;
            center = top;
            index += plane;
        }
    }
}

static void launchStencil(const Real* input, Real* output, const Options& options,
                          size_t begin, size_t end, cudaStream_t stream,
                          bool faces = false, size_t secondFace = 0) {
    if (options.nx <= 2 || options.ny <= 2 || begin >= end) return;
    const size_t tilesX = (options.nx - 2 + 31) / 32;
    const size_t tilesXY = tilesX * ((options.ny - 2 + 7) / 8);
    const size_t tilesZ = faces ? (secondFace ? 2 : 1) : (end - begin + 7) / 8;
    const size_t tiles = tilesXY * tilesZ;
    const unsigned blocks = static_cast<unsigned>(std::min(tiles, size_t{65535}));
    const dim3 threads(32, 8);
    if (faces) {
        stencilKernel<1><<<blocks, threads, 0, stream>>>(input, output,
            options.nx, options.ny, options.nx * options.ny, tilesX, tilesXY,
            tiles, begin, end, secondFace);
    } else {
        stencilKernel<8><<<blocks, threads, 0, stream>>>(input, output,
            options.nx, options.ny, options.nx * options.ny, tilesX, tilesXY,
            tiles, begin, end, 0);
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

static bool cudaAwareMPI(MPI_Comm comm) {
    int supported = 0;
    #if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    supported = MPIX_Query_cuda_support();
    #endif
    // Other MPI implementations have no common runtime query. An explicit
    // override permits their device-buffer path without an extra dependency.
    if (const char* setting = std::getenv("STENCIL3D_CUDA_AWARE_MPI")) {
        if (std::strcmp(setting, "0") == 0) supported = 0;
        else if (std::strcmp(setting, "1") == 0) supported = 1;
    }
    int allSupported = 0;
    MPI_CHECK(MPI_Allreduce(&supported, &allSupported, 1, MPI_INT, MPI_MIN, comm));
    return allSupported != 0;
}

// Segment messages to support planes larger than MPI's int count limit.
static void postPlane(Real* buffer, size_t count, int peer, int tag,
                      bool receive, MPI_Comm comm, std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, size_t{INT_MAX}));
        MPI_Request request;
        if (receive) {
            MPI_CHECK(MPI_Irecv(buffer + offset, chunk, MPI_DOUBLE, peer, tag, comm, &request));
        } else {
            MPI_CHECK(MPI_Isend(buffer + offset, chunk, MPI_DOUBLE, peer, tag, comm, &request));
        }
        requests.push_back(request);
        offset += static_cast<size_t>(chunk);
    }
}

static void waitRequests(std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < requests.size();) {
        const int chunk = static_cast<int>(std::min(requests.size() - offset, size_t{INT_MAX}));
        MPI_CHECK(MPI_Waitall(chunk, requests.data() + offset, MPI_STATUSES_IGNORE));
        offset += static_cast<size_t>(chunk);
    }
}

static void printGlobalResults(const Real* owned, const Options& options,
                               Slab slab, int rank, int ranks, MPI_Comm comm) {
    const size_t plane = options.nx * options.ny;
    const size_t total = plane * options.nz;
    std::vector<Real> grid;
    if (rank == 0) grid.resize(total);
    if (total <= INT_MAX) {
        std::vector<int> counts(ranks), displacements(ranks);
        for (int r = 0; r < ranks; ++r) {
            const Slab other = partition(options.nz, r, ranks);
            counts[r] = static_cast<int>(other.count * plane);
            displacements[r] = static_cast<int>(other.first * plane);
        }
        MPI_CHECK(MPI_Gatherv(owned, static_cast<int>(slab.count * plane), MPI_DOUBLE,
                              grid.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                              0, comm));
    } else if (rank == 0) {
        std::copy_n(owned, slab.count * plane, grid.data());
        for (int r = 1; r < ranks; ++r) {
            const Slab other = partition(options.nz, r, ranks);
            for (size_t offset = 0; offset < other.count * plane;) {
                const int chunk = static_cast<int>(std::min(other.count * plane - offset,
                                                           size_t{INT_MAX}));
                MPI_CHECK(MPI_Recv(grid.data() + other.first * plane + offset, chunk,
                                   MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE));
                offset += static_cast<size_t>(chunk);
            }
        }
    } else {
        for (size_t offset = 0; offset < slab.count * plane;) {
            const int chunk = static_cast<int>(std::min(slab.count * plane - offset,
                                                       size_t{INT_MAX}));
            MPI_CHECK(MPI_Send(owned + offset, chunk, MPI_DOUBLE, 0, 2, comm));
            offset += static_cast<size_t>(chunk);
        }
    }
    if (rank == 0) print_results(grid, "Grid");
}

static int runBenchmark(const Options& options, MPI_Comm comm) {
    int rank = 0, ranks = 1;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));
    MPI_CHECK(MPI_Comm_size(comm, &ranks));
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0, devices = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        std::fprintf(stderr, "Rank %d: a CUDA GPU is required.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_CHECK(MPI_Comm_free(&nodeComm));
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::min(8, omp_get_max_threads()));

    const Slab slab = partition(options.nz, rank, ranks);
    const size_t plane = options.nx * options.ny;
    const size_t cells = (slab.count + 2) * plane;
    const size_t bytes = cells * sizeof(Real);
    const size_t planeBytes = plane * sizeof(Real);
    const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank == ranks - 1 ? MPI_PROC_NULL : rank + 1;
    const size_t begin = slab.first == 0 ? 2 : 1;
    const size_t end = slab.count + 1 - (slab.first + slab.count == options.nz);
    const bool exchange = ranks > 1;
    const bool direct = exchange && cudaAwareMPI(comm);

    if (rank == 0) {
        std::printf("MPI ranks: %d; OpenMP threads per rank: %d\n", ranks, omp_get_max_threads());
        if (exchange) std::printf("Halo exchange: %s\n", direct ? "CUDA-aware MPI" : "pinned host staging");
        std::printf("Initializing grid...\n");
    }
    std::unique_ptr<Real[]> host(new Real[cells]);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells; ++i) {
        const size_t localZ = i / plane;
        if ((localZ == 0 && slab.first == 0) ||
            (localZ == slab.count + 1 && slab.first + slab.count == options.nz)) {
            host[i] = 0.0; // Unused halos outside the physical domain.
        } else {
            const size_t globalZ = localZ == 0 ? slab.first - 1 : slab.first + localZ - 1;
            host[i] = static_cast<Real>((globalZ * plane + i % plane) % 19);
        }
    }
    Real *first = nullptr, *second = nullptr, *staging = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&first), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&second), bytes));
    CUDA_CHECK(cudaMemcpy(first, host.get(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(second, first, bytes, cudaMemcpyDeviceToDevice));
    if (!options.validate && !options.printResults) host.reset();
    if (exchange && !direct) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&staging), 4 * planeBytes));
    }

    cudaStream_t compute, transfers;
    cudaEvent_t boundariesReady, interiorsReady[2];
    int leastPriority = 0, greatestPriority = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithPriority(&transfers, cudaStreamNonBlocking, greatestPriority));
    CUDA_CHECK(cudaEventCreateWithFlags(&boundariesReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&interiorsReady[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&interiorsReady[1], cudaEventDisableTiming));
    std::vector<MPI_Request> requests;
    requests.reserve(4 * ((plane - 1) / INT_MAX + 1));

    // Warm the stencil kernel without changing the initial input grid.
    if (options.iterations > 0) launchStencil(first, second, options, begin, end, compute);
    CUDA_CHECK(cudaStreamSynchronize(compute));

    // When no communication is necessary, replay even-length CUDA graphs.
    // Both pointers return to their original roles after each graph replay.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    const int graphSteps = !exchange && begin < end && options.nx > 2 && options.ny > 2
                         ? std::min(options.iterations / 2 * 2, 64) : 0;
    if (graphSteps) {
        CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < graphSteps; ++i) {
            launchStencil(i % 2 ? second : first, i % 2 ? first : second,
                          options, begin, end, compute);
        }
        CUDA_CHECK(cudaStreamEndCapture(compute, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(executable, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }

    if (rank == 0) std::printf("Running stencil computation...\n");
    MPI_CHECK(MPI_Barrier(comm));
    const double start = MPI_Wtime();
    Real* input = first;
    Real* output = second;
    int iteration = begin >= end || options.nx <= 2 || options.ny <= 2
                  ? options.iterations : 0;
    if (graphSteps) {
        for (; iteration <= options.iterations - graphSteps; iteration += graphSteps)
            CUDA_CHECK(cudaGraphLaunch(executable, compute));
    }
    for (; iteration < options.iterations; ++iteration) {
        if (!exchange) {
            launchStencil(input, output, options, begin, end, compute);
        } else {
            // The interior never reads incoming halos and executes while MPI
            // progresses. Only the one or two interface planes wait for halos.
            launchStencil(input, output, options,
                          begin + (previous != MPI_PROC_NULL),
                          end - (next != MPI_PROC_NULL), compute);
            CUDA_CHECK(cudaEventRecord(interiorsReady[iteration % 2], compute));
            requests.clear();
            Real* recvPrevious = direct ? input : staging + 2 * plane;
            Real* recvNext = direct ? input + (slab.count + 1) * plane : staging + 3 * plane;
            if (previous != MPI_PROC_NULL) postPlane(recvPrevious, plane, previous, 1, true, comm, requests);
            if (next != MPI_PROC_NULL) postPlane(recvNext, plane, next, 0, true, comm, requests);
            if (!direct) {
                if (previous != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(staging, input + plane, planeBytes,
                                               cudaMemcpyDeviceToHost, transfers));
                if (next != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(staging + plane, input + slab.count * plane,
                                               planeBytes, cudaMemcpyDeviceToHost, transfers));
                CUDA_CHECK(cudaStreamSynchronize(transfers));
            }
            if (previous != MPI_PROC_NULL)
                postPlane(direct ? input + plane : staging, plane, previous, 0, false, comm, requests);
            if (next != MPI_PROC_NULL)
                postPlane(direct ? input + slab.count * plane : staging + plane,
                          plane, next, 1, false, comm, requests);
            waitRequests(requests);
            if (!direct) {
                if (previous != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(input, recvPrevious, planeBytes,
                                               cudaMemcpyHostToDevice, transfers));
                if (next != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(input + (slab.count + 1) * plane, recvNext,
                                               planeBytes, cudaMemcpyHostToDevice, transfers));
            }
            // A face reads the adjacent owned plane from the previous step.
            // Wait for that step's interior, rather than the current interior,
            // so the face kernel and the next exchange can run ahead of it.
            if (iteration > 0)
                CUDA_CHECK(cudaStreamWaitEvent(transfers, interiorsReady[(iteration - 1) % 2], 0));
            const size_t face = previous != MPI_PROC_NULL ? 1 : slab.count;
            const size_t otherFace = previous != MPI_PROC_NULL && next != MPI_PROC_NULL
                                  && slab.count > 1 ? slab.count : 0;
            launchStencil(input, output, options, face, face + 1, transfers, true, otherFace);
            CUDA_CHECK(cudaEventRecord(boundariesReady, transfers));
            // Input faces for the next exchange must be complete before MPI
            // reads them. The interior remains queued in its own stream;
            // double buffering and stream order protect its dependencies.
            CUDA_CHECK(cudaEventSynchronize(boundariesReady));
        }
        std::swap(input, output);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (rank == 0) {
        const double updates = static_cast<double>(options.nx > 2 ? options.nx - 2 : 0)
                             * static_cast<double>(options.ny > 2 ? options.ny - 2 : 0)
                             * static_cast<double>(options.nz > 2 ? options.nz - 2 : 0)
                             * options.iterations;
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0.0);
    }

    if (options.printResults || options.validate)
        CUDA_CHECK(cudaMemcpy(host.get() + plane, input + plane,
                              slab.count * planeBytes, cudaMemcpyDeviceToHost));
    if (options.printResults)
        printGlobalResults(host.get() + plane, options, slab, rank, ranks, comm);
    int invalid = 0;
    if (options.validate) {
        if (rank == 0) std::printf("Validating result...\n");
        Real minimum = std::numeric_limits<Real>::infinity();
        Real maximum = -std::numeric_limits<Real>::infinity();
        #pragma omp parallel for schedule(static) reduction(min:minimum) reduction(max:maximum) reduction(|:invalid)
        for (size_t i = plane; i < (slab.count + 1) * plane; ++i) {
            const Real value = host[i];
            invalid |= !std::isfinite(value);
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        Real globalMinimum, globalMaximum;
        int globalInvalid;
        MPI_CHECK(MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_MAX, comm));
        MPI_CHECK(MPI_Allreduce(&minimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN, comm));
        MPI_CHECK(MPI_Allreduce(&maximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX, comm));
        invalid = globalInvalid || globalMinimum < -1e6 || globalMaximum > 1e6;
        if (rank == 0) {
            if (globalInvalid) std::printf("Validation failed: found NaN or Inf value\n");
            else std::printf("Value range: [%.6f, %.6f]\n", globalMinimum, globalMaximum);
            if (!globalInvalid && invalid) std::printf("Validation failed: values out of expected range\n");
            std::printf("Validation: %s\n", invalid ? "FAILED" : "PASSED");
        }
    }

    if (executable) CUDA_CHECK(cudaGraphExecDestroy(executable));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(interiorsReady[1]));
    CUDA_CHECK(cudaEventDestroy(interiorsReady[0]));
    CUDA_CHECK(cudaEventDestroy(boundariesReady));
    CUDA_CHECK(cudaStreamDestroy(transfers));
    CUDA_CHECK(cudaStreamDestroy(compute));
    if (staging) CUDA_CHECK(cudaFreeHost(staging));
    CUDA_CHECK(cudaFree(second));
    CUDA_CHECK(cudaFree(first));
    return invalid;
}

int main(int argc, char** argv) {
    int provided = 0;
    const int init = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (init != MPI_SUCCESS) return 1;
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank = 0, worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI_THREAD_FUNNELED support is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    Options options;
    bool badArguments = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-v") == 0) options.validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) options.printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { help = true; break; }
        else if ((std::strcmp(argv[i], "-x") == 0 || std::strcmp(argv[i], "-y") == 0 ||
                  std::strcmp(argv[i], "-z") == 0 || std::strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            const char option = argv[i][1];
            size_t value = 0;
            if (!parseNumber(argv[++i], value)) { badArguments = true; break; }
            if (option == 'x') options.nx = value;
            else if (option == 'y') options.ny = value;
            else if (option == 'z') options.nz = value;
            else if (value <= INT_MAX) options.iterations = static_cast<int>(value);
            else { badArguments = true; break; }
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            badArguments = true;
            break;
        }
    }
    if (options.ny == 0) options.ny = options.nx;
    if (options.nz == 0) options.nz = options.nx;
    // Check global and halo-extended local allocation arithmetic before use.
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (!options.nx || !options.ny || !options.nz || options.nx > limit / options.ny)
        badArguments = true;
    else {
        const size_t plane = options.nx * options.ny;
        if (options.nz > limit / plane || limit / plane < 4 ||
            options.nz > limit / plane - 2) badArguments = true;
    }
    if (help || badArguments) {
        if (rank == 0) {
            if (badArguments && !help) std::fprintf(stderr, "Invalid arguments or grid size.\n");
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return help ? 0 : 1;
    }

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", options.nx, options.ny, options.nz);
        std::printf("Iterations: %d\n", options.iterations);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
    }
    // Excess ranks participate in world completion, but never own empty slabs.
    const size_t usefulRanks = options.nx > 2 && options.ny > 2 && options.nz > 2
                             ? options.nz - 2 : 1;
    const int activeRanks = static_cast<int>(std::min(usefulRanks, static_cast<size_t>(worldSize)));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &comm));
    int status = 0;
    try {
        if (comm != MPI_COMM_NULL) status = runBenchmark(options, comm);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    int globalStatus = 0;
    MPI_CHECK(MPI_Allreduce(&status, &globalStatus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    if (comm != MPI_COMM_NULL) MPI_CHECK(MPI_Comm_free(&comm));
    MPI_CHECK(MPI_Finalize());
    return globalStatus;
}
