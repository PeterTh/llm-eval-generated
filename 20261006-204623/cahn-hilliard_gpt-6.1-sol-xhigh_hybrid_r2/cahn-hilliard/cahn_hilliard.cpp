#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <mpi.h>
#ifdef CH_MPI_CUDA_QUERY
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

[[noreturn]] void fail(const char* message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        fail("CUDA operation failed");
    }
}

void checkMpi(int error, const char* operation) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        std::fprintf(stderr, "%s: %.*s\n", operation, length, message);
        fail("MPI operation failed");
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)
#define MPI_CHECK(operation) checkMpi((operation), #operation)

struct MpiSession {
    MpiSession(int& argc, char**& argv) {
        int provided = MPI_THREAD_SINGLE;
        MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
        if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
        MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    }
    ~MpiSession() { MPI_Finalize(); }
};

struct DeviceBuffer {
    double* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(double)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct PinnedBuffer {
    double* data = nullptr;
    explicit PinnedBuffer(size_t count) {
        if (count) CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), count * sizeof(double)));
    }
    ~PinnedBuffer() { if (data) cudaFreeHost(data); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

struct Stream {
    cudaStream_t handle = nullptr;
    explicit Stream(int priority = 0) {
        CUDA_CHECK(cudaStreamCreateWithPriority(&handle, cudaStreamNonBlocking, priority));
    }
    ~Stream() { cudaStreamDestroy(handle); }
};

struct Event {
    cudaEvent_t handle = nullptr;
    Event() { CUDA_CHECK(cudaEventCreateWithFlags(&handle, cudaEventDisableTiming)); }
    ~Event() { cudaEventDestroy(handle); }
};

// The longest dimension becomes the slab direction. The other two dimensions
// form contiguous GPU planes. Concentration has two ghost planes at either end.
struct Domain {
    size_t nx, ny, nz, globalNz, first;
    int axis;
};

size_t slabSize(size_t extent, int ranks, int rank) {
    return extent / ranks + (static_cast<size_t>(rank) < extent % ranks);
}

size_t slabStart(size_t extent, int ranks, int rank) {
    return (extent / ranks) * rank + std::min(static_cast<size_t>(rank), extent % ranks);
}

int slabOwner(size_t plane, size_t extent, int ranks) {
    const size_t base = extent / ranks;
    const size_t extra = extent % ranks;
    const size_t cutoff = (base + 1) * extra;
    return static_cast<int>(plane < cutoff ? plane / (base + 1) :
                            extra + (plane - cutoff) / base);
}

size_t originalIndex(size_t x, size_t y, size_t z, int axis, size_t nx, size_t ny) {
    if (axis == 0) return (y * ny + x) * nx + z;
    if (axis == 1) return (y * ny + z) * nx + x;
    return (z * ny + y) * nx + x;
}

int hostThreads(size_t cells) {
    const size_t useful = std::max<size_t>(1, std::min<size_t>(32, cells / 32768));
    return std::min(omp_get_max_threads(), static_cast<int>(useful));
}

template<int Axis, bool Potential>
__device__ __forceinline__ void evaluateCell(
        const double* __restrict__ field, const double* __restrict__ cold,
        double* __restrict__ output, size_t i, size_t xm, size_t xp,
        size_t ym, size_t yp, double below, double center, double above) {
    const double a = field[xp] + field[xm] - 2.0 * center;
    const double b = field[yp] + field[ym] - 2.0 * center;
    const double c = above + below - 2.0 * center;
    // Preserve the original X + Y + Z evaluation order after axis permutation.
    double laplacian;
    if constexpr (Axis == 0) laplacian = (c + a) + b;
    else if constexpr (Axis == 1) laplacian = (a + c) + b;
    else laplacian = (a + b) + c;

    if constexpr (Potential) {
        constexpr double eAA = -(2.0 / 9.0);
        constexpr double eBB = -(2.0 / 9.0);
        constexpr double eAB = 2.0 / 9.0;
        output[i] = 4.5 * ((center + 1.0) * eAA + (center - 1.0) * eBB -
                          2.0 * center * eAB) +
                    3.0 * center + center * center * center - 0.5 * laplacian;
    } else {
        output[i] = cold[i] + 0.01 * laplacian;
    }
}

// Threads traverse short Z columns, reusing three consecutive plane values in
// registers. X/Y accesses are coalesced and shared by the GPU read cache.
template<int Axis, bool Potential>
__global__ void interiorStencil(const double* __restrict__ field,
        const double* __restrict__ cold, double* __restrict__ output,
        Domain d, size_t first, size_t count) {
    const size_t plane = d.nx * d.ny;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t xy = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         xy < plane; xy += stride) {
        const size_t x = xy % d.nx;
        const size_t y = xy / d.nx;
        const size_t xm = x ? xy - 1 : xy;
        const size_t xp = x + 1 < d.nx ? xy + 1 : xy;
        const size_t ym = y ? xy - d.nx : xy;
        const size_t yp = y + 1 < d.ny ? xy + d.nx : xy;
        for (size_t base = static_cast<size_t>(blockIdx.y) * 4; base < count;
             base += static_cast<size_t>(gridDim.y) * 4) {
            size_t z = first + base;
            size_t gz = d.first + z - 2;
            size_t offset = z * plane;
            double center = field[offset + xy];
            double below = gz ? field[offset + xy - plane] : center;
            double above = gz + 1 < d.globalNz ? field[offset + xy + plane] : center;
            const size_t end = base + (count - base < 4 ? count - base : 4);
            for (size_t q = base; q < end; ++q) {
                evaluateCell<Axis, Potential>(field, cold, output, offset + xy,
                    offset + xm, offset + xp, offset + ym, offset + yp,
                    below, center, above);
                if (q + 1 < end) {
                    offset += plane;
                    ++gz;
                    below = center;
                    center = above;
                    above = gz + 1 < d.globalNz ? field[offset + xy + plane] : center;
                }
            }
        }
    }
}

// Only the four edge planes depend on communication. Potential also covers the
// two adjacent ghost planes, eliminating a second exchange of chemical potential.
template<int Axis, bool Potential>
__global__ void boundaryStencil(const double* __restrict__ field,
        const double* __restrict__ cold, double* __restrict__ output, Domain d) {
    const size_t q = blockIdx.y;
    size_t z;
    if constexpr (Potential) {
        z = d.nz == 1 ? q + 1 : (q < 2 ? q + 1 : d.nz + q - 1);
    } else {
        z = d.nz <= 4 ? q + 2 : (q < 2 ? q + 2 : d.nz + q - 2);
    }
    const int64_t gz = static_cast<int64_t>(d.first) + static_cast<int64_t>(z) - 2;
    if (gz < 0 || static_cast<size_t>(gz) >= d.globalNz) return;
    const size_t plane = d.nx * d.ny;
    const size_t offset = z * plane;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t xy = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         xy < plane; xy += stride) {
        const size_t x = xy % d.nx;
        const size_t y = xy / d.nx;
        const double center = field[offset + xy];
        const double below = gz ? field[offset + xy - plane] : center;
        const double above = static_cast<size_t>(gz) + 1 < d.globalNz ?
                             field[offset + xy + plane] : center;
        evaluateCell<Axis, Potential>(field, cold, output, offset + xy,
            offset + (x ? xy - 1 : xy),
            offset + (x + 1 < d.nx ? xy + 1 : xy),
            offset + (y ? xy - d.nx : xy),
            offset + (y + 1 < d.ny ? xy + d.nx : xy),
            below, center, above);
    }
}

template<int Axis, bool Potential, bool Boundary>
void launchStencil(const double* field, const double* cold, double* output,
                   Domain d, size_t first, size_t count, cudaStream_t stream) {
    if (!count) return;
    constexpr unsigned threads = 256;
    const size_t plane = d.nx * d.ny;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>(65535, (plane + threads - 1) / threads));
    if constexpr (Boundary) {
        boundaryStencil<Axis, Potential><<<dim3(blocks, static_cast<unsigned>(count)), threads, 0, stream>>>(
            field, cold, output, d);
    } else {
        const unsigned columns = static_cast<unsigned>(std::min<size_t>(65535, (count + 3) / 4));
        interiorStencil<Axis, Potential><<<dim3(blocks, columns), threads, 0, stream>>>(
            field, cold, output, d, first, count);
    }
    CUDA_CHECK(cudaGetLastError());
}

template<bool Potential, bool Boundary = false>
void stencil(const double* field, const double* cold, double* output,
             Domain d, size_t first, size_t count, cudaStream_t stream) {
    if (d.axis == 0) launchStencil<0, Potential, Boundary>(field, cold, output, d, first, count, stream);
    else if (d.axis == 1) launchStencil<1, Potential, Boundary>(field, cold, output, d, first, count, stream);
    else launchStencil<2, Potential, Boundary>(field, cold, output, d, first, count, stream);
}

template<class Function>
cudaGraphExec_t captureGraph(cudaStream_t stream, Function function) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    function();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphUpload(executable, stream));
    return executable;
}

struct Transfer {
    int peer, tag;
    size_t z, planes;
};

struct HaloExchange {
    std::vector<Transfer> sends, receives;
    std::vector<MPI_Request> requests;
    size_t plane;
    MPI_Comm comm;
    bool deviceMpi;
    PinnedBuffer sendBuffer, receiveBuffer;

    static std::vector<Transfer> receivePlan(Domain d, int ranks, int rank) {
        std::vector<Transfer> plan;
        const int64_t positions[4] = {
            static_cast<int64_t>(d.first) - 2, static_cast<int64_t>(d.first) - 1,
            static_cast<int64_t>(d.first + d.nz), static_cast<int64_t>(d.first + d.nz) + 1
        };
        const size_t offsets[4] = {0, 1, d.nz + 2, d.nz + 3};
        for (int slot = 0; slot < 4; ++slot) {
            if (positions[slot] < 0 || static_cast<size_t>(positions[slot]) >= d.globalNz) continue;
            const int owner = slabOwner(static_cast<size_t>(positions[slot]), d.globalNz, ranks);
            if (owner == rank) continue;
            // Normal slabs exchange two contiguous planes in a single message.
            // Thin slabs retain separate transfers to the actual plane owners.
            if (!plan.empty() && plan.back().peer == owner &&
                plan.back().tag + static_cast<int>(plan.back().planes) == slot &&
                plan.back().z + plan.back().planes == offsets[slot]) {
                ++plan.back().planes;
            } else {
                plan.push_back({owner, slot, offsets[slot], 1});
            }
        }
        return plan;
    }

    static std::vector<Transfer> sendPlan(Domain d, int ranks, int rank) {
        std::vector<Transfer> plan;
        // With nonempty slabs, a radius-two halo can reach at most two ranks.
        for (int peer = std::max(0, rank - 2); peer <= std::min(ranks - 1, rank + 2); ++peer) {
            if (peer == rank) continue;
            Domain other = d;
            other.first = slabStart(d.globalNz, ranks, peer);
            other.nz = slabSize(d.globalNz, ranks, peer);
            for (const Transfer& receive : receivePlan(other, ranks, peer)) {
                if (receive.peer != rank) continue;
                const int64_t global = static_cast<int64_t>(other.first) +
                                       static_cast<int64_t>(receive.z) - 2;
                plan.push_back({peer, receive.tag, static_cast<size_t>(global) - d.first + 2,
                                receive.planes});
            }
        }
        return plan;
    }

    static size_t totalPlanes(const std::vector<Transfer>& plan) {
        size_t count = 0;
        for (const Transfer& transfer : plan) count += transfer.planes;
        return count;
    }

    HaloExchange(Domain d, int ranks, int rank, MPI_Comm communicator, bool direct) :
        sends(sendPlan(d, ranks, rank)), receives(receivePlan(d, ranks, rank)),
        plane(d.nx * d.ny), comm(communicator), deviceMpi(direct),
        sendBuffer(direct ? 0 : totalPlanes(sends) * plane),
        receiveBuffer(direct ? 0 : totalPlanes(receives) * plane) {
        requests.reserve((totalPlanes(sends) + totalPlanes(receives)) * ((plane - 1) / INT_MAX + 1));
    }

    void postReceives(double* field) {
        requests.clear();
        size_t staged = 0;
        for (size_t entry = 0; entry < receives.size(); ++entry) {
            const Transfer& transfer = receives[entry];
            const size_t cells = transfer.planes * plane;
            double* buffer = deviceMpi ? field + transfer.z * plane : receiveBuffer.data + staged;
            for (size_t offset = 0; offset < cells; ) {
                const int count = static_cast<int>(std::min<size_t>(INT_MAX, cells - offset));
                requests.push_back(MPI_REQUEST_NULL);
                MPI_CHECK(MPI_Irecv(buffer + offset, count, MPI_DOUBLE, transfer.peer,
                                    transfer.tag, comm, &requests.back()));
                offset += count;
            }
            staged += cells;
        }
    }

    void stageSends(const double* field, cudaStream_t stream) {
        if (!deviceMpi) {
            size_t staged = 0;
            for (size_t entry = 0; entry < sends.size(); ++entry) {
                const size_t cells = sends[entry].planes * plane;
                CUDA_CHECK(cudaMemcpyAsync(sendBuffer.data + staged,
                    field + sends[entry].z * plane, cells * sizeof(double),
                    cudaMemcpyDeviceToHost, stream));
                staged += cells;
            }
        }
    }

    void postSends(const double* field) {
        size_t staged = 0;
        for (size_t entry = 0; entry < sends.size(); ++entry) {
            const Transfer& transfer = sends[entry];
            const size_t cells = transfer.planes * plane;
            const double* buffer = deviceMpi ? field + transfer.z * plane : sendBuffer.data + staged;
            for (size_t offset = 0; offset < cells; ) {
                const int count = static_cast<int>(std::min<size_t>(INT_MAX, cells - offset));
                requests.push_back(MPI_REQUEST_NULL);
                MPI_CHECK(MPI_Isend(buffer + offset, count, MPI_DOUBLE, transfer.peer,
                                    transfer.tag, comm, &requests.back()));
                offset += count;
            }
            staged += cells;
        }
    }

    void finish(double* field, cudaStream_t stream) {
        for (size_t offset = 0; offset < requests.size(); ) {
            const int count = static_cast<int>(std::min<size_t>(INT_MAX, requests.size() - offset));
            MPI_CHECK(MPI_Waitall(count, requests.data() + offset, MPI_STATUSES_IGNORE));
            offset += count;
        }
        if (!deviceMpi) {
            size_t staged = 0;
            for (size_t entry = 0; entry < receives.size(); ++entry) {
                const size_t cells = receives[entry].planes * plane;
                CUDA_CHECK(cudaMemcpyAsync(field + receives[entry].z * plane,
                    receiveBuffer.data + staged, cells * sizeof(double),
                    cudaMemcpyHostToDevice, stream));
                staged += cells;
            }
        }
    }
};

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool readNumber(const char* text, size_t& value) {
    if (!*text || *text == '-') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || *end || parsed > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(parsed);
    return true;
}

// Gather only when full result output is requested. Chunked transfers avoid
// MPI_Gatherv's int-sized displacement limit. Root restores original XYZ order.
void printDistributedResults(const std::vector<double>& local, Domain d,
                             size_t nx, size_t ny, size_t volume,
                             int rank, int ranks, MPI_Comm comm) {
    const size_t plane = d.nx * d.ny;
    if (rank != 0) {
        for (size_t offset = 0; offset < local.size(); ) {
            const int count = static_cast<int>(std::min<size_t>(INT_MAX, local.size() - offset));
            MPI_CHECK(MPI_Send(local.data() + offset, count, MPI_DOUBLE, 0, 4, comm));
            offset += count;
        }
        return;
    }
    std::vector<double> result(volume);
    std::vector<double> temporary;
    if (d.axis != 2 && ranks > 1) temporary.resize(local.size());
    for (int peer = 0; peer < ranks; ++peer) {
        const size_t first = slabStart(d.globalNz, ranks, peer);
        const size_t cells = slabSize(d.globalNz, ranks, peer) * plane;
        const double* source = local.data();
        if (peer) {
            double* destination = d.axis == 2 ? result.data() + first * plane : temporary.data();
            for (size_t offset = 0; offset < cells; ) {
                const int count = static_cast<int>(std::min<size_t>(INT_MAX, cells - offset));
                MPI_CHECK(MPI_Recv(destination + offset, count, MPI_DOUBLE, peer, 4, comm, MPI_STATUS_IGNORE));
                offset += count;
            }
            source = destination;
        }
        if (d.axis == 2) {
            if (!peer) std::copy(local.begin(), local.end(), result.begin());
        } else {
            const int threads = hostThreads(cells);
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (size_t i = 0; i < cells; ++i) {
                const size_t xy = i % plane;
                result[originalIndex(xy % d.nx, xy / d.nx, first + i / plane, d.axis, nx, ny)] = source[i];
            }
        }
    }
    print_results(result, "Concentration");
}

bool validateDistributed(const std::vector<double>& local, int rank, MPI_Comm comm) {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    int invalid = 0;
    const int threads = hostThreads(local.size());
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(min:minimum) reduction(max:maximum) reduction(|:invalid)
    for (size_t i = 0; i < local.size(); ++i) {
        const double value = local[i];
        invalid |= !std::isfinite(value);
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    double globalMinimum, globalMaximum;
    int globalInvalid;
    MPI_CHECK(MPI_Allreduce(&minimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN, comm));
    MPI_CHECK(MPI_Allreduce(&maximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX, comm));
    MPI_CHECK(MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_BOR, comm));
    const bool valid = !globalInvalid && globalMinimum >= -10.0 && globalMaximum <= 10.0;
    if (!rank) {
        std::printf("Validating result...\n");
        if (globalInvalid) std::printf("Validation failed: found NaN or Inf value\n");
        else {
            std::printf("Concentration range: [%.6f, %.6f]\n", globalMinimum, globalMaximum);
            if (!valid) std::printf("Validation failed: values out of expected range\n");
        }
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

int run(int argc, char** argv) {
    int rank, worldSize;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    size_t nx = 64, ny = 0, nz = 0, steps = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        size_t* number = nullptr;
        if (!std::strcmp(argv[i], "-x")) number = &nx;
        else if (!std::strcmp(argv[i], "-y")) number = &ny;
        else if (!std::strcmp(argv[i], "-z")) number = &nz;
        else if (!std::strcmp(argv[i], "-i")) number = &steps;
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
        if (number && (i + 1 == argc || !readNumber(argv[++i], *number))) {
            if (!rank) std::fprintf(stderr, "Invalid numeric argument\n");
            return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
    if (!nx || !ny || !nz || nx > limit / ny || nx * ny > limit / nz ||
        nx > static_cast<size_t>(INT64_MAX - 4) ||
        ny > static_cast<size_t>(INT64_MAX - 4) ||
        nz > static_cast<size_t>(INT64_MAX - 4) || steps > INT_MAX) {
        if (!rank) std::fprintf(stderr, "Grid dimensions must be positive and fit in memory; time steps must fit in int\n");
        return 1;
    }
    const size_t volume = nx * ny * nz;
    const int axis = nx > nz && nx >= ny ? 0 : (ny > nz ? 1 : 2);
    const size_t extent = axis == 0 ? nx : (axis == 1 ? ny : nz);
    const int ranks = static_cast<int>(std::min<size_t>(worldSize, extent));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < ranks ? 0 : MPI_UNDEFINED, rank, &comm));
    if (rank >= ranks) return 0;
    MPI_CHECK(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN));
    Domain d{axis == 0 ? ny : nx, axis == 2 ? ny : nz,
             slabSize(extent, ranks, rank), extent, slabStart(extent, ranks, rank), axis};
    const size_t plane = d.nx * d.ny;
    if (d.nz > limit / plane || limit / plane - d.nz < 4) fail("Local grid including halos is too large");
    const size_t cells = d.nz * plane;

    // Node-local ranks select GPUs correctly even when global rank ordering is
    // not contiguous by node. CUDA_VISIBLE_DEVICES restrictions are respected.
    MPI_Comm node = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank, devices = 0;
    MPI_CHECK(MPI_Comm_rank(node, &localRank));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) fail("At least one CUDA device per participating node is required");
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_CHECK(MPI_Comm_free(&node));
    bool deviceMpi = false;
#ifdef CH_MPI_CUDA_QUERY
    deviceMpi = MPIX_Query_cuda_support() != 0;
#endif
    // All communicating ranks must use the same transport mode.
    int direct = deviceMpi ? 1 : 0, allDirect = 0;
    MPI_CHECK(MPI_Allreduce(&direct, &allDirect, 1, MPI_INT, MPI_MIN, comm));
    deviceMpi = allDirect != 0;

    if (!rank) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %zu\n", steps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallelization: MPI + OpenMP + CUDA (%d ranks, %c slabs, %s halos)\n",
                    ranks, "XYZ"[axis], deviceMpi ? "device MPI" : "pinned host");
        std::printf("Initializing concentration field...\n");
    }

    DeviceBuffer a((d.nz + 4) * plane), b((d.nz + 4) * plane), potential((d.nz + 4) * plane);
    Stream compute;
    int lowestPriority, highestPriority;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowestPriority, &highestPriority));
    Stream transfer(highestPriority);
    Event fieldReady, haloReady;
    std::vector<double> local(cells);
    const int threads = hostThreads(cells);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < cells; ++i) {
        const size_t xy = i % plane;
        const size_t global = originalIndex(xy % d.nx, xy / d.nx, d.first + i / plane, axis, nx, ny);
        const double pseudo = (((global + 1) * size_t{1299709}) % volume) / static_cast<double>(volume);
        local[i] = -1.0 + 2.0 * pseudo;
    }
    CUDA_CHECK(cudaMemcpy(a.data + 2 * plane, local.data(), cells * sizeof(double), cudaMemcpyHostToDevice));
    if (!validate && !printResults) std::vector<double>().swap(local);
    HaloExchange exchange(d, ranks, rank, comm, deviceMpi);

    cudaGraphExec_t interior[2] = {nullptr, nullptr};
    cudaGraphExec_t boundary[2] = {nullptr, nullptr};
    cudaGraphExec_t pair = nullptr, single = nullptr;
    if (ranks == 1) {
        auto step = [&](double* cold, double* next) {
            stencil<true>(cold, cold, potential.data, d, 2, d.nz, compute.handle);
            stencil<false>(potential.data, cold, next, d, 2, d.nz, compute.handle);
        };
        pair = captureGraph(compute.handle, [&] { step(a.data, b.data); step(b.data, a.data); });
        single = captureGraph(compute.handle, [&] { step(a.data, b.data); });
    } else {
        for (int parity = 0; parity < 2; ++parity) {
            double* cold = parity ? b.data : a.data;
            double* next = parity ? a.data : b.data;
            if (d.nz > 2) {
                interior[parity] = captureGraph(compute.handle, [&] {
                    stencil<true>(cold, cold, potential.data, d, 3, d.nz - 2, compute.handle);
                    if (d.nz > 4)
                        stencil<false>(potential.data, cold, next, d, 4, d.nz - 4, compute.handle);
                });
            }
            boundary[parity] = captureGraph(compute.handle, [&] {
                stencil<true, true>(cold, cold, potential.data, d, 0, d.nz == 1 ? 3 : 4, compute.handle);
                stencil<false, true>(potential.data, cold, next, d, 0, std::min<size_t>(4, d.nz), compute.handle);
            });
        }
    }
    CUDA_CHECK(cudaEventRecord(fieldReady.handle, compute.handle));
    CUDA_CHECK(cudaStreamSynchronize(compute.handle));
    if (!rank) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_CHECK(MPI_Barrier(comm));
    const double start = MPI_Wtime();
    double* cold = a.data;
    if (ranks == 1) {
        // A two-step CUDA graph avoids CPU launch overhead and returns to buffer A.
        for (size_t t = 0; t < steps / 2; ++t) CUDA_CHECK(cudaGraphLaunch(pair, compute.handle));
        if (steps % 2) {
            CUDA_CHECK(cudaGraphLaunch(single, compute.handle));
            cold = b.data;
        }
    } else {
        for (size_t t = 0; t < steps; ++t) {
            const int parity = static_cast<int>(t % 2);
            cold = parity ? b.data : a.data;
            // Receive staging cannot be reused before the preceding H2D copy.
            if (t && !deviceMpi) CUDA_CHECK(cudaEventSynchronize(haloReady.handle));
            exchange.postReceives(cold);
            CUDA_CHECK(cudaStreamWaitEvent(transfer.handle, fieldReady.handle, 0));
            exchange.stageSends(cold, transfer.handle);
            if (interior[parity]) CUDA_CHECK(cudaGraphLaunch(interior[parity], compute.handle));
            // Only edge data must be ready for MPI; interior GPU work continues.
            CUDA_CHECK(cudaStreamSynchronize(transfer.handle));
            exchange.postSends(cold);
            exchange.finish(cold, transfer.handle);
            CUDA_CHECK(cudaEventRecord(haloReady.handle, transfer.handle));
            CUDA_CHECK(cudaStreamWaitEvent(compute.handle, haloReady.handle, 0));
            CUDA_CHECK(cudaGraphLaunch(boundary[parity], compute.handle));
            CUDA_CHECK(cudaEventRecord(fieldReady.handle, compute.handle));
        }
        cold = steps % 2 ? b.data : a.data;
    }
    CUDA_CHECK(cudaStreamSynchronize(compute.handle));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (!rank) {
        std::printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        const double mcups = seconds > 0 ? static_cast<double>(volume) * steps / seconds / 1e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    if (validate || printResults) {
        CUDA_CHECK(cudaMemcpy(local.data(), cold + 2 * plane, cells * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (printResults) printDistributedResults(local, d, nx, ny, volume, rank, ranks, comm);
    const bool valid = !validate || validateDistributed(local, rank, comm);

    for (int parity = 0; parity < 2; ++parity) {
        if (interior[parity]) CUDA_CHECK(cudaGraphExecDestroy(interior[parity]));
        if (boundary[parity]) CUDA_CHECK(cudaGraphExecDestroy(boundary[parity]));
    }
    if (pair) CUDA_CHECK(cudaGraphExecDestroy(pair));
    if (single) CUDA_CHECK(cudaGraphExecDestroy(single));
    MPI_CHECK(MPI_Comm_free(&comm));
    return valid ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    MpiSession mpi(argc, argv);
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        fail(error.what());
    }
}
