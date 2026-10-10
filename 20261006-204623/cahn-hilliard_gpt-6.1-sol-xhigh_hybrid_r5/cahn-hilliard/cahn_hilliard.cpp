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
#include <omp.h>
#include <cuda_runtime.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

static void checkMPI(int error, const char* operation) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "%s: %.*s\n", operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}
static void checkCUDA(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}
#define MPI_CHECK(call) checkMPI((call), #call)
#define CUDA_CHECK(call) checkCUDA((call), #call)

// All coordinates in this structure are local, zero based. The fields have
// one ghost cell on each face; no edge or corner halos are needed by either
// seven-point stencil. The inner box excludes only communicating faces.
struct Grid {
    size_t n[3], strideY, strideZ, cells;
    size_t low[3], high[3], inner[3];
    size_t faceOffset[7], borderOffset[7];
    unsigned physical;
};

__device__ __forceinline__ size_t fieldIndex(const Grid& g, size_t x, size_t y, size_t z) {
    return (z + 1) * g.strideZ + (y + 1) * g.strideY + x + 1;
}

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                           const Grid& g, size_t x, size_t y, size_t z,
                                           size_t index) {
    const size_t xm = (x == 0 && (g.physical & 1)) ? index : index - 1;
    const size_t xp = (x + 1 == g.n[0] && (g.physical & 2)) ? index : index + 1;
    const size_t ym = (y == 0 && (g.physical & 4)) ? index : index - g.strideY;
    const size_t yp = (y + 1 == g.n[1] && (g.physical & 8)) ? index : index + g.strideY;
    const size_t zm = (z == 0 && (g.physical & 16)) ? index : index - g.strideZ;
    const size_t zp = (z + 1 == g.n[2] && (g.physical & 32)) ? index : index + g.strideZ;
    const double center = field[index];
    // dx = dy = dz = 1 in the original benchmark. Keep its evaluation order.
    const double cxx = field[xp] + field[xm] - 2.0 * center;
    const double cyy = field[yp] + field[ym] - 2.0 * center;
    const double czz = field[zp] + field[zm] - 2.0 * center;
    return cxx + cyy + czz;
}

template<bool Chemical>
__device__ __forceinline__ void stencilCell(const double* __restrict__ input,
                                           const double* __restrict__ concentration,
                                           double* __restrict__ output, const Grid& g,
                                           size_t x, size_t y, size_t z) {
    const size_t index = fieldIndex(g, x, y, z);
    const double lap = laplacian(input, g, x, y, z, index);
    if constexpr (Chemical) {
        constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
        const double cv = input[index];
        output[index] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv - 0.5 * lap;
    } else {
        output[index] = concentration[index] + 0.01 * lap;
    }
}

template<bool Chemical>
__global__ void interiorKernel(const double* __restrict__ input,
                               const double* __restrict__ concentration,
                               double* __restrict__ output, Grid g) {
    // X-contiguous warps give coalesced accesses. Grid strides also accommodate
    // dimensions exceeding CUDA's Y/Z grid limits.
    for (size_t z = g.low[2] + blockIdx.z * blockDim.z + threadIdx.z;
         z < g.high[2]; z += size_t(gridDim.z) * blockDim.z)
        for (size_t y = g.low[1] + blockIdx.y * blockDim.y + threadIdx.y;
             y < g.high[1]; y += size_t(gridDim.y) * blockDim.y)
            for (size_t x = g.low[0] + blockIdx.x * blockDim.x + threadIdx.x;
                 x < g.high[0]; x += size_t(gridDim.x) * blockDim.x)
                stencilCell<Chemical>(input, concentration, output, g, x, y, z);
}

template<bool Chemical>
__global__ void borderKernel(const double* __restrict__ input,
                             const double* __restrict__ concentration,
                             double* __restrict__ output, Grid g) {
    // Six disjoint slabs partition the complement of the inner box. In
    // particular, edges and one-cell-wide subdomains are written exactly once.
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < g.borderOffset[6]; i += size_t(blockDim.x) * gridDim.x) {
        int face = 0;
        while (i >= g.borderOffset[face + 1]) ++face;
        const size_t j = i - g.borderOffset[face];
        size_t x, y, z;
        if (face < 2) {
            x = face == 0 ? 0 : g.high[0];
            y = j % g.n[1];
            z = j / g.n[1];
        } else if (face < 4) {
            x = g.low[0] + j % g.inner[0];
            y = face == 2 ? 0 : g.high[1];
            z = j / g.inner[0];
        } else {
            x = g.low[0] + j % g.inner[0];
            y = g.low[1] + (j / g.inner[0]) % g.inner[1];
            z = face == 4 ? 0 : g.high[2];
        }
        stencilCell<Chemical>(input, concentration, output, g, x, y, z);
    }
}

template<bool Pack>
__global__ void faceKernel(double* field, double* buffer, Grid g) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < g.faceOffset[6]; i += size_t(blockDim.x) * gridDim.x) {
        int face = 0;
        while (i >= g.faceOffset[face + 1]) ++face;
        const size_t j = i - g.faceOffset[face];
        const int axis = face / 2;
        size_t x, y, z;
        const size_t coordinate = (face & 1) ? g.n[axis] + (Pack ? 0 : 1) : (Pack ? 1 : 0);
        if (axis == 0) {
            x = coordinate; y = 1 + j % g.n[1]; z = 1 + j / g.n[1];
        } else if (axis == 1) {
            x = 1 + j % g.n[0]; y = coordinate; z = 1 + j / g.n[0];
        } else {
            x = 1 + j % g.n[0]; y = 1 + j / g.n[0]; z = coordinate;
        }
        const size_t index = z * g.strideZ + y * g.strideY + x;
        if constexpr (Pack) buffer[i] = field[index];
        else field[index] = buffer[i];
    }
}

template<bool Scatter>
__global__ void transferKernel(double* field, double* contiguous, Grid g) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < g.cells; i += size_t(blockDim.x) * gridDim.x) {
        const size_t x = i % g.n[0];
        const size_t y = (i / g.n[0]) % g.n[1];
        const size_t z = i / (g.n[0] * g.n[1]);
        const size_t index = fieldIndex(g, x, y, z);
        if constexpr (Scatter) field[index] = contiguous[i];
        else contiguous[i] = field[index];
    }
}

static unsigned blocksFor(size_t count) {
    return static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535));
}

template<bool Chemical>
static void launchInterior(const double* input, const double* concentration,
                           double* output, const Grid& g, cudaStream_t stream) {
    if (!g.inner[0] || !g.inner[1] || !g.inner[2]) return;
    const dim3 threads(32, 4, 2);
    const dim3 blocks(static_cast<unsigned>(std::min<size_t>((g.inner[0] + 31) / 32, 65535)),
                      static_cast<unsigned>(std::min<size_t>((g.inner[1] + 3) / 4, 65535)),
                      static_cast<unsigned>(std::min<size_t>((g.inner[2] + 1) / 2, 65535)));
    interiorKernel<Chemical><<<blocks, threads, 0, stream>>>(input, concentration, output, g);
    CUDA_CHECK(cudaGetLastError());
}

// Select a surface-area-efficient process grid subject to nonempty blocks.
// This also handles thin grids and prime rank counts without empty GPU work.
static int chooseProcessGrid(int ranks, const size_t global[3], int dims[3]) {
    for (int active = ranks; active > 0; --active) {
        double best = std::numeric_limits<double>::infinity();
        std::vector<int> divisors;
        for (int divisor = 1; divisor <= active / divisor; ++divisor) {
            if (active % divisor) continue;
            divisors.push_back(divisor);
            if (divisor != active / divisor) divisors.push_back(active / divisor);
        }
        for (int px : divisors) {
            if (size_t(px) > global[0]) continue;
            const int rest = active / px;
            for (int py : divisors) {
                if (size_t(py) > global[1] || rest % py) continue;
                const int pz = rest / py;
                if (size_t(pz) > global[2]) continue;
                const double cost = double(px - 1) / global[0]
                                  + double(py - 1) / global[1]
                                  + double(pz - 1) / global[2];
                if (cost < best) {
                    best = cost;
                    dims[0] = px; dims[1] = py; dims[2] = pz;
                }
            }
        }
        if (std::isfinite(best)) return active;
    }
    return 1;
}

static Grid makeGrid(const size_t local[3], const int peers[6]) {
    Grid g{};
    for (int axis = 0; axis < 3; ++axis) {
        g.n[axis] = local[axis];
        g.low[axis] = peers[2 * axis] == MPI_PROC_NULL ? 0 : 1;
        g.high[axis] = std::max(g.low[axis], local[axis] - (peers[2 * axis + 1] == MPI_PROC_NULL ? 0 : 1));
        g.inner[axis] = g.high[axis] - g.low[axis];
    }
    g.strideY = local[0] + 2;
    g.strideZ = g.strideY * (local[1] + 2);
    g.cells = local[0] * local[1] * local[2];
    for (int face = 0; face < 6; ++face) {
        if (peers[face] == MPI_PROC_NULL) g.physical |= 1u << face;
        const int axis = face / 2;
        const size_t count = peers[face] == MPI_PROC_NULL ? 0 : local[(axis + 1) % 3] * local[(axis + 2) % 3];
        g.faceOffset[face + 1] = g.faceOffset[face] + count;
        size_t border;
        if (axis == 0) border = local[1] * local[2];
        else if (axis == 1) border = g.inner[0] * local[2];
        else border = g.inner[0] * g.inner[1];
        border *= (face & 1) ? local[axis] - g.high[axis] : g.low[axis];
        g.borderOffset[face + 1] = g.borderOffset[face] + border;
    }
    return g;
}

struct GPU {
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr, *host = nullptr;
    double *send = nullptr, *receive = nullptr, *hostSend = nullptr, *hostReceive = nullptr;
    cudaStream_t compute{}, communication{};
    cudaEvent_t phaseReady{}, sendReady{}, haloReady{};
    cudaGraph_t graph{};
    cudaGraphExec_t graphExec{};

    GPU(const Grid& g, bool direct) {
        const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
        if (g.strideZ > limit / (g.n[2] + 2) || g.faceOffset[6] > limit) {
            std::fprintf(stderr, "Local GPU allocation size overflow\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        const size_t bytes = g.strideZ * (g.n[2] + 2) * sizeof(double);
        CUDA_CHECK(cudaMalloc(&cold, bytes));
        CUDA_CHECK(cudaMalloc(&cnew, bytes));
        CUDA_CHECK(cudaMalloc(&mu, bytes));
        CUDA_CHECK(cudaMallocHost(&host, g.cells * sizeof(double)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        int leastPriority, greatestPriority;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&communication, cudaStreamNonBlocking, greatestPriority));
        CUDA_CHECK(cudaEventCreateWithFlags(&phaseReady, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&sendReady, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&haloReady, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(haloReady, communication));
        if (g.faceOffset[6]) {
            const size_t faceBytes = g.faceOffset[6] * sizeof(double);
            CUDA_CHECK(cudaMalloc(&send, faceBytes));
            CUDA_CHECK(cudaMalloc(&receive, faceBytes));
            if (!direct) {
                CUDA_CHECK(cudaMallocHost(&hostSend, faceBytes));
                CUDA_CHECK(cudaMallocHost(&hostReceive, faceBytes));
            }
        }
    }
    ~GPU() {
        if (graphExec) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu));
        CUDA_CHECK(cudaFreeHost(host));
        if (send) CUDA_CHECK(cudaFree(send));
        if (receive) CUDA_CHECK(cudaFree(receive));
        if (hostSend) CUDA_CHECK(cudaFreeHost(hostSend));
        if (hostReceive) CUDA_CHECK(cudaFreeHost(hostReceive));
        CUDA_CHECK(cudaEventDestroy(phaseReady));
        CUDA_CHECK(cudaEventDestroy(sendReady));
        CUDA_CHECK(cudaEventDestroy(haloReady));
        CUDA_CHECK(cudaStreamDestroy(compute));
        CUDA_CHECK(cudaStreamDestroy(communication));
    }
};

static void copyInterior(GPU& gpu, const Grid& g, bool toDevice) {
    // mu is unused before and after the simulation, so it can hold a compact
    // transfer buffer without an extra allocation. A single contiguous PCIe
    // transfer avoids the per-row overhead of a narrow pitched 3D copy.
    if (toDevice) {
        CUDA_CHECK(cudaMemcpyAsync(gpu.mu, gpu.host, g.cells * sizeof(double),
                                   cudaMemcpyHostToDevice, gpu.compute));
        transferKernel<true><<<blocksFor(g.cells), 256, 0, gpu.compute>>>(gpu.cold, gpu.mu, g);
        CUDA_CHECK(cudaGetLastError());
    } else {
        transferKernel<false><<<blocksFor(g.cells), 256, 0, gpu.compute>>>(gpu.cold, gpu.mu, g);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(gpu.host, gpu.mu, g.cells * sizeof(double),
                                   cudaMemcpyDeviceToHost, gpu.compute));
    }
}

class HaloExchange {
    GPU& gpu;
    const Grid& grid;
    const int* peers;
    MPI_Comm communicator;
    bool direct;
    std::vector<MPI_Request> requests;

    void post(bool sending) {
        double* buffer = sending ? (direct ? gpu.send : gpu.hostSend)
                                 : (direct ? gpu.receive : gpu.hostReceive);
        for (int face = 0; face < 6; ++face) {
            if (peers[face] == MPI_PROC_NULL) continue;
            size_t begin = grid.faceOffset[face];
            const size_t end = grid.faceOffset[face + 1];
            while (begin < end) {
                const int count = static_cast<int>(std::min<size_t>(end - begin, INT_MAX));
                MPI_Request request;
                if (sending) {
                    MPI_CHECK(MPI_Isend(buffer + begin, count, MPI_DOUBLE, peers[face], face,
                                        communicator, &request));
                } else {
                    MPI_CHECK(MPI_Irecv(buffer + begin, count, MPI_DOUBLE, peers[face], face ^ 1,
                                        communicator, &request));
                }
                requests.push_back(request);
                begin += count;
            }
        }
    }
public:
    HaloExchange(GPU& state, const Grid& g, const int neighbors[6], MPI_Comm comm, bool cudaAware)
        : gpu(state), grid(g), peers(neighbors), communicator(comm), direct(cudaAware) {
        size_t messages = 0;
        for (int face = 0; face < 6; ++face)
            messages += (grid.faceOffset[face + 1] - grid.faceOffset[face] + INT_MAX - 1) / INT_MAX;
        requests.reserve(2 * messages);
    }

    template<bool Chemical>
    void step(double* input, const double* concentration, double* output) {
        // A previous asynchronous unpack must release the receive buffer before
        // MPI writes the next phase's messages into it.
        CUDA_CHECK(cudaEventSynchronize(gpu.haloReady));
        requests.clear();
        post(false);
        CUDA_CHECK(cudaStreamWaitEvent(gpu.communication, gpu.phaseReady, 0));
        faceKernel<true><<<blocksFor(grid.faceOffset[6]), 256, 0, gpu.communication>>>(input, gpu.send, grid);
        CUDA_CHECK(cudaGetLastError());
        if (!direct)
            CUDA_CHECK(cudaMemcpyAsync(gpu.hostSend, gpu.send, grid.faceOffset[6] * sizeof(double),
                                       cudaMemcpyDeviceToHost, gpu.communication));
        CUDA_CHECK(cudaEventRecord(gpu.sendReady, gpu.communication));
        launchInterior<Chemical>(input, concentration, output, grid, gpu.compute);
        CUDA_CHECK(cudaEventSynchronize(gpu.sendReady));
        post(true);
        // MPI progresses while the independent interior stencil runs on CUDA.
        MPI_CHECK(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
        if (!direct)
            CUDA_CHECK(cudaMemcpyAsync(gpu.receive, gpu.hostReceive, grid.faceOffset[6] * sizeof(double),
                                       cudaMemcpyHostToDevice, gpu.communication));
        faceKernel<false><<<blocksFor(grid.faceOffset[6]), 256, 0, gpu.communication>>>(input, gpu.receive, grid);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(gpu.haloReady, gpu.communication));
        CUDA_CHECK(cudaStreamWaitEvent(gpu.compute, gpu.haloReady, 0));
        if (grid.borderOffset[6]) {
            borderKernel<Chemical><<<blocksFor(grid.borderOffset[6]), 256, 0, gpu.compute>>>(input, concentration, output, grid);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(gpu.phaseReady, gpu.compute));
    }
};

static void gatherResults(const double* local, const Grid& g, const size_t origin[3],
                          const size_t global[3], MPI_Comm comm, int rank, int ranks) {
    unsigned long long description[6];
    for (int axis = 0; axis < 3; ++axis) {
        description[axis] = origin[axis];
        description[axis + 3] = g.n[axis];
    }
    std::vector<unsigned long long> descriptions(rank == 0 ? size_t(6) * ranks : 0);
    MPI_CHECK(MPI_Gather(description, 6, MPI_UNSIGNED_LONG_LONG, descriptions.data(), 6,
                         MPI_UNSIGNED_LONG_LONG, 0, comm));
    if (rank != 0) {
        for (size_t i = 0; i < g.cells;) {
            const int count = static_cast<int>(std::min<size_t>(g.cells - i, INT_MAX));
            MPI_CHECK(MPI_Send(local + i, count, MPI_DOUBLE, 0, 100, comm));
            i += count;
        }
        return;
    }
    std::vector<double> result(global[0] * global[1] * global[2]);
    std::vector<double> receive;
    for (int source = 0; source < ranks; ++source) {
        const auto* info = descriptions.data() + 6 * source;
        const size_t nx = info[3], ny = info[4], nz = info[5];
        const size_t count = nx * ny * nz;
        const double* block = local;
        if (source) {
            receive.resize(count);
            for (size_t i = 0; i < count;) {
                const int chunk = static_cast<int>(std::min<size_t>(count - i, INT_MAX));
                MPI_CHECK(MPI_Recv(receive.data() + i, chunk, MPI_DOUBLE, source, 100, comm, MPI_STATUS_IGNORE));
                i += chunk;
            }
            block = receive.data();
        }
        #pragma omp parallel for collapse(2) schedule(static) if(count >= 32768)
        for (size_t z = 0; z < nz; ++z)
            for (size_t y = 0; y < ny; ++y) {
                const size_t destination = ((z + info[2]) * global[1] + y + info[1]) * global[0] + info[0];
                std::copy_n(block + (z * ny + y) * nx, nx, result.data() + destination);
            }
    }
    print_results(result, "Concentration");
}

static bool validateResult(const double* data, size_t count, MPI_Comm comm, int rank) {
    int invalid = 0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    #pragma omp parallel for schedule(static) reduction(|:invalid) reduction(min:minimum) reduction(max:maximum) if(count >= 32768)
    for (size_t i = 0; i < count; ++i) {
        const double value = data[i];
        invalid |= !std::isfinite(value);
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_BOR, comm));
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_DOUBLE, MPI_MIN, comm));
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_DOUBLE, MPI_MAX, comm));
    if (invalid) {
        if (rank == 0) std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (rank == 0) std::printf("Concentration range: [%.6f, %.6f]\n", minimum, maximum);
    if (maximum > 10.0 || minimum < -10.0) {
        if (rank == 0) std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

static bool parseNumber(const char* text, size_t& number) {
    if (!*text || *text == '-') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno || *end || value > std::numeric_limits<size_t>::max()) return false;
    number = static_cast<size_t>(value);
    return true;
}

static int run(int argc, char** argv, int worldRank, int worldSize) {
    size_t global[3] = {64, 0, 0};
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-h")) {
            if (!worldRank) printUsage(argv[0]);
            return 0;
        }
        if (!std::strcmp(argv[i], "-v")) { validate = true; continue; }
        if (!std::strcmp(argv[i], "-r")) { printResults = true; continue; }
        int option = -1;
        if (!std::strcmp(argv[i], "-x")) option = 0;
        if (!std::strcmp(argv[i], "-y")) option = 1;
        if (!std::strcmp(argv[i], "-z")) option = 2;
        if (!std::strcmp(argv[i], "-i")) option = 3;
        size_t number = 0;
        if (option < 0 || i + 1 == argc || !parseNumber(argv[i + 1], number)
            || (option == 0 && !number) || (option == 3 && number > INT_MAX)) {
            if (!worldRank) {
                std::fprintf(stderr, "Invalid option or value: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
        ++i;
        if (option == 3) iterations = static_cast<int>(number);
        else global[option] = number;
    }
    if (!global[1]) global[1] = global[0];
    if (!global[2]) global[2] = global[0];
    // The padded field uses at most 27 times the interior volume. Bounding
    // the total here makes all later size_t/byte arithmetic safe, even for 1^3.
    const size_t limit = std::min<size_t>(std::numeric_limits<size_t>::max() / (27 * sizeof(double)),
                                        std::numeric_limits<ptrdiff_t>::max());
    if (global[0] > limit / global[1] || global[0] * global[1] > limit / global[2]) {
        if (!worldRank) std::fprintf(stderr, "Grid size overflow\n");
        return 1;
    }
    const size_t volume = global[0] * global[1] * global[2];
    int dims[3] = {1, 1, 1};
    const int active = chooseProcessGrid(static_cast<int>(std::min<size_t>(worldSize, volume)), global, dims);
    MPI_Comm workers = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &workers));
    int status = 0;
    if (workers != MPI_COMM_NULL) {
        const int periods[3] = {0, 0, 0};
        MPI_Comm cart;
        MPI_CHECK(MPI_Cart_create(workers, 3, dims, periods, 0, &cart));
        int rank, coordinates[3], peers[6];
        MPI_CHECK(MPI_Comm_rank(cart, &rank));
        MPI_CHECK(MPI_Cart_coords(cart, rank, 3, coordinates));
        size_t origin[3], local[3];
        for (int axis = 0; axis < 3; ++axis) {
            MPI_CHECK(MPI_Cart_shift(cart, axis, 1, &peers[2 * axis], &peers[2 * axis + 1]));
            const size_t quotient = global[axis] / dims[axis], remainder = global[axis] % dims[axis];
            local[axis] = quotient + (size_t(coordinates[axis]) < remainder);
            origin[axis] = size_t(coordinates[axis]) * quotient + std::min<size_t>(coordinates[axis], remainder);
        }
        MPI_Comm shared;
        MPI_CHECK(MPI_Comm_split_type(cart, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared));
        int localRank;
        MPI_CHECK(MPI_Comm_rank(shared, &localRank));
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (!deviceCount) {
            std::fprintf(stderr, "A CUDA GPU is required on every active MPI rank\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
        MPI_CHECK(MPI_Comm_free(&shared));
        int cudaAware = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        cudaAware = MPIX_Query_cuda_support() != 0;
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        cudaAware = 1;
#endif
        // All peers must agree on the memory space of their messages.
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &cudaAware, 1, MPI_INT, MPI_MIN, cart));
        omp_set_dynamic(0);
        if (!std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::min(8, omp_get_max_threads()));
        const Grid grid = makeGrid(local, peers);
        {
            GPU gpu(grid, cudaAware != 0);
            if (!rank) {
                std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
                std::printf("Grid size: %zu x %zu x %zu\n", global[0], global[1], global[2]);
                std::printf("Time steps: %d\n", iterations);
                std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
                std::printf("MPI process grid: %d x %d x %d (%d active ranks)\n", dims[0], dims[1], dims[2], active);
                std::printf("Halo transport: %s\n", cudaAware ? "CUDA-aware MPI" : "pinned host staging");
                std::printf("Initializing concentration field...\n");
            }
            #pragma omp parallel for schedule(static) if(grid.cells >= 32768)
            for (size_t i = 0; i < grid.cells; ++i) {
                const size_t x = i % local[0];
                const size_t y = (i / local[0]) % local[1];
                const size_t z = i / (local[0] * local[1]);
                const size_t linear = ((z + origin[2]) * global[1] + y + origin[1]) * global[0] + x + origin[0];
                const double pseudo = (((linear + 1) * size_t(1299709)) % volume) / static_cast<double>(volume);
                gpu.host[i] = -1.0 + 2.0 * pseudo;
            }
            copyInterior(gpu, grid, true);
            CUDA_CHECK(cudaEventRecord(gpu.phaseReady, gpu.compute));
            CUDA_CHECK(cudaStreamSynchronize(gpu.compute));
            HaloExchange exchange(gpu, grid, peers, cart, cudaAware != 0);
            if (!grid.faceOffset[6] && iterations >= 2) {
                // A two-step graph returns to the original concentration
                // buffer, avoiding per-step launch overhead on a single rank.
                CUDA_CHECK(cudaStreamBeginCapture(gpu.compute, cudaStreamCaptureModeThreadLocal));
                launchInterior<true>(gpu.cold, gpu.cold, gpu.mu, grid, gpu.compute);
                launchInterior<false>(gpu.mu, gpu.cold, gpu.cnew, grid, gpu.compute);
                launchInterior<true>(gpu.cnew, gpu.cnew, gpu.mu, grid, gpu.compute);
                launchInterior<false>(gpu.mu, gpu.cnew, gpu.cold, grid, gpu.compute);
                CUDA_CHECK(cudaStreamEndCapture(gpu.compute, &gpu.graph));
                CUDA_CHECK(cudaGraphInstantiate(&gpu.graphExec, gpu.graph, nullptr, nullptr, 0));
                CUDA_CHECK(cudaGraphUpload(gpu.graphExec, gpu.compute));
            }
            // Load kernels and initialize the communication path before timing.
            // The trial update writes only cnew, leaving the initial cold intact.
            if (iterations > 0) {
                if (grid.faceOffset[6]) {
                    exchange.step<true>(gpu.cold, gpu.cold, gpu.mu);
                    exchange.step<false>(gpu.mu, gpu.cold, gpu.cnew);
                } else {
                    launchInterior<true>(gpu.cold, gpu.cold, gpu.mu, grid, gpu.compute);
                    launchInterior<false>(gpu.mu, gpu.cold, gpu.cnew, grid, gpu.compute);
                }
                CUDA_CHECK(cudaStreamSynchronize(gpu.compute));
            }
            if (!rank) std::printf("Running Cahn-Hilliard simulation...\n");
            MPI_CHECK(MPI_Barrier(cart));
            const double start = MPI_Wtime();
            if (!grid.faceOffset[6]) {
                int t = 0;
                for (; t + 1 < iterations; t += 2) CUDA_CHECK(cudaGraphLaunch(gpu.graphExec, gpu.compute));
                if (t < iterations) {
                    launchInterior<true>(gpu.cold, gpu.cold, gpu.mu, grid, gpu.compute);
                    launchInterior<false>(gpu.mu, gpu.cold, gpu.cnew, grid, gpu.compute);
                    std::swap(gpu.cold, gpu.cnew);
                }
            } else {
                for (int t = 0; t < iterations; ++t) {
                    exchange.step<true>(gpu.cold, gpu.cold, gpu.mu);
                    exchange.step<false>(gpu.mu, gpu.cold, gpu.cnew);
                    std::swap(gpu.cold, gpu.cnew);
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(gpu.compute));
            const double localTime = MPI_Wtime() - start;
            double elapsed = 0;
            MPI_CHECK(MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart));
            if (!rank) {
                std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
                const double mcups = elapsed > 0 ? double(volume) * iterations / elapsed / 1e6 : 0.0;
                std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
            }
            if (printResults || validate) {
                copyInterior(gpu, grid, false);
                CUDA_CHECK(cudaStreamSynchronize(gpu.compute));
            }
            if (printResults) gatherResults(gpu.host, grid, origin, global, cart, rank, active);
            if (validate) {
                if (!rank) std::printf("Validating result...\n");
                const bool valid = validateResult(gpu.host, grid.cells, cart, rank);
                if (!rank) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
        MPI_CHECK(MPI_Comm_free(&cart));
        MPI_CHECK(MPI_Comm_free(&workers));
    }
    // Inactive ranks participate in shutdown and receive the same exit status.
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int status = 1;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Finalize());
    return status;
}
