#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#ifdef CH_MPI_CUDA_QUERY_EXT
#include <mpi-ext.h>
#endif
#include <omp.h>

#include "../common/results_output.hpp"

// MPI is called only by the initializing thread (MPI_THREAD_FUNNELED).
static void mpiCheck(int error, const char* operation) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0, rank = 0;
    MPI_Error_string(error, message, &length);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
#define MPI_CHECK(operation) mpiCheck((operation), #operation)
#define CUDA_CHECK(operation) cudaCheck((operation), #operation)

struct Options {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, results = false;
};

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

// Returns 2 for help, 1 for an invalid command line, and 0 on success.
static int parseOptions(int argc, char** argv, Options& o, int rank) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 2;
        }
        if (std::strcmp(argv[i], "-v") == 0) { o.validate = true; continue; }
        if (std::strcmp(argv[i], "-r") == 0) { o.results = true; continue; }
        const char* option = argv[i];
        const bool numeric = std::strcmp(option, "-x") == 0 || std::strcmp(option, "-y") == 0 ||
                             std::strcmp(option, "-z") == 0 || std::strcmp(option, "-i") == 0;
        if (!numeric || i + 1 == argc) {
            if (rank == 0) { std::printf("Unknown or incomplete option: %s\n", option); printUsage(argv[0]); }
            return 1;
        }
        const char* value = argv[++i];
        char* end = nullptr;
        errno = 0;
        const unsigned long long number = std::strtoull(value, &end, 10);
        const bool timestep = std::strcmp(option, "-i") == 0;
        if (*value == '-' || end == value || *end != '\0' || errno != 0 ||
            number > static_cast<unsigned long long>(timestep ? INT_MAX : INT_MAX - 2) ||
            (!timestep && std::strcmp(option, "-x") == 0 && number == 0)) {
            if (rank == 0) std::fprintf(stderr, "Invalid value for %s: %s\n", option, value);
            return 1;
        }
        if (timestep) o.iterations = static_cast<int>(number);
        else if (std::strcmp(option, "-x") == 0) o.nx = static_cast<size_t>(number);
        else if (std::strcmp(option, "-y") == 0) o.ny = static_cast<size_t>(number);
        else o.nz = static_cast<size_t>(number);
    }
    if (o.ny == 0) o.ny = o.nx;
    if (o.nz == 0) o.nz = o.nx;
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
    if (o.nx > limit / o.ny || o.nx * o.ny > limit / o.nz) {
        if (rank == 0) std::fprintf(stderr, "Grid size is too large\n");
        return 1;
    }
    return 0;
}

// Choose a nonperiodic Cartesian decomposition with no empty subdomains.
// Maximize useful ranks, then minimize the total area of partition interfaces.
// Thin grids and more ranks than cells are handled by leaving excess ranks idle.
static int chooseDecomposition(const Options& o, int ranks, int dims[3]) {
    int active = 0;
    long double bestArea = std::numeric_limits<long double>::infinity();
    const int maxX = static_cast<int>(std::min(o.nx, static_cast<size_t>(ranks)));
    for (int px = 1; px <= maxX; ++px) {
        const int maxY = static_cast<int>(std::min(o.ny, static_cast<size_t>(ranks / px)));
        for (int py = 1; py <= maxY; ++py) {
            const int pz = static_cast<int>(std::min(o.nz, static_cast<size_t>(ranks / px / py)));
            const int count = px * py * pz;
            const long double area = static_cast<long double>(px - 1) * o.ny * o.nz +
                                     static_cast<long double>(py - 1) * o.nx * o.nz +
                                     static_cast<long double>(pz - 1) * o.nx * o.ny;
            if (count > active || (count == active && area < bestArea)) {
                active = count;
                bestArea = area;
                dims[0] = px; dims[1] = py; dims[2] = pz;
            }
        }
    }
    return active;
}

static int localExtent(size_t global, int partitions, int coordinate) {
    return static_cast<int>(global / partitions + (static_cast<size_t>(coordinate) < global % partitions));
}
static size_t localOffset(size_t global, int partitions, int coordinate) {
    return static_cast<size_t>(coordinate) * (global / partitions) +
           std::min(static_cast<size_t>(coordinate), global % partitions);
}

struct Geometry {
    int nx, ny, nz;
    size_t pitch, slice;
    int physical[6];
};
struct Box {
    int lo[3], hi[3];                 // Half-open coordinates in the padded field.
    size_t volume;
};
static Box makeBox(int x0, int x1, int y0, int y1, int z0, int z1) {
    return {{x0, y0, z0}, {x1, y1, z1},
            static_cast<size_t>(x1 - x0) * (y1 - y0) * (z1 - z0)};
}
struct Shell {
    Box boxes[6];
    size_t offsets[7];
};
struct HaloLayout {
    size_t offsets[7];
};

// Ghost faces are needed only across MPI interfaces. At global faces the
// stencil clamps to its center, exactly as in the original implementation.
// Keep the double-precision arithmetic and expression order unchanged.
template<bool Chemical>
__device__ __forceinline__ void stencilCell(const double* __restrict__ input,
                                           const double* __restrict__ cold,
                                           double* __restrict__ output,
                                           Geometry g, int x, int y, int z) {
    const size_t i = static_cast<size_t>(z) * g.slice + static_cast<size_t>(y) * g.pitch + x;
    const size_t xm = (x == 1 && g.physical[0]) ? i : i - 1;
    const size_t xp = (x == g.nx && g.physical[1]) ? i : i + 1;
    const size_t ym = (y == 1 && g.physical[2]) ? i : i - g.pitch;
    const size_t yp = (y == g.ny && g.physical[3]) ? i : i + g.pitch;
    const size_t zm = (z == 1 && g.physical[4]) ? i : i - g.slice;
    const size_t zp = (z == g.nz && g.physical[5]) ? i : i + g.slice;
    const double cv = input[i];
    const double cxx = input[xp] + input[xm] - 2.0 * cv;
    const double cyy = input[yp] + input[ym] - 2.0 * cv;
    const double czz = input[zp] + input[zm] - 2.0 * cv;
    const double laplacian = cxx + cyy + czz;
    if constexpr (Chemical) {
        constexpr double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
        output[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
                    + 3.0 * cv + cv * cv * cv - 0.5 * laplacian;
    } else {
        output[i] = cold[i] + 0.01 * laplacian;
    }
}

template<bool Chemical>
__global__ void stencilBox(const double* __restrict__ input, const double* __restrict__ cold,
                           double* __restrict__ output, Geometry g, Box box) {
    // X is contiguous; a warp processes 32 neighboring cells. Grid-stride loops
    // also support grids larger than CUDA's Y/Z grid-dimension limits.
    for (size_t z = box.lo[2] + blockIdx.z * static_cast<size_t>(blockDim.z) + threadIdx.z;
         z < static_cast<size_t>(box.hi[2]); z += static_cast<size_t>(gridDim.z) * blockDim.z)
        for (size_t y = box.lo[1] + blockIdx.y * static_cast<size_t>(blockDim.y) + threadIdx.y;
             y < static_cast<size_t>(box.hi[1]); y += static_cast<size_t>(gridDim.y) * blockDim.y)
            for (size_t x = box.lo[0] + blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
                 x < static_cast<size_t>(box.hi[0]); x += static_cast<size_t>(gridDim.x) * blockDim.x)
                stencilCell<Chemical>(input, cold, output, g, static_cast<int>(x),
                                      static_cast<int>(y), static_cast<int>(z));
}

// A single launch handles the six disjoint interface regions, including
// subdomains only one or two cells wide, without updating any cell twice.
template<bool Chemical>
__global__ void stencilShell(const double* __restrict__ input, const double* __restrict__ cold,
                             double* __restrict__ output, Geometry g, Shell shell) {
    for (size_t t = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         t < shell.offsets[6]; t += static_cast<size_t>(gridDim.x) * blockDim.x) {
        int face = 0;
        while (t >= shell.offsets[face + 1]) ++face;
        const Box b = shell.boxes[face];
        const size_t v = t - shell.offsets[face];
        const size_t width = b.hi[0] - b.lo[0], height = b.hi[1] - b.lo[1];
        const int x = b.lo[0] + static_cast<int>(v % width);
        const int y = b.lo[1] + static_cast<int>((v / width) % height);
        const int z = b.lo[2] + static_cast<int>(v / (width * height));
        stencilCell<Chemical>(input, cold, output, g, x, y, z);
    }
}

template<bool Pack>
__global__ void transferFaces(double* __restrict__ field, double* __restrict__ buffer,
                              Geometry g, HaloLayout layout) {
    for (size_t t = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         t < layout.offsets[6]; t += static_cast<size_t>(gridDim.x) * blockDim.x) {
        int face = 0;
        while (t >= layout.offsets[face + 1]) ++face;
        const size_t v = t - layout.offsets[face];
        size_t x, y, z;
        if (face < 2) {
            x = (face == 0) ? (Pack ? 1 : 0) : (Pack ? g.nx : g.nx + 1);
            y = v % g.ny + 1; z = v / g.ny + 1;
        } else if (face < 4) {
            x = v % g.nx + 1; z = v / g.nx + 1;
            y = (face == 2) ? (Pack ? 1 : 0) : (Pack ? g.ny : g.ny + 1);
        } else {
            x = v % g.nx + 1; y = v / g.nx + 1;
            z = (face == 4) ? (Pack ? 1 : 0) : (Pack ? g.nz : g.nz + 1);
        }
        const size_t i = z * g.slice + y * g.pitch + x;
        if constexpr (Pack) buffer[t] = field[i];
        else field[i] = buffer[t];
    }
}

static unsigned int linearBlocks(size_t count) {
    return static_cast<unsigned int>(std::min<size_t>((count + 255) / 256, 65535));
}
template<bool Chemical>
static void launchBox(const double* input, const double* cold, double* output,
                      Geometry g, Box box, cudaStream_t stream) {
    if (box.volume == 0) return;
    const dim3 block(32, 4, 2);
    const dim3 grid(static_cast<unsigned int>((static_cast<size_t>(box.hi[0] - box.lo[0]) + 31) / 32),
                    static_cast<unsigned int>(std::min<size_t>((static_cast<size_t>(box.hi[1] - box.lo[1]) + 3) / 4, 65535)),
                    static_cast<unsigned int>(std::min<size_t>((static_cast<size_t>(box.hi[2] - box.lo[2]) + 1) / 2, 65535)));
    stencilBox<Chemical><<<grid, block, 0, stream>>>(input, cold, output, g, box);
    CUDA_CHECK(cudaGetLastError());
}

// Use device buffers when MPI reports CUDA support. Otherwise stage
// asynchronously through pinned memory, without sending GPU pointers to MPI.
class Solver {
public:
    Geometry g{};
    size_t owned = 0;
    double *cold = nullptr, *next = nullptr, *mu = nullptr;
    cudaStream_t compute = nullptr;

    Solver(const Options& o, MPI_Comm communicator, const int dims[3], const int coords[3], bool directMPI)
        : comm(communicator), deviceMPI(directMPI) {
        const size_t global[3] = {o.nx, o.ny, o.nz};
        int sizes[3];
        for (int axis = 0; axis < 3; ++axis) {
            sizes[axis] = localExtent(global[axis], dims[axis], coords[axis]);
            origin[axis] = localOffset(global[axis], dims[axis], coords[axis]);
            MPI_CHECK(MPI_Cart_shift(comm, axis, 1, &neighbors[2 * axis], &neighbors[2 * axis + 1]));
        }
        g.nx = sizes[0]; g.ny = sizes[1]; g.nz = sizes[2];
        g.pitch = static_cast<size_t>(g.nx) + 2;
        g.slice = g.pitch * (static_cast<size_t>(g.ny) + 2);
        const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(double);
        if (g.pitch > maxElements / (static_cast<size_t>(g.ny) + 2) ||
            g.slice > maxElements / (static_cast<size_t>(g.nz) + 2)) {
            throw std::runtime_error("Local padded grid is too large");
        }
        const size_t padded = g.slice * (static_cast<size_t>(g.nz) + 2);
        owned = static_cast<size_t>(g.nx) * g.ny * g.nz;
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cold), padded * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&next), padded * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mu), padded * sizeof(double)));

        std::vector<double> initial(owned);
        const size_t volume = o.nx * o.ny * o.nz;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int z = 0; z < g.nz; ++z) {
            for (int y = 0; y < g.ny; ++y) {
                const size_t globalRow = (origin[2] + z) * o.nx * o.ny + (origin[1] + y) * o.nx + origin[0];
                const size_t localRow = (static_cast<size_t>(z) * g.ny + y) * g.nx;
                for (int x = 0; x < g.nx; ++x) {
                    const size_t id = globalRow + x;
                    const double pseudo = (((id + 1) * 1299709) % volume) / static_cast<double>(volume);
                    initial[localRow + x] = -1.0 + 2.0 * pseudo;
                }
            }
        }
        copyOwned(initial.data(), cold, cudaMemcpyHostToDevice);

        full = makeBox(1, g.nx + 1, 1, g.ny + 1, 1, g.nz + 1);
        int low[3], high[3];
        for (int axis = 0; axis < 3; ++axis) {
            g.physical[2 * axis] = neighbors[2 * axis] == MPI_PROC_NULL;
            g.physical[2 * axis + 1] = neighbors[2 * axis + 1] == MPI_PROC_NULL;
            low[axis] = g.physical[2 * axis] ? 1 : 2;
            high[axis] = std::max(low[axis], g.physical[2 * axis + 1] ? sizes[axis] + 1 : sizes[axis]);
        }
        core = makeBox(low[0], high[0], low[1], high[1], low[2], high[2]);
        shell.boxes[0] = makeBox(1, low[0], 1, g.ny + 1, 1, g.nz + 1);
        shell.boxes[1] = makeBox(high[0], g.nx + 1, 1, g.ny + 1, 1, g.nz + 1);
        shell.boxes[2] = makeBox(low[0], high[0], 1, low[1], 1, g.nz + 1);
        shell.boxes[3] = makeBox(low[0], high[0], high[1], g.ny + 1, 1, g.nz + 1);
        shell.boxes[4] = makeBox(low[0], high[0], low[1], high[1], 1, low[2]);
        shell.boxes[5] = makeBox(low[0], high[0], low[1], high[1], high[2], g.nz + 1);
        for (int face = 0; face < 6; ++face) {
            shell.offsets[face + 1] = shell.offsets[face] + shell.boxes[face].volume;
            const size_t area = face < 2 ? static_cast<size_t>(g.ny) * g.nz :
                                face < 4 ? static_cast<size_t>(g.nx) * g.nz :
                                           static_cast<size_t>(g.nx) * g.ny;
            layout.offsets[face + 1] = layout.offsets[face] + (neighbors[face] == MPI_PROC_NULL ? 0 : area);
        }
        if (layout.offsets[6] > 0) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&fieldReady, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&packed, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&received, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(fieldReady, compute));
            const size_t bytes = layout.offsets[6] * sizeof(double);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceSend), bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceRecv), bytes));
            if (!deviceMPI) {
                CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostSend), bytes));
                CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostRecv), bytes));
            }
            // Derived types avoid MPI's int count limit for large faces.
            for (int axis = 0; axis < 3; ++axis) {
                const int shape[2] = {sizes[(axis + 1) % 3], sizes[(axis + 2) % 3]};
                const int zero[2] = {0, 0};
                MPI_CHECK(MPI_Type_create_subarray(2, shape, shape, zero, MPI_ORDER_C, MPI_DOUBLE, &faceTypes[axis]));
                MPI_CHECK(MPI_Type_commit(&faceTypes[axis]));
            }
        } else if (o.iterations >= 2) {
            // With no MPI interfaces, replay two complete steps per CUDA graph
            // launch. The two swaps return to the original buffer arrangement.
            CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
            launchBox<true>(cold, cold, mu, g, full, compute);
            launchBox<false>(mu, cold, next, g, full, compute);
            launchBox<true>(next, next, mu, g, full, compute);
            launchBox<false>(mu, next, cold, g, full, compute);
            CUDA_CHECK(cudaStreamEndCapture(compute, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        }
    }

    ~Solver() {
        if (graphExec) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
        if (exchange) {
            CUDA_CHECK(cudaStreamDestroy(exchange));
            CUDA_CHECK(cudaEventDestroy(fieldReady));
            CUDA_CHECK(cudaEventDestroy(packed));
            CUDA_CHECK(cudaEventDestroy(received));
            CUDA_CHECK(cudaFree(deviceSend)); CUDA_CHECK(cudaFree(deviceRecv));
            if (!deviceMPI) {
                CUDA_CHECK(cudaFreeHost(hostSend)); CUDA_CHECK(cudaFreeHost(hostRecv));
            }
            for (auto& type : faceTypes) MPI_CHECK(MPI_Type_free(&type));
        }
        CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(next)); CUDA_CHECK(cudaFree(mu));
        CUDA_CHECK(cudaStreamDestroy(compute));
    }

    void run(int iterations) {
        if (graphExec) {
            for (int t = 0; t < iterations / 2; ++t) CUDA_CHECK(cudaGraphLaunch(graphExec, compute));
            if (iterations % 2) {
                launchBox<true>(cold, cold, mu, g, full, compute);
                launchBox<false>(mu, cold, next, g, full, compute);
                std::swap(cold, next);
            }
        } else {
            for (int t = 0; t < iterations; ++t) {
                phase<true>(cold, cold, mu, 100);
                phase<false>(mu, cold, next, 200);
                std::swap(cold, next);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }

    std::vector<double> download() {
        std::vector<double> local(owned);
        copyOwned(local.data(), cold, cudaMemcpyDeviceToHost);
        return local;
    }

private:
    MPI_Comm comm;
    size_t origin[3]{};
    int neighbors[6]{};
    MPI_Datatype faceTypes[3]{MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    Box full{}, core{};
    Shell shell{};
    HaloLayout layout{};
    cudaStream_t exchange = nullptr;
    cudaEvent_t fieldReady = nullptr, packed = nullptr, received = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    double *deviceSend = nullptr, *deviceRecv = nullptr, *hostSend = nullptr, *hostRecv = nullptr;
    bool receiveBufferUsed = false;
    bool deviceMPI;

    void copyOwned(double* host, double* device, cudaMemcpyKind direction) {
        cudaMemcpy3DParms copy{};
        const auto cpu = make_cudaPitchedPtr(host, static_cast<size_t>(g.nx) * sizeof(double),
                                           static_cast<size_t>(g.nx) * sizeof(double), g.ny);
        const auto gpu = make_cudaPitchedPtr(device, g.pitch * sizeof(double), g.pitch * sizeof(double), g.ny + 2);
        const cudaPos ghostOffset = make_cudaPos(sizeof(double), 1, 1);
        if (direction == cudaMemcpyHostToDevice) {
            copy.srcPtr = cpu; copy.dstPtr = gpu; copy.dstPos = ghostOffset;
        } else {
            copy.srcPtr = gpu; copy.srcPos = ghostOffset; copy.dstPtr = cpu;
        }
        copy.extent = make_cudaExtent(static_cast<size_t>(g.nx) * sizeof(double), g.ny, g.nz);
        copy.kind = direction;
        CUDA_CHECK(cudaMemcpy3D(&copy));
    }

    template<bool Chemical>
    void phase(double* input, const double* concentration, double* output, int tag) {
        if (!exchange) {
            launchBox<Chemical>(input, concentration, output, g, full, compute);
            return;
        }
        // Finish the previous H2D transfer/unpack before MPI may reuse receive
        // buffers. Send completion is waited below before reusing send buffers.
        if (receiveBufferUsed) CUDA_CHECK(cudaEventSynchronize(received));
        MPI_Request receives[6], sends[6];
        int nrecv = 0, nsend = 0;
        for (int face = 0; face < 6; ++face) {
            if (neighbors[face] != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Irecv((deviceMPI ? deviceRecv : hostRecv) + layout.offsets[face], 1, faceTypes[face / 2],
                                    neighbors[face], tag + (face ^ 1), comm, &receives[nrecv++]));
            }
        }
        CUDA_CHECK(cudaStreamWaitEvent(exchange, fieldReady, 0));
        transferFaces<true><<<linearBlocks(layout.offsets[6]), 256, 0, exchange>>>(input, deviceSend, g, layout);
        CUDA_CHECK(cudaGetLastError());
        const size_t bytes = layout.offsets[6] * sizeof(double);
        if (!deviceMPI) CUDA_CHECK(cudaMemcpyAsync(hostSend, deviceSend, bytes, cudaMemcpyDeviceToHost, exchange));
        CUDA_CHECK(cudaEventRecord(packed, exchange));
        launchBox<Chemical>(input, concentration, output, g, core, compute);
        CUDA_CHECK(cudaEventSynchronize(packed));
        for (int face = 0; face < 6; ++face) {
            if (neighbors[face] != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Isend((deviceMPI ? deviceSend : hostSend) + layout.offsets[face], 1, faceTypes[face / 2],
                                    neighbors[face], tag + face, comm, &sends[nsend++]));
            }
        }
        // MPI progresses on the host while the independent GPU interior runs.
        MPI_CHECK(MPI_Waitall(nrecv, receives, MPI_STATUSES_IGNORE));
        if (!deviceMPI) CUDA_CHECK(cudaMemcpyAsync(deviceRecv, hostRecv, bytes, cudaMemcpyHostToDevice, exchange));
        transferFaces<false><<<linearBlocks(layout.offsets[6]), 256, 0, exchange>>>(input, deviceRecv, g, layout);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(received, exchange));
        receiveBufferUsed = true;
        CUDA_CHECK(cudaStreamWaitEvent(compute, received, 0));
        stencilShell<Chemical><<<linearBlocks(shell.offsets[6]), 256, 0, compute>>>(input, concentration, output, g, shell);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(fieldReady, compute));
        MPI_CHECK(MPI_Waitall(nsend, sends, MPI_STATUSES_IGNORE));
    }
};

// Gather only for -r. Receive Cartesian blocks directly into their positions in
// the original global X-fastest ordering, preserving Kahan sum and hash order.
static void printDistributedResults(const std::vector<double>& local, const Options& o,
                                    MPI_Comm comm, int rank, int ranks, const int dims[3]) {
    if (rank == 0) {
        std::vector<double> global(o.nx * o.ny * o.nz);
        std::vector<MPI_Request> requests(ranks - 1);
        const int shape[3] = {static_cast<int>(o.nz), static_cast<int>(o.ny), static_cast<int>(o.nx)};
        for (int source = 0; source < ranks; ++source) {
            int coords[3], sub[3], start[3];
            MPI_CHECK(MPI_Cart_coords(comm, source, 3, coords));
            const size_t extents[3] = {o.nx, o.ny, o.nz};
            for (int axis = 0; axis < 3; ++axis) {
                sub[2 - axis] = localExtent(extents[axis], dims[axis], coords[axis]);
                start[2 - axis] = static_cast<int>(localOffset(extents[axis], dims[axis], coords[axis]));
            }
            if (source == 0) {
                #pragma omp parallel for collapse(2) schedule(static)
                for (int z = 0; z < sub[0]; ++z) {
                    for (int y = 0; y < sub[1]; ++y) {
                        const size_t dst = (static_cast<size_t>(z + start[0]) * o.ny + y + start[1]) * o.nx + start[2];
                        const size_t src = (static_cast<size_t>(z) * sub[1] + y) * sub[2];
                        std::copy_n(local.data() + src, sub[2], global.data() + dst);
                    }
                }
            } else {
                MPI_Datatype block;
                MPI_CHECK(MPI_Type_create_subarray(3, shape, sub, start, MPI_ORDER_C, MPI_DOUBLE, &block));
                MPI_CHECK(MPI_Type_commit(&block));
                MPI_CHECK(MPI_Irecv(global.data(), 1, block, source, 300, comm, &requests[source - 1]));
                // MPI retains the datatype until the outstanding receive ends.
                MPI_CHECK(MPI_Type_free(&block));
            }
        }
        MPI_CHECK(MPI_Waitall(ranks - 1, requests.data(), MPI_STATUSES_IGNORE));
        print_results(global, "Concentration");
    } else {
        int coords[3];
        MPI_CHECK(MPI_Cart_coords(comm, rank, 3, coords));
        const int shape[3] = {localExtent(o.nz, dims[2], coords[2]),
                              localExtent(o.ny, dims[1], coords[1]),
                              localExtent(o.nx, dims[0], coords[0])};
        const int zero[3] = {0, 0, 0};
        MPI_Datatype block;
        MPI_CHECK(MPI_Type_create_subarray(3, shape, shape, zero, MPI_ORDER_C, MPI_DOUBLE, &block));
        MPI_CHECK(MPI_Type_commit(&block));
        MPI_CHECK(MPI_Send(local.data(), 1, block, 0, 300, comm));
        MPI_CHECK(MPI_Type_free(&block));
    }
}

static int validateDistributed(const std::vector<double>& local, MPI_Comm comm, int rank) {
    double low = std::numeric_limits<double>::infinity();
    double high = -std::numeric_limits<double>::infinity();
    int bad = 0;
    #pragma omp parallel for reduction(min:low) reduction(max:high) reduction(|:bad) schedule(static)
    for (size_t i = 0; i < local.size(); ++i) {
        const double value = local[i];
        bad |= !std::isfinite(value);
        low = std::min(low, value);
        high = std::max(high, value);
    }
    double globalLow = 0.0, globalHigh = 0.0;
    int globalBad = 0;
    MPI_CHECK(MPI_Reduce(&low, &globalLow, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
    MPI_CHECK(MPI_Reduce(&high, &globalHigh, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    MPI_CHECK(MPI_Reduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, 0, comm));
    int status = 0;
    if (rank == 0) {
        std::printf("Validating result...\n");
        if (globalBad) {
            std::printf("Validation failed: found NaN or Inf value\n");
            status = 1;
        } else {
            std::printf("Concentration range: [%.6f, %.6f]\n", globalLow, globalHigh);
            if (globalHigh > 10.0 || globalLow < -10.0) {
                std::printf("Validation failed: values out of expected range\n");
                status = 1;
            }
        }
        std::printf("Validation: %s\n", status ? "FAILED" : "PASSED");
    }
    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int worldRank = 0, worldSize = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (provided < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) std::fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    Options o;
    const int parsed = parseOptions(argc, argv, o, worldRank);
    if (parsed) { MPI_CHECK(MPI_Finalize()); return parsed == 2 ? 0 : 1; }

    int status = 0;
    try {
        int dims[3] = {1, 1, 1};
        const int active = chooseDecomposition(o, worldSize, dims);
        MPI_Comm work = MPI_COMM_NULL;
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &work));
        if (work != MPI_COMM_NULL) {
            const int periods[3] = {0, 0, 0};
            MPI_Comm cart = MPI_COMM_NULL, node = MPI_COMM_NULL;
            MPI_CHECK(MPI_Cart_create(work, 3, dims, periods, 0, &cart));
            MPI_CHECK(MPI_Comm_set_errhandler(cart, MPI_ERRORS_RETURN));
            MPI_CHECK(MPI_Comm_free(&work));
            int rank = 0, coords[3], nodeRank = 0, nodeSize = 0;
            MPI_CHECK(MPI_Comm_rank(cart, &rank));
            MPI_CHECK(MPI_Cart_coords(cart, rank, 3, coords));
            MPI_CHECK(MPI_Comm_split_type(cart, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
            MPI_CHECK(MPI_Comm_rank(node, &nodeRank));
            MPI_CHECK(MPI_Comm_size(node, &nodeSize));
            if (std::getenv("OMP_NUM_THREADS") == nullptr)
                omp_set_num_threads(std::max(1, omp_get_num_procs() / nodeSize));
            int devices = 0;
            CUDA_CHECK(cudaGetDeviceCount(&devices));
            if (devices == 0) throw std::runtime_error("A CUDA device is required on every active MPI rank");
            // CUDA_VISIBLE_DEVICES can expose either one assigned GPU per rank
            // or the whole node; both conventions work with this mapping.
            CUDA_CHECK(cudaSetDevice(nodeRank % devices));
            int directMPI = 0;
#ifdef CH_MPI_CUDA_QUERY
            directMPI = MPIX_Query_cuda_support() != 0;
#endif
            // All peers must agree on their buffer protocol.
            MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &directMPI, 1, MPI_INT, MPI_MIN, cart));
            MPI_CHECK(MPI_Comm_free(&node));
            if (rank == 0) {
                std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
                std::printf("Grid size: %zu x %zu x %zu\n", o.nx, o.ny, o.nz);
                std::printf("Time steps: %d\n", o.iterations);
                std::printf("Validation: %s\n", o.validate ? "enabled" : "disabled");
                std::printf("Parallelism: %d MPI ranks (%d x %d x %d), OpenMP, CUDA\n", active, dims[0], dims[1], dims[2]);
                std::printf("Initializing concentration field...\n");
            }
            {
                Solver solver(o, cart, dims, coords, directMPI != 0);
                if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
                MPI_CHECK(MPI_Barrier(cart));
                const double start = MPI_Wtime();
                solver.run(o.iterations);
                const double elapsed = MPI_Wtime() - start;
                double seconds = 0.0;
                MPI_CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart));
                if (rank == 0) {
                    std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
                    const double updates = static_cast<double>(o.nx * o.ny * o.nz) * o.iterations;
                    std::printf("Performance: %.3f MCellUpdates/s\n", seconds > 0.0 ? updates / seconds / 1e6 : 0.0);
                }
                if (o.results || o.validate) {
                    const auto local = solver.download();
                    if (o.results) printDistributedResults(local, o, cart, rank, active, dims);
                    if (o.validate) status = validateDistributed(local, cart, rank);
                }
            }
            MPI_CHECK(MPI_Comm_free(&cart));
        }
        // Idle ranks take part in the final status broadcast and finalization.
        MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_CHECK(MPI_Finalize());
    return status;
}
