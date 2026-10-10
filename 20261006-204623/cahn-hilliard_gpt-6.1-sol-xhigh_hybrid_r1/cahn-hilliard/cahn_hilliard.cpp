#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#ifdef OMPI_MAJOR_VERSION
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

int worldRank = 0;

void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        throw std::runtime_error(std::string(message, length));
    }
}

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

size_t checkedProduct(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::runtime_error("Grid size exceeds the addressable memory range");
    return a * b;
}

struct Options {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
};

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

size_t parseNumber(const char* text) {
    if (!*text || *text == '-') throw std::runtime_error("Expected a nonnegative integer");
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno || *end || value > std::numeric_limits<size_t>::max())
        throw std::runtime_error("Invalid integer argument");
    return static_cast<size_t>(value);
}

// All stencil arithmetic remains double precision, with the original evaluation
// order and clamping at the physical (not MPI subdomain) boundaries. Each thread
// walks a short column, reusing the preceding/current/following z values.
template <bool Chemical>
__global__ void stencil(const double* __restrict__ field,
                        const double* __restrict__ concentration,
                        double* __restrict__ result,
                        size_t nx, size_t ny, size_t localZ,
                        size_t firstZ, size_t depth, bool lowerWall, bool upperWall) {
    constexpr size_t zTile = 4;
    const size_t plane = nx * ny;
    for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
         y < ny; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
        for (size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             x < nx; x += static_cast<size_t>(gridDim.x) * blockDim.x) {
            const size_t row = y * nx + x;
            const size_t xm = x ? 1 : 0;
            const size_t xp = x + 1 < nx ? 1 : 0;
            const size_t ym = y ? nx : 0;
            const size_t yp = y + 1 < ny ? nx : 0;
            for (size_t tile = static_cast<size_t>(blockIdx.z) * zTile;
                 tile < depth; tile += static_cast<size_t>(gridDim.z) * zTile) {
                const size_t begin = firstZ + tile;
                const size_t end = firstZ + (tile + zTile < depth ? tile + zTile : depth);
                size_t index = begin * plane + row;
                double center = field[index];
                double below = field[index - ((lowerWall && begin == 1) ? 0 : plane)];
                for (size_t z = begin; z < end; ++z, index += plane) {
                    const double above = field[index + ((upperWall && z == localZ) ? 0 : plane)];
                    const double cxx = field[index + xp] + field[index - xm] - 2.0 * center;
                    const double cyy = field[index + yp] + field[index - ym] - 2.0 * center;
                    const double czz = above + below - 2.0 * center;
                    const double laplacian = cxx + cyy + czz;
                    if constexpr (Chemical) {
                        constexpr double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
                        result[index] = 4.5 * ((center + 1.0) * eAA + (center - 1.0) * eBB
                                             - 2.0 * center * eAB)
                                      + 3.0 * center + center * center * center - 0.5 * laplacian;
                    } else {
                        result[index] = concentration[index] + 0.01 * laplacian;
                    }
                    below = center;
                    center = above;
                }
            }
        }
    }
}

template <bool Chemical>
void launchStencil(const double* field, const double* concentration, double* result,
                   const Options& o, size_t localZ, size_t firstZ, size_t depth,
                   bool lowerWall, bool upperWall, cudaStream_t stream) {
    if (!depth) return;
    const dim3 block(32, 4);
    const dim3 grid(static_cast<unsigned>(std::min<size_t>((o.nx - 1) / block.x + 1, 65535)),
                    static_cast<unsigned>(std::min<size_t>((o.ny - 1) / block.y + 1, 65535)),
                    static_cast<unsigned>(std::min<size_t>((depth - 1) / 4 + 1, 65535)));
    stencil<Chemical><<<grid, block, 0, stream>>>(field, concentration, result,
        o.nx, o.ny, localZ, firstZ, depth, lowerWall, upperWall);
    cudaCheck(cudaGetLastError());
}

struct DeviceBuffer {
    double* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), checkedProduct(count, sizeof(double))));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Execution {
    cudaStream_t interior = nullptr, boundary = nullptr;
    cudaEvent_t ready = nullptr, edgesDone = nullptr, sendsReady = nullptr, receivesConsumed = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executableGraph = nullptr;
    Execution() {
        cudaCheck(cudaStreamCreateWithFlags(&interior, cudaStreamNonBlocking));
        cudaCheck(cudaStreamCreateWithFlags(&boundary, cudaStreamNonBlocking));
        cudaCheck(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        cudaCheck(cudaEventCreateWithFlags(&edgesDone, cudaEventDisableTiming));
        cudaCheck(cudaEventCreateWithFlags(&sendsReady, cudaEventDisableTiming));
        cudaCheck(cudaEventCreateWithFlags(&receivesConsumed, cudaEventDisableTiming));
        cudaCheck(cudaEventRecord(ready, interior));
        cudaCheck(cudaEventRecord(receivesConsumed, boundary));
    }
    ~Execution() {
        if (executableGraph) cudaGraphExecDestroy(executableGraph);
        if (graph) cudaGraphDestroy(graph);
        cudaEventDestroy(receivesConsumed);
        cudaEventDestroy(sendsReady);
        cudaEventDestroy(edgesDone);
        cudaEventDestroy(ready);
        cudaStreamDestroy(boundary);
        cudaStreamDestroy(interior);
    }
};

bool cudaAwareMPI() {
    // Open MPI advertises this at runtime; other implementations use portable
    // pinned-host staging unless the installation explicitly opts into GPU MPI.
    const char* setting = std::getenv("CAHN_HILLIARD_CUDA_AWARE_MPI");
    if (setting) return std::strcmp(setting, "1") == 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

// One reusable pinned send/receive buffer per rank. MPI calls are confined to
// the main thread (MPI_THREAD_FUNNELED). Receives are posted before the device
// copies; MPI progresses while the GPU computes the independent interior.
struct HaloExchange {
    MPI_Comm comm;
    int lower, upper;
    size_t plane, localZ;
    bool direct;
    double* host = nullptr;
    std::vector<MPI_Request> requests;

    HaloExchange(MPI_Comm communicator, int rank, int ranks, size_t planeSize, size_t zSize)
        : comm(communicator), lower(rank ? rank - 1 : MPI_PROC_NULL),
          upper(rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL),
          plane(planeSize), localZ(zSize), direct(cudaAwareMPI()) {
        if (!direct && ranks > 1)
            cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&host),
                                    checkedProduct(checkedProduct(plane, 4), sizeof(double))));
        requests.reserve(4 * ((plane - 1) / INT_MAX + 1));
    }
    ~HaloExchange() { if (host) cudaFreeHost(host); }

    void post(double* buffer, int peer, int tag, bool send) {
        if (peer == MPI_PROC_NULL) return;
        for (size_t offset = 0; offset < plane;) {
            const int count = static_cast<int>(std::min<size_t>(plane - offset, INT_MAX));
            requests.push_back(MPI_REQUEST_NULL);
            if (send)
                mpiCheck(MPI_Isend(buffer + offset, count, MPI_DOUBLE, peer, tag, comm, &requests.back()));
            else
                mpiCheck(MPI_Irecv(buffer + offset, count, MPI_DOUBLE, peer, tag, comm, &requests.back()));
            offset += count;
        }
    }

    void begin(double* field, Execution& execution) {
        requests.clear();
        cudaCheck(cudaStreamWaitEvent(execution.boundary, execution.ready, 0));
        if (direct) {
            // MPI must not access device storage before the producing kernels finish.
            cudaCheck(cudaEventSynchronize(execution.ready));
            post(field, lower, 1, false);
            post(field + (localZ + 1) * plane, upper, 0, false);
        } else {
            // The preceding pass may still be consuming this pinned receive
            // buffer on the GPU copy engine. Finish those copies before MPI
            // is allowed to overwrite it; the preceding kernels can continue.
            cudaCheck(cudaEventSynchronize(execution.receivesConsumed));
            post(host + 2 * plane, lower, 1, false);
            post(host + 3 * plane, upper, 0, false);
            if (lower != MPI_PROC_NULL)
                cudaCheck(cudaMemcpyAsync(host, field + plane, plane * sizeof(double),
                                           cudaMemcpyDeviceToHost, execution.boundary));
            if (upper != MPI_PROC_NULL)
                cudaCheck(cudaMemcpyAsync(host + plane, field + localZ * plane, plane * sizeof(double),
                                           cudaMemcpyDeviceToHost, execution.boundary));
            cudaCheck(cudaEventRecord(execution.sendsReady, execution.boundary));
        }
    }

    void finish(double* field, Execution& execution) {
        if (!direct) cudaCheck(cudaEventSynchronize(execution.sendsReady));
        post(direct ? field + plane : host, lower, 0, true);
        post(direct ? field + localZ * plane : host + plane, upper, 1, true);
        mpiCheck(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
        if (!direct) {
            if (lower != MPI_PROC_NULL)
                cudaCheck(cudaMemcpyAsync(field, host + 2 * plane, plane * sizeof(double),
                                           cudaMemcpyHostToDevice, execution.boundary));
            if (upper != MPI_PROC_NULL)
                cudaCheck(cudaMemcpyAsync(field + (localZ + 1) * plane, host + 3 * plane,
                                           plane * sizeof(double), cudaMemcpyHostToDevice, execution.boundary));
            cudaCheck(cudaEventRecord(execution.receivesConsumed, execution.boundary));
        }
    }
};

template <bool Chemical>
void distributedPass(double* field, const double* concentration, double* result,
                     const Options& o, size_t localZ, bool lowerWall, bool upperWall,
                     HaloExchange& halo, Execution& execution) {
    halo.begin(field, execution);
    if (localZ > 2)
        launchStencil<Chemical>(field, concentration, result, o, localZ, 2, localZ - 2,
                                lowerWall, upperWall, execution.interior);
    halo.finish(field, execution);
    launchStencil<Chemical>(field, concentration, result, o, localZ, 1, 1,
                            lowerWall, upperWall, execution.boundary);
    if (localZ > 1)
        launchStencil<Chemical>(field, concentration, result, o, localZ, localZ, 1,
                                lowerWall, upperWall, execution.boundary);
    cudaCheck(cudaEventRecord(execution.edgesDone, execution.boundary));
    cudaCheck(cudaStreamWaitEvent(execution.interior, execution.edgesDone, 0));
    cudaCheck(cudaEventRecord(execution.ready, execution.interior));
}

void singleRankSimulation(double* cold, double* cnew, double* mu, const Options& o, Execution& execution) {
    // Capture two steps so graph replays need no pointer patches and leave the
    // final concentration in cold. This avoids per-step host launch overhead.
    if (o.iterations >= 2) {
        cudaCheck(cudaStreamBeginCapture(execution.interior, cudaStreamCaptureModeThreadLocal));
        launchStencil<true>(cold, nullptr, mu, o, o.nz, 1, o.nz, true, true, execution.interior);
        launchStencil<false>(mu, cold, cnew, o, o.nz, 1, o.nz, true, true, execution.interior);
        launchStencil<true>(cnew, nullptr, mu, o, o.nz, 1, o.nz, true, true, execution.interior);
        launchStencil<false>(mu, cnew, cold, o, o.nz, 1, o.nz, true, true, execution.interior);
        cudaCheck(cudaStreamEndCapture(execution.interior, &execution.graph));
        cudaCheck(cudaGraphInstantiate(&execution.executableGraph, execution.graph, nullptr, nullptr, 0));
        cudaCheck(cudaGraphUpload(execution.executableGraph, execution.interior));
        cudaCheck(cudaStreamSynchronize(execution.interior));
    }
}

bool validateResult(const std::vector<double>& local, MPI_Comm comm, int rank) {
    int nonfinite = 0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    #pragma omp parallel for schedule(static) reduction(|:nonfinite) reduction(min:minimum) reduction(max:maximum)
    for (size_t i = 0; i < local.size(); ++i) {
        const double value = local[i];
        nonfinite |= !std::isfinite(value);
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    int globalNonfinite = 0;
    double globalMinimum = 0, globalMaximum = 0;
    mpiCheck(MPI_Allreduce(&nonfinite, &globalNonfinite, 1, MPI_INT, MPI_MAX, comm));
    mpiCheck(MPI_Allreduce(&minimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN, comm));
    mpiCheck(MPI_Allreduce(&maximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX, comm));
    if (globalNonfinite) {
        if (!rank) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (!rank) printf("Concentration range: [%.6f, %.6f]\n", globalMinimum, globalMaximum);
    if (globalMaximum > 10.0 || globalMinimum < -10.0) {
        if (!rank) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

size_t slabDepth(size_t nz, int ranks, int rank) {
    return nz / ranks + (static_cast<size_t>(rank) < nz % ranks ? 1 : 0);
}

size_t slabStart(size_t nz, int ranks, int rank) {
    return static_cast<size_t>(rank) * (nz / ranks) + std::min<size_t>(rank, nz % ranks);
}

void printGlobalResults(const std::vector<double>& local, const Options& o, MPI_Comm comm, int rank, int ranks) {
    const size_t plane = o.nx * o.ny;
    const size_t volume = plane * o.nz;
    if (ranks == 1) {
        print_results(local, "Concentration");
        return;
    }
    std::vector<double> global;
    if (!rank) global.resize(volume);
    if (volume <= INT_MAX) {
        std::vector<int> counts(ranks), offsets(ranks);
        for (int r = 0; r < ranks; ++r) {
            counts[r] = static_cast<int>(slabDepth(o.nz, ranks, r) * plane);
            offsets[r] = static_cast<int>(slabStart(o.nz, ranks, r) * plane);
        }
        mpiCheck(MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                            global.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, comm));
    } else {
        // MPI-3 counts/displacements are ints: chunk very large output instead
        // of truncating them. Gathering is only performed when -r is requested.
        if (!rank) std::copy(local.begin(), local.end(), global.begin());
        for (int r = 1; r < ranks; ++r) {
            const size_t count = slabDepth(o.nz, ranks, r) * plane;
            const size_t start = slabStart(o.nz, ranks, r) * plane;
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
                if (!rank)
                    mpiCheck(MPI_Recv(global.data() + start + offset, chunk, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE));
                else if (rank == r)
                    mpiCheck(MPI_Send(local.data() + offset, chunk, MPI_DOUBLE, 0, 2, comm));
                offset += chunk;
            }
        }
    }
    if (!rank) print_results(global, "Concentration");
}

int run(const Options& o, int worldSize) {
    // Empty slabs do no work and join the final world collective. In particular,
    // nz=1 and runs with more ranks than planes retain the original semantics.
    const int activeRanks = static_cast<int>(std::min<size_t>(worldSize, o.nz));
    MPI_Comm comm = MPI_COMM_NULL;
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &comm));
    if (comm == MPI_COMM_NULL) return 0;
    int rank = 0;
    mpiCheck(MPI_Comm_rank(comm, &rank));
    mpiCheck(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN));

    MPI_Comm nodeComm = MPI_COMM_NULL;
    mpiCheck(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int nodeRank = 0;
    mpiCheck(MPI_Comm_rank(nodeComm, &nodeRank));
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("A CUDA accelerator is required on every active MPI rank");
    // CUDA_VISIBLE_DEVICES may already restrict each process to its assigned GPU.
    cudaCheck(cudaSetDevice(nodeRank % devices));
    mpiCheck(MPI_Comm_free(&nodeComm));

    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::min(8, omp_get_num_procs()));
    const size_t plane = checkedProduct(o.nx, o.ny);
    const size_t volume = checkedProduct(plane, o.nz);
    const size_t localZ = slabDepth(o.nz, activeRanks, rank);
    const size_t startZ = slabStart(o.nz, activeRanks, rank);
    const size_t localSize = checkedProduct(plane, localZ);
    const size_t allocation = checkedProduct(plane, localZ + 2);
    DeviceBuffer first(allocation), second(allocation), potential(allocation);
    Execution execution;
    double* cold = first.data;
    double* cnew = second.data;
    std::vector<double> local(localSize);

    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", o.nx, o.ny, o.nz);
        printf("Time steps: %d\n", o.iterations);
        printf("Validation: %s\n", o.validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localSize; ++i) {
        const size_t linearId = startZ * plane + i;
        const double pseudo = (((linearId + 1) * size_t{1299709}) % volume) / static_cast<double>(volume);
        local[i] = -1.0 + 2.0 * pseudo;
    }
    cudaCheck(cudaMemcpy(cold + plane, local.data(), localSize * sizeof(double), cudaMemcpyHostToDevice));
    if (!o.validate && !o.printResults) std::vector<double>().swap(local);

    // Allocation, initialization, and CUDA graph setup are outside the timed loop.
    // There is no CPU/serial solver path: CUDA, MPI and OpenMP are required.
    // Load both kernels now, including any PTX JIT compilation under CUDA's
    // lazy module loading, so it cannot contaminate distributed timings.
    cudaFuncAttributes attributes;
    cudaCheck(cudaFuncGetAttributes(&attributes, stencil<true>));
    cudaCheck(cudaFuncGetAttributes(&attributes, stencil<false>));
    if (activeRanks == 1) singleRankSimulation(cold, cnew, potential.data, o, execution);
    HaloExchange halo(comm, rank, activeRanks, plane, localZ);
    cudaCheck(cudaDeviceSynchronize());
    if (!rank) printf("Running Cahn-Hilliard simulation...\n");
    mpiCheck(MPI_Barrier(comm));
    const double start = MPI_Wtime();
    if (activeRanks == 1) {
        for (int t = 0; t < o.iterations / 2; ++t)
            cudaCheck(cudaGraphLaunch(execution.executableGraph, execution.interior));
        if (o.iterations % 2) {
            launchStencil<true>(cold, nullptr, potential.data, o, localZ, 1, localZ, true, true, execution.interior);
            launchStencil<false>(potential.data, cold, cnew, o, localZ, 1, localZ, true, true, execution.interior);
            std::swap(cold, cnew);
        }
    } else {
        for (int t = 0; t < o.iterations; ++t) {
            distributedPass<true>(cold, nullptr, potential.data, o, localZ, rank == 0, rank == activeRanks - 1, halo, execution);
            distributedPass<false>(potential.data, cold, cnew, o, localZ, rank == 0, rank == activeRanks - 1, halo, execution);
            std::swap(cold, cnew);
        }
    }
    cudaCheck(cudaStreamSynchronize(execution.interior));
    const double elapsed = MPI_Wtime() - start;
    double slowest = 0;
    mpiCheck(MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (!rank) {
        printf("Computation time: %.0f ms\n", std::floor(slowest * 1000.0));
        const double mcups = slowest > 0 ? static_cast<double>(volume) * o.iterations / slowest / 1e6 : 0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    if (o.validate || o.printResults)
        cudaCheck(cudaMemcpy(local.data(), cold + plane, localSize * sizeof(double), cudaMemcpyDeviceToHost));
    if (o.printResults) printGlobalResults(local, o, comm, rank, activeRanks);
    int status = 0;
    if (o.validate) {
        if (!rank) printf("Validating result...\n");
        const bool valid = validateResult(local, comm, rank);
        if (!rank) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        status = valid ? 0 : 1;
    }
    mpiCheck(MPI_Comm_free(&comm));
    return status;
}

} // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int result = 0;
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED support is required");
        Options o;
        bool help = false;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "-h")) { help = true; break; }
            if (!std::strcmp(argv[i], "-v")) o.validate = true;
            else if (!std::strcmp(argv[i], "-r")) o.printResults = true;
            else if ((!std::strcmp(argv[i], "-x") || !std::strcmp(argv[i], "-y") ||
                      !std::strcmp(argv[i], "-z") || !std::strcmp(argv[i], "-i")) && i + 1 < argc) {
                const char option = argv[i][1];
                const size_t value = parseNumber(argv[++i]);
                if (option == 'x') o.nx = value;
                else if (option == 'y') o.ny = value;
                else if (option == 'z') o.nz = value;
                else {
                    if (value > INT_MAX) throw std::runtime_error("Too many time steps");
                    o.iterations = static_cast<int>(value);
                }
            } else throw std::runtime_error(std::string("Unknown or incomplete option: ") + argv[i]);
        }
        if (help) {
            if (!worldRank) printUsage(argv[0]);
        } else {
            if (!o.nx) throw std::runtime_error("Grid dimensions must be positive");
            if (!o.ny) o.ny = o.nx;
            if (!o.nz) o.nz = o.nx;
            const size_t volume = checkedProduct(checkedProduct(o.nx, o.ny), o.nz);
            checkedProduct(volume, sizeof(double));
            int worldSize = 0;
            mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
            result = run(o, worldSize);
            mpiCheck(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return result;
}
