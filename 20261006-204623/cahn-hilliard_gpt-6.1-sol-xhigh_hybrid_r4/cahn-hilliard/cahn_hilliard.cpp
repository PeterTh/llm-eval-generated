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
#ifdef HAVE_MPI_CUDA_QUERY
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#endif
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {
int worldRank = 0;

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", worldRank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

void mpiCheck(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(status, message, &length);
        fail(message);
    }
}

size_t slabBegin(int rank, int ranks, size_t nz) {
    return (nz / ranks) * rank + std::min<size_t>(rank, nz % ranks);
}

// Both device fields have two concentration halo planes on either side.
// Only one chemical-potential halo is needed; it is recomputed locally.
struct Grid {
    size_t nx, ny, nz, plane, localNz, begin;
};

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                           const Grid& g, size_t x, size_t y,
                                           long long z) {
    const size_t p = static_cast<size_t>(z + 2) * g.plane + y * g.nx + x;
    const size_t xp = x + 1 < g.nx ? p + 1 : p;
    const size_t xn = x > 0 ? p - 1 : p;
    const size_t yp = y + 1 < g.ny ? p + g.nx : p;
    const size_t yn = y > 0 ? p - g.nx : p;
    const long long globalZ = static_cast<long long>(g.begin) + z;
    const size_t zp = globalZ + 1 < static_cast<long long>(g.nz) ? p + g.plane : p;
    const size_t zn = globalZ > 0 ? p - g.plane : p;
    const double value = field[p];
    // dx = dy = dz = 1, as in the original benchmark.
    const double cxx = field[xp] + field[xn] - 2.0 * value;
    const double cyy = field[yp] + field[yn] - 2.0 * value;
    const double czz = field[zp] + field[zn] - 2.0 * value;
    return cxx + cyy + czz;
}

template<bool Boundary>
__global__ void chemicalPotential(const double* __restrict__ c,
                                  double* __restrict__ mu, Grid g,
                                  size_t first, size_t planes) {
    for (size_t j = blockIdx.z; j < planes; j += gridDim.z) {
        long long z;
        if constexpr (Boundary) {
            // {-1, 0, localNz-1, localNz}; remove the duplicate for a one-plane slab.
            z = g.localNz == 1 ? static_cast<long long>(j) - 1
                : (j < 2 ? static_cast<long long>(j) - 1
                         : static_cast<long long>(g.localNz) - 1 + static_cast<long long>(j - 2));
        } else {
            z = static_cast<long long>(first + j);
        }
        const long long outputZ = z;
        // A clamped mu ghost must equal mu at the physical boundary, including
        // that boundary cell's own clamped concentration stencil.
        if (g.begin == 0 && z < 0) z = 0;
        if (g.begin + g.localNz == g.nz && z >= static_cast<long long>(g.localNz))
            z = static_cast<long long>(g.localNz) - 1;
        for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
             y < g.ny; y += size_t(gridDim.y) * blockDim.y) {
            for (size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
                 x < g.nx; x += size_t(gridDim.x) * blockDim.x) {
                const size_t xy = y * g.nx + x;
                const double cv = c[static_cast<size_t>(z + 2) * g.plane + xy];
                constexpr double e_AA = -(2.0 / 9.0);
                constexpr double e_BB = -(2.0 / 9.0);
                constexpr double e_AB = 2.0 / 9.0;
                mu[static_cast<size_t>(outputZ + 2) * g.plane + xy] =
                    4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                    + 3.0 * cv + cv * cv * cv - 0.5 * laplacian(c, g, x, y, z);
            }
        }
    }
}

template<bool Boundary>
__global__ void updateConcentration(double* __restrict__ next,
                                    const double* __restrict__ old,
                                    const double* __restrict__ mu, Grid g,
                                    size_t first, size_t planes) {
    for (size_t j = blockIdx.z; j < planes; j += gridDim.z) {
        size_t z;
        if constexpr (Boundary) {
            z = g.localNz <= 4 ? j : (j < 2 ? j : g.localNz - 4 + j);
        } else {
            z = first + j;
        }
        for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
             y < g.ny; y += size_t(gridDim.y) * blockDim.y) {
            for (size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
                 x < g.nx; x += size_t(gridDim.x) * blockDim.x) {
                const size_t p = (z + 2) * g.plane + y * g.nx + x;
                next[p] = old[p] + 0.01 * laplacian(mu, g, x, y, static_cast<long long>(z));
            }
        }
    }
}

struct DeviceFields {
    double* c[2] = {nullptr, nullptr};
    double* mu = nullptr;
    cudaStream_t compute = nullptr, boundary = nullptr;
    cudaEvent_t boundaryReady = nullptr, muReady = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    dim3 block, gridXY;

    explicit DeviceFields(const Grid& g) {
        const size_t bytes = (g.localNz + 4) * g.plane * sizeof(double);
        for (auto& field : c) cudaCheck(cudaMalloc(&field, bytes));
        cudaCheck(cudaMalloc(&mu, bytes));
        cudaCheck(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        cudaCheck(cudaStreamCreateWithFlags(&boundary, cudaStreamNonBlocking));
        cudaCheck(cudaEventCreateWithFlags(&boundaryReady, cudaEventDisableTiming));
        cudaCheck(cudaEventCreateWithFlags(&muReady, cudaEventDisableTiming));
        unsigned bx = 1;
        while (bx < 32 && bx < g.nx) bx *= 2;
        block = dim3(bx, 128 / bx, 1);
        gridXY = dim3(static_cast<unsigned>(std::min<size_t>((g.nx - 1) / block.x + 1, INT_MAX)),
                      static_cast<unsigned>(std::min<size_t>((g.ny - 1) / block.y + 1, 65535)), 1);
    }

    dim3 launchGrid(size_t planes) const {
        return dim3(gridXY.x, gridXY.y, static_cast<unsigned>(std::min<size_t>(planes, 65535)));
    }

    void fullStep(const Grid& g, int parity) {
        const dim3 launch = launchGrid(g.localNz);
        chemicalPotential<false><<<launch, block, 0, compute>>>(c[parity], mu, g, 0, g.localNz);
        updateConcentration<false><<<launch, block, 0, compute>>>(c[1 - parity], c[parity], mu, g, 0, g.localNz);
    }

    ~DeviceFields() {
        if (graphExec) cudaGraphExecDestroy(graphExec);
        if (graph) cudaGraphDestroy(graph);
        cudaEventDestroy(boundaryReady);
        cudaEventDestroy(muReady);
        cudaStreamDestroy(compute);
        cudaStreamDestroy(boundary);
        cudaFree(mu);
        for (auto field : c) cudaFree(field);
    }
};

struct Transfer {
    int peer;
    size_t deviceOffset, count, hostOffset;
};

// The depth-two halo can span two ranks when slabs contain only one plane.
// Precompute intersections rather than assuming both planes have one owner.
struct HaloExchange {
    std::vector<Transfer> sends, receives;
    std::vector<MPI_Request> requests[2];
    double* sendHost = nullptr;
    double* receiveHost = nullptr;
    size_t receiveElements = 0;
    bool direct = false;

    HaloExchange(const Grid& g, int rank, int ranks, MPI_Comm comm, DeviceFields& d) {
#ifdef HAVE_MPI_CUDA_QUERY
        int supported = MPIX_Query_cuda_support() != 0;
        int allSupported;
        mpiCheck(MPI_Allreduce(&supported, &allSupported, 1, MPI_INT, MPI_MIN, comm));
        direct = allSupported != 0;
#endif
        size_t sendCount = 0, receiveCount = 0;
        auto append = [](std::vector<Transfer>& list, size_t& total, int peer,
                         size_t begin, size_t end, size_t localBegin, size_t plane) {
            if (begin >= end) return;
            // +2 before subtraction avoids unsigned underflow for lower ghosts.
            const size_t offset = (begin + 2 - localBegin) * plane;
            size_t left = (end - begin) * plane;
            size_t done = 0;
            while (left) {
                const size_t count = std::min<size_t>(left, INT_MAX);
                list.push_back({peer, offset + done, count, total});
                total += count;
                done += count;
                left -= count;
            }
        };
        const size_t end = g.begin + g.localNz;
        for (int peer = std::max(0, rank - 2); peer <= std::min(ranks - 1, rank + 2); ++peer) {
            if (peer == rank) continue;
            const size_t pb = slabBegin(peer, ranks, g.nz);
            const size_t pe = slabBegin(peer + 1, ranks, g.nz);
            const size_t lower = g.begin > 2 ? g.begin - 2 : 0;
            const size_t peerLower = pb > 2 ? pb - 2 : 0;
            append(receives, receiveCount, peer, std::max(lower, pb), std::min(g.begin, pe), g.begin, g.plane);
            append(receives, receiveCount, peer, std::max(end, pb), std::min(g.nz, std::min(end + 2, pe)), g.begin, g.plane);
            append(sends, sendCount, peer, std::max(peerLower, g.begin), std::min(pb, end), g.begin, g.plane);
            append(sends, sendCount, peer, std::max(pe, g.begin), std::min(g.nz, std::min(pe + 2, end)), g.begin, g.plane);
        }
        if (!direct) {
            if (sendCount) cudaCheck(cudaMallocHost(&sendHost, sendCount * sizeof(double)));
            // The next iteration may receive before this iteration's async
            // host-to-device copy has completed. Give each parity its own buffer.
            receiveElements = receiveCount;
            if (receiveCount > std::numeric_limits<size_t>::max() / (2 * sizeof(double)))
                fail("Pinned receive buffers are too large.");
            if (receiveCount) cudaCheck(cudaMallocHost(&receiveHost, 2 * receiveCount * sizeof(double)));
        }
        for (int parity = 0; parity < 2; ++parity) {
            auto& req = requests[parity];
            req.resize(receives.size() + sends.size(), MPI_REQUEST_NULL);
            size_t index = 0;
            for (const auto& t : receives) {
                double* buffer = direct ? d.c[parity] + t.deviceOffset
                                        : receiveHost + parity * receiveElements + t.hostOffset;
                mpiCheck(MPI_Recv_init(buffer, static_cast<int>(t.count), MPI_DOUBLE,
                                      t.peer, 0, comm, &req[index++]));
            }
            for (const auto& t : sends) {
                double* buffer = direct ? d.c[parity] + t.deviceOffset : sendHost + t.hostOffset;
                mpiCheck(MPI_Send_init(buffer, static_cast<int>(t.count), MPI_DOUBLE,
                                      t.peer, 0, comm, &req[index++]));
            }
        }
    }

    void start(int parity, DeviceFields& d) {
        // Receives can progress while the outgoing GPU planes are copied.
        auto& req = requests[parity];
        const int nr = static_cast<int>(receives.size());
        if (nr) mpiCheck(MPI_Startall(nr, req.data()));
        if (direct) {
            cudaCheck(cudaEventSynchronize(d.boundaryReady));
        } else {
            for (const auto& t : sends)
                cudaCheck(cudaMemcpyAsync(sendHost + t.hostOffset, d.c[parity] + t.deviceOffset,
                                          t.count * sizeof(double), cudaMemcpyDeviceToHost, d.boundary));
            cudaCheck(cudaStreamSynchronize(d.boundary));
        }
        const int ns = static_cast<int>(sends.size());
        if (ns) mpiCheck(MPI_Startall(ns, req.data() + nr));
    }

    void finish(int parity, DeviceFields& d) {
        auto& req = requests[parity];
        if (!req.empty()) mpiCheck(MPI_Waitall(static_cast<int>(req.size()), req.data(), MPI_STATUSES_IGNORE));
        if (!direct) {
            for (const auto& t : receives)
                cudaCheck(cudaMemcpyAsync(d.c[parity] + t.deviceOffset,
                                          receiveHost + parity * receiveElements + t.hostOffset,
                                          t.count * sizeof(double), cudaMemcpyHostToDevice, d.boundary));
        }
    }

    ~HaloExchange() {
        for (auto& set : requests)
            for (auto& req : set) MPI_Request_free(&req);
        if (sendHost) cudaFreeHost(sendHost);
        if (receiveHost) cudaFreeHost(receiveHost);
    }
};

void distributedStep(const Grid& g, int parity, DeviceFields& d, HaloExchange& halo) {
    cudaCheck(cudaStreamWaitEvent(d.compute, d.boundaryReady, 0));
    // The interior is independent of the incoming concentration halo.
    if (g.localNz > 2)
        chemicalPotential<false><<<d.launchGrid(g.localNz - 2), d.block, 0, d.compute>>>
            (d.c[parity], d.mu, g, 1, g.localNz - 2);
    cudaCheck(cudaEventRecord(d.muReady, d.compute));
    if (g.localNz > 4)
        updateConcentration<false><<<d.launchGrid(g.localNz - 4), d.block, 0, d.compute>>>
            (d.c[1 - parity], d.c[parity], d.mu, g, 2, g.localNz - 4);

    halo.start(parity, d);
    halo.finish(parity, d);
    const size_t muPlanes = g.localNz == 1 ? 3 : 4;
    chemicalPotential<true><<<d.launchGrid(muPlanes), d.block, 0, d.boundary>>>
        (d.c[parity], d.mu, g, 0, muPlanes);
    cudaCheck(cudaStreamWaitEvent(d.boundary, d.muReady, 0));
    const size_t cPlanes = std::min<size_t>(4, g.localNz);
    updateConcentration<true><<<d.launchGrid(cPlanes), d.block, 0, d.boundary>>>
        (d.c[1 - parity], d.c[parity], d.mu, g, 0, cPlanes);
    // The next exchange needs only these first/last two output planes.
    cudaCheck(cudaEventRecord(d.boundaryReady, d.boundary));
    cudaCheck(cudaGetLastError());
}

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
    std::printf("Launch one MPI rank per GPU; use OMP_NUM_THREADS to set host workers.\n");
}

bool parseNumber(const char* text, size_t& value) {
    if (!*text || *text == '-') return false;
    char* end;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || *end || parsed > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(parsed);
    return true;
}

// Gathering is reserved for -r: retain the original ordered Kahan sum, samples,
// and byte hash without replicating the entire grid on every rank.
void outputResults(const std::vector<double>& local, const Grid& g,
                   int rank, int ranks, MPI_Comm comm) {
    if (ranks == 1) {
        print_results(local, "Concentration");
        return;
    }
    constexpr size_t chunk = INT_MAX;
    if (rank == 0) {
        std::vector<double> global(g.plane * g.nz);
        std::copy(local.begin(), local.end(), global.begin());
        for (int peer = 1; peer < ranks; ++peer) {
            const size_t begin = slabBegin(peer, ranks, g.nz) * g.plane;
            const size_t count = (slabBegin(peer + 1, ranks, g.nz) * g.plane) - begin;
            for (size_t offset = 0; offset < count;) {
                const int n = static_cast<int>(std::min(chunk, count - offset));
                mpiCheck(MPI_Recv(global.data() + begin + offset, n, MPI_DOUBLE, peer, 1, comm, MPI_STATUS_IGNORE));
                offset += n;
            }
        }
        print_results(global, "Concentration");
    } else {
        for (size_t offset = 0; offset < local.size();) {
            const int n = static_cast<int>(std::min(chunk, local.size() - offset));
            mpiCheck(MPI_Send(local.data() + offset, n, MPI_DOUBLE, 0, 1, comm));
            offset += n;
        }
    }
}

bool validateResult(const std::vector<double>& local, MPI_Comm comm, int rank) {
    int bad = 0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    #pragma omp parallel for schedule(static) reduction(|:bad) reduction(min:minimum) reduction(max:maximum)
    for (size_t i = 0; i < local.size(); ++i) {
        bad |= !std::isfinite(local[i]);
        minimum = std::min(minimum, local[i]);
        maximum = std::max(maximum, local[i]);
    }
    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, comm));
    double range[2] = {minimum, -maximum};
    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, range, 2, MPI_DOUBLE, MPI_MIN, comm));
    minimum = range[0];
    maximum = -range[1];
    if (rank == 0) {
        if (bad) std::printf("Validation failed: found NaN or Inf value\n");
        else {
            std::printf("Concentration range: [%.6f, %.6f]\n", minimum, maximum);
            if (maximum > 10.0 || minimum < -10.0)
                std::printf("Validation failed: values out of expected range\n");
        }
    }
    return !bad && maximum <= 10.0 && minimum >= -10.0;
}

int run(size_t nx, size_t ny, size_t nz, int iterations, bool validate,
        bool printResults, MPI_Comm comm) {
    int rank, ranks;
    mpiCheck(MPI_Comm_rank(comm, &rank));
    mpiCheck(MPI_Comm_size(comm, &ranks));
    MPI_Comm node;
    mpiCheck(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int nodeRank;
    mpiCheck(MPI_Comm_rank(node, &nodeRank));
    mpiCheck(MPI_Comm_free(&node));
    int devices;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) fail("This benchmark requires a CUDA GPU on every participating rank.");
    cudaCheck(cudaSetDevice(nodeRank % devices));
    // Avoid hundreds of host workers per GPU unless explicitly requested.
    if (!std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::min(8, omp_get_max_threads()));

    const size_t begin = slabBegin(rank, ranks, nz);
    const size_t localNz = slabBegin(rank + 1, ranks, nz) - begin;
    Grid g{nx, ny, nz, nx * ny, localNz, begin};
    const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(double);
    if (localNz + 4 > maxElements / g.plane) fail("Local grid including halos is too large.");
    DeviceFields d(g);
    HaloExchange halo(g, rank, ranks, comm, d);
    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel execution: %d MPI ranks, %d OpenMP threads/rank, CUDA (%s halos)\n",
                    ranks, omp_get_max_threads(), halo.direct ? "device" : "pinned host");
        std::printf("Initializing concentration field...\n");
    }
    std::vector<double> local(g.plane * localNz);
    const size_t volume = g.plane * nz;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local.size(); ++i) {
        const size_t linearId = begin * g.plane + i;
        const double pseudo = (((linearId + 1) * size_t(1299709)) % volume) / static_cast<double>(volume);
        local[i] = -1.0 + 2.0 * pseudo;
    }
    cudaCheck(cudaMemcpy(d.c[0] + 2 * g.plane, local.data(), local.size() * sizeof(double), cudaMemcpyHostToDevice));
    if (!validate && !printResults) std::vector<double>().swap(local);
    cudaCheck(cudaEventRecord(d.boundaryReady, d.boundary));
    cudaCheck(cudaEventSynchronize(d.boundaryReady));

    // Two captured steps return to the same concentration buffer. Graphs avoid
    // repeated host launch overhead on small single-GPU benchmark grids.
    if (ranks == 1 && iterations >= 2) {
        cudaCheck(cudaStreamBeginCapture(d.compute, cudaStreamCaptureModeThreadLocal));
        d.fullStep(g, 0);
        d.fullStep(g, 1);
        cudaCheck(cudaStreamEndCapture(d.compute, &d.graph));
        cudaCheck(cudaGraphInstantiate(&d.graphExec, d.graph, nullptr, nullptr, 0));
        cudaCheck(cudaGraphUpload(d.graphExec, d.compute));
        cudaCheck(cudaStreamSynchronize(d.compute));
    }
    if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    mpiCheck(MPI_Barrier(comm));
    const double start = MPI_Wtime();
    if (ranks == 1) {
        for (int t = 0; t < iterations / 2; ++t)
            cudaCheck(cudaGraphLaunch(d.graphExec, d.compute));
        if (iterations % 2) d.fullStep(g, 0);
    } else {
        for (int t = 0; t < iterations; ++t) distributedStep(g, t % 2, d, halo);
    }
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamSynchronize(d.compute));
    cudaCheck(cudaStreamSynchronize(d.boundary));
    double elapsed = MPI_Wtime() - start;
    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, comm));
    if (rank == 0) {
        std::printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
        const double mcups = elapsed > 0 ? static_cast<double>(volume) * iterations / elapsed / 1e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    if (printResults || validate)
        cudaCheck(cudaMemcpy(local.data(), d.c[iterations % 2] + 2 * g.plane,
                             local.size() * sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) outputResults(local, g, rank, ranks, comm);
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        const bool valid = validateResult(local, comm, rank);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED support is required.");
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false, help = false, invalid = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { help = true; break; }
        else if (i + 1 < argc && (!std::strcmp(argv[i], "-x") || !std::strcmp(argv[i], "-y") ||
                                  !std::strcmp(argv[i], "-z") || !std::strcmp(argv[i], "-i"))) {
            const char option = argv[i][1];
            size_t value = 0;
            if (!parseNumber(argv[++i], value)) { invalid = true; break; }
            if (option == 'x') nx = value;
            else if (option == 'y') ny = value;
            else if (option == 'z') nz = value;
            else if (value <= INT_MAX) iterations = static_cast<int>(value);
            else { invalid = true; break; }
        } else {
            if (worldRank == 0) std::printf("Unknown option: %s\n", argv[i]);
            invalid = true;
            break;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
    if (!nx || !ny || !nz || nz > static_cast<size_t>(LLONG_MAX) - 4 ||
        nx > limit / ny || (nx * ny) > limit / nz) invalid = true;
    if (help || invalid) {
        if (worldRank == 0) {
            if (invalid) std::fprintf(stderr, "Invalid arguments or grid size.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return invalid ? 1 : 0;
    }
    int worldSize;
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    const int activeRanks = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &comm));
    int result = 0;
    try {
        if (comm != MPI_COMM_NULL) {
            result = run(nx, ny, nz, iterations, validate, printResults, comm);
            mpiCheck(MPI_Comm_free(&comm));
        }
    } catch (const std::exception& error) {
        fail(error.what());
    }
    mpiCheck(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    mpiCheck(MPI_Finalize());
    return result;
}
