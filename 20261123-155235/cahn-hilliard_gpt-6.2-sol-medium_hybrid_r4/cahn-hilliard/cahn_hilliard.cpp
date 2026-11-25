#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

struct Range {
    int begin;
    int end;

    int size() const { return end - begin; }
};

Range slabs(int depth, int ranks, int rank) {
    const int base = depth / ranks;
    const int extra = depth % ranks;
    const int begin = rank * base + std::min(rank, extra);
    return {begin, begin + base + (rank < extra)};
}

Range extend(Range slabs, int width, int depth) {
    slabs.begin -= std::min(slabs.begin, width);
    slabs.end += std::min(depth - slabs.end, width);
    return slabs;
}

Range intersect(Range a, Range b) {
    const int begin = std::max(a.begin, b.begin);
    return {begin, std::max(begin, std::min(a.end, b.end))};
}

Range interior(Range owned, int width, int depth) {
    const int begin = owned.begin == 0 ? 0 : owned.begin + std::min(width, owned.size());
    const int end = owned.end == depth ? owned.end : std::max(begin, owned.end - width);
    return {begin, end};
}

struct Grid {
    int x, y, z;
    int columns, rows, depth;
    int axis;
    size_t plane, volume;

    Grid(int nx, int ny, int nz, int ranks) : x(nx), y(ny), z(nz) {
        // Keep the original layout when Z offers enough interior work per rank.
        // For thin grids, split along the longest axis and keep planes contiguous.
        axis = 2;
        if (ranks > 1 && z / ranks < 4) {
            if (y > z) axis = 1;
            if (x > (axis == 1 ? y : z)) axis = 0;
        }
        columns = axis == 0 ? y : x;
        rows = axis == 2 ? y : z;
        depth = axis == 0 ? x : (axis == 1 ? y : z);
        plane = static_cast<size_t>(columns) * rows;
        if (plane > std::numeric_limits<size_t>::max() / sizeof(double) / depth) {
            throw std::length_error("Grid size exceeds addressable memory");
        }
        volume = plane * depth;
    }

    size_t originalIndex(size_t index) const {
        if (axis == 2) return index;
        const size_t column = index % columns;
        const size_t row = index / columns % rows;
        const size_t slab = index / plane;
        return axis == 1 ? (row * y + slab) * x + column
                         : (row * y + column) * x + slab;
    }
};

template <bool Potential, int Axis, typename Index>
__global__ void stencil(const double* __restrict__ input, const double* __restrict__ cold,
                        double* __restrict__ output, unsigned columns, unsigned rows, unsigned depth,
                        Index plane, unsigned first, unsigned firstCount, unsigned second, unsigned count) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= columns) return;

    for (unsigned row = blockIdx.y * blockDim.y + threadIdx.y; row < rows;
         row += gridDim.y * blockDim.y) {
        for (unsigned slab = blockIdx.z * blockDim.z + threadIdx.z; slab < count;
             slab += gridDim.z * blockDim.z) {
            const unsigned z = slab < firstCount ? first + slab : second + (slab - firstCount);
            const Index i = z * plane + static_cast<Index>(row) * columns + x;
            const double value = input[i];
            const double twice = 2.0 * value;
            const double dColumn = (input[i + (x + 1 < columns)] + input[i - (x > 0)]) - twice;
            const double dRow = (input[i + (row + 1 < rows ? columns : 0)]
                               + input[i - (row > 0 ? columns : 0)]) - twice;
            // Ranges always leave a neighbor in storage at internal rank boundaries.
            const double dSlab = (input[i + (z + 1 < depth ? plane : 0)]
                                + input[i - (z > 0 ? plane : 0)]) - twice;
            double laplacian;
            if constexpr (Axis == 0) {
                laplacian = (dSlab + dColumn) + dRow;
            } else if constexpr (Axis == 1) {
                laplacian = (dColumn + dSlab) + dRow;
            } else {
                laplacian = (dColumn + dRow) + dSlab;
            }

            if constexpr (Potential) {
                constexpr double eAA = -(2.0 / 9.0);
                constexpr double eBB = -(2.0 / 9.0);
                constexpr double eAB = 2.0 / 9.0;
                const double bonds = __fma_rn(-twice, eAB, __fma_rn(value + 1.0, eAA, (value - 1.0) * eBB));
                const double bulk = __fma_rn(value * value, value, __fma_rn(4.5, bonds, 3.0 * value));
                output[i] = __fma_rn(-0.5, laplacian, bulk);
            } else {
                output[i] = __fma_rn(0.01, laplacian, cold[i]);
            }
        }
    }
}

template <bool Potential, typename Index>
void dispatchStencil(const Grid& grid, int depth, int first, int firstCount, int second, int count,
                     Index plane, dim3 blocks, dim3 threads, const double* input, const double* cold,
                     double* output, cudaStream_t stream) {
    if (grid.axis == 2) {
        stencil<Potential, 2><<<blocks, threads, 0, stream>>>(input, cold, output, grid.columns, grid.rows,
            depth, plane, first, firstCount, second, count);
    } else if (grid.axis == 1) {
        stencil<Potential, 1><<<blocks, threads, 0, stream>>>(input, cold, output, grid.columns, grid.rows,
            depth, plane, first, firstCount, second, count);
    } else {
        stencil<Potential, 0><<<blocks, threads, 0, stream>>>(input, cold, output, grid.columns, grid.rows,
            depth, plane, first, firstCount, second, count);
    }
}

template <bool Potential>
void launchStencil(const Grid& grid, Range storage, Range first, Range second,
                   const double* input, const double* cold, double* output, cudaStream_t stream) {
    const int count = first.size() + second.size();
    if (count == 0) return;

    dim3 threads(32, 4);
    if (grid.columns < 32 || grid.rows < 4) {
        // Fill the block along Z when the contiguous plane is narrow.
        threads.x = 1;
        while (threads.x < grid.columns && threads.x < 32) threads.x *= 2;
        threads.y = 1;
        while (threads.y < grid.rows && threads.y < 128 / threads.x) threads.y *= 2;
        while (threads.x < grid.columns && threads.x * threads.y < 128) threads.x *= 2;
        while (threads.z < count && threads.x * threads.y * threads.z < 128) threads.z *= 2;
    }
    const dim3 blocks((grid.columns - 1) / threads.x + 1,
                      std::min((grid.rows - 1) / threads.y + 1, 65535U),
                      std::min((count - 1) / threads.z + 1, 65535U));
    const int firstOffset = first.begin - storage.begin;
    const int secondOffset = second.begin - storage.begin;
    if (storage.size() * grid.plane <= UINT_MAX) {
        dispatchStencil<Potential>(grid, storage.size(), firstOffset, first.size(), secondOffset, count,
            static_cast<unsigned>(grid.plane), blocks, threads, input, cold, output, stream);
    } else {
        dispatchStencil<Potential>(grid, storage.size(), firstOffset, first.size(), secondOffset, count,
            grid.plane, blocks, threads, input, cold, output, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool cudaAwareMPI() {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

struct Transfer {
    int peer;
    Range planes;
    double* staging = nullptr;
};

class HaloExchange {
public:
    HaloExchange(const Grid& grid, Range owned, Range storage, int rank, int ranks, MPI_Comm comm)
        : grid(grid), storage(storage), comm(comm), direct(cudaAwareMPI()) {
        // Two consecutive halo planes can belong to different ranks for thin slabs.
        for (int peer = std::max(0, rank - 2); peer < ranks && peer <= rank + 2; ++peer) {
            if (peer == rank) continue;
            const Range other = slabs(grid.depth, ranks, peer);
            addTransfer(receives, peer, intersect(other, storage));
            addTransfer(sends, peer, intersect(owned, extend(other, 2, grid.depth)));
        }
        if (!sends.empty() && !direct) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&received, cudaEventDisableTiming));
        }
        requests.reserve(2 * (sends.size() + receives.size()));
    }

    ~HaloExchange() {
        for (const auto& transfer : sends) {
            if (transfer.staging) CUDA_CHECK(cudaFreeHost(transfer.staging));
        }
        for (const auto& transfer : receives) {
            if (transfer.staging) CUDA_CHECK(cudaFreeHost(transfer.staging));
        }
        if (stream) {
            CUDA_CHECK(cudaEventDestroy(received));
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
    }

    void postReceives(double* cold) {
        // MPI may recycle the pinned receive buffers once the previous copy finishes.
        if (received) CUDA_CHECK(cudaEventSynchronize(received));
        requests.clear();
        for (const auto& transfer : receives) {
            post(transfer, direct ? deviceBuffer(cold, transfer) : transfer.staging, false);
        }
    }

    void finish(double* cold, cudaEvent_t ready, cudaStream_t compute) {
        if (direct) {
            CUDA_CHECK(cudaEventSynchronize(ready));
        } else {
            CUDA_CHECK(cudaStreamWaitEvent(stream, ready, 0));
            for (const auto& transfer : sends) {
                CUDA_CHECK(cudaMemcpyAsync(transfer.staging, deviceBuffer(cold, transfer),
                    elements(transfer) * sizeof(double), cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        for (const auto& transfer : sends) {
            post(transfer, direct ? deviceBuffer(cold, transfer) : transfer.staging, true);
        }
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

        if (!direct) {
            for (const auto& transfer : receives) {
                CUDA_CHECK(cudaMemcpyAsync(deviceBuffer(cold, transfer), transfer.staging,
                    elements(transfer) * sizeof(double), cudaMemcpyHostToDevice, stream));
            }
            CUDA_CHECK(cudaEventRecord(received, stream));
            CUDA_CHECK(cudaStreamWaitEvent(compute, received, 0));
        }
    }

private:
    const Grid& grid;
    Range storage;
    MPI_Comm comm;
    const bool direct;
    cudaStream_t stream = nullptr;
    cudaEvent_t received = nullptr;
    std::vector<Transfer> receives, sends;
    std::vector<MPI_Request> requests;

    size_t elements(const Transfer& t) const { return t.planes.size() * grid.plane; }

    double* deviceBuffer(double* cold, const Transfer& t) const {
        return cold + (t.planes.begin - storage.begin) * grid.plane;
    }

    void addTransfer(std::vector<Transfer>& transfers, int peer, Range planes) {
        if (planes.size() == 0) return;
        Transfer t{peer, planes};
        if (!direct) CUDA_CHECK(cudaMallocHost(&t.staging, elements(t) * sizeof(double)));
        transfers.push_back(t);
    }

    void post(const Transfer& transfer, double* buffer, bool send) {
        for (size_t left = elements(transfer); left != 0;) {
            const int count = static_cast<int>(std::min(left, static_cast<size_t>(INT_MAX)));
            auto& request = requests.emplace_back();
            if (send) {
                MPI_Isend(buffer, count, MPI_DOUBLE, transfer.peer, 0, comm, &request);
            } else {
                MPI_Irecv(buffer, count, MPI_DOUBLE, transfer.peer, 0, comm, &request);
            }
            buffer += count;
            left -= count;
        }
    }
};

void selectDevice(MPI_Comm comm) {
    MPI_Comm local;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int localRank;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);

    int devices;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) checkCuda(cudaErrorNoDevice, "Selecting a CUDA device");
    CUDA_CHECK(cudaSetDevice(localRank % devices));
}

int hostThreads(size_t count) {
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(), (count - 1) / 65536 + 1));
}

void initialize(double* field, const Grid& grid, Range owned, size_t count) {
    const size_t offset = owned.begin * grid.plane;
    const double volume = static_cast<double>(grid.volume);
    const int threads = hostThreads(count);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < count; ++i) {
        const size_t id = grid.originalIndex(offset + i);
        const double pseudo = ((id + 1) * 1299709 % grid.volume) / volume;
        field[i] = -1.0 + 2.0 * pseudo;
    }
}

void gatherResults(const double* local, const Grid& grid, Range owned, int rank, int ranks,
                   MPI_Comm comm) {
    // Keep each collective's counts and displacements representable as MPI ints.
    std::vector<double> gathered(rank == 0 ? grid.volume : 0);
    std::vector<int> counts(rank == 0 ? ranks : 0), displacements(counts.size());
    const size_t localBegin = owned.begin * grid.plane;
    const size_t localEnd = owned.end * grid.plane;
    for (size_t begin = 0; begin < grid.volume;) {
        const size_t end = begin + std::min(grid.volume - begin, static_cast<size_t>(INT_MAX));
        if (rank == 0) {
            for (int peer = 0; peer < ranks; ++peer) {
                const Range part = slabs(grid.depth, ranks, peer);
                const size_t from = std::max(begin, part.begin * grid.plane);
                const size_t to = std::min(end, part.end * grid.plane);
                counts[peer] = from < to ? static_cast<int>(to - from) : 0;
                displacements[peer] = counts[peer] ? static_cast<int>(from - begin) : 0;
            }
        }
        const size_t from = std::max(begin, localBegin);
        const size_t to = std::min(end, localEnd);
        const int count = from < to ? static_cast<int>(to - from) : 0;
        MPI_Gatherv(count ? local + (from - localBegin) : local, count, MPI_DOUBLE,
                    rank == 0 ? gathered.data() + begin : nullptr, counts.data(), displacements.data(),
                    MPI_DOUBLE, 0, comm);
        begin = end;
    }
    if (rank != 0) return;

    if (grid.axis != 2) {
        std::vector<double> ordered(grid.volume);
        const int threads = hostThreads(grid.volume);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (size_t i = 0; i < grid.volume; ++i) {
            ordered[grid.originalIndex(i)] = gathered[i];
        }
        gathered.swap(ordered);
    }
    print_results(gathered, "Concentration");
}

bool validateResult(const double* local, size_t count, int rank, MPI_Comm comm) {
    double minValue = std::numeric_limits<double>::infinity();
    double maxValue = -minValue;
    int nonfinite = 0;
    const int threads = hostThreads(count);
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(min:minValue) reduction(max:maxValue) reduction(|:nonfinite)
    for (size_t i = 0; i < count; ++i) {
        const double value = local[i];
        if (std::isfinite(value)) {
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
        } else {
            nonfinite = 1;
        }
    }

    // Negating the maximum allows both extrema to use one collective.
    const double localStats[] = {minValue, -maxValue, -static_cast<double>(nonfinite)};
    double stats[3];
    MPI_Allreduce(localStats, stats, 3, MPI_DOUBLE, MPI_MIN, comm);
    const bool valid = stats[2] == 0 && stats[0] >= -10.0 && -stats[1] <= 10.0;
    if (rank == 0) {
        if (stats[2] != 0) {
            printf("Validation failed: found NaN or Inf value\n");
        } else {
            printf("Concentration range: [%.6f, %.6f]\n", stats[0], -stats[1]);
            if (!valid) printf("Validation failed: values out of expected range\n");
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

int run(const Grid& grid, int rank, int ranks, int iterations, bool validate,
        bool printResults, MPI_Comm comm) {
    const Range owned = slabs(grid.depth, ranks, rank);
    const Range storage = extend(owned, 2, grid.depth);
    const size_t offset = (owned.begin - storage.begin) * grid.plane;
    const size_t count = owned.size() * grid.plane;
    const size_t bytes = storage.size() * grid.plane * sizeof(double);
    double *cold, *cnew, *mu, *host;
    CUDA_CHECK(cudaMalloc(&cold, bytes));
    CUDA_CHECK(cudaMalloc(&cnew, bytes));
    CUDA_CHECK(cudaMalloc(&mu, bytes));
    CUDA_CHECK(cudaMallocHost(&host, count * sizeof(double)));
    cudaStream_t compute;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    if (rank == 0) printf("Initializing concentration field...\n");
    initialize(host, grid, owned, count);
    CUDA_CHECK(cudaMemcpyAsync(cold + offset, host, count * sizeof(double), cudaMemcpyHostToDevice, compute));
    CUDA_CHECK(cudaStreamSynchronize(compute));
    if (!validate && !printResults) CUDA_CHECK(cudaFreeHost(host));

    HaloExchange halo(grid, owned, storage, rank, ranks, comm);
    cudaEvent_t ready = nullptr;
    if (ranks > 1) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(ready, compute));
    }
    const Range potential = extend(owned, 1, grid.depth);
    const Range innerPotential = interior(owned, 1, grid.depth);
    const Range innerUpdate = interior(owned, 2, grid.depth);
    const Range empty{storage.begin, storage.begin};

    const auto advanceInterior = [&](const double* current, double* next) {
        launchStencil<true>(grid, storage, innerPotential, empty, current, current, mu, compute);
        launchStencil<false>(grid, storage, innerUpdate, empty, mu, current, next, compute);
    };
    // Batch the single-rank stencil launches. Repeated batches have even length
    // so that the graph can reuse the same pair of concentration buffers.
    const int batch = ranks == 1 && iterations > 1 ? std::min(iterations, 32) : 0;
    cudaGraphExec_t graphExec = nullptr;
    if (batch) {
        cudaGraph_t graph;
        double* current = cold;
        double* next = cnew;
        CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
        for (int step = 0; step < batch; ++step) {
            advanceInterior(current, next);
            std::swap(current, next);
        }
        CUDA_CHECK(cudaStreamEndCapture(compute, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaGraphUpload(graphExec, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    int step = 0;
    if (graphExec) {
        for (; step <= iterations - batch; step += batch) {
            CUDA_CHECK(cudaGraphLaunch(graphExec, compute));
        }
        if (step % 2) std::swap(cold, cnew);
    }
    for (; step < iterations; ++step) {
        advanceInterior(cold, cnew);
        if (ranks > 1) {
            halo.postReceives(cold);
            halo.finish(cold, ready, compute);
            launchStencil<true>(grid, storage, {potential.begin, innerPotential.begin},
                {innerPotential.end, potential.end}, cold, cold, mu, compute);
            launchStencil<false>(grid, storage, {owned.begin, innerUpdate.begin},
                {innerUpdate.end, owned.end}, mu, cold, cnew, compute);
            CUDA_CHECK(cudaEventRecord(ready, compute));
        }
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    const double localSeconds = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double updates = static_cast<double>(grid.volume) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / seconds / 1e6);
    }

    bool valid = true;
    if (validate || printResults) {
        CUDA_CHECK(cudaMemcpy(host, cold + offset, count * sizeof(double), cudaMemcpyDeviceToHost));
        if (printResults) gatherResults(host, grid, owned, rank, ranks, comm);
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            valid = validateResult(host, count, rank, comm);
        }
        CUDA_CHECK(cudaFreeHost(host));
    }
    if (ready) CUDA_CHECK(cudaEventDestroy(ready));
    if (graphExec) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaFree(mu));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(cold));
    return valid ? 0 : 1;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

bool readInteger(const char* value, int& result) {
    char* end;
    errno = 0;
    const long number = strtol(value, &end, 10);
    if (errno || end == value || *end || number < INT_MIN || number > INT_MAX) return false;
    result = static_cast<int>(number);
    return true;
}

int benchmark(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int nx = 64, ny = 0, nz = 0, iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        const char* option = argv[i];
        int* value = nullptr;
        if (strcmp(option, "-x") == 0) value = &nx;
        else if (strcmp(option, "-y") == 0) value = &ny;
        else if (strcmp(option, "-z") == 0) value = &nz;
        else if (strcmp(option, "-i") == 0) value = &iterations;
        else if (strcmp(option, "-v") == 0) validate = true;
        else if (strcmp(option, "-r") == 0) printResults = true;
        else if (strcmp(option, "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", option);
                printUsage(argv[0]);
            }
            return 1;
        }
        if (value && (i + 1 == argc || !readInteger(argv[++i], *value))) {
            if (rank == 0) fprintf(stderr, "Invalid integer for %s\n", option);
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be positive\n");
        return 1;
    }

    const Grid grid(nx, ny, nz, ranks);
    const int activeRanks = std::min(ranks, grid.depth);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &active);
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %d x %d x %d\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    int status = 0;
    if (rank < activeRanks) {
        selectDevice(active);
        status = run(grid, rank, activeRanks, iterations, validate, printResults, active);
        MPI_Comm_free(&active);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

} // namespace

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI must support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int status = 1;
    try {
        status = benchmark(argc, argv);
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
