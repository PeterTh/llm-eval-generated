#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#if defined(OMPI_MAJOR_VERSION) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

namespace {

constexpr int haloWidth = 2;
constexpr int haloTag = 0;
constexpr size_t mpiMaxCount = std::numeric_limits<int>::max();
constexpr size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(double);

[[noreturn]] void abortRun(const char* message) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        abortRun(cudaGetErrorString(status));
    }
}

struct Slab {
    size_t begin;
    size_t end;
};

Slab slabFor(int rank, int ranks, size_t nz) {
    const size_t base = nz / ranks;
    const size_t extra = nz % ranks;
    const size_t begin = rank * base + std::min(static_cast<size_t>(rank), extra);
    return {begin, begin + base + (static_cast<size_t>(rank) < extra)};
}

Slab withHalo(Slab slab, size_t nz) {
    return {slab.begin - std::min<size_t>(slab.begin, haloWidth),
            slab.end + std::min<size_t>(nz - slab.end, haloWidth)};
}

// A launch visits one contiguous interval, or two disjoint boundary intervals.
struct Planes {
    size_t begin;
    size_t firstCount;
    size_t secondBegin;
    size_t count;
};

struct StencilWork {
    Planes interior;
    Planes boundary;
};

StencilWork splitWork(size_t begin, size_t end, size_t safeBegin, size_t safeEnd) {
    safeBegin = std::min(safeBegin, end);
    safeEnd = std::max(safeEnd, safeBegin);
    return {{safeBegin, safeEnd - safeBegin, 0, safeEnd - safeBegin},
            {begin, safeBegin - begin, safeEnd, safeBegin - begin + end - safeEnd}};
}

struct Grid {
    unsigned nx;
    unsigned ny;
    size_t plane;
    size_t zHigh;
};

__device__ __forceinline__ double chemicalPotential(double c, double lap) {
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = 2.0 / 9.0;
    // Match the contraction order of the original optimized host calculation.
    double energy = __fma_rn(c + 1.0, eAA, (c - 1.0) * eBB);
    energy = __fma_rn(-2.0 * c, eAB, energy);
    const double bulk = __fma_rn(c * c, c, __fma_rn(4.5, energy, 3.0 * c));
    return __fma_rn(-0.5, lap, bulk);
}

template <bool computePotential, unsigned zTile>
__global__ void stencil(double* __restrict__ out, const double* __restrict__ c,
                        const double* __restrict__ stencilInput, Grid grid, Planes planes) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= grid.nx) return;

    const unsigned xNext = x + 1 < grid.nx;
    const unsigned xPrev = x > 0;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < grid.ny;
         y += gridDim.y * blockDim.y) {
        const unsigned yNext = y + 1 < grid.ny ? grid.nx : 0;
        const unsigned yPrev = y > 0 ? grid.nx : 0;
        const size_t xy = static_cast<size_t>(y) * grid.nx + x;
        for (size_t p = blockIdx.z * zTile; p < planes.count; p += gridDim.z * zTile) {
            size_t z = planes.begin + p;
            if constexpr (zTile == 1) {
                if (p >= planes.firstCount) z = planes.secondBegin + p - planes.firstCount;
            }
            size_t index = z * grid.plane + xy;
            double current = stencilInput[index];
            double previous = stencilInput[index - (z > 0 ? grid.plane : 0)];

            // Consecutive planes reuse the z neighbors already loaded by the thread.
            #pragma unroll
            for (unsigned k = 0; k < zTile; ++k) {
                if (p + k >= planes.count) break;
                const double* center = stencilInput + index;
                const double next = center[z < grid.zHigh ? grid.plane : 0];
                const double twice = 2.0 * current;
                const double xx = *(center + xNext) + *(center - xPrev) - twice;
                const double yy = *(center + yNext) + *(center - yPrev) - twice;
                const double zz = next + previous - twice;
                const double lap = (xx + yy) + zz;
                if constexpr (computePotential) {
                    out[index] = chemicalPotential(current, lap);
                } else {
                    out[index] = __fma_rn(0.01, lap, c[index]);
                }
                previous = current;
                current = next;
                index += grid.plane;
                ++z;
            }
        }
    }
}

template <bool computePotential, unsigned zTile = 4>
void launchStencil(double* out, const double* c, const double* input, const Grid& grid,
                   const Planes& planes, cudaStream_t stream) {
    if (planes.count == 0) return;
    const dim3 threads(32, 4);
    const dim3 blocks((grid.nx + threads.x - 1) / threads.x,
                      std::min(65535u, (grid.ny + threads.y - 1) / threads.y),
                      std::min<size_t>(65535, (planes.count + zTile - 1) / zTile));
    stencil<computePotential, zTile><<<blocks, threads, 0, stream>>>(out, c, input, grid, planes);
    checkCuda(cudaGetLastError());
}

void selectDevice(MPI_Comm comm) {
    MPI_Comm local;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int localRank;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);

    int devices;
    checkCuda(cudaGetDeviceCount(&devices));
    if (devices == 0) abortRun("No visible CUDA devices");
    checkCuda(cudaSetDevice(localRank % devices));
}

bool deviceBuffersSupported(MPI_Comm comm) {
    int local = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    local = MPIX_Query_cuda_support();
#endif
    int supported;
    MPI_Allreduce(&local, &supported, 1, MPI_INT, MPI_MIN, comm);
    return supported != 0;
}

struct Transfer {
    int peer;
    int count;
    size_t deviceOffset;
    size_t hostOffset;
};

class HaloExchange {
public:
    HaloExchange(MPI_Comm comm, int rank, int ranks, size_t nz, size_t plane, Slab own, Slab storage)
        : comm(comm), direct(ranks > 1 && deviceBuffersSupported(comm)) {
        if (ranks == 1) return;

        // Radius two can reach a second neighbor when a rank owns a single plane.
        for (int peer = std::max(0, rank - haloWidth);
             peer <= rank + std::min(haloWidth, ranks - 1 - rank); ++peer) {
            if (peer == rank) continue;
            const Slab other = slabFor(peer, ranks, nz);
            const Slab send = intersection(own, withHalo(other, nz));
            const Slab recv = intersection(other, storage);
            addTransfers(sends, send, peer, storage.begin, plane, sendSize);
            addTransfers(receives, recv, peer, storage.begin, plane, receiveSize);
        }
        requests.resize(sends.size() + receives.size());
        if (!direct) {
            if (sendSize > maxCells || receiveSize > (maxCells - sendSize) / 2) {
                abortRun("Halo staging buffers exceed the addressable allocation size");
            }
            checkCuda(cudaMallocHost(&staging, (sendSize + 2 * receiveSize) * sizeof(double)));
        }
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        checkCuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    }

    ~HaloExchange() {
        if (ready) checkCuda(cudaEventDestroy(ready));
        if (stream) checkCuda(cudaStreamDestroy(stream));
        if (staging) checkCuda(cudaFreeHost(staging));
    }

    void begin(double* field, cudaStream_t compute) {
        checkCuda(cudaEventRecord(ready, compute));
        if (!direct) {
            checkCuda(cudaStreamWaitEvent(stream, ready));
            for (const auto& send : sends) {
                checkCuda(cudaMemcpyAsync(staging + send.hostOffset, field + send.deviceOffset,
                                          send.count * sizeof(double), cudaMemcpyDeviceToHost, stream));
            }
        }
        for (size_t i = 0; i < receives.size(); ++i) {
            const auto& recv = receives[i];
            double* dest = direct ? field + recv.deviceOffset : receiveBuffer(recv);
            MPI_Irecv(dest, recv.count, MPI_DOUBLE, recv.peer, haloTag, comm,
                      &requests[i]);
        }
    }

    void finish(double* field, cudaStream_t compute) {
        if (direct) {
            checkCuda(cudaEventSynchronize(ready));
        } else {
            checkCuda(cudaStreamSynchronize(stream));
        }
        for (size_t i = 0; i < sends.size(); ++i) {
            const auto& send = sends[i];
            double* source = direct ? field + send.deviceOffset : staging + send.hostOffset;
            MPI_Isend(source, send.count, MPI_DOUBLE, send.peer, haloTag, comm,
                      &requests[receives.size() + i]);
        }
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

        if (!direct) {
            for (const auto& recv : receives) {
                checkCuda(cudaMemcpyAsync(field + recv.deviceOffset, receiveBuffer(recv),
                                          recv.count * sizeof(double), cudaMemcpyHostToDevice, stream));
            }
            checkCuda(cudaEventRecord(ready, stream));
            checkCuda(cudaStreamWaitEvent(compute, ready));
            // The next MPI receive may start before these uploads have finished.
            receiveSlot ^= 1;
        }
    }

    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

private:
    static Slab intersection(Slab a, Slab b) {
        return {std::max(a.begin, b.begin), std::min(a.end, b.end)};
    }

    static void addTransfers(std::vector<Transfer>& transfers, Slab range, int peer,
                             size_t storageStart, size_t plane, size_t& stagingSize) {
        if (range.begin >= range.end) return;
        size_t offset = (range.begin - storageStart) * plane;
        size_t remaining = (range.end - range.begin) * plane;
        while (remaining > 0) {
            const size_t count = std::min(remaining, mpiMaxCount);
            transfers.push_back({peer, static_cast<int>(count), offset, stagingSize});
            offset += count;
            stagingSize += count;
            remaining -= count;
        }
    }

    double* receiveBuffer(const Transfer& transfer) const {
        return staging + sendSize + receiveSlot * receiveSize + transfer.hostOffset;
    }

    MPI_Comm comm;
    bool direct;
    std::vector<Transfer> sends;
    std::vector<Transfer> receives;
    std::vector<MPI_Request> requests;
    size_t sendSize = 0;
    size_t receiveSize = 0;
    size_t receiveSlot = 0;
    double* staging = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr;
};

int hostThreads(size_t count) {
    const int maxThreads = omp_get_max_threads();
    const int requested = std::getenv("OMP_NUM_THREADS") ? maxThreads : std::min(8, maxThreads);
    return static_cast<int>(std::min<size_t>(requested, 1 + count / 16384));
}

void initializeConcentration(double* local, size_t count, size_t offset, size_t volume) {
    [[maybe_unused]] const int threads = hostThreads(count);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < count; ++i) {
        const double pseudo = ((offset + i + 1) * 1299709 % volume) / static_cast<double>(volume);
        local[i] = -1.0 + 2.0 * pseudo;
    }
}

void gatherResults(const double* local, Slab own, size_t plane, size_t nz,
                   int rank, int ranks, MPI_Comm comm) {
    const size_t volume = plane * nz;
    std::vector<double> global(rank == 0 ? volume : 0);
    std::vector<int> counts(rank == 0 ? ranks : 0);
    std::vector<int> offsets(rank == 0 ? ranks : 0);
    // Gather in global z order, including grids larger than MPI's int displacements.
    for (size_t base = 0; base < volume;) {
        const size_t end = base + std::min(volume - base, mpiMaxCount);
        if (rank == 0) {
            for (int peer = 0; peer < ranks; ++peer) {
                const Slab slab = slabFor(peer, ranks, nz);
                const size_t first = std::max(base, slab.begin * plane);
                const size_t last = std::min(end, slab.end * plane);
                counts[peer] = last > first ? static_cast<int>(last - first) : 0;
                offsets[peer] = counts[peer] ? static_cast<int>(first - base) : 0;
            }
        }
        const size_t first = std::max(base, own.begin * plane);
        const size_t last = std::min(end, own.end * plane);
        const int count = last > first ? static_cast<int>(last - first) : 0;
        const double* send = count ? local + (first - own.begin * plane) : local;
        MPI_Gatherv(send, count, MPI_DOUBLE, rank == 0 ? global.data() + base : nullptr,
                    counts.data(), offsets.data(), MPI_DOUBLE, 0, comm);
        base = end;
    }
    if (rank == 0) print_results(global, "Concentration");
}

bool validateResult(const double* local, size_t count, int rank, MPI_Comm comm) {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -minimum;
    int invalid = 0;
    [[maybe_unused]] const int threads = hostThreads(count);
    #pragma omp parallel for schedule(static) num_threads(threads) \
        reduction(min:minimum) reduction(max:maximum) reduction(|:invalid)
    for (size_t i = 0; i < count; ++i) {
        const double value = local[i];
        if (std::isfinite(value)) {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        } else {
            invalid = 1;
        }
    }
    // Reduce both extrema with one collective by negating the minimum.
    const double stats[] = {-minimum, maximum, static_cast<double>(invalid)};
    double global[3];
    MPI_Reduce(stats, global, 3, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank != 0) return true;
    if (global[2] != 0) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    printf("Concentration range: [%.6f, %.6f]\n", -global[0], global[1]);
    if (global[1] > 10.0 || global[0] > 10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

int runSimulation(unsigned nx, unsigned ny, size_t nz, int iterations,
                  bool validate, bool printResults, MPI_Comm comm) {
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    const size_t plane = static_cast<size_t>(nx) * ny;
    const size_t volume = plane * nz;
    const Slab own = slabFor(rank, ranks, nz);
    const Slab storage = withHalo(own, nz);
    const size_t depth = own.end - own.begin;
    const size_t begin = own.begin - storage.begin;
    const size_t end = own.end - storage.begin;
    const size_t storageDepth = storage.end - storage.begin;
    const Grid grid{nx, ny, plane, storageDepth - 1};
    const bool left = own.begin > 0;
    const bool right = own.end < nz;
    const StencilWork potential = splitWork(begin - left, end + right,
                                            begin + left, end - right);
    const StencilWork update = splitWork(begin, end, begin + haloWidth * left,
                                         end - std::min<size_t>(depth, haloWidth * right));
    const size_t localCount = depth * plane;
    const size_t deviceBytes = storageDepth * plane * sizeof(double);
    const size_t localBytes = localCount * sizeof(double);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %u x %u x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }

    selectDevice(comm);
    HaloExchange exchange(comm, rank, ranks, nz, plane, own, storage);
    cudaStream_t compute;
    double *cold, *cnew, *mu, *local;
    checkCuda(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    checkCuda(cudaMalloc(&cold, deviceBytes));
    checkCuda(cudaMalloc(&cnew, deviceBytes));
    checkCuda(cudaMalloc(&mu, deviceBytes));
    checkCuda(cudaMallocHost(&local, localBytes));
    initializeConcentration(local, localCount, own.begin * plane, volume);
    checkCuda(cudaMemcpyAsync(cold + begin * plane, local, localBytes, cudaMemcpyHostToDevice, compute));
    checkCuda(cudaStreamSynchronize(compute));
    if (!validate && !printResults) checkCuda(cudaFreeHost(local));

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int step = 0; step < iterations; ++step) {
        if (ranks > 1) exchange.begin(cold, compute);
        launchStencil<true>(mu, cold, cold, grid, potential.interior, compute);
        launchStencil<false>(cnew, cold, mu, grid, update.interior, compute);
        if (ranks > 1) {
            exchange.finish(cold, compute);
            launchStencil<true, 1>(mu, cold, cold, grid, potential.boundary, compute);
            launchStencil<false, 1>(cnew, cold, mu, grid, update.boundary, compute);
        }
        std::swap(cold, cnew);
    }
    checkCuda(cudaStreamSynchronize(compute));
    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", std::floor(duration * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(volume) * iterations / duration / 1e6);
    }

    bool valid = true;
    if (validate || printResults) {
        checkCuda(cudaMemcpy(local, cold + begin * plane, localBytes, cudaMemcpyDeviceToHost));
        if (printResults) gatherResults(local, own, plane, nz, rank, ranks, comm);
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            valid = validateResult(local, localCount, rank, comm);
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        checkCuda(cudaFreeHost(local));
    }
    checkCuda(cudaFree(mu));
    checkCuda(cudaFree(cnew));
    checkCuda(cudaFree(cold));
    checkCuda(cudaStreamDestroy(compute));
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

int run(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int nx = 64;
    int ny = 0;
    int nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx <= 0 || ny <= 0 || nz <= 0 || maxCells / nx / ny < static_cast<size_t>(nz)) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be positive and fit in memory.\n");
        return 1;
    }

    MPI_Comm active = MPI_COMM_WORLD;
    if (ranks > nz) {
        MPI_Comm_split(MPI_COMM_WORLD, rank < nz ? 0 : MPI_UNDEFINED, rank, &active);
    }
    int status = 0;
    if (active != MPI_COMM_NULL) {
        status = runSimulation(nx, ny, nz, iterations, validate, printResults, active);
    }
    if (ranks > nz && active != MPI_COMM_NULL) MPI_Comm_free(&active);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

}  // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (provided < MPI_THREAD_FUNNELED) abortRun("MPI_THREAD_FUNNELED is required");
    int status;
    try {
        status = run(argc, argv);
    } catch (const std::exception& error) {
        abortRun(error.what());
    }
    MPI_Finalize();
    return status;
}
