#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/hash.hpp"

using Real = double;
constexpr size_t transferChunk = 1 << 20;
constexpr size_t mpiChunk = INT_MAX / sizeof(Real);

struct Options {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
};

struct Slab {
    size_t firstGlobalPlane;
    size_t allocatedPlanes;
    size_t ownedOffset;
    size_t ownedPlanes;
    size_t workPlanes;
};

Slab partition(const Options& options, int rank, int ranks) {
    if (options.nx <= 2 || options.ny <= 2 || options.nz <= 2) {
        return {0, options.nz, 0, options.nz, 0};
    }

    const size_t interior = options.nz - 2;
    const size_t remainder = interior % ranks;
    const size_t first = (interior / ranks) * rank + std::min(size_t(rank), remainder);
    const size_t work = interior / ranks + (size_t(rank) < remainder);
    const bool ownsBottom = rank == 0;
    const bool ownsTop = rank == ranks - 1;
    return {first, work + 2, size_t(!ownsBottom), work + ownsBottom + ownsTop, work};
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

void selectDevice(MPI_Comm comm) {
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);

    int devices;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "stencil3d requires a visible CUDA device on each active MPI rank\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
}

int hostThreads(size_t count) {
    constexpr size_t elementsPerThread = 1 << 17;
    return std::min(omp_get_max_threads(), int(1 + (count - 1) / elementsPerThread));
}

void initializeGrid(Real* grid, size_t count, size_t globalStart, cudaStream_t stream) {
    const int slots = count > transferChunk ? 2 : 1;
    Real* staging[2];
    cudaEvent_t copied[2];
    for (int slot = 0; slot < slots; ++slot) {
        CUDA_CHECK(cudaMallocHost(&staging[slot], std::min(count, transferChunk) * sizeof(Real)));
        CUDA_CHECK(cudaEventCreateWithFlags(&copied[slot], cudaEventDisableTiming));
    }

    for (size_t offset = 0, chunk = 0; offset < count; offset += transferChunk, ++chunk) {
        const int slot = chunk % slots;
        if (chunk >= size_t(slots)) {
            CUDA_CHECK(cudaEventSynchronize(copied[slot]));
        }
        const size_t length = std::min(count - offset, transferChunk);
        Real* const host = staging[slot];
        const size_t base = globalStart + offset;
        #pragma omp parallel for schedule(static) num_threads(hostThreads(length))
        for (size_t i = 0; i < length; ++i) {
            host[i] = Real((base + i) % 19);
        }
        CUDA_CHECK(cudaMemcpyAsync(grid + offset, host, length * sizeof(Real),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(copied[slot], stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int slot = 0; slot < slots; ++slot) {
        CUDA_CHECK(cudaEventDestroy(copied[slot]));
        CUDA_CHECK(cudaFreeHost(staging[slot]));
    }
}

__device__ __forceinline__ Real average(Real sum) {
    if (sum < 0x1p-900) {
        return sum / 7.0;
    }
    // Correct the reciprocal product using its exact division residual. For
    // normal results the quotient is at least 1/14 ulp from a rounding midpoint
    // (the divisor is 7), so the last FMA rounds as division does.
    constexpr Real reciprocal = 1.0 / 7.0;
    const Real quotient = __dmul_rn(sum, reciprocal);
    const Real residual = __fma_rn(-7.0, quotient, sum);
    return __fma_rn(residual, reciprocal, quotient);
}

template <int PlanesPerBlock>
__global__ void stencil(const Real* __restrict__ input, Real* __restrict__ output,
                        size_t nx, size_t ny, size_t planeSize, size_t firstPlane,
                        size_t planes, size_t planeStep) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x == 0 || x >= nx - 1) {
        return;
    }

    // Advance a few planes per thread so the Z neighbors can stay in registers.
    const size_t tiles = (planes + PlanesPerBlock - 1) / PlanesPerBlock;
    for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
         y < ny - 1; y += size_t(gridDim.y) * blockDim.y) {
        for (size_t tile = blockIdx.z; tile < tiles; tile += gridDim.z) {
            const size_t z = firstPlane + tile * planeStep;
            const size_t remaining = planes - tile * PlanesPerBlock;
            size_t index = z * planeSize + y * nx + x;
            Real bottom = input[index - planeSize];
            Real center = input[index];
            #pragma unroll
            for (int step = 0; step < PlanesPerBlock; ++step) {
                if (size_t(step) >= remaining) {
                    break;
                }
                const Real top = input[index + planeSize];
                const Real sum = center + input[index - 1] + input[index + 1]
                               + input[index - nx] + input[index + nx] + bottom + top;
                output[index] = average(sum);
                bottom = center;
                center = top;
                index += planeSize;
            }
        }
    }
}

template <int PlanesPerBlock>
void launchStencil(const Real* input, Real* output, const Options& options,
                   size_t first, size_t count, size_t stride, cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    const dim3 threads(32, 4);
    const dim3 blocks((options.nx - 2) / threads.x + 1,
                      std::min((options.ny - 3) / threads.y + 1, size_t(65535)),
                      std::min((count - 1) / PlanesPerBlock + 1, size_t(65535)));
    stencil<PlanesPerBlock><<<blocks, threads, 0, stream>>>(
        input, output, options.nx, options.ny, options.nx * options.ny, first, count, stride);
    CUDA_CHECK(cudaGetLastError());
}

struct Halo {
    int neighbor;
    int tag;
    size_t owned;
    size_t ghost;
    Real* outgoing = nullptr;
    Real* incoming[2] = {};
};

bool deviceMpiAvailable(MPI_Comm comm) {
    int available = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    available = MPIX_Query_cuda_support() != 0;
#endif
    int everywhere;
    MPI_Allreduce(&available, &everywhere, 1, MPI_INT, MPI_MIN, comm);
    return everywhere != 0;
}

void postHalo(Real* buffer, size_t planeSize, const Halo& halo, bool receiving,
              MPI_Comm comm, std::vector<MPI_Request>& pending) {
    for (size_t offset = 0; offset < planeSize; offset += mpiChunk) {
        const int count = std::min(planeSize - offset, mpiChunk);
        pending.emplace_back();
        if (receiving) {
            MPI_Irecv(buffer + offset, count, MPI_DOUBLE, halo.neighbor, 1 - halo.tag,
                      comm, &pending.back());
        } else {
            MPI_Isend(buffer + offset, count, MPI_DOUBLE, halo.neighbor, halo.tag,
                      comm, &pending.back());
        }
    }
}

double iterate(Real*& input, Real*& output, const Options& options, const Slab& slab,
               int rank, int ranks, MPI_Comm comm, cudaStream_t compute) {
    const size_t plane = options.nx * options.ny;
    const size_t bytes = plane * sizeof(Real);
    const bool lower = rank > 0;
    const bool upper = rank + 1 < ranks;
    std::vector<Halo> halos;
    if (options.iterations > 1) {
        if (lower) halos.push_back({rank - 1, 0, plane, 0});
        if (upper) halos.push_back({rank + 1, 1, slab.workPlanes * plane, (slab.workPlanes + 1) * plane});
    }

    cudaStream_t exchange = nullptr;
    cudaEvent_t bulkReady = nullptr;
    cudaEvent_t halosReady = nullptr;
    bool direct = false;
    Real* staging = nullptr;
    std::vector<MPI_Request> pending;
    if (!halos.empty()) {
        direct = deviceMpiAvailable(comm);
        CUDA_CHECK(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&bulkReady, cudaEventDisableTiming));
        if (!direct) {
            CUDA_CHECK(cudaEventCreateWithFlags(&halosReady, cudaEventDisableTiming));
            CUDA_CHECK(cudaMallocHost(&staging, 3 * halos.size() * bytes));
            for (size_t i = 0; i < halos.size(); ++i) {
                halos[i].outgoing = staging + (3 * i) * plane;
                halos[i].incoming[0] = halos[i].outgoing + plane;
                halos[i].incoming[1] = halos[i].outgoing + 2 * plane;
            }
        }
        pending.reserve(2 * halos.size() * ((plane - 1) / mpiChunk + 1));
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    if (slab.workPlanes != 0) {
        for (int iter = 0; iter < options.iterations; ++iter) {
            if (halos.empty() || iter == options.iterations - 1) {
                launchStencil<4>(input, output, options, 1, slab.workPlanes, 4, compute);
            } else {
                // The initial ghosts are known from the global initialization. Exchange
                // each step's output ghosts while the same step computes the slab interior.
                const int slot = iter % 2;
                pending.clear();
                for (const Halo& halo : halos) {
                    Real* receive = direct ? output + halo.ghost : halo.incoming[slot];
                    postHalo(receive, plane, halo, true, comm, pending);
                }

                const size_t faceCount = slab.workPlanes == 1 ? 1 : halos.size();
                const size_t firstFace = lower ? 1 : slab.workPlanes;
                launchStencil<1>(input, output, options, firstFace, faceCount,
                                 slab.workPlanes - 1, exchange);
                if (!direct) {
                    for (const Halo& halo : halos) {
                        CUDA_CHECK(cudaMemcpyAsync(halo.outgoing, output + halo.owned, bytes,
                                                   cudaMemcpyDeviceToHost, exchange));
                    }
                }
                launchStencil<4>(input, output, options, 1 + lower, slab.workPlanes - faceCount,
                                 4, compute);
                CUDA_CHECK(cudaEventRecord(bulkReady, compute));

                CUDA_CHECK(cudaStreamSynchronize(exchange));
                for (const Halo& halo : halos) {
                    Real* send = direct ? output + halo.owned : halo.outgoing;
                    postHalo(send, plane, halo, false, comm, pending);
                }
                MPI_Waitall(pending.size(), pending.data(), MPI_STATUSES_IGNORE);

                if (!direct) {
                    // Alternate receive buffers so MPI can post the next receive
                    // while these buffers are still being read by the GPU.
                    for (const Halo& halo : halos) {
                        CUDA_CHECK(cudaMemcpyAsync(output + halo.ghost, halo.incoming[slot], bytes,
                                                   cudaMemcpyHostToDevice, exchange));
                    }
                    CUDA_CHECK(cudaEventRecord(halosReady, exchange));
                    CUDA_CHECK(cudaStreamWaitEvent(compute, halosReady, 0));
                }
                CUDA_CHECK(cudaStreamWaitEvent(exchange, bulkReady, 0));
            }
            std::swap(input, output);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    const double elapsed = MPI_Wtime() - start;
    double longest = 0.0;
    MPI_Reduce(&elapsed, &longest, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (exchange) {
        CUDA_CHECK(cudaStreamDestroy(exchange));
        CUDA_CHECK(cudaEventDestroy(bulkReady));
    }
    if (staging) {
        CUDA_CHECK(cudaEventDestroy(halosReady));
        CUDA_CHECK(cudaFreeHost(staging));
    }
    return longest;
}

struct Range {
    Real min = std::numeric_limits<Real>::infinity();
    Real max = -std::numeric_limits<Real>::infinity();
    int invalid = 0;
};

struct Results {
    Range range;
    Real sum = 0.0;
    Real correction = 0.0;
    hash digest;
    std::array<size_t, 5> indices;
    std::array<Real, 5> samples{};
    size_t nextSample = 0;
    size_t index = 0;

    explicit Results(size_t count)
        : indices{0, count / 4, count / 2, (count / 4) * 3 + (count % 4) * 3 / 4, count - 1} {}

    void append(const Real* data, size_t count, bool validate) {
        for (size_t i = 0; i < count; ++i, ++index) {
            const Real value = data[i];
            const Real y = value - correction;
            const Real t = sum + y;
            correction = (t - sum) - y;
            sum = t;
            range.min = std::min(range.min, value);
            range.max = std::max(range.max, value);
            range.invalid |= validate && !std::isfinite(value);
            digest.add(value);
            while (nextSample < indices.size() && indices[nextSample] == index) {
                samples[nextSample++] = value;
            }
        }
    }

    void print() const {
        printf("=== RESULTS ===\nName: Grid\nElements: %zu\n", index);
        printf("Sum: %.17e\nMin: %.17e\nMax: %.17e\n", sum, range.min, range.max);
        for (size_t i = 0; i < indices.size(); ++i) {
            printf("Sample[%zu]: %.17e\n", indices[i], samples[i]);
        }
        printf("Hash: %016" PRIx64 "\n=== END RESULTS ===\n", digest.get());
    }
};

Range printResults(const Real* grid, const Options& options, int rank, int ranks, MPI_Comm comm) {
    const size_t plane = options.nx * options.ny;
    Results results(plane * options.nz);
    std::vector<Real> buffer(std::min(transferChunk, plane * options.nz));
    // Read the slabs in global order so the Kahan sum and FNV-1a hash match
    // the serial output, with bounded host memory even on the root rank.
    for (int owner = rank; owner < (rank == 0 ? ranks : rank + 1); ++owner) {
        const Slab slab = partition(options, owner, ranks);
        const size_t count = slab.ownedPlanes * plane;
        for (size_t offset = 0; offset < count; offset += transferChunk) {
            const size_t length = std::min(count - offset, transferChunk);
            if (owner == rank) {
                CUDA_CHECK(cudaMemcpy(buffer.data(), grid + slab.ownedOffset * plane + offset,
                                      length * sizeof(Real), cudaMemcpyDeviceToHost));
                if (rank != 0) {
                    MPI_Send(buffer.data(), length, MPI_DOUBLE, 0, 0, comm);
                }
            } else {
                MPI_Recv(buffer.data(), length, MPI_DOUBLE, owner, 0, comm, MPI_STATUS_IGNORE);
            }
            if (rank == 0) {
                results.append(buffer.data(), length, options.validate);
            }
        }
    }
    if (rank == 0) results.print();
    return results.range;
}

Range validateGrid(const Real* grid, size_t count, MPI_Comm comm) {
    std::vector<Real> buffer(std::min(transferChunk, count));
    Range local;
    for (size_t offset = 0; offset < count; offset += transferChunk) {
        const size_t length = std::min(count - offset, transferChunk);
        CUDA_CHECK(cudaMemcpy(buffer.data(), grid + offset, length * sizeof(Real), cudaMemcpyDeviceToHost));
        Real min = local.min;
        Real max = local.max;
        int invalid = local.invalid;
        #pragma omp parallel for schedule(static) num_threads(hostThreads(length)) reduction(min:min) reduction(max:max) reduction(|:invalid)
        for (size_t i = 0; i < length; ++i) {
            const Real value = buffer[i];
            invalid |= !std::isfinite(value);
            min = std::min(min, value);
            max = std::max(max, value);
        }
        local = {min, max, invalid};
    }
    Range global;
    MPI_Reduce(&local.min, &global.min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local.max, &global.max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local.invalid, &global.invalid, 1, MPI_INT, MPI_LOR, 0, comm);
    return global;
}

int runBenchmark(const Options& options, int rank, int ranks, MPI_Comm comm) {
    selectDevice(comm);
    const Slab slab = partition(options, rank, ranks);
    const size_t plane = options.nx * options.ny;
    const size_t count = plane * slab.allocatedPlanes;
    Real* input;
    Real* output = nullptr;
    cudaStream_t compute;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&input, count * sizeof(Real)));

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", options.nx, options.ny, options.nz);
        printf("Iterations: %d\n", options.iterations);
        printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    initializeGrid(input, count, slab.firstGlobalPlane * plane, compute);
    if (options.iterations > 0 && slab.workPlanes > 0) {
        CUDA_CHECK(cudaMalloc(&output, count * sizeof(Real)));
        // Both buffers keep the physical boundaries for every ping-pong step.
        CUDA_CHECK(cudaMemcpyAsync(output, input, count * sizeof(Real), cudaMemcpyDeviceToDevice, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }

    const double seconds = iterate(input, output, options, slab, rank, ranks, comm, compute);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        const double updates = double(slab.workPlanes == 0 ? 0 :
                                      (options.nx - 2) * (options.ny - 2) * (options.nz - 2))
                               * options.iterations;
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0.0);
    }

    Range range;
    if (options.printResults) {
        range = printResults(input, options, rank, ranks, comm);
    }
    if (options.validate && !options.printResults) {
        if (rank == 0) printf("Validating result...\n");
        range = validateGrid(input + slab.ownedOffset * plane, slab.ownedPlanes * plane, comm);
    }

    int exitCode = 0;
    if (options.validate && rank == 0) {
        if (options.printResults) printf("Validating result...\n");
        if (range.invalid) {
            printf("Validation failed: found NaN or Inf value\n");
        } else {
            printf("Value range: [%.6f, %.6f]\n", range.min, range.max);
            if (range.min < -1e6 || range.max > 1e6) {
                printf("Validation failed: values out of expected range\n");
            }
        }
        exitCode = range.invalid || range.min < -1e6 || range.max > 1e6;
        printf("Validation: %s\n", exitCode ? "FAILED" : "PASSED");
    }
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaStreamDestroy(compute));
    return exitCode;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

bool parseNumber(const char* text, size_t& value) {
    char* end;
    errno = 0;
    const unsigned long long number = std::strtoull(text, &end, 10);
    if (errno || end == text || *end != '\0' || number > INT_MAX) {
        return false;
    }
    value = number;
    return true;
}

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    Options options;
    for (int i = 1; i < argc; ++i) {
        const char* flag = argv[i];
        if (strcmp(flag, "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        }
        if (strcmp(flag, "-v") == 0) {
            options.validate = true;
        } else if (strcmp(flag, "-r") == 0) {
            options.printResults = true;
        } else if (strcmp(flag, "-x") == 0 || strcmp(flag, "-y") == 0 ||
                   strcmp(flag, "-z") == 0 || strcmp(flag, "-i") == 0) {
            size_t value;
            if (i + 1 < argc && parseNumber(argv[++i], value)) {
                if (strcmp(flag, "-x") == 0) options.nx = value;
                else if (strcmp(flag, "-y") == 0) options.ny = value;
                else if (strcmp(flag, "-z") == 0) options.nz = value;
                else options.iterations = value;
            } else {
                if (rank == 0) {
                    printf("Invalid value for %s\n", flag);
                    printUsage(argv[0]);
                }
                MPI_Finalize();
                return 1;
            }
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", flag);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (options.ny == 0) options.ny = options.nx;
    if (options.nz == 0) options.nz = options.nx;
    if (options.nx == 0 ||
        options.nx > std::numeric_limits<size_t>::max() / sizeof(Real) / options.ny / options.nz) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions\n");
        MPI_Finalize();
        return 1;
    }

    const bool hasInterior = options.nx > 2 && options.ny > 2 && options.nz > 2;
    const int active = hasInterior ? std::min(size_t(ranks), options.nz - 2) : 1;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    int status = 0;
    if (comm != MPI_COMM_NULL) {
        status = runBenchmark(options, rank, active, comm);
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
