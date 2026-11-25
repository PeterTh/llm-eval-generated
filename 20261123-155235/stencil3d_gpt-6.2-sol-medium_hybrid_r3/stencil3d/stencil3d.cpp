#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif
#include <omp.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

constexpr int haloTag = 0;
constexpr int resultTag = 1;

[[noreturn]] void abortBenchmark(const char* message) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) abortBenchmark(cudaGetErrorString(status));
}

bool cudaAwareMpi() {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

struct Slab {
    size_t first;  // Global index of the first owned plane
    size_t count;
    size_t begin;  // Local half-open range of updatable planes, after the lower ghost
    size_t end;
};

Slab splitGrid(size_t nz, bool hasInterior, int rank, int ranks) {
    const size_t planes = hasInterior ? nz - 2 : nz;
    const size_t quotient = planes / ranks;
    const size_t remainder = planes % ranks;
    const size_t offset = rank * quotient + std::min<size_t>(rank, remainder);
    const size_t count = quotient + (static_cast<size_t>(rank) < remainder);
    if (count == 0 || !hasInterior) return {offset, count, 1, 1};

    const bool lowerFace = rank == 0;
    const bool upperFace = offset + count == planes;
    const size_t begin = 1 + lowerFace;
    return {offset + !lowerFace, count + lowerFace + upperFace, begin, begin + count};
}

constexpr unsigned tileX = 32;
constexpr unsigned tileY = 4;
constexpr unsigned tileZ = 4;
constexpr unsigned maxGridYZ = 65535;

__device__ Real divideBySeven(Real sum) {
    // Correct the rounded reciprocal product with its FMA residual. For the
    // stencil's bounded sums, the correction error is far smaller than the
    // distance of sum/7 from a rounding midpoint (at least 1/14 ulp).
    constexpr Real reciprocal = 1.0 / 7.0;
    const Real quotient = __dmul_rn(sum, reciprocal);
    const Real residual = __fma_rn(-7.0, quotient, sum);
    return __fma_rn(residual, reciprocal, quotient);
}

template <bool Halo>
__global__ void stencil(const Real* __restrict__ input, Real* __restrict__ output,
                        size_t nx, size_t ny, size_t plane, size_t firstZ,
                        size_t lastZ, size_t zTiles) {
    const size_t x = blockIdx.x * static_cast<size_t>(tileX) + threadIdx.x + 1;
    if (x >= nx - 1) return;

    // Adjacent threads traverse X. Each thread reuses two Z neighbors as it advances.
    for (size_t y = blockIdx.y * static_cast<size_t>(tileY) + threadIdx.y + 1;
         y < ny - 1; y += gridDim.y * tileY) {
        for (size_t zTile = blockIdx.z; zTile < zTiles; zTile += gridDim.z) {
            const size_t z = Halo ? (zTile == 0 ? firstZ : lastZ) : firstZ + zTile * tileZ;
            const size_t remaining = lastZ - z + 1;
            const unsigned depth = Halo ? 1 : (remaining < tileZ ? remaining : tileZ);
            size_t index = z * plane + y * nx + x;

            Real bottom = input[index - plane];
            Real center = input[index];
            for (unsigned dz = 0; dz < depth; ++dz) {
                const Real left = input[index - 1];
                const Real right = input[index + 1];
                const Real front = input[index - nx];
                const Real back = input[index + nx];
                const Real top = input[index + plane];
                output[index] = divideBySeven(center + left + right + front + back + bottom + top);
                bottom = center;
                center = top;
                index += plane;
            }
        }
    }
}

template <bool Halo>
void launchStencil(const Real* input, Real* output, size_t nx, size_t ny, size_t plane,
                   size_t begin, size_t end, cudaStream_t stream) {
    if (begin >= end) return;
    const size_t depth = end - begin;
    const size_t zTiles = Halo ? 1 + (depth > 1) : (depth + tileZ - 1) / tileZ;
    const size_t yTiles = (ny - 2 + tileY - 1) / tileY;
    const dim3 blocks((nx - 2 + tileX - 1) / tileX, std::min<size_t>(yTiles, maxGridYZ),
                      std::min<size_t>(zTiles, maxGridYZ));
    stencil<Halo><<<blocks, dim3(tileX, tileY), 0, stream>>>(
        input, output, nx, ny, plane, begin, end - 1, zTiles);
    checkCuda(cudaGetLastError());
}

struct HaloPlane {
    int neighbor;
    size_t edge;
    size_t ghost;
    Real* send = nullptr;
    std::array<Real*, 2> receive{};
};

// MPI-3 uses int element counts. Split large messages without narrowing their size.
template <typename Operation>
void mpiChunks(size_t count, Operation operation) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min<size_t>(count - offset,
                                                           std::numeric_limits<int>::max()));
        operation(offset, chunk);
        offset += chunk;
    }
}

void iterate(Real*& input, Real*& output, const Slab& slab, size_t nx, size_t ny,
             int iterations, int rank, size_t nz, bool direct, MPI_Comm comm) {
    const size_t plane = nx * ny;
    const size_t bytes = plane * sizeof(Real);
    std::array<HaloPlane, 2> halos;
    size_t haloCount = 0;
    if (slab.first != 0) halos[haloCount++] = {rank - 1, slab.begin, 0};
    if (slab.first + slab.count != nz) {
        halos[haloCount++] = {rank + 1, slab.end - 1, slab.count + 1};
    }

    cudaStream_t interiorStream;
    checkCuda(cudaStreamCreateWithFlags(&interiorStream, cudaStreamNonBlocking));
    cudaStream_t haloStream = nullptr;
    cudaEvent_t interiorReady = nullptr;
    if (haloCount != 0) {
        int highPriority;
        checkCuda(cudaDeviceGetStreamPriorityRange(nullptr, &highPriority));
        checkCuda(cudaStreamCreateWithPriority(&haloStream, cudaStreamNonBlocking, highPriority));
        checkCuda(cudaEventCreateWithFlags(&interiorReady, cudaEventDisableTiming));
    }

    for (size_t side = 0; side < haloCount && !direct && iterations > 1; ++side) {
        checkCuda(cudaMallocHost(&halos[side].send, bytes));
        for (Real*& receive : halos[side].receive) checkCuda(cudaMallocHost(&receive, bytes));
    }

    const size_t haloBegin = slab.first == 0 ? slab.end - 1 : slab.begin;
    const size_t haloEnd = slab.first + slab.count == nz ? slab.begin + 1 : slab.end;
    const size_t interiorBegin = slab.begin + (slab.first != 0);
    const size_t interiorEnd = slab.end - (slab.first + slab.count != nz);
    std::vector<MPI_Request> requests;
    requests.reserve(2 * haloCount * (1 + (plane - 1) / std::numeric_limits<int>::max()));

    // Resolve lazy CUDA kernel loading before measuring the iteration loop.
    cudaFuncAttributes attributes;
    checkCuda(cudaFuncGetAttributes(&attributes, stencil<false>));
    if (haloStream) checkCuda(cudaFuncGetAttributes(&attributes, stencil<true>));
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int step = 0; step < iterations; ++step) {
        if (haloCount == 0) {
            launchStencil<false>(input, output, nx, ny, plane, slab.begin, slab.end, interiorStream);
        } else {
            const bool exchange = step + 1 < iterations;
            const size_t receiveSlot = step % 2;
            if (exchange) {
                requests.clear();
                for (size_t side = 0; side < haloCount; ++side) {
                    const HaloPlane& halo = halos[side];
                    Real* receive = direct ? output + halo.ghost * plane : halo.receive[receiveSlot];
                    mpiChunks(plane, [&](size_t offset, int count) {
                        requests.emplace_back();
                        MPI_Irecv(receive + offset, count, MPI_DOUBLE, halo.neighbor, haloTag,
                                  comm, &requests.back());
                    });
                }
            }

            // The edge kernel reads the new halos and the previous step's adjacent
            // interior planes. Its own stream orders it after any host-to-device copies.
            if (step != 0) checkCuda(cudaStreamWaitEvent(haloStream, interiorReady, 0));
            launchStencil<true>(input, output, nx, ny, plane, haloBegin, haloEnd, haloStream);
            if (exchange && !direct) {
                for (size_t side = 0; side < haloCount; ++side) {
                    const HaloPlane& halo = halos[side];
                    checkCuda(cudaMemcpyAsync(halo.send, output + halo.edge * plane, bytes,
                                              cudaMemcpyDeviceToHost, haloStream));
                }
            }

            launchStencil<false>(input, output, nx, ny, plane, interiorBegin, interiorEnd,
                                 interiorStream);
            checkCuda(cudaEventRecord(interiorReady, interiorStream));

            if (exchange) {
                checkCuda(cudaStreamSynchronize(haloStream));
                for (size_t side = 0; side < haloCount; ++side) {
                    const HaloPlane& halo = halos[side];
                    const Real* send = direct ? output + halo.edge * plane : halo.send;
                    mpiChunks(plane, [&](size_t offset, int count) {
                        requests.emplace_back();
                        MPI_Isend(send + offset, count, MPI_DOUBLE, halo.neighbor, haloTag,
                                  comm, &requests.back());
                    });
                }
                MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
                if (!direct) {
                    for (size_t side = 0; side < haloCount; ++side) {
                        const HaloPlane& halo = halos[side];
                        checkCuda(cudaMemcpyAsync(output + halo.ghost * plane, halo.receive[receiveSlot],
                                                  bytes, cudaMemcpyHostToDevice, haloStream));
                    }
                    // The next receive uses the alternate pinned buffer. The next
                    // halo stream sync will retire this copy before that slot is reused.
                }
            }
        }
        std::swap(input, output);
    }
    checkCuda(cudaStreamSynchronize(interiorStream));
    if (haloStream) checkCuda(cudaStreamSynchronize(haloStream));

    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const double updates = static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations;
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f MCellUpdates/s\n", updates / (seconds * 1e6));
    }

    for (size_t side = 0; side < haloCount && !direct && iterations > 1; ++side) {
        checkCuda(cudaFreeHost(halos[side].send));
        for (Real* receive : halos[side].receive) checkCuda(cudaFreeHost(receive));
    }
    if (haloStream) {
        checkCuda(cudaEventDestroy(interiorReady));
        checkCuda(cudaStreamDestroy(haloStream));
    }
    checkCuda(cudaStreamDestroy(interiorStream));
}

void initialize(Real* input, Real* output, const Slab& slab, size_t plane, size_t nz,
                int threads) {
    // Populate real neighbor planes from the same global initial condition. Both
    // buffers retain their original physical boundary values throughout the run.
    const size_t lowerGhost = slab.first != 0;
    const size_t upperGhost = slab.first + slab.count != nz;
    const size_t initialized = (slab.count + lowerGhost + upperGhost) * plane;
    const size_t globalStart = (slab.first - lowerGhost) * plane;
    const size_t deviceStart = (1 - lowerGhost) * plane;
    const size_t chunkSize = std::min<size_t>(initialized, 1 << 20);
    Real* staging;
    checkCuda(cudaMallocHost(&staging, chunkSize * sizeof(Real)));
    for (size_t offset = 0; offset < initialized; offset += chunkSize) {
        const size_t count = std::min(chunkSize, initialized - offset);
        #pragma omp parallel for num_threads(threads) schedule(static) if(count >= 8192)
        for (size_t i = 0; i < count; ++i) {
            staging[i] = (globalStart + offset + i) % 19;
        }
        checkCuda(cudaMemcpy(input + deviceStart + offset, staging, count * sizeof(Real),
                             cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMemcpy(output + deviceStart, input + deviceStart, initialized * sizeof(Real),
                         cudaMemcpyDeviceToDevice));
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFreeHost(staging));
}

bool validateResult(const std::vector<Real>& local, int rank, int threads, MPI_Comm comm) {
    Real minValue = std::numeric_limits<Real>::infinity();
    Real maxValue = -minValue;
    int nonfinite = 0;
    #pragma omp parallel for num_threads(threads) schedule(static) if(local.size() >= 8192) \
        reduction(min:minValue) reduction(max:maxValue) reduction(|:nonfinite)
    for (size_t i = 0; i < local.size(); ++i) {
        const Real value = local[i];
        nonfinite |= !std::isfinite(value);
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }
    MPI_Allreduce(MPI_IN_PLACE, &nonfinite, 1, MPI_INT, MPI_BOR, comm);
    MPI_Allreduce(MPI_IN_PLACE, &minValue, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &maxValue, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (nonfinite) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    const bool valid = maxValue <= 1e6 && minValue >= -1e6;
    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
        if (!valid) printf("Validation failed: values out of expected range\n");
    }
    return valid;
}

void reportResults(std::vector<Real>& local, size_t plane, size_t nz, bool hasInterior,
                   int rank, int ranks, MPI_Comm comm) {
    if (ranks == 1) {
        print_results(local, "Grid");
    } else if (rank == 0) {
        const size_t localSize = local.size();
        local.resize(plane * nz);
        for (int source = 1; source < ranks; ++source) {
            const Slab slab = splitGrid(nz, hasInterior, source, ranks);
            mpiChunks(slab.count * plane, [&](size_t offset, int count) {
                MPI_Recv(local.data() + slab.first * plane + offset, count, MPI_DOUBLE,
                         source, resultTag, comm, MPI_STATUS_IGNORE);
            });
        }
        print_results(local, "Grid");
        local.resize(localSize);
    } else {
        mpiChunks(local.size(), [&](size_t offset, int count) {
            MPI_Send(local.data() + offset, count, MPI_DOUBLE, 0, resultTag, comm);
        });
    }
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

template <typename Integer>
bool parseNumber(const char* text, Integer& value) {
    if (*text == '+') ++text;
    const char* end = text + strlen(text);
    const auto result = std::from_chars(text, end, value);
    return result.ec == std::errc{} && result.ptr == end;
}

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        const char* option = argv[i];
        bool valid;
        if (strcmp(option, "-x") == 0) {
            valid = ++i < argc && parseNumber(argv[i], nx);
        } else if (strcmp(option, "-y") == 0) {
            valid = ++i < argc && parseNumber(argv[i], ny);
        } else if (strcmp(option, "-z") == 0) {
            valid = ++i < argc && parseNumber(argv[i], nz);
        } else if (strcmp(option, "-i") == 0) {
            valid = ++i < argc && parseNumber(argv[i], iterations) && iterations >= 0;
        } else if (strcmp(option, "-v") == 0) {
            validate = valid = true;
        } else if (strcmp(option, "-r") == 0) {
            printResults = valid = true;
        } else if (strcmp(option, "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            valid = false;
        }
        if (!valid) {
            if (rank == 0) {
                printf("Invalid option or value: %s\n", option);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx == 0 || nx > maxElements / ny || nx * ny > maxElements / nz) {
        if (rank == 0) fprintf(stderr, "Grid dimensions are zero or too large\n");
        return 1;
    }
    const size_t plane = nx * ny;
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;

    // Drop ranks with no owned planes. The remaining ranks are contiguous in
    // world order, so local slab neighbors retain their world rank numbers.
    const int activeRanks = static_cast<int>(std::min<size_t>(ranks, hasInterior ? nz - 2 : nz));
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &active);
    if (active == MPI_COMM_NULL) return 0;

    const Slab slab = splitGrid(nz, hasInterior, rank, activeRanks);
    if (slab.count + 2 > maxElements / plane) abortBenchmark("Local slab is too large");

    MPI_Comm local;
    MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);
    int devices;
    checkCuda(cudaGetDeviceCount(&devices));
    if (devices == 0) abortBenchmark("No CUDA devices are visible");
    checkCuda(cudaSetDevice(localRank % devices));

    const int threads = std::getenv("OMP_NUM_THREADS") ? omp_get_max_threads()
                                                      : std::min(8, omp_get_max_threads());
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    Real* input;
    Real* output;
    const size_t bytes = (slab.count + 2) * plane * sizeof(Real);
    checkCuda(cudaMalloc(&input, bytes));
    checkCuda(cudaMalloc(&output, bytes));
    initialize(input, output, slab, plane, nz, threads);

    if (rank == 0) printf("Running stencil computation...\n");
    if (hasInterior && iterations != 0) {
        const bool direct = cudaAwareMpi();
        iterate(input, output, slab, nx, ny, iterations, rank, nz, direct, active);
    } else if (rank == 0) {
        printf("Computation time: 0.000 ms\n");
        printf("Performance: 0.000 MCellUpdates/s\n");
    }

    std::vector<Real> result;
    if (printResults || validate) {
        result.resize(slab.count * plane);
        checkCuda(cudaMemcpy(result.data(), input + plane, result.size() * sizeof(Real),
                             cudaMemcpyDeviceToHost));
    }
    checkCuda(cudaFree(input));
    checkCuda(cudaFree(output));

    bool valid = true;
    if (printResults) reportResults(result, plane, nz, hasInterior, rank, activeRanks, active);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateResult(result, rank, threads, active);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Comm_free(&active);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    if (provided < MPI_THREAD_FUNNELED) abortBenchmark("MPI_THREAD_FUNNELED is required");
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int status;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        abortBenchmark(error.what());
    }
    MPI_Finalize();
    return status;
}
