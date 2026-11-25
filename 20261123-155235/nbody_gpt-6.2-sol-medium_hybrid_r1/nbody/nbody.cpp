#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#ifdef OMPI_MAJOR_VERSION
#include <mpi-ext.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int FORCE_THREADS = 64;
constexpr int WEIGHT_TARGETS = 32;
constexpr int WEIGHT_SOURCES = 8;
constexpr size_t MAX_WEIGHTS = 4 * 1024 * 1024;
constexpr int ENERGY_THREADS = 128;

static_assert(sizeof(double3) == 3 * sizeof(double));

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

template <typename T, bool Pinned>
struct CudaDeleter {
    void operator()(T* ptr) const {
        if constexpr (Pinned) {
            cudaFreeHost(ptr);
        } else {
            cudaFree(ptr);
        }
    }
};

template <typename T, bool Pinned = false>
auto cudaBuffer(size_t count) {
    T* ptr = nullptr;
    if (count != 0) {
        if constexpr (Pinned) {
            CUDA_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
        } else {
            CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
        }
    }
    return std::unique_ptr<T, CudaDeleter<T, Pinned>>(ptr);
}

int selectDevice() {
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int devices;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        checkCuda(cudaErrorNoDevice, "Selecting a CUDA device");
    }
    // This also respects launchers that expose only the assigned GPU to each rank.
    const int device = localRank % devices;
    CUDA_CHECK(cudaSetDevice(device));
    int multiprocessors;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    return multiprocessors;
}

bool cudaAwareMpi() {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

int blocksFor(int count, int threads) {
    return count == 0 ? 0 : 1 + (count - 1) / threads;
}

[[maybe_unused]] int hostThreads(int count) {
    return std::min(omp_get_max_threads(), 1 + count / 512);
}

// Use explicit rounding for the arithmetic in the original optimized loops,
// including the separate rounding of the square root and reciprocal.
__device__ __forceinline__ double inverseDistance(double dx, double dy, double dz) {
    const double xy = __fma_rn(dx, dx, dy * dy);
    const double distanceSquared = __fma_rn(dz, dz, xy) + SOFTENING;
    return __drcp_rn(__dsqrt_rn(distanceSquared));
}

__device__ __forceinline__ double inverseDistanceCubed(double dx, double dy, double dz) {
    const double invDist = inverseDistance(dx, dy, dz);
    return invDist * invDist * invDist;
}

// Small target partitions cannot fill the GPU with one thread per target.
// Compute independent weights in parallel, then consume them in source order.
__global__ void computeWeights(const double3* __restrict__ positions, double* __restrict__ weights,
                               int total, int offset, int count) {
    const int i = blockIdx.x * WEIGHT_TARGETS + threadIdx.x;
    const int j = blockIdx.y * WEIGHT_SOURCES + threadIdx.y;
    if (i < count && j < total) {
        const double3 target = positions[offset + i];
        const double3 source = positions[j];
        weights[static_cast<size_t>(j) * count + i] =
            inverseDistanceCubed(source.x - target.x, source.y - target.y, source.z - target.z);
    }
}

template <bool Precomputed>
__device__ __forceinline__ void addForce(double3 source, double3 target, double3& force,
                                        const double* weights, int i, int j, int count) {
    const double dx = source.x - target.x;
    const double dy = source.y - target.y;
    const double dz = source.z - target.z;
    const double invDist3 = Precomputed ? weights[static_cast<size_t>(j) * count + i]
                                        : inverseDistanceCubed(dx, dy, dz);
    force.x = __fma_rn(dx, invDist3, force.x);
    force.y = __fma_rn(dy, invDist3, force.y);
    force.z = __fma_rn(dz, invDist3, force.z);
}

template <bool Precomputed>
__global__ void advanceBodies(const double3* __restrict__ positions,
                              double3* __restrict__ nextPositions,
                              double3* __restrict__ velocities,
                              const double* __restrict__ weights,
                              int total, int offset, int count) {
    __shared__ double3 sources[FORCE_THREADS];
    const int lane = threadIdx.x;
    const int i = blockIdx.x * FORCE_THREADS + lane;
    const bool active = i < count;
    const double3 target = active ? positions[offset + i] : double3{};
    double3 force{};

    int base = 0;
    while (base < total) {
        const int length = min(FORCE_THREADS, total - base);
        if (lane < length) {
            sources[lane] = positions[base + lane];
        }
        __syncthreads();

        // Each target visits the global sources in their original index order.
        if (active) {
            if (length == FORCE_THREADS) {
#pragma unroll 4
                for (int j = 0; j < FORCE_THREADS; ++j) {
                    addForce<Precomputed>(sources[j], target, force, weights, i, base + j, count);
                }
            } else {
                for (int j = 0; j < length; ++j) {
                    addForce<Precomputed>(sources[j], target, force, weights, i, base + j, count);
                }
            }
        }
        __syncthreads();
        base += length;
    }

    if (active) {
        double3 velocity = velocities[i];
        velocity.x = __dadd_rn(velocity.x, __dmul_rn(DT, force.x));
        velocity.y = __dadd_rn(velocity.y, __dmul_rn(DT, force.y));
        velocity.z = __dadd_rn(velocity.z, __dmul_rn(DT, force.z));
        velocities[i] = velocity;
        nextPositions[offset + i] = make_double3(__fma_rn(DT, velocity.x, target.x),
                                                 __fma_rn(DT, velocity.y, target.y),
                                                 __fma_rn(DT, velocity.z, target.z));
    }
}

__global__ void potentialEnergy(const double3* __restrict__ positions, double* partial,
                                int total, int offset, int count) {
    __shared__ double warpSums[ENERGY_THREADS / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int local = blockIdx.x * ENERGY_THREADS + threadIdx.x;
    double energy = 0;
    if (local < count) {
        // Pair long and short rows of the upper triangle to balance the ranks.
        const int row = offset + local;
        const int i = (row % 2 == 0) ? row / 2 : total - 1 - row / 2;
        const double3 target = positions[i];
        for (int j = i + 1; j < total; ++j) {
            const double3 source = positions[j];
            energy -= inverseDistance(source.x - target.x, source.y - target.y, source.z - target.z);
        }
    }

    for (int delta = 16; delta > 0; delta /= 2) {
        energy += __shfl_down_sync(0xffffffff, energy, delta);
    }
    if (lane == 0) {
        warpSums[warp] = energy;
    }
    __syncthreads();
    if (warp == 0) {
        energy = lane < ENERGY_THREADS / 32 ? warpSums[lane] : 0;
        for (int delta = ENERGY_THREADS / 64; delta > 0; delta /= 2) {
            energy += __shfl_down_sync(0xffffffff, energy, delta);
        }
        if (lane == 0) {
            partial[blockIdx.x] = energy;
        }
    }
}

struct Samples {
    int values[6];
};

void initializeBodies(double3* positions, double3* velocities, int total, int rank,
                      const std::vector<int>& counts, const std::vector<int>& offsets) {
    static_assert(sizeof(Samples) == 6 * sizeof(int));
    std::vector<Samples> samples(rank == 0 ? total : 0);
    if (rank == 0) {
        // rand_r has a sequential state. Generate the original stream once.
        unsigned int seed = 42;
        for (auto& body : samples) {
            for (int& value : body.values) {
                value = rand_r(&seed);
            }
        }
    }

    const int count = counts[rank];
    std::vector<Samples> local(rank == 0 ? 0 : count);
    if (counts.size() > 1) {
        MPI_Datatype sampleType;
        MPI_Type_contiguous(6, MPI_INT, &sampleType);
        MPI_Type_commit(&sampleType);
        MPI_Scatterv(samples.data(), counts.data(), offsets.data(), sampleType,
                     rank == 0 ? MPI_IN_PLACE : local.data(), count, sampleType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&sampleType);
    }

    const Samples* input = rank == 0 ? samples.data() : local.data();
    const auto coordinate = [](int value) { return 2.0 * (value / double(RAND_MAX)) - 1.0; };
#pragma omp parallel for schedule(static) num_threads(hostThreads(count))
    for (int i = 0; i < count; ++i) {
        const auto& v = input[i].values;
        positions[i] = make_double3(coordinate(v[0]), coordinate(v[1]), coordinate(v[2]));
        velocities[i] = make_double3(coordinate(v[3]), coordinate(v[4]), coordinate(v[5]));
    }
}

void exchangePositions(double3* devicePositions, double3* staging, int total, int count,
                       int offset, const std::vector<int>& counts, const std::vector<int>& offsets,
                       MPI_Datatype vectorType, cudaStream_t stream, bool stagingReady = false) {
    if (!stagingReady) {
        if (staging && count != 0) {
            CUDA_CHECK(cudaMemcpyAsync(staging + offset, devicePositions + offset,
                                       count * sizeof(double3), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    double3* received = staging ? staging : devicePositions;
    if (counts.front() == counts.back()) {
        MPI_Allgather(MPI_IN_PLACE, 0, vectorType, received, count, vectorType, MPI_COMM_WORLD);
    } else {
        MPI_Allgatherv(MPI_IN_PLACE, 0, vectorType, received, counts.data(), offsets.data(),
                       vectorType, MPI_COMM_WORLD);
    }
    if (staging) {
        CUDA_CHECK(cudaMemcpyAsync(devicePositions, staging, total * sizeof(double3),
                                   cudaMemcpyHostToDevice, stream));
        // MPI can only reuse this staging buffer after the upload completes.
        // The next exchange synchronizes this stream before calling MPI.
    }
}

void reportResults(const double3* positions, const double3* velocities, int total, int rank,
                   const std::vector<int>& counts, const std::vector<int>& offsets,
                   MPI_Datatype vectorType) {
    const int gathered = rank == 0 && counts.size() > 1 ? total : 0;
    std::vector<double3> allPositions(gathered);
    std::vector<double3> allVelocities(gathered);
    if (counts.size() > 1 && total != 0) {
        MPI_Gatherv(positions, counts[rank], vectorType, allPositions.data(),
                    counts.data(), offsets.data(), vectorType, 0, MPI_COMM_WORLD);
        MPI_Gatherv(velocities, counts[rank], vectorType, allVelocities.data(),
                    counts.data(), offsets.data(), vectorType, 0, MPI_COMM_WORLD);
        positions = allPositions.data();
        velocities = allVelocities.data();
    }
    if (rank == 0) {
        std::vector<double> bodyData(static_cast<size_t>(total) * 6);
#pragma omp parallel for schedule(static) num_threads(hostThreads(total))
        for (int i = 0; i < total; ++i) {
            const double3 p = positions[i];
            const double3 v = velocities[i];
            double* body = bodyData.data() + static_cast<size_t>(i) * 6;
            body[0] = p.x;
            body[1] = p.y;
            body[2] = p.z;
            body[3] = v.x;
            body[4] = v.y;
            body[5] = v.z;
        }
        print_results(bodyData, "Bodies");
    }
}

bool finite(double3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

bool exceedsBounds(double3 v) {
    constexpr double LIMIT = 1e6;
    return std::abs(v.x) > LIMIT || std::abs(v.y) > LIMIT || std::abs(v.z) > LIMIT;
}

int validateState(const double3* positions, const double3* velocities, int count,
                  int offset, int total, int rank, double& kinetic) {
    // Encode the first failure in body order, with the original within-body priority.
    const long long valid = 3LL * total;
    long long failure = valid;
#pragma omp parallel for schedule(static) reduction(min : failure) reduction(+ : kinetic) num_threads(hostThreads(count))
    for (int i = 0; i < count; ++i) {
        const double3 p = positions[i];
        const double3 v = velocities[i];
        int reason = -1;
        if (!finite(p) || !finite(v)) {
            reason = 0;
        } else if (exceedsBounds(p)) {
            reason = 1;
        } else if (exceedsBounds(v)) {
            reason = 2;
        }
        if (reason >= 0) {
            failure = std::min(failure, 3LL * (offset + i) + reason);
        }
        kinetic += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    long long firstFailure;
    MPI_Allreduce(&failure, &firstFailure, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
    if (firstFailure == valid) {
        return 0;
    }
    if (rank == 0) {
        const char* messages[] = {"found NaN or Inf value in body state",
                                  "body position exceeds reasonable bounds",
                                  "body velocity exceeds reasonable bounds"};
        printf("Validation failed: %s\n", messages[firstFailure % 3]);
        printf("Validation: FAILED\n");
    }
    return 1;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runSimulation(int numBodies, int numSteps, bool validate, bool printResults, int rank, int ranks) {
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts(ranks);
    std::vector<int> offsets(ranks);
    int assigned = 0;
    for (int r = 0; r < ranks; ++r) {
        offsets[r] = assigned;
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        assigned += counts[r];
    }
    const int count = counts[rank];
    const int offset = offsets[rank];

    const int multiprocessors = selectDevice();
    MPI_Datatype vectorType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vectorType);
    MPI_Type_commit(&vectorType);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const bool distributed = ranks > 1 && numBodies != 0;
    const bool exchangeNeeded = distributed && (numSteps > 0 || validate);
    auto staging = cudaBuffer<double3, true>(exchangeNeeded && !cudaAwareMpi() ? numBodies : 0);
    auto localPositions = cudaBuffer<double3, true>(staging ? 0 : count);
    auto localVelocities = cudaBuffer<double3, true>(count);
    auto first = cudaBuffer<double3>(numBodies);
    auto second = cudaBuffer<double3>(numSteps > 0 ? numBodies : 0);
    auto deviceVelocities = cudaBuffer<double3>(count);
    double3* positions = first.get();
    double3* nextPositions = second.get();
    double3* hostPositions = staging ? staging.get() + offset : localPositions.get();

    initializeBodies(hostPositions, localVelocities.get(), numBodies, rank, counts, offsets);
    if (count != 0) {
        if (!staging) {
            CUDA_CHECK(cudaMemcpyAsync(positions + offset, hostPositions, count * sizeof(double3),
                                       cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(deviceVelocities.get(), localVelocities.get(), count * sizeof(double3),
                                   cudaMemcpyHostToDevice, stream));
    }
    if (exchangeNeeded) {
        exchangePositions(positions, staging.get(), numBodies, count, offset, counts, offsets,
                          vectorType, stream, static_cast<bool>(staging));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int forceBlocks = blocksFor(count, FORCE_THREADS);
    const size_t weightCount = static_cast<size_t>(count) * numBodies;
    const int weightRows = blocksFor(numBodies, WEIGHT_SOURCES);
    const bool useWeights = numSteps > 0 && forceBlocks < 2 * multiprocessors &&
                            weightCount <= MAX_WEIGHTS && weightRows <= 65535;
    auto weights = cudaBuffer<double>(useWeights ? weightCount : 0);
    const dim3 weightGrid(blocksFor(count, WEIGHT_TARGETS), weightRows);
    const dim3 weightBlock(WEIGHT_TARGETS, WEIGHT_SOURCES);
    if (forceBlocks != 0 && numSteps > 0) {
        cudaFuncAttributes attributes;
        if (weights) {
            CUDA_CHECK(cudaFuncGetAttributes(&attributes, computeWeights));
            CUDA_CHECK(cudaFuncGetAttributes(&attributes, advanceBodies<true>));
        } else {
            CUDA_CHECK(cudaFuncGetAttributes(&attributes, advanceBodies<false>));
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps && numBodies != 0; ++step) {
        if (forceBlocks != 0) {
            if (weights) {
                computeWeights<<<weightGrid, weightBlock, 0, stream>>>(
                    positions, weights.get(), numBodies, offset, count);
                CUDA_CHECK(cudaGetLastError());
                advanceBodies<true><<<forceBlocks, FORCE_THREADS, 0, stream>>>(
                    positions, nextPositions, deviceVelocities.get(), weights.get(), numBodies, offset, count);
            } else {
                advanceBodies<false><<<forceBlocks, FORCE_THREADS, 0, stream>>>(
                    positions, nextPositions, deviceVelocities.get(), nullptr, numBodies, offset, count);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        std::swap(positions, nextPositions);
        if (distributed && step + 1 < numSteps) {
            exchangePositions(positions, staging.get(), numBodies, count, offset, counts, offsets,
                              vectorType, stream);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Simulation time: %lld ms\n", static_cast<long long>(1000 * maxElapsed));
    }

    if ((printResults || validate) && count != 0 && numSteps > 0) {
        CUDA_CHECK(cudaMemcpyAsync(hostPositions, positions + offset, count * sizeof(double3),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(localVelocities.get(), deviceVelocities.get(), count * sizeof(double3),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (printResults) {
        reportResults(hostPositions, localVelocities.get(), numBodies, rank, counts, offsets,
                      vectorType);
    }

    int status = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }
        double energy = 0;
        status = validateState(hostPositions, localVelocities.get(), count, offset,
                               numBodies, rank, energy);
        if (status == 0) {
            if (distributed && numSteps > 0) {
                exchangePositions(positions, staging.get(), numBodies, count, offset, counts, offsets,
                                  vectorType, stream, static_cast<bool>(staging));
            }
            const int energyBlocks = blocksFor(count, ENERGY_THREADS);
            if (energyBlocks != 0) {
                auto partialDevice = cudaBuffer<double>(energyBlocks);
                auto partialHost = cudaBuffer<double, true>(energyBlocks);
                potentialEnergy<<<energyBlocks, ENERGY_THREADS, 0, stream>>>(
                    positions, partialDevice.get(), numBodies, offset, count);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(partialHost.get(), partialDevice.get(),
                                           energyBlocks * sizeof(double), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                for (int i = 0; i < energyBlocks; ++i) {
                    energy += partialHost.get()[i];
                }
            }
            double totalEnergy;
            MPI_Reduce(&energy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Final energy: %.6f\n", totalEnergy);
                printf("Validation: PASSED\n");
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Type_free(&vectorType);
    return status;
}

}  // namespace

int main(int argc, char** argv) {
    int threadLevel;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &threadLevel) != MPI_SUCCESS) {
        return 1;
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (threadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int status = 0;
    bool run = true;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            run = false;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
            run = false;
            break;
        }
    }

    if (run && numBodies < 0) {
        if (rank == 0) {
            fprintf(stderr, "Number of bodies must be nonnegative\n");
        }
        status = 1;
    } else if (run) {
        status = runSimulation(numBodies, numSteps, validate, printResults, rank, ranks);
    }
    MPI_Finalize();
    return status;
}
