#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#ifdef OMPI_MAJOR_VERSION
#include <mpi-ext.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int SOURCE_TILE_SIZE = 128;
constexpr int ENERGY_THREADS = 128;

struct Vec3 {
    double x, y, z;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double));

int worldRank;

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA error: %s\n", worldRank, cudaGetErrorString(result));
        if (!omp_in_parallel()) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::abort();
    }
}

template <typename T>
T* allocateDevice(size_t count) {
    T* data = nullptr;
    checkCuda(cudaMalloc(&data, count * sizeof(T)));
    return data;
}

template <typename T>
T* allocateHost(size_t count) {
    T* data = nullptr;
    checkCuda(cudaHostAlloc(&data, std::max(count, size_t{1}) * sizeof(T), cudaHostAllocPortable));
    return data;
}

int blocksFor(int count, int threads) {
    return count / threads + (count % threads != 0);
}

__device__ __forceinline__ double inverseDistance(double dx, double dy, double dz) {
    // Match the double-precision arithmetic of the optimized serial force loop.
    const double distanceSquared = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
    return 1.0 / sqrt(distanceSquared);
}

__device__ __forceinline__ void accumulateForce(Vec3 source, Vec3 target, Vec3& force) {
    const double dx = source.x - target.x;
    const double dy = source.y - target.y;
    const double dz = source.z - target.z;
    const double inverse = inverseDistance(dx, dy, dz);
    const double inverseCubed = inverse * inverse * inverse;
    force.x = fma(dx, inverseCubed, force.x);
    force.y = fma(dy, inverseCubed, force.y);
    force.z = fma(dz, inverseCubed, force.z);
}

template <int Threads>
__global__ void advance(const Vec3* __restrict__ positions, Vec3* __restrict__ next,
                        Vec3* __restrict__ velocities, int total, int first, int count) {
    __shared__ Vec3 sources[SOURCE_TILE_SIZE];
    const int thread = threadIdx.x;
    const int targetIndex = blockIdx.x * Threads + thread;
    const bool active = targetIndex < count;
    const Vec3 target = active ? positions[first + targetIndex] : Vec3{};
    Vec3 force{};

    const int fullTiles = total / SOURCE_TILE_SIZE;
    for (int tile = 0; tile < fullTiles; ++tile) {
        #pragma unroll
        for (int j = thread; j < SOURCE_TILE_SIZE; j += Threads) {
            sources[j] = positions[tile * SOURCE_TILE_SIZE + j];
        }
        __syncthreads();
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < SOURCE_TILE_SIZE; ++j) {
                accumulateForce(sources[j], target, force);
            }
        }
        __syncthreads();
    }

    const int tail = total % SOURCE_TILE_SIZE;
    if (tail > 0) {
        for (int j = thread; j < tail; j += Threads) {
            sources[j] = positions[fullTiles * SOURCE_TILE_SIZE + j];
        }
        __syncthreads();
        if (active) {
            for (int j = 0; j < tail; ++j) {
                accumulateForce(sources[j], target, force);
            }
        }
    }

    if (active) {
        Vec3 velocity = velocities[targetIndex];
        velocity.x += DT * force.x;
        velocity.y += DT * force.y;
        velocity.z += DT * force.z;
        velocities[targetIndex] = velocity;
        next[targetIndex] = {fma(DT, velocity.x, target.x), fma(DT, velocity.y, target.y),
                             fma(DT, velocity.z, target.z)};
    }
}

__global__ void bodyEnergies(const Vec3* __restrict__ positions,
                             const Vec3* __restrict__ velocities,
                             double* __restrict__ energies, int total, int first, int count) {
    const int local = blockIdx.x * ENERGY_THREADS + threadIdx.x;
    if (local >= count) {
        return;
    }

    const int i = first + local;
    const Vec3 p = positions[i];
    const Vec3 v = velocities[local];
    double energy = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);

    // Count each pair once with a balanced number of partners per target.
    // For even totals, the first half also counts the diametrically opposite pair.
    int partners = (total - 1) / 2 + (total % 2 == 0 && i < total / 2);
    const int beforeWrap = min(partners, total - i - 1);
    for (int j = i + 1; j < i + 1 + beforeWrap; ++j) {
        const Vec3 q = positions[j];
        energy -= inverseDistance(q.x - p.x, q.y - p.y, q.z - p.z);
    }
    partners -= beforeWrap;
    for (int j = 0; j < partners; ++j) {
        const Vec3 q = positions[j];
        energy -= inverseDistance(q.x - p.x, q.y - p.y, q.z - p.z);
    }
    energies[local] = energy;
}

struct GpuWork {
    int device;
    int first;
    int count;
    int forceThreads = 32;
    cudaStream_t stream = nullptr;
    Vec3* positions = nullptr;
    Vec3* next = nullptr;
    Vec3* velocities = nullptr;
};

template <int Threads>
void launchAdvance(const GpuWork& gpu, int total) {
    advance<Threads><<<blocksFor(gpu.count, Threads), Threads, 0, gpu.stream>>>(
        gpu.positions, gpu.next, gpu.velocities, total, gpu.first, gpu.count);
}

template <typename Operation>
void onGpus(std::vector<GpuWork>& workers, int threads, const Operation& task) {
    #pragma omp parallel for num_threads(threads) if(threads > 1)
    for (size_t g = 0; g < workers.size(); ++g) {
        auto& gpu = workers[g];
        checkCuda(cudaSetDevice(gpu.device));
        task(gpu);
    }
}

bool cudaAwareMpi() {
#ifdef MPIX_CUDA_AWARE_SUPPORT
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

int hostThreads(int count) {
    // Keep the OpenMP team proportional to the amount of linear host work.
    return std::max(1, std::min(omp_get_max_threads(), count / 4096));
}

// The MPI rank owns a contiguous range, split into contiguous GPU ranges.
int partitionPoint(int size, int partitions, int index) {
    return (size / partitions) * index + std::min(index, size % partitions);
}

void initializeBodies(Vec3* positions, Vec3* velocities, int count) {
    unsigned int seed = 42;
    auto sample = [&seed] { return 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0; };
    for (int i = 0; i < count; ++i) {
        positions[i] = {sample(), sample(), sample()};
        velocities[i] = {sample(), sample(), sample()};
    }
}

int validationIssue(Vec3 p, Vec3 v) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
        return 1;
    }
    if (std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6) {
        return 2;
    }
    if (std::abs(v.x) > 1e6 || std::abs(v.y) > 1e6 || std::abs(v.z) > 1e6) {
        return 3;
    }
    return 0;
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

int run(int argc, char** argv, int rank, int ranks) {
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (numBodies < 0) {
        if (rank == 0) {
            fprintf(stderr, "Number of bodies must be nonnegative\n");
        }
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribute visible GPUs among ranks on the same node. A rank with a single
    // scheduler-assigned device always uses that device.
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, localRanks;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localRanks);
    MPI_Comm_free(&node);

    int visibleDevices = 0;
    checkCuda(cudaGetDeviceCount(&visibleDevices));
    if (visibleDevices == 0) {
        checkCuda(cudaErrorNoDevice);
    }
    const int firstDevice = localRank % visibleDevices;
    const int assignedDevices = 1 + (visibleDevices - 1 - firstDevice) / localRanks;
    std::vector<int> devicesPerRank(ranks);
    MPI_Allgather(&assignedDevices, 1, MPI_INT, devicesPerRank.data(), 1, MPI_INT, MPI_COMM_WORLD);
    const int totalDevices = std::accumulate(devicesPerRank.begin(), devicesPerRank.end(), 0);

    std::vector<int> offsets(ranks), counts(ranks);
    int deviceOffset = 0;
    for (int r = 0; r < ranks; ++r) {
        offsets[r] = partitionPoint(numBodies, totalDevices, deviceOffset);
        deviceOffset += devicesPerRank[r];
        counts[r] = partitionPoint(numBodies, totalDevices, deviceOffset) - offsets[r];
    }
    const int first = offsets[rank];
    const int localCount = counts[rank];
    const bool singleGpu = totalDevices == 1;
    const int rankDeviceOffset = std::accumulate(devicesPerRank.begin(), devicesPerRank.begin() + rank, 0);
    std::vector<GpuWork> workers;
    for (int g = 0; g < assignedDevices; ++g) {
        const int begin = partitionPoint(numBodies, totalDevices, rankDeviceOffset + g);
        const int end = partitionPoint(numBodies, totalDevices, rankDeviceOffset + g + 1);
        if (begin != end) {
            workers.push_back({firstDevice + g * localRanks, begin, end - begin});
        }
    }
    const int gpuThreads = std::max(1, std::min(static_cast<int>(workers.size()), omp_get_max_threads()));

    // CUDA-aware MPI can exchange device buffers directly when each rank owns
    // one GPU. Otherwise use two portable pinned snapshots for DMA and MPI.
    int deviceExchange = 0;
    if (ranks > 1 && totalDevices == ranks && numBodies >= ranks && numSteps > 1) {
        const int localSupport = cudaAwareMpi();
        MPI_Allreduce(&localSupport, &deviceExchange, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }
    const bool stagePositions = !singleGpu && !deviceExchange;
    const bool equalCounts = std::all_of(counts.begin(), counts.end(),
                                         [localCount](int count) { return count == localCount; });
    checkCuda(cudaSetDevice(firstDevice));
    Vec3* positions = allocateHost<Vec3>(numBodies);
    Vec3* nextPositions = stagePositions && numSteps > 1 && numBodies > 0
                              ? allocateHost<Vec3>(numBodies) : nullptr;
    Vec3* const velocities = allocateHost<Vec3>(localCount);
    std::vector<Vec3> rootVelocities(rank == 0 && ranks > 1 ? numBodies : 0);
    MPI_Datatype vectorType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vectorType);
    MPI_Type_commit(&vectorType);

    const auto exchangePositions = [&](const void* local, Vec3* all) {
        if (equalCounts) {
            MPI_Allgather(local, localCount, vectorType, all, localCount, vectorType, MPI_COMM_WORLD);
        } else {
            MPI_Allgatherv(local, localCount, vectorType, all, counts.data(),
                           offsets.data(), vectorType, MPI_COMM_WORLD);
        }
    };

    if (rank == 0) {
        initializeBodies(positions, ranks == 1 ? velocities : rootVelocities.data(), numBodies);
    }
    if (ranks > 1) {
        MPI_Bcast(positions, numBodies, vectorType, 0, MPI_COMM_WORLD);
        MPI_Scatterv(rootVelocities.data(), counts.data(), offsets.data(), vectorType,
                     velocities, localCount, vectorType, 0, MPI_COMM_WORLD);
        if (!printResults) {
            std::vector<Vec3>().swap(rootVelocities);
        }
    }

    onGpus(workers, gpuThreads, [&](GpuWork& gpu) {
        int multiprocessors;
        checkCuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, gpu.device));
        // Smaller blocks spread short target ranges over more multiprocessors.
        if (gpu.count >= 256 * multiprocessors) {
            gpu.forceThreads = 128;
        } else if (gpu.count >= 64 * multiprocessors) {
            gpu.forceThreads = 64;
        }
        checkCuda(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking));
        gpu.positions = allocateDevice<Vec3>(numBodies);
        gpu.next = allocateDevice<Vec3>(gpu.count);
        gpu.velocities = allocateDevice<Vec3>(gpu.count);
        checkCuda(cudaMemcpyAsync(gpu.positions, positions, numBodies * sizeof(Vec3),
                                  cudaMemcpyHostToDevice, gpu.stream));
        checkCuda(cudaMemcpyAsync(gpu.velocities, velocities + (gpu.first - first),
                                  gpu.count * sizeof(Vec3), cudaMemcpyHostToDevice, gpu.stream));
        checkCuda(cudaStreamSynchronize(gpu.stream));
    });

    const bool needState = printResults || validate;
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps && numBodies > 0; ++step) {
        const bool anotherStep = step + 1 < numSteps;

        onGpus(workers, gpuThreads, [&](GpuWork& gpu) {
            if (stagePositions && step > 0) {
                checkCuda(cudaMemcpyAsync(gpu.positions, positions, numBodies * sizeof(Vec3),
                                          cudaMemcpyHostToDevice, gpu.stream));
            }
            switch (gpu.forceThreads) {
                case 32: launchAdvance<32>(gpu, numBodies); break;
                case 64: launchAdvance<64>(gpu, numBodies); break;
                default: launchAdvance<128>(gpu, numBodies); break;
            }
            checkCuda(cudaGetLastError());

            if (singleGpu) {
                std::swap(gpu.positions, gpu.next);
            } else if (stagePositions && anotherStep) {
                checkCuda(cudaMemcpyAsync(nextPositions + gpu.first, gpu.next,
                                          gpu.count * sizeof(Vec3), cudaMemcpyDeviceToHost, gpu.stream));
            }
        });

        if (!singleGpu && anotherStep) {
            onGpus(workers, gpuThreads, [](const GpuWork& gpu) {
                checkCuda(cudaStreamSynchronize(gpu.stream));
            });
            if (deviceExchange) {
                const auto& gpu = workers.front();
                exchangePositions(gpu.next, gpu.positions);
            } else {
                if (ranks > 1) {
                    exchangePositions(MPI_IN_PLACE, nextPositions);
                }
                std::swap(positions, nextPositions);
            }
        }
    }

    onGpus(workers, gpuThreads, [](const GpuWork& gpu) {
        checkCuda(cudaStreamSynchronize(gpu.stream));
    });
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(1000 * maxElapsed));
    }

    if (needState && numSteps > 0) {
        onGpus(workers, gpuThreads, [&](const GpuWork& gpu) {
            checkCuda(cudaMemcpyAsync(positions + gpu.first, singleGpu ? gpu.positions : gpu.next,
                                      gpu.count * sizeof(Vec3), cudaMemcpyDeviceToHost, gpu.stream));
            checkCuda(cudaMemcpyAsync(velocities + (gpu.first - first), gpu.velocities,
                                      gpu.count * sizeof(Vec3), cudaMemcpyDeviceToHost, gpu.stream));
            checkCuda(cudaStreamSynchronize(gpu.stream));
        });
    }

    if (printResults) {
        if (ranks > 1) {
            if (numSteps > 0) {
                if (validate) {
                    exchangePositions(MPI_IN_PLACE, positions);
                } else {
                    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : positions + first, localCount, vectorType,
                                positions, counts.data(), offsets.data(), vectorType, 0, MPI_COMM_WORLD);
                }
            }
            MPI_Gatherv(velocities, localCount, vectorType, rootVelocities.data(),
                        counts.data(), offsets.data(), vectorType, 0, MPI_COMM_WORLD);
        }
        if (rank == 0) {
            const Vec3* allVelocities = ranks == 1 ? velocities : rootVelocities.data();
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
            const int threads = hostThreads(numBodies);
            #pragma omp parallel for num_threads(threads) if(threads > 1)
            for (int i = 0; i < numBodies; ++i) {
                const Vec3 p = positions[i];
                const Vec3 v = allVelocities[i];
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

    int status = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }
        const int threads = hostThreads(localCount);
        uint64_t firstIssue = UINT64_MAX;
        #pragma omp parallel for num_threads(threads) if(threads > 1) reduction(min:firstIssue)
        for (int i = 0; i < localCount; ++i) {
            const int issue = validationIssue(positions[first + i], velocities[i]);
            if (issue != 0) {
                firstIssue = std::min(firstIssue, 4 * static_cast<uint64_t>(first + i) + issue);
            }
        }
        uint64_t globalIssue;
        MPI_Allreduce(&firstIssue, &globalIssue, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
        if (globalIssue == UINT64_MAX) {
            if (!printResults && ranks > 1 && numSteps > 0) {
                exchangePositions(MPI_IN_PLACE, positions);
            }
            double* const energies = allocateHost<double>(localCount);
            onGpus(workers, gpuThreads, [&](const GpuWork& gpu) {
                if (!singleGpu && numSteps > 0) {
                    checkCuda(cudaMemcpyAsync(gpu.positions, positions, numBodies * sizeof(Vec3),
                                              cudaMemcpyHostToDevice, gpu.stream));
                }
                double* const gpuEnergies = allocateDevice<double>(gpu.count);
                bodyEnergies<<<blocksFor(gpu.count, ENERGY_THREADS), ENERGY_THREADS, 0, gpu.stream>>>(
                    gpu.positions, gpu.velocities, gpuEnergies, numBodies, gpu.first, gpu.count);
                checkCuda(cudaGetLastError());
                checkCuda(cudaMemcpyAsync(energies + (gpu.first - first), gpuEnergies,
                                          gpu.count * sizeof(double), cudaMemcpyDeviceToHost, gpu.stream));
                checkCuda(cudaStreamSynchronize(gpu.stream));
                checkCuda(cudaFree(gpuEnergies));
            });
            double localEnergy = 0.0;
            #pragma omp parallel for num_threads(threads) if(threads > 1) reduction(+:localEnergy)
            for (int i = 0; i < localCount; ++i) {
                localEnergy += energies[i];
            }
            checkCuda(cudaFreeHost(energies));
            double energy = 0.0;
            MPI_Reduce(&localEnergy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Final energy: %.6f\n", energy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                const char* messages[] = {"", "found NaN or Inf value in body state",
                                          "body position exceeds reasonable bounds",
                                          "body velocity exceeds reasonable bounds"};
                printf("Validation failed: %s\n", messages[globalIssue % 4]);
                printf("Validation: FAILED\n");
            }
            status = 1;
        }
    }

    onGpus(workers, gpuThreads, [](const GpuWork& gpu) {
        checkCuda(cudaFree(gpu.velocities));
        checkCuda(cudaFree(gpu.next));
        checkCuda(cudaFree(gpu.positions));
        checkCuda(cudaStreamDestroy(gpu.stream));
    });
    checkCuda(cudaFreeHost(velocities));
    checkCuda(cudaFreeHost(positions));
    if (nextPositions != nullptr) {
        checkCuda(cudaFreeHost(nextPositions));
    }
    MPI_Type_free(&vectorType);
    return status;
}

} // namespace

int main(int argc, char** argv) {
    int threadSupport;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &threadSupport);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    worldRank = rank;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (threadSupport < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        }
        MPI_Finalize();
        return 1;
    }
    const int status = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
