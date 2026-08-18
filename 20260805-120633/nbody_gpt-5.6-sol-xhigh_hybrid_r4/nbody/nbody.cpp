#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int CUDA_BLOCK_SIZE = 128;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

void checkCuda(const cudaError_t error, const char* expression, const char* file, const int line) {
    if (error == cudaSuccess) {
        return;
    }

    int initialized = 0;
    int rank = -1;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    std::fprintf(stderr, "Rank %d: CUDA error at %s:%d for %s: %s\n", rank, file, line,
                 expression, cudaGetErrorString(error));
    if (initialized) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(const size_t count) : count_(count) {
        if (count_ != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    size_t size() const noexcept { return count_; }

  private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

template <typename T>
class PinnedBuffer {
  public:
    explicit PinnedBuffer(const size_t count) : count_(count) {
        if (count_ != 0) {
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T), cudaHostAllocPortable));
        }
    }

    ~PinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    T& operator[](const size_t index) noexcept { return data_[index]; }
    const T& operator[](const size_t index) const noexcept { return data_[index]; }

  private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // rand_r carries state from one value to the next, so retaining this serial
    // initialization preserves the exact initial state of the original code.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

template <int BlockSize>
__global__ __launch_bounds__(BlockSize) void advanceBodiesKernel(const double4* __restrict__ allPositions,
                                                                 double4* __restrict__ localVelocities,
                                                                 double4* __restrict__ nextLocalPositions,
                                                                 const int numBodies,
                                                                 const int localOffset,
                                                                 const int localCount) {
    __shared__ double4 positionTile[BlockSize];

    const int localIndex = blockIdx.x * BlockSize + threadIdx.x;
    const bool active = localIndex < localCount;
    const int globalIndex = localOffset + localIndex;
    double4 target = make_double4(0.0, 0.0, 0.0, 0.0);
    if (active) {
        target = allPositions[globalIndex];
    }

    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileStart = 0; tileStart < numBodies; tileStart += BlockSize) {
        const int sourceIndex = tileStart + threadIdx.x;
        positionTile[threadIdx.x] = sourceIndex < numBodies
                                          ? allPositions[sourceIndex]
                                          : make_double4(0.0, 0.0, 0.0, 0.0);
        __syncthreads();

        if (active) {
            const int tileCount = min(BlockSize, numBodies - tileStart);
#pragma unroll 8
            for (int j = 0; j < tileCount; ++j) {
                const double dx = positionTile[j].x - target.x;
                const double dy = positionTile[j].y - target.y;
                const double dz = positionTile[j].z - target.z;
                const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inverseDistance = 1.0 / sqrt(distanceSquared);
                const double inverseDistanceCubed = inverseDistance * inverseDistance * inverseDistance;
                forceX += dx * inverseDistanceCubed;
                forceY += dy * inverseDistanceCubed;
                forceZ += dz * inverseDistanceCubed;
            }
        }
        __syncthreads();
    }

    if (active) {
        double4 velocity = localVelocities[localIndex];
        // The original x86 loop materializes DT * force before adding it to
        // velocity.  Keep those two roundings explicit so the CUDA path has
        // the same floating-point update semantics instead of contracting a
        // new FMA at this boundary.
        velocity.x = __dadd_rn(velocity.x, __dmul_rn(DT, forceX));
        velocity.y = __dadd_rn(velocity.y, __dmul_rn(DT, forceY));
        velocity.z = __dadd_rn(velocity.z, __dmul_rn(DT, forceZ));
        localVelocities[localIndex] = velocity;
        nextLocalPositions[localIndex] = make_double4(__fma_rn(velocity.x, DT, target.x),
                                                      __fma_rn(velocity.y, DT, target.y),
                                                      __fma_rn(velocity.z, DT, target.z), 0.0);
    }
}

struct Partition {
    int offset;
    int count;
};

Partition partitionBodies(const int numBodies, const int rank, const int ranks) {
    const int quotient = numBodies / ranks;
    const int remainder = numBodies % ranks;
    return {rank * quotient + std::min(rank, remainder), quotient + (rank < remainder ? 1 : 0)};
}

void buildMpiLayout(const int numBodies, const int ranks, std::vector<int>& counts,
                    std::vector<int>& displacements) {
    counts.resize(ranks);
    displacements.resize(ranks);
    for (int rank = 0; rank < ranks; ++rank) {
        const Partition partition = partitionBodies(numBodies, rank, ranks);
        counts[rank] = 4 * partition.count;
        displacements[rank] = 4 * partition.offset;
    }
}

bool validateLocalState(const double4* positions, const double4* localVelocities,
                        const Partition partition, int flags[3]) {
    int nonFinite = 0;
    int extremePosition = 0;
    int extremeVelocity = 0;

#pragma omp parallel for schedule(static) reduction(| : nonFinite, extremePosition, extremeVelocity)
    for (int localIndex = 0; localIndex < partition.count; ++localIndex) {
        const double4 position = positions[partition.offset + localIndex];
        const double4 velocity = localVelocities[localIndex];
        nonFinite |= !std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z) ||
                     !std::isfinite(velocity.x) || !std::isfinite(velocity.y) || !std::isfinite(velocity.z);
        extremePosition |= std::abs(position.x) > 1e6 || std::abs(position.y) > 1e6 ||
                           std::abs(position.z) > 1e6;
        extremeVelocity |= std::abs(velocity.x) > 1e6 || std::abs(velocity.y) > 1e6 ||
                           std::abs(velocity.z) > 1e6;
    }

    flags[0] = nonFinite;
    flags[1] = extremePosition;
    flags[2] = extremeVelocity;
    return nonFinite == 0 && extremePosition == 0 && extremeVelocity == 0;
}

double computeLocalEnergy(const double4* positions, const double4* localVelocities,
                          const int numBodies, const Partition partition,
                          const int rank, const int ranks) {
    double kineticEnergy = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : kineticEnergy)
    for (int localIndex = 0; localIndex < partition.count; ++localIndex) {
        const double4 velocity = localVelocities[localIndex];
        kineticEnergy += 0.5 *
                         (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
    }

    // Cyclic assignment keeps the triangular potential-energy work balanced
    // between MPI ranks; dynamic scheduling balances it between OpenMP threads.
    double potentialMagnitude = 0.0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : potentialMagnitude)
    for (int i = rank; i < numBodies; i += ranks) {
        const double4 first = positions[i];
        for (int j = i + 1; j < numBodies; ++j) {
            const double dx = positions[j].x - first.x;
            const double dy = positions[j].y - first.y;
            const double dz = positions[j].z - first.z;
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potentialMagnitude += 1.0 / distance;
        }
    }
    return kineticEnergy - potentialMagnitude;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel) != MPI_SUCCESS) {
        std::fprintf(stderr, "Unable to initialize MPI\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide the required MPI_THREAD_FUNNELED support\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
            break;
        }
    }

    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (numBodies < 0 || numSteps < 0 || numBodies > INT_MAX / 4) {
        if (rank == 0) {
            std::fprintf(stderr, "The body and step counts must be non-negative, and the body count must not exceed %d\n",
                         INT_MAX / 4);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Comm nodeCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeCommunicator);
    int localRank = 0;
    MPI_Comm_rank(nodeCommunicator, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::fprintf(stderr, "Rank %d: no CUDA device is visible\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const Partition localPartition = partitionBodies(numBodies, rank, ranks);
    std::vector<int> mpiCounts;
    std::vector<int> mpiDisplacements;
    buildMpiLayout(numBodies, ranks, mpiCounts, mpiDisplacements);

    int exitCode = EXIT_SUCCESS;
    {
        PinnedBuffer<double4> allPositions(static_cast<size_t>(numBodies));
        PinnedBuffer<double4> localVelocities(static_cast<size_t>(localPartition.count));

        std::vector<Body> initialBodies(static_cast<size_t>(numBodies));
        randomizeBodies(initialBodies);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const Body& body = initialBodies[i];
            allPositions[i] = make_double4(body.pos.x, body.pos.y, body.pos.z, 0.0);
        }
#pragma omp parallel for schedule(static)
        for (int i = 0; i < localPartition.count; ++i) {
            const Body& body = initialBodies[localPartition.offset + i];
            localVelocities[i] = make_double4(body.vel.x, body.vel.y, body.vel.z, 0.0);
        }
        std::vector<Body>().swap(initialBodies);

        DeviceBuffer<double4> devicePositions(static_cast<size_t>(numBodies));
        DeviceBuffer<double4> deviceLocalVelocities(static_cast<size_t>(localPartition.count));
        DeviceBuffer<double4> deviceNextLocalPositions(static_cast<size_t>(localPartition.count));

        if (numBodies != 0) {
            CUDA_CHECK(cudaMemcpyAsync(devicePositions.data(), allPositions.data(),
                                       static_cast<size_t>(numBodies) * sizeof(double4),
                                       cudaMemcpyHostToDevice, stream));
        }
        if (localPartition.count != 0) {
            CUDA_CHECK(cudaMemcpyAsync(deviceLocalVelocities.data(), localVelocities.data(),
                                       static_cast<size_t>(localPartition.count) * sizeof(double4),
                                       cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            std::printf("N-Body Simulation\n");
            std::printf("Number of bodies: %d\n", numBodies);
            std::printf("Number of steps: %d\n", numSteps);
            std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        }

        const double startTime = MPI_Wtime();
        for (int step = 0; step < numSteps; ++step) {
            if (localPartition.count != 0) {
                const int gridSize = (localPartition.count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
                advanceBodiesKernel<CUDA_BLOCK_SIZE><<<gridSize, CUDA_BLOCK_SIZE, 0, stream>>>(
                    devicePositions.data(), deviceLocalVelocities.data(), deviceNextLocalPositions.data(),
                    numBodies, localPartition.offset, localPartition.count);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(allPositions.data() + localPartition.offset,
                                           deviceNextLocalPositions.data(),
                                           static_cast<size_t>(localPartition.count) * sizeof(double4),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            if (numBodies != 0) {
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, allPositions.data(), mpiCounts.data(),
                               mpiDisplacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
                if (step + 1 < numSteps) {
                    CUDA_CHECK(cudaMemcpyAsync(devicePositions.data(), allPositions.data(),
                                               static_cast<size_t>(numBodies) * sizeof(double4),
                                               cudaMemcpyHostToDevice, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            }
        }
        const double localElapsed = MPI_Wtime() - startTime;
        double elapsed = 0.0;
        MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            const long durationMilliseconds = static_cast<long>(elapsed * 1000.0);
            std::printf("Simulation time: %ld ms\n", durationMilliseconds);
        }

        if (printResults || validate) {
            if (localPartition.count != 0) {
                CUDA_CHECK(cudaMemcpyAsync(localVelocities.data(), deviceLocalVelocities.data(),
                                           static_cast<size_t>(localPartition.count) * sizeof(double4),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }

        if (printResults) {
            std::vector<double4> gatheredVelocities;
            if (rank == 0) {
                gatheredVelocities.resize(static_cast<size_t>(numBodies));
            }
            MPI_Gatherv(localVelocities.data(), 4 * localPartition.count, MPI_DOUBLE,
                        rank == 0 ? gatheredVelocities.data() : nullptr, mpiCounts.data(),
                        mpiDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
                for (int i = 0; i < numBodies; ++i) {
                    const size_t output = static_cast<size_t>(i) * 6;
                    bodyData[output] = allPositions[i].x;
                    bodyData[output + 1] = allPositions[i].y;
                    bodyData[output + 2] = allPositions[i].z;
                    bodyData[output + 3] = gatheredVelocities[i].x;
                    bodyData[output + 4] = gatheredVelocities[i].y;
                    bodyData[output + 5] = gatheredVelocities[i].z;
                }
                print_results(bodyData, "Bodies");
            }
        }

        if (validate) {
            int localFlags[3] = {0, 0, 0};
            int globalFlags[3] = {0, 0, 0};
            validateLocalState(allPositions.data(), localVelocities.data(), localPartition, localFlags);
            MPI_Allreduce(localFlags, globalFlags, 3, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            const bool valid = globalFlags[0] == 0 && globalFlags[1] == 0 && globalFlags[2] == 0;

            double localEnergy = 0.0;
            double finalEnergy = 0.0;
            if (valid) {
                localEnergy = computeLocalEnergy(allPositions.data(), localVelocities.data(), numBodies,
                                                 localPartition, rank, ranks);
            }
            MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                std::printf("Validating simulation results...\n");
                if (globalFlags[0] != 0) {
                    std::printf("Validation failed: found NaN or Inf value in body state\n");
                } else if (globalFlags[1] != 0) {
                    std::printf("Validation failed: body position exceeds reasonable bounds\n");
                } else if (globalFlags[2] != 0) {
                    std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
                }

                if (valid) {
                    std::printf("Final energy: %.6f\n", finalEnergy);
                    std::printf("Validation: PASSED\n");
                } else {
                    std::printf("Validation: FAILED\n");
                    exitCode = EXIT_FAILURE;
                }
            }
            MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Comm_free(&nodeCommunicator);
    MPI_Finalize();
    return exitCode;
}
