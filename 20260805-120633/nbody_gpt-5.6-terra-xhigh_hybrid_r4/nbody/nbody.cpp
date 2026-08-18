#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int CUDA_BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be packed as three doubles");
static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be packed as six doubles");
static_assert(sizeof(double3) == sizeof(Vec3), "CUDA and host position layouts must agree");

[[noreturn]] void cudaAbort(cudaError_t status, const char* operation) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        cudaAbort(status, operation);
    }
}

void checkMpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        char error[MPI_MAX_ERROR_STRING];
        int errorLength = 0;
        MPI_Error_string(status, error, &errorLength);
        std::fprintf(stderr, "MPI error during %s: %.*s\n", operation, errorLength, error);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

// MPI only requires host pointers, so pin the communication buffers.  This
// allows the same executable to run with both CUDA-aware and standard MPI
// installations while retaining high-bandwidth asynchronous GPU transfers.
template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t size) : size_(size) {
        void* allocation = nullptr;
        checkCuda(cudaMallocHost(&allocation, std::max<size_t>(size, 1) * sizeof(T)), "cudaMallocHost");
        data_ = static_cast<T*>(allocation);
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    ~PinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }

    T& operator[](size_t index) noexcept { return data_[index]; }
    const T& operator[](size_t index) const noexcept { return data_[index]; }

private:
    T* data_ = nullptr;
    size_t size_ = 0;
};

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

// Every rank owns a contiguous particle range.  The positions of every range
// are all-gathered on GPU memory before this kernel advances the local range.
// The source-particle loop is intentionally ordered from 0 to n-1, matching
// the original direct N-body calculation.
__global__ void advanceBodies(double3* __restrict__ localPositions,
                              double3* __restrict__ localVelocities,
                              const double3* __restrict__ globalPositions,
                              int localCount,
                              int globalCount) {
    extern __shared__ double3 positionTile[];

    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIndex < localCount;

    double3 position{};
    double3 velocity{};
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    if (active) {
        position = localPositions[localIndex];
        velocity = localVelocities[localIndex];
    }

    for (int tileStart = 0; tileStart < globalCount; tileStart += blockDim.x) {
        const int sourceIndex = tileStart + threadIdx.x;
        if (sourceIndex < globalCount) {
            positionTile[threadIdx.x] = globalPositions[sourceIndex];
        }
        __syncthreads();

        if (active) {
            const int tileCount = min(blockDim.x, globalCount - tileStart);
            for (int j = 0; j < tileCount; ++j) {
                const double dx = positionTile[j].x - position.x;
                const double dy = positionTile[j].y - position.y;
                const double dz = positionTile[j].z - position.z;
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
        velocity.x += DT * forceX;
        velocity.y += DT * forceY;
        velocity.z += DT * forceZ;
        position.x += DT * velocity.x;
        position.y += DT * velocity.y;
        position.z += DT * velocity.z;

        localPositions[localIndex] = position;
        localVelocities[localIndex] = velocity;
    }
}

bool validateLocalBodies(const double3* positions, const double3* velocities, int count) {
    int finite = 1;
    int bounded = 1;

#pragma omp parallel for schedule(static) reduction(&:finite, bounded)
    for (int i = 0; i < count; ++i) {
        const double3 position = positions[static_cast<size_t>(i)];
        const double3 velocity = velocities[static_cast<size_t>(i)];
        if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z) ||
            !std::isfinite(velocity.x) || !std::isfinite(velocity.y) || !std::isfinite(velocity.z)) {
            finite = 0;
        }

        constexpr double maxPosition = 1e6;
        constexpr double maxVelocity = 1e6;
        if (std::abs(position.x) > maxPosition || std::abs(position.y) > maxPosition ||
            std::abs(position.z) > maxPosition || std::abs(velocity.x) > maxVelocity ||
            std::abs(velocity.y) > maxVelocity || std::abs(velocity.z) > maxVelocity) {
            bounded = 0;
        }
    }
    return finite != 0 && bounded != 0;
}

double computeDistributedTotalEnergy(const double3* localPositions,
                                     const double3* localVelocities,
                                     int localCount,
                                     const double3* globalPositions,
                                     int globalCount,
                                     int globalOffset) {
    double kineticEnergy = 0.0;
    double potentialEnergy = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : kineticEnergy, potentialEnergy)
    for (int localIndex = 0; localIndex < localCount; ++localIndex) {
        const double3 velocity = localVelocities[static_cast<size_t>(localIndex)];
        kineticEnergy += 0.5 * (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);

        const int globalIndex = globalOffset + localIndex;
        const double3 position = localPositions[static_cast<size_t>(localIndex)];
        for (int j = globalIndex + 1; j < globalCount; ++j) {
            const double dx = globalPositions[static_cast<size_t>(j)].x - position.x;
            const double dy = globalPositions[static_cast<size_t>(j)].y - position.y;
            const double dz = globalPositions[static_cast<size_t>(j)].z - position.z;
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potentialEnergy -= 1.0 / distance;
        }
    }
    return kineticEnergy + potentialEnergy;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    checkMpi(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel), "MPI_Init_thread");

    int rank = 0;
    int worldSize = 0;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), "MPI_Comm_size");

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide the MPI_THREAD_FUNNELED level required by this program\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    int nodeRank = 0;
    MPI_Comm nodeComm = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm),
             "MPI_Comm_split_type");
    checkMpi(MPI_Comm_rank(nodeComm, &nodeRank), "MPI_Comm_rank for node");

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA devices are available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    checkCuda(cudaSetDevice(nodeRank % deviceCount), "cudaSetDevice");
    checkMpi(MPI_Comm_free(&nodeComm), "MPI_Comm_free");

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            checkMpi(MPI_Finalize(), "MPI_Finalize");
            return EXIT_SUCCESS;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            checkMpi(MPI_Finalize(), "MPI_Finalize");
            return EXIT_FAILURE;
        }
    }

    if (numBodies < 0 || numSteps < 0 || numBodies > std::numeric_limits<int>::max() / 6) {
        if (rank == 0) {
            std::fprintf(stderr, "The number of bodies and steps must be non-negative, and body count must fit MPI counts\n");
        }
        checkMpi(MPI_Finalize(), "MPI_Finalize");
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", worldSize);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
    }

    const int baseCount = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    std::vector<int> bodyCounts(static_cast<size_t>(worldSize));
    std::vector<int> bodyOffsets(static_cast<size_t>(worldSize));
    std::vector<int> positionCounts(static_cast<size_t>(worldSize));
    std::vector<int> positionOffsets(static_cast<size_t>(worldSize));
    std::vector<int> bodyDataCounts(static_cast<size_t>(worldSize));
    std::vector<int> bodyDataOffsets(static_cast<size_t>(worldSize));
    int offset = 0;
    for (int process = 0; process < worldSize; ++process) {
        const int count = baseCount + (process < remainder ? 1 : 0);
        bodyCounts[static_cast<size_t>(process)] = count;
        bodyOffsets[static_cast<size_t>(process)] = offset;
        positionCounts[static_cast<size_t>(process)] = 3 * count;
        positionOffsets[static_cast<size_t>(process)] = 3 * offset;
        bodyDataCounts[static_cast<size_t>(process)] = 6 * count;
        bodyDataOffsets[static_cast<size_t>(process)] = 6 * offset;
        offset += count;
    }

    const int localCount = bodyCounts[static_cast<size_t>(rank)];
    const int globalOffset = bodyOffsets[static_cast<size_t>(rank)];

    std::vector<Body> initialBodies;
    if (rank == 0) {
        initialBodies.resize(static_cast<size_t>(numBodies));
        randomizeBodies(initialBodies);
    }
    std::vector<Body> localBodies(static_cast<size_t>(localCount));
    checkMpi(MPI_Scatterv(rank == 0 ? initialBodies.data() : nullptr,
                          bodyDataCounts.data(), bodyDataOffsets.data(), MPI_DOUBLE,
                          localBodies.data(), 6 * localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD),
             "MPI_Scatterv initial body state");

    PinnedBuffer<double3> localPositions(static_cast<size_t>(localCount));
    PinnedBuffer<double3> localVelocities(static_cast<size_t>(localCount));
    PinnedBuffer<double3> globalPositions(static_cast<size_t>(numBodies));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) {
        const Body& body = localBodies[static_cast<size_t>(i)];
        localPositions[static_cast<size_t>(i)] = make_double3(body.pos.x, body.pos.y, body.pos.z);
        localVelocities[static_cast<size_t>(i)] = make_double3(body.vel.x, body.vel.y, body.vel.z);
    }
    localBodies.clear();
    localBodies.shrink_to_fit();

    const size_t localAllocationCount = std::max<size_t>(1, localPositions.size());
    const size_t globalAllocationCount = std::max(1, numBodies);
    double3* deviceLocalPositions = nullptr;
    double3* deviceLocalVelocities = nullptr;
    double3* deviceGlobalPositions = nullptr;
    checkCuda(cudaMalloc(&deviceLocalPositions, localAllocationCount * sizeof(*deviceLocalPositions)),
              "cudaMalloc local positions");
    checkCuda(cudaMalloc(&deviceLocalVelocities, localAllocationCount * sizeof(*deviceLocalVelocities)),
              "cudaMalloc local velocities");
    checkCuda(cudaMalloc(&deviceGlobalPositions, globalAllocationCount * sizeof(*deviceGlobalPositions)),
              "cudaMalloc global positions");
    if (localCount > 0) {
        checkCuda(cudaMemcpy(deviceLocalPositions, localPositions.data(), localPositions.size() * sizeof(double3),
                             cudaMemcpyHostToDevice),
                  "copy initial positions to device");
        checkCuda(cudaMemcpy(deviceLocalVelocities, localVelocities.data(), localVelocities.size() * sizeof(double3),
                             cudaMemcpyHostToDevice),
                  "copy initial velocities to device");
    }

    cudaStream_t simulationStream = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&simulationStream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier before timing");
    const double startTime = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        // The initial host buffer is already authoritative.  Thereafter the
        // preceding GPU update is staged back before the MPI collective.
        if (step != 0 && localCount > 0) {
            checkCuda(cudaMemcpyAsync(localPositions.data(), deviceLocalPositions,
                                      localPositions.size() * sizeof(double3), cudaMemcpyDeviceToHost,
                                      simulationStream),
                      "stage local positions from device");
        }
        checkCuda(cudaStreamSynchronize(simulationStream), "local position staging completion");
        checkMpi(MPI_Allgatherv(localPositions.data(), 3 * localCount, MPI_DOUBLE,
                                globalPositions.data(), positionCounts.data(), positionOffsets.data(), MPI_DOUBLE,
                                MPI_COMM_WORLD),
                 "MPI_Allgatherv positions");

        if (localCount > 0) {
            checkCuda(cudaMemcpyAsync(deviceGlobalPositions, globalPositions.data(),
                                      static_cast<size_t>(numBodies) * sizeof(double3), cudaMemcpyHostToDevice,
                                      simulationStream),
                      "stage global positions to device");
            const int gridSize = (localCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            advanceBodies<<<gridSize, CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE * sizeof(double3), simulationStream>>>(
                deviceLocalPositions, deviceLocalVelocities, deviceGlobalPositions, localCount, numBodies);
            checkCuda(cudaGetLastError(), "advanceBodies kernel launch");
        }
    }
    checkCuda(cudaStreamSynchronize(simulationStream), "final simulation kernel completion");
    const double localElapsed = MPI_Wtime() - startTime;
    double elapsed = 0.0;
    checkMpi(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
             "MPI_Reduce elapsed time");

    if (localCount > 0) {
        checkCuda(cudaMemcpyAsync(localPositions.data(), deviceLocalPositions, localPositions.size() * sizeof(double3),
                                  cudaMemcpyDeviceToHost, simulationStream),
                  "copy final positions from device");
        checkCuda(cudaMemcpyAsync(localVelocities.data(), deviceLocalVelocities, localVelocities.size() * sizeof(double3),
                                  cudaMemcpyDeviceToHost, simulationStream),
                  "copy final velocities from device");
        checkCuda(cudaStreamSynchronize(simulationStream), "final state transfer completion");
    }

    if (rank == 0) {
        std::printf("Simulation time: %.3f ms\n", elapsed * 1000.0);
    }

    int exitCode = EXIT_SUCCESS;
    if (printResults) {
        std::vector<double3> finalPositions;
        std::vector<double3> finalVelocities;
        if (rank == 0) {
            finalPositions.resize(static_cast<size_t>(numBodies));
            finalVelocities.resize(static_cast<size_t>(numBodies));
        }
        checkMpi(MPI_Gatherv(localPositions.data(), 3 * localCount, MPI_DOUBLE,
                             rank == 0 ? finalPositions.data() : nullptr,
                             positionCounts.data(), positionOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD),
                 "MPI_Gatherv final positions");
        checkMpi(MPI_Gatherv(localVelocities.data(), 3 * localCount, MPI_DOUBLE,
                             rank == 0 ? finalVelocities.data() : nullptr,
                             positionCounts.data(), positionOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD),
                 "MPI_Gatherv final velocities");

        if (rank == 0) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                const size_t index = static_cast<size_t>(i);
                bodyData[6 * index] = finalPositions[index].x;
                bodyData[6 * index + 1] = finalPositions[index].y;
                bodyData[6 * index + 2] = finalPositions[index].z;
                bodyData[6 * index + 3] = finalVelocities[index].x;
                bodyData[6 * index + 4] = finalVelocities[index].y;
                bodyData[6 * index + 5] = finalVelocities[index].z;
            }
            print_results(bodyData, "Bodies");
        }
    }

    if (validate) {
        int localValid = validateLocalBodies(localPositions.data(), localVelocities.data(), localCount) ? 1 : 0;
        int globallyValid = 0;
        checkMpi(MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD),
                 "MPI_Allreduce validation state");

        if (globallyValid != 0) {
            PinnedBuffer<double3> finalGlobalPositions(static_cast<size_t>(numBodies));
            checkMpi(MPI_Allgatherv(localPositions.data(), 3 * localCount, MPI_DOUBLE,
                                    finalGlobalPositions.data(), positionCounts.data(), positionOffsets.data(), MPI_DOUBLE,
                                    MPI_COMM_WORLD),
                     "MPI_Allgatherv final positions");
            const double localEnergy = computeDistributedTotalEnergy(
                localPositions.data(), localVelocities.data(), localCount, finalGlobalPositions.data(), numBodies,
                globalOffset);
            double finalEnergy = 0.0;
            checkMpi(MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD),
                     "MPI_Reduce total energy");
            if (rank == 0) {
                std::printf("Validating simulation results...\n");
                std::printf("Final energy: %.6f\n", finalEnergy);
                std::printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                std::printf("Validating simulation results...\n");
                std::printf("Validation failed: found non-finite or out-of-bounds body state\n");
                std::printf("Validation: FAILED\n");
            }
            exitCode = EXIT_FAILURE;
        }
    }

    checkCuda(cudaStreamDestroy(simulationStream), "cudaStreamDestroy");
    checkCuda(cudaFree(deviceGlobalPositions), "cudaFree global positions");
    checkCuda(cudaFree(deviceLocalVelocities), "cudaFree local velocities");
    checkCuda(cudaFree(deviceLocalPositions), "cudaFree local positions");
    checkMpi(MPI_Finalize(), "MPI_Finalize");
    return exitCode;
}
