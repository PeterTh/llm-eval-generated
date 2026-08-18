#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int CUDA_BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(std::is_standard_layout_v<Vec3> && sizeof(Vec3) == 3 * sizeof(double));

namespace {

int worldRank = 0;

[[noreturn]] void abortWithCudaError(const cudaError_t error, const char* expression,
                                     const char* file, const int line) {
    std::fprintf(stderr, "Rank %d: CUDA call %s failed at %s:%d: %s\n", worldRank,
                 expression, file, line, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

[[noreturn]] void abortWithMpiError(const int error, const char* expression,
                                    const char* file, const int line) {
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "Rank %d: MPI call %s failed at %s:%d: %.*s\n", worldRank,
                 expression, file, line, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        const cudaError_t cuda_check_error = (call);                                        \
        if (cuda_check_error != cudaSuccess) {                                              \
            abortWithCudaError(cuda_check_error, #call, __FILE__, __LINE__);                \
        }                                                                                   \
    } while (false)

#define MPI_CHECK(call)                                                                     \
    do {                                                                                    \
        const int mpi_check_error = (call);                                                 \
        if (mpi_check_error != MPI_SUCCESS) {                                               \
            abortWithMpiError(mpi_check_error, #call, __FILE__, __LINE__);                  \
        }                                                                                   \
    } while (false)

__global__ void advanceBodiesKernel(const Vec3* __restrict__ globalPositions,
                                    Vec3* __restrict__ localVelocities,
                                    Vec3* __restrict__ nextLocalPositions,
                                    const int globalOffset, const int localCount,
                                    const int numBodies) {
    __shared__ Vec3 positionTile[CUDA_BLOCK_SIZE];

    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIndex < localCount;
    Vec3 position;
    if (active) {
        position = globalPositions[globalOffset + localIndex];
    }

    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileBase = 0; tileBase < numBodies; tileBase += CUDA_BLOCK_SIZE) {
        const int sourceIndex = tileBase + threadIdx.x;
        if (sourceIndex < numBodies) {
            positionTile[threadIdx.x] = globalPositions[sourceIndex];
        }
        __syncthreads();

        if (active) {
            const int tileSize = min(CUDA_BLOCK_SIZE, numBodies - tileBase);
#pragma unroll 4
            for (int j = 0; j < tileSize; ++j) {
                const double dx = positionTile[j].x - position.x;
                const double dy = positionTile[j].y - position.y;
                const double dz = positionTile[j].z - position.z;
                const double distanceSquared =
                    dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inverseDistance = 1.0 / sqrt(distanceSquared);
                const double inverseDistanceCubed =
                    inverseDistance * inverseDistance * inverseDistance;
                forceX += dx * inverseDistanceCubed;
                forceY += dy * inverseDistanceCubed;
                forceZ += dz * inverseDistanceCubed;
            }
        }
        __syncthreads();
    }

    if (active) {
        Vec3 velocity = localVelocities[localIndex];
        velocity.x += DT * forceX;
        velocity.y += DT * forceY;
        velocity.z += DT * forceZ;
        position.x += velocity.x * DT;
        position.y += velocity.y * DT;
        position.z += velocity.z * DT;
        localVelocities[localIndex] = velocity;
        nextLocalPositions[localIndex] = position;
    }
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // This loop intentionally remains ordered so that the initial state is bit-for-bit
    // identical to the original rand_r sequence.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

std::vector<int> makeCounts(const int total, const int ranks) {
    std::vector<int> counts(ranks);
    const int base = total / ranks;
    const int remainder = total % ranks;
    for (int rank = 0; rank < ranks; ++rank) {
        counts[rank] = base + (rank < remainder ? 1 : 0);
    }
    return counts;
}

std::vector<int> makeDisplacements(const std::vector<int>& counts) {
    std::vector<int> displacements(counts.size(), 0);
    for (std::size_t rank = 1; rank < counts.size(); ++rank) {
        displacements[rank] = displacements[rank - 1] + counts[rank - 1];
    }
    return displacements;
}

std::vector<int> scaleByThree(const std::vector<int>& values) {
    std::vector<int> scaled(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        scaled[i] = 3 * values[i];
    }
    return scaled;
}

bool validateSimulation(const Vec3* positions, const std::vector<Vec3>& velocities,
                        const int numBodies) {
    int hasNonFinite = 0;
    int positionOutOfBounds = 0;
    int velocityOutOfBounds = 0;

#pragma omp parallel for schedule(static) reduction(| : hasNonFinite, positionOutOfBounds, velocityOutOfBounds)
    for (int i = 0; i < numBodies; ++i) {
        const Vec3 position = positions[i];
        const Vec3 velocity = velocities[i];
        hasNonFinite |=
            !(std::isfinite(position.x) && std::isfinite(position.y) &&
              std::isfinite(position.z) && std::isfinite(velocity.x) &&
              std::isfinite(velocity.y) && std::isfinite(velocity.z));
        positionOutOfBounds |= std::abs(position.x) > 1e6 || std::abs(position.y) > 1e6 ||
                               std::abs(position.z) > 1e6;
        velocityOutOfBounds |= std::abs(velocity.x) > 1e6 || std::abs(velocity.y) > 1e6 ||
                               std::abs(velocity.z) > 1e6;
    }

    if (hasNonFinite) {
        std::printf("Validation failed: found NaN or Inf value in body state\n");
        return false;
    }
    if (positionOutOfBounds) {
        std::printf("Validation failed: body position exceeds reasonable bounds\n");
        return false;
    }
    if (velocityOutOfBounds) {
        std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
        return false;
    }
    return true;
}

double computeTotalEnergy(const Vec3* positions, const std::vector<Vec3>& velocities,
                          const int numBodies) {
    double kineticEnergy = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : kineticEnergy)
    for (int i = 0; i < numBodies; ++i) {
        const Vec3 velocity = velocities[i];
        kineticEnergy += 0.5 * (velocity.x * velocity.x + velocity.y * velocity.y +
                                velocity.z * velocity.z);
    }

    double potentialEnergy = 0.0;
#pragma omp parallel for schedule(dynamic, 4) reduction(+ : potentialEnergy)
    for (int i = 0; i < numBodies; ++i) {
        double bodyPotential = 0.0;
        for (int j = i + 1; j < numBodies; ++j) {
            const double dx = positions[j].x - positions[i].x;
            const double dy = positions[j].y - positions[i].y;
            const double dz = positions[j].z - positions[i].z;
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            bodyPotential -= 1.0 / distance;
        }
        potentialEnergy += bodyPotential;
    }
    return kineticEnergy + potentialEnergy;
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

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel));

    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool invalidArguments = false;

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
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            invalidArguments = true;
        }
    }

    if (numBodies <= 0 || numBodies > INT_MAX / 3 || numSteps < 0) {
        if (worldRank == 0) {
            std::fprintf(stderr, "The body count must be positive and the step count non-negative.\n");
        }
        invalidArguments = true;
    }
    if (showHelp || invalidArguments) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return invalidArguments ? 1 : 0;
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                  MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    int localSize = 1;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
    MPI_CHECK(MPI_Comm_size(localCommunicator, &localSize));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::fprintf(stderr, "Rank %d: no CUDA device is available\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    if (localRank == 0 && localSize > deviceCount) {
        std::fprintf(stderr,
                     "Warning: %d MPI ranks share %d CUDA devices on a node; one rank per GPU "
                     "is recommended.\n",
                     localSize, deviceCount);
    }

    // Use GPU-resident collectives when the MPI implementation advertises
    // CUDA-awareness.  The pinned staging path below retains portability to
    // MPI installations without GPUDirect support.
    bool cudaAwareMpi = false;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    cudaAwareMpi = MPIX_Query_cuda_support() != 0;
#endif
    int cudaAwareValue = cudaAwareMpi ? 1 : 0;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &cudaAwareValue, 1, MPI_INT, MPI_MIN,
                            MPI_COMM_WORLD));
    cudaAwareMpi = cudaAwareValue != 0;

    if (worldRank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const std::vector<int> bodyCounts = makeCounts(numBodies, worldSize);
    const std::vector<int> bodyDisplacements = makeDisplacements(bodyCounts);
    const std::vector<int> vectorCounts = scaleByThree(bodyCounts);
    const std::vector<int> vectorDisplacements = scaleByThree(bodyDisplacements);
    const int localCount = bodyCounts[worldRank];
    const int globalOffset = bodyDisplacements[worldRank];

    Vec3* hostGlobalPositions = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostGlobalPositions),
                              static_cast<std::size_t>(numBodies) * sizeof(Vec3)));

    std::vector<Vec3> allInitialVelocities;
    if (worldRank == 0) {
        std::vector<Body> initialBodies(numBodies);
        randomizeBodies(initialBodies);
        allInitialVelocities.resize(numBodies);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            hostGlobalPositions[i] = initialBodies[i].pos;
            allInitialVelocities[i] = initialBodies[i].vel;
        }
    }

    MPI_CHECK(MPI_Bcast(hostGlobalPositions, 3 * numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    std::vector<Vec3> localVelocities(std::max(localCount, 1));
    MPI_CHECK(MPI_Scatterv(worldRank == 0 ? allInitialVelocities.data() : nullptr,
                           vectorCounts.data(), vectorDisplacements.data(), MPI_DOUBLE,
                           localVelocities.data(), 3 * localCount, MPI_DOUBLE, 0,
                           MPI_COMM_WORLD));
    allInitialVelocities.clear();
    allInitialVelocities.shrink_to_fit();

    Vec3* deviceGlobalPositions = nullptr;
    Vec3* deviceLocalPositions = nullptr;
    Vec3* deviceLocalVelocities = nullptr;
    const std::size_t globalBytes = static_cast<std::size_t>(numBodies) * sizeof(Vec3);
    const std::size_t localBytes =
        static_cast<std::size_t>(std::max(localCount, 1)) * sizeof(Vec3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGlobalPositions), globalBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceLocalPositions),
                          cudaAwareMpi ? globalBytes : localBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceLocalVelocities), localBytes));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(deviceGlobalPositions, hostGlobalPositions, globalBytes,
                               cudaMemcpyHostToDevice, stream));
    if (localCount > 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceLocalVelocities, localVelocities.data(),
                                   static_cast<std::size_t>(localCount) * sizeof(Vec3),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    const double startTime = MPI_Wtime();
    const int gridSize = (localCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            Vec3* nextLocalPositions =
                deviceLocalPositions + (cudaAwareMpi ? globalOffset : 0);
            advanceBodiesKernel<<<gridSize, CUDA_BLOCK_SIZE, 0, stream>>>(
                deviceGlobalPositions, deviceLocalVelocities, nextLocalPositions,
                globalOffset, localCount, numBodies);
            CUDA_CHECK(cudaGetLastError());
            if (!cudaAwareMpi) {
                CUDA_CHECK(cudaMemcpyAsync(hostGlobalPositions + globalOffset,
                                           deviceLocalPositions,
                                           static_cast<std::size_t>(localCount) * sizeof(Vec3),
                                           cudaMemcpyDeviceToHost, stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (cudaAwareMpi) {
            MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, deviceLocalPositions,
                                     vectorCounts.data(), vectorDisplacements.data(), MPI_DOUBLE,
                                     MPI_COMM_WORLD));
            std::swap(deviceGlobalPositions, deviceLocalPositions);
        } else {
            MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, hostGlobalPositions,
                                     vectorCounts.data(), vectorDisplacements.data(), MPI_DOUBLE,
                                     MPI_COMM_WORLD));

            if (step + 1 < numSteps) {
                CUDA_CHECK(cudaMemcpyAsync(deviceGlobalPositions, hostGlobalPositions,
                                           globalBytes, cudaMemcpyHostToDevice, stream));
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double localElapsed = MPI_Wtime() - startTime;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (worldRank == 0) {
        const auto milliseconds = static_cast<long>(elapsed * 1000.0);
        std::printf("Simulation time: %ld ms\n", milliseconds);
    }

    std::vector<Vec3> allVelocities;
    if (printResults || validate) {
        if (localCount > 0) {
            CUDA_CHECK(cudaMemcpyAsync(localVelocities.data(), deviceLocalVelocities,
                                       static_cast<std::size_t>(localCount) * sizeof(Vec3),
                                       cudaMemcpyDeviceToHost, stream));
        }
        if (cudaAwareMpi && worldRank == 0) {
            CUDA_CHECK(cudaMemcpyAsync(hostGlobalPositions, deviceGlobalPositions, globalBytes,
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (worldRank == 0) {
            allVelocities.resize(numBodies);
        }
        MPI_CHECK(MPI_Gatherv(localVelocities.data(), 3 * localCount, MPI_DOUBLE,
                              worldRank == 0 ? allVelocities.data() : nullptr,
                              vectorCounts.data(), vectorDisplacements.data(), MPI_DOUBLE, 0,
                              MPI_COMM_WORLD));
    }

    int exitCode = 0;
    if (worldRank == 0 && printResults) {
        std::vector<double> bodyData(static_cast<std::size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const std::size_t outputIndex = static_cast<std::size_t>(i) * 6;
            bodyData[outputIndex] = hostGlobalPositions[i].x;
            bodyData[outputIndex + 1] = hostGlobalPositions[i].y;
            bodyData[outputIndex + 2] = hostGlobalPositions[i].z;
            bodyData[outputIndex + 3] = allVelocities[i].x;
            bodyData[outputIndex + 4] = allVelocities[i].y;
            bodyData[outputIndex + 5] = allVelocities[i].z;
        }
        print_results(bodyData, "Bodies");
    }

    if (worldRank == 0 && validate) {
        std::printf("Validating simulation results...\n");
        if (validateSimulation(hostGlobalPositions, allVelocities, numBodies)) {
            const double finalEnergy =
                computeTotalEnergy(hostGlobalPositions, allVelocities, numBodies);
            std::printf("Final energy: %.6f\n", finalEnergy);
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceLocalVelocities));
    CUDA_CHECK(cudaFree(deviceLocalPositions));
    CUDA_CHECK(cudaFree(deviceGlobalPositions));
    CUDA_CHECK(cudaFreeHost(hostGlobalPositions));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
