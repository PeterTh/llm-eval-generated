#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

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

static_assert(std::is_standard_layout_v<Body>);
static_assert(sizeof(Body) == 6 * sizeof(double));

void checkCuda(cudaError_t error, const char* expression, int rank) {
    if (error == cudaSuccess) {
        return;
    }
    std::fprintf(stderr, "MPI rank %d: CUDA failure in %s: %s\n", rank, expression,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(call) checkCuda((call), #call, rank)

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Keep the original deterministic random-number stream. This is intentionally
    // performed on rank zero before the bodies are distributed.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

// Each block reuses a shared-memory tile of source positions for 256 target
// bodies. Ranks own disjoint target ranges, so no atomics or inter-rank force
// reductions are needed. Integration is fused into the force kernel to avoid a
// second launch and another pass over device memory.
__global__ void computeForcesAndIntegrate(const double4* __restrict__ globalPositions,
                                          double4* __restrict__ localPositions,
                                          double4* __restrict__ localVelocities,
                                          int globalBodyCount, int globalOffset,
                                          int localBodyCount) {
    extern __shared__ double4 positionTile[];

    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIndex < localBodyCount;
    double4 position = make_double4(0.0, 0.0, 0.0, 0.0);
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    if (active) {
        position = globalPositions[globalOffset + localIndex];
    }

    for (int tileBase = 0; tileBase < globalBodyCount; tileBase += blockDim.x) {
        const int sourceIndex = tileBase + threadIdx.x;
        if (sourceIndex < globalBodyCount) {
            positionTile[threadIdx.x] = globalPositions[sourceIndex];
        }
        __syncthreads();

        if (active) {
            const int tileCount = min(blockDim.x, globalBodyCount - tileBase);
#pragma unroll 8
            for (int j = 0; j < tileCount; ++j) {
                const double4 source = positionTile[j];
                const double dx = source.x - position.x;
                const double dy = source.y - position.y;
                const double dz = source.z - position.z;
                const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
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
        double4 velocity = localVelocities[localIndex];
        velocity.x += DT * forceX;
        velocity.y += DT * forceY;
        velocity.z += DT * forceZ;
        position.x += velocity.x * DT;
        position.y += velocity.y * DT;
        position.z += velocity.z * DT;
        localVelocities[localIndex] = velocity;
        localPositions[localIndex] = position;
    }
}

void makeDistribution(int bodyCount, int rankCount, std::vector<int>& counts,
                      std::vector<int>& displacements) {
    counts.resize(rankCount);
    displacements.resize(rankCount);
    const int quotient = bodyCount / rankCount;
    const int remainder = bodyCount % rankCount;
    int offset = 0;
    for (int r = 0; r < rankCount; ++r) {
        counts[r] = quotient + (r < remainder ? 1 : 0);
        displacements[r] = offset;
        offset += counts[r];
    }
}

bool queryCudaAwareMpi(int rank) {
    bool cudaAware = false;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    cudaAware = MPIX_Query_cuda_support() != 0;
#endif

    // This opt-in also covers CUDA-aware MPI implementations that do not expose
    // Open MPI's query extension (for example, some MVAPICH builds).
    if (const char* setting = std::getenv("NBODY_CUDA_AWARE_MPI")) {
        if (std::strcmp(setting, "1") == 0) {
            cudaAware = true;
        } else if (std::strcmp(setting, "0") == 0) {
            cudaAware = false;
        } else if (rank == 0) {
            std::fprintf(stderr,
                         "Ignoring invalid NBODY_CUDA_AWARE_MPI value '%s' (expected 0 or 1)\n",
                         setting);
        }
    }
    return cudaAware;
}

void exchangePositions(double4* deviceLocalPositions, double4* deviceGlobalPositions,
                       double4* hostLocalPositions, double4* hostGlobalPositions,
                       int localCount, int globalCount, const std::vector<int>& counts4,
                       const std::vector<int>& displacements4, bool cudaAwareMpi,
                       cudaStream_t stream, int rank) {
    if (cudaAwareMpi) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Allgatherv(deviceLocalPositions, 4 * localCount, MPI_DOUBLE,
                       deviceGlobalPositions, counts4.data(), displacements4.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        return;
    }

    if (localCount != 0) {
        CUDA_CHECK(cudaMemcpyAsync(hostLocalPositions, deviceLocalPositions,
                                   static_cast<size_t>(localCount) * sizeof(double4),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Allgatherv(hostLocalPositions, 4 * localCount, MPI_DOUBLE, hostGlobalPositions,
                   counts4.data(), displacements4.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpyAsync(deviceGlobalPositions, hostGlobalPositions,
                               static_cast<size_t>(globalCount) * sizeof(double4),
                               cudaMemcpyHostToDevice, stream));
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double kineticEnergy = 0.0;
    const long long bodyCount = static_cast<long long>(bodies.size());

#pragma omp parallel for reduction(+ : kineticEnergy) schedule(static)
    for (long long i = 0; i < bodyCount; ++i) {
        const Body& body = bodies[static_cast<size_t>(i)];
        kineticEnergy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y +
                               body.vel.z * body.vel.z);
    }

    double potentialEnergy = 0.0;
#pragma omp parallel for reduction(+ : potentialEnergy) schedule(static)
    for (long long i = 0; i < bodyCount; ++i) {
        for (long long j = i + 1; j < bodyCount; ++j) {
            const Body& first = bodies[static_cast<size_t>(i)];
            const Body& second = bodies[static_cast<size_t>(j)];
            const double dx = second.pos.x - first.pos.x;
            const double dy = second.pos.y - first.pos.y;
            const double dz = second.pos.z - first.pos.z;
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potentialEnergy -= 1.0 / distance;
        }
    }
    return kineticEnergy + potentialEnergy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int valid = 1;
    const long long bodyCount = static_cast<long long>(bodies.size());
#pragma omp parallel for reduction(& : valid) schedule(static)
    for (long long i = 0; i < bodyCount; ++i) {
        const Body& body = bodies[static_cast<size_t>(i)];
        constexpr double maxPosition = 1e6;
        constexpr double maxVelocity = 1e6;
        const bool finite = std::isfinite(body.pos.x) && std::isfinite(body.pos.y) &&
                            std::isfinite(body.pos.z) && std::isfinite(body.vel.x) &&
                            std::isfinite(body.vel.y) && std::isfinite(body.vel.z);
        const bool bounded = std::abs(body.pos.x) <= maxPosition &&
                             std::abs(body.pos.y) <= maxPosition &&
                             std::abs(body.pos.z) <= maxPosition &&
                             std::abs(body.vel.x) <= maxVelocity &&
                             std::abs(body.vel.y) <= maxVelocity &&
                             std::abs(body.vel.z) <= maxVelocity;
        valid &= finite && bounded;
    }
    if (!valid) {
        std::printf("Validation failed: found a non-finite or out-of-bounds body state\n");
    }
    return valid != 0;
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
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int rankCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rankCount);
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
        }
    }

    if (numBodies <= 0 || numBodies > std::numeric_limits<int>::max() / 4 || numSteps < 0) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "The body count must be in [1, INT_MAX/4] and the step count must be nonnegative\n");
        }
        argumentsValid = false;
    }
    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localCommunicator);
    int localRank = 0;
    int localRankCount = 1;
    MPI_Comm_rank(localCommunicator, &localRank);
    MPI_Comm_size(localCommunicator, &localRankCount);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA-capable device is available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));
    const bool cudaAwareMpi = queryCudaAwareMpi(rank);

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s)/rank\n",
                    rankCount, omp_get_max_threads());
        std::printf("Rank 0 CUDA device: %s\n", deviceProperties.name);
        std::printf("MPI GPU buffers: %s\n",
                    cudaAwareMpi ? "CUDA-aware (GPUDirect capable)" : "pinned-host staging");
    }
    if (localRank == 0 && localRankCount > deviceCount) {
        std::fprintf(stderr,
                     "Warning: %d local MPI ranks share %d CUDA device(s); one rank per GPU is recommended\n",
                     localRankCount, deviceCount);
    }

    std::vector<int> counts;
    std::vector<int> displacements;
    makeDistribution(numBodies, rankCount, counts, displacements);
    const int localCount = counts[rank];
    const int globalOffset = displacements[rank];

    std::vector<int> counts4(rankCount);
    std::vector<int> displacements4(rankCount);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rankCount; ++r) {
        counts4[r] = 4 * counts[r];
        displacements4[r] = 4 * displacements[r];
    }

    MPI_Datatype bodyType = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    std::vector<Body> allBodies;
    if (rank == 0) {
        allBodies.resize(numBodies);
        randomizeBodies(allBodies);
    }
    std::vector<Body> localBodies(static_cast<size_t>(localCount));
    MPI_Scatterv(rank == 0 ? allBodies.data() : nullptr, counts.data(), displacements.data(),
                 bodyType, localBodies.data(), localCount, bodyType, 0, MPI_COMM_WORLD);

    const size_t allocatedLocalCount = static_cast<size_t>(std::max(localCount, 1));
    double4* hostLocalPositions = nullptr;
    double4* hostLocalVelocities = nullptr;
    double4* hostGlobalPositions = nullptr;
    CUDA_CHECK(cudaMallocHost(&hostLocalPositions, allocatedLocalCount * sizeof(double4)));
    CUDA_CHECK(cudaMallocHost(&hostLocalVelocities, allocatedLocalCount * sizeof(double4)));
    if (!cudaAwareMpi) {
        CUDA_CHECK(cudaMallocHost(&hostGlobalPositions,
                                  static_cast<size_t>(numBodies) * sizeof(double4)));
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) {
        const Body& body = localBodies[static_cast<size_t>(i)];
        hostLocalPositions[i] = make_double4(body.pos.x, body.pos.y, body.pos.z, 0.0);
        hostLocalVelocities[i] = make_double4(body.vel.x, body.vel.y, body.vel.z, 0.0);
    }

    double4* deviceLocalPositions = nullptr;
    double4* deviceLocalVelocities = nullptr;
    double4* deviceGlobalPositions = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceLocalPositions, allocatedLocalCount * sizeof(double4)));
    CUDA_CHECK(cudaMalloc(&deviceLocalVelocities, allocatedLocalCount * sizeof(double4)));
    CUDA_CHECK(cudaMalloc(&deviceGlobalPositions,
                          static_cast<size_t>(numBodies) * sizeof(double4)));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (localCount != 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceLocalPositions, hostLocalPositions,
                                   static_cast<size_t>(localCount) * sizeof(double4),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(deviceLocalVelocities, hostLocalVelocities,
                                   static_cast<size_t>(localCount) * sizeof(double4),
                                   cudaMemcpyHostToDevice, stream));
    }
    exchangePositions(deviceLocalPositions, deviceGlobalPositions, hostLocalPositions,
                      hostGlobalPositions, localCount, numBodies, counts4, displacements4,
                      cudaAwareMpi, stream, rank);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Initialization and the initial position exchange are deliberately outside
    // the timed simulation region, matching the original benchmark's timing.
    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    const int blockCount = (localCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    for (int step = 0; step < numSteps; ++step) {
        if (blockCount != 0) {
            computeForcesAndIntegrate<<<blockCount, CUDA_BLOCK_SIZE,
                                        CUDA_BLOCK_SIZE * sizeof(double4), stream>>>(
                deviceGlobalPositions, deviceLocalPositions, deviceLocalVelocities,
                numBodies, globalOffset, localCount);
            CUDA_CHECK(cudaGetLastError());
        }

        if (step + 1 < numSteps) {
            exchangePositions(deviceLocalPositions, deviceGlobalPositions,
                              hostLocalPositions, hostGlobalPositions, localCount,
                              numBodies, counts4, displacements4, cudaAwareMpi, stream,
                              rank);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const double localElapsed = MPI_Wtime() - startTime;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Simulation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
    }

    if (printResults || validate) {
        if (localCount != 0) {
            CUDA_CHECK(cudaMemcpyAsync(hostLocalPositions, deviceLocalPositions,
                                       static_cast<size_t>(localCount) * sizeof(double4),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(hostLocalVelocities, deviceLocalVelocities,
                                       static_cast<size_t>(localCount) * sizeof(double4),
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

#pragma omp parallel for schedule(static)
        for (int i = 0; i < localCount; ++i) {
            const double4 position = hostLocalPositions[i];
            const double4 velocity = hostLocalVelocities[i];
            localBodies[static_cast<size_t>(i)] = {
                Vec3(position.x, position.y, position.z),
                Vec3(velocity.x, velocity.y, velocity.z)};
        }
        if (rank == 0 && allBodies.size() != static_cast<size_t>(numBodies)) {
            allBodies.resize(numBodies);
        }
        MPI_Gatherv(localBodies.data(), localCount, bodyType,
                    rank == 0 ? allBodies.data() : nullptr, counts.data(),
                    displacements.data(), bodyType, 0, MPI_COMM_WORLD);
    }

    int exitCode = EXIT_SUCCESS;
    if (rank == 0 && printResults) {
        std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const Body& body = allBodies[static_cast<size_t>(i)];
            const size_t outputOffset = static_cast<size_t>(i) * 6;
            bodyData[outputOffset + 0] = body.pos.x;
            bodyData[outputOffset + 1] = body.pos.y;
            bodyData[outputOffset + 2] = body.pos.z;
            bodyData[outputOffset + 3] = body.vel.x;
            bodyData[outputOffset + 4] = body.vel.y;
            bodyData[outputOffset + 5] = body.vel.z;
        }
        print_results(bodyData, "Bodies");
    }

    if (rank == 0 && validate) {
        std::printf("Validating simulation results...\n");
        if (validateSimulation(allBodies)) {
            const double finalEnergy = computeTotalEnergy(allBodies);
            std::printf("Final energy: %.6f\n", finalEnergy);
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            exitCode = EXIT_FAILURE;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceGlobalPositions));
    CUDA_CHECK(cudaFree(deviceLocalVelocities));
    CUDA_CHECK(cudaFree(deviceLocalPositions));
    if (hostGlobalPositions != nullptr) {
        CUDA_CHECK(cudaFreeHost(hostGlobalPositions));
    }
    CUDA_CHECK(cudaFreeHost(hostLocalVelocities));
    CUDA_CHECK(cudaFreeHost(hostLocalPositions));
    MPI_Type_free(&bodyType);
    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return exitCode;
}
