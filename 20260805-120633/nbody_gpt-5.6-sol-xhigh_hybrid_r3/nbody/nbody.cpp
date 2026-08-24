#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#if defined(OMPI_MAJOR_VERSION) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#define NBODY_HAS_MPI_CUDA_QUERY 1
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int CUDA_BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept
        : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// Four doubles keep every element naturally aligned for coalesced CUDA loads.
// The fourth component is padding and is deliberately excluded from results.
struct alignas(32) DeviceVec3 {
    double x, y, z, padding;
};

static_assert(sizeof(DeviceVec3) == 4 * sizeof(double));

[[noreturn]] void abortMpi(const char* operation, const int error, const char* file, const int line) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char errorString[MPI_MAX_ERROR_STRING] = {};
    int errorLength = 0;
    MPI_Error_string(error, errorString, &errorLength);
    std::fprintf(stderr, "Rank %d: MPI failure at %s:%d in %s: %.*s\n", rank, file, line,
                 operation, errorLength, errorString);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

[[noreturn]] void abortCuda(const char* operation, const cudaError_t error, const char* file,
                            const int line) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: CUDA failure at %s:%d in %s: %s\n", rank, file, line,
                 operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define MPI_CHECK(operation)                                                                    \
    do {                                                                                         \
        const int mpi_check_error = (operation);                                                 \
        if (mpi_check_error != MPI_SUCCESS) {                                                    \
            abortMpi(#operation, mpi_check_error, __FILE__, __LINE__);                           \
        }                                                                                        \
    } while (false)

#define CUDA_CHECK(operation)                                                                   \
    do {                                                                                         \
        const cudaError_t cuda_check_error = (operation);                                       \
        if (cuda_check_error != cudaSuccess) {                                                   \
            abortCuda(#operation, cuda_check_error, __FILE__, __LINE__);                         \
        }                                                                                        \
    } while (false)

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Keep the original single random stream so initialization is deterministic for every rank
    // count and exactly matches the serial benchmark.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

// Each rank computes only its owned bodies. All blocks read an immutable snapshot of the global
// positions, and write to a separate next-position array; this is the same force-then-integrate
// ordering as the original program. Global positions are tiled through shared memory so each rank
// reads each source body from device memory only once per active block.
__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void nbodyStepKernel(const DeviceVec3* __restrict__ positions,
                     DeviceVec3* __restrict__ velocities,
                     DeviceVec3* __restrict__ nextLocalPositions, const int numBodies,
                     const int localCount, const int globalOffset) {
    extern __shared__ DeviceVec3 positionTile[];

    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIndex < localCount;
    const int globalIndex = globalOffset + localIndex;

    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    if (active) {
        const DeviceVec3 position = positions[globalIndex];
        px = position.x;
        py = position.y;
        pz = position.z;
    }

    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileStart = 0; tileStart < numBodies; tileStart += blockDim.x) {
        const int sourceIndex = tileStart + threadIdx.x;
        if (sourceIndex < numBodies) {
            positionTile[threadIdx.x] = positions[sourceIndex];
        }
        __syncthreads();

        if (active) {
            const int tileCount = min(static_cast<int>(blockDim.x), numBodies - tileStart);
#pragma unroll 8
            for (int j = 0; j < tileCount; ++j) {
                const double dx = positionTile[j].x - px;
                const double dy = positionTile[j].y - py;
                const double dz = positionTile[j].z - pz;
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
        DeviceVec3 velocity = velocities[localIndex];
        velocity.x += DT * forceX;
        velocity.y += DT * forceY;
        velocity.z += DT * forceZ;
        velocities[localIndex] = velocity;

        nextLocalPositions[localIndex] = {
            px + velocity.x * DT, py + velocity.y * DT, pz + velocity.z * DT, 0.0};
    }
}

bool validateSimulation(const DeviceVec3* positions, const std::vector<DeviceVec3>& velocities,
                        const int numBodies) {
    int hasNonFinite = 0;
    int hasBadPosition = 0;
    int hasBadVelocity = 0;

#pragma omp parallel for schedule(static) reduction(| : hasNonFinite, hasBadPosition, hasBadVelocity)
    for (int i = 0; i < numBodies; ++i) {
        const DeviceVec3 position = positions[i];
        const DeviceVec3 velocity = velocities[i];
        hasNonFinite |= !std::isfinite(position.x) || !std::isfinite(position.y) ||
                        !std::isfinite(position.z) || !std::isfinite(velocity.x) ||
                        !std::isfinite(velocity.y) || !std::isfinite(velocity.z);
        hasBadPosition |= std::abs(position.x) > 1e6 || std::abs(position.y) > 1e6 ||
                          std::abs(position.z) > 1e6;
        hasBadVelocity |= std::abs(velocity.x) > 1e6 || std::abs(velocity.y) > 1e6 ||
                          std::abs(velocity.z) > 1e6;
    }

    if (hasNonFinite) {
        std::printf("Validation failed: found NaN or Inf value in body state\n");
        return false;
    }
    if (hasBadPosition) {
        std::printf("Validation failed: body position exceeds reasonable bounds\n");
        return false;
    }
    if (hasBadVelocity) {
        std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
        return false;
    }
    return true;
}

double computeTotalEnergy(const DeviceVec3* positions,
                          const std::vector<DeviceVec3>& velocities, const int numBodies) {
    double kineticEnergy = 0.0;
    double potentialEnergy = 0.0;

#pragma omp parallel for schedule(static) reduction(+ : kineticEnergy)
    for (int i = 0; i < numBodies; ++i) {
        const DeviceVec3 velocity = velocities[i];
        kineticEnergy += 0.5 * (velocity.x * velocity.x + velocity.y * velocity.y +
                                velocity.z * velocity.z);
    }

#pragma omp parallel for schedule(dynamic, 8) reduction(+ : potentialEnergy)
    for (int i = 0; i < numBodies; ++i) {
        double localPotential = 0.0;
        const DeviceVec3 firstPosition = positions[i];
        for (int j = i + 1; j < numBodies; ++j) {
            const double dx = positions[j].x - firstPosition.x;
            const double dy = positions[j].y - firstPosition.y;
            const double dz = positions[j].z - firstPosition.z;
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            localPotential -= 1.0 / distance;
        }
        potentialEnergy += localPotential;
    }
    return kineticEnergy + potentialEnergy;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks finite/range values)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int runSimulation(const int rank, const int worldSize, const int numBodies, const int numSteps,
                  const bool validate, const bool printResults) {
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &localCommunicator));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA devices are visible. This benchmark requires CUDA.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));  // Create the context before the timed region.

    std::vector<int> counts(worldSize);
    std::vector<int> displacements(worldSize);
    const int baseCount = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    int runningDisplacement = 0;
    for (int process = 0; process < worldSize; ++process) {
        counts[process] = baseCount + (process < remainder ? 1 : 0);
        displacements[process] = runningDisplacement;
        runningDisplacement += counts[process];
    }
    const int localCount = counts[rank];
    const int globalOffset = displacements[rank];
    const size_t localAllocationCount = static_cast<size_t>(localCount > 0 ? localCount : 1);

    // Open MPI exposes a runtime query for GPU-buffer support. Use GPUDirect collectives whenever
    // that query is available and true, while retaining a pinned staging path for portable MPI
    // installations. Both paths use the same rank decomposition and CUDA kernel.
    bool cudaAwareMpi = false;
#if defined(NBODY_HAS_MPI_CUDA_QUERY)
    cudaAwareMpi = MPIX_Query_cuda_support() != 0;
#endif

    MPI_Datatype mpiDeviceVec3 = MPI_DATATYPE_NULL;
    MPI_CHECK(MPI_Type_contiguous(4, MPI_DOUBLE, &mpiDeviceVec3));
    MPI_CHECK(MPI_Type_commit(&mpiDeviceVec3));

    DeviceVec3* hostGlobalPositions = nullptr;
    DeviceVec3* hostLocalPositions = nullptr;
    DeviceVec3* hostLocalVelocities = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostGlobalPositions),
                              static_cast<size_t>(numBodies) * sizeof(DeviceVec3)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostLocalPositions),
                              localAllocationCount * sizeof(DeviceVec3)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostLocalVelocities),
                              localAllocationCount * sizeof(DeviceVec3)));

    std::vector<DeviceVec3> initialVelocities;
    if (rank == 0) {
        std::vector<Body> bodies(static_cast<size_t>(numBodies));
        randomizeBodies(bodies);
        initialVelocities.resize(static_cast<size_t>(numBodies));
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            hostGlobalPositions[i] = {
                bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0};
            initialVelocities[i] = {
                bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0};
        }
    }

    MPI_CHECK(MPI_Bcast(hostGlobalPositions, numBodies, mpiDeviceVec3, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatterv(rank == 0 ? initialVelocities.data() : nullptr, counts.data(),
                           displacements.data(), mpiDeviceVec3, hostLocalVelocities, localCount,
                           mpiDeviceVec3, 0, MPI_COMM_WORLD));
    initialVelocities.clear();
    initialVelocities.shrink_to_fit();

    DeviceVec3* deviceGlobalPositions = nullptr;
    DeviceVec3* deviceLocalPositions = nullptr;
    DeviceVec3* deviceLocalVelocities = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGlobalPositions),
                          static_cast<size_t>(numBodies) * sizeof(DeviceVec3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceLocalPositions),
                          localAllocationCount * sizeof(DeviceVec3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceLocalVelocities),
                          localAllocationCount * sizeof(DeviceVec3)));
    CUDA_CHECK(cudaMemcpy(deviceGlobalPositions, hostGlobalPositions,
                          static_cast<size_t>(numBodies) * sizeof(DeviceVec3),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceLocalVelocities, hostLocalVelocities,
                          static_cast<size_t>(localCount) * sizeof(DeviceVec3),
                          cudaMemcpyHostToDevice));

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), %d OpenMP thread(s) per rank, "
                    "%d CUDA device(s) on rank 0's node\n",
                    worldSize, omp_get_max_threads(), deviceCount);
        std::printf("MPI GPU transport: %s\n",
                    cudaAwareMpi ? "CUDA-aware (direct device collectives)"
                                 : "portable pinned-host staging");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto start = std::chrono::steady_clock::now();

    const size_t sharedMemoryBytes = CUDA_BLOCK_SIZE * sizeof(DeviceVec3);
    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            const int blocks = (localCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            nbodyStepKernel<<<blocks, CUDA_BLOCK_SIZE, sharedMemoryBytes>>>(
                deviceGlobalPositions, deviceLocalVelocities, deviceLocalPositions, numBodies,
                localCount, globalOffset);
            CUDA_CHECK(cudaGetLastError());
        }

        if (cudaAwareMpi) {
            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_CHECK(MPI_Allgatherv(deviceLocalPositions, localCount, mpiDeviceVec3,
                                     deviceGlobalPositions, counts.data(), displacements.data(),
                                     mpiDeviceVec3, MPI_COMM_WORLD));
        } else {
            if (localCount > 0) {
                CUDA_CHECK(cudaMemcpy(hostLocalPositions, deviceLocalPositions,
                                      static_cast<size_t>(localCount) * sizeof(DeviceVec3),
                                      cudaMemcpyDeviceToHost));
            }
            // Pinned buffers keep the benchmark efficient and functional when the MPI library was
            // not built with GPU-buffer transport.
            MPI_CHECK(MPI_Allgatherv(hostLocalPositions, localCount, mpiDeviceVec3,
                                     hostGlobalPositions, counts.data(), displacements.data(),
                                     mpiDeviceVec3, MPI_COMM_WORLD));

            if (step + 1 < numSteps) {
                CUDA_CHECK(cudaMemcpy(deviceGlobalPositions, hostGlobalPositions,
                                      static_cast<size_t>(numBodies) * sizeof(DeviceVec3),
                                      cudaMemcpyHostToDevice));
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long localDurationMilliseconds = static_cast<long>(duration.count());
    long simulationDurationMilliseconds = 0;
    MPI_CHECK(MPI_Reduce(&localDurationMilliseconds, &simulationDurationMilliseconds, 1, MPI_LONG,
                         MPI_MAX, 0, MPI_COMM_WORLD));

    if (cudaAwareMpi && rank == 0) {
        CUDA_CHECK(cudaMemcpy(hostGlobalPositions, deviceGlobalPositions,
                              static_cast<size_t>(numBodies) * sizeof(DeviceVec3),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaMemcpy(hostLocalVelocities, deviceLocalVelocities,
                          static_cast<size_t>(localCount) * sizeof(DeviceVec3),
                          cudaMemcpyDeviceToHost));
    std::vector<DeviceVec3> finalVelocities;
    if (rank == 0) {
        finalVelocities.resize(static_cast<size_t>(numBodies));
    }
    MPI_CHECK(MPI_Gatherv(hostLocalVelocities, localCount, mpiDeviceVec3,
                          rank == 0 ? finalVelocities.data() : nullptr, counts.data(),
                          displacements.data(), mpiDeviceVec3, 0, MPI_COMM_WORLD));

    int exitCode = EXIT_SUCCESS;
    if (rank == 0) {
        std::printf("Simulation time: %ld ms\n", simulationDurationMilliseconds);

        if (printResults) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                const size_t outputIndex = static_cast<size_t>(i) * 6;
                bodyData[outputIndex] = hostGlobalPositions[i].x;
                bodyData[outputIndex + 1] = hostGlobalPositions[i].y;
                bodyData[outputIndex + 2] = hostGlobalPositions[i].z;
                bodyData[outputIndex + 3] = finalVelocities[i].x;
                bodyData[outputIndex + 4] = finalVelocities[i].y;
                bodyData[outputIndex + 5] = finalVelocities[i].z;
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            std::printf("Validating simulation results...\n");
            if (validateSimulation(hostGlobalPositions, finalVelocities, numBodies)) {
                const double finalEnergy =
                    computeTotalEnergy(hostGlobalPositions, finalVelocities, numBodies);
                std::printf("Final energy: %.6f\n", finalEnergy);
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = EXIT_FAILURE;
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));

    CUDA_CHECK(cudaFree(deviceLocalVelocities));
    CUDA_CHECK(cudaFree(deviceLocalPositions));
    CUDA_CHECK(cudaFree(deviceGlobalPositions));
    CUDA_CHECK(cudaFreeHost(hostLocalVelocities));
    CUDA_CHECK(cudaFreeHost(hostLocalPositions));
    CUDA_CHECK(cudaFreeHost(hostGlobalPositions));
    MPI_CHECK(MPI_Type_free(&mpiDeviceVec3));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));
    return exitCode;
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    const int initializationError =
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    if (initializationError != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI initialization failed\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide the required MPI_THREAD_FUNNELED level\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool validArguments = true;

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
            validArguments = false;
        }
    }
    if (numBodies <= 0 || numSteps < 0) {
        if (rank == 0) {
            std::fprintf(stderr, "The body count must be positive and the step count non-negative.\n");
        }
        validArguments = false;
    }

    int exitCode = EXIT_SUCCESS;
    if (showHelp || !validArguments) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        exitCode = validArguments ? EXIT_SUCCESS : EXIT_FAILURE;
    } else {
        exitCode = runSimulation(rank, worldSize, numBodies, numSteps, validate, printResults);
    }

    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
