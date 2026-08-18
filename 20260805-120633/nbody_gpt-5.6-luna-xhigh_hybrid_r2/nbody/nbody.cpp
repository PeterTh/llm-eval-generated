#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

void checkCuda(const cudaError_t result, const char* operation, const int rank) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "MPI rank %d: CUDA error in %s: %s\n", rank, operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation, rank) checkCuda((operation), #operation, (rank))

void checkMpi(const int result, const char* operation, const int rank) {
    if (result != MPI_SUCCESS) {
        char errorString[MPI_MAX_ERROR_STRING]{};
        int errorLength = 0;
        MPI_Error_string(result, errorString, &errorLength);
        std::fprintf(stderr, "MPI rank %d: MPI error in %s: %.*s\n", rank, operation, errorLength, errorString);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

#define MPI_CHECK(operation, rank) checkMpi((operation), #operation, (rank))

// Each CUDA thread owns one target body. The source bodies are loaded in tiles
// so that every position read by a thread block is reused from shared memory.
__global__ void computeForcesKernel(const double* __restrict__ posX,
                                     const double* __restrict__ posY,
                                     const double* __restrict__ posZ,
                                     double* __restrict__ velX,
                                     double* __restrict__ velY,
                                     double* __restrict__ velZ,
                                     const int globalStart,
                                     const int localCount,
                                     const int bodyCount) {
    extern __shared__ double sourcePositions[];
    double* sourceX = sourcePositions;
    double* sourceY = sourceX + blockDim.x;
    double* sourceZ = sourceY + blockDim.x;

    const int localIndex = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const bool active = localIndex < localCount;
    const int targetIndex = globalStart + localIndex;

    const double targetX = active ? posX[targetIndex] : 0.0;
    const double targetY = active ? posY[targetIndex] : 0.0;
    const double targetZ = active ? posZ[targetIndex] : 0.0;
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileStart = 0; tileStart < bodyCount; tileStart += blockDim.x) {
        const int sourceIndex = tileStart + static_cast<int>(threadIdx.x);
        if (sourceIndex < bodyCount) {
            sourceX[threadIdx.x] = posX[sourceIndex];
            sourceY[threadIdx.x] = posY[sourceIndex];
            sourceZ[threadIdx.x] = posZ[sourceIndex];
        } else {
            sourceX[threadIdx.x] = 0.0;
            sourceY[threadIdx.x] = 0.0;
            sourceZ[threadIdx.x] = 0.0;
        }
        __syncthreads();

        int tileCount = bodyCount - tileStart;
        if (tileCount > blockDim.x) {
            tileCount = blockDim.x;
        }

        if (active) {
            #pragma unroll 4
            for (int k = 0; k < tileCount; ++k) {
                const double dx = sourceX[k] - targetX;
                const double dy = sourceY[k] - targetY;
                const double dz = sourceZ[k] - targetZ;
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
        velX[localIndex] += DT * forceX;
        velY[localIndex] += DT * forceY;
        velZ[localIndex] += DT * forceZ;
    }
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Keep the original rand_r stream and order so initialization remains
    // deterministic and compatible with the serial benchmark.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void bodiesToSoA(const std::vector<Body>& bodies,
                 std::vector<double>& posX,
                 std::vector<double>& posY,
                 std::vector<double>& posZ,
                 std::vector<double>& velX,
                 std::vector<double>& velY,
                 std::vector<double>& velZ) {
    const long long bodyCount = static_cast<long long>(bodies.size());
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < bodyCount; ++i) {
        const Body& body = bodies[static_cast<std::size_t>(i)];
        posX[static_cast<std::size_t>(i)] = body.pos.x;
        posY[static_cast<std::size_t>(i)] = body.pos.y;
        posZ[static_cast<std::size_t>(i)] = body.pos.z;
        velX[static_cast<std::size_t>(i)] = body.vel.x;
        velY[static_cast<std::size_t>(i)] = body.vel.y;
        velZ[static_cast<std::size_t>(i)] = body.vel.z;
    }
}

void soaToBodies(const std::vector<double>& posX,
                 const std::vector<double>& posY,
                 const std::vector<double>& posZ,
                 const std::vector<double>& velX,
                 const std::vector<double>& velY,
                 const std::vector<double>& velZ,
                 std::vector<Body>& bodies) {
    const long long bodyCount = static_cast<long long>(bodies.size());
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < bodyCount; ++i) {
        Body& body = bodies[static_cast<std::size_t>(i)];
        body.pos.x = posX[static_cast<std::size_t>(i)];
        body.pos.y = posY[static_cast<std::size_t>(i)];
        body.pos.z = posZ[static_cast<std::size_t>(i)];
        body.vel.x = velX[static_cast<std::size_t>(i)];
        body.vel.y = velY[static_cast<std::size_t>(i)];
        body.vel.z = velZ[static_cast<std::size_t>(i)];
    }
}

void initializeLocalVelocities(const std::vector<double>& velX,
                               const std::vector<double>& velY,
                               const std::vector<double>& velZ,
                               double* localVelX,
                               double* localVelY,
                               double* localVelZ,
                               const int localStart,
                               const int localCount) {
    #pragma omp parallel for schedule(static)
    for (long long localIndex = 0; localIndex < localCount; ++localIndex) {
        const std::size_t globalIndex = static_cast<std::size_t>(localStart + localIndex);
        localVelX[localIndex] = velX[globalIndex];
        localVelY[localIndex] = velY[globalIndex];
        localVelZ[localIndex] = velZ[globalIndex];
    }
}

void integrateLocalBodies(std::vector<double>& posX,
                          std::vector<double>& posY,
                          std::vector<double>& posZ,
                          const double* localVelX,
                          const double* localVelY,
                          const double* localVelZ,
                          const int localStart,
                          const int localCount) {
    #pragma omp parallel for schedule(static)
    for (long long localIndex = 0; localIndex < localCount; ++localIndex) {
        const std::size_t globalIndex = static_cast<std::size_t>(localStart + localIndex);
        posX[globalIndex] += localVelX[localIndex] * DT;
        posY[globalIndex] += localVelY[localIndex] * DT;
        posZ[globalIndex] += localVelZ[localIndex] * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const int bodyCount = static_cast<int>(bodies.size());
    double kineticEnergy = 0.0;

    #pragma omp parallel for reduction(+:kineticEnergy) schedule(static)
    for (int i = 0; i < bodyCount; ++i) {
        const Body& body = bodies[static_cast<std::size_t>(i)];
        kineticEnergy += 0.5 * (body.vel.x * body.vel.x +
                                body.vel.y * body.vel.y +
                                body.vel.z * body.vel.z);
    }

    double potentialEnergy = 0.0;
    #pragma omp parallel for reduction(+:potentialEnergy) schedule(static)
    for (int i = 0; i < bodyCount; ++i) {
        const Body& bodyI = bodies[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < bodyCount; ++j) {
            const Body& bodyJ = bodies[static_cast<std::size_t>(j)];
            const double dx = bodyJ.pos.x - bodyI.pos.x;
            const double dy = bodyJ.pos.y - bodyI.pos.y;
            const double dz = bodyJ.pos.z - bodyI.pos.z;
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potentialEnergy -= 1.0 / distance;
        }
    }

    return kineticEnergy + potentialEnergy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int finiteValid = 1;
    int positionValid = 1;
    int velocityValid = 1;
    const long long bodyCount = static_cast<long long>(bodies.size());
    #pragma omp parallel for reduction(&:finiteValid, positionValid, velocityValid) schedule(static)
    for (long long i = 0; i < bodyCount; ++i) {
        const Body& body = bodies[static_cast<std::size_t>(i)];
        const bool finite = std::isfinite(body.pos.x) && std::isfinite(body.pos.y) && std::isfinite(body.pos.z) &&
                            std::isfinite(body.vel.x) && std::isfinite(body.vel.y) && std::isfinite(body.vel.z);
        if (!finite) {
            finiteValid = 0;
        }
        if (std::abs(body.pos.x) > 1e6 || std::abs(body.pos.y) > 1e6 || std::abs(body.pos.z) > 1e6) {
            positionValid = 0;
        }
        if (std::abs(body.vel.x) > 1e6 || std::abs(body.vel.y) > 1e6 || std::abs(body.vel.z) > 1e6) {
            velocityValid = 0;
        }
    }

    if (!finiteValid) {
        std::printf("Validation failed: found NaN or Inf value in body state\n");
    } else if (!positionValid) {
        std::printf("Validation failed: body position exceeds reasonable bounds\n");
    } else if (!velocityValid) {
        std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
    }
    return finiteValid != 0 && positionValid != 0 && velocityValid != 0;
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
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel) != MPI_SUCCESS) {
        return EXIT_FAILURE;
    }

    int rank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank), rank);
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), rank);

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;
    const char* invalidOption = nullptr;

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
            break;
        } else {
            parseError = true;
            if (invalidOption == nullptr) {
                invalidOption = argv[i];
            }
        }
    }

    if (showHelp || parseError || numBodies < 0 || numSteps < 0) {
        if (rank == 0) {
            if (parseError) {
                std::printf("Unknown option: %s\n", invalidOption);
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError || numBodies < 0 || numSteps < 0 ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localCommunicator), rank);
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank), rank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount), rank);
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA device is available for the hybrid nbody benchmark\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount), rank);

    const int baseBodyCount = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    const int localStart = rank * baseBodyCount + std::min(rank, remainder);
    const int localCount = baseBodyCount + (rank < remainder ? 1 : 0);
    const std::size_t bodyCount = static_cast<std::size_t>(numBodies);
    const std::size_t bodyBytes = bodyCount * sizeof(double);
    const std::size_t localBytes = static_cast<std::size_t>(localCount) * sizeof(double);

    std::vector<int> receiveCounts(static_cast<std::size_t>(worldSize));
    std::vector<int> displacements(static_cast<std::size_t>(worldSize));
    for (int process = 0; process < worldSize; ++process) {
        receiveCounts[static_cast<std::size_t>(process)] = baseBodyCount + (process < remainder ? 1 : 0);
        displacements[static_cast<std::size_t>(process)] =
            process * baseBodyCount + std::min(process, remainder);
    }

    std::vector<Body> bodies(bodyCount);
    randomizeBodies(bodies);

    std::vector<double> posX(bodyCount), posY(bodyCount), posZ(bodyCount);
    std::vector<double> velX(bodyCount), velY(bodyCount), velZ(bodyCount);
    bodiesToSoA(bodies, posX, posY, posZ, velX, velY, velZ);

    double* localVelX = nullptr;
    double* localVelY = nullptr;
    double* localVelZ = nullptr;
    if (localCount > 0) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&localVelX), localBytes), rank);
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&localVelY), localBytes), rank);
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&localVelZ), localBytes), rank);
        initializeLocalVelocities(velX, velY, velZ, localVelX, localVelY, localVelZ, localStart, localCount);
    }

    double* devicePosX = nullptr;
    double* devicePosY = nullptr;
    double* devicePosZ = nullptr;
    double* deviceVelX = nullptr;
    double* deviceVelY = nullptr;
    double* deviceVelZ = nullptr;
    if (numBodies > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePosX), bodyBytes), rank);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePosY), bodyBytes), rank);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePosZ), bodyBytes), rank);
    }
    if (localCount > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceVelX), localBytes), rank);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceVelY), localBytes), rank);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceVelZ), localBytes), rank);
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), rank);
    if (numBodies > 0) {
        CUDA_CHECK(cudaMemcpyAsync(devicePosX, posX.data(), bodyBytes, cudaMemcpyHostToDevice, stream), rank);
        CUDA_CHECK(cudaMemcpyAsync(devicePosY, posY.data(), bodyBytes, cudaMemcpyHostToDevice, stream), rank);
        CUDA_CHECK(cudaMemcpyAsync(devicePosZ, posZ.data(), bodyBytes, cudaMemcpyHostToDevice, stream), rank);
    }
    if (localCount > 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceVelX, localVelX, localBytes, cudaMemcpyHostToDevice, stream), rank);
        CUDA_CHECK(cudaMemcpyAsync(deviceVelY, localVelY, localBytes, cudaMemcpyHostToDevice, stream), rank);
        CUDA_CHECK(cudaMemcpyAsync(deviceVelZ, localVelZ, localBytes, cudaMemcpyHostToDevice, stream), rank);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream), rank);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD), rank);
    const double simulationStart = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            const dim3 block(CUDA_BLOCK_SIZE);
            const dim3 grid((localCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
            const std::size_t sharedBytes = 3 * CUDA_BLOCK_SIZE * sizeof(double);
            computeForcesKernel<<<grid, block, sharedBytes, stream>>>(
                devicePosX, devicePosY, devicePosZ,
                deviceVelX, deviceVelY, deviceVelZ,
                localStart, localCount, numBodies);
            CUDA_CHECK(cudaGetLastError(), rank);

            CUDA_CHECK(cudaMemcpyAsync(localVelX, deviceVelX, localBytes, cudaMemcpyDeviceToHost, stream), rank);
            CUDA_CHECK(cudaMemcpyAsync(localVelY, deviceVelY, localBytes, cudaMemcpyDeviceToHost, stream), rank);
            CUDA_CHECK(cudaMemcpyAsync(localVelZ, deviceVelZ, localBytes, cudaMemcpyDeviceToHost, stream), rank);
            CUDA_CHECK(cudaStreamSynchronize(stream), rank);
        }

        // The force phase is GPU-bound; the independent O(N) update is kept on
        // the CPU and threaded with OpenMP while MPI exchanges the new positions.
        integrateLocalBodies(posX, posY, posZ, localVelX, localVelY, localVelZ, localStart, localCount);

        MPI_CHECK(MPI_Allgatherv(
            localCount > 0 ? posX.data() + localStart : nullptr, localCount, MPI_DOUBLE,
            posX.data(), receiveCounts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD), rank);
        MPI_CHECK(MPI_Allgatherv(
            localCount > 0 ? posY.data() + localStart : nullptr, localCount, MPI_DOUBLE,
            posY.data(), receiveCounts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD), rank);
        MPI_CHECK(MPI_Allgatherv(
            localCount > 0 ? posZ.data() + localStart : nullptr, localCount, MPI_DOUBLE,
            posZ.data(), receiveCounts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD), rank);

        if (numBodies > 0) {
            CUDA_CHECK(cudaMemcpyAsync(devicePosX, posX.data(), bodyBytes, cudaMemcpyHostToDevice, stream), rank);
            CUDA_CHECK(cudaMemcpyAsync(devicePosY, posY.data(), bodyBytes, cudaMemcpyHostToDevice, stream), rank);
            CUDA_CHECK(cudaMemcpyAsync(devicePosZ, posZ.data(), bodyBytes, cudaMemcpyHostToDevice, stream), rank);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream), rank);
    const double localSimulationTime = MPI_Wtime() - simulationStart;
    double maximumSimulationTime = 0.0;
    MPI_CHECK(MPI_Reduce(&localSimulationTime, &maximumSimulationTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), rank);

    // Velocities are independent across target partitions, so they only need
    // one final all-gather for validation and result serialization.
    MPI_CHECK(MPI_Allgatherv(
        localCount > 0 ? localVelX : nullptr, localCount, MPI_DOUBLE,
        velX.data(), receiveCounts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD), rank);
    MPI_CHECK(MPI_Allgatherv(
        localCount > 0 ? localVelY : nullptr, localCount, MPI_DOUBLE,
        velY.data(), receiveCounts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD), rank);
    MPI_CHECK(MPI_Allgatherv(
        localCount > 0 ? localVelZ : nullptr, localCount, MPI_DOUBLE,
        velZ.data(), receiveCounts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD), rank);

    if (rank == 0) {
        soaToBodies(posX, posY, posZ, velX, velY, velZ, bodies);
        const long long elapsedMilliseconds = static_cast<long long>(maximumSimulationTime * 1000.0);
        std::printf("Simulation time: %lld ms\n", elapsedMilliseconds);

        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(bodyCount * 6);
            for (const Body& body : bodies) {
                bodyData.push_back(body.pos.x);
                bodyData.push_back(body.pos.y);
                bodyData.push_back(body.pos.z);
                bodyData.push_back(body.vel.x);
                bodyData.push_back(body.vel.y);
                bodyData.push_back(body.vel.z);
            }
            print_results(bodyData, "Bodies");
        }
    }

    int validationOk = 1;
    if (validate && rank == 0) {
        std::printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) {
            const double finalEnergy = computeTotalEnergy(bodies);
            std::printf("Final energy: %.6f\n", finalEnergy);
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            validationOk = 0;
        }
    }
    MPI_CHECK(MPI_Bcast(&validationOk, 1, MPI_INT, 0, MPI_COMM_WORLD), rank);

    CUDA_CHECK(cudaFree(devicePosX), rank);
    CUDA_CHECK(cudaFree(devicePosY), rank);
    CUDA_CHECK(cudaFree(devicePosZ), rank);
    CUDA_CHECK(cudaFree(deviceVelX), rank);
    CUDA_CHECK(cudaFree(deviceVelY), rank);
    CUDA_CHECK(cudaFree(deviceVelZ), rank);
    CUDA_CHECK(cudaStreamDestroy(stream), rank);
    if (localVelX != nullptr) {
        CUDA_CHECK(cudaFreeHost(localVelX), rank);
        CUDA_CHECK(cudaFreeHost(localVelY), rank);
        CUDA_CHECK(cudaFreeHost(localVelZ), rank);
    }
    MPI_CHECK(MPI_Comm_free(&localCommunicator), rank);
    MPI_CHECK(MPI_Finalize(), rank);
    return validationOk ? EXIT_SUCCESS : EXIT_FAILURE;
}
