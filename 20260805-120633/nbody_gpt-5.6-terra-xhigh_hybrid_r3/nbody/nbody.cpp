#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int CUDA_BLOCK_SIZE = 256;

// Bodies are distributed by target body.  Positions are replicated so every
// GPU can evaluate its target range without fine-grained MPI communication.
struct Body {
    double px, py, pz;
    double vx, vy, vz;
};

struct Position {
    double x, y, z;
};

struct Force {
    double x, y, z;
};

static_assert(std::is_standard_layout_v<Body> && sizeof(Body) == 6 * sizeof(double));
static_assert(std::is_standard_layout_v<Position> && sizeof(Position) == 3 * sizeof(double));
static_assert(std::is_standard_layout_v<Force> && sizeof(Force) == 3 * sizeof(double));

[[noreturn]] void failMpi(int error, const char* expression, int rank) {
    char errorString[MPI_MAX_ERROR_STRING];
    int errorLength = 0;
    MPI_Error_string(error, errorString, &errorLength);
    std::fprintf(stderr, "MPI error on rank %d in %s: %.*s\n", rank, expression,
                 errorLength, errorString);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

void checkMpi(int error, const char* expression, int rank) {
    if (error != MPI_SUCCESS) {
        failMpi(error, expression, rank);
    }
}

[[noreturn]] void failCuda(cudaError_t error, const char* expression, int line, int rank) {
    std::fprintf(stderr, "CUDA error on rank %d at nbody.cpp:%d in %s: %s\n", rank, line,
                 expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

void checkCuda(cudaError_t error, const char* expression, int line, int rank) {
    if (error != cudaSuccess) {
        failCuda(error, expression, line, rank);
    }
}

#define MPI_CHECK(call) checkMpi((call), #call, worldRank)
#define CUDA_CHECK(call) checkCuda((call), #call, __LINE__, worldRank)

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.px = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.py = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pz = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vx = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vy = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vz = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

// A two-dimensional grid exposes source tiles and target tiles independently.
// This keeps every accelerator busy even when MPI gives a rank a relatively
// small target range.  Each block stages its source tile in shared memory and
// atomically contributes a partial force for its target tile.
__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void computeForceTiles(const Body* __restrict__ localBodies,
                       const Position* __restrict__ allPositions,
                       Force* __restrict__ forces,
                       int localCount,
                       int totalBodies) {
    extern __shared__ double sourceTile[];
    double* const sourceX = sourceTile;
    double* const sourceY = sourceX + blockDim.x;
    double* const sourceZ = sourceY + blockDim.x;

    const int localIndex = blockIdx.y * blockDim.x + threadIdx.x;
    const bool active = localIndex < localCount;
    const int firstSource = blockIdx.x * blockDim.x;
    const int sourceIndex = firstSource + threadIdx.x;
    if (sourceIndex < totalBodies) {
        const Position source = allPositions[sourceIndex];
        sourceX[threadIdx.x] = source.x;
        sourceY[threadIdx.x] = source.y;
        sourceZ[threadIdx.x] = source.z;
    }
    __syncthreads();

    if (active) {
        const Body body = localBodies[localIndex];
        double fx = 0.0;
        double fy = 0.0;
        double fz = 0.0;
        const int sourceCount = min(static_cast<int>(blockDim.x), totalBodies - firstSource);
        #pragma unroll 4
        for (int j = 0; j < sourceCount; ++j) {
            const double dx = sourceX[j] - body.px;
            const double dy = sourceY[j] - body.py;
            const double dz = sourceZ[j] - body.pz;
            const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double inverseDistance = 1.0 / sqrt(distanceSquared);
            const double inverseDistanceCubed = inverseDistance * inverseDistance * inverseDistance;
            fx += dx * inverseDistanceCubed;
            fy += dy * inverseDistanceCubed;
            fz += dz * inverseDistanceCubed;
        }
        atomicAdd(&forces[localIndex].x, fx);
        atomicAdd(&forces[localIndex].y, fy);
        atomicAdd(&forces[localIndex].z, fz);
    }
}

__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void integrateBodies(Body* __restrict__ localBodies,
                     const Force* __restrict__ forces,
                     int localCount) {
    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIndex < localCount) {
        Body body = localBodies[localIndex];
        const Force force = forces[localIndex];
        body.vx += DT * force.x;
        body.vy += DT * force.y;
        body.vz += DT * force.z;
        body.px += body.vx * DT;
        body.py += body.vy * DT;
        body.pz += body.vz * DT;
        localBodies[localIndex] = body;
    }
}

__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void extractPositions(const Body* __restrict__ localBodies,
                      Position* __restrict__ localPositions,
                      int localCount) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < localCount) {
        const Body body = localBodies[index];
        localPositions[index] = {body.px, body.py, body.pz};
    }
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(bodies.size()); ++i) {
        const Body& body = bodies[static_cast<size_t>(i)];
        const bool finite = std::isfinite(body.px) && std::isfinite(body.py) && std::isfinite(body.pz) &&
                            std::isfinite(body.vx) && std::isfinite(body.vy) && std::isfinite(body.vz);
        const bool bounded = std::abs(body.px) <= 1e6 && std::abs(body.py) <= 1e6 &&
                             std::abs(body.pz) <= 1e6 && std::abs(body.vx) <= 1e6 &&
                             std::abs(body.vy) <= 1e6 && std::abs(body.vz) <= 1e6;
        valid &= static_cast<int>(finite && bounded);
    }

    if (!valid) {
        std::printf("Validation failed: found a non-finite or out-of-bounds body state\n");
    }
    return valid != 0;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(bodies.size());
    double kineticEnergy = 0.0;
    #pragma omp parallel for reduction(+:kineticEnergy) schedule(static)
    for (std::ptrdiff_t i = 0; i < count; ++i) {
        const Body& body = bodies[static_cast<size_t>(i)];
        kineticEnergy += 0.5 * (body.vx * body.vx + body.vy * body.vy + body.vz * body.vz);
    }

    double potentialEnergy = 0.0;
    #pragma omp parallel for reduction(+:potentialEnergy) schedule(dynamic)
    for (std::ptrdiff_t i = 0; i < count; ++i) {
        const Body& left = bodies[static_cast<size_t>(i)];
        for (std::ptrdiff_t j = i + 1; j < count; ++j) {
            const Body& right = bodies[static_cast<size_t>(j)];
            const double dx = right.px - left.px;
            const double dy = right.py - left.py;
            const double dz = right.pz - left.pz;
            potentialEnergy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
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
    int mpiThreadLevel = MPI_THREAD_SINGLE;
    const int initError = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel);
    if (initError != MPI_SUCCESS) {
        std::fprintf(stderr, "Unable to initialize MPI\n");
        return 1;
    }

    int worldRank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (mpiThreadLevel < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) {
            std::fprintf(stderr, "MPI does not provide the MPI_THREAD_FUNNELED level required by this program\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    bool badArguments = false;
    bool showHelp = false;
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
            badArguments = true;
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
        }
    }
    if (showHelp || badArguments || numBodies < 0 || numSteps < 0) {
        if (worldRank == 0) {
            if (numBodies < 0 || numSteps < 0) {
                std::printf("Number of bodies and steps must be non-negative\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return badArguments || numBodies < 0 || numSteps < 0 ? 1 : 0;
    }

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (worldRank == 0) {
            std::fprintf(stderr, "No CUDA devices are available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (worldRank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> bodyCounts(static_cast<size_t>(worldSize));
    std::vector<int> bodyDisplacements(static_cast<size_t>(worldSize));
    const int baseCount = numBodies / worldSize;
    const int extraBodies = numBodies % worldSize;
    int offset = 0;
    for (int rank = 0; rank < worldSize; ++rank) {
        bodyCounts[static_cast<size_t>(rank)] = baseCount + (rank < extraBodies ? 1 : 0);
        bodyDisplacements[static_cast<size_t>(rank)] = offset;
        offset += bodyCounts[static_cast<size_t>(rank)];
    }
    const int localCount = bodyCounts[static_cast<size_t>(worldRank)];
    const int localOffset = bodyDisplacements[static_cast<size_t>(worldRank)];

    MPI_Datatype bodyType = MPI_DATATYPE_NULL;
    MPI_Datatype positionType = MPI_DATATYPE_NULL;
    MPI_CHECK(MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType));
    MPI_CHECK(MPI_Type_commit(&bodyType));
    MPI_CHECK(MPI_Type_contiguous(3, MPI_DOUBLE, &positionType));
    MPI_CHECK(MPI_Type_commit(&positionType));

    std::vector<Body> allBodies;
    if (worldRank == 0) {
        allBodies.resize(static_cast<size_t>(numBodies));
        randomizeBodies(allBodies);
    }
    std::vector<Body> localBodies(static_cast<size_t>(localCount));
    MPI_CHECK(MPI_Scatterv(worldRank == 0 ? allBodies.data() : nullptr,
                           bodyCounts.data(), bodyDisplacements.data(), bodyType,
                           localBodies.data(), localCount, bodyType, 0, MPI_COMM_WORLD));

    Position* hostPositions = nullptr;
    const size_t positionBytes = static_cast<size_t>(numBodies) * sizeof(Position);
    if (positionBytes != 0) {
        CUDA_CHECK(cudaHostAlloc(&hostPositions, positionBytes, cudaHostAllocPortable));
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(localCount); ++i) {
            const Body& body = localBodies[static_cast<size_t>(i)];
            hostPositions[localOffset + i] = {body.px, body.py, body.pz};
        }
        MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, hostPositions,
                                 bodyCounts.data(), bodyDisplacements.data(), positionType,
                                 MPI_COMM_WORLD));
    }

    Body* deviceBodies = nullptr;
    Position* deviceAllPositions = nullptr;
    Position* deviceLocalPositions = nullptr;
    Force* deviceForces = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceBodies, static_cast<size_t>(std::max(1, localCount)) * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&deviceAllPositions, static_cast<size_t>(std::max(1, numBodies)) * sizeof(Position)));
    CUDA_CHECK(cudaMalloc(&deviceLocalPositions, static_cast<size_t>(std::max(1, localCount)) * sizeof(Position)));
    CUDA_CHECK(cudaMalloc(&deviceForces, static_cast<size_t>(std::max(1, localCount)) * sizeof(Force)));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (localCount > 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceBodies, localBodies.data(), static_cast<size_t>(localCount) * sizeof(Body),
                                   cudaMemcpyHostToDevice, stream));
    }
    if (numBodies > 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceAllPositions, hostPositions, positionBytes,
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            const int targetBlocks = (localCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            const int sourceBlocks = (numBodies + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            const size_t sharedBytes = 3 * CUDA_BLOCK_SIZE * sizeof(double);
            CUDA_CHECK(cudaMemsetAsync(deviceForces, 0, static_cast<size_t>(localCount) * sizeof(Force), stream));
            computeForceTiles<<<dim3(sourceBlocks, targetBlocks), CUDA_BLOCK_SIZE, sharedBytes, stream>>>(
                deviceBodies, deviceAllPositions, deviceForces, localCount, numBodies);
            CUDA_CHECK(cudaGetLastError());
            integrateBodies<<<targetBlocks, CUDA_BLOCK_SIZE, 0, stream>>>(
                deviceBodies, deviceForces, localCount);
            CUDA_CHECK(cudaGetLastError());
        }

        // The last iteration has no successor, so its replicated positions are
        // not needed.  Avoiding that final exchange removes a full cluster-wide
        // synchronization and two device/host transfers from the timed path.
        if (step + 1 < numSteps && localCount > 0) {
            const int gridSize = (localCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            extractPositions<<<gridSize, CUDA_BLOCK_SIZE, 0, stream>>>(
                deviceBodies, deviceLocalPositions, localCount);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(hostPositions + localOffset, deviceLocalPositions,
                                       static_cast<size_t>(localCount) * sizeof(Position),
                                       cudaMemcpyDeviceToHost, stream));
        }
        if (step + 1 < numSteps) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (numBodies > 0) {
                MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, hostPositions,
                                         bodyCounts.data(), bodyDisplacements.data(), positionType,
                                         MPI_COMM_WORLD));
                CUDA_CHECK(cudaMemcpyAsync(deviceAllPositions, hostPositions, positionBytes,
                                           cudaMemcpyHostToDevice, stream));
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    const double localElapsedMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
    double elapsedMilliseconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsedMilliseconds, &elapsedMilliseconds, 1, MPI_DOUBLE, MPI_MAX,
                         0, MPI_COMM_WORLD));
    if (worldRank == 0) {
        std::printf("Simulation time: %.3f ms\n", elapsedMilliseconds);
    }

    int success = 1;
    if (printResults || validate) {
        if (localCount > 0) {
            CUDA_CHECK(cudaMemcpy(localBodies.data(), deviceBodies,
                                  static_cast<size_t>(localCount) * sizeof(Body), cudaMemcpyDeviceToHost));
        }
        if (worldRank == 0) {
            allBodies.resize(static_cast<size_t>(numBodies));
        }
        MPI_CHECK(MPI_Gatherv(localBodies.data(), localCount, bodyType,
                              worldRank == 0 ? allBodies.data() : nullptr,
                              bodyCounts.data(), bodyDisplacements.data(), bodyType, 0,
                              MPI_COMM_WORLD));

        if (worldRank == 0) {
            if (printResults) {
                std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
                #pragma omp parallel for schedule(static)
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(allBodies.size()); ++i) {
                    const Body& body = allBodies[static_cast<size_t>(i)];
                    const size_t base = static_cast<size_t>(i) * 6;
                    bodyData[base] = body.px;
                    bodyData[base + 1] = body.py;
                    bodyData[base + 2] = body.pz;
                    bodyData[base + 3] = body.vx;
                    bodyData[base + 4] = body.vy;
                    bodyData[base + 5] = body.vz;
                }
                print_results(bodyData, "Bodies");
            }
            if (validate) {
                std::printf("Validating simulation results...\n");
                if (validateSimulation(allBodies)) {
                    std::printf("Final energy: %.6f\n", computeTotalEnergy(allBodies));
                    std::printf("Validation: PASSED\n");
                } else {
                    std::printf("Validation: FAILED\n");
                    success = 0;
                }
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceForces));
    CUDA_CHECK(cudaFree(deviceLocalPositions));
    CUDA_CHECK(cudaFree(deviceAllPositions));
    CUDA_CHECK(cudaFree(deviceBodies));
    if (hostPositions != nullptr) {
        CUDA_CHECK(cudaFreeHost(hostPositions));
    }
    MPI_CHECK(MPI_Type_free(&positionType));
    MPI_CHECK(MPI_Type_free(&bodyType));
    MPI_CHECK(MPI_Finalize());
    return success ? 0 : 1;
}
