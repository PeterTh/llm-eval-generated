#include <cuda_runtime.h>

#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <cstddef>
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
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be tightly packed");
static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be tightly packed");

// Every thread evaluates one target body. The source positions are tiled in
// shared memory, which avoids reloading each source body once per target
// thread in a block.
__global__ void computeForcesKernel(const double* __restrict__ posX,
                                    const double* __restrict__ posY,
                                    const double* __restrict__ posZ,
                                    double* __restrict__ velX,
                                    double* __restrict__ velY,
                                    double* __restrict__ velZ,
                                    const int numBodies,
                                    const int firstBody,
                                    const int bodyCount) {
    __shared__ double tileX[CUDA_BLOCK_SIZE];
    __shared__ double tileY[CUDA_BLOCK_SIZE];
    __shared__ double tileZ[CUDA_BLOCK_SIZE];

    const int localIndex = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = localIndex < bodyCount;
    const int target = firstBody + localIndex;

    const double xi = active ? posX[target] : 0.0;
    const double yi = active ? posY[target] : 0.0;
    const double zi = active ? posZ[target] : 0.0;

    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileStart = 0; tileStart < numBodies; tileStart += blockDim.x) {
        const int source = tileStart + static_cast<int>(threadIdx.x);
        if (source < numBodies) {
            tileX[threadIdx.x] = posX[source];
            tileY[threadIdx.x] = posY[source];
            tileZ[threadIdx.x] = posZ[source];
        } else {
            tileX[threadIdx.x] = 0.0;
            tileY[threadIdx.x] = 0.0;
            tileZ[threadIdx.x] = 0.0;
        }
        __syncthreads();

        const int tileCount = (numBodies - tileStart < blockDim.x) ?
                                  (numBodies - tileStart) : blockDim.x;
        if (active) {
            // Keep this loop in source order to retain the original reduction
            // semantics for each target body.
            for (int k = 0; k < tileCount; ++k) {
                const double dx = tileX[k] - xi;
                const double dy = tileY[k] - yi;
                const double dz = tileZ[k] - zi;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                forceX += dx * invDist3;
                forceY += dy * invDist3;
                forceZ += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        velX[target] += DT * forceX;
        velY[target] += DT * forceY;
        velZ[target] += DT * forceZ;
    }
}

void checkCuda(const cudaError_t result, const char* operation, const int rank) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "MPI rank %d: CUDA operation '%s' failed: %s\n",
                     rank, operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation, rank)

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void unpackBodies(const std::vector<Body>& bodies,
                  std::vector<double>& posX,
                  std::vector<double>& posY,
                  std::vector<double>& posZ,
                  std::vector<double>& velX,
                  std::vector<double>& velY,
                  std::vector<double>& velZ) {
    const int numBodies = static_cast<int>(bodies.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        posX[i] = bodies[i].pos.x;
        posY[i] = bodies[i].pos.y;
        posZ[i] = bodies[i].pos.z;
        velX[i] = bodies[i].vel.x;
        velY[i] = bodies[i].vel.y;
        velZ[i] = bodies[i].vel.z;
    }
}

void integrateLocalBodies(std::vector<Body>& bodies,
                          const std::vector<double>& velX,
                          const std::vector<double>& velY,
                          const std::vector<double>& velZ,
                          std::vector<double>& posX,
                          std::vector<double>& posY,
                          std::vector<double>& posZ,
                          const int firstBody,
                          const int bodyCount) {
#pragma omp parallel for schedule(static)
    for (int localIndex = 0; localIndex < bodyCount; ++localIndex) {
        const int i = firstBody + localIndex;
        bodies[i].vel.x = velX[i];
        bodies[i].vel.y = velY[i];
        bodies[i].vel.z = velZ[i];
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
        posX[i] = bodies[i].pos.x;
        posY[i] = bodies[i].pos.y;
        posZ[i] = bodies[i].pos.z;
    }
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            std::printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            std::printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                         body.vel.y * body.vel.y +
                         body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
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
    int providedThreadLevel = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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

    // Parse command line arguments identically on every rank.
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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) {
            std::fprintf(stderr, "Number of bodies and steps must be non-negative\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select one accelerator per MPI rank, using the node-local rank so that
    // ranks on different nodes independently select device zero, one, ... .
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    int localRank = 0;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localCommunicator);
    MPI_Comm_rank(localCommunicator, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA accelerator is available for the nbody benchmark\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    MPI_Datatype mpiBody;
    MPI_Type_contiguous(6, MPI_DOUBLE, &mpiBody);
    MPI_Type_commit(&mpiBody);

    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    if (numBodies > 0) {
        MPI_Bcast(bodies.data(), numBodies, mpiBody, 0, MPI_COMM_WORLD);
    }

    const int firstBody = static_cast<int>((static_cast<long long>(numBodies) * rank) / worldSize);
    const int lastBody = static_cast<int>((static_cast<long long>(numBodies) * (rank + 1)) / worldSize);
    const int bodyCount = lastBody - firstBody;

    std::vector<int> bodyCounts(worldSize);
    std::vector<int> bodyDisplacements(worldSize);
    for (int process = 0; process < worldSize; ++process) {
        const int begin = static_cast<int>((static_cast<long long>(numBodies) * process) / worldSize);
        const int end = static_cast<int>((static_cast<long long>(numBodies) * (process + 1)) / worldSize);
        bodyDisplacements[process] = begin;
        bodyCounts[process] = end - begin;
    }

    std::vector<double> posX(static_cast<size_t>(numBodies));
    std::vector<double> posY(static_cast<size_t>(numBodies));
    std::vector<double> posZ(static_cast<size_t>(numBodies));
    std::vector<double> velX(static_cast<size_t>(numBodies));
    std::vector<double> velY(static_cast<size_t>(numBodies));
    std::vector<double> velZ(static_cast<size_t>(numBodies));
    unpackBodies(bodies, posX, posY, posZ, velX, velY, velZ);

    double* devicePosX = nullptr;
    double* devicePosY = nullptr;
    double* devicePosZ = nullptr;
    double* deviceVelX = nullptr;
    double* deviceVelY = nullptr;
    double* deviceVelZ = nullptr;
    if (numBodies > 0) {
        const size_t bytes = static_cast<size_t>(numBodies) * sizeof(double);
        CUDA_CHECK(cudaMalloc(&devicePosX, bytes));
        CUDA_CHECK(cudaMalloc(&devicePosY, bytes));
        CUDA_CHECK(cudaMalloc(&devicePosZ, bytes));
        CUDA_CHECK(cudaMalloc(&deviceVelX, bytes));
        CUDA_CHECK(cudaMalloc(&deviceVelY, bytes));
        CUDA_CHECK(cudaMalloc(&deviceVelZ, bytes));
    }

    omp_set_dynamic(0);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        if (numBodies > 0) {
            const size_t positionBytes = static_cast<size_t>(numBodies) * sizeof(double);
            CUDA_CHECK(cudaMemcpy(devicePosX, posX.data(), positionBytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(devicePosY, posY.data(), positionBytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(devicePosZ, posZ.data(), positionBytes, cudaMemcpyHostToDevice));

            if (bodyCount > 0) {
                const size_t localBytes = static_cast<size_t>(bodyCount) * sizeof(double);
                CUDA_CHECK(cudaMemcpy(deviceVelX + firstBody, velX.data() + firstBody,
                                      localBytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(deviceVelY + firstBody, velY.data() + firstBody,
                                      localBytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(deviceVelZ + firstBody, velZ.data() + firstBody,
                                      localBytes, cudaMemcpyHostToDevice));

                const int blocks = (bodyCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
                computeForcesKernel<<<blocks, CUDA_BLOCK_SIZE>>>(
                    devicePosX, devicePosY, devicePosZ,
                    deviceVelX, deviceVelY, deviceVelZ,
                    numBodies, firstBody, bodyCount);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaMemcpy(velX.data() + firstBody, deviceVelX + firstBody,
                                      localBytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(velY.data() + firstBody, deviceVelY + firstBody,
                                      localBytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(velZ.data() + firstBody, deviceVelZ + firstBody,
                                      localBytes, cudaMemcpyDeviceToHost));
            }
        }

        // Each rank integrates its own updated partition. The in-place
        // all-gather then makes the complete state available to every GPU for
        // the next force evaluation.
        integrateLocalBodies(bodies, velX, velY, velZ,
                             posX, posY, posZ, firstBody, bodyCount);
        if (numBodies > 0) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, mpiBody,
                           bodies.data(), bodyCounts.data(), bodyDisplacements.data(),
                           mpiBody, MPI_COMM_WORLD);
            unpackBodies(bodies, posX, posY, posZ, velX, velY, velZ);
        }
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long long elapsedMilliseconds = static_cast<long long>(elapsed * 1000.0);
        std::printf("Simulation time: %lld ms\n", elapsedMilliseconds);
    }

    if (rank == 0 && printResults) {
        // Serialize body positions and velocities for hashing. Parallelizing
        // the independent stores preserves the original output order.
        std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const size_t offset = static_cast<size_t>(i) * 6;
            bodyData[offset + 0] = bodies[i].pos.x;
            bodyData[offset + 1] = bodies[i].pos.y;
            bodyData[offset + 2] = bodies[i].pos.z;
            bodyData[offset + 3] = bodies[i].vel.x;
            bodyData[offset + 4] = bodies[i].vel.y;
            bodyData[offset + 5] = bodies[i].vel.z;
        }
        print_results(bodyData, "Bodies");
    }

    int exitCode = 0;
    if (rank == 0 && validate) {
        std::printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            // Report final energy for reference
            const double finalEnergy = computeTotalEnergy(bodies);
            std::printf("Final energy: %.6f\n", finalEnergy);
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (numBodies > 0) {
        CUDA_CHECK(cudaFree(devicePosX));
        CUDA_CHECK(cudaFree(devicePosY));
        CUDA_CHECK(cudaFree(devicePosZ));
        CUDA_CHECK(cudaFree(deviceVelX));
        CUDA_CHECK(cudaFree(deviceVelY));
        CUDA_CHECK(cudaFree(deviceVelZ));
    }
    MPI_Type_free(&mpiBody);
    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return exitCode;
}

#undef CUDA_CHECK
