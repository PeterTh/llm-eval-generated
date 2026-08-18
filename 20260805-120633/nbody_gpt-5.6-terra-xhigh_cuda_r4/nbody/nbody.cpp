#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr unsigned int THREADS_PER_BLOCK = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

void cudaCheck(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) cudaCheck((operation), #operation)

// Each block computes the next state for a contiguous group of bodies.  The
// source and destination arrays are distinct, so no block can observe a
// position written by another block during the same integration step.
__global__ void advanceBodies(const double4* __restrict__ sourcePositions,
                              const double4* __restrict__ sourceVelocities,
                              double4* __restrict__ destinationPositions,
                              double4* __restrict__ destinationVelocities,
                              const unsigned int numBodies) {
    extern __shared__ double4 positionTile[];

    const unsigned int localIndex = threadIdx.x;
    const unsigned int bodyIndex = blockIdx.x * blockDim.x + localIndex;
    const bool isActive = bodyIndex < numBodies;

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (isActive) {
        const double4 position = sourcePositions[bodyIndex];
        x = position.x;
        y = position.y;
        z = position.z;
    }

    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (unsigned int tileStart = 0; tileStart < numBodies; tileStart += blockDim.x) {
        const unsigned int sourceIndex = tileStart + localIndex;
        if (sourceIndex < numBodies) {
            positionTile[localIndex] = sourcePositions[sourceIndex];
        }
        __syncthreads();

        if (isActive) {
            const unsigned int tileSize = min(blockDim.x, numBodies - tileStart);
#pragma unroll 8
            for (unsigned int j = 0; j < tileSize; ++j) {
                const double dx = positionTile[j].x - x;
                const double dy = positionTile[j].y - y;
                const double dz = positionTile[j].z - z;
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

    if (isActive) {
        const double4 velocity = sourceVelocities[bodyIndex];
        const double nextVelocityX = velocity.x + DT * forceX;
        const double nextVelocityY = velocity.y + DT * forceY;
        const double nextVelocityZ = velocity.z + DT * forceZ;

        destinationVelocities[bodyIndex] = make_double4(nextVelocityX, nextVelocityY, nextVelocityZ, 0.0);
        destinationPositions[bodyIndex] = make_double4(x + DT * nextVelocityX,
                                                        y + DT * nextVelocityY,
                                                        z + DT * nextVelocityZ,
                                                        0.0);
    }
}

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

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
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

int main(int argc, char** argv) {
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Initialization stays on the host to retain the benchmark's deterministic
    // random sequence.  The simulation state itself is entirely device-resident.
    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    randomizeBodies(bodies);

    if (numBodies == 0) {
        printf("Simulation time: 0 ms\n");
        if (printResults) {
            print_results(std::vector<double>{}, "Bodies");
        }
        if (validate) {
            printf("Validating simulation results...\n");
            printf("Final energy: %.6f\n", computeTotalEnergy(bodies));
            printf("Validation: PASSED\n");
        }
        return 0;
    }

    const size_t stateBytes = static_cast<size_t>(numBodies) * sizeof(double4);
    std::vector<double4> hostPositions(static_cast<size_t>(numBodies));
    std::vector<double4> hostVelocities(static_cast<size_t>(numBodies));
    for (int i = 0; i < numBodies; ++i) {
        hostPositions[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
        hostVelocities[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
    }

    double4* positionBuffers[2] = {nullptr, nullptr};
    double4* velocityBuffers[2] = {nullptr, nullptr};
    for (int buffer = 0; buffer < 2; ++buffer) {
        CUDA_CHECK(cudaMalloc(&positionBuffers[buffer], stateBytes));
        CUDA_CHECK(cudaMalloc(&velocityBuffers[buffer], stateBytes));
    }
    CUDA_CHECK(cudaMemcpy(positionBuffers[0], hostPositions.data(), stateBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(velocityBuffers[0], hostVelocities.data(), stateBytes, cudaMemcpyHostToDevice));

    cudaEvent_t startEvent;
    cudaEvent_t endEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&endEvent));

    const unsigned int blockCount =
        (static_cast<unsigned int>(numBodies) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const size_t sharedMemoryBytes = THREADS_PER_BLOCK * sizeof(double4);
    int currentBuffer = 0;

    // The event timing covers only the GPU simulation, matching the original
    // benchmark's exclusion of initialization and result serialization.
    CUDA_CHECK(cudaEventRecord(startEvent));
    for (int step = 0; step < numSteps; ++step) {
        const int nextBuffer = currentBuffer ^ 1;
        advanceBodies<<<blockCount, THREADS_PER_BLOCK, sharedMemoryBytes>>>(
            positionBuffers[currentBuffer], velocityBuffers[currentBuffer],
            positionBuffers[nextBuffer], velocityBuffers[nextBuffer],
            static_cast<unsigned int>(numBodies));
        CUDA_CHECK(cudaGetLastError());
        currentBuffer = nextBuffer;
    }
    CUDA_CHECK(cudaEventRecord(endEvent));
    CUDA_CHECK(cudaEventSynchronize(endEvent));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, endEvent));
    printf("Simulation time: %ld ms\n", static_cast<long>(elapsedMilliseconds));

    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(hostPositions.data(), positionBuffers[currentBuffer], stateBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostVelocities.data(), velocityBuffers[currentBuffer], stateBytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = Vec3(hostPositions[i].x, hostPositions[i].y, hostPositions[i].z);
            bodies[i].vel = Vec3(hostVelocities[i].x, hostVelocities[i].y, hostVelocities[i].z);
        }
    }

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(endEvent));
    for (int buffer = 0; buffer < 2; ++buffer) {
        CUDA_CHECK(cudaFree(positionBuffers[buffer]));
        CUDA_CHECK(cudaFree(velocityBuffers[buffer]));
    }

    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (const auto& body : bodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            // Report final energy for reference
            const double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            return 0;
        }

        printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
