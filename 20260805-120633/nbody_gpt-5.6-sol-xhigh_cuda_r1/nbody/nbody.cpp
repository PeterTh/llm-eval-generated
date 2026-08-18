#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

constexpr int CUDA_BLOCK_SIZE = 32;
constexpr int CUDA_UNROLL = 8;

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
                 file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_error = (expression);                        \
        if (cuda_check_error != cudaSuccess) {                                    \
            cudaFailure(cuda_check_error, #expression, __FILE__, __LINE__);       \
        }                                                                         \
    } while (false)

// A warp advances 32 bodies and cooperatively stages 32 source positions at a
// time. Warp synchronization is sufficient for the one-warp block, minimizing
// barriers while retaining broadcast-friendly shared-memory reads. Positions
// are double-buffered so velocity update and integration can safely stay fused.
__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void simulationStepKernel(const double3* __restrict__ positions,
                          double3* __restrict__ nextPositions,
                          double3* __restrict__ velocities,
                          const int numBodies) {
    __shared__ double3 positionTile[CUDA_BLOCK_SIZE];

    const int bodyIndex = static_cast<int>(blockIdx.x) * CUDA_BLOCK_SIZE
                        + static_cast<int>(threadIdx.x);

    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    if (bodyIndex < numBodies) {
        const double3 position = positions[bodyIndex];
        px = position.x;
        py = position.y;
        pz = position.z;
    }

    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileStart = 0; tileStart < numBodies;
         tileStart += CUDA_BLOCK_SIZE) {
        const int sourceIndex = tileStart + static_cast<int>(threadIdx.x);
        if (sourceIndex < numBodies) {
            positionTile[threadIdx.x] = positions[sourceIndex];
        }
        __syncwarp(__activemask());

        const int tileSize = min(CUDA_BLOCK_SIZE, numBodies - tileStart);
        if (bodyIndex < numBodies) {
#pragma unroll CUDA_UNROLL
            for (int j = 0; j < tileSize; ++j) {
                const double3 source = positionTile[j];
                const double dx = source.x - px;
                const double dy = source.y - py;
                const double dz = source.z - pz;
                const double distanceSquared =
                    dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inverseDistance = rsqrt(distanceSquared);
                const double inverseDistanceCubed = inverseDistance
                                                   * inverseDistance
                                                   * inverseDistance;

                forceX += dx * inverseDistanceCubed;
                forceY += dy * inverseDistanceCubed;
                forceZ += dz * inverseDistanceCubed;
            }
        }
        __syncwarp(__activemask());
    }

    if (bodyIndex < numBodies) {
        double3 velocity = velocities[bodyIndex];
        velocity.x += DT * forceX;
        velocity.y += DT * forceY;
        velocity.z += DT * forceZ;
        velocities[bodyIndex] = velocity;
        nextPositions[bodyIndex] = make_double3(
            px + velocity.x * DT,
            py + velocity.y * DT,
            pz + velocity.z * DT);
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

    if (numBodies < 0 || numSteps < 0) {
        std::fprintf(stderr, "Number of bodies and steps must be non-negative\n");
        return 1;
    }
    
    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Pack the host representation into aligned vectors for fully coalesced
    // device accesses. Transfers and allocation are initialization/finalization
    // work and intentionally remain outside the measured simulation interval.
    std::vector<double3> positions(static_cast<size_t>(numBodies));
    std::vector<double3> velocities(static_cast<size_t>(numBodies));
    for (int i = 0; i < numBodies; ++i) {
        positions[i] = make_double3(bodies[i].pos.x, bodies[i].pos.y,
                                    bodies[i].pos.z);
        velocities[i] = make_double3(bodies[i].vel.x, bodies[i].vel.y,
                                     bodies[i].vel.z);
    }

    double3* devicePositions = nullptr;
    double3* deviceNextPositions = nullptr;
    double3* deviceVelocities = nullptr;

    // This also makes absence of a CUDA-capable device an explicit error; the
    // benchmark never falls back to a serial CPU simulation.
    int activeDevice = 0;
    CUDA_CHECK(cudaGetDevice(&activeDevice));

    const size_t deviceBytes = static_cast<size_t>(numBodies) * sizeof(double3);
    if (numBodies > 0) {
        CUDA_CHECK(cudaMalloc(&devicePositions, deviceBytes));
        CUDA_CHECK(cudaMalloc(&deviceNextPositions, deviceBytes));
        CUDA_CHECK(cudaMalloc(&deviceVelocities, deviceBytes));
        CUDA_CHECK(cudaMemcpy(devicePositions, positions.data(), deviceBytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceVelocities, velocities.data(), deviceBytes,
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    if (numBodies > 0) {
        const int gridSize = (numBodies + CUDA_BLOCK_SIZE - 1)
                           / CUDA_BLOCK_SIZE;
        for (int step = 0; step < numSteps; ++step) {
            simulationStepKernel<<<gridSize, CUDA_BLOCK_SIZE>>>(
                devicePositions, deviceNextPositions, deviceVelocities,
                numBodies);
            CUDA_CHECK(cudaGetLastError());
            double3* const oldPositions = devicePositions;
            devicePositions = deviceNextPositions;
            deviceNextPositions = oldPositions;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    if (numBodies > 0) {
        CUDA_CHECK(cudaMemcpy(positions.data(), devicePositions, deviceBytes,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velocities.data(), deviceVelocities, deviceBytes,
                              cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos = Vec3(positions[i].x, positions[i].y, positions[i].z);
        bodies[i].vel = Vec3(velocities[i].x, velocities[i].y, velocities[i].z);
    }

    if (devicePositions != nullptr) {
        CUDA_CHECK(cudaFree(devicePositions));
        CUDA_CHECK(cudaFree(deviceNextPositions));
        CUDA_CHECK(cudaFree(deviceVelocities));
    }
    
    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
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
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
