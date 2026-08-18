#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int THREADS_PER_BLOCK = 128;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                              \
    do {                                                                                              \
        const cudaError_t status = (call);                                                            \
        if (status != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                 \
                         cudaGetErrorString(status));                                                 \
            std::exit(EXIT_FAILURE);                                                                  \
        }                                                                                             \
    } while (0)

// Each block cooperatively caches a tile of source positions.  Positions are double-buffered:
// every thread reads only currentPositions and writes only nextPositions, so the velocity update
// and following position update have the same all-body ordering as the original two CPU loops.
__global__ void advanceBodies(const double4* __restrict__ currentPositions,
                              double4* __restrict__ nextPositions,
                              double4* __restrict__ velocities,
                              const int numBodies) {
    extern __shared__ double4 positionTile[];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    const bool active = i < numBodies;
    const double4 myPosition = active ? currentPositions[i] : make_double4(0.0, 0.0, 0.0, 0.0);

    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;

    for (int tileStart = 0; tileStart < numBodies; tileStart += blockDim.x) {
        const int j = tileStart + tid;
        positionTile[tid] = j < numBodies ? currentPositions[j] : make_double4(0.0, 0.0, 0.0, 0.0);
        __syncthreads();

        const int tileSize = min(blockDim.x, numBodies - tileStart);
        if (active) {
            #pragma unroll 4
            for (int k = 0; k < tileSize; ++k) {
                const double4 otherPosition = positionTile[k];
                const double dx = otherPosition.x - myPosition.x;
                const double dy = otherPosition.y - myPosition.y;
                const double dz = otherPosition.z - myPosition.z;
                const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inverseDistance = 1.0 / sqrt(distanceSquared);
                const double inverseDistanceCubed = inverseDistance * inverseDistance * inverseDistance;

                fx = fma(dx, inverseDistanceCubed, fx);
                fy = fma(dy, inverseDistanceCubed, fy);
                fz = fma(dz, inverseDistanceCubed, fz);
            }
        }
        __syncthreads();
    }

    if (active) {
        double4 velocity = velocities[i];
        velocity.x = fma(DT, fx, velocity.x);
        velocity.y = fma(DT, fy, velocity.y);
        velocity.z = fma(DT, fz, velocity.z);
        velocities[i] = velocity;

        nextPositions[i] = make_double4(fma(velocity.x, DT, myPosition.x),
                                         fma(velocity.y, DT, myPosition.y),
                                         fma(velocity.z, DT, myPosition.z), 0.0);
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

long runSimulation(std::vector<Body>& bodies, const int numSteps) {
    const int numBodies = static_cast<int>(bodies.size());
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::fprintf(stderr, "No CUDA-capable device is available.\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(0));

    if (numBodies == 0 || numSteps == 0) {
        return 0;
    }

    std::vector<double4> hostPositions(static_cast<size_t>(numBodies));
    std::vector<double4> hostVelocities(static_cast<size_t>(numBodies));
    for (int i = 0; i < numBodies; ++i) {
        hostPositions[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
        hostVelocities[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
    }

    const size_t bytes = static_cast<size_t>(numBodies) * sizeof(double4);
    double4* currentPositions = nullptr;
    double4* nextPositions = nullptr;
    double4* deviceVelocities = nullptr;
    CUDA_CHECK(cudaMalloc(&currentPositions, bytes));
    CUDA_CHECK(cudaMalloc(&nextPositions, bytes));
    CUDA_CHECK(cudaMalloc(&deviceVelocities, bytes));
    CUDA_CHECK(cudaMemcpy(currentPositions, hostPositions.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceVelocities, hostVelocities.data(), bytes, cudaMemcpyHostToDevice));

    const int blocks = (numBodies + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const size_t sharedBytes = static_cast<size_t>(THREADS_PER_BLOCK) * sizeof(double4);

    // Load the module and initialize the device execution path before timing.  A zero-body
    // launch cannot modify simulation state, but prevents one-time CUDA startup work from
    // being reported as simulation time.
    advanceBodies<<<1, THREADS_PER_BLOCK, sharedBytes>>>(currentPositions, nextPositions,
                                                         deviceVelocities, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int step = 0; step < numSteps; ++step) {
        advanceBodies<<<blocks, THREADS_PER_BLOCK, sharedBytes>>>(currentPositions, nextPositions,
                                                                   deviceVelocities, numBodies);
        CUDA_CHECK(cudaGetLastError());
        double4* const completedPositions = currentPositions;
        currentPositions = nextPositions;
        nextPositions = completedPositions;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, stop));

    CUDA_CHECK(cudaMemcpy(hostPositions.data(), currentPositions, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostVelocities.data(), deviceVelocities, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(currentPositions));
    CUDA_CHECK(cudaFree(nextPositions));
    CUDA_CHECK(cudaFree(deviceVelocities));

    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos = Vec3(hostPositions[i].x, hostPositions[i].y, hostPositions[i].z);
        bodies[i].vel = Vec3(hostVelocities[i].x, hostVelocities[i].y, hostVelocities[i].z);
    }
    return static_cast<long>(elapsedMilliseconds);
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
        printf("Number of bodies and simulation steps must be non-negative.\n");
        return 1;
    }

    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Initialize bodies
    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    randomizeBodies(bodies);

    const long durationMilliseconds = runSimulation(bodies, numSteps);
    printf("Simulation time: %ld ms\n", durationMilliseconds);

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
