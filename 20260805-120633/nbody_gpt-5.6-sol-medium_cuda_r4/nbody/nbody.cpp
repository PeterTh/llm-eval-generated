#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

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

[[noreturn]] void cudaFailure(cudaError_t error, const char* operation,
                              const char* file, int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while %s: %s\n", file, line,
                 operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(operation)                                                   \
    do {                                                                        \
        const cudaError_t error = (operation);                                  \
        if (error != cudaSuccess) {                                             \
            cudaFailure(error, #operation, __FILE__, __LINE__);                 \
        }                                                                       \
    } while (false)

// The state is stored as six contiguous structure-of-arrays planes.  Two states
// are ping-ponged so every block reads a consistent snapshot while simultaneously
// producing the next simulation step.
template <int BlockSize>
__global__ __launch_bounds__(BlockSize)
void advanceBodies(const double* __restrict__ input,
                   double* __restrict__ output, int n) {
    extern __shared__ double tile[];
    double* const tileX = tile;
    double* const tileY = tile + BlockSize;
    double* const tileZ = tile + 2 * BlockSize;

    const double* const posX = input;
    const double* const posY = input + n;
    const double* const posZ = input + 2 * n;
    const double* const velX = input + 3 * n;
    const double* const velY = input + 4 * n;
    const double* const velZ = input + 5 * n;

    const int i = static_cast<int>(blockIdx.x) * BlockSize + threadIdx.x;
    const bool active = i < n;
    const double xi = active ? posX[i] : 0.0;
    const double yi = active ? posY[i] : 0.0;
    const double zi = active ? posZ[i] : 0.0;
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    // Each position is fetched from global memory once per block and reused by
    // all 256 threads.  The inner loop retains the original j-order summation.
    for (int base = 0; base < n; base += BlockSize) {
        const int j = base + threadIdx.x;
        if (j < n) {
            tileX[threadIdx.x] = posX[j];
            tileY[threadIdx.x] = posY[j];
            tileZ[threadIdx.x] = posZ[j];
        }
        __syncthreads();

        if (active) {
            const int count = min(BlockSize, n - base);
#pragma unroll 8
            for (int k = 0; k < count; ++k) {
                const double dx = tileX[k] - xi;
                const double dy = tileY[k] - yi;
                const double dz = tileZ[k] - zi;
                const double distanceSquared =
                    dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inverseDistance = rsqrt(distanceSquared);
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
        const double newVelX = velX[i] + DT * forceX;
        const double newVelY = velY[i] + DT * forceY;
        const double newVelZ = velZ[i] + DT * forceZ;
        output[i] = xi + DT * newVelX;
        output[n + i] = yi + DT * newVelY;
        output[2 * n + i] = zi + DT * newVelZ;
        output[3 * n + i] = newVelX;
        output[4 * n + i] = newVelY;
        output[5 * n + i] = newVelZ;
    }
}

void copyBodiesToPlanes(const std::vector<Body>& bodies,
                        std::vector<double>& planes) {
    const size_t n = bodies.size();
    for (size_t i = 0; i < n; ++i) {
        planes[i] = bodies[i].pos.x;
        planes[n + i] = bodies[i].pos.y;
        planes[2 * n + i] = bodies[i].pos.z;
        planes[3 * n + i] = bodies[i].vel.x;
        planes[4 * n + i] = bodies[i].vel.y;
        planes[5 * n + i] = bodies[i].vel.z;
    }
}

void copyPlanesToBodies(const std::vector<double>& planes,
                        std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(planes[i], planes[n + i], planes[2 * n + i]);
        bodies[i].vel =
            Vec3(planes[3 * n + i], planes[4 * n + i], planes[5 * n + i]);
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
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    const size_t stateValues = static_cast<size_t>(numBodies) * 6;
    const size_t stateBytes = stateValues * sizeof(double);
    std::vector<double> hostState(stateValues);
    copyBodiesToPlanes(bodies, hostState);

    double* deviceCurrent = nullptr;
    double* deviceNext = nullptr;
    if (stateBytes != 0) {
        CUDA_CHECK(cudaMalloc(&deviceCurrent, stateBytes));
        CUDA_CHECK(cudaMalloc(&deviceNext, stateBytes));
        CUDA_CHECK(cudaMemcpy(deviceCurrent, hostState.data(), stateBytes,
                              cudaMemcpyHostToDevice));
    }

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    if (numBodies > 0) {
        const int gridSize = (numBodies + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        const size_t sharedBytes = 3 * CUDA_BLOCK_SIZE * sizeof(double);
        for (int step = 0; step < numSteps; ++step) {
            advanceBodies<CUDA_BLOCK_SIZE><<<gridSize, CUDA_BLOCK_SIZE,
                                              sharedBytes>>>(
                deviceCurrent, deviceNext, numBodies);
            CUDA_CHECK(cudaGetLastError());
            std::swap(deviceCurrent, deviceNext);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    if (stateBytes != 0) {
        CUDA_CHECK(cudaMemcpy(hostState.data(), deviceCurrent, stateBytes,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(deviceCurrent));
        CUDA_CHECK(cudaFree(deviceNext));
        copyPlanesToBodies(hostState, bodies);
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
