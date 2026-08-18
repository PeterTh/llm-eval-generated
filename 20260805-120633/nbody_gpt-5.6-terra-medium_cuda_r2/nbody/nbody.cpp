#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int THREADS_PER_BLOCK = 256;

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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each block caches a tile of source positions.  Positions are only read by
// this kernel, so the following integration kernel cannot race with a force
// evaluation.
__global__ void computeForcesKernel(const double4* __restrict__ positions,
                                    double4* __restrict__ velocities, int n) {
    extern __shared__ double4 positionTile[];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double4 myPosition{};
    if (i < n) myPosition = positions[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int tile = 0; tile < n; tile += blockDim.x) {
        const int j = tile + threadIdx.x;
        if (j < n) positionTile[threadIdx.x] = positions[j];
        __syncthreads();

        const int tileSize = min(blockDim.x, n - tile);
        if (i < n) {
            #pragma unroll 4
            for (int k = 0; k < tileSize; ++k) {
                const double dx = positionTile[k].x - myPosition.x;
                const double dy = positionTile[k].y - myPosition.y;
                const double dz = positionTile[k].z - myPosition.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        double4 velocity = velocities[i];
        velocity.x += DT * fx;
        velocity.y += DT * fy;
        velocity.z += DT * fz;
        velocities[i] = velocity;
    }
}

__global__ void integrateBodiesKernel(double4* __restrict__ positions,
                                      const double4* __restrict__ velocities, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double4 position = positions[i];
        const double4 velocity = velocities[i];
        position.x += velocity.x * DT;
        position.y += velocity.y * DT;
        position.z += velocity.z * DT;
        positions[i] = position;
    }
}

void runSimulationCuda(std::vector<Body>& bodies, int numSteps, float& elapsedMs) {
    const int n = static_cast<int>(bodies.size());
    if (n == 0 || numSteps <= 0) {
        elapsedMs = 0.0f;
        return;
    }

    std::vector<double4> positions(n), velocities(n);
    for (int i = 0; i < n; ++i) {
        positions[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
        velocities[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
    }

    double4 *devicePositions = nullptr, *deviceVelocities = nullptr;
    checkCuda(cudaMalloc(&devicePositions, static_cast<size_t>(n) * sizeof(double4)), "allocating positions");
    checkCuda(cudaMalloc(&deviceVelocities, static_cast<size_t>(n) * sizeof(double4)), "allocating velocities");
    checkCuda(cudaMemcpy(devicePositions, positions.data(), static_cast<size_t>(n) * sizeof(double4), cudaMemcpyHostToDevice), "copying positions to device");
    checkCuda(cudaMemcpy(deviceVelocities, velocities.data(), static_cast<size_t>(n) * sizeof(double4), cudaMemcpyHostToDevice), "copying velocities to device");

    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&end), "creating end event");
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const size_t sharedBytes = THREADS_PER_BLOCK * sizeof(double4);

    checkCuda(cudaEventRecord(start), "recording start event");
    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<blocks, THREADS_PER_BLOCK, sharedBytes>>>(devicePositions, deviceVelocities, n);
        checkCuda(cudaGetLastError(), "launching force kernel");
        integrateBodiesKernel<<<blocks, THREADS_PER_BLOCK>>>(devicePositions, deviceVelocities, n);
        checkCuda(cudaGetLastError(), "launching integration kernel");
    }
    checkCuda(cudaEventRecord(end), "recording end event");
    checkCuda(cudaEventSynchronize(end), "synchronizing simulation");
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, end), "measuring simulation time");

    checkCuda(cudaMemcpy(positions.data(), devicePositions, static_cast<size_t>(n) * sizeof(double4), cudaMemcpyDeviceToHost), "copying positions from device");
    checkCuda(cudaMemcpy(velocities.data(), deviceVelocities, static_cast<size_t>(n) * sizeof(double4), cudaMemcpyDeviceToHost), "copying velocities from device");
    for (int i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(positions[i].x, positions[i].y, positions[i].z);
        bodies[i].vel = Vec3(velocities[i].x, velocities[i].y, velocities[i].z);
    }

    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(end), "destroying end event");
    checkCuda(cudaFree(devicePositions), "freeing positions");
    checkCuda(cudaFree(deviceVelocities), "freeing velocities");
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
    
    // Run simulation
    float elapsedMs = 0.0f;
    runSimulationCuda(bodies, numSteps, elapsedMs);
    printf("Simulation time: %.3f ms\n", elapsedMs);
    
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
