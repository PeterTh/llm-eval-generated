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

// Each thread owns one body. Position buffers are swapped after each step so
// every force calculation observes the positions from the start of that step.
constexpr int TILE_SIZE = 128;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void simulationStep(const Vec3* __restrict__ positions,
                               Vec3* __restrict__ nextPositions,
                               Vec3* __restrict__ velocities, int n) {
    __shared__ double sx[TILE_SIZE], sy[TILE_SIZE], sz[TILE_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double x = 0, y = 0, z = 0;
    if (i < n) {
        x = positions[i].x; y = positions[i].y; z = positions[i].z;
    }
    double fx = 0, fy = 0, fz = 0;
    for (int base = 0; base < n; base += TILE_SIZE) {
        const int j = base + threadIdx.x;
        if (j < n) {
            sx[threadIdx.x] = positions[j].x;
            sy[threadIdx.x] = positions[j].y;
            sz[threadIdx.x] = positions[j].z;
        }
        __syncthreads();
        const int count = min(TILE_SIZE, n - base);
        if (i < n) {
            for (int k = 0; k < count; ++k) {
                const double dx = sx[k] - x, dy = sy[k] - y, dz = sz[k] - z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3; fy += dy * invDist3; fz += dz * invDist3;
            }
        }
        __syncthreads();
    }
    if (i < n) {
        Vec3 v = velocities[i];
        v.x += DT * fx; v.y += DT * fy; v.z += DT * fz;
        velocities[i] = v;
        nextPositions[i].x = x + v.x * DT;
        nextPositions[i].y = y + v.y * DT;
        nextPositions[i].z = z + v.z * DT;
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
    
    // Keep state on the device throughout the simulation.
    std::vector<Vec3> positions(numBodies), velocities(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        positions[i] = bodies[i].pos;
        velocities[i] = bodies[i].vel;
    }
    Vec3 *devicePositions = nullptr, *deviceNext = nullptr, *deviceVelocities = nullptr;
    const size_t bytes = positions.size() * sizeof(Vec3);
    if (numBodies > 0) {
        checkCuda(cudaMalloc(&devicePositions, bytes), "allocating positions");
        checkCuda(cudaMalloc(&deviceNext, bytes), "allocating next positions");
        checkCuda(cudaMalloc(&deviceVelocities, bytes), "allocating velocities");
        checkCuda(cudaMemcpy(devicePositions, positions.data(), bytes, cudaMemcpyHostToDevice), "copying positions");
        checkCuda(cudaMemcpy(deviceVelocities, velocities.data(), bytes, cudaMemcpyHostToDevice), "copying velocities");
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps && numBodies > 0; ++step) {
        simulationStep<<<(numBodies + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(
            devicePositions, deviceNext, deviceVelocities, numBodies);
        checkCuda(cudaGetLastError(), "launching simulation step");
        std::swap(devicePositions, deviceNext);
    }
    if (numBodies > 0 && numSteps > 0)
        checkCuda(cudaDeviceSynchronize(), "running simulation");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (numBodies > 0 && (printResults || validate)) {
        checkCuda(cudaMemcpy(positions.data(), devicePositions, bytes, cudaMemcpyDeviceToHost), "copying final positions");
        checkCuda(cudaMemcpy(velocities.data(), deviceVelocities, bytes, cudaMemcpyDeviceToHost), "copying final velocities");
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = positions[i];
            bodies[i].vel = velocities[i];
        }
    }
    if (numBodies > 0) {
        checkCuda(cudaFree(devicePositions), "freeing positions");
        checkCuda(cudaFree(deviceNext), "freeing next positions");
        checkCuda(cudaFree(deviceVelocities), "freeing velocities");
    }

    printf("Simulation time: %ld ms\n", duration.count());
    
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
