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

// Every thread accumulates forces in the same body order as the original.
// The input positions remain unchanged until all bodies have been updated.
constexpr int BLOCK_SIZE = 128;

__global__ void advanceBodies(const double4* __restrict__ positions,
                              double4* __restrict__ nextPositions,
                              double4* __restrict__ velocities, int n) {
    __shared__ double4 tile[BLOCK_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    double4 position = make_double4(0.0, 0.0, 0.0, 0.0);
    if (active) position = positions[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        const int count = min(BLOCK_SIZE, n - base);
        if (active) {
            for (int k = 0; k < count; ++k) {
                const double dx = tile[k].x - position.x;
                const double dy = tile[k].y - position.y;
                const double dz = tile[k].z - position.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = rsqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        double4 velocity = velocities[i];
        velocity.x += DT * fx;
        velocity.y += DT * fy;
        velocity.z += DT * fz;
        position.x += velocity.x * DT;
        position.y += velocity.y * DT;
        position.z += velocity.z * DT;
        velocities[i] = velocity;
        nextPositions[i] = position;
    }
}

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

// Each thread owns one body's kinetic energy and its i < j pair energies.
// Positions are shared across a block so each tile is loaded only once.
__global__ void energyContributions(const double4* __restrict__ positions,
                                    const double4* __restrict__ velocities,
                                    double* __restrict__ contributions, int n) {
    __shared__ double4 tile[BLOCK_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    double4 position = make_double4(0.0, 0.0, 0.0, 0.0);
    double energy = 0.0;
    if (active) {
        position = positions[i];
        const double4 velocity = velocities[i];
        energy = 0.5 * (velocity.x * velocity.x + velocity.y * velocity.y +
                        velocity.z * velocity.z);
    }
    for (int base = blockIdx.x * BLOCK_SIZE; base < n; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        if (active) {
            const int count = min(BLOCK_SIZE, n - base);
            const int first = max(0, i - base + 1);
            for (int k = first; k < count; ++k) {
                const double dx = tile[k].x - position.x;
                const double dy = tile[k].y - position.y;
                const double dz = tile[k].z - position.z;
                const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                energy -= 1.0 / dist;
            }
        }
        __syncthreads();
    }
    if (active) contributions[i] = energy;
}

double computeTotalEnergy(const double4* positions, const double4* velocities, int n) {
    if (n == 0) return 0.0;
    double* deviceContributions = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    checkCuda(cudaMalloc(&deviceContributions, bytes), "allocating energy contributions");
    energyContributions<<<1 + (n - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        positions, velocities, deviceContributions, n);
    checkCuda(cudaGetLastError(), "launching energy kernel");
    std::vector<double> contributions(n);
    checkCuda(cudaMemcpy(contributions.data(), deviceContributions, bytes, cudaMemcpyDeviceToHost),
              "copying energy contributions from device");
    checkCuda(cudaFree(deviceContributions), "freeing energy contributions");
    double energy = 0.0;
    for (double contribution : contributions) energy += contribution;
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
    
    // Keep all simulation state on the device between steps.
    checkCuda(cudaFree(nullptr), "CUDA initialization");
    double4 *positions = nullptr, *nextPositions = nullptr, *velocities = nullptr;
    std::vector<double4> hostPositions(numBodies), hostVelocities(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        hostPositions[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
        hostVelocities[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
    }
    const size_t bytes = static_cast<size_t>(numBodies) * sizeof(double4);
    if (numBodies > 0) {
        checkCuda(cudaMalloc(&positions, bytes), "allocating positions");
        checkCuda(cudaMalloc(&nextPositions, bytes), "allocating next positions");
        checkCuda(cudaMalloc(&velocities, bytes), "allocating velocities");
        checkCuda(cudaMemcpy(positions, hostPositions.data(), bytes, cudaMemcpyHostToDevice),
                  "copying positions to device");
        checkCuda(cudaMemcpy(velocities, hostVelocities.data(), bytes, cudaMemcpyHostToDevice),
                  "copying velocities to device");
    }

    auto start = std::chrono::high_resolution_clock::now();
    if (numBodies > 0) {
        const int blocks = 1 + (numBodies - 1) / BLOCK_SIZE;
        for (int step = 0; step < numSteps; ++step) {
            advanceBodies<<<blocks, BLOCK_SIZE>>>(positions, nextPositions, velocities, numBodies);
            double4* tmp = positions;
            positions = nextPositions;
            nextPositions = tmp;
        }
        checkCuda(cudaGetLastError(), "launching simulation kernel");
        checkCuda(cudaDeviceSynchronize(), "running simulation");
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    if (numBodies > 0 && (printResults || validate)) {
        checkCuda(cudaMemcpy(hostPositions.data(), positions, bytes, cudaMemcpyDeviceToHost),
                  "copying positions from device");
        checkCuda(cudaMemcpy(hostVelocities.data(), velocities, bytes, cudaMemcpyDeviceToHost),
                  "copying velocities from device");
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = Vec3(hostPositions[i].x, hostPositions[i].y, hostPositions[i].z);
            bodies[i].vel = Vec3(hostVelocities[i].x, hostVelocities[i].y, hostVelocities[i].z);
        }
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
    int exitStatus = 0;
    if (validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(positions, velocities, numBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitStatus = 1;
        }
    }

    checkCuda(cudaFree(positions), "freeing positions");
    checkCuda(cudaFree(nextPositions), "freeing next positions");
    checkCuda(cudaFree(velocities), "freeing velocities");
    return exitStatus;
}
