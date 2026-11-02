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

// Each block cooperatively loads a tile of the old positions.  All threads,
// including those past n in the final block, participate in the barriers.
constexpr int BLOCK_SIZE = 128;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void simulateStep(const double4* __restrict__ positions,
                             double4* __restrict__ nextPositions,
                             double4* __restrict__ velocities, int n) {
    __shared__ double4 tile[BLOCK_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const double4 p = active ? positions[i] : make_double4(0, 0, 0, 0);
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        const int count = min(BLOCK_SIZE, n - base);
        if (active) {
            for (int k = 0; k < count; ++k) {
                const double dx = tile[k].x - p.x;
                const double dy = tile[k].y - p.y;
                const double dz = tile[k].z - p.z;
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
        double4 v = velocities[i];
        v.x += DT * fx;
        v.y += DT * fy;
        v.z += DT * fz;
        velocities[i] = v;
        nextPositions[i] = make_double4(p.x + v.x * DT, p.y + v.y * DT,
                                       p.z + v.z * DT, 0.0);
    }
}

// Sum each pair once, as in the original energy calculation.  Blocks produce
// partial sums so validation does not require a quadratic CPU calculation.
__global__ void energyPartials(const double4* positions, const double4* velocities,
                               int n, double* partials) {
    __shared__ double sums[BLOCK_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double energy = 0.0;
    if (i < n) {
        const double4 p = positions[i];
        const double4 v = velocities[i];
        energy = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
        for (int j = i + 1; j < n; ++j) {
            const double4 q = positions[j];
            const double dx = q.x - p.x;
            const double dy = q.y - p.y;
            const double dz = q.z - p.z;
            energy -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    sums[threadIdx.x] = energy;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = sums[0];
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
    
    if (numBodies < 0) {
        std::fprintf(stderr, "Number of bodies must be non-negative\n");
        return 1;
    }

    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    std::vector<double4> positions(numBodies), velocities(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        const Body& b = bodies[i];
        positions[i] = make_double4(b.pos.x, b.pos.y, b.pos.z, 0.0);
        velocities[i] = make_double4(b.vel.x, b.vel.y, b.vel.z, 0.0);
    }

    double4 *devicePositions = nullptr, *nextPositions = nullptr, *deviceVelocities = nullptr;
    const size_t bytes = static_cast<size_t>(numBodies) * sizeof(double4);
    const int blocks = numBodies / BLOCK_SIZE + (numBodies % BLOCK_SIZE != 0);
    // Initialize the CUDA runtime even for an empty simulation.
    checkCuda(cudaFree(nullptr), "CUDA initialization");
    if (numBodies > 0) {
        checkCuda(cudaMalloc(&devicePositions, bytes), "allocating positions");
        checkCuda(cudaMalloc(&nextPositions, bytes), "allocating next positions");
        checkCuda(cudaMalloc(&deviceVelocities, bytes), "allocating velocities");
        checkCuda(cudaMemcpy(devicePositions, positions.data(), bytes, cudaMemcpyHostToDevice),
                  "copying positions to device");
        checkCuda(cudaMemcpy(deviceVelocities, velocities.data(), bytes, cudaMemcpyHostToDevice),
                  "copying velocities to device");
    }

    cudaFuncAttributes attributes;
    checkCuda(cudaFuncGetAttributes(&attributes, simulateStep), "loading simulation kernel");

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps && numBodies > 0; ++step) {
        simulateStep<<<blocks, BLOCK_SIZE>>>(devicePositions, nextPositions, deviceVelocities, numBodies);
        std::swap(devicePositions, nextPositions);
    }
    checkCuda(cudaGetLastError(), "launching simulation");
    checkCuda(cudaDeviceSynchronize(), "running simulation");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Simulation time: %ld ms\n", duration.count());

    double finalEnergy = 0.0;
    if (validate && numBodies > 0) {
        double* devicePartials = nullptr;
        std::vector<double> partials(blocks);
        checkCuda(cudaMalloc(&devicePartials, partials.size() * sizeof(double)), "allocating energy sums");
        energyPartials<<<blocks, BLOCK_SIZE>>>(devicePositions, deviceVelocities, numBodies, devicePartials);
        checkCuda(cudaGetLastError(), "launching energy calculation");
        checkCuda(cudaMemcpy(partials.data(), devicePartials, partials.size() * sizeof(double), cudaMemcpyDeviceToHost),
                  "copying energy sums");
        checkCuda(cudaFree(devicePartials), "freeing energy sums");
        for (double partial : partials) finalEnergy += partial;
    }
    if ((validate || printResults) && numBodies > 0) {
        checkCuda(cudaMemcpy(positions.data(), devicePositions, bytes, cudaMemcpyDeviceToHost),
                  "copying positions to host");
        checkCuda(cudaMemcpy(velocities.data(), deviceVelocities, bytes, cudaMemcpyDeviceToHost),
                  "copying velocities to host");
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = Vec3(positions[i].x, positions[i].y, positions[i].z);
            bodies[i].vel = Vec3(velocities[i].x, velocities[i].y, velocities[i].z);
        }
    }
    checkCuda(cudaFree(devicePositions), "freeing positions");
    checkCuda(cudaFree(nextPositions), "freeing next positions");
    checkCuda(cudaFree(deviceVelocities), "freeing velocities");

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
