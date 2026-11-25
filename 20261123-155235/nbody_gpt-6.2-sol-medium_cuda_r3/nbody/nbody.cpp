#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
};

void randomizeBodies(std::vector<Vec3>& positions, std::vector<Vec3>& velocities) {
    unsigned int seed = 42;
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        positions[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        positions[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int TILE_SIZE = 128;
constexpr unsigned int ENERGY_THREADS = 128;

__device__ __forceinline__ double inverseDistance(Vec3 delta) {
    // Match the native CPU build's contraction order and round sqrt and reciprocal separately.
    const double xy = __fma_rn(delta.x, delta.x, delta.y * delta.y);
    const double distSqr = __fma_rn(delta.z, delta.z, xy) + SOFTENING;
    return __drcp_rn(__dsqrt_rn(distSqr));
}

__device__ __forceinline__ void accumulateForce(Vec3 position, Vec3 source, Vec3& force) {
    const Vec3 delta{source.x - position.x, source.y - position.y, source.z - position.z};
    const double invDist = inverseDistance(delta);
    const double invDist3 = invDist * invDist * invDist;
    force.x = __fma_rn(delta.x, invDist3, force.x);
    force.y = __fma_rn(delta.y, invDist3, force.y);
    force.z = __fma_rn(delta.z, invDist3, force.z);
}

__global__ void advanceBodies(const Vec3* __restrict__ positions,
                              Vec3* __restrict__ nextPositions,
                              Vec3* __restrict__ velocities, unsigned int count) {
    __shared__ Vec3 sources[TILE_SIZE];
    const unsigned int i = blockIdx.x * WARP_SIZE + threadIdx.x;
    const Vec3 position = i < count ? positions[i] : Vec3{};
    Vec3 force{};

    // Visit sources in serial order for each target, sharing each tile across the block.
    const unsigned int fullTiles = count / TILE_SIZE;
    for (unsigned int tile = 0; tile < fullTiles; ++tile) {
        #pragma unroll
        for (unsigned int j = threadIdx.x; j < TILE_SIZE; j += WARP_SIZE) {
            sources[j] = positions[tile * TILE_SIZE + j];
        }
        __syncwarp();

        #pragma unroll 4
        for (unsigned int j = 0; j < TILE_SIZE; ++j) {
            accumulateForce(position, sources[j], force);
        }
        __syncwarp();
    }

    const unsigned int tail = count % TILE_SIZE;
    if (tail != 0) {
        for (unsigned int j = threadIdx.x; j < tail; j += WARP_SIZE) {
            sources[j] = positions[fullTiles * TILE_SIZE + j];
        }
        __syncwarp();
    }
    if (i >= count) {
        return;
    }
    for (unsigned int j = 0; j < tail; ++j) {
        accumulateForce(position, sources[j], force);
    }

    const Vec3 oldVelocity = velocities[i];
    const Vec3 velocity{oldVelocity.x + DT * force.x,
                        oldVelocity.y + DT * force.y,
                        oldVelocity.z + DT * force.z};
    velocities[i] = velocity;
    // Every thread reads the old positions for this step and writes to the other buffer.
    nextPositions[i] = {__fma_rn(velocity.x, DT, position.x),
                        __fma_rn(velocity.y, DT, position.y),
                        __fma_rn(velocity.z, DT, position.z)};
}

__device__ double warpSum(double value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void computeEnergy(const Vec3* __restrict__ positions,
                              const Vec3* __restrict__ velocities,
                              double* __restrict__ blockEnergies, unsigned int count) {
    __shared__ double warpEnergies[ENERGY_THREADS / WARP_SIZE];
    const unsigned int i = blockIdx.x * ENERGY_THREADS + threadIdx.x;
    double energy = 0.0;
    if (i < count) {
        const Vec3 position = positions[i];
        const Vec3 velocity = velocities[i];
        energy = 0.5 * (__fma_rn(velocity.z, velocity.z,
                                 __fma_rn(velocity.x, velocity.x, velocity.y * velocity.y)));
        // Count each unordered pair once.
        for (unsigned int j = i + 1; j < count; ++j) {
            const Vec3 source = positions[j];
            energy -= inverseDistance({source.x - position.x, source.y - position.y,
                                       source.z - position.z});
        }
    }
    energy = warpSum(energy);
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        warpEnergies[threadIdx.x / WARP_SIZE] = energy;
    }
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
        energy = warpSum(lane < ENERGY_THREADS / WARP_SIZE ? warpEnergies[lane] : 0.0);
        if (lane == 0) {
            blockEnergies[blockIdx.x] = energy;
        }
    }
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities) {
    for (size_t i = 0; i < positions.size(); ++i) {
        const Vec3 pos = positions[i];
        const Vec3 vel = velocities[i];
        // Check for NaN or Inf values
        if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
            !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(pos.x) > maxPos || std::abs(pos.y) > maxPos || std::abs(pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vel.x) > maxVel || std::abs(vel.y) > maxVel || std::abs(vel.z) > maxVel) {
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
    std::vector<Vec3> positions(numBodies);
    std::vector<Vec3> velocities(numBodies);
    randomizeBodies(positions, velocities);

    const auto count = static_cast<unsigned int>(positions.size());
    Vec3* devicePositions = nullptr;
    Vec3* deviceVelocities = nullptr;
    Vec3* nextPositions = nullptr;
    const size_t bytes = positions.size() * sizeof(Vec3);
    if (count != 0 && (numSteps > 0 || validate)) {
        checkCuda(cudaMalloc(&devicePositions, bytes), "Allocate device positions");
        checkCuda(cudaMalloc(&deviceVelocities, bytes), "Allocate device velocities");
        checkCuda(cudaMemcpy(devicePositions, positions.data(), bytes, cudaMemcpyHostToDevice),
                  "Upload positions");
        checkCuda(cudaMemcpy(deviceVelocities, velocities.data(), bytes, cudaMemcpyHostToDevice),
                  "Upload velocities");
    }
    if (count != 0 && numSteps > 0) {
        checkCuda(cudaMalloc(&nextPositions, bytes), "Allocate next positions");
        // Load the kernel during setup, including any driver compilation of embedded PTX.
        cudaFuncAttributes attributes;
        checkCuda(cudaFuncGetAttributes(&attributes, advanceBodies), "Load simulation kernel");
    }
    
    // Time the simulation, including completion of the last GPU step.
    auto start = std::chrono::high_resolution_clock::now();
    if (nextPositions) {
        const unsigned int blocks = (count + WARP_SIZE - 1) / WARP_SIZE;
        for (int step = 0; step < numSteps; ++step) {
            advanceBodies<<<blocks, WARP_SIZE>>>(devicePositions, nextPositions, deviceVelocities, count);
            std::swap(devicePositions, nextPositions);
        }
        checkCuda(cudaGetLastError(), "Launch simulation");
        checkCuda(cudaDeviceSynchronize(), "Run simulation");
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());

    if (nextPositions && (printResults || validate)) {
        checkCuda(cudaMemcpy(positions.data(), devicePositions, bytes, cudaMemcpyDeviceToHost),
                  "Download positions");
        checkCuda(cudaMemcpy(velocities.data(), deviceVelocities, bytes, cudaMemcpyDeviceToHost),
                  "Download velocities");
    }
    
    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(positions.size() * 6);
        for (size_t i = 0; i < positions.size(); ++i) {
            const Vec3 pos = positions[i];
            const Vec3 vel = velocities[i];
            bodyData.insert(bodyData.end(), {pos.x, pos.y, pos.z, vel.x, vel.y, vel.z});
        }
        print_results(bodyData, "Bodies");
    }
    
    // Validation: check that simulation produces finite, reasonable values
    int exitCode = 0;
    if (validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(positions, velocities)) {
            double finalEnergy = 0.0;
            if (count != 0) {
                const unsigned int blocks = (count + ENERGY_THREADS - 1) / ENERGY_THREADS;
                std::vector<double> blockEnergies(blocks);
                double* deviceEnergies;
                checkCuda(cudaMalloc(&deviceEnergies, blocks * sizeof(double)), "Allocate energy sums");
                computeEnergy<<<blocks, ENERGY_THREADS>>>(
                    devicePositions, deviceVelocities, deviceEnergies, count);
                checkCuda(cudaGetLastError(), "Launch energy calculation");
                checkCuda(cudaMemcpy(blockEnergies.data(), deviceEnergies, blocks * sizeof(double),
                                     cudaMemcpyDeviceToHost), "Download energy sums");
                checkCuda(cudaFree(deviceEnergies), "Free energy sums");
                finalEnergy = kahan_sum(blockEnergies);
            }
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }
    
    if (devicePositions) {
        checkCuda(cudaFree(devicePositions), "Free device positions");
        checkCuda(cudaFree(deviceVelocities), "Free device velocities");
    }
    if (nextPositions) {
        checkCuda(cudaFree(nextPositions), "Free next positions");
    }
    return exitCode;
}
