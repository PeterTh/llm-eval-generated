#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

using Vec3 = double3;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

void randomizeBodies(Vec3* positions, Vec3* velocities, unsigned int n,
                     unsigned int seed = 42) {
    auto randomVector = [&seed]() {
        return Vec3{
            2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0,
            2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0,
            2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0
        };
    };
    for (unsigned int i = 0; i < n; ++i) {
        positions[i] = randomVector();
        velocities[i] = randomVector();
    }
}

__device__ __forceinline__ double inverseDistance(double dx, double dy, double dz) {
    const double xy = __fma_rn(dx, dx, dy * dy);
    const double distSqr = __fma_rn(dz, dz, xy) + SOFTENING;
    // Round the square root and reciprocal separately, as in the original calculation.
    return __drcp_rn(__dsqrt_rn(distSqr));
}

__device__ __forceinline__ void accumulateForce(Vec3 target, Vec3 source, Vec3& force) {
    const double dx = source.x - target.x;
    const double dy = source.y - target.y;
    const double dz = source.z - target.z;
    const double invDist = inverseDistance(dx, dy, dz);
    const double invDist3 = invDist * invDist * invDist;
    force.x = __fma_rn(dx, invDist3, force.x);
    force.y = __fma_rn(dy, invDist3, force.y);
    force.z = __fma_rn(dz, invDist3, force.z);
}

constexpr unsigned int WarpSize = 32;

__global__ void advanceBodies(const Vec3* __restrict__ positions,
                              Vec3* __restrict__ nextPositions,
                              Vec3* __restrict__ velocities, unsigned int n) {
    __shared__ Vec3 tile[WarpSize];
    const unsigned int lane = threadIdx.x;
    const unsigned int i = blockIdx.x * WarpSize + lane;
    const Vec3 target = i < n ? positions[i] : Vec3{};
    Vec3 force{};

    // Every thread helps load a tile, including threads past the last body.
    const unsigned int fullTiles = n / WarpSize;
    for (unsigned int t = 0; t < fullTiles; ++t) {
        tile[lane] = positions[t * WarpSize + lane];
        __syncwarp();
        #pragma unroll 8
        for (unsigned int j = 0; j < WarpSize; ++j) {
            accumulateForce(target, tile[j], force);
        }
        __syncwarp();
    }

    const unsigned int tail = n % WarpSize;
    if (tail != 0) {
        if (lane < tail) {
            tile[lane] = positions[fullTiles * WarpSize + lane];
        }
        __syncwarp();
        for (unsigned int j = 0; j < tail; ++j) {
            accumulateForce(target, tile[j], force);
        }
    }

    if (i < n) {
        Vec3 velocity = velocities[i];
        velocity.x += DT * force.x;
        velocity.y += DT * force.y;
        velocity.z += DT * force.z;
        velocities[i] = velocity;

        // All force reads go to the old position buffer for this step.
        nextPositions[i] = Vec3{
            __fma_rn(velocity.x, DT, target.x),
            __fma_rn(velocity.y, DT, target.y),
            __fma_rn(velocity.z, DT, target.z)
        };
    }
}

void simulate(Vec3*& positions, Vec3*& nextPositions, Vec3* velocities,
              unsigned int n, int steps) {
    const unsigned int blocks = (n + WarpSize - 1) / WarpSize;
    for (int step = 0; step < steps; ++step) {
        advanceBodies<<<blocks, WarpSize>>>(positions, nextPositions, velocities, n);
        std::swap(positions, nextPositions);
    }
    checkCuda(cudaGetLastError(), "Launching simulation");
    checkCuda(cudaDeviceSynchronize(), "Running simulation");
}

constexpr unsigned int EnergyBlockSize = 128;

__global__ void energyByBlock(const Vec3* __restrict__ positions,
                              const Vec3* __restrict__ velocities,
                              double* __restrict__ partialEnergy, unsigned int n) {
    __shared__ double warpEnergy[EnergyBlockSize / WarpSize];
    const unsigned int i = blockIdx.x * EnergyBlockSize + threadIdx.x;
    double energy = 0.0;
    if (i < n) {
        const Vec3 pos = positions[i];
        const Vec3 vel = velocities[i];
        energy = 0.5 * (__fma_rn(vel.z, vel.z, __fma_rn(vel.x, vel.x, vel.y * vel.y)));
        for (unsigned int j = i + 1; j < n; ++j) {
            const Vec3 other = positions[j];
            energy -= inverseDistance(other.x - pos.x, other.y - pos.y, other.z - pos.z);
        }
    }

    for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
        energy += __shfl_down_sync(0xffffffff, energy, offset);
    }
    const unsigned int lane = threadIdx.x % WarpSize;
    if (lane == 0) {
        warpEnergy[threadIdx.x / WarpSize] = energy;
    }
    __syncthreads();

    if (threadIdx.x < WarpSize) {
        energy = lane < EnergyBlockSize / WarpSize ? warpEnergy[lane] : 0.0;
        for (int offset = EnergyBlockSize / WarpSize / 2; offset > 0; offset /= 2) {
            energy += __shfl_down_sync(0xffffffff, energy, offset);
        }
        if (lane == 0) {
            partialEnergy[blockIdx.x] = energy;
        }
    }
}

double computeTotalEnergy(const Vec3* positions, const Vec3* velocities,
                          Vec3* scratch, unsigned int n) {
    if (n == 0) {
        return 0.0;
    }
    const unsigned int blocks = (n + EnergyBlockSize - 1) / EnergyBlockSize;
    // The inactive position buffer can hold the block sums after simulation.
    auto* deviceEnergy = reinterpret_cast<double*>(scratch);
    energyByBlock<<<blocks, EnergyBlockSize>>>(positions, velocities, deviceEnergy, n);
    checkCuda(cudaGetLastError(), "Launching energy calculation");
    std::vector<double> energy(blocks);
    checkCuda(cudaMemcpy(energy.data(), deviceEnergy, blocks * sizeof(double),
                         cudaMemcpyDeviceToHost), "Downloading energy");
    return kahan_sum(energy);
}

// Validate that simulation produces finite, reasonable values.
bool validateSimulation(const Vec3* positions, const Vec3* velocities, unsigned int n) {
    for (unsigned int i = 0; i < n; ++i) {
        const Vec3 pos = positions[i];
        const Vec3 vel = velocities[i];
        if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
            !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

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
    
    if (numBodies < 0) {
        fprintf(stderr, "Number of bodies must be nonnegative\n");
        return 1;
    }
    const auto n = static_cast<unsigned int>(numBodies);
    std::vector<Vec3> bodies(2 * static_cast<size_t>(n));
    Vec3* hostPositions = bodies.data();
    Vec3* hostVelocities = n == 0 ? hostPositions : hostPositions + n;
    randomizeBodies(hostPositions, hostVelocities, n);

    Vec3* positions = nullptr;
    Vec3* velocities = nullptr;
    Vec3* nextPositions = nullptr;
    std::unique_ptr<Vec3, decltype(&cudaFree)> deviceBodies(nullptr, cudaFree);
    const size_t bytes = n * sizeof(Vec3);
    if (n != 0 && (numSteps > 0 || validate)) {
        checkCuda(cudaMalloc(&positions, 3 * bytes), "Allocating device bodies");
        deviceBodies.reset(positions);
        velocities = positions + n;
        nextPositions = velocities + n;
        checkCuda(cudaMemcpy(positions, hostPositions, 2 * bytes, cudaMemcpyHostToDevice),
                  "Uploading initial bodies");
    }

    const auto start = std::chrono::high_resolution_clock::now();
    if (n != 0 && numSteps > 0) {
        simulate(positions, nextPositions, velocities, n, numSteps);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Simulation time: %ld ms\n", duration.count());

    if (n != 0 && numSteps > 0 && (printResults || validate)) {
        checkCuda(cudaMemcpy(hostPositions, positions, bytes, cudaMemcpyDeviceToHost),
                  "Downloading final positions");
        checkCuda(cudaMemcpy(hostVelocities, velocities, bytes, cudaMemcpyDeviceToHost),
                  "Downloading final velocities");
    }

    if (printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(6 * static_cast<size_t>(n));
        for (unsigned int i = 0; i < n; ++i) {
            const Vec3 pos = hostPositions[i];
            const Vec3 vel = hostVelocities[i];
            bodyData.insert(bodyData.end(), {pos.x, pos.y, pos.z, vel.x, vel.y, vel.z});
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(hostPositions, hostVelocities, n)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(positions, velocities, nextPositions, n);
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
