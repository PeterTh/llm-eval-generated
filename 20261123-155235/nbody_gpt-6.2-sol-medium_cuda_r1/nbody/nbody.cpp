#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
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
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ double squaredDistance(double dx, double dy, double dz) {
    // Use the same contraction order as the optimized CPU force loop.
    return __dadd_rn(__fma_rn(dz, dz, __fma_rn(dx, dx, dy * dy)), SOFTENING);
}

__device__ __forceinline__ void accumulateForce(Vec3 target, double x, double y, double z,
                                               Vec3& force) {
    const double dx = x - target.x;
    const double dy = y - target.y;
    const double dz = z - target.z;
    const double invDist = __drcp_rn(__dsqrt_rn(squaredDistance(dx, dy, dz)));
    const double invDist3 = invDist * invDist * invDist;
    force.x = __fma_rn(dx, invDist3, force.x);
    force.y = __fma_rn(dy, invDist3, force.y);
    force.z = __fma_rn(dz, invDist3, force.z);
}

template <unsigned BlockSize>
__device__ __forceinline__ void syncTile() {
    if constexpr (BlockSize == 32) {
        __syncwarp();
    } else {
        __syncthreads();
    }
}

template <unsigned BlockSize>
__global__ void advanceBodies(const Vec3* __restrict__ positions,
                              Vec3* __restrict__ nextPositions,
                              Vec3* __restrict__ velocities, unsigned count) {
    __shared__ double tileX[BlockSize];
    __shared__ double tileY[BlockSize];
    __shared__ double tileZ[BlockSize];

    const unsigned lane = threadIdx.x;
    const unsigned i = blockIdx.x * BlockSize + lane;
    const Vec3 target = i < count ? positions[i] : Vec3{};
    Vec3 force{};

    // Reuse source positions across the block while keeping each body's sum in source order.
    for (unsigned base = 0; base < count; base += BlockSize) {
        const unsigned j = base + lane;
        if (j < count) {
            const Vec3 source = positions[j];
            tileX[lane] = source.x;
            tileY[lane] = source.y;
            tileZ[lane] = source.z;
        }
        syncTile<BlockSize>();

        const unsigned tileSize = min(BlockSize, count - base);
        #pragma unroll 4
        for (unsigned k = 0; k < tileSize; ++k) {
            accumulateForce(target, tileX[k], tileY[k], tileZ[k], force);
        }
        syncTile<BlockSize>();
    }

    if (i < count) {
        Vec3 velocity = velocities[i];
        // Round the velocity increment before using the updated velocity to integrate position.
        velocity.x += __dmul_rn(DT, force.x);
        velocity.y += __dmul_rn(DT, force.y);
        velocity.z += __dmul_rn(DT, force.z);
        velocities[i] = velocity;
        nextPositions[i] = {__fma_rn(velocity.x, DT, target.x),
                            __fma_rn(velocity.y, DT, target.y),
                            __fma_rn(velocity.z, DT, target.z)};
    }
}

template <unsigned BlockSize>
long runSteps(Vec3*& positions, Vec3*& nextPositions, Vec3* velocities,
              unsigned count, int steps) {
    const unsigned blocks = (count + BlockSize - 1) / BlockSize;
    // Load the kernel before timing so the first launch does not include JIT compilation.
    cudaFuncAttributes attributes;
    checkCuda(cudaFuncGetAttributes(&attributes, advanceBodies<BlockSize>),
              "Loading CUDA simulation kernel");

    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        advanceBodies<BlockSize><<<blocks, BlockSize>>>(positions, nextPositions, velocities, count);
        std::swap(positions, nextPositions);
    }
    checkCuda(cudaGetLastError(), "Launching CUDA simulation");
    checkCuda(cudaDeviceSynchronize(), "Running CUDA simulation");
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
}

__global__ void bodyEnergies(const Vec3* __restrict__ positions,
                             const Vec3* __restrict__ velocities,
                             double* __restrict__ energies, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const Vec3 position = positions[i];
    const Vec3 velocity = velocities[i];
    double energy = 0.5 * (velocity.x * velocity.x + velocity.y * velocity.y +
                           velocity.z * velocity.z);
    for (unsigned j = i + 1; j < count; ++j) {
        const Vec3 source = positions[j];
        const double distance = squaredDistance(source.x - position.x, source.y - position.y,
                                                source.z - position.z);
        energy -= __drcp_rn(__dsqrt_rn(distance));
    }
    energies[i] = energy;
}

double computeTotalEnergy(const Vec3* positions, const Vec3* velocities, unsigned count) {
    std::vector<double> energies(count);
    const size_t bytes = count * sizeof(double);
    double* deviceEnergies = nullptr;
    checkCuda(cudaMalloc(&deviceEnergies, bytes), "Allocating CUDA energy sums");
    constexpr unsigned blockSize = 64;
    bodyEnergies<<<(count + blockSize - 1) / blockSize, blockSize>>>(
        positions, velocities, deviceEnergies, count);
    checkCuda(cudaGetLastError(), "Launching CUDA energy calculation");
    checkCuda(cudaMemcpy(energies.data(), deviceEnergies, bytes, cudaMemcpyDeviceToHost),
              "Downloading CUDA energy sums");
    checkCuda(cudaFree(deviceEnergies), "Freeing CUDA energy sums");
    return kahan_sum(energies);
}

struct SimulationResult {
    long milliseconds = 0;
    double energy = 0.0;
    const char* validationError = nullptr;
};

const char* validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            return "found NaN or Inf value in body state";
        }

        constexpr double maxPos = 1e6;
        constexpr double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            return "body position exceeds reasonable bounds";
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            return "body velocity exceeds reasonable bounds";
        }
    }
    return nullptr;
}

SimulationResult simulate(std::vector<Body>& bodies, int steps, bool printResults, bool validate) {
    const unsigned count = bodies.size();
    if (count == 0 || (steps <= 0 && !validate)) {
        return {};
    }

    std::vector<Vec3> state(2 * size_t{count});
    for (unsigned i = 0; i < count; ++i) {
        state[i] = bodies[i].pos;
        state[count + i] = bodies[i].vel;
    }

    Vec3* deviceState = nullptr;
    const size_t bytes = count * sizeof(Vec3);
    checkCuda(cudaMalloc(&deviceState, 3 * bytes), "Allocating CUDA body state");
    Vec3* positions = deviceState;
    Vec3* velocities = deviceState + count;
    Vec3* nextPositions = velocities + count;
    checkCuda(cudaMemcpy(deviceState, state.data(), 2 * bytes, cudaMemcpyHostToDevice),
              "Uploading body state");

    SimulationResult result;
    if (steps > 0) {
        // Smaller blocks spread moderate workloads across more multiprocessors.
        if (count <= 32768) {
            result.milliseconds = runSteps<32>(positions, nextPositions, velocities, count, steps);
        } else {
            result.milliseconds = runSteps<64>(positions, nextPositions, velocities, count, steps);
        }
    }

    if (steps > 0 && (printResults || validate)) {
        checkCuda(cudaMemcpy(state.data(), positions, bytes, cudaMemcpyDeviceToHost),
                  "Downloading body positions");
        checkCuda(cudaMemcpy(state.data() + count, velocities, bytes, cudaMemcpyDeviceToHost),
                  "Downloading body velocities");
        for (unsigned i = 0; i < count; ++i) {
            bodies[i] = {state[i], state[count + i]};
        }
    }
    if (validate) {
        result.validationError = validateSimulation(bodies);
        if (result.validationError == nullptr) {
            result.energy = computeTotalEnergy(positions, velocities, count);
        }
    }
    checkCuda(cudaFree(deviceState), "Freeing CUDA body state");
    return result;
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
    
    const SimulationResult result = simulate(bodies, numSteps, printResults, validate);
    printf("Simulation time: %ld ms\n", result.milliseconds);
    
    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(bodies.size() * 6);
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
        
        if (result.validationError == nullptr) {
            printf("Final energy: %.6f\n", result.energy);
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation failed: %s\n", result.validationError);
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
