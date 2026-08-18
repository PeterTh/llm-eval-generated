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
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

constexpr int CUDA_BLOCK_SIZE = 256;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

// One thread advances one body.  Every block stages a contiguous tile of
// source positions in shared memory, which turns each global position load
// into 256 force interactions.  Positions are double-buffered: all reads are
// from the state at the start of the step, matching the original force then
// integrate ordering without an inter-block read/write race.
__global__ void advanceBodiesKernel(const double* __restrict__ posX,
                                    const double* __restrict__ posY,
                                    const double* __restrict__ posZ,
                                    double* __restrict__ nextPosX,
                                    double* __restrict__ nextPosY,
                                    double* __restrict__ nextPosZ,
                                    double* __restrict__ velX,
                                    double* __restrict__ velY,
                                    double* __restrict__ velZ,
                                    int numBodies) {
    __shared__ double tileX[CUDA_BLOCK_SIZE];
    __shared__ double tileY[CUDA_BLOCK_SIZE];
    __shared__ double tileZ[CUDA_BLOCK_SIZE];

    const int i = static_cast<int>(blockIdx.x) * CUDA_BLOCK_SIZE + threadIdx.x;
    const bool active = i < numBodies;

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    if (active) {
        x = posX[i];
        y = posY[i];
        z = posZ[i];
    }

    for (int base = 0; base < numBodies; base += CUDA_BLOCK_SIZE) {
        const int source = base + threadIdx.x;
        if (source < numBodies) {
            tileX[threadIdx.x] = posX[source];
            tileY[threadIdx.x] = posY[source];
            tileZ[threadIdx.x] = posZ[source];
        }
        __syncthreads();

        const int tileSize = min(CUDA_BLOCK_SIZE, numBodies - base);
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < tileSize; ++j) {
                const double dx = tileX[j] - x;
                const double dy = tileY[j] - y;
                const double dz = tileZ[j] - z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                forceX += dx * invDist3;
                forceY += dy * invDist3;
                forceZ += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        const double updatedVelX = velX[i] + DT * forceX;
        const double updatedVelY = velY[i] + DT * forceY;
        const double updatedVelZ = velZ[i] + DT * forceZ;

        velX[i] = updatedVelX;
        velY[i] = updatedVelY;
        velZ[i] = updatedVelZ;
        nextPosX[i] = x + updatedVelX * DT;
        nextPosY[i] = y + updatedVelY * DT;
        nextPosZ[i] = z + updatedVelZ * DT;
    }
}

// For workloads with too few target tiles to fill the GPU, each target tile is
// split across contiguous ranges of source bodies.  The partial sums are kept
// separate and are reduced in source-range order by reduceAndIntegrateKernel.
// This adds parallelism without ever allowing a timestep to observe a newly
// written position.
__global__ void computeForcePartialsKernel(const double* __restrict__ posX,
                                           const double* __restrict__ posY,
                                           const double* __restrict__ posZ,
                                           double* __restrict__ partialForceX,
                                           double* __restrict__ partialForceY,
                                           double* __restrict__ partialForceZ,
                                           int numBodies,
                                           int sourcePartitions) {
    __shared__ double tileX[CUDA_BLOCK_SIZE];
    __shared__ double tileY[CUDA_BLOCK_SIZE];
    __shared__ double tileZ[CUDA_BLOCK_SIZE];

    const int i = static_cast<int>(blockIdx.x) * CUDA_BLOCK_SIZE + threadIdx.x;
    const int partition = blockIdx.y;
    const int sourceBegin = static_cast<int>(
        (static_cast<long long>(partition) * numBodies) / sourcePartitions);
    const int sourceEnd = static_cast<int>(
        (static_cast<long long>(partition + 1) * numBodies) / sourcePartitions);
    const bool active = i < numBodies;

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;
    if (active) {
        x = posX[i];
        y = posY[i];
        z = posZ[i];
    }

    for (int base = sourceBegin; base < sourceEnd; base += CUDA_BLOCK_SIZE) {
        const int source = base + threadIdx.x;
        if (source < sourceEnd) {
            tileX[threadIdx.x] = posX[source];
            tileY[threadIdx.x] = posY[source];
            tileZ[threadIdx.x] = posZ[source];
        }
        __syncthreads();

        const int tileSize = min(CUDA_BLOCK_SIZE, sourceEnd - base);
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < tileSize; ++j) {
                const double dx = tileX[j] - x;
                const double dy = tileY[j] - y;
                const double dz = tileZ[j] - z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                forceX += dx * invDist3;
                forceY += dy * invDist3;
                forceZ += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        const size_t partialIndex = static_cast<size_t>(partition) * numBodies + i;
        partialForceX[partialIndex] = forceX;
        partialForceY[partialIndex] = forceY;
        partialForceZ[partialIndex] = forceZ;
    }
}

__global__ void reduceAndIntegrateKernel(const double* __restrict__ posX,
                                         const double* __restrict__ posY,
                                         const double* __restrict__ posZ,
                                         double* __restrict__ nextPosX,
                                         double* __restrict__ nextPosY,
                                         double* __restrict__ nextPosZ,
                                         double* __restrict__ velX,
                                         double* __restrict__ velY,
                                         double* __restrict__ velZ,
                                         const double* __restrict__ partialForceX,
                                         const double* __restrict__ partialForceY,
                                         const double* __restrict__ partialForceZ,
                                         int numBodies,
                                         int sourcePartitions) {
    const int i = static_cast<int>(blockIdx.x) * CUDA_BLOCK_SIZE + threadIdx.x;
    if (i >= numBodies) {
        return;
    }

    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;
    for (int partition = 0; partition < sourcePartitions; ++partition) {
        const size_t partialIndex = static_cast<size_t>(partition) * numBodies + i;
        forceX += partialForceX[partialIndex];
        forceY += partialForceY[partialIndex];
        forceZ += partialForceZ[partialIndex];
    }

    const double updatedVelX = velX[i] + DT * forceX;
    const double updatedVelY = velY[i] + DT * forceY;
    const double updatedVelZ = velZ[i] + DT * forceZ;
    velX[i] = updatedVelX;
    velY[i] = updatedVelY;
    velZ[i] = updatedVelZ;
    nextPosX[i] = posX[i] + updatedVelX * DT;
    nextPosY[i] = posY[i] + updatedVelY * DT;
    nextPosZ[i] = posZ[i] + updatedVelZ * DT;
}

struct DeviceBodies {
    DeviceBodies(size_t count, int sourcePartitions) : count(count), sourcePartitions(sourcePartitions) {
        if (count == 0) {
            return;
        }

        const size_t bytes = count * sizeof(double);
        CUDA_CHECK(cudaMalloc(&posX, bytes));
        CUDA_CHECK(cudaMalloc(&posY, bytes));
        CUDA_CHECK(cudaMalloc(&posZ, bytes));
        CUDA_CHECK(cudaMalloc(&nextPosX, bytes));
        CUDA_CHECK(cudaMalloc(&nextPosY, bytes));
        CUDA_CHECK(cudaMalloc(&nextPosZ, bytes));
        CUDA_CHECK(cudaMalloc(&velX, bytes));
        CUDA_CHECK(cudaMalloc(&velY, bytes));
        CUDA_CHECK(cudaMalloc(&velZ, bytes));

        if (sourcePartitions > 1) {
            const size_t partialBytes = count * static_cast<size_t>(sourcePartitions) * sizeof(double);
            CUDA_CHECK(cudaMalloc(&partialForceX, partialBytes));
            CUDA_CHECK(cudaMalloc(&partialForceY, partialBytes));
            CUDA_CHECK(cudaMalloc(&partialForceZ, partialBytes));
        }
    }

    DeviceBodies(const DeviceBodies&) = delete;
    DeviceBodies& operator=(const DeviceBodies&) = delete;

    ~DeviceBodies() {
        cudaFree(posX);
        cudaFree(posY);
        cudaFree(posZ);
        cudaFree(nextPosX);
        cudaFree(nextPosY);
        cudaFree(nextPosZ);
        cudaFree(velX);
        cudaFree(velY);
        cudaFree(velZ);
        cudaFree(partialForceX);
        cudaFree(partialForceY);
        cudaFree(partialForceZ);
    }

    size_t count;
    int sourcePartitions;
    double* posX = nullptr;
    double* posY = nullptr;
    double* posZ = nullptr;
    double* nextPosX = nullptr;
    double* nextPosY = nullptr;
    double* nextPosZ = nullptr;
    double* velX = nullptr;
    double* velY = nullptr;
    double* velZ = nullptr;
    double* partialForceX = nullptr;
    double* partialForceY = nullptr;
    double* partialForceZ = nullptr;
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

    // Split the host array-of-structures into the structure-of-arrays layout
    // used by the CUDA kernel.  This makes each position component a coalesced
    // load while retaining the original host representation and output order.
    std::vector<double> posX(bodies.size());
    std::vector<double> posY(bodies.size());
    std::vector<double> posZ(bodies.size());
    std::vector<double> velX(bodies.size());
    std::vector<double> velY(bodies.size());
    std::vector<double> velZ(bodies.size());
    for (size_t i = 0; i < bodies.size(); ++i) {
        posX[i] = bodies[i].pos.x;
        posY[i] = bodies[i].pos.y;
        posZ[i] = bodies[i].pos.z;
        velX[i] = bodies[i].vel.x;
        velY[i] = bodies[i].vel.y;
        velZ[i] = bodies[i].vel.z;
    }

    // A single long-running block per target tile underuses wide GPUs for
    // modest N.  Select enough source partitions to keep four blocks resident
    // per SM, then deterministically reduce those contiguous source ranges.
    int sourcePartitions = 1;
    if (!bodies.empty() && numSteps > 0) {
        int device = 0;
        int multiprocessorCount = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessorCount,
                                          cudaDevAttrMultiProcessorCount, device));
        const int targetBlocks = (numBodies + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        const int desiredBlocks = multiprocessorCount * 4;
        sourcePartitions = (desiredBlocks + targetBlocks - 1) / targetBlocks;
        sourcePartitions = min(sourcePartitions, numBodies);
    }

    DeviceBodies deviceBodies(bodies.size(), sourcePartitions);
    const size_t bytes = bodies.size() * sizeof(double);
    if (!bodies.empty()) {
        CUDA_CHECK(cudaMemcpy(deviceBodies.posX, posX.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceBodies.posY, posY.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceBodies.posZ, posZ.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceBodies.velX, velX.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceBodies.velY, velY.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceBodies.velZ, velZ.data(), bytes, cudaMemcpyHostToDevice));
    }

    // Run the entire timestep sequence on the GPU.  Transfers and result
    // formatting remain outside the timed region, as initialization and
    // reporting did in the original benchmark.
    auto start = std::chrono::high_resolution_clock::now();

    if (!bodies.empty()) {
        const int gridSize = (numBodies + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        for (int step = 0; step < numSteps; ++step) {
            if (sourcePartitions == 1) {
                advanceBodiesKernel<<<gridSize, CUDA_BLOCK_SIZE>>>(
                    deviceBodies.posX, deviceBodies.posY, deviceBodies.posZ,
                    deviceBodies.nextPosX, deviceBodies.nextPosY, deviceBodies.nextPosZ,
                    deviceBodies.velX, deviceBodies.velY, deviceBodies.velZ, numBodies);
            } else {
                computeForcePartialsKernel<<<dim3(gridSize, sourcePartitions), CUDA_BLOCK_SIZE>>>(
                    deviceBodies.posX, deviceBodies.posY, deviceBodies.posZ,
                    deviceBodies.partialForceX, deviceBodies.partialForceY, deviceBodies.partialForceZ,
                    numBodies, sourcePartitions);
                CUDA_CHECK(cudaGetLastError());

                reduceAndIntegrateKernel<<<gridSize, CUDA_BLOCK_SIZE>>>(
                    deviceBodies.posX, deviceBodies.posY, deviceBodies.posZ,
                    deviceBodies.nextPosX, deviceBodies.nextPosY, deviceBodies.nextPosZ,
                    deviceBodies.velX, deviceBodies.velY, deviceBodies.velZ,
                    deviceBodies.partialForceX, deviceBodies.partialForceY, deviceBodies.partialForceZ,
                    numBodies, sourcePartitions);
            }
            CUDA_CHECK(cudaGetLastError());

            std::swap(deviceBodies.posX, deviceBodies.nextPosX);
            std::swap(deviceBodies.posY, deviceBodies.nextPosY);
            std::swap(deviceBodies.posZ, deviceBodies.nextPosZ);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    if (!bodies.empty()) {
        CUDA_CHECK(cudaMemcpy(posX.data(), deviceBodies.posX, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(posY.data(), deviceBodies.posY, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(posZ.data(), deviceBodies.posZ, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velX.data(), deviceBodies.velX, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velY.data(), deviceBodies.velY, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velZ.data(), deviceBodies.velZ, bytes, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < bodies.size(); ++i) {
            bodies[i].pos.x = posX[i];
            bodies[i].pos.y = posY[i];
            bodies[i].pos.z = posZ[i];
            bodies[i].vel.x = velX[i];
            bodies[i].vel.y = velY[i];
            bodies[i].vel.z = velZ[i];
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
