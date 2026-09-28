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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err_));                          \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

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

// Structure-of-arrays layout on the device for coalesced access.
// Forces are accumulated over all j in ascending order (tile by tile), matching
// the summation order of the original sequential loop. The block size is a
// template parameter so a smaller block can be chosen for small n, spreading
// the work over more SMs (each body's j-loop must stay serial to preserve the
// reference summation order, so grid parallelism is capped at n threads).
template <int BS>
__global__ void computeForcesKernel(const double* __restrict__ posX,
                                    const double* __restrict__ posY,
                                    const double* __restrict__ posZ,
                                    double* __restrict__ velX,
                                    double* __restrict__ velY,
                                    double* __restrict__ velZ,
                                    int n) {
    __shared__ double shX[BS];
    __shared__ double shY[BS];
    __shared__ double shZ[BS];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (i < n) {
        xi = posX[i];
        yi = posY[i];
        zi = posZ[i];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < n; tile += BS) {
        const int j = tile + threadIdx.x;
        if (j < n) {
            shX[threadIdx.x] = posX[j];
            shY[threadIdx.x] = posY[j];
            shZ[threadIdx.x] = posZ[j];
        }
        __syncthreads();

        const int tileCount = min(BS, n - tile);
        if (i < n) {
#pragma unroll 8
            // Explicit fma() calls replicate the exact contraction pattern gcc
            // emits for the original scalar loop, so results are bitwise
            // identical to the CPU reference (kernel is built with -fmad=false).
            for (int k = 0; k < tileCount; ++k) {
                const double dx = shX[k] - xi;
                const double dy = shY[k] - yi;
                const double dz = shZ[k] - zi;
                const double distSqr = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx = fma(dx, invDist3, Fx);
                Fy = fma(dy, invDist3, Fy);
                Fz = fma(dz, invDist3, Fz);
            }
        }
        __syncthreads();
    }

    if (i < n) {
        velX[i] += DT * Fx;
        velY[i] += DT * Fy;
        velZ[i] += DT * Fz;
    }
}

__global__ void integrateBodiesKernel(double* __restrict__ posX,
                                      double* __restrict__ posY,
                                      double* __restrict__ posZ,
                                      const double* __restrict__ velX,
                                      const double* __restrict__ velY,
                                      const double* __restrict__ velZ,
                                      int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        posX[i] = fma(velX[i], DT, posX[i]);
        posY[i] = fma(velY[i], DT, posY[i]);
        posZ[i] = fma(velZ[i], DT, posZ[i]);
    }
}

void launchComputeForces(int blockSize, int numBlocks,
                         double* posX, double* posY, double* posZ,
                         double* velX, double* velY, double* velZ, int n) {
    switch (blockSize) {
        case 32:  computeForcesKernel<32><<<numBlocks, 32>>>(posX, posY, posZ, velX, velY, velZ, n); break;
        case 64:  computeForcesKernel<64><<<numBlocks, 64>>>(posX, posY, posZ, velX, velY, velZ, n); break;
        case 128: computeForcesKernel<128><<<numBlocks, 128>>>(posX, posY, posZ, velX, velY, velZ, n); break;
        default:  computeForcesKernel<256><<<numBlocks, 256>>>(posX, posY, posZ, velX, velY, velZ, n); break;
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

    // Upload to device in structure-of-arrays layout
    const size_t nBytes = (size_t)numBodies * sizeof(double);
    std::vector<double> host(6 * (size_t)numBodies);
    for (int i = 0; i < numBodies; ++i) {
        host[0 * numBodies + i] = bodies[i].pos.x;
        host[1 * numBodies + i] = bodies[i].pos.y;
        host[2 * numBodies + i] = bodies[i].pos.z;
        host[3 * numBodies + i] = bodies[i].vel.x;
        host[4 * numBodies + i] = bodies[i].vel.y;
        host[5 * numBodies + i] = bodies[i].vel.z;
    }

    double* dData = nullptr;
    CUDA_CHECK(cudaMalloc(&dData, 6 * nBytes));
    CUDA_CHECK(cudaMemcpy(dData, host.data(), 6 * nBytes, cudaMemcpyHostToDevice));
    double* dPosX = dData + 0 * numBodies;
    double* dPosY = dData + 1 * numBodies;
    double* dPosZ = dData + 2 * numBodies;
    double* dVelX = dData + 3 * numBodies;
    double* dVelY = dData + 4 * numBodies;
    double* dVelZ = dData + 5 * numBodies;

    // Pick the largest block size that still produces enough blocks to keep
    // all SMs busy; per-body work is serial, so grid width is what scales.
    int smCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));
    int blockSize = 256;
    while (blockSize > 32 && (numBodies + blockSize - 1) / blockSize < 2 * smCount) {
        blockSize /= 2;
    }
    const int numBlocks = (numBodies + blockSize - 1) / blockSize;

    // Warm-up launch with n=0 (a no-op on the data) so module load / kernel
    // initialization cost is not attributed to the timed simulation loop.
    integrateBodiesKernel<<<1, 32>>>(dPosX, dPosY, dPosZ, dVelX, dVelY, dVelZ, 0);
    launchComputeForces(blockSize, 1, dPosX, dPosY, dPosZ, dVelX, dVelY, dVelZ, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        launchComputeForces(blockSize, numBlocks, dPosX, dPosY, dPosZ,
                            dVelX, dVelY, dVelZ, numBodies);
        integrateBodiesKernel<<<numBlocks, blockSize>>>(dPosX, dPosY, dPosZ,
                                                        dVelX, dVelY, dVelZ, numBodies);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Download results back into the array-of-structs representation
    CUDA_CHECK(cudaMemcpy(host.data(), dData, 6 * nBytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = host[0 * numBodies + i];
        bodies[i].pos.y = host[1 * numBodies + i];
        bodies[i].pos.z = host[2 * numBodies + i];
        bodies[i].vel.x = host[3 * numBodies + i];
        bodies[i].vel.y = host[4 * numBodies + i];
        bodies[i].vel.z = host[5 * numBodies + i];
    }
    CUDA_CHECK(cudaFree(dData));

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
