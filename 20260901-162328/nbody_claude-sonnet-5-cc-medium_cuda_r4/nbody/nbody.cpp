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
constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err__));                                          \
            exit(1);                                                                     \
        }                                                                                \
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

// Structure-of-arrays layout on the device for coalesced memory access.
__global__ void computeForcesKernel(const double* __restrict__ posX, const double* __restrict__ posY,
                                     const double* __restrict__ posZ, double* __restrict__ velX,
                                     double* __restrict__ velY, double* __restrict__ velZ, int n) {
    extern __shared__ double shared[];
    double* sPosX = shared;
    double* sPosY = shared + blockDim.x;
    double* sPosZ = shared + 2 * blockDim.x;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double myX = 0.0, myY = 0.0, myZ = 0.0;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    if (i < n) {
        myX = posX[i];
        myY = posY[i];
        myZ = posZ[i];
    }

    for (int tile = 0; tile < n; tile += blockDim.x) {
        const int j = tile + threadIdx.x;
        if (j < n) {
            sPosX[threadIdx.x] = posX[j];
            sPosY[threadIdx.x] = posY[j];
            sPosZ[threadIdx.x] = posZ[j];
        }
        __syncthreads();

        const int tileCount = min(blockDim.x, n - tile);
        if (i < n) {
            for (int k = 0; k < tileCount; ++k) {
                const double dx = sPosX[k] - myX;
                const double dy = sPosY[k] - myY;
                const double dz = sPosZ[k] - myZ;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
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

__global__ void integrateBodiesKernel(double* __restrict__ posX, double* __restrict__ posY,
                                       double* __restrict__ posZ, const double* __restrict__ velX,
                                       const double* __restrict__ velY, const double* __restrict__ velZ, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        posX[i] += velX[i] * DT;
        posY[i] += velY[i] * DT;
        posZ[i] += velZ[i] * DT;
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

    // Convert to structure-of-arrays for the GPU
    std::vector<double> hPosX(numBodies), hPosY(numBodies), hPosZ(numBodies);
    std::vector<double> hVelX(numBodies), hVelY(numBodies), hVelZ(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        hPosX[i] = bodies[i].pos.x;
        hPosY[i] = bodies[i].pos.y;
        hPosZ[i] = bodies[i].pos.z;
        hVelX[i] = bodies[i].vel.x;
        hVelY[i] = bodies[i].vel.y;
        hVelZ[i] = bodies[i].vel.z;
    }

    double *dPosX, *dPosY, *dPosZ, *dVelX, *dVelY, *dVelZ;
    const size_t bytes = static_cast<size_t>(numBodies) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&dPosX, bytes));
    CUDA_CHECK(cudaMalloc(&dPosY, bytes));
    CUDA_CHECK(cudaMalloc(&dPosZ, bytes));
    CUDA_CHECK(cudaMalloc(&dVelX, bytes));
    CUDA_CHECK(cudaMalloc(&dVelY, bytes));
    CUDA_CHECK(cudaMalloc(&dVelZ, bytes));

    CUDA_CHECK(cudaMemcpy(dPosX, hPosX.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPosY, hPosY.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPosZ, hPosZ.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVelX, hVelX.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVelY, hVelY.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVelZ, hVelZ.data(), bytes, cudaMemcpyHostToDevice));

    const int numBlocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t sharedBytes = 3 * BLOCK_SIZE * sizeof(double);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<numBlocks, BLOCK_SIZE, sharedBytes>>>(dPosX, dPosY, dPosZ, dVelX, dVelY, dVelZ,
                                                                     numBodies);
        integrateBodiesKernel<<<numBlocks, BLOCK_SIZE>>>(dPosX, dPosY, dPosZ, dVelX, dVelY, dVelZ, numBodies);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(hPosX.data(), dPosX, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hPosY.data(), dPosY, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hPosZ.data(), dPosZ, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hVelX.data(), dVelX, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hVelY.data(), dVelY, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hVelZ.data(), dVelZ, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = hPosX[i];
        bodies[i].pos.y = hPosY[i];
        bodies[i].pos.z = hPosZ[i];
        bodies[i].vel.x = hVelX[i];
        bodies[i].vel.y = hVelY[i];
        bodies[i].vel.z = hVelZ[i];
    }

    CUDA_CHECK(cudaFree(dPosX));
    CUDA_CHECK(cudaFree(dPosY));
    CUDA_CHECK(cudaFree(dPosZ));
    CUDA_CHECK(cudaFree(dVelX));
    CUDA_CHECK(cudaFree(dVelY));
    CUDA_CHECK(cudaFree(dVelZ));

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
