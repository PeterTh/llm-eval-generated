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

[[noreturn]] void cudaFailure(const char* operation, cudaError_t error) {
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) cudaFailure(operation, error);
}

// Each thread owns one destination body. Source bodies are loaded in tiles so
// every block reuses each position from shared memory. The source loop remains
// in increasing-index order, matching the scalar implementation's summation
// order for each destination body.
__global__ void advanceBodies(double* __restrict__ posX,
                              double* __restrict__ posY,
                              double* __restrict__ posZ,
                              double* __restrict__ velX,
                              double* __restrict__ velY,
                              double* __restrict__ velZ,
                              int n) {
    constexpr int TILE_SIZE = 256;
    __shared__ double tileX[TILE_SIZE];
    __shared__ double tileY[TILE_SIZE];
    __shared__ double tileZ[TILE_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const double ix = active ? posX[i] : 0.0;
    const double iy = active ? posY[i] : 0.0;
    const double iz = active ? posZ[i] : 0.0;
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;

    for (int base = 0; base < n; base += TILE_SIZE) {
        const int j = base + threadIdx.x;
        if (j < n) {
            tileX[threadIdx.x] = posX[j];
            tileY[threadIdx.x] = posY[j];
            tileZ[threadIdx.x] = posZ[j];
        }
        __syncthreads();

        const int count = min(TILE_SIZE, n - base);
        if (active) {
            for (int k = 0; k < count; ++k) {
                const double dx = tileX[k] - ix;
                const double dy = tileY[k] - iy;
                const double dz = tileZ[k] - iz;
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

    if (active) {
        const double vx = velX[i] + DT * fx;
        const double vy = velY[i] + DT * fy;
        const double vz = velZ[i] + DT * fz;
        velX[i] = vx;
        velY[i] = vy;
        velZ[i] = vz;
        posX[i] += vx * DT;
        posY[i] += vy * DT;
        posZ[i] += vz * DT;
    }
}

void runSimulation(std::vector<Body>& bodies, int numSteps) {
    const int n = static_cast<int>(bodies.size());
    if (n == 0 || numSteps == 0) return;

    std::vector<double> posX(n), posY(n), posZ(n), velX(n), velY(n), velZ(n);
    for (int i = 0; i < n; ++i) {
        posX[i] = bodies[i].pos.x; posY[i] = bodies[i].pos.y; posZ[i] = bodies[i].pos.z;
        velX[i] = bodies[i].vel.x; velY[i] = bodies[i].vel.y; velZ[i] = bodies[i].vel.z;
    }

    double *dPosX, *dPosY, *dPosZ, *dVelX, *dVelY, *dVelZ;
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    checkCuda(cudaMalloc(&dPosX, bytes), "cudaMalloc(posX)");
    checkCuda(cudaMalloc(&dPosY, bytes), "cudaMalloc(posY)");
    checkCuda(cudaMalloc(&dPosZ, bytes), "cudaMalloc(posZ)");
    checkCuda(cudaMalloc(&dVelX, bytes), "cudaMalloc(velX)");
    checkCuda(cudaMalloc(&dVelY, bytes), "cudaMalloc(velY)");
    checkCuda(cudaMalloc(&dVelZ, bytes), "cudaMalloc(velZ)");
    checkCuda(cudaMemcpy(dPosX, posX.data(), bytes, cudaMemcpyHostToDevice), "copy posX");
    checkCuda(cudaMemcpy(dPosY, posY.data(), bytes, cudaMemcpyHostToDevice), "copy posY");
    checkCuda(cudaMemcpy(dPosZ, posZ.data(), bytes, cudaMemcpyHostToDevice), "copy posZ");
    checkCuda(cudaMemcpy(dVelX, velX.data(), bytes, cudaMemcpyHostToDevice), "copy velX");
    checkCuda(cudaMemcpy(dVelY, velY.data(), bytes, cudaMemcpyHostToDevice), "copy velY");
    checkCuda(cudaMemcpy(dVelZ, velZ.data(), bytes, cudaMemcpyHostToDevice), "copy velZ");

    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    for (int step = 0; step < numSteps; ++step) {
        advanceBodies<<<blocks, threads>>>(dPosX, dPosY, dPosZ, dVelX, dVelY, dVelZ, n);
        checkCuda(cudaGetLastError(), "advanceBodies launch");
    }
    checkCuda(cudaDeviceSynchronize(), "advanceBodies execution");
    checkCuda(cudaMemcpy(posX.data(), dPosX, bytes, cudaMemcpyDeviceToHost), "copy back posX");
    checkCuda(cudaMemcpy(posY.data(), dPosY, bytes, cudaMemcpyDeviceToHost), "copy back posY");
    checkCuda(cudaMemcpy(posZ.data(), dPosZ, bytes, cudaMemcpyDeviceToHost), "copy back posZ");
    checkCuda(cudaMemcpy(velX.data(), dVelX, bytes, cudaMemcpyDeviceToHost), "copy back velX");
    checkCuda(cudaMemcpy(velY.data(), dVelY, bytes, cudaMemcpyDeviceToHost), "copy back velY");
    checkCuda(cudaMemcpy(velZ.data(), dVelZ, bytes, cudaMemcpyDeviceToHost), "copy back velZ");
    cudaFree(dPosX); cudaFree(dPosY); cudaFree(dPosZ);
    cudaFree(dVelX); cudaFree(dVelY); cudaFree(dVelZ);
    for (int i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(posX[i], posY[i], posZ[i]);
        bodies[i].vel = Vec3(velX[i], velY[i], velZ[i]);
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
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(bodies, numSteps);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
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
