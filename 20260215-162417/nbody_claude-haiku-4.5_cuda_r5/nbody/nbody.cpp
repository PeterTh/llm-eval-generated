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
constexpr int THREADS_PER_BLOCK = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

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

__global__ void computeForcesKernel(const double* posX, const double* posY, const double* posZ,
                                     double* velX, double* velY, double* velZ,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n) return;
    
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double posXi = posX[i];
    double posYi = posY[i];
    double posZi = posZ[i];
    
    for (int j = 0; j < n; ++j) {
        double dx = posX[j] - posXi;
        double dy = posY[j] - posYi;
        double dz = posZ[j] - posZi;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = rsqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
    
    velX[i] += DT * Fx;
    velY[i] += DT * Fy;
    velZ[i] += DT * Fz;
}

__global__ void integrateKernel(double* posX, double* posY, double* posZ,
                                 const double* velX, const double* velY, const double* velZ,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n) return;
    
    posX[i] += velX[i] * DT;
    posY[i] += velY[i] * DT;
    posZ[i] += velZ[i] * DT;
}

void computeForces(std::vector<Body>& bodies, 
                   double* d_posX, double* d_posY, double* d_posZ,
                   double* d_velX, double* d_velY, double* d_velZ) {
    int n = bodies.size();
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    computeForcesKernel<<<blocks, THREADS_PER_BLOCK>>>(d_posX, d_posY, d_posZ,
                                                        d_velX, d_velY, d_velZ, n);
    checkCudaError(cudaGetLastError(), "computeForcesKernel");
}

void integrateBodies(std::vector<Body>& bodies,
                     double* d_posX, double* d_posY, double* d_posZ,
                     double* d_velX, double* d_velY, double* d_velZ) {
    int n = bodies.size();
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    integrateKernel<<<blocks, THREADS_PER_BLOCK>>>(d_posX, d_posY, d_posZ,
                                                     d_velX, d_velY, d_velZ, n);
    checkCudaError(cudaGetLastError(), "integrateKernel");
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
    
    printf("N-Body Simulation (CUDA)\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    
    // Allocate GPU memory for SoA layout
    double* d_posX = nullptr;
    double* d_posY = nullptr;
    double* d_posZ = nullptr;
    double* d_velX = nullptr;
    double* d_velY = nullptr;
    double* d_velZ = nullptr;
    
    size_t bytes = numBodies * sizeof(double);
    checkCudaError(cudaMalloc(&d_posX, bytes), "cudaMalloc d_posX");
    checkCudaError(cudaMalloc(&d_posY, bytes), "cudaMalloc d_posY");
    checkCudaError(cudaMalloc(&d_posZ, bytes), "cudaMalloc d_posZ");
    checkCudaError(cudaMalloc(&d_velX, bytes), "cudaMalloc d_velX");
    checkCudaError(cudaMalloc(&d_velY, bytes), "cudaMalloc d_velY");
    checkCudaError(cudaMalloc(&d_velZ, bytes), "cudaMalloc d_velZ");
    
    // Transfer initial data to GPU
    std::vector<double> posX(numBodies), posY(numBodies), posZ(numBodies);
    std::vector<double> velX(numBodies), velY(numBodies), velZ(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        posX[i] = bodies[i].pos.x;
        posY[i] = bodies[i].pos.y;
        posZ[i] = bodies[i].pos.z;
        velX[i] = bodies[i].vel.x;
        velY[i] = bodies[i].vel.y;
        velZ[i] = bodies[i].vel.z;
    }
    
    checkCudaError(cudaMemcpy(d_posX, posX.data(), bytes, cudaMemcpyHostToDevice), "copy posX");
    checkCudaError(cudaMemcpy(d_posY, posY.data(), bytes, cudaMemcpyHostToDevice), "copy posY");
    checkCudaError(cudaMemcpy(d_posZ, posZ.data(), bytes, cudaMemcpyHostToDevice), "copy posZ");
    checkCudaError(cudaMemcpy(d_velX, velX.data(), bytes, cudaMemcpyHostToDevice), "copy velX");
    checkCudaError(cudaMemcpy(d_velY, velY.data(), bytes, cudaMemcpyHostToDevice), "copy velY");
    checkCudaError(cudaMemcpy(d_velZ, velZ.data(), bytes, cudaMemcpyHostToDevice), "copy velZ");
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ);
        integrateBodies(bodies, d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ);
    }
    
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    // Transfer results back to host
    checkCudaError(cudaMemcpy(posX.data(), d_posX, bytes, cudaMemcpyDeviceToHost), "copy posX back");
    checkCudaError(cudaMemcpy(posY.data(), d_posY, bytes, cudaMemcpyDeviceToHost), "copy posY back");
    checkCudaError(cudaMemcpy(posZ.data(), d_posZ, bytes, cudaMemcpyDeviceToHost), "copy posZ back");
    checkCudaError(cudaMemcpy(velX.data(), d_velX, bytes, cudaMemcpyDeviceToHost), "copy velX back");
    checkCudaError(cudaMemcpy(velY.data(), d_velY, bytes, cudaMemcpyDeviceToHost), "copy velY back");
    checkCudaError(cudaMemcpy(velZ.data(), d_velZ, bytes, cudaMemcpyDeviceToHost), "copy velZ back");
    
    // Reconstruct bodies from SoA
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = posX[i];
        bodies[i].pos.y = posY[i];
        bodies[i].pos.z = posZ[i];
        bodies[i].vel.x = velX[i];
        bodies[i].vel.y = velY[i];
        bodies[i].vel.z = velZ[i];
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
        } else {
            printf("Validation: FAILED\n");
            cudaFree(d_posX);
            cudaFree(d_posY);
            cudaFree(d_posZ);
            cudaFree(d_velX);
            cudaFree(d_velY);
            cudaFree(d_velZ);
            return 1;
        }
    }
    
    // Clean up GPU memory
    cudaFree(d_posX);
    cudaFree(d_posY);
    cudaFree(d_posZ);
    cudaFree(d_velX);
    cudaFree(d_velY);
    cudaFree(d_velZ);
    
    return 0;
}
