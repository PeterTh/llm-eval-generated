#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// CUDA kernel for computing forces
__global__ void computeForcesKernel(const double* posX, const double* posY, const double* posZ,
                                     double* velX, double* velY, double* velZ,
                                     int n, double softening, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double ix = posX[i], iy = posY[i], iz = posZ[i];
    
    for (int j = 0; j < n; ++j) {
        double dx = posX[j] - ix;
        double dy = posY[j] - iy;
        double dz = posZ[j] - iz;
        double distSqr = dx * dx + dy * dy + dz * dz + softening;
        double invDist = rsqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
    
    velX[i] += dt * Fx;
    velY[i] += dt * Fy;
    velZ[i] += dt * Fz;
}

// CUDA kernel for integrating bodies
__global__ void integrateBodiesKernel(double* posX, double* posY, double* posZ,
                                       const double* velX, const double* velY, const double* velZ,
                                       int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    posX[i] += velX[i] * dt;
    posY[i] += velY[i] * dt;
    posZ[i] += velZ[i] * dt;
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
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

void computeForces(std::vector<Body>& bodies, double* d_posX, double* d_posY, double* d_posZ,
                   double* d_velX, double* d_velY, double* d_velZ, int n) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computeForcesKernel<<<gridSize, BLOCK_SIZE>>>(d_posX, d_posY, d_posZ,
                                                    d_velX, d_velY, d_velZ,
                                                    n, SOFTENING, DT);
    checkCudaError(cudaGetLastError(), "computeForcesKernel");
}

void integrateBodies(std::vector<Body>& bodies, double* d_posX, double* d_posY, double* d_posZ,
                     const double* d_velX, const double* d_velY, const double* d_velZ, int n) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    integrateBodiesKernel<<<gridSize, BLOCK_SIZE>>>(d_posX, d_posY, d_posZ,
                                                      d_velX, d_velY, d_velZ,
                                                      n, DT);
    checkCudaError(cudaGetLastError(), "integrateBodiesKernel");
}

double computeTotalEnergy(const std::vector<Body>& bodies, const double* d_posX, const double* d_posY, const double* d_posZ,
                         const double* d_velX, const double* d_velY, const double* d_velZ) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Copy data back from device to host for energy computation
    std::vector<double> h_posX(n), h_posY(n), h_posZ(n);
    std::vector<double> h_velX(n), h_velY(n), h_velZ(n);
    
    checkCudaError(cudaMemcpy(h_posX.data(), d_posX, n * sizeof(double), cudaMemcpyDeviceToHost), "memcpy posX");
    checkCudaError(cudaMemcpy(h_posY.data(), d_posY, n * sizeof(double), cudaMemcpyDeviceToHost), "memcpy posY");
    checkCudaError(cudaMemcpy(h_posZ.data(), d_posZ, n * sizeof(double), cudaMemcpyDeviceToHost), "memcpy posZ");
    checkCudaError(cudaMemcpy(h_velX.data(), d_velX, n * sizeof(double), cudaMemcpyDeviceToHost), "memcpy velX");
    checkCudaError(cudaMemcpy(h_velY.data(), d_velY, n * sizeof(double), cudaMemcpyDeviceToHost), "memcpy velY");
    checkCudaError(cudaMemcpy(h_velZ.data(), d_velZ, n * sizeof(double), cudaMemcpyDeviceToHost), "memcpy velZ");
    
    // Kinetic energy (assuming unit mass)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (h_velX[i] * h_velX[i] + 
                        h_velY[i] * h_velY[i] + 
                        h_velZ[i] * h_velZ[i]);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = h_posX[j] - h_posX[i];
            const double dy = h_posY[j] - h_posY[i];
            const double dz = h_posZ[j] - h_posZ[i];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    
    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const double* d_posX, const double* d_posY, const double* d_posZ,
                       const double* d_velX, const double* d_velY, const double* d_velZ, int n) {
    std::vector<double> h_posX(n), h_posY(n), h_posZ(n);
    std::vector<double> h_velX(n), h_velY(n), h_velZ(n);
    
    checkCudaError(cudaMemcpy(h_posX.data(), d_posX, n * sizeof(double), cudaMemcpyDeviceToHost), "memcpy posX");
    checkCudaError(cudaMemcpy(h_posY.data(), d_posY, n * sizeof(double), cudaMemcpyDeviceToHost), "memcpy posY");
    checkCudaError(cudaMemcpy(h_posZ.data(), d_posZ, n * sizeof(double), cudaMemcpyDeviceToHost), "memcpy posZ");
    checkCudaError(cudaMemcpy(h_velX.data(), d_velX, n * sizeof(double), cudaMemcpyDeviceToHost), "memcpy velX");
    checkCudaError(cudaMemcpy(h_velY.data(), d_velY, n * sizeof(double), cudaMemcpyDeviceToHost), "memcpy velY");
    checkCudaError(cudaMemcpy(h_velZ.data(), d_velZ, n * sizeof(double), cudaMemcpyDeviceToHost), "memcpy velZ");
    
    for (int i = 0; i < n; ++i) {
        // Check for NaN or Inf values
        if (!std::isfinite(h_posX[i]) || !std::isfinite(h_posY[i]) || !std::isfinite(h_posZ[i]) ||
            !std::isfinite(h_velX[i]) || !std::isfinite(h_velY[i]) || !std::isfinite(h_velZ[i])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(h_posX[i]) > maxPos || std::abs(h_posY[i]) > maxPos || std::abs(h_posZ[i]) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(h_velX[i]) > maxVel || std::abs(h_velY[i]) > maxVel || std::abs(h_velZ[i]) > maxVel) {
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
    
    // Initialize bodies on host
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    
    // Allocate device memory
    double* d_posX, *d_posY, *d_posZ;
    double* d_velX, *d_velY, *d_velZ;
    size_t bytes = numBodies * sizeof(double);
    
    checkCudaError(cudaMalloc(&d_posX, bytes), "malloc posX");
    checkCudaError(cudaMalloc(&d_posY, bytes), "malloc posY");
    checkCudaError(cudaMalloc(&d_posZ, bytes), "malloc posZ");
    checkCudaError(cudaMalloc(&d_velX, bytes), "malloc velX");
    checkCudaError(cudaMalloc(&d_velY, bytes), "malloc velY");
    checkCudaError(cudaMalloc(&d_velZ, bytes), "malloc velZ");
    
    // Copy initial data to device
    std::vector<double> h_posX(numBodies), h_posY(numBodies), h_posZ(numBodies);
    std::vector<double> h_velX(numBodies), h_velY(numBodies), h_velZ(numBodies);
    
    for (int i = 0; i < numBodies; ++i) {
        h_posX[i] = bodies[i].pos.x;
        h_posY[i] = bodies[i].pos.y;
        h_posZ[i] = bodies[i].pos.z;
        h_velX[i] = bodies[i].vel.x;
        h_velY[i] = bodies[i].vel.y;
        h_velZ[i] = bodies[i].vel.z;
    }
    
    checkCudaError(cudaMemcpy(d_posX, h_posX.data(), bytes, cudaMemcpyHostToDevice), "memcpy posX to device");
    checkCudaError(cudaMemcpy(d_posY, h_posY.data(), bytes, cudaMemcpyHostToDevice), "memcpy posY to device");
    checkCudaError(cudaMemcpy(d_posZ, h_posZ.data(), bytes, cudaMemcpyHostToDevice), "memcpy posZ to device");
    checkCudaError(cudaMemcpy(d_velX, h_velX.data(), bytes, cudaMemcpyHostToDevice), "memcpy velX to device");
    checkCudaError(cudaMemcpy(d_velY, h_velY.data(), bytes, cudaMemcpyHostToDevice), "memcpy velY to device");
    checkCudaError(cudaMemcpy(d_velZ, h_velZ.data(), bytes, cudaMemcpyHostToDevice), "memcpy velZ to device");
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, numBodies);
        integrateBodies(bodies, d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, numBodies);
    }
    
    checkCudaError(cudaDeviceSynchronize(), "deviceSynchronize");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    // Print results for external validation
    if (printResults) {
        // Copy results back to host
        checkCudaError(cudaMemcpy(h_posX.data(), d_posX, bytes, cudaMemcpyDeviceToHost), "memcpy posX to host");
        checkCudaError(cudaMemcpy(h_posY.data(), d_posY, bytes, cudaMemcpyDeviceToHost), "memcpy posY to host");
        checkCudaError(cudaMemcpy(h_posZ.data(), d_posZ, bytes, cudaMemcpyDeviceToHost), "memcpy posZ to host");
        checkCudaError(cudaMemcpy(h_velX.data(), d_velX, bytes, cudaMemcpyDeviceToHost), "memcpy velX to host");
        checkCudaError(cudaMemcpy(h_velY.data(), d_velY, bytes, cudaMemcpyDeviceToHost), "memcpy velY to host");
        checkCudaError(cudaMemcpy(h_velZ.data(), d_velZ, bytes, cudaMemcpyDeviceToHost), "memcpy velZ to host");
        
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            bodyData.push_back(h_posX[i]);
            bodyData.push_back(h_posY[i]);
            bodyData.push_back(h_posZ[i]);
            bodyData.push_back(h_velX[i]);
            bodyData.push_back(h_velY[i]);
            bodyData.push_back(h_velZ[i]);
        }
        print_results(bodyData, "Bodies");
    }
    
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, numBodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies, d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    // Free device memory
    checkCudaError(cudaFree(d_posX), "free posX");
    checkCudaError(cudaFree(d_posY), "free posY");
    checkCudaError(cudaFree(d_posZ), "free posZ");
    checkCudaError(cudaFree(d_velX), "free velX");
    checkCudaError(cudaFree(d_velY), "free velY");
    checkCudaError(cudaFree(d_velZ), "free velZ");
    
    return validate ? (validateSimulation(d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, numBodies) ? 0 : 1) : 0;
}
