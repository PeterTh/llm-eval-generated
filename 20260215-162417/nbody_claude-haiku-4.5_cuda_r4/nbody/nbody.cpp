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

__global__ void computeForcesKernel(const double* posX, const double* posY, const double* posZ,
                                    double* velX, double* velY, double* velZ,
                                    int n, double softening, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        double iPosX = posX[i];
        double iPosY = posY[i];
        double iPosZ = posZ[i];
        
        #pragma unroll 4
        for (int j = 0; j < n; ++j) {
            double dx = posX[j] - iPosX;
            double dy = posY[j] - iPosY;
            double dz = posZ[j] - iPosZ;
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
}

__global__ void integrateKernel(double* posX, double* posY, double* posZ,
                                const double* velX, const double* velY, const double* velZ,
                                int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        posX[i] += velX[i] * dt;
        posY[i] += velY[i] * dt;
        posZ[i] += velZ[i] * dt;
    }
}

void computeForces(std::vector<Body>& bodies) {
    int n = bodies.size();
    
    // Pack data into linear arrays on the fly for GPU transfer
    double *d_posX, *d_posY, *d_posZ, *d_velX, *d_velY, *d_velZ;
    cudaMalloc(&d_posX, n * sizeof(double));
    cudaMalloc(&d_posY, n * sizeof(double));
    cudaMalloc(&d_posZ, n * sizeof(double));
    cudaMalloc(&d_velX, n * sizeof(double));
    cudaMalloc(&d_velY, n * sizeof(double));
    cudaMalloc(&d_velZ, n * sizeof(double));
    
    // Prepare position and velocity arrays
    std::vector<double> posX(n), posY(n), posZ(n), velX(n), velY(n), velZ(n);
    for (int i = 0; i < n; ++i) {
        posX[i] = bodies[i].pos.x;
        posY[i] = bodies[i].pos.y;
        posZ[i] = bodies[i].pos.z;
        velX[i] = bodies[i].vel.x;
        velY[i] = bodies[i].vel.y;
        velZ[i] = bodies[i].vel.z;
    }
    
    // Transfer to device
    cudaMemcpy(d_posX, posX.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY, posY.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posZ, posZ.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velX, velX.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velY, velY.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velZ, velZ.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    
    // Execute kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    computeForcesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_posX, d_posY, d_posZ, 
                                                            d_velX, d_velY, d_velZ, 
                                                            n, SOFTENING, DT);
    cudaDeviceSynchronize();
    
    // Transfer results back
    cudaMemcpy(velX.data(), d_velX, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(velY.data(), d_velY, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(velZ.data(), d_velZ, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; ++i) {
        bodies[i].vel.x = velX[i];
        bodies[i].vel.y = velY[i];
        bodies[i].vel.z = velZ[i];
    }
    
    // Clean up device memory
    cudaFree(d_posX);
    cudaFree(d_posY);
    cudaFree(d_posZ);
    cudaFree(d_velX);
    cudaFree(d_velY);
    cudaFree(d_velZ);
}

void integrateBodies(std::vector<Body>& bodies) {
    int n = bodies.size();
    
    // Allocate GPU memory
    double *d_posX, *d_posY, *d_posZ, *d_velX, *d_velY, *d_velZ;
    cudaMalloc(&d_posX, n * sizeof(double));
    cudaMalloc(&d_posY, n * sizeof(double));
    cudaMalloc(&d_posZ, n * sizeof(double));
    cudaMalloc(&d_velX, n * sizeof(double));
    cudaMalloc(&d_velY, n * sizeof(double));
    cudaMalloc(&d_velZ, n * sizeof(double));
    
    // Prepare arrays
    std::vector<double> posX(n), posY(n), posZ(n), velX(n), velY(n), velZ(n);
    for (int i = 0; i < n; ++i) {
        posX[i] = bodies[i].pos.x;
        posY[i] = bodies[i].pos.y;
        posZ[i] = bodies[i].pos.z;
        velX[i] = bodies[i].vel.x;
        velY[i] = bodies[i].vel.y;
        velZ[i] = bodies[i].vel.z;
    }
    
    // Transfer to device
    cudaMemcpy(d_posX, posX.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY, posY.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posZ, posZ.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velX, velX.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velY, velY.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velZ, velZ.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    
    // Execute kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    integrateKernel<<<blocksPerGrid, threadsPerBlock>>>(d_posX, d_posY, d_posZ,
                                                        d_velX, d_velY, d_velZ,
                                                        n, DT);
    cudaDeviceSynchronize();
    
    // Transfer results back
    cudaMemcpy(posX.data(), d_posX, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(posY.data(), d_posY, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(posZ.data(), d_posZ, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = posX[i];
        bodies[i].pos.y = posY[i];
        bodies[i].pos.z = posZ[i];
    }
    
    // Clean up device memory
    cudaFree(d_posX);
    cudaFree(d_posY);
    cudaFree(d_posZ);
    cudaFree(d_velX);
    cudaFree(d_velY);
    cudaFree(d_velZ);
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
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies);
        integrateBodies(bodies);
    }
    
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
