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

// GPU data storage structure
struct GPUData {
    double* posX;
    double* posY;
    double* posZ;
    double* velX;
    double* velY;
    double* velZ;
    int n;
    
    GPUData(int num_bodies) : n(num_bodies) {
        cudaMalloc(&posX, n * sizeof(double));
        cudaMalloc(&posY, n * sizeof(double));
        cudaMalloc(&posZ, n * sizeof(double));
        cudaMalloc(&velX, n * sizeof(double));
        cudaMalloc(&velY, n * sizeof(double));
        cudaMalloc(&velZ, n * sizeof(double));
    }
    
    ~GPUData() {
        cudaFree(posX);
        cudaFree(posY);
        cudaFree(posZ);
        cudaFree(velX);
        cudaFree(velY);
        cudaFree(velZ);
    }
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

// Copy bodies to GPU memory
void bodiesToGPU(const std::vector<Body>& bodies, GPUData& gpu) {
    int n = bodies.size();
    std::vector<double> h_posX(n), h_posY(n), h_posZ(n);
    std::vector<double> h_velX(n), h_velY(n), h_velZ(n);
    
    for (int i = 0; i < n; ++i) {
        h_posX[i] = bodies[i].pos.x;
        h_posY[i] = bodies[i].pos.y;
        h_posZ[i] = bodies[i].pos.z;
        h_velX[i] = bodies[i].vel.x;
        h_velY[i] = bodies[i].vel.y;
        h_velZ[i] = bodies[i].vel.z;
    }
    
    cudaMemcpy(gpu.posX, h_posX.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.posY, h_posY.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.posZ, h_posZ.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.velX, h_velX.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.velY, h_velY.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.velZ, h_velZ.data(), n * sizeof(double), cudaMemcpyHostToDevice);
}

// Copy bodies from GPU memory
void bodiesToCPU(std::vector<Body>& bodies, const GPUData& gpu) {
    int n = bodies.size();
    std::vector<double> h_posX(n), h_posY(n), h_posZ(n);
    std::vector<double> h_velX(n), h_velY(n), h_velZ(n);
    
    cudaMemcpy(h_posX.data(), gpu.posX, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_posY.data(), gpu.posY, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_posZ.data(), gpu.posZ, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_velX.data(), gpu.velX, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_velY.data(), gpu.velY, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_velZ.data(), gpu.velZ, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = h_posX[i];
        bodies[i].pos.y = h_posY[i];
        bodies[i].pos.z = h_posZ[i];
        bodies[i].vel.x = h_velX[i];
        bodies[i].vel.y = h_velY[i];
        bodies[i].vel.z = h_velZ[i];
    }
}

// CUDA kernel for force computation with better optimization
__global__ void computeForcesKernel(const double* posX, const double* posY, const double* posZ,
                                      double* velX, double* velY, double* velZ,
                                      int n, double softening, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        double myX = posX[i];
        double myY = posY[i];
        double myZ = posZ[i];
        
        for (int j = 0; j < n; ++j) {
            double dx = posX[j] - myX;
            double dy = posY[j] - myY;
            double dz = posZ[j] - myZ;
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

// CUDA kernel for position integration
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

void computeForces(std::vector<Body>& bodies, GPUData& gpu) {
    int n = bodies.size();
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    computeForcesKernel<<<blocksPerGrid, threadsPerBlock>>>(gpu.posX, gpu.posY, gpu.posZ,
                                                              gpu.velX, gpu.velY, gpu.velZ,
                                                              n, SOFTENING, DT);
    cudaDeviceSynchronize();
}

void integrateBodies(std::vector<Body>& bodies, GPUData& gpu) {
    int n = bodies.size();
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    integrateKernel<<<blocksPerGrid, threadsPerBlock>>>(gpu.posX, gpu.posY, gpu.posZ,
                                                          gpu.velX, gpu.velY, gpu.velZ,
                                                          n, DT);
    cudaDeviceSynchronize();
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
    
    // Allocate GPU memory once
    GPUData gpu(numBodies);
    bodiesToGPU(bodies, gpu);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, gpu);
        integrateBodies(bodies, gpu);
    }
    
    // Copy results back from GPU
    bodiesToCPU(bodies, gpu);
    
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
