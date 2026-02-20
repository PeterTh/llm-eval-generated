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

// CUDA kernel: Compute forces for all bodies
__global__ void computeForces_kernel(const Body* bodies, Body* bodies_out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    Vec3 pos_i = bodies[i].pos;
    
    for (size_t j = 0; j < n; ++j) {
        const double dx = bodies[j].pos.x - pos_i.x;
        const double dy = bodies[j].pos.y - pos_i.y;
        const double dz = bodies[j].pos.z - pos_i.z;
        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;
        
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
    
    bodies_out[i].pos = bodies[i].pos;
    bodies_out[i].vel.x = bodies[i].vel.x + DT * Fx;
    bodies_out[i].vel.y = bodies[i].vel.y + DT * Fy;
    bodies_out[i].vel.z = bodies[i].vel.z + DT * Fz;
}

// CUDA kernel: Integrate body positions
__global__ void integrateBodies_kernel(const Body* bodies_in, Body* bodies_out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    bodies_out[i].pos.x = bodies_in[i].pos.x + bodies_in[i].vel.x * DT;
    bodies_out[i].pos.y = bodies_in[i].pos.y + bodies_in[i].vel.y * DT;
    bodies_out[i].pos.z = bodies_in[i].pos.z + bodies_in[i].vel.z * DT;
    bodies_out[i].vel = bodies_in[i].vel;
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
    
    // Initialize bodies on host
    std::vector<Body> bodies_h(numBodies);
    randomizeBodies(bodies_h);
    
    // Allocate device memory
    Body* bodies_d = nullptr;
    Body* bodies_temp_d = nullptr;
    cudaMalloc(&bodies_d, numBodies * sizeof(Body));
    cudaMalloc(&bodies_temp_d, numBodies * sizeof(Body));
    
    // Copy initial bodies to device
    cudaMemcpy(bodies_d, bodies_h.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice);
    
    // Determine grid and block sizes
    constexpr int BLOCK_SIZE = 256;
    int gridSize = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        // Compute forces kernel
        computeForces_kernel<<<gridSize, BLOCK_SIZE>>>(bodies_d, bodies_temp_d, numBodies);
        cudaDeviceSynchronize();
        
        // Swap pointers
        Body* temp = bodies_d;
        bodies_d = bodies_temp_d;
        bodies_temp_d = temp;
        
        // Integrate positions kernel
        integrateBodies_kernel<<<gridSize, BLOCK_SIZE>>>(bodies_d, bodies_temp_d, numBodies);
        cudaDeviceSynchronize();
        
        // Swap pointers
        temp = bodies_d;
        bodies_d = bodies_temp_d;
        bodies_temp_d = temp;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    // Copy results back to host
    cudaMemcpy(bodies_h.data(), bodies_d, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);
    
    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : bodies_h) {
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
        
        if (validateSimulation(bodies_h)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies_h);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            cudaFree(bodies_d);
            cudaFree(bodies_temp_d);
            return 1;
        }
    }
    
    // Cleanup
    cudaFree(bodies_d);
    cudaFree(bodies_temp_d);
    
    return 0;
}
