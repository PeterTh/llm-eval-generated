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

__global__ void stepBodies(const Body* source, Body* destination, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double tileX[256];
    __shared__ double tileY[256];
    __shared__ double tileZ[256];
    const bool active = i < n;
    const double x = active ? source[i].pos.x : 0.0;
    const double y = active ? source[i].pos.y : 0.0;
    const double z = active ? source[i].pos.z : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + threadIdx.x;
        if (j < n) {
            tileX[threadIdx.x] = source[j].pos.x;
            tileY[threadIdx.x] = source[j].pos.y;
            tileZ[threadIdx.x] = source[j].pos.z;
        }
        __syncthreads();
        const int count = min(static_cast<int>(blockDim.x), n - base);
        for (int k = 0; k < count; ++k) {
            const double dx = tileX[k] - x;
            const double dy = tileY[k] - y;
            const double dz = tileZ[k] - z;
            const double d2 = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double inv = 1.0 / sqrt(d2);
            const double inv3 = inv * inv * inv;
            fx += dx * inv3;
            fy += dy * inv3;
            fz += dz * inv3;
        }
        __syncthreads();
    }
    if (!active) return;
    destination[i].vel.x = source[i].vel.x + DT * fx;
    destination[i].vel.y = source[i].vel.y + DT * fy;
    destination[i].vel.z = source[i].vel.z + DT * fz;
    destination[i].pos.x = x + destination[i].vel.x * DT;
    destination[i].pos.y = y + destination[i].vel.y * DT;
    destination[i].pos.z = z + destination[i].vel.z * DT;
}

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(1);
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
    
    // Run simulation on the GPU.
    Body* deviceBodies = nullptr;
    Body* deviceScratch = nullptr;
    if (numBodies > 0) {
        checkCuda(cudaMalloc(&deviceBodies, sizeof(Body) * static_cast<size_t>(numBodies)));
        checkCuda(cudaMalloc(&deviceScratch, sizeof(Body) * static_cast<size_t>(numBodies)));
        checkCuda(cudaMemcpy(deviceBodies, bodies.data(), sizeof(Body) * static_cast<size_t>(numBodies), cudaMemcpyHostToDevice));
    }
    Body* current = deviceBodies;
    Body* next = deviceScratch;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        if (numBodies > 0) {
            constexpr int threads = 256;
            stepBodies<<<(numBodies + threads - 1) / threads, threads>>>(current, next, numBodies);
            checkCuda(cudaGetLastError());
            Body* swap = current;
            current = next;
            next = swap;
        }
    }
    checkCuda(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());

    if (numBodies > 0) {
        checkCuda(cudaMemcpy(bodies.data(), current, sizeof(Body) * static_cast<size_t>(numBodies), cudaMemcpyDeviceToHost));
        checkCuda(cudaFree(deviceBodies));
        checkCuda(cudaFree(deviceScratch));
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
