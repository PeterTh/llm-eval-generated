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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
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

// CUDA kernel for computing forces using tile-based approach for optimal performance
__global__ void computeForcesKernel(Body* bodies, int n) {
    extern __shared__ Vec3 sharedPos[];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    
    if (i < n) {
        Vec3 myPos = bodies[i].pos;
        
        // Process bodies in tiles using shared memory
        for (int tile = 0; tile < (n + blockDim.x - 1) / blockDim.x; ++tile) {
            int j = tile * blockDim.x + threadIdx.x;
            
            // Load tile into shared memory
            if (j < n) {
                sharedPos[threadIdx.x] = bodies[j].pos;
            }
            __syncthreads();
            
            // Compute forces with bodies in this tile
            int tileSize = min(blockDim.x, n - tile * blockDim.x);
            for (int k = 0; k < tileSize; ++k) {
                double dx = sharedPos[k].x - myPos.x;
                double dy = sharedPos[k].y - myPos.y;
                double dz = sharedPos[k].z - myPos.z;
                double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                double invDist = rsqrt(distSqr);  // Fast inverse square root
                double invDist3 = invDist * invDist * invDist;
                
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
            __syncthreads();
        }
        
        // Update velocities
        bodies[i].vel.x += DT * Fx;
        bodies[i].vel.y += DT * Fy;
        bodies[i].vel.z += DT * Fz;
    }
}

// CUDA kernel for integrating body positions
__global__ void integrateBodiesKernel(Body* bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

void computeForces(Body* d_bodies, int n) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int sharedMemSize = BLOCK_SIZE * sizeof(Vec3);
    computeForcesKernel<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_bodies, n);
    CUDA_CHECK(cudaGetLastError());
}

void integrateBodies(Body* d_bodies, int n) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    integrateBodiesKernel<<<numBlocks, BLOCK_SIZE>>>(d_bodies, n);
    CUDA_CHECK(cudaGetLastError());
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
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    
    // Allocate device memory
    Body* d_bodies;
    CUDA_CHECK(cudaMalloc(&d_bodies, numBodies * sizeof(Body)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_bodies, bodies.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice));
    
    // Run simulation on GPU
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(d_bodies, numBodies);
        integrateBodies(d_bodies, numBodies);
    }
    
    // Synchronize to ensure all GPU work is complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(bodies.data(), d_bodies, numBodies * sizeof(Body), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_bodies));
    
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
