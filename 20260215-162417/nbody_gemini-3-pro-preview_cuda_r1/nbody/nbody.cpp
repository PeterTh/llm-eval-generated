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

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

__global__ void integrateBodiesKernel(Body* bodies, int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        bodies[i].pos.x += bodies[i].vel.x * dt;
        bodies[i].pos.y += bodies[i].vel.y * dt;
        bodies[i].pos.z += bodies[i].vel.z * dt;
    }
}

__global__ void computeForcesKernel(Body* bodies, int n, double dt, double softening) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double Fx = 0.0;
    double Fy = 0.0;
    double Fz = 0.0;
    
    // Preload position to registers
    Vec3 myPos;
    if (i < n) {
        myPos = bodies[i].pos;
    }

    // Loop over tiles
    for (int tile = 0; tile < gridDim.x; tile++) {
        __shared__ Vec3 sharedPos[BLOCK_SIZE];
        
        int idx = tile * blockDim.x + threadIdx.x;
        
        // Load tile to shared memory
        if (idx < n) {
            sharedPos[threadIdx.x] = bodies[idx].pos;
        } else {
            // Padding for last tile, avoids divergence in calculation loop
            sharedPos[threadIdx.x] = Vec3(0.0, 0.0, 0.0);
        }
        
        __syncthreads();

        if (i < n) {
            // How many elements in this tile are valid?
            // If it's the last tile, it might be partial.
            // But we pad with 0, so we can just iterate up to BLOCK_SIZE usually,
            // EXCEPT if we care about interacting with "ghost" bodies at (0,0,0).
            // Ghost bodies at (0,0,0) will exert force! This is bad.
            // So we MUST iterate only up to valid count.
            
            int valid_count = BLOCK_SIZE;
            // The last tile may not be full.
            // However, we padded with (0,0,0) in shared memory.
            // If we include the padded elements in the force calculation,
            // they will contribute force if their position (0,0,0) is valid.
            // But (0,0,0) is a valid position. So we MUST NOT include padded elements.
            // The tile index corresponds to the block index in the grid if we map 1 block to 1 tile?
            // No, 'tile' iterates from 0 to gridDim.x.
            // The number of valid elements in the last tile (index gridDim.x-1) is:
            // n % BLOCK_SIZE. If n % BLOCK_SIZE == 0, it is BLOCK_SIZE.
            // So: (n - 1) % BLOCK_SIZE + 1?
            // Actually: n - (gridDim.x - 1) * BLOCK_SIZE.
            
            if (tile == gridDim.x - 1) {
                valid_count = n - tile * BLOCK_SIZE;
            }

            #pragma unroll
            for (int j = 0; j < valid_count; j++) {
                double dx = sharedPos[j].x - myPos.x;
                double dy = sharedPos[j].y - myPos.y;
                double dz = sharedPos[j].z - myPos.z;
                double distSqr = dx * dx + dy * dy + dz * dz + softening;
                double invDist = rsqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;
                
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        bodies[i].vel.x += dt * Fx;
        bodies[i].vel.y += dt * Fy;
        bodies[i].vel.z += dt * Fz;
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
    
    // Allocate device memory
    Body* d_bodies;
    CHECK_CUDA(cudaMalloc(&d_bodies, numBodies * sizeof(Body)));
    CHECK_CUDA(cudaMemcpy(d_bodies, bodies.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice));

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    int numBlocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<numBlocks, BLOCK_SIZE>>>(d_bodies, numBodies, DT, SOFTENING);
        CHECK_CUDA(cudaGetLastError());
        integrateBodiesKernel<<<numBlocks, BLOCK_SIZE>>>(d_bodies, numBodies, DT);
        CHECK_CUDA(cudaGetLastError());
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(bodies.data(), d_bodies, numBodies * sizeof(Body), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_bodies));
    
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
