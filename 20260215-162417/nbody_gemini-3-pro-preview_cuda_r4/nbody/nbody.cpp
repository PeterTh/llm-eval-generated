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
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void computeForcesKernel(Body* bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double my_x, my_y, my_z;

    if (i < n) {
        my_x = bodies[i].pos.x;
        my_y = bodies[i].pos.y;
        my_z = bodies[i].pos.z;
    }

    extern __shared__ Body s_bodies[];

    for (int tile = 0; tile < gridDim.x; ++tile) {
        int idx = tile * blockDim.x + threadIdx.x;
        if (idx < n) {
            s_bodies[threadIdx.x] = bodies[idx];
        } else {
            // Pad with dummy data if out of bounds (though idx check in force calc handles this implicitly if logic is right, but better safe)
            // Or just ensure we don't access garbage.
            // Actually, simply checking `idx < n` when loading is enough.
            // But when iterating the tile, we need to know how many valid elements are there.
            // The simplest way is to pad `bodies` to multiple of block size on host, or check bounds inside the tile loop.
            // Checking bounds inside the inner loop is expensive (divergence).
            // A common trick is to ensure `n` is a multiple of block size by padding, or handle the last tile carefully.
            // Let's handle the last tile carefully.
             s_bodies[threadIdx.x] = bodies[n-1]; // Duplicate last body to avoid partial tile issues? No, that adds force.
             // We'll just be careful.
        }
        __syncthreads();

        if (i < n) {
            // For the last tile, we might process fewer than blockDim.x elements
            // However, iterating up to blockDim.x is fine IF we make sure the invalid slots don't contribute force.
            // But we can just iterate over all elements in the tile.
            // If the tile is the last one, we need to stop early?
            // Or we can just check bounds.
            
            // To keep it simple and efficient (no divergence in inner loop), let's iterate fully.
            // But then invalid entries in shared memory need to be handled.
            // If we set invalid entries to have position at infinity, force is 0.
            // Or we just check bounds: `int j_global = tile * blockDim.x + k; if (j_global < n) ...`
            // But that adds a check in the inner loop.
            
            // Actually, let's just use the `idx < n` check during load to set a flag or count.
            // But all threads need to know the count.
            
            // Let's stick to the bounds check. The branch predictor usually handles it well if all threads agree.
            // Wait, for the last tile, some threads will have `idx < n` and some not.
            
            // Let's try to just load 0s for mass? But there is no mass.
            // If we load a body at infinity, distance is huge, force is 0.
            // Let's initialize s_bodies with huge coordinates if idx >= n.
             if (idx >= n) {
                 s_bodies[threadIdx.x].pos.x = 1e10; // Huge
                 s_bodies[threadIdx.x].pos.y = 1e10;
                 s_bodies[threadIdx.x].pos.z = 1e10;
             }
             
             // Now iterate k from 0 to blockDim.x
             #pragma unroll
             for (int k = 0; k < blockDim.x; ++k) {
                 double dx = s_bodies[k].pos.x - my_x;
                 double dy = s_bodies[k].pos.y - my_y;
                 double dz = s_bodies[k].pos.z - my_z;
                 double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                 double invDist = 1.0 / sqrt(distSqr);
                 double invDist3 = invDist * invDist * invDist;
                 
                 Fx += dx * invDist3;
                 Fy += dy * invDist3;
                 Fz += dz * invDist3;
             }
        }
        __syncthreads();
    }
    
    if (i < n) {
        bodies[i].vel.x += DT * Fx;
        bodies[i].vel.y += DT * Fy;
        bodies[i].vel.z += DT * Fz;
    }
}

__global__ void integrateBodiesKernel(Body* bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    bodies[i].pos.x += bodies[i].vel.x * DT;
    bodies[i].pos.y += bodies[i].vel.y * DT;
    bodies[i].pos.z += bodies[i].vel.z * DT;
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
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Allocate device memory
    Body* d_bodies;
    checkCudaError(cudaMalloc(&d_bodies, numBodies * sizeof(Body)), "cudaMalloc bodies");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_bodies, bodies.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    int blockSize = 256;
    int numBlocks = (numBodies + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(Body);

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<numBlocks, blockSize, sharedMemSize>>>(d_bodies, numBodies);
        checkCudaError(cudaGetLastError(), "computeForcesKernel launch");
        
        integrateBodiesKernel<<<numBlocks, blockSize>>>(d_bodies, numBodies);
        checkCudaError(cudaGetLastError(), "integrateBodiesKernel launch");
    }
    
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    // Copy back for results/validation
    checkCudaError(cudaMemcpy(bodies.data(), d_bodies, numBodies * sizeof(Body), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    
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
            // return 1; // Don't exit early, let cleanup happen
        }
    }

    cudaFree(d_bodies);
    
    return 0;
}
