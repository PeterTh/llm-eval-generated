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
constexpr int CUDA_BLOCK_SIZE = 256;

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

__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void advanceBodies(Body* __restrict__ bodies, const int n) {
    // A tile is reused by every thread in the block, reducing global reads
    // from O(N^2) to O(N^2 / blockSize) while retaining one output per body.
    __shared__ Vec3 tile[CUDA_BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const Vec3 position = active ? bodies[i].pos : Vec3{};
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;

    for (int tileStart = 0; tileStart < n; tileStart += blockDim.x) {
        const int source = tileStart + threadIdx.x;
        if (source < n) {
            tile[threadIdx.x] = bodies[source].pos;
        }
        __syncthreads();

        const int tileCount = min(blockDim.x, n - tileStart);
        #pragma unroll 4
        for (int k = 0; k < tileCount; ++k) {
            if (active) {
                const double dx = tile[k].x - position.x;
                const double dy = tile[k].y - position.y;
                const double dz = tile[k].z - position.z;
                const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inverseDistance = 1.0 / sqrt(distanceSquared);
                const double inverseDistanceCubed = inverseDistance * inverseDistance * inverseDistance;
                fx += dx * inverseDistanceCubed;
                fy += dy * inverseDistanceCubed;
                fz += dz * inverseDistanceCubed;
            }
        }
        __syncthreads();
    }

    if (!active) {
        return;
    }
    Vec3 velocity = bodies[i].vel;
    velocity.x += DT * fx;
    velocity.y += DT * fy;
    velocity.z += DT * fz;
    bodies[i].vel = velocity;
    bodies[i].pos.x = position.x + velocity.x * DT;
    bodies[i].pos.y = position.y + velocity.y * DT;
    bodies[i].pos.z = position.z + velocity.z * DT;
}

[[noreturn]] void cudaFailure(const char* operation, const cudaError_t error) {
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void checkCuda(const char* operation, const cudaError_t error) {
    if (error != cudaSuccess) {
        cudaFailure(operation, error);
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

    Body* deviceBodies = nullptr;
    if (numBodies > 0) {
        checkCuda("cudaMalloc", cudaMalloc(&deviceBodies, bodies.size() * sizeof(Body)));
        checkCuda("cudaMemcpy host-to-device", cudaMemcpy(deviceBodies, bodies.data(),
                                                           bodies.size() * sizeof(Body),
                                                           cudaMemcpyHostToDevice));
    }
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    if (numBodies > 0) {
        for (int step = 0; step < numSteps; ++step) {
            const int blocks = (numBodies + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            advanceBodies<<<blocks, CUDA_BLOCK_SIZE>>>(deviceBodies, numBodies);
            checkCuda("advanceBodies launch", cudaGetLastError());
        }

        checkCuda("advanceBodies completion", cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();

    if (numBodies > 0) {
        checkCuda("cudaMemcpy device-to-host", cudaMemcpy(bodies.data(), deviceBodies,
                                                           bodies.size() * sizeof(Body),
                                                           cudaMemcpyDeviceToHost));
        checkCuda("cudaFree", cudaFree(deviceBodies));
    }

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
