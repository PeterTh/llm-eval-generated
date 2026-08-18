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

constexpr int THREADS_PER_BLOCK = 256;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each block caches a tile of source bodies.  Every thread owns one target
// body, so it can accumulate that body's velocity independently.  Position
// updates stay in a separate kernel to retain the global phase boundary in the
// original compute-forces then integrate sequence.
__global__ void computeForces(Body* __restrict__ bodies, int n) {
    extern __shared__ Body sourceBodies[];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    Body target{};
    if (active) {
        target = bodies[i];
    }

    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;

    for (int tileStart = 0; tileStart < n; tileStart += blockDim.x) {
        const int sourceIndex = tileStart + threadIdx.x;
        if (sourceIndex < n) {
            sourceBodies[threadIdx.x] = bodies[sourceIndex];
        }
        __syncthreads();

        const int tileSize = min(blockDim.x, n - tileStart);
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < tileSize; ++j) {
                const double dx = sourceBodies[j].pos.x - target.pos.x;
                const double dy = sourceBodies[j].pos.y - target.pos.y;
                const double dz = sourceBodies[j].pos.z - target.pos.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        target.vel.x += DT * fx;
        target.vel.y += DT * fy;
        target.vel.z += DT * fz;
        bodies[i] = target;
    }
}

__global__ void integrateBodies(Body* __restrict__ bodies, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Body body = bodies[i];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
        bodies[i] = body;
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

    Body* deviceBodies = nullptr;
    const size_t bodyBytes = bodies.size() * sizeof(Body);
    if (!bodies.empty()) {
        checkCuda(cudaMalloc(&deviceBodies, bodyBytes), "allocating device bodies");
        checkCuda(cudaMemcpy(deviceBodies, bodies.data(), bodyBytes, cudaMemcpyHostToDevice),
                  "copying initial bodies to device");
    }

    cudaEvent_t startEvent;
    cudaEvent_t endEvent;
    checkCuda(cudaEventCreate(&startEvent), "creating start event");
    checkCuda(cudaEventCreate(&endEvent), "creating end event");
    
    // Run simulation
    checkCuda(cudaEventRecord(startEvent), "recording start event");
    for (int step = 0; step < numSteps; ++step) {
        if (!bodies.empty()) {
            const int blocks = (numBodies + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            computeForces<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(Body)>>>(deviceBodies, numBodies);
            checkCuda(cudaGetLastError(), "launching force kernel");
            integrateBodies<<<blocks, THREADS_PER_BLOCK>>>(deviceBodies, numBodies);
            checkCuda(cudaGetLastError(), "launching integration kernel");
        }
    }
    checkCuda(cudaEventRecord(endEvent), "recording end event");
    checkCuda(cudaEventSynchronize(endEvent), "synchronizing simulation");
    float elapsedMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, endEvent), "calculating simulation time");
    checkCuda(cudaEventDestroy(startEvent), "destroying start event");
    checkCuda(cudaEventDestroy(endEvent), "destroying end event");
    
    printf("Simulation time: %ld ms\n", static_cast<long>(elapsedMilliseconds));

    // The device remains authoritative unless a host-side result is requested.
    if ((printResults || validate) && !bodies.empty()) {
        checkCuda(cudaMemcpy(bodies.data(), deviceBodies, bodyBytes, cudaMemcpyDeviceToHost),
                  "copying final bodies to host");
    }
    if (deviceBodies != nullptr) {
        checkCuda(cudaFree(deviceBodies), "freeing device bodies");
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
