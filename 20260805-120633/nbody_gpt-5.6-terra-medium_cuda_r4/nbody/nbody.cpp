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
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept
        : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

namespace {

constexpr int THREADS_PER_BLOCK = 256;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each block stages one tile of source bodies in shared memory.  Every thread
// still visits source bodies in increasing index order, matching the original
// force accumulation order while avoiding repeated global-memory loads.
__global__ void computeForcesKernel(Body* bodies, int n) {
    extern __shared__ Body sourceTile[];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    Body body;
    if (active) {
        body = bodies[i];
    }

    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;

    for (int tileStart = 0; tileStart < n; tileStart += blockDim.x) {
        const int sourceIndex = tileStart + threadIdx.x;
        if (sourceIndex < n) {
            sourceTile[threadIdx.x] = bodies[sourceIndex];
        }
        __syncthreads();

        const int tileSize = min(blockDim.x, n - tileStart);
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < tileSize; ++j) {
                const double dx = sourceTile[j].pos.x - body.pos.x;
                const double dy = sourceTile[j].pos.y - body.pos.y;
                const double dz = sourceTile[j].pos.z - body.pos.z;
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
        bodies[i].vel.x = body.vel.x + DT * fx;
        bodies[i].vel.y = body.vel.y + DT * fy;
        bodies[i].vel.z = body.vel.z + DT * fz;
    }
}

__global__ void integrateBodiesKernel(Body* bodies, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Body body = bodies[i];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
        bodies[i] = body;
    }
}

void simulateOnGpu(Body* deviceBodies, int numBodies, int numSteps) {
    if (numBodies == 0) {
        return;
    }

    const dim3 block(THREADS_PER_BLOCK);
    const dim3 grid((numBodies + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    const size_t sharedBytes = THREADS_PER_BLOCK * sizeof(Body);
    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<grid, block, sharedBytes>>>(deviceBodies, numBodies);
        checkCuda(cudaGetLastError(), "force-kernel launch");
        integrateBodiesKernel<<<grid, block>>>(deviceBodies, numBodies);
        checkCuda(cudaGetLastError(), "integration-kernel launch");
    }
}

}  // namespace

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
    
    // Keep simulation state on the GPU for all steps; transfers occur only
    // before the run and when host-side output or validation is requested.
    Body* deviceBodies = nullptr;
    if (numBodies > 0) {
        checkCuda(cudaMalloc(&deviceBodies, static_cast<size_t>(numBodies) * sizeof(Body)), "device allocation");
        checkCuda(cudaMemcpy(deviceBodies, bodies.data(), static_cast<size_t>(numBodies) * sizeof(Body),
                             cudaMemcpyHostToDevice), "initial state upload");
    }

    cudaEvent_t start;
    cudaEvent_t end;
    checkCuda(cudaEventCreate(&start), "start-event creation");
    checkCuda(cudaEventCreate(&end), "end-event creation");
    checkCuda(cudaEventRecord(start), "start-event record");
    simulateOnGpu(deviceBodies, numBodies, numSteps);
    checkCuda(cudaEventRecord(end), "end-event record");
    checkCuda(cudaEventSynchronize(end), "simulation synchronization");

    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, end), "elapsed-time measurement");
    checkCuda(cudaEventDestroy(start), "start-event destruction");
    checkCuda(cudaEventDestroy(end), "end-event destruction");
    printf("Simulation time: %ld ms\n", static_cast<long>(elapsedMs));

    if (numBodies > 0 && (printResults || validate)) {
        checkCuda(cudaMemcpy(bodies.data(), deviceBodies, static_cast<size_t>(numBodies) * sizeof(Body),
                             cudaMemcpyDeviceToHost), "final state download");
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
            if (deviceBodies != nullptr) {
                checkCuda(cudaFree(deviceBodies), "device memory release");
            }
            return 0;
        } else {
            printf("Validation: FAILED\n");
            if (deviceBodies != nullptr) {
                checkCuda(cudaFree(deviceBodies), "device memory release");
            }
            return 1;
        }
    }
    
    if (deviceBodies != nullptr) {
        checkCuda(cudaFree(deviceBodies), "device memory release");
    }
    return 0;
}
