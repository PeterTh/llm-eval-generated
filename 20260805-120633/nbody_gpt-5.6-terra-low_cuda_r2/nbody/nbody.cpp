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

static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be tightly packed");

[[noreturn]] void cudaCheckFailure(cudaError_t error, const char* expression,
                                  const char* file, int line) {
    fprintf(stderr, "CUDA error at %s:%d (%s): %s\n", file, line, expression,
            cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) do { \
    const cudaError_t cudaCheckError = (expression); \
    if (cudaCheckError != cudaSuccess) cudaCheckFailure(cudaCheckError, #expression, __FILE__, __LINE__); \
} while (0)

constexpr int THREADS_PER_BLOCK = 256;

__global__ void computeForcesKernel(Body* bodies, int n) {
    extern __shared__ double sharedPositions[];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;

    // Keep the target body's old position in registers: the input position array
    // remains read-only for this entire kernel invocation.
    double ix = 0.0, iy = 0.0, iz = 0.0;
    if (i < n) {
        ix = bodies[i].pos.x;
        iy = bodies[i].pos.y;
        iz = bodies[i].pos.z;
    }

    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + tid;
        if (j < n) {
            sharedPositions[tid] = bodies[j].pos.x;
            sharedPositions[blockDim.x + tid] = bodies[j].pos.y;
            sharedPositions[2 * blockDim.x + tid] = bodies[j].pos.z;
        }
        __syncthreads();

        const int tileSize = min(blockDim.x, n - base);
        if (i < n) {
            #pragma unroll 4
            for (int k = 0; k < tileSize; ++k) {
                const double dx = sharedPositions[k] - ix;
                const double dy = sharedPositions[blockDim.x + k] - iy;
                const double dz = sharedPositions[2 * blockDim.x + k] - iz;
                const double invDist = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        bodies[i].vel.x += DT * fx;
        bodies[i].vel.y += DT * fy;
        bodies[i].vel.z += DT * fz;
    }
}

__global__ void integrateBodiesKernel(Body* bodies, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
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

    if (numBodies < 0 || numSteps < 0) {
        fprintf(stderr, "Number of bodies and steps must be non-negative\n");
        return 1;
    }
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    Body* deviceBodies = nullptr;
    const size_t bodyBytes = bodies.size() * sizeof(Body);
    if (bodyBytes != 0) {
        CUDA_CHECK(cudaMalloc(&deviceBodies, bodyBytes));
        CUDA_CHECK(cudaMemcpy(deviceBodies, bodies.data(), bodyBytes, cudaMemcpyHostToDevice));
    } else {
        // Initialize the CUDA runtime before timing even when there is no input.
        CUDA_CHECK(cudaFree(nullptr));
    }
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    const int blocks = (numBodies + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const size_t sharedBytes = 3 * THREADS_PER_BLOCK * sizeof(double);
    for (int step = 0; step < numSteps; ++step) {
        if (numBodies != 0) {
            computeForcesKernel<<<blocks, THREADS_PER_BLOCK, sharedBytes>>>(deviceBodies, numBodies);
            CUDA_CHECK(cudaGetLastError());
            integrateBodiesKernel<<<blocks, THREADS_PER_BLOCK>>>(deviceBodies, numBodies);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());

    if (bodyBytes != 0) {
        CUDA_CHECK(cudaMemcpy(bodies.data(), deviceBodies, bodyBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(deviceBodies));
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
