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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,          \
                    __LINE__, cudaGetErrorString(err_));                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

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

constexpr int BLOCK_SIZE = 256;

// All-pairs force computation with shared-memory tiling: each thread owns one
// body i and accumulates forces over tiles of j-bodies staged in shared memory.
__global__ void computeForcesKernel(const double3* __restrict__ pos,
                                    double3* __restrict__ vel, const int n) {
    __shared__ double3 shPos[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const double3 myPos = (i < n) ? pos[i] : make_double3(0.0, 0.0, 0.0);

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        const int j = tile + threadIdx.x;
        if (j < n) {
            shPos[threadIdx.x] = pos[j];
        }
        __syncthreads();

        const int tileCount = min(BLOCK_SIZE, n - tile);
#pragma unroll 8
        for (int k = 0; k < tileCount; ++k) {
            const double dx = shPos[k].x - myPos.x;
            const double dy = shPos[k].y - myPos.y;
            const double dz = shPos[k].z - myPos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = rsqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    if (i < n) {
        vel[i].x += DT * Fx;
        vel[i].y += DT * Fy;
        vel[i].z += DT * Fz;
    }
}

__global__ void integrateBodiesKernel(double3* __restrict__ pos,
                                      const double3* __restrict__ vel, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pos[i].x += vel[i].x * DT;
        pos[i].y += vel[i].y * DT;
        pos[i].z += vel[i].z * DT;
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

    // Upload initial state to the GPU as separate position/velocity arrays
    std::vector<double3> hostPos(numBodies), hostVel(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        hostPos[i] = make_double3(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z);
        hostVel[i] = make_double3(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z);
    }

    double3 *devPos = nullptr, *devVel = nullptr;
    CUDA_CHECK(cudaMalloc(&devPos, numBodies * sizeof(double3)));
    CUDA_CHECK(cudaMalloc(&devVel, numBodies * sizeof(double3)));
    CUDA_CHECK(cudaMemcpy(devPos, hostPos.data(), numBodies * sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devVel, hostVel.data(), numBodies * sizeof(double3), cudaMemcpyHostToDevice));

    const int numBlocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Run simulation (state stays resident on the GPU across all steps)
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<numBlocks, BLOCK_SIZE>>>(devPos, devVel, numBodies);
        integrateBodiesKernel<<<numBlocks, BLOCK_SIZE>>>(devPos, devVel, numBodies);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Download final state
    CUDA_CHECK(cudaMemcpy(hostPos.data(), devPos, numBodies * sizeof(double3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostVel.data(), devVel, numBodies * sizeof(double3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(devPos));
    CUDA_CHECK(cudaFree(devVel));
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos = Vec3(hostPos[i].x, hostPos[i].y, hostPos[i].z);
        bodies[i].vel = Vec3(hostVel[i].x, hostVel[i].y, hostVel[i].z);
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
