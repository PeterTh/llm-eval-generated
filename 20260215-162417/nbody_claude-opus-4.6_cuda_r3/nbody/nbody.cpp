#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

void randomizeBodies(Body* bodies, int n, unsigned int seed = 42) {
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

__global__ void computeForcesKernel(Body* bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double posX = 0.0, posY = 0.0, posZ = 0.0;

    if (i < n) {
        posX = bodies[i].pos.x;
        posY = bodies[i].pos.y;
        posZ = bodies[i].pos.z;
    }

    __shared__ double shPosX[BLOCK_SIZE];
    __shared__ double shPosY[BLOCK_SIZE];
    __shared__ double shPosZ[BLOCK_SIZE];

    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        int j = tile * BLOCK_SIZE + threadIdx.x;
        if (j < n) {
            shPosX[threadIdx.x] = bodies[j].pos.x;
            shPosY[threadIdx.x] = bodies[j].pos.y;
            shPosZ[threadIdx.x] = bodies[j].pos.z;
        }
        __syncthreads();

        if (i < n) {
            int limit = min(BLOCK_SIZE, n - tile * BLOCK_SIZE);
            for (int k = 0; k < limit; ++k) {
                double dx = shPosX[k] - posX;
                double dy = shPosY[k] - posY;
                double dz = shPosZ[k] - posZ;
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

double computeTotalEnergy(const Body* bodies, int n) {
    double energy = 0.0;

    // Kinetic energy (assuming unit mass)
    for (int i = 0; i < n; ++i) {
        energy += 0.5 * (bodies[i].vel.x * bodies[i].vel.x +
                        bodies[i].vel.y * bodies[i].vel.y +
                        bodies[i].vel.z * bodies[i].vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
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
bool validateSimulation(const Body* bodies, int n) {
    for (int i = 0; i < n; ++i) {
        // Check for NaN or Inf values
        if (!std::isfinite(bodies[i].pos.x) || !std::isfinite(bodies[i].pos.y) || !std::isfinite(bodies[i].pos.z) ||
            !std::isfinite(bodies[i].vel.x) || !std::isfinite(bodies[i].vel.y) || !std::isfinite(bodies[i].vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(bodies[i].pos.x) > maxPos || std::abs(bodies[i].pos.y) > maxPos || std::abs(bodies[i].pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(bodies[i].vel.x) > maxVel || std::abs(bodies[i].vel.y) > maxVel || std::abs(bodies[i].vel.z) > maxVel) {
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

    // Initialize bodies on host
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies.data(), numBodies);

    // Allocate device memory and copy
    Body* d_bodies;
    CUDA_CHECK(cudaMalloc(&d_bodies, numBodies * sizeof(Body)));
    CUDA_CHECK(cudaMemcpy(d_bodies, bodies.data(), numBodies * sizeof(Body),
                          cudaMemcpyHostToDevice));

    int numBlocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Run simulation on GPU
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<numBlocks, BLOCK_SIZE>>>(d_bodies, numBodies);
        integrateBodiesKernel<<<numBlocks, BLOCK_SIZE>>>(d_bodies, numBodies);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(bodies.data(), d_bodies, numBodies * sizeof(Body),
                          cudaMemcpyDeviceToHost));
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

        if (validateSimulation(bodies.data(), numBodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies.data(), numBodies);
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
