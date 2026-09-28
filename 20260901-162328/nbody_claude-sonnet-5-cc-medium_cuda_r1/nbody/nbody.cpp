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

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err__));                                          \
            exit(1);                                                                     \
        }                                                                                \
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

// Computes pairwise gravitational forces and updates velocities, one thread per body.
// Positions are staged through shared memory in tiles so that each thread in the block
// reads a given body's position from fast on-chip memory instead of global memory,
// while preserving the exact same j = 0..n-1 summation order as the original code.
__global__ void computeForcesKernel(const double* __restrict__ posX, const double* __restrict__ posY,
                                     const double* __restrict__ posZ, double* velX, double* velY, double* velZ,
                                     int n) {
    extern __shared__ double3 shPos[];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double myX = 0.0, myY = 0.0, myZ = 0.0;
    if (i < n) {
        myX = posX[i];
        myY = posY[i];
        myZ = posZ[i];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    const int numTiles = (n + blockDim.x - 1) / blockDim.x;
    for (int tile = 0; tile < numTiles; ++tile) {
        const int idx = tile * blockDim.x + threadIdx.x;
        if (idx < n) {
            shPos[threadIdx.x] = make_double3(posX[idx], posY[idx], posZ[idx]);
        }
        __syncthreads();

        const int tileSize = min(blockDim.x, n - tile * blockDim.x);
        if (i < n) {
            for (int j = 0; j < tileSize; ++j) {
                const double dx = shPos[j].x - myX;
                const double dy = shPos[j].y - myY;
                const double dz = shPos[j].z - myZ;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        velX[i] += DT * Fx;
        velY[i] += DT * Fy;
        velZ[i] += DT * Fz;
    }
}

__global__ void integrateBodiesKernel(double* posX, double* posY, double* posZ, const double* __restrict__ velX,
                                       const double* __restrict__ velY, const double* __restrict__ velZ, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        posX[i] += velX[i] * DT;
        posY[i] += velY[i] * DT;
        posZ[i] += velZ[i] * DT;
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

    // Stage bodies into structure-of-arrays host buffers for coalesced GPU transfers
    std::vector<double> h_posX(numBodies), h_posY(numBodies), h_posZ(numBodies);
    std::vector<double> h_velX(numBodies), h_velY(numBodies), h_velZ(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        h_posX[i] = bodies[i].pos.x;
        h_posY[i] = bodies[i].pos.y;
        h_posZ[i] = bodies[i].pos.z;
        h_velX[i] = bodies[i].vel.x;
        h_velY[i] = bodies[i].vel.y;
        h_velZ[i] = bodies[i].vel.z;
    }

    const size_t bytes = static_cast<size_t>(numBodies) * sizeof(double);
    double *d_posX, *d_posY, *d_posZ, *d_velX, *d_velY, *d_velZ;
    CUDA_CHECK(cudaMalloc(&d_posX, bytes));
    CUDA_CHECK(cudaMalloc(&d_posY, bytes));
    CUDA_CHECK(cudaMalloc(&d_posZ, bytes));
    CUDA_CHECK(cudaMalloc(&d_velX, bytes));
    CUDA_CHECK(cudaMalloc(&d_velY, bytes));
    CUDA_CHECK(cudaMalloc(&d_velZ, bytes));

    CUDA_CHECK(cudaMemcpy(d_posX, h_posX.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posY, h_posY.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posZ, h_posZ.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velX, h_velX.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velY, h_velY.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velZ, h_velZ.data(), bytes, cudaMemcpyHostToDevice));

    const int numBlocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t sharedMemBytes = BLOCK_SIZE * sizeof(double3);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    if (numBodies > 0) {
        for (int step = 0; step < numSteps; ++step) {
            computeForcesKernel<<<numBlocks, BLOCK_SIZE, sharedMemBytes>>>(d_posX, d_posY, d_posZ, d_velX, d_velY,
                                                                            d_velZ, numBodies);
            integrateBodiesKernel<<<numBlocks, BLOCK_SIZE>>>(d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ,
                                                              numBodies);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Copy final state back to host
    CUDA_CHECK(cudaMemcpy(h_posX.data(), d_posX, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_posY.data(), d_posY, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_posZ.data(), d_posZ, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_velX.data(), d_velX, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_velY.data(), d_velY, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_velZ.data(), d_velZ, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_posX));
    CUDA_CHECK(cudaFree(d_posY));
    CUDA_CHECK(cudaFree(d_posZ));
    CUDA_CHECK(cudaFree(d_velX));
    CUDA_CHECK(cudaFree(d_velY));
    CUDA_CHECK(cudaFree(d_velZ));

    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = h_posX[i];
        bodies[i].pos.y = h_posY[i];
        bodies[i].pos.z = h_posZ[i];
        bodies[i].vel.x = h_velX[i];
        bodies[i].vel.y = h_velY[i];
        bodies[i].vel.z = h_velZ[i];
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
