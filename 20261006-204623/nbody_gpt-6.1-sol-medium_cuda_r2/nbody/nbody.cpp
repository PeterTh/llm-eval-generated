#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
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

// CUDA errors are fatal: the benchmark always runs its simulation on the GPU.
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

constexpr int TILE_SIZE = 32;

// Positions are double buffered. Every force reads the previous step's state,
// while velocity and the next position are computed together without another
// kernel launch. The planes provide coalesced global loads and stores.
template <int BLOCK_SIZE>
__global__ void advanceBodies(const double* __restrict__ positions,
                              double* __restrict__ nextPositions,
                              double* __restrict__ velocities,
                              size_t n, size_t stride) {
    __shared__ double tileX[TILE_SIZE];
    __shared__ double tileY[TILE_SIZE];
    __shared__ double tileZ[TILE_SIZE];
    const int lane = threadIdx.x;
    const size_t i = size_t(blockIdx.x) * BLOCK_SIZE + lane;
    const bool active = i < n;
    const double x = active ? positions[i] : 0.0;
    const double y = active ? positions[stride + i] : 0.0;
    const double z = active ? positions[2 * stride + i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (size_t base = 0; base < n; base += TILE_SIZE) {
        #pragma unroll
        for (int offset = lane; offset < TILE_SIZE; offset += BLOCK_SIZE) {
            const size_t j = base + offset;
            if (j < n) {
                tileX[offset] = positions[j];
                tileY[offset] = positions[stride + j];
                tileZ[offset] = positions[2 * stride + j];
            }
        }
        __syncthreads();
        const int count = int(n - base < TILE_SIZE ? n - base : TILE_SIZE);
        if (active) {
            // Keep the original j order and double precision arithmetic.
            #pragma unroll 4
            for (int k = 0; k < count; ++k) {
                const double dx = tileX[k] - x;
                const double dy = tileY[k] - y;
                const double dz = tileZ[k] - z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = __drcp_rn(__dsqrt_rn(distSqr));
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }
    if (active) {
        const double vx = velocities[i] + DT * fx;
        const double vy = velocities[stride + i] + DT * fy;
        const double vz = velocities[2 * stride + i] + DT * fz;
        velocities[i] = vx;
        velocities[stride + i] = vy;
        velocities[2 * stride + i] = vz;
        nextPositions[i] = x + vx * DT;
        nextPositions[stride + i] = y + vy * DT;
        nextPositions[2 * stride + i] = z + vz * DT;
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
    
    // Allocate once and retain all state on the GPU throughout the simulation.
    const size_t n = bodies.size();
    const size_t stride = std::max(size_t(TILE_SIZE),
                                   ((n + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE);
    const size_t planeBytes = 3 * stride * sizeof(double);
    std::vector<double> state(6 * stride, 0.0);
    for (size_t i = 0; i < n; ++i) {
        state[i] = bodies[i].pos.x;
        state[stride + i] = bodies[i].pos.y;
        state[2 * stride + i] = bodies[i].pos.z;
        state[3 * stride + i] = bodies[i].vel.x;
        state[4 * stride + i] = bodies[i].vel.y;
        state[5 * stride + i] = bodies[i].vel.z;
    }
    double* deviceState = nullptr;
    checkCuda(cudaMalloc(&deviceState, 3 * planeBytes), "allocation");
    double* positions = deviceState;
    double* velocities = deviceState + 3 * stride;
    double* nextPositions = deviceState + 6 * stride;
    checkCuda(cudaMemcpy(deviceState, state.data(), 2 * planeBytes,
                         cudaMemcpyHostToDevice), "initial state upload");

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps && n != 0; ++step) {
        advanceBodies<TILE_SIZE><<<(n + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(
            positions, nextPositions, velocities, n, stride);
        std::swap(positions, nextPositions);
    }
    checkCuda(cudaGetLastError(), "simulation launch");
    checkCuda(cudaDeviceSynchronize(), "simulation execution");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    checkCuda(cudaMemcpy(state.data(), positions, planeBytes,
                         cudaMemcpyDeviceToHost), "position download");
    checkCuda(cudaMemcpy(state.data() + 3 * stride, velocities, planeBytes,
                         cudaMemcpyDeviceToHost), "velocity download");
    checkCuda(cudaFree(deviceState), "deallocation");
    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(state[i], state[stride + i], state[2 * stride + i]);
        bodies[i].vel = Vec3(state[3 * stride + i], state[4 * stride + i], state[5 * stride + i]);
    }

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
