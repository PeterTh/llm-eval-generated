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

// A failed CUDA operation is fatal: this benchmark has no CPU fallback.
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

constexpr int TILE_SIZE = 64;

// Each thread sums one body's interactions in the original j order. All threads,
// including the inactive tail lanes, participate in the shared-memory barriers.
__global__ void advanceBodies(const double4* __restrict__ positions,
                              double4* __restrict__ nextPositions,
                              double4* __restrict__ velocities, size_t n) {
    __shared__ double tileX[TILE_SIZE];
    __shared__ double tileY[TILE_SIZE];
    __shared__ double tileZ[TILE_SIZE];
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const double4 p = active ? positions[i] : make_double4(0, 0, 0, 0);
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (size_t base = 0; base < n; base += TILE_SIZE) {
        const size_t j = base + threadIdx.x;
        const double4 q = j < n ? positions[j] : make_double4(0, 0, 0, 0);
        tileX[threadIdx.x] = q.x;
        tileY[threadIdx.x] = q.y;
        tileZ[threadIdx.x] = q.z;
        __syncthreads();

        const int count = int(n - base < TILE_SIZE ? n - base : TILE_SIZE);
        if (active) {
            #pragma unroll 8
            for (int k = 0; k < count; ++k) {
                const double dx = tileX[k] - p.x;
                const double dy = tileY[k] - p.y;
                const double dz = tileZ[k] - p.z;
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

    if (active) {
        double4 v = velocities[i];
        v.x += DT * Fx;
        v.y += DT * Fy;
        v.z += DT * Fz;
        velocities[i] = v;
        // The next launch reads only this output buffer, so integration cannot
        // change positions still being read by another block in this step.
        nextPositions[i] = make_double4(p.x + v.x * DT, p.y + v.y * DT,
                                       p.z + v.z * DT, 0.0);
    }
}

void simulate(std::vector<Body>& bodies, int numSteps) {
    const size_t n = bodies.size();
    if (n == 0 || numSteps <= 0) return;

    std::vector<double4> state(2 * n);
    for (size_t i = 0; i < n; ++i) {
        const Body& b = bodies[i];
        state[i] = make_double4(b.pos.x, b.pos.y, b.pos.z, 0.0);
        state[n + i] = make_double4(b.vel.x, b.vel.y, b.vel.z, 0.0);
    }
    double4* storage = nullptr;
    const size_t bytes = n * sizeof(double4);
    checkCuda(cudaMalloc(&storage, 3 * bytes), "state allocation");
    double4* positions = storage;
    double4* velocities = storage + n;
    double4* nextPositions = storage + 2 * n;
    checkCuda(cudaMemcpy(storage, state.data(), 2 * bytes, cudaMemcpyHostToDevice),
              "initial state upload");

    const unsigned int blocks = static_cast<unsigned int>((n + TILE_SIZE - 1) / TILE_SIZE);
    for (int step = 0; step < numSteps; ++step) {
        advanceBodies<<<blocks, TILE_SIZE>>>(positions, nextPositions, velocities, n);
        checkCuda(cudaGetLastError(), "simulation kernel launch");
        std::swap(positions, nextPositions);
    }
    // Synchronous copies wait for every step and surface asynchronous errors.
    checkCuda(cudaMemcpy(state.data(), positions, bytes, cudaMemcpyDeviceToHost),
              "position download");
    checkCuda(cudaMemcpy(state.data() + n, velocities, bytes, cudaMemcpyDeviceToHost),
              "velocity download");
    checkCuda(cudaFree(storage), "state release");
    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(state[i].x, state[i].y, state[i].z);
        bodies[i].vel = Vec3(state[n + i].x, state[n + i].y, state[n + i].z);
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
    
    if (numBodies < 0) {
        fprintf(stderr, "Number of bodies must be nonnegative\n");
        return 1;
    }

    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    
    // Initialize the required GPU before timing the simulation, including for
    // empty inputs. Transfers and GPU execution remain inside the timed region.
    checkCuda(cudaSetDevice(0), "device initialization");
    cudaFuncAttributes attributes;
    checkCuda(cudaFuncGetAttributes(&attributes, advanceBodies), "kernel initialization");

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    simulate(bodies, numSteps);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(bodies.size() * 6);
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
