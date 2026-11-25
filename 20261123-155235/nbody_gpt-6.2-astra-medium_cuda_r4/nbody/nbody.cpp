#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
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

constexpr unsigned TILE = 64;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__device__ inline double interactionWeight(Vec3 d) {
    const double r2 = fma(d.z, d.z, fma(d.x, d.x, d.y * d.y)) + SOFTENING;
    const double inv = __ddiv_rn(1.0, __dsqrt_rn(r2));
    return inv * inv * inv;
}

__device__ inline Vec3 displacement(Vec3 source, Vec3 target) {
    return {source.x - target.x, source.y - target.y, source.z - target.z};
}

__device__ inline void accumulateForce(Vec3 source, Vec3 target, Vec3& force) {
    const Vec3 d = displacement(source, target);
    const double weight = interactionWeight(d);
    force.x = fma(d.x, weight, force.x);
    force.y = fma(d.y, weight, force.y);
    force.z = fma(d.z, weight, force.z);
}

__device__ inline Body integrate(Body body, Vec3 force) {
    body.vel.x += __dmul_rn(force.x, DT);
    body.vel.y += __dmul_rn(force.y, DT);
    body.vel.z += __dmul_rn(force.z, DT);
    body.pos.x = fma(body.vel.x, DT, body.pos.x);
    body.pos.y = fma(body.vel.y, DT, body.pos.y);
    body.pos.z = fma(body.vel.z, DT, body.pos.z);
    return body;
}

// With few targets, a block cooperates on one body's inverse distances.
// Three lanes then accumulate the components in the original source order.
constexpr unsigned SOURCE_BATCH = 128;

__global__ void advanceSmall(const Body* __restrict__ current, Body* __restrict__ next,
                             unsigned n) {
    __shared__ double delta[SOURCE_BATCH][3];
    __shared__ double weights[SOURCE_BATCH];
    __shared__ double components[3];
    const unsigned lane = threadIdx.x;
    const unsigned i = blockIdx.x;
    const Vec3 target = current[i].pos;
    double force = 0;
    for (unsigned base = 0; base < n; base += SOURCE_BATCH) {
        if (base + lane < n) {
            const Vec3 d = displacement(current[base + lane].pos, target);
            delta[lane][0] = d.x;
            delta[lane][1] = d.y;
            delta[lane][2] = d.z;
            weights[lane] = interactionWeight(d);
        }
        __syncthreads();
        if (lane < 3) {
            const unsigned count = min(SOURCE_BATCH, n - base);
            #pragma unroll 8
            for (unsigned j = 0; j < count; ++j) {
                force = fma(delta[j][lane], weights[j], force);
            }
        }
        __syncthreads();
    }
    if (lane < 3) components[lane] = force;
    __syncthreads();
    if (lane == 0) next[i] = integrate(current[i], {components[0], components[1], components[2]});
}

__global__ void advance(const Body* __restrict__ current, Body* __restrict__ next,
                        unsigned n) {
    __shared__ Vec3 sources[TILE];
    const unsigned lane = threadIdx.x;
    const unsigned i = blockIdx.x * TILE + lane;
    const Vec3 target = i < n ? current[i].pos : Vec3{};
    Vec3 force{};

    // All lanes participate in loading each tile, including the last partial block.
    unsigned base = 0;
    for (; base + TILE <= n; base += TILE) {
        sources[lane] = current[base + lane].pos;
        __syncthreads();
        #pragma unroll 8
        for (unsigned j = 0; j < TILE; ++j) {
            accumulateForce(sources[j], target, force);
        }
        __syncthreads();
    }
    if (base < n) {
        if (base + lane < n) sources[lane] = current[base + lane].pos;
        __syncthreads();
        for (unsigned j = 0; j < n - base; ++j) {
            accumulateForce(sources[j], target, force);
        }
    }

    if (i < n) {
        next[i] = integrate(current[i], force);
    }
}

std::chrono::milliseconds simulate(std::vector<Body>& bodies, int numSteps, bool readBack) {
    if (bodies.empty() || numSteps <= 0) return std::chrono::milliseconds{0};

    const unsigned n = static_cast<unsigned>(bodies.size());
    const size_t bytes = bodies.size() * sizeof(Body);
    Body *current = nullptr, *next = nullptr;
    checkCuda(cudaMalloc(&current, bytes));
    checkCuda(cudaMalloc(&next, bytes));
    checkCuda(cudaMemcpy(current, bodies.data(), bytes, cudaMemcpyHostToDevice));

    const unsigned blocks = (n + TILE - 1) / TILE;
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (n <= 2048) {
            advanceSmall<<<n, SOURCE_BATCH>>>(current, next, n);
        } else {
            advance<<<blocks, TILE>>>(current, next, n);
        }
        checkCuda(cudaGetLastError());
        std::swap(current, next);
    }
    checkCuda(cudaDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();

    if (readBack) checkCuda(cudaMemcpy(bodies.data(), current, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(current));
    checkCuda(cudaFree(next));
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
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
    
    const auto duration = simulate(bodies, numSteps, validate || printResults);
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
