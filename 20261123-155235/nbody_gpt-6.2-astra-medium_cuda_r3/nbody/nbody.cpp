#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

constexpr unsigned TILE = 64;

__device__ double inverseDistance(Vec3 d) {
    const double xy = fma(d.x, d.x, d.y * d.y);
    const double squared = fma(d.z, d.z, xy) + SOFTENING;
    return __ddiv_rn(1.0, __dsqrt_rn(squared));
}

__device__ Vec3 displacement(Vec3 source, Vec3 target) {
    return {source.x - target.x, source.y - target.y, source.z - target.z};
}

// Small systems have too few targets to fill the GPU. Evaluate their
// expensive pair weights independently, then consume them in source order.
__global__ void pairWeights(const Body* bodies, double* weights, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        const double inv = inverseDistance(displacement(bodies[j].pos, bodies[i].pos));
        weights[size_t(j) * n + i] = inv * inv * inv;
    }
}

template<bool cachedWeights>
__global__ void advance(const Body* __restrict__ current, Body* __restrict__ next,
                        const double* __restrict__ weights, unsigned n) {
    __shared__ Vec3 sources[TILE];
    const unsigned lane = threadIdx.x;
    const unsigned i = blockIdx.x * TILE + lane;
    const Vec3 pos = i < n ? current[i].pos : Vec3{};
    Vec3 force{};
    for (unsigned base = 0; base < n; base += TILE) {
        if (base + lane < n) sources[lane] = current[base + lane].pos;
        __syncthreads();
        const unsigned count = min(TILE, n - base);
        if (i < n) {
            #pragma unroll 4
            for (unsigned k = 0; k < count; ++k) {
                const Vec3 d = displacement(sources[k], pos);
                double weight;
                if constexpr (cachedWeights) {
                    weight = weights[size_t(base + k) * n + i];
                } else {
                    const double inv = inverseDistance(d);
                    weight = inv * inv * inv;
                }
                force.x = fma(d.x, weight, force.x);
                force.y = fma(d.y, weight, force.y);
                force.z = fma(d.z, weight, force.z);
            }
        }
        __syncthreads();
    }

    if (i < n) {
        const Vec3 old = current[i].vel;
        const Vec3 vel = {old.x + __dmul_rn(DT, force.x),
                          old.y + __dmul_rn(DT, force.y),
                          old.z + __dmul_rn(DT, force.z)};
        next[i] = {{fma(vel.x, DT, pos.x), fma(vel.y, DT, pos.y), fma(vel.z, DT, pos.z)}, vel};
    }
}

__global__ void energyRows(const Body* bodies, double* rows, unsigned n) {
    const unsigned i = blockIdx.x;
    const Vec3 pos = bodies[i].pos;
    double sum = 0.0;
    for (unsigned j = i + 1 + threadIdx.x; j < n; j += blockDim.x) {
        sum -= inverseDistance(displacement(bodies[j].pos, pos));
    }
    for (int offset = 16; offset; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ double warps[4];
    if ((threadIdx.x % 32) == 0) warps[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        const Vec3 v = bodies[i].vel;
        rows[i] = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z)
                  + ((warps[0] + warps[1]) + (warps[2] + warps[3]));
    }
}

double computeTotalEnergy(const Body* bodies, unsigned n) {
    if (n == 0) return 0.0;

    std::vector<double> hostRows(n);
    double* rows;
    const size_t bytes = hostRows.size() * sizeof(double);
    checkCuda(cudaMalloc(&rows, bytes), "Allocate energy rows");
    energyRows<<<n, 128>>>(bodies, rows, n);
    checkCuda(cudaGetLastError(), "Launch energy calculation");
    checkCuda(cudaMemcpy(hostRows.data(), rows, bytes, cudaMemcpyDeviceToHost), "Download energy rows");
    checkCuda(cudaFree(rows), "Free energy rows");

    double total = 0.0;
    for (double value : hostRows) total += value;
    return total;
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
    
    Body *current = nullptr, *next = nullptr;
    double* weights = nullptr;
    const size_t bytes = bodies.size() * sizeof(Body);
    const unsigned n = static_cast<unsigned>(numBodies);
    const unsigned blocks = (n + TILE - 1) / TILE;
    if (n) {
        checkCuda(cudaMalloc(&current, bytes), "Allocate body state");
        checkCuda(cudaMemcpy(current, bodies.data(), bytes, cudaMemcpyHostToDevice), "Upload bodies");
        if (numSteps > 0) {
            checkCuda(cudaMalloc(&next, bytes), "Allocate next body state");
            if (n <= 2048)
                checkCuda(cudaMalloc(&weights, size_t(n) * n * sizeof(double)), "Allocate pair weights");
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; n && step < numSteps; ++step) {
        if (weights) {
            const dim3 threads(32, 8);
            const dim3 grid((n + 31) / 32, (n + 7) / 8);
            pairWeights<<<grid, threads>>>(current, weights, n);
            checkCuda(cudaGetLastError(), "Launch pair weights");
            advance<true><<<blocks, TILE>>>(current, next, weights, n);
        } else {
            advance<false><<<blocks, TILE>>>(current, next, nullptr, n);
        }
        checkCuda(cudaGetLastError(), "Launch simulation step");
        std::swap(current, next);
    }
    checkCuda(cudaDeviceSynchronize(), "Finish simulation");
    auto end = std::chrono::high_resolution_clock::now();
    if ((printResults || validate) && n)
        checkCuda(cudaMemcpy(bodies.data(), current, bytes, cudaMemcpyDeviceToHost), "Download bodies");

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
    int result = 0;
    if (validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            const double finalEnergy = computeTotalEnergy(current, n);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }

    checkCuda(cudaFree(weights), "Free pair weights");
    checkCuda(cudaFree(next), "Free next body state");
    checkCuda(cudaFree(current), "Free body state");
    return result;
}
