#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>

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

// Each component is contiguous so both tile loads and state writes coalesce.
constexpr int TILE = 32;

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void advanceBodies(const double* __restrict__ pos,
                              double* __restrict__ next,
                              double* __restrict__ vel, size_t n) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];
    const int lane = threadIdx.x;
    const size_t i = size_t(blockIdx.x) * TILE + lane;
    const bool active = i < n;
    const double x = active ? pos[i] : 0.0;
    const double y = active ? pos[n + i] : 0.0;
    const double z = active ? pos[2 * n + i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (size_t base = 0; base < n; base += TILE) {
        const size_t j = base + lane;
        if (j < n) {
            sx[lane] = pos[j];
            sy[lane] = pos[n + j];
            sz[lane] = pos[2 * n + j];
        }
        __syncthreads();
        const int count = int(n - base < TILE ? n - base : TILE);
        if (active) {
            // Keep the original ascending summation order and precise FP64 math.
            #pragma unroll 8
            for (int k = 0; k < count; ++k) {
                const double dx = sx[k] - x;
                const double dy = sy[k] - y;
                const double dz = sz[k] - z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = __drcp_rn(__dsqrt_rn(distSqr));
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        // Inactive lanes still participate in both tile barriers.
        __syncthreads();
    }
    if (active) {
        const double vx = vel[i] + DT * fx;
        const double vy = vel[n + i] + DT * fy;
        const double vz = vel[2 * n + i] + DT * fz;
        vel[i] = vx;
        vel[n + i] = vy;
        vel[2 * n + i] = vz;
        // Separate output positions let integration share the force kernel
        // without exposing partially updated positions to other blocks.
        next[i] = x + vx * DT;
        next[n + i] = y + vy * DT;
        next[2 * n + i] = z + vz * DT;
    }
}

__global__ void energyPartials(const double* __restrict__ pos,
                               const double* __restrict__ vel,
                               double* __restrict__ partials, size_t n) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE], sums[TILE];
    const int lane = threadIdx.x;
    const size_t i = size_t(blockIdx.x) * TILE + lane;
    const bool active = i < n;
    const double x = active ? pos[i] : 0.0;
    const double y = active ? pos[n + i] : 0.0;
    const double z = active ? pos[2 * n + i] : 0.0;
    double energy = 0.0;
    if (active) {
        const double vx = vel[i], vy = vel[n + i], vz = vel[2 * n + i];
        energy = 0.5 * (vx * vx + vy * vy + vz * vz);
    }
    for (size_t base = size_t(blockIdx.x) * TILE; base < n; base += TILE) {
        const size_t j = base + lane;
        if (j < n) {
            sx[lane] = pos[j];
            sy[lane] = pos[n + j];
            sz[lane] = pos[2 * n + j];
        }
        __syncthreads();
        const int count = int(n - base < TILE ? n - base : TILE);
        const int first = base > i ? 0 : int(i - base + 1);
        if (active) {
            for (int k = first; k < count; ++k) {
                const double dx = sx[k] - x, dy = sy[k] - y, dz = sz[k] - z;
                energy -= __drcp_rn(__dsqrt_rn(dx * dx + dy * dy + dz * dz + SOFTENING));
            }
        }
        __syncthreads();
    }
    sums[lane] = energy;
    __syncthreads();
    for (int stride = TILE / 2; stride > 0; stride /= 2) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        __syncthreads();
    }
    if (lane == 0) partials[blockIdx.x] = sums[0];
}

class GpuSimulation {
    size_t n;
    unsigned int blocks;
    double *storage = nullptr, *pos = nullptr, *next = nullptr, *vel = nullptr;

public:
    explicit GpuSimulation(const std::vector<Body>& bodies)
        : n(bodies.size()), blocks(static_cast<unsigned int>((n + TILE - 1) / TILE)) {
        // Require CUDA even for an empty or zero-step simulation.
        checkCuda(cudaFree(nullptr), "initialization");
        if (n == 0) return;
        checkCuda(cudaMalloc(&storage, 9 * n * sizeof(double)), "allocation");
        pos = storage;
        next = pos + 3 * n;
        vel = next + 3 * n;
        std::vector<double> state(6 * n);
        for (size_t i = 0; i < n; ++i) {
            state[i] = bodies[i].pos.x;
            state[n + i] = bodies[i].pos.y;
            state[2 * n + i] = bodies[i].pos.z;
            state[3 * n + i] = bodies[i].vel.x;
            state[4 * n + i] = bodies[i].vel.y;
            state[5 * n + i] = bodies[i].vel.z;
        }
        checkCuda(cudaMemcpy(pos, state.data(), 3 * n * sizeof(double), cudaMemcpyHostToDevice), "position upload");
        checkCuda(cudaMemcpy(vel, state.data() + 3 * n, 3 * n * sizeof(double), cudaMemcpyHostToDevice), "velocity upload");
    }

    GpuSimulation(const GpuSimulation&) = delete;
    GpuSimulation& operator=(const GpuSimulation&) = delete;
    ~GpuSimulation() { if (storage) cudaFree(storage); }

    void advance() {
        if (n == 0) return;
        advanceBodies<<<blocks, TILE>>>(pos, next, vel, n);
        checkCuda(cudaGetLastError(), "simulation launch");
        std::swap(pos, next);
    }

    void download(std::vector<Body>& bodies) const {
        if (n == 0) return;
        std::vector<double> state(6 * n);
        checkCuda(cudaMemcpy(state.data(), pos, 3 * n * sizeof(double), cudaMemcpyDeviceToHost), "position download");
        checkCuda(cudaMemcpy(state.data() + 3 * n, vel, 3 * n * sizeof(double), cudaMemcpyDeviceToHost), "velocity download");
        for (size_t i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(state[i], state[n + i], state[2 * n + i]);
            bodies[i].vel = Vec3(state[3 * n + i], state[4 * n + i], state[5 * n + i]);
        }
    }

    double totalEnergy() const {
        if (n == 0) return 0.0;
        // The unused position buffer is large enough for one sum per block.
        energyPartials<<<blocks, TILE>>>(pos, vel, next, n);
        checkCuda(cudaGetLastError(), "energy launch");
        std::vector<double> partials(blocks);
        checkCuda(cudaMemcpy(partials.data(), next, blocks * sizeof(double), cudaMemcpyDeviceToHost), "energy download");
        return kahan_sum(partials);
    }
};

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
    
    if (numBodies < 0) {
        fprintf(stderr, "Number of bodies must be nonnegative\n");
        return 1;
    }

    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    GpuSimulation simulation(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        simulation.advance();
    }
    
    checkCuda(cudaDeviceSynchronize(), "simulation execution");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    simulation.download(bodies);

    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(size_t(numBodies) * 6);
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
            double finalEnergy = simulation.totalEnergy();
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
