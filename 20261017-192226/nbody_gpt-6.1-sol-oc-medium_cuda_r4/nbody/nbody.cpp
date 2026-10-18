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

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

constexpr int BLOCK_SIZE = 32;
constexpr int TILE_SIZE = 128;

// Each thread owns one target body. Source positions are loaded once per
// block into shared memory and consumed in their original summation order.
// Separate input/output positions let force and integration share a launch
// without exposing new positions to other blocks during the same step.
__global__ void advanceBodies(const double* __restrict__ pos,
                              double* __restrict__ next,
                              double* __restrict__ vel, size_t n) {
    __shared__ double sx[TILE_SIZE], sy[TILE_SIZE], sz[TILE_SIZE];
    const unsigned int lane = threadIdx.x;
    const size_t i = size_t(blockIdx.x) * BLOCK_SIZE + lane;
    const bool active = i < n;
    const double x = active ? pos[i] : 0.0;
    const double y = active ? pos[n + i] : 0.0;
    const double z = active ? pos[2 * n + i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (size_t base = 0; base < n; base += TILE_SIZE) {
        #pragma unroll
        for (int offset = lane; offset < TILE_SIZE; offset += BLOCK_SIZE) {
            const size_t source = base + offset;
            if (source < n) {
                sx[offset] = pos[source];
                sy[offset] = pos[n + source];
                sz[offset] = pos[2 * n + source];
            }
        }
        __syncthreads();
        const int count = n - base < TILE_SIZE ? int(n - base) : TILE_SIZE;
        if (active) {
            #pragma unroll 8
            for (int j = 0; j < count; ++j) {
                const double dx = sx[j] - x;
                const double dy = sy[j] - y;
                const double dz = sz[j] - z;
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
        const double vx = vel[i] + DT * fx;
        const double vy = vel[n + i] + DT * fy;
        const double vz = vel[2 * n + i] + DT * fz;
        vel[i] = vx;
        vel[n + i] = vy;
        vel[2 * n + i] = vz;
        next[i] = x + vx * DT;
        next[n + i] = y + vy * DT;
        next[2 * n + i] = z + vz * DT;
    }
}

// Validation's quadratic potential-energy calculation also runs on the GPU.
// Only one partial sum per body is transferred for the final host reduction.
__global__ void computeEnergy(const double* __restrict__ pos,
                             const double* __restrict__ vel,
                             double* __restrict__ energy, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double x = pos[i], y = pos[n + i], z = pos[2 * n + i];
    const double vx = vel[i], vy = vel[n + i], vz = vel[2 * n + i];
    double potential = 0.0;
    for (size_t j = i + 1; j < n; ++j) {
        const double dx = pos[j] - x;
        const double dy = pos[n + j] - y;
        const double dz = pos[2 * n + j] - z;
        potential -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
    }
    energy[i] = 0.5 * (vx * vx + vy * vy + vz * vz) + potential;
}

class CudaSimulation {
    size_t n;
    double *pos = nullptr, *next = nullptr, *vel = nullptr;

public:
    explicit CudaSimulation(const std::vector<Body>& bodies) : n(bodies.size()) {
        // Require CUDA even for an empty simulation; there is no CPU fallback.
        checkCuda(cudaFree(nullptr));
        if (n == 0) return;
        const size_t bytes = 3 * n * sizeof(double);
        checkCuda(cudaMalloc(&pos, bytes));
        checkCuda(cudaMalloc(&next, bytes));
        checkCuda(cudaMalloc(&vel, bytes));
        std::vector<double> positions(3 * n), velocities(3 * n);
        for (size_t i = 0; i < n; ++i) {
            positions[i] = bodies[i].pos.x;
            positions[n + i] = bodies[i].pos.y;
            positions[2 * n + i] = bodies[i].pos.z;
            velocities[i] = bodies[i].vel.x;
            velocities[n + i] = bodies[i].vel.y;
            velocities[2 * n + i] = bodies[i].vel.z;
        }
        checkCuda(cudaMemcpy(pos, positions.data(), bytes, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(vel, velocities.data(), bytes, cudaMemcpyHostToDevice));
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    ~CudaSimulation() {
        cudaFree(vel);
        cudaFree(next);
        cudaFree(pos);
    }

    void run(int steps) {
        if (n == 0) return;
        const unsigned int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int step = 0; step < steps; ++step) {
            advanceBodies<<<blocks, BLOCK_SIZE>>>(pos, next, vel, n);
            checkCuda(cudaGetLastError());
            std::swap(pos, next);
        }
        // Include all GPU work in the reported simulation time.
        checkCuda(cudaDeviceSynchronize());
    }

    void download(std::vector<Body>& bodies) const {
        if (n == 0) return;
        const size_t bytes = 3 * n * sizeof(double);
        std::vector<double> positions(3 * n), velocities(3 * n);
        checkCuda(cudaMemcpy(positions.data(), pos, bytes, cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(velocities.data(), vel, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(positions[i], positions[n + i], positions[2 * n + i]);
            bodies[i].vel = Vec3(velocities[i], velocities[n + i], velocities[2 * n + i]);
        }
    }

    double totalEnergy() const {
        if (n == 0) return 0.0;
        // The unused position buffer is scratch space after the final step.
        computeEnergy<<<(n + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(pos, vel, next, n);
        checkCuda(cudaGetLastError());
        std::vector<double> partial(n);
        checkCuda(cudaMemcpy(partial.data(), next, n * sizeof(double), cudaMemcpyDeviceToHost));
        double energy = 0.0;
        for (double value : partial) energy += value;
        return energy;
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
    CudaSimulation simulation(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    simulation.run(numSteps);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    if (printResults || validate) simulation.download(bodies);
    
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
