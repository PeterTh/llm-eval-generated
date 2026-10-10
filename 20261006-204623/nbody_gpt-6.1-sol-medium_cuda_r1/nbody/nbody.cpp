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

// Keep positions in separate input/output buffers: every force must use the
// positions at the beginning of the step, including across thread blocks.
// One warp per tile also spreads small simulations across more SMs.
constexpr int TILE_SIZE = 32;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void advanceBodies(const double* __restrict__ positions,
                              double* __restrict__ nextPositions,
                              double* __restrict__ velocities, int n) {
    __shared__ double tileX[TILE_SIZE];
    __shared__ double tileY[TILE_SIZE];
    __shared__ double tileZ[TILE_SIZE];
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = size_t(n);
    const bool active = i < count;
    const double x = active ? positions[i] : 0.0;
    const double y = active ? positions[count + i] : 0.0;
    const double z = active ? positions[2 * count + i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (size_t base = 0; base < count; base += TILE_SIZE) {
        const size_t j = base + threadIdx.x;
        if (j < count) {
            tileX[threadIdx.x] = positions[j];
            tileY[threadIdx.x] = positions[count + j];
            tileZ[threadIdx.x] = positions[2 * count + j];
        }
        __syncthreads();
        const int tileCount = int(count - base < TILE_SIZE ? count - base : TILE_SIZE);
        if (active) {
            // Preserve the original source-body summation order and precision.
            #pragma unroll 8
            for (int j = 0; j < tileCount; ++j) {
                const double dx = tileX[j] - x;
                const double dy = tileY[j] - y;
                const double dz = tileZ[j] - z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        // Inactive lanes also participate before the next tile is loaded.
        __syncthreads();
    }
    if (active) {
        const double vx = velocities[i] + DT * fx;
        const double vy = velocities[count + i] + DT * fy;
        const double vz = velocities[2 * count + i] + DT * fz;
        velocities[i] = vx;
        velocities[count + i] = vy;
        velocities[2 * count + i] = vz;
        nextPositions[i] = x + vx * DT;
        nextPositions[count + i] = y + vy * DT;
        nextPositions[2 * count + i] = z + vz * DT;
    }
}

// A single allocation and coalesced structure-of-arrays storage. Transfers and
// allocation happen only once, outside the timed simulation loop.
class DeviceBodies {
public:
    explicit DeviceBodies(const std::vector<Body>& bodies) : n(bodies.size()) {
        checkCuda(cudaFree(nullptr)); // Initialize CUDA even for an empty input.
        if (n == 0) return;
        std::vector<double> data(6 * n);
        for (size_t i = 0; i < n; ++i) {
            data[i] = bodies[i].pos.x;
            data[n + i] = bodies[i].pos.y;
            data[2 * n + i] = bodies[i].pos.z;
            data[3 * n + i] = bodies[i].vel.x;
            data[4 * n + i] = bodies[i].vel.y;
            data[5 * n + i] = bodies[i].vel.z;
        }
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&storage), 9 * n * sizeof(double)));
        positions = storage;
        velocities = storage + 3 * n;
        nextPositions = storage + 6 * n;
        checkCuda(cudaMemcpy(storage, data.data(), data.size() * sizeof(double), cudaMemcpyHostToDevice));
    }

    ~DeviceBodies() { if (storage) cudaFree(storage); }
    DeviceBodies(const DeviceBodies&) = delete;
    DeviceBodies& operator=(const DeviceBodies&) = delete;

    void advance() {
        if (n == 0) return;
        advanceBodies<<<(n + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(
            positions, nextPositions, velocities, int(n));
        checkCuda(cudaGetLastError());
        std::swap(positions, nextPositions);
    }

    void download(std::vector<Body>& bodies) const {
        if (n == 0) return;
        std::vector<double> data(6 * n);
        checkCuda(cudaMemcpy(data.data(), positions, 3 * n * sizeof(double), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(data.data() + 3 * n, velocities, 3 * n * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(data[i], data[n + i], data[2 * n + i]);
            bodies[i].vel = Vec3(data[3 * n + i], data[4 * n + i], data[5 * n + i]);
        }
    }

private:
    size_t n;
    double* storage = nullptr;
    double* positions = nullptr;
    double* nextPositions = nullptr;
    double* velocities = nullptr;
};

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
    
    DeviceBodies deviceBodies(bodies);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        deviceBodies.advance();
    }
    
    checkCuda(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    deviceBodies.download(bodies);

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
