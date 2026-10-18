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

static void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

constexpr int TILE_SIZE = 32;

// Structure-of-arrays storage gives coalesced loads and stores. Each thread
// accumulates one body's interactions in the original increasing-j order.
// Separate position buffers allow integration to be fused with force evaluation
// without letting any block observe partially updated positions.
template <bool Energy>
__global__ void evaluateBodies(const double* __restrict__ positions,
                               double* __restrict__ velocities,
                               double* __restrict__ output, size_t n) {
    __shared__ double tile[3][TILE_SIZE];
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const double x = active ? positions[i] : 0.0;
    const double y = active ? positions[n + i] : 0.0;
    const double z = active ? positions[2 * n + i] : 0.0;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    // All threads, including those in a partial final block, reach each barrier.
    for (size_t base = 0; base < n; base += TILE_SIZE) {
        if constexpr (Energy) {
            if (base + TILE_SIZE <= size_t(blockIdx.x) * blockDim.x) continue;
        }
        const size_t j = base + threadIdx.x;
        if (j < n) {
            tile[0][threadIdx.x] = positions[j];
            tile[1][threadIdx.x] = positions[n + j];
            tile[2][threadIdx.x] = positions[2 * n + j];
        }
        __syncthreads();
        const int count = n - base < TILE_SIZE ? int(n - base) : TILE_SIZE;
        if (active) {
            #pragma unroll 4
            for (int k = 0; k < count; ++k) {
                if (Energy && base + k <= i) continue;
                const double dx = tile[0][k] - x;
                const double dy = tile[1][k] - y;
                const double dz = tile[2][k] - z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = rsqrt(distSqr);
                if constexpr (Energy) {
                    Fx -= invDist;
                } else {
                    const double invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            }
        }
        __syncthreads();
    }
    if (active) {
        double vx = velocities[i];
        double vy = velocities[n + i];
        double vz = velocities[2 * n + i];
        if constexpr (Energy) {
            output[i] = 0.5 * (vx * vx + vy * vy + vz * vz) + Fx;
        } else {
            vx += DT * Fx;
            vy += DT * Fy;
            vz += DT * Fz;
            velocities[i] = vx;
            velocities[n + i] = vy;
            velocities[2 * n + i] = vz;
            output[i] = x + vx * DT;
            output[n + i] = y + vy * DT;
            output[2 * n + i] = z + vz * DT;
        }
    }
}

class GpuSimulation {
    size_t n;
    double *storage = nullptr, *positions = nullptr, *nextPositions = nullptr, *velocities = nullptr;
public:
    explicit GpuSimulation(const std::vector<Body>& bodies) : n(bodies.size()) {
        // Initialize CUDA even for an empty simulation: there is no CPU fallback.
        checkCuda(cudaFree(nullptr));
        if (n == 0) return;
        checkCuda(cudaMalloc(&storage, 9 * n * sizeof(double)));
        positions = storage;
        nextPositions = storage + 3 * n;
        velocities = storage + 6 * n;
        std::vector<double> packed(6 * n);
        for (size_t i = 0; i < n; ++i) {
            packed[i] = bodies[i].pos.x;
            packed[n + i] = bodies[i].pos.y;
            packed[2 * n + i] = bodies[i].pos.z;
            packed[3 * n + i] = bodies[i].vel.x;
            packed[4 * n + i] = bodies[i].vel.y;
            packed[5 * n + i] = bodies[i].vel.z;
        }
        checkCuda(cudaMemcpy(positions, packed.data(), 3 * n * sizeof(double), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(velocities, packed.data() + 3 * n, 3 * n * sizeof(double), cudaMemcpyHostToDevice));
        // Load the kernel before timing (CUDA may lazily load device code).
        cudaFuncAttributes attributes;
        checkCuda(cudaFuncGetAttributes(&attributes, evaluateBodies<false>));
    }
    ~GpuSimulation() { if (storage) cudaFree(storage); }
    GpuSimulation(const GpuSimulation&) = delete;
    GpuSimulation& operator=(const GpuSimulation&) = delete;

    void step() {
        if (n == 0) return;
        evaluateBodies<false><<<(n + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(
            positions, velocities, nextPositions, n);
        checkCuda(cudaGetLastError());
        std::swap(positions, nextPositions);
    }

    void download(std::vector<Body>& bodies) const {
        if (n == 0) return;
        std::vector<double> packed(6 * n);
        checkCuda(cudaMemcpy(packed.data(), positions, 3 * n * sizeof(double), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(packed.data() + 3 * n, velocities, 3 * n * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(packed[i], packed[n + i], packed[2 * n + i]);
            bodies[i].vel = Vec3(packed[3 * n + i], packed[4 * n + i], packed[5 * n + i]);
        }
    }

    double totalEnergy() const {
        if (n == 0) return 0.0;
        // Reuse the unused position buffer for per-body energies.
        evaluateBodies<true><<<(n + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(
            positions, velocities, nextPositions, n);
        checkCuda(cudaGetLastError());
        std::vector<double> energies(n);
        checkCuda(cudaMemcpy(energies.data(), nextPositions, n * sizeof(double), cudaMemcpyDeviceToHost));
        return kahan_sum(energies);
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
    
    if (numBodies < 0 || numSteps < 0) {
        fprintf(stderr, "Number of bodies and steps must be nonnegative\n");
        return 1;
    }

    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    GpuSimulation simulation(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        simulation.step();
    }
    checkCuda(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    simulation.download(bodies);
    
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
