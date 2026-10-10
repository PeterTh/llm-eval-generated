#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each thread owns one target body. Source positions are loaded once per
// block into shared memory, then reused by every target in that block.
template <int TileSize>
__global__ void advanceBodies(const double* __restrict__ positions,
                              double* __restrict__ nextPositions,
                              double* __restrict__ velocities, size_t n) {
    __shared__ double sourceX[TileSize];
    __shared__ double sourceY[TileSize];
    __shared__ double sourceZ[TileSize];
    const size_t i = size_t(blockIdx.x) * TileSize + threadIdx.x;
    const bool active = i < n;
    const double x = active ? positions[i] : 0.0;
    const double y = active ? positions[n + i] : 0.0;
    const double z = active ? positions[2 * n + i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (size_t base = 0; base < n; base += TileSize) {
        const size_t j = base + threadIdx.x;
        if (j < n) {
            sourceX[threadIdx.x] = positions[j];
            sourceY[threadIdx.x] = positions[n + j];
            sourceZ[threadIdx.x] = positions[2 * n + j];
        }
        // Inactive threads must also participate in both barriers.
        __syncthreads();
        const int count = int(n - base < TileSize ? n - base : TileSize);
        if (active) {
            #pragma unroll 4
            for (int k = 0; k < count; ++k) {
                const double dx = sourceX[k] - x;
                const double dy = sourceY[k] - y;
                const double dz = sourceZ[k] - z;
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
        const double vx = velocities[i] + DT * fx;
        const double vy = velocities[n + i] + DT * fy;
        const double vz = velocities[2 * n + i] + DT * fz;
        velocities[i] = vx;
        velocities[n + i] = vy;
        velocities[2 * n + i] = vz;
        // Separate output storage prevents other blocks from seeing positions
        // from the next step while they are still calculating current forces.
        nextPositions[i] = x + vx * DT;
        nextPositions[n + i] = y + vy * DT;
        nextPositions[2 * n + i] = z + vz * DT;
    }
}

class CudaSimulation {
    double* storage = nullptr;
    double* positions = nullptr;
    double* nextPositions = nullptr;
    double* velocities = nullptr;
    size_t n;

public:
    explicit CudaSimulation(const std::vector<Body>& bodies) : n(bodies.size()) {
        // Initialize CUDA even for empty input; there is no CPU fallback.
        checkCuda(cudaFree(nullptr), "initializing the CUDA device");
        // Resolve kernels before timing so lazy module loading is setup cost.
        cudaFuncAttributes attributes;
        checkCuda(cudaFuncGetAttributes(&attributes, advanceBodies<32>), "loading simulation kernel");
        checkCuda(cudaFuncGetAttributes(&attributes, advanceBodies<64>), "loading simulation kernel");
        checkCuda(cudaFuncGetAttributes(&attributes, advanceBodies<128>), "loading simulation kernel");
        if (n == 0) return;
        checkCuda(cudaMalloc(&storage, 9 * n * sizeof(double)), "allocating body state");
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
        checkCuda(cudaMemcpy(positions, packed.data(), 3 * n * sizeof(double),
                             cudaMemcpyHostToDevice), "uploading positions");
        checkCuda(cudaMemcpy(velocities, packed.data() + 3 * n, 3 * n * sizeof(double),
                             cudaMemcpyHostToDevice), "uploading velocities");
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;
    ~CudaSimulation() { if (storage) cudaFree(storage); }

    void advance(int steps) {
        for (int step = 0; step < steps && n != 0; ++step) {
            // Smaller blocks expose more parallelism for small populations.
            if (n < 4096) {
                advanceBodies<32><<<unsigned((n + 31) / 32), 32>>>(
                    positions, nextPositions, velocities, n);
            } else if (n < 8192) {
                advanceBodies<64><<<unsigned((n + 63) / 64), 64>>>(
                    positions, nextPositions, velocities, n);
            } else {
                advanceBodies<128><<<unsigned((n + 127) / 128), 128>>>(
                    positions, nextPositions, velocities, n);
            }
            std::swap(positions, nextPositions);
        }
        checkCuda(cudaGetLastError(), "launching simulation kernels");
        checkCuda(cudaDeviceSynchronize(), "executing simulation kernels");
    }

    void download(std::vector<Body>& bodies) const {
        if (n == 0) return;
        std::vector<double> packed(6 * n);
        checkCuda(cudaMemcpy(packed.data(), positions, 3 * n * sizeof(double),
                             cudaMemcpyDeviceToHost), "downloading positions");
        checkCuda(cudaMemcpy(packed.data() + 3 * n, velocities, 3 * n * sizeof(double),
                             cudaMemcpyDeviceToHost), "downloading velocities");
        for (size_t i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(packed[i], packed[n + i], packed[2 * n + i]);
            bodies[i].vel = Vec3(packed[3 * n + i], packed[4 * n + i], packed[5 * n + i]);
        }
    }
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
    
    CudaSimulation simulation(bodies);

    // Run simulation with state resident on the GPU.
    auto start = std::chrono::high_resolution_clock::now();
    
    simulation.advance(numSteps);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %lld ms\n", static_cast<long long>(duration.count()));
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
