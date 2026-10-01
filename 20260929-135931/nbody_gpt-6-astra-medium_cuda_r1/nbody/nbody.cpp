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

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

constexpr int TILE_SIZE = 128;

// Each lane owns one body. Positions are read from an immutable snapshot;
// writing the next snapshot lets integration share the force kernel safely.
template <int TileSize>
__global__ void advanceBodies(const double* __restrict__ positions,
                              double* __restrict__ nextPositions,
                              double* __restrict__ velocities, size_t n) {
    __shared__ double tileX[TileSize], tileY[TileSize], tileZ[TileSize];
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x;
    const bool active = i < n;
    const double x = active ? positions[i] : 0.0;
    const double y = active ? positions[n + i] : 0.0;
    const double z = active ? positions[2 * n + i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (size_t base = 0; base < n; base += TileSize) {
        const size_t j = base + lane;
        if (j < n) {
            tileX[lane] = positions[j];
            tileY[lane] = positions[n + j];
            tileZ[lane] = positions[2 * n + j];
        }
        __syncthreads();
        const int count = int(n - base < TileSize ? n - base : TileSize);
        if (active) {
            // Keep the original summation order, including the self interaction.
            #pragma unroll 4
            for (int k = 0; k < count; ++k) {
                const double dx = tileX[k] - x;
                const double dy = tileY[k] - y;
                const double dz = tileZ[k] - z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        // Inactive lanes must participate, including in the final partial tile.
        __syncthreads();
    }
    if (active) {
        const double vx = velocities[i] + DT * fx;
        const double vy = velocities[n + i] + DT * fy;
        const double vz = velocities[2 * n + i] + DT * fz;
        velocities[i] = vx;
        velocities[n + i] = vy;
        velocities[2 * n + i] = vz;
        nextPositions[i] = x + vx * DT;
        nextPositions[n + i] = y + vy * DT;
        nextPositions[2 * n + i] = z + vz * DT;
    }
}

__global__ void bodyEnergies(const double* __restrict__ positions,
                             const double* __restrict__ velocities,
                             double* __restrict__ energies, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double x = positions[i], y = positions[n + i], z = positions[2 * n + i];
    const double vx = velocities[i], vy = velocities[n + i], vz = velocities[2 * n + i];
    double energy = 0.5 * (vx * vx + vy * vy + vz * vz);
    for (size_t j = i + 1; j < n; ++j) {
        const double dx = positions[j] - x;
        const double dy = positions[n + j] - y;
        const double dz = positions[2 * n + j] - z;
        energy -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
    }
    energies[i] = energy;
}

class DeviceSimulation {
    size_t n;
    double* storage = nullptr;
    double *positions, *nextPositions, *velocities, *energies;

public:
    explicit DeviceSimulation(const std::vector<Body>& bodies) : n(bodies.size()) {
        // CUDA is required even for an empty simulation; there is no CPU fallback.
        checkCuda(cudaMalloc(&storage, std::max(size_t(1), 10 * n) * sizeof(double)), "allocate bodies");
        positions = storage;
        velocities = storage + 3 * n;
        nextPositions = storage + 6 * n;
        energies = storage + 9 * n;
        std::vector<double> packed(6 * n);
        for (size_t i = 0; i < n; ++i) {
            packed[i] = bodies[i].pos.x;
            packed[n + i] = bodies[i].pos.y;
            packed[2 * n + i] = bodies[i].pos.z;
            packed[3 * n + i] = bodies[i].vel.x;
            packed[4 * n + i] = bodies[i].vel.y;
            packed[5 * n + i] = bodies[i].vel.z;
        }
        if (n) checkCuda(cudaMemcpy(storage, packed.data(), packed.size() * sizeof(double),
                                    cudaMemcpyHostToDevice), "upload bodies");
    }

    DeviceSimulation(const DeviceSimulation&) = delete;
    DeviceSimulation& operator=(const DeviceSimulation&) = delete;
    ~DeviceSimulation() { cudaFree(storage); }

    void run(int steps) {
        if (n) {
            const int threads = n <= 8192 ? 32 : TILE_SIZE;
            const unsigned int blocks = (n + threads - 1) / threads;
            for (int step = 0; step < steps; ++step) {
                // Smaller tiles expose more independent blocks for small systems.
                if (threads == 32)
                    advanceBodies<32><<<blocks, 32>>>(positions, nextPositions, velocities, n);
                else
                    advanceBodies<TILE_SIZE><<<blocks, TILE_SIZE>>>(positions, nextPositions, velocities, n);
                checkCuda(cudaGetLastError(), "advance bodies");
                std::swap(positions, nextPositions);
            }
        }
        // Include actual GPU execution, not just kernel submission, in the timer.
        checkCuda(cudaDeviceSynchronize(), "finish simulation");
    }

    void download(std::vector<Body>& bodies) const {
        if (!n) return;
        std::vector<double> packed(6 * n);
        checkCuda(cudaMemcpy(packed.data(), positions, 3 * n * sizeof(double),
                             cudaMemcpyDeviceToHost), "download positions");
        checkCuda(cudaMemcpy(packed.data() + 3 * n, velocities, 3 * n * sizeof(double),
                             cudaMemcpyDeviceToHost), "download velocities");
        for (size_t i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(packed[i], packed[n + i], packed[2 * n + i]);
            bodies[i].vel = Vec3(packed[3 * n + i], packed[4 * n + i], packed[5 * n + i]);
        }
    }

    double totalEnergy() const {
        if (!n) return 0.0;
        bodyEnergies<<<(n + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(positions, velocities, energies, n);
        checkCuda(cudaGetLastError(), "compute energy");
        std::vector<double> partial(n);
        checkCuda(cudaMemcpy(partial.data(), energies, n * sizeof(double),
                             cudaMemcpyDeviceToHost), "download energy");
        // Only the O(n) final reduction is on the host; all pair work runs on CUDA.
        return kahan_sum(partial);
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
    
    DeviceSimulation simulation(bodies);

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
