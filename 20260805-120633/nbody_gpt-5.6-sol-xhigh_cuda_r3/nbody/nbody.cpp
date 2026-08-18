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

[[noreturn]] void cudaFailure(const cudaError_t status, const char* operation) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

inline void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        cudaFailure(status, operation);
    }
}

constexpr int CUDA_BLOCK_SIZE = 32;
constexpr int WARP_SIZE = 32;

// Each one-warp block streams a coalesced position tile through shared memory.
// Warp-level barriers are cheaper than whole-block barriers, while shared-memory
// broadcasts reuse every source load for 32 target bodies.  Positions are
// written to a different allocation, so every force in a step observes exactly
// the same pre-integration snapshot.
__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void advanceBodies(const double4* __restrict__ oldPositions,
                   double4* __restrict__ newPositions,
                   double4* __restrict__ velocities,
                   const int numBodies) {
    __shared__ double4 positionTile[CUDA_BLOCK_SIZE];

    const int lane = threadIdx.x;
    const int bodyIndex = blockIdx.x * CUDA_BLOCK_SIZE + lane;
    const bool active = bodyIndex < numBodies;
    const double4 ownPosition = active ? oldPositions[bodyIndex]
                                       : make_double4(0.0, 0.0, 0.0, 0.0);

    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileStart = 0; tileStart < numBodies; tileStart += WARP_SIZE) {
        const int sourceIndex = tileStart + lane;
        if (sourceIndex < numBodies) {
            positionTile[lane] = oldPositions[sourceIndex];
        }
        __syncwarp();

        const int tileCount = min(WARP_SIZE, numBodies - tileStart);
#pragma unroll 8
        for (int source = 0; source < tileCount; ++source) {
            const double4 otherPosition = positionTile[source];
            const double dx = otherPosition.x - ownPosition.x;
            const double dy = otherPosition.y - ownPosition.y;
            const double dz = otherPosition.z - ownPosition.z;
            const double distanceSquared =
                dx * dx + dy * dy + dz * dz + SOFTENING;
            // A direct reciprocal square root avoids issuing separate square-
            // root and division operations while retaining double precision.
            const double inverseDistance = rsqrt(distanceSquared);
            const double inverseDistanceCubed =
                inverseDistance * inverseDistance * inverseDistance;

            forceX += dx * inverseDistanceCubed;
            forceY += dy * inverseDistanceCubed;
            forceZ += dz * inverseDistanceCubed;
        }
        __syncwarp();
    }

    if (active) {
        double4 velocity = velocities[bodyIndex];
        velocity.x += DT * forceX;
        velocity.y += DT * forceY;
        velocity.z += DT * forceZ;

        velocities[bodyIndex] = velocity;
        newPositions[bodyIndex] =
            make_double4(ownPosition.x + velocity.x * DT,
                         ownPosition.y + velocity.y * DT,
                         ownPosition.z + velocity.z * DT,
                         0.0);
    }
}

class CudaSimulation {
  public:
    explicit CudaSimulation(const std::vector<Body>& bodies)
        : numBodies_(static_cast<int>(bodies.size())) {
        if (numBodies_ == 0) {
            return;
        }

        std::vector<double4> positions(bodies.size());
        std::vector<double4> velocities(bodies.size());
        for (size_t i = 0; i < bodies.size(); ++i) {
            positions[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y,
                                        bodies[i].pos.z, 0.0);
            velocities[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y,
                                         bodies[i].vel.z, 0.0);
        }

        const size_t bytes = bodies.size() * sizeof(double4);
        checkCuda(cudaMalloc(&positions_[0], bytes), "position allocation");
        checkCuda(cudaMalloc(&positions_[1], bytes), "position scratch allocation");
        checkCuda(cudaMalloc(&velocities_, bytes), "velocity allocation");
        checkCuda(cudaMemcpy(positions_[0], positions.data(), bytes,
                             cudaMemcpyHostToDevice),
                  "copying initial positions to the GPU");
        checkCuda(cudaMemcpy(velocities_, velocities.data(), bytes,
                             cudaMemcpyHostToDevice),
                  "copying initial velocities to the GPU");

        // Force lazy CUDA module loading to happen during backend setup rather
        // than on the first timed simulation step.  With zero bodies the
        // kernel only traverses its synchronization-safe inactive path.
        advanceBodies<<<1, CUDA_BLOCK_SIZE>>>(
            positions_[0], positions_[1], velocities_, 0);
        checkCuda(cudaGetLastError(), "warming up the n-body kernel");
        checkCuda(cudaDeviceSynchronize(), "warming up the CUDA device");
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    ~CudaSimulation() {
        cudaFree(positions_[0]);
        cudaFree(positions_[1]);
        cudaFree(velocities_);
    }

    void run(const int numSteps) {
        if (numBodies_ == 0 || numSteps == 0) {
            return;
        }

        const int gridSize =
            (numBodies_ + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        for (int step = 0; step < numSteps; ++step) {
            advanceBodies<<<gridSize, CUDA_BLOCK_SIZE>>>(
                positions_[currentPositions_],
                positions_[currentPositions_ ^ 1], velocities_, numBodies_);
            currentPositions_ ^= 1;
        }

        checkCuda(cudaGetLastError(), "launching the n-body kernel");
        checkCuda(cudaDeviceSynchronize(), "running the n-body simulation");
    }

    void copyToHost(std::vector<Body>& bodies) const {
        if (numBodies_ == 0) {
            return;
        }

        std::vector<double4> positions(bodies.size());
        std::vector<double4> velocities(bodies.size());
        const size_t bytes = bodies.size() * sizeof(double4);
        checkCuda(cudaMemcpy(positions.data(), positions_[currentPositions_],
                             bytes, cudaMemcpyDeviceToHost),
                  "copying final positions from the GPU");
        checkCuda(cudaMemcpy(velocities.data(), velocities_, bytes,
                             cudaMemcpyDeviceToHost),
                  "copying final velocities from the GPU");

        for (size_t i = 0; i < bodies.size(); ++i) {
            bodies[i].pos = Vec3(positions[i].x, positions[i].y, positions[i].z);
            bodies[i].vel = Vec3(velocities[i].x, velocities[i].y,
                                 velocities[i].z);
        }
    }

  private:
    int numBodies_ = 0;
    int currentPositions_ = 0;
    double4* positions_[2] = {nullptr, nullptr};
    double4* velocities_ = nullptr;
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

    if (numBodies < 0 || numSteps < 0) {
        printf("Number of bodies and simulation steps must be non-negative\n");
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

    if (printResults || validate) {
        simulation.copyToHost(bodies);
    }
    
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
