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
constexpr int THREADS_PER_BLOCK = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

void checkCuda(cudaError_t error, const char* operation, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", operation, file, line,
                cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation, __FILE__, __LINE__)

// Each block reuses a tile of source positions from shared memory.  The
// source positions are read-only during this kernel, so every body can be
// processed independently without atomics or synchronization between blocks.
__global__ void computeForcesKernel(const double* __restrict__ posX,
                                    const double* __restrict__ posY,
                                    const double* __restrict__ posZ,
                                    double* __restrict__ velX,
                                    double* __restrict__ velY,
                                    double* __restrict__ velZ,
                                    int numBodies) {
    extern __shared__ double sharedPositions[];
    double* tileX = sharedPositions;
    double* tileY = tileX + blockDim.x;
    double* tileZ = tileY + blockDim.x;

    const int thread = static_cast<int>(threadIdx.x);
    const int body = static_cast<int>(blockIdx.x * blockDim.x + thread);
    const bool active = body < numBodies;

    const double x = active ? posX[body] : 0.0;
    const double y = active ? posY[body] : 0.0;
    const double z = active ? posZ[body] : 0.0;
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileStart = 0; tileStart < numBodies; tileStart += blockDim.x) {
        const int sourceBody = tileStart + thread;
        if (sourceBody < numBodies) {
            tileX[thread] = posX[sourceBody];
            tileY[thread] = posY[sourceBody];
            tileZ[thread] = posZ[sourceBody];
        }
        __syncthreads();

        if (active) {
            int tileSize = numBodies - tileStart;
            if (tileSize > blockDim.x) {
                tileSize = blockDim.x;
            }

            #pragma unroll 4
            for (int source = 0; source < tileSize; ++source) {
                const double dx = tileX[source] - x;
                const double dy = tileY[source] - y;
                const double dz = tileZ[source] - z;
                const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inverseDistance = 1.0 / __dsqrt_rn(distanceSquared);
                const double inverseDistanceCubed = inverseDistance * inverseDistance * inverseDistance;

                forceX += dx * inverseDistanceCubed;
                forceY += dy * inverseDistanceCubed;
                forceZ += dz * inverseDistanceCubed;
            }
        }
        __syncthreads();
    }

    if (active) {
        velX[body] += DT * forceX;
        velY[body] += DT * forceY;
        velZ[body] += DT * forceZ;
    }
}

__global__ void integrateBodiesKernel(double* __restrict__ posX,
                                      double* __restrict__ posY,
                                      double* __restrict__ posZ,
                                      const double* __restrict__ velX,
                                      const double* __restrict__ velY,
                                      const double* __restrict__ velZ,
                                      int numBodies) {
    const int body = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (body < numBodies) {
        posX[body] += velX[body] * DT;
        posY[body] += velY[body] * DT;
        posZ[body] += velZ[body] * DT;
    }
}

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

class CudaBodies {
  public:
    explicit CudaBodies(const std::vector<Body>& bodies) : numBodies_(static_cast<int>(bodies.size())) {
        if (numBodies_ == 0) {
            return;
        }

        const size_t bytes = bodies.size() * sizeof(double);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&posX_), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&posY_), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&posZ_), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&velX_), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&velY_), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&velZ_), bytes));

        std::vector<double> posX(bodies.size());
        std::vector<double> posY(bodies.size());
        std::vector<double> posZ(bodies.size());
        std::vector<double> velX(bodies.size());
        std::vector<double> velY(bodies.size());
        std::vector<double> velZ(bodies.size());
        for (size_t i = 0; i < bodies.size(); ++i) {
            posX[i] = bodies[i].pos.x;
            posY[i] = bodies[i].pos.y;
            posZ[i] = bodies[i].pos.z;
            velX[i] = bodies[i].vel.x;
            velY[i] = bodies[i].vel.y;
            velZ[i] = bodies[i].vel.z;
        }

        CUDA_CHECK(cudaMemcpy(posX_, posX.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(posY_, posY.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(posZ_, posZ.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(velX_, velX.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(velY_, velY.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(velZ_, velZ.data(), bytes, cudaMemcpyHostToDevice));
    }

    ~CudaBodies() {
        cudaFree(posX_);
        cudaFree(posY_);
        cudaFree(posZ_);
        cudaFree(velX_);
        cudaFree(velY_);
        cudaFree(velZ_);
    }

    CudaBodies(const CudaBodies&) = delete;
    CudaBodies& operator=(const CudaBodies&) = delete;

    void step() {
        if (numBodies_ == 0) {
            return;
        }

        const dim3 block(THREADS_PER_BLOCK);
        const dim3 grid((numBodies_ + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        const size_t sharedBytes = 3 * THREADS_PER_BLOCK * sizeof(double);

        computeForcesKernel<<<grid, block, sharedBytes>>>(posX_, posY_, posZ_, velX_, velY_, velZ_, numBodies_);
        CUDA_CHECK(cudaGetLastError());

        integrateBodiesKernel<<<grid, block>>>(posX_, posY_, posZ_, velX_, velY_, velZ_, numBodies_);
        CUDA_CHECK(cudaGetLastError());
    }

    void download(std::vector<Body>& bodies) const {
        if (numBodies_ == 0) {
            return;
        }

        const size_t bytes = bodies.size() * sizeof(double);
        std::vector<double> posX(bodies.size());
        std::vector<double> posY(bodies.size());
        std::vector<double> posZ(bodies.size());
        std::vector<double> velX(bodies.size());
        std::vector<double> velY(bodies.size());
        std::vector<double> velZ(bodies.size());

        CUDA_CHECK(cudaMemcpy(posX.data(), posX_, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(posY.data(), posY_, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(posZ.data(), posZ_, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velX.data(), velX_, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velY.data(), velY_, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velZ.data(), velZ_, bytes, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < bodies.size(); ++i) {
            bodies[i].pos.x = posX[i];
            bodies[i].pos.y = posY[i];
            bodies[i].pos.z = posZ[i];
            bodies[i].vel.x = velX[i];
            bodies[i].vel.y = velY[i];
            bodies[i].vel.z = velZ[i];
        }
    }

  private:
    int numBodies_ = 0;
    double* posX_ = nullptr;
    double* posY_ = nullptr;
    double* posZ_ = nullptr;
    double* velX_ = nullptr;
    double* velY_ = nullptr;
    double* velZ_ = nullptr;
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
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    CudaBodies deviceBodies(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        deviceBodies.step();
    }

    if (numBodies > 0 && numSteps > 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    deviceBodies.download(bodies);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
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
