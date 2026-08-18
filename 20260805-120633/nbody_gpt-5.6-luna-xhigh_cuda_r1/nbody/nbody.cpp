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

namespace {

constexpr int THREADS_PER_BLOCK = 256;

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error while %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// One block cooperatively loads a position tile. Every thread then computes
// one body's complete force, so force accumulation remains deterministic in
// increasing j order and requires no atomics.
__global__ void nbodyStepKernel(const double* __restrict__ posX,
                                const double* __restrict__ posY,
                                const double* __restrict__ posZ,
                                double* __restrict__ nextPosX,
                                double* __restrict__ nextPosY,
                                double* __restrict__ nextPosZ,
                                double* __restrict__ velX,
                                double* __restrict__ velY,
                                double* __restrict__ velZ,
                                int n) {
    extern __shared__ double positionTile[];
    double* tileX = positionTile;
    double* tileY = tileX + blockDim.x;
    double* tileZ = tileY + blockDim.x;

    const int lane = static_cast<int>(threadIdx.x);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    double ix = 0.0;
    double iy = 0.0;
    double iz = 0.0;
    if (i < n) {
        ix = posX[i];
        iy = posY[i];
        iz = posZ[i];
    }

    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;

    for (int tileStart = 0; tileStart < n; tileStart += blockDim.x) {
        const int j = tileStart + lane;
        if (j < n) {
            tileX[lane] = posX[j];
            tileY[lane] = posY[j];
            tileZ[lane] = posZ[j];
        }
        __syncthreads();

        if (i < n) {
            const int tileSize = min(static_cast<int>(blockDim.x), n - tileStart);
            #pragma unroll 4
            for (int k = 0; k < tileSize; ++k) {
                const double dx = tileX[k] - ix;
                const double dy = tileY[k] - iy;
                const double dz = tileZ[k] - iz;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = __ddiv_rn(1.0, __dsqrt_rn(distSqr));
                const double invDist3 = invDist * invDist * invDist;

                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        // This is computeForces followed by integrateBodies, in the same
        // order as the scalar implementation.
        const double updatedVx = velX[i] + DT * fx;
        const double updatedVy = velY[i] + DT * fy;
        const double updatedVz = velZ[i] + DT * fz;
        velX[i] = updatedVx;
        velY[i] = updatedVy;
        velZ[i] = updatedVz;
        nextPosX[i] = ix + updatedVx * DT;
        nextPosY[i] = iy + updatedVy * DT;
        nextPosZ[i] = iz + updatedVz * DT;
    }
}

class GpuSimulation {
  public:
    explicit GpuSimulation(const std::vector<Body>& bodies)
        : n_(static_cast<int>(bodies.size())) {
        if (n_ == 0) {
            return;
        }

        const size_t bodyCount = static_cast<size_t>(n_);
        const size_t valueCount = 6 * bodyCount;
        hostState_.resize(valueCount);

        double* posX = hostState_.data();
        double* posY = posX + bodyCount;
        double* posZ = posY + bodyCount;
        double* velX = posZ + bodyCount;
        double* velY = velX + bodyCount;
        double* velZ = velY + bodyCount;
        for (size_t i = 0; i < bodyCount; ++i) {
            posX[i] = bodies[i].pos.x;
            posY[i] = bodies[i].pos.y;
            posZ[i] = bodies[i].pos.z;
            velX[i] = bodies[i].vel.x;
            velY[i] = bodies[i].vel.y;
            velZ[i] = bodies[i].vel.z;
        }

        stateBytes_ = valueCount * sizeof(double);
        const size_t allocationBytes = 9 * bodyCount * sizeof(double);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceState_), allocationBytes),
                  "allocating device state");
        checkCuda(cudaMemcpy(deviceState_, hostState_.data(), stateBytes_, cudaMemcpyHostToDevice),
                  "uploading initial state");
    }

    GpuSimulation(const GpuSimulation&) = delete;
    GpuSimulation& operator=(const GpuSimulation&) = delete;

    ~GpuSimulation() {
        if (deviceState_ != nullptr) {
            cudaFree(deviceState_);
        }
    }

    void run(const int steps) {
        if (n_ == 0 || steps <= 0) {
            return;
        }

        double* posX = deviceState_;
        double* posY = posX + n_;
        double* posZ = posY + n_;
        double* velX = posZ + n_;
        double* velY = velX + n_;
        double* velZ = velY + n_;
        double* nextPosX = velZ + n_;
        double* nextPosY = nextPosX + n_;
        double* nextPosZ = nextPosY + n_;

        if (positionsAlternate_) {
            std::swap(posX, nextPosX);
            std::swap(posY, nextPosY);
            std::swap(posZ, nextPosZ);
        }

        const int blockCount = (n_ + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const size_t sharedBytes = 3 * THREADS_PER_BLOCK * sizeof(double);
        for (int step = 0; step < steps; ++step) {
            nbodyStepKernel<<<blockCount, THREADS_PER_BLOCK, sharedBytes>>>(
                posX, posY, posZ, nextPosX, nextPosY, nextPosZ, velX, velY, velZ, n_);
            std::swap(posX, nextPosX);
            std::swap(posY, nextPosY);
            std::swap(posZ, nextPosZ);
        }

        checkCuda(cudaGetLastError(), "launching simulation kernels");
        checkCuda(cudaDeviceSynchronize(), "running simulation kernels");
        positionsAlternate_ = (steps & 1) != 0 ? !positionsAlternate_ : positionsAlternate_;
    }

    void download(std::vector<Body>& bodies) {
        if (n_ == 0) {
            return;
        }

        const size_t bodyCount = static_cast<size_t>(n_);
        double* hostPosX = hostState_.data();
        double* hostPosY = hostPosX + bodyCount;
        double* hostPosZ = hostPosY + bodyCount;
        double* hostVelX = hostPosZ + bodyCount;
        double* hostVelY = hostVelX + bodyCount;
        double* hostVelZ = hostVelY + bodyCount;

        const double* posX = deviceState_;
        const double* posY = posX + n_;
        const double* posZ = posY + n_;
        const double* velX = posZ + n_;
        const double* velY = velX + n_;
        const double* velZ = velY + n_;
        const double* alternatePosX = velZ + n_;
        const double* alternatePosY = alternatePosX + n_;
        const double* alternatePosZ = alternatePosY + n_;
        if (positionsAlternate_) {
            posX = alternatePosX;
            posY = alternatePosY;
            posZ = alternatePosZ;
        }

        checkCuda(cudaMemcpy(hostPosX, posX, bodyCount * sizeof(double), cudaMemcpyDeviceToHost),
                  "downloading x positions");
        checkCuda(cudaMemcpy(hostPosY, posY, bodyCount * sizeof(double), cudaMemcpyDeviceToHost),
                  "downloading y positions");
        checkCuda(cudaMemcpy(hostPosZ, posZ, bodyCount * sizeof(double), cudaMemcpyDeviceToHost),
                  "downloading z positions");
        checkCuda(cudaMemcpy(hostVelX, velX, bodyCount * sizeof(double), cudaMemcpyDeviceToHost),
                  "downloading x velocities");
        checkCuda(cudaMemcpy(hostVelY, velY, bodyCount * sizeof(double), cudaMemcpyDeviceToHost),
                  "downloading y velocities");
        checkCuda(cudaMemcpy(hostVelZ, velZ, bodyCount * sizeof(double), cudaMemcpyDeviceToHost),
                  "downloading z velocities");

        for (size_t i = 0; i < bodyCount; ++i) {
            bodies[i].pos.x = hostPosX[i];
            bodies[i].pos.y = hostPosY[i];
            bodies[i].pos.z = hostPosZ[i];
            bodies[i].vel.x = hostVelX[i];
            bodies[i].vel.y = hostVelY[i];
            bodies[i].vel.z = hostVelZ[i];
        }
    }

  private:
    int n_;
    size_t stateBytes_ = 0;
    std::vector<double> hostState_;
    double* deviceState_ = nullptr;
    bool positionsAlternate_ = false;
};

} // namespace

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

    // Keep allocation and host-to-device transfer outside the benchmarked
    // simulation interval, matching the original timing boundary.
    GpuSimulation simulation(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    simulation.run(numSteps);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    simulation.download(bodies);
    
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
