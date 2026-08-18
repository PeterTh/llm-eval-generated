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

constexpr int FORCE_BLOCK_SIZE = 256;

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Positions are staged by tiles so every global position load feeds an entire
// thread block.  Each thread keeps its force accumulator and its target body's
// position in registers, and visits j in ascending order just as the original
// scalar loop does.
__global__ void advanceBodiesKernel(
    const double* __restrict__ posX,
    const double* __restrict__ posY,
    const double* __restrict__ posZ,
    double* __restrict__ nextPosX,
    double* __restrict__ nextPosY,
    double* __restrict__ nextPosZ,
    double* __restrict__ velX,
    double* __restrict__ velY,
    double* __restrict__ velZ,
    const size_t numBodies) {
    __shared__ double tileX[FORCE_BLOCK_SIZE];
    __shared__ double tileY[FORCE_BLOCK_SIZE];
    __shared__ double tileZ[FORCE_BLOCK_SIZE];

    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = i < numBodies;

    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    if (active) {
        px = posX[i];
        py = posY[i];
        pz = posZ[i];
    }

    for (size_t tileStart = 0; tileStart < numBodies; tileStart += FORCE_BLOCK_SIZE) {
        const size_t tileCount = min(static_cast<size_t>(FORCE_BLOCK_SIZE), numBodies - tileStart);
        if (threadIdx.x < tileCount) {
            const size_t j = tileStart + threadIdx.x;
            tileX[threadIdx.x] = posX[j];
            tileY[threadIdx.x] = posY[j];
            tileZ[threadIdx.x] = posZ[j];
        }
        __syncthreads();

        if (active) {
            #pragma unroll 8
            for (size_t j = 0; j < tileCount; ++j) {
                const double dx = tileX[j] - px;
                const double dy = tileY[j] - py;
                const double dz = tileZ[j] - pz;
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
        const double updatedVelX = velX[i] + DT * fx;
        const double updatedVelY = velY[i] + DT * fy;
        const double updatedVelZ = velZ[i] + DT * fz;
        velX[i] = updatedVelX;
        velY[i] = updatedVelY;
        velZ[i] = updatedVelZ;
        nextPosX[i] = px + updatedVelX * DT;
        nextPosY[i] = py + updatedVelY * DT;
        nextPosZ[i] = pz + updatedVelZ * DT;
    }
}

class CudaBodies {
public:
    explicit CudaBodies(const std::vector<Body>& bodies) : numBodies_(bodies.size()) {
        if (numBodies_ == 0) {
            return;
        }

        const size_t bytes = numBodies_ * sizeof(double);
        checkCuda(cudaMalloc(&posX_, bytes), "allocating position x");
        checkCuda(cudaMalloc(&posY_, bytes), "allocating position y");
        checkCuda(cudaMalloc(&posZ_, bytes), "allocating position z");
        checkCuda(cudaMalloc(&nextPosX_, bytes), "allocating next position x");
        checkCuda(cudaMalloc(&nextPosY_, bytes), "allocating next position y");
        checkCuda(cudaMalloc(&nextPosZ_, bytes), "allocating next position z");
        checkCuda(cudaMalloc(&velX_, bytes), "allocating velocity x");
        checkCuda(cudaMalloc(&velY_, bytes), "allocating velocity y");
        checkCuda(cudaMalloc(&velZ_, bytes), "allocating velocity z");

        std::vector<double> hostX(numBodies_);
        std::vector<double> hostY(numBodies_);
        std::vector<double> hostZ(numBodies_);
        for (size_t i = 0; i < numBodies_; ++i) {
            hostX[i] = bodies[i].pos.x;
            hostY[i] = bodies[i].pos.y;
            hostZ[i] = bodies[i].pos.z;
        }
        copyToDevice(posX_, hostX, "copying position x to device");
        copyToDevice(posY_, hostY, "copying position y to device");
        copyToDevice(posZ_, hostZ, "copying position z to device");

        for (size_t i = 0; i < numBodies_; ++i) {
            hostX[i] = bodies[i].vel.x;
            hostY[i] = bodies[i].vel.y;
            hostZ[i] = bodies[i].vel.z;
        }
        copyToDevice(velX_, hostX, "copying velocity x to device");
        copyToDevice(velY_, hostY, "copying velocity y to device");
        copyToDevice(velZ_, hostZ, "copying velocity z to device");
    }

    CudaBodies(const CudaBodies&) = delete;
    CudaBodies& operator=(const CudaBodies&) = delete;

    ~CudaBodies() {
        cudaFree(posX_);
        cudaFree(posY_);
        cudaFree(posZ_);
        cudaFree(nextPosX_);
        cudaFree(nextPosY_);
        cudaFree(nextPosZ_);
        cudaFree(velX_);
        cudaFree(velY_);
        cudaFree(velZ_);
    }

    void advance() {
        if (numBodies_ == 0) {
            return;
        }

        const dim3 block(FORCE_BLOCK_SIZE);
        const dim3 grid(static_cast<unsigned int>((numBodies_ + FORCE_BLOCK_SIZE - 1) / FORCE_BLOCK_SIZE));
        advanceBodiesKernel<<<grid, block>>>(
            posX_, posY_, posZ_, nextPosX_, nextPosY_, nextPosZ_, velX_, velY_, velZ_, numBodies_);
        checkCuda(cudaGetLastError(), "launching timestep kernel");

        std::swap(posX_, nextPosX_);
        std::swap(posY_, nextPosY_);
        std::swap(posZ_, nextPosZ_);
    }

    void copyToHost(std::vector<Body>& bodies) const {
        if (numBodies_ == 0) {
            return;
        }

        std::vector<double> hostX(numBodies_);
        std::vector<double> hostY(numBodies_);
        std::vector<double> hostZ(numBodies_);
        copyToHost(posX_, hostX, "copying position x from device");
        copyToHost(posY_, hostY, "copying position y from device");
        copyToHost(posZ_, hostZ, "copying position z from device");
        for (size_t i = 0; i < numBodies_; ++i) {
            bodies[i].pos.x = hostX[i];
            bodies[i].pos.y = hostY[i];
            bodies[i].pos.z = hostZ[i];
        }

        copyToHost(velX_, hostX, "copying velocity x from device");
        copyToHost(velY_, hostY, "copying velocity y from device");
        copyToHost(velZ_, hostZ, "copying velocity z from device");
        for (size_t i = 0; i < numBodies_; ++i) {
            bodies[i].vel.x = hostX[i];
            bodies[i].vel.y = hostY[i];
            bodies[i].vel.z = hostZ[i];
        }
    }

    void synchronize() const {
        if (numBodies_ != 0) {
            checkCuda(cudaDeviceSynchronize(), "synchronizing simulation");
        }
    }

private:
    static void copyToDevice(double* destination, const std::vector<double>& source, const char* operation) {
        if (!source.empty()) {
            checkCuda(cudaMemcpy(destination, source.data(), source.size() * sizeof(double), cudaMemcpyHostToDevice), operation);
        }
    }

    static void copyToHost(const double* source, std::vector<double>& destination, const char* operation) {
        checkCuda(cudaMemcpy(destination.data(), source, destination.size() * sizeof(double), cudaMemcpyDeviceToHost), operation);
    }

    size_t numBodies_ = 0;
    double* posX_ = nullptr;
    double* posY_ = nullptr;
    double* posZ_ = nullptr;
    double* nextPosX_ = nullptr;
    double* nextPosY_ = nullptr;
    double* nextPosZ_ = nullptr;
    double* velX_ = nullptr;
    double* velY_ = nullptr;
    double* velZ_ = nullptr;
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
    
    // Keep the complete timestep on the GPU.  The initial upload and final
    // download deliberately sit outside the timed simulation region, matching
    // the original benchmark's treatment of host-side initialization/output.
    CudaBodies deviceBodies(bodies);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        deviceBodies.advance();
    }
    deviceBodies.synchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());

    deviceBodies.copyToHost(bodies);
    
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
