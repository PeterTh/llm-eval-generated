#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int CUDA_BLOCK_SIZE = 32;

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

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
                 file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_status__ = (expression);                          \
        if (cuda_status__ != cudaSuccess) {                                      \
            cudaFailure(cuda_status__, #expression, __FILE__, __LINE__);         \
        }                                                                        \
    } while (false)

// Every thread advances one body. Positions are double-buffered so all force
// calculations in a step see the same input positions; this also allows force
// evaluation and integration to share one kernel launch.
template <int BlockSize>
__global__ __launch_bounds__(BlockSize)
void advanceBodies(const double* __restrict__ inX,
                   const double* __restrict__ inY,
                   const double* __restrict__ inZ,
                   double* __restrict__ outX,
                   double* __restrict__ outY,
                   double* __restrict__ outZ,
                   double* __restrict__ velX,
                   double* __restrict__ velY,
                   double* __restrict__ velZ,
                   const int count) {
    __shared__ double tileX[BlockSize];
    __shared__ double tileY[BlockSize];
    __shared__ double tileZ[BlockSize];

    const int bodyIndex = blockIdx.x * BlockSize + threadIdx.x;
    const bool active = bodyIndex < count;
    const double bodyX = active ? inX[bodyIndex] : 0.0;
    const double bodyY = active ? inY[bodyIndex] : 0.0;
    const double bodyZ = active ? inZ[bodyIndex] : 0.0;
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    const int fullTileCount = count / BlockSize;
    for (int tile = 0; tile < fullTileCount; ++tile) {
        const int sourceIndex = tile * BlockSize + threadIdx.x;
        tileX[threadIdx.x] = inX[sourceIndex];
        tileY[threadIdx.x] = inY[sourceIndex];
        tileZ[threadIdx.x] = inZ[sourceIndex];
        __syncwarp();

#pragma unroll 8
        for (int j = 0; j < BlockSize; ++j) {
            const double dx = tileX[j] - bodyX;
            const double dy = tileY[j] - bodyY;
            const double dz = tileZ[j] - bodyZ;
            const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double inverseDistance = rsqrt(distanceSquared);
            const double inverseDistanceCubed = inverseDistance * inverseDistance * inverseDistance;
            forceX += dx * inverseDistanceCubed;
            forceY += dy * inverseDistanceCubed;
            forceZ += dz * inverseDistanceCubed;
        }
        __syncwarp();
    }

    const int remainder = count - fullTileCount * BlockSize;
    if (remainder != 0) {
        const int sourceIndex = fullTileCount * BlockSize + threadIdx.x;
        if (threadIdx.x < remainder) {
            tileX[threadIdx.x] = inX[sourceIndex];
            tileY[threadIdx.x] = inY[sourceIndex];
            tileZ[threadIdx.x] = inZ[sourceIndex];
        }
        __syncwarp();

#pragma unroll 8
        for (int j = 0; j < remainder; ++j) {
            const double dx = tileX[j] - bodyX;
            const double dy = tileY[j] - bodyY;
            const double dz = tileZ[j] - bodyZ;
            const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double inverseDistance = rsqrt(distanceSquared);
            const double inverseDistanceCubed = inverseDistance * inverseDistance * inverseDistance;
            forceX += dx * inverseDistanceCubed;
            forceY += dy * inverseDistanceCubed;
            forceZ += dz * inverseDistanceCubed;
        }
    }

    if (active) {
        const double newVelocityX = velX[bodyIndex] + DT * forceX;
        const double newVelocityY = velY[bodyIndex] + DT * forceY;
        const double newVelocityZ = velZ[bodyIndex] + DT * forceZ;
        velX[bodyIndex] = newVelocityX;
        velY[bodyIndex] = newVelocityY;
        velZ[bodyIndex] = newVelocityZ;
        outX[bodyIndex] = bodyX + newVelocityX * DT;
        outY[bodyIndex] = bodyY + newVelocityY * DT;
        outZ[bodyIndex] = bodyZ + newVelocityZ * DT;
    }
}

// Device arrays use a structure-of-arrays layout so every warp performs
// contiguous global-memory accesses. A padded stride keeps each array aligned.
class DeviceBodies {
public:
    explicit DeviceBodies(const size_t count)
        : stride_((count + 31u) & ~size_t{31u}) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&storage_),
                              9u * stride_ * sizeof(double)));
        positionX_[0] = storage_;
        positionY_[0] = positionX_[0] + stride_;
        positionZ_[0] = positionY_[0] + stride_;
        positionX_[1] = positionZ_[0] + stride_;
        positionY_[1] = positionX_[1] + stride_;
        positionZ_[1] = positionY_[1] + stride_;
        velocityX_ = positionZ_[1] + stride_;
        velocityY_ = velocityX_ + stride_;
        velocityZ_ = velocityY_ + stride_;
    }

    DeviceBodies(const DeviceBodies&) = delete;
    DeviceBodies& operator=(const DeviceBodies&) = delete;

    ~DeviceBodies() {
        if (storage_ != nullptr) {
            cudaFree(storage_);
        }
    }

    double* positionX(const int buffer) const { return positionX_[buffer]; }
    double* positionY(const int buffer) const { return positionY_[buffer]; }
    double* positionZ(const int buffer) const { return positionZ_[buffer]; }
    double* velocityX() const { return velocityX_; }
    double* velocityY() const { return velocityY_; }
    double* velocityZ() const { return velocityZ_; }

private:
    size_t stride_ = 0;
    double* storage_ = nullptr;
    double* positionX_[2]{};
    double* positionY_[2]{};
    double* positionZ_[2]{};
    double* velocityX_ = nullptr;
    double* velocityY_ = nullptr;
    double* velocityZ_ = nullptr;
};

long long runSimulationCuda(std::vector<Body>& bodies, const int steps,
                            const bool copyResults) {
    const size_t count = bodies.size();
    if (count == 0) {
        // Initialize the CUDA runtime even for this degenerate case: CUDA is
        // deliberately the only execution backend.
        CUDA_CHECK(cudaFree(nullptr));
        return 0;
    }

    CUDA_CHECK(cudaFree(nullptr));
    DeviceBodies device(count);

    std::vector<double> hostState(6u * count);
    double* const hostX = hostState.data();
    double* const hostY = hostX + count;
    double* const hostZ = hostY + count;
    double* const hostVelocityX = hostZ + count;
    double* const hostVelocityY = hostVelocityX + count;
    double* const hostVelocityZ = hostVelocityY + count;

    for (size_t i = 0; i < count; ++i) {
        hostX[i] = bodies[i].pos.x;
        hostY[i] = bodies[i].pos.y;
        hostZ[i] = bodies[i].pos.z;
        hostVelocityX[i] = bodies[i].vel.x;
        hostVelocityY[i] = bodies[i].vel.y;
        hostVelocityZ[i] = bodies[i].vel.z;
    }

    const size_t bytes = count * sizeof(double);
    CUDA_CHECK(cudaMemcpy(device.positionX(0), hostX, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.positionY(0), hostY, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.positionZ(0), hostZ, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.velocityX(), hostVelocityX, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.velocityY(), hostVelocityY, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.velocityZ(), hostVelocityZ, bytes, cudaMemcpyHostToDevice));

    int inputBuffer = 0;
    int outputBuffer = 1;
    const int blockCount = static_cast<int>((count + CUDA_BLOCK_SIZE - 1u) / CUDA_BLOCK_SIZE);

    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < steps; ++step) {
        advanceBodies<CUDA_BLOCK_SIZE><<<blockCount, CUDA_BLOCK_SIZE>>>(
            device.positionX(inputBuffer), device.positionY(inputBuffer), device.positionZ(inputBuffer),
            device.positionX(outputBuffer), device.positionY(outputBuffer), device.positionZ(outputBuffer),
            device.velocityX(), device.velocityY(), device.velocityZ(), static_cast<int>(count));
        std::swap(inputBuffer, outputBuffer);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();

    if (copyResults) {
        CUDA_CHECK(cudaMemcpy(hostX, device.positionX(inputBuffer), bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostY, device.positionY(inputBuffer), bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostZ, device.positionZ(inputBuffer), bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostVelocityX, device.velocityX(), bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostVelocityY, device.velocityY(), bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostVelocityZ, device.velocityZ(), bytes, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < count; ++i) {
            bodies[i].pos = Vec3(hostX[i], hostY[i], hostZ[i]);
            bodies[i].vel = Vec3(hostVelocityX[i], hostVelocityY[i], hostVelocityZ[i]);
        }
    }

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
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
            std::printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        constexpr double maxPos = 1e6;
        constexpr double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            std::printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (numBodies < 0) {
        std::fprintf(stderr, "Number of bodies must be non-negative\n");
        return 1;
    }

    std::printf("N-Body Simulation\n");
    std::printf("Number of bodies: %d\n", numBodies);
    std::printf("Number of steps: %d\n", numSteps);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Initialize bodies
    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    randomizeBodies(bodies);

    const long long duration = runSimulationCuda(bodies, numSteps, validate || printResults);
    std::printf("Simulation time: %lld ms\n", duration);

    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6u);
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
        std::printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            // Report final energy for reference
            const double finalEnergy = computeTotalEnergy(bodies);
            std::printf("Final energy: %.6f\n", finalEnergy);
            std::printf("Validation: PASSED\n");
            return 0;
        }

        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
