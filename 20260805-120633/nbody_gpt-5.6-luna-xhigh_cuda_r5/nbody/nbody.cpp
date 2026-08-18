#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int CUDA_BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

namespace {

void checkCuda(const cudaError_t error, const char* expression, const char* file, const int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n", file, line, expression,
                     cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

// Each block loads a tile of source positions into shared memory. The target
// body remains in registers while it accumulates interactions in j order,
// matching the scalar implementation's summation order.
template <int BLOCK_SIZE>
__global__ void advanceBodiesKernel(const double* __restrict__ currentX,
                                    const double* __restrict__ currentY,
                                    const double* __restrict__ currentZ,
                                    double* __restrict__ nextX,
                                    double* __restrict__ nextY,
                                    double* __restrict__ nextZ,
                                    double* __restrict__ velocityX,
                                    double* __restrict__ velocityY,
                                    double* __restrict__ velocityZ,
                                    const int bodyCount) {
    __shared__ double tileX[BLOCK_SIZE];
    __shared__ double tileY[BLOCK_SIZE];
    __shared__ double tileZ[BLOCK_SIZE];

    const int lane = static_cast<int>(threadIdx.x);
    const int target = static_cast<int>(blockIdx.x) * BLOCK_SIZE + lane;

    double targetX = 0.0;
    double targetY = 0.0;
    double targetZ = 0.0;
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    if (target < bodyCount) {
        targetX = currentX[target];
        targetY = currentY[target];
        targetZ = currentZ[target];
    }

    for (int tileStart = 0; tileStart < bodyCount; tileStart += BLOCK_SIZE) {
        const int source = tileStart + lane;
        if (source < bodyCount) {
            tileX[lane] = currentX[source];
            tileY[lane] = currentY[source];
            tileZ[lane] = currentZ[source];
        } else {
            tileX[lane] = 0.0;
            tileY[lane] = 0.0;
            tileZ[lane] = 0.0;
        }
        __syncthreads();

        if (target < bodyCount) {
            const int tileCount = (bodyCount - tileStart < BLOCK_SIZE) ? bodyCount - tileStart : BLOCK_SIZE;
#pragma unroll 4
            for (int sourceInTile = 0; sourceInTile < tileCount; ++sourceInTile) {
                const double dx = tileX[sourceInTile] - targetX;
                const double dy = tileY[sourceInTile] - targetY;
                const double dz = tileZ[sourceInTile] - targetZ;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                // Keep round-to-nearest sqrt/division semantics of the
                // original scalar expression; this also prevents nvcc from
                // replacing it with a lower-precision reciprocal sqrt.
                const double distance = __dsqrt_rn(distSqr);
                const double invDist = __ddiv_rn(1.0, distance);
                const double invDist3 = invDist * invDist * invDist;

                forceX += dx * invDist3;
                forceY += dy * invDist3;
                forceZ += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (target < bodyCount) {
        // Integrate from the updated velocity into a separate position
        // buffer. This avoids a grid-wide synchronization between force
        // evaluation and integration.
        // The original performs this velocity update before the separate
        // integration loop, so keep its multiply and add as distinct
        // round-to-nearest operations.
        double updatedVelocityX = __dadd_rn(velocityX[target], __dmul_rn(DT, forceX));
        double updatedVelocityY = __dadd_rn(velocityY[target], __dmul_rn(DT, forceY));
        double updatedVelocityZ = __dadd_rn(velocityZ[target], __dmul_rn(DT, forceZ));

        velocityX[target] = updatedVelocityX;
        velocityY[target] = updatedVelocityY;
        velocityZ[target] = updatedVelocityZ;

        nextX[target] = targetX + updatedVelocityX * DT;
        nextY[target] = targetY + updatedVelocityY * DT;
        nextZ[target] = targetZ + updatedVelocityZ * DT;
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

class GpuNBody {
public:
    explicit GpuNBody(const std::vector<Body>& bodies)
        : bodyCount_(static_cast<int>(bodies.size())) {
        if (bodyCount_ == 0) {
            return;
        }

        const size_t count = static_cast<size_t>(bodyCount_);
        const size_t positionBytes = 3 * count * sizeof(double);
        const size_t velocityBytes = 3 * count * sizeof(double);

        for (int buffer = 0; buffer < 2; ++buffer) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&positionStorage_[buffer]), positionBytes));
            positionX_[buffer] = positionStorage_[buffer];
            positionY_[buffer] = positionStorage_[buffer] + count;
            positionZ_[buffer] = positionStorage_[buffer] + 2 * count;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&velocityStorage_), velocityBytes));
        velocityX_ = velocityStorage_;
        velocityY_ = velocityStorage_ + count;
        velocityZ_ = velocityStorage_ + 2 * count;

        std::vector<double> hostX(count), hostY(count), hostZ(count);
        std::vector<double> hostVelocityX(count), hostVelocityY(count), hostVelocityZ(count);
        for (size_t i = 0; i < count; ++i) {
            hostX[i] = bodies[i].pos.x;
            hostY[i] = bodies[i].pos.y;
            hostZ[i] = bodies[i].pos.z;
            hostVelocityX[i] = bodies[i].vel.x;
            hostVelocityY[i] = bodies[i].vel.y;
            hostVelocityZ[i] = bodies[i].vel.z;
        }

        CUDA_CHECK(cudaMemcpy(positionX_[0], hostX.data(), count * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(positionY_[0], hostY.data(), count * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(positionZ_[0], hostZ.data(), count * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(velocityX_, hostVelocityX.data(), count * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(velocityY_, hostVelocityY.data(), count * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(velocityZ_, hostVelocityZ.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    }

    GpuNBody(const GpuNBody&) = delete;
    GpuNBody& operator=(const GpuNBody&) = delete;

    ~GpuNBody() {
        CUDA_CHECK(cudaFree(velocityStorage_));
        CUDA_CHECK(cudaFree(positionStorage_[0]));
        CUDA_CHECK(cudaFree(positionStorage_[1]));
    }

    void simulate(const int steps) {
        if (bodyCount_ == 0) {
            return;
        }

        const dim3 block(CUDA_BLOCK_SIZE);
        const dim3 grid((static_cast<unsigned int>(bodyCount_) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
        for (int step = 0; step < steps; ++step) {
            const int nextBuffer = currentBuffer_ ^ 1;
            advanceBodiesKernel<CUDA_BLOCK_SIZE><<<grid, block>>>(
                positionX_[currentBuffer_], positionY_[currentBuffer_], positionZ_[currentBuffer_],
                positionX_[nextBuffer], positionY_[nextBuffer], positionZ_[nextBuffer],
                velocityX_, velocityY_, velocityZ_, bodyCount_);
            CUDA_CHECK(cudaGetLastError());
            currentBuffer_ = nextBuffer;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void download(std::vector<Body>& bodies) const {
        if (bodyCount_ == 0) {
            return;
        }

        const size_t count = static_cast<size_t>(bodyCount_);
        std::vector<double> hostX(count), hostY(count), hostZ(count);
        std::vector<double> hostVelocityX(count), hostVelocityY(count), hostVelocityZ(count);

        CUDA_CHECK(cudaMemcpy(hostX.data(), positionX_[currentBuffer_], count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostY.data(), positionY_[currentBuffer_], count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostZ.data(), positionZ_[currentBuffer_], count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostVelocityX.data(), velocityX_, count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostVelocityY.data(), velocityY_, count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostVelocityZ.data(), velocityZ_, count * sizeof(double), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < count; ++i) {
            bodies[i].pos.x = hostX[i];
            bodies[i].pos.y = hostY[i];
            bodies[i].pos.z = hostZ[i];
            bodies[i].vel.x = hostVelocityX[i];
            bodies[i].vel.y = hostVelocityY[i];
            bodies[i].vel.z = hostVelocityZ[i];
        }
    }

private:
    int bodyCount_ = 0;
    int currentBuffer_ = 0;
    double* positionStorage_[2] = {nullptr, nullptr};
    double* velocityStorage_ = nullptr;
    double* positionX_[2] = {nullptr, nullptr};
    double* positionY_[2] = {nullptr, nullptr};
    double* positionZ_[2] = {nullptr, nullptr};
    double* velocityX_ = nullptr;
    double* velocityY_ = nullptr;
    double* velocityZ_ = nullptr;
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

} // namespace

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
        printf("Number of bodies and steps must be non-negative\n");
        return 1;
    }

    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Initialize bodies
    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    randomizeBodies(bodies);

    // Run simulation on the CUDA device.
    GpuNBody simulation(bodies);
    auto start = std::chrono::high_resolution_clock::now();
    simulation.simulate(numSteps);
    auto end = std::chrono::high_resolution_clock::now();
    simulation.download(bodies);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
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
