#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int THREADS_PER_TARGET = 32;
constexpr int TARGETS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_TARGET;

void checkCuda(cudaError_t error, const char* operation, const char* file, int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
                     file, line, operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation, __FILE__, __LINE__)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept
        : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

// A warp cooperatively evaluates one target body. Source positions are staged
// once per block, so global reads are coalesced and shared by all targets in
// that block.
// The destination position arrays form the step barrier: a kernel only reads
// the immutable current arrays and only writes the next arrays.
__global__ __launch_bounds__(THREADS_PER_BLOCK, 4)
void advanceBodies(const double* __restrict__ positionX,
                   const double* __restrict__ positionY,
                   const double* __restrict__ positionZ,
                   double* __restrict__ nextPositionX,
                   double* __restrict__ nextPositionY,
                   double* __restrict__ nextPositionZ,
                   double* __restrict__ velocityX,
                   double* __restrict__ velocityY,
                   double* __restrict__ velocityZ,
                   int bodyCount) {
    __shared__ double tileX[THREADS_PER_BLOCK];
    __shared__ double tileY[THREADS_PER_BLOCK];
    __shared__ double tileZ[THREADS_PER_BLOCK];

    const int lane = threadIdx.x & (THREADS_PER_TARGET - 1);
    const int targetInBlock = threadIdx.x / THREADS_PER_TARGET;
    const int target = static_cast<int>(blockIdx.x) * TARGETS_PER_BLOCK + targetInBlock;
    const bool active = target < bodyCount;

    const double targetX = active ? positionX[target] : 0.0;
    const double targetY = active ? positionY[target] : 0.0;
    const double targetZ = active ? positionZ[target] : 0.0;
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileBase = 0; tileBase < bodyCount; tileBase += THREADS_PER_BLOCK) {
        const int source = tileBase + static_cast<int>(threadIdx.x);
        if (source < bodyCount) {
            tileX[threadIdx.x] = positionX[source];
            tileY[threadIdx.x] = positionY[source];
            tileZ[threadIdx.x] = positionZ[source];
        }
        __syncthreads();

        if (active) {
            const int tileSize = min(THREADS_PER_BLOCK, bodyCount - tileBase);
            for (int index = lane; index < tileSize; index += THREADS_PER_TARGET) {
                const double dx = tileX[index] - targetX;
                const double dy = tileY[index] - targetY;
                const double dz = tileZ[index] - targetZ;
                const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inverseDistance = rsqrt(distanceSquared);
                const double inverseDistanceCubed =
                    inverseDistance * inverseDistance * inverseDistance;

                forceX += dx * inverseDistanceCubed;
                forceY += dy * inverseDistanceCubed;
                forceZ += dz * inverseDistanceCubed;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int offset = THREADS_PER_TARGET / 2; offset > 0; offset >>= 1) {
        forceX += __shfl_down_sync(0xffffffffu, forceX, offset, THREADS_PER_TARGET);
        forceY += __shfl_down_sync(0xffffffffu, forceY, offset, THREADS_PER_TARGET);
        forceZ += __shfl_down_sync(0xffffffffu, forceZ, offset, THREADS_PER_TARGET);
    }

    if (lane == 0 && active) {
        const double newVelocityX = velocityX[target] + DT * forceX;
        const double newVelocityY = velocityY[target] + DT * forceY;
        const double newVelocityZ = velocityZ[target] + DT * forceZ;

        velocityX[target] = newVelocityX;
        velocityY[target] = newVelocityY;
        velocityZ[target] = newVelocityZ;
        nextPositionX[target] = targetX + newVelocityX * DT;
        nextPositionY[target] = targetY + newVelocityY * DT;
        nextPositionZ[target] = targetZ + newVelocityZ * DT;
    }
}

class DeviceSimulation {
  public:
    explicit DeviceSimulation(const std::vector<Body>& bodies)
        : bodyCount_(static_cast<int>(bodies.size())) {
        if (bodyCount_ == 0) {
            return;
        }

        const size_t count = bodies.size();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state_), 9 * count * sizeof(double)));
        hostState_.resize(6 * count);

        for (size_t i = 0; i < count; ++i) {
            hostState_[i] = bodies[i].pos.x;
            hostState_[count + i] = bodies[i].pos.y;
            hostState_[2 * count + i] = bodies[i].pos.z;
            hostState_[3 * count + i] = bodies[i].vel.x;
            hostState_[4 * count + i] = bodies[i].vel.y;
            hostState_[5 * count + i] = bodies[i].vel.z;
        }

        CUDA_CHECK(cudaMemcpy(position(0, 0), hostState_.data(), 3 * count * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(velocity(0), hostState_.data() + 3 * count,
                              3 * count * sizeof(double), cudaMemcpyHostToDevice));

        // Host staging is only needed again when results are requested.
        hostState_.clear();
        hostState_.shrink_to_fit();
    }

    DeviceSimulation(const DeviceSimulation&) = delete;
    DeviceSimulation& operator=(const DeviceSimulation&) = delete;

    ~DeviceSimulation() {
        if (state_ != nullptr) {
            cudaFree(state_);
        }
    }

    float run(int stepCount) {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        if (bodyCount_ > 0) {
            const int blockCount = (bodyCount_ + TARGETS_PER_BLOCK - 1) / TARGETS_PER_BLOCK;
            for (int step = 0; step < stepCount; ++step) {
                advanceBodies<<<blockCount, THREADS_PER_BLOCK>>>(
                    position(activePosition_, 0), position(activePosition_, 1),
                    position(activePosition_, 2), position(activePosition_ ^ 1, 0),
                    position(activePosition_ ^ 1, 1), position(activePosition_ ^ 1, 2),
                    velocity(0), velocity(1), velocity(2), bodyCount_);
                activePosition_ ^= 1;
            }
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsedMilliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return elapsedMilliseconds;
    }

    void download(std::vector<Body>& bodies) {
        if (bodyCount_ == 0) {
            return;
        }

        const size_t count = bodies.size();
        hostState_.resize(6 * count);
        CUDA_CHECK(cudaMemcpy(hostState_.data(), position(activePosition_, 0),
                              3 * count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostState_.data() + 3 * count, velocity(0),
                              3 * count * sizeof(double), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < count; ++i) {
            bodies[i].pos =
                Vec3(hostState_[i], hostState_[count + i], hostState_[2 * count + i]);
            bodies[i].vel = Vec3(hostState_[3 * count + i], hostState_[4 * count + i],
                                 hostState_[5 * count + i]);
        }
    }

  private:
    double* position(int buffer, int component) const {
        return state_ + static_cast<size_t>(buffer * 3 + component) * bodyCount_;
    }

    double* velocity(int component) const {
        return state_ + static_cast<size_t>(6 + component) * bodyCount_;
    }

    int bodyCount_ = 0;
    int activePosition_ = 0;
    double* state_ = nullptr;
    std::vector<double> hostState_;
};

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t count = bodies.size();

    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    for (size_t i = 0; i < count; ++i) {
        for (size_t j = i + 1; j < count; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / distance;
        }
    }
    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) ||
            !std::isfinite(body.pos.z) || !std::isfinite(body.vel.x) ||
            !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            std::printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        constexpr double maxPosition = 1e6;
        constexpr double maxVelocity = 1e6;
        if (std::abs(body.pos.x) > maxPosition || std::abs(body.pos.y) > maxPosition ||
            std::abs(body.pos.z) > maxPosition) {
            std::printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVelocity || std::abs(body.vel.y) > maxVelocity ||
            std::abs(body.vel.z) > maxVelocity) {
            std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

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

    if (numBodies < 0 || numSteps < 0) {
        std::fprintf(stderr, "The number of bodies and simulation steps must be non-negative.\n");
        return 1;
    }

    std::printf("N-Body Simulation\n");
    std::printf("Number of bodies: %d\n", numBodies);
    std::printf("Number of steps: %d\n", numSteps);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Initializing the runtime here also makes absence of a CUDA device a hard
    // error; this executable intentionally has no serial fallback.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));

    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    randomizeBodies(bodies);
    DeviceSimulation simulation(bodies);
    const float elapsedMilliseconds = simulation.run(numSteps);

    std::printf("Simulation time: %ld ms\n", static_cast<long>(elapsedMilliseconds));

    if (printResults || validate) {
        simulation.download(bodies);
    }

    if (printResults) {
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

    if (validate) {
        std::printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) {
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
