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

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression, const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while evaluating %s: %s\n",
                 file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                        \
    do {                                                                              \
        const cudaError_t cuda_check_result = (expression);                           \
        if (cuda_check_result != cudaSuccess) {                                       \
            cudaFailure(cuda_check_result, #expression, __FILE__, __LINE__);          \
        }                                                                             \
    } while (false)

// For large problems, one thread owns one body. Source positions are staged in
// shared memory so every global load is reused by the entire thread block.
template <int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void advanceBodiesTiled(const double* __restrict__ inX,
                        const double* __restrict__ inY,
                        const double* __restrict__ inZ,
                        double* __restrict__ outX,
                        double* __restrict__ outY,
                        double* __restrict__ outZ,
                        double* __restrict__ velX,
                        double* __restrict__ velY,
                        double* __restrict__ velZ,
                        const int n) {
    __shared__ double tileX[BLOCK_SIZE];
    __shared__ double tileY[BLOCK_SIZE];
    __shared__ double tileZ[BLOCK_SIZE];

    const int bodyIndex = static_cast<int>(blockIdx.x) * BLOCK_SIZE + static_cast<int>(threadIdx.x);
    const bool active = bodyIndex < n;
    const double bodyX = active ? inX[bodyIndex] : 0.0;
    const double bodyY = active ? inY[bodyIndex] : 0.0;
    const double bodyZ = active ? inZ[bodyIndex] : 0.0;
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileBase = 0; tileBase < n; tileBase += BLOCK_SIZE) {
        const int sourceIndex = tileBase + static_cast<int>(threadIdx.x);
        if (sourceIndex < n) {
            tileX[threadIdx.x] = inX[sourceIndex];
            tileY[threadIdx.x] = inY[sourceIndex];
            tileZ[threadIdx.x] = inZ[sourceIndex];
        }
        __syncthreads();

        if (active) {
            const int tileCount = min(BLOCK_SIZE, n - tileBase);
#pragma unroll 8
            for (int j = 0; j < tileCount; ++j) {
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
        __syncthreads();
    }

    if (active) {
        const double newVelocityX = velX[bodyIndex] + DT * forceX;
        const double newVelocityY = velY[bodyIndex] + DT * forceY;
        const double newVelocityZ = velZ[bodyIndex] + DT * forceZ;
        velX[bodyIndex] = newVelocityX;
        velY[bodyIndex] = newVelocityY;
        velZ[bodyIndex] = newVelocityZ;
        outX[bodyIndex] = bodyX + DT * newVelocityX;
        outY[bodyIndex] = bodyY + DT * newVelocityY;
        outZ[bodyIndex] = bodyZ + DT * newVelocityZ;
    }
}

// Small and medium problems do not contain enough bodies to occupy a modern
// GPU with one thread per body. Here a power-of-two group of lanes cooperates
// on each body's force and reduces its partial sums with warp shuffles.
template <int LANES_PER_BODY, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void advanceBodiesCooperative(const double* __restrict__ inX,
                              const double* __restrict__ inY,
                              const double* __restrict__ inZ,
                              double* __restrict__ outX,
                              double* __restrict__ outY,
                              double* __restrict__ outZ,
                              double* __restrict__ velX,
                              double* __restrict__ velY,
                              double* __restrict__ velZ,
                              const int n) {
    constexpr unsigned int FULL_WARP_MASK = 0xffffffffu;
    const int globalThread = static_cast<int>(blockIdx.x) * BLOCK_SIZE + static_cast<int>(threadIdx.x);
    const int bodyIndex = globalThread / LANES_PER_BODY;
    const int lane = globalThread & (LANES_PER_BODY - 1);
    const bool active = bodyIndex < n;
    const double bodyX = active ? inX[bodyIndex] : 0.0;
    const double bodyY = active ? inY[bodyIndex] : 0.0;
    const double bodyZ = active ? inZ[bodyIndex] : 0.0;
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    if (active) {
#pragma unroll 4
        for (int j = lane; j < n; j += LANES_PER_BODY) {
            const double dx = inX[j] - bodyX;
            const double dy = inY[j] - bodyY;
            const double dz = inZ[j] - bodyZ;
            const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double inverseDistance = rsqrt(distanceSquared);
            const double inverseDistanceCubed = inverseDistance * inverseDistance * inverseDistance;
            forceX += dx * inverseDistanceCubed;
            forceY += dy * inverseDistanceCubed;
            forceZ += dz * inverseDistanceCubed;
        }
    }

#pragma unroll
    for (int offset = LANES_PER_BODY / 2; offset > 0; offset /= 2) {
        forceX += __shfl_down_sync(FULL_WARP_MASK, forceX, offset, LANES_PER_BODY);
        forceY += __shfl_down_sync(FULL_WARP_MASK, forceY, offset, LANES_PER_BODY);
        forceZ += __shfl_down_sync(FULL_WARP_MASK, forceZ, offset, LANES_PER_BODY);
    }

    if (active && lane == 0) {
        const double newVelocityX = velX[bodyIndex] + DT * forceX;
        const double newVelocityY = velY[bodyIndex] + DT * forceY;
        const double newVelocityZ = velZ[bodyIndex] + DT * forceZ;
        velX[bodyIndex] = newVelocityX;
        velY[bodyIndex] = newVelocityY;
        velZ[bodyIndex] = newVelocityZ;
        outX[bodyIndex] = bodyX + DT * newVelocityX;
        outY[bodyIndex] = bodyY + DT * newVelocityY;
        outZ[bodyIndex] = bodyZ + DT * newVelocityZ;
    }
}

class CudaSimulation {
  public:
    CudaSimulation(const std::vector<Body>& bodies, const int steps)
        : count_(static_cast<int>(bodies.size())), steps_(steps) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

        if (count_ == 0) {
            return;
        }

        const size_t componentBytes = static_cast<size_t>(count_) * sizeof(double);
        std::vector<double> hostState(static_cast<size_t>(count_) * 6);
        double* hostPosX = hostState.data();
        double* hostPosY = hostPosX + count_;
        double* hostPosZ = hostPosY + count_;
        double* hostVelX = hostPosZ + count_;
        double* hostVelY = hostVelX + count_;
        double* hostVelZ = hostVelY + count_;
        for (int i = 0; i < count_; ++i) {
            hostPosX[i] = bodies[i].pos.x;
            hostPosY[i] = bodies[i].pos.y;
            hostPosZ[i] = bodies[i].pos.z;
            hostVelX[i] = bodies[i].vel.x;
            hostVelY[i] = bodies[i].vel.y;
            hostVelZ[i] = bodies[i].vel.z;
        }

        CUDA_CHECK(cudaMalloc(&deviceState_, static_cast<size_t>(count_) * 9 * sizeof(double)));
        currentX_ = deviceState_;
        currentY_ = currentX_ + count_;
        currentZ_ = currentY_ + count_;
        nextX_ = currentZ_ + count_;
        nextY_ = nextX_ + count_;
        nextZ_ = nextY_ + count_;
        velocityX_ = nextZ_ + count_;
        velocityY_ = velocityX_ + count_;
        velocityZ_ = velocityY_ + count_;

        CUDA_CHECK(cudaMemcpyAsync(currentX_, hostPosX, componentBytes * 3,
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(velocityX_, hostVelX, componentBytes * 3,
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        buildExecutionGraph();
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    ~CudaSimulation() {
        if (batchGraphExecutable_ != nullptr) {
            cudaGraphExecDestroy(batchGraphExecutable_);
        }
        if (batchGraph_ != nullptr) {
            cudaGraphDestroy(batchGraph_);
        }
        if (tailGraphExecutable_ != nullptr) {
            cudaGraphExecDestroy(tailGraphExecutable_);
        }
        if (tailGraph_ != nullptr) {
            cudaGraphDestroy(tailGraph_);
        }
        if (deviceState_ != nullptr) {
            cudaFree(deviceState_);
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    void advance() {
        for (int batch = 0; batch < batchLaunches_; ++batch) {
            CUDA_CHECK(cudaGraphLaunch(batchGraphExecutable_, stream_));
        }
        if (tailGraphExecutable_ != nullptr) {
            CUDA_CHECK(cudaGraphLaunch(tailGraphExecutable_, stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void copyToHost(std::vector<Body>& bodies) const {
        if (count_ == 0) {
            return;
        }

        const size_t componentBytes = static_cast<size_t>(count_) * sizeof(double);
        std::vector<double> hostState(static_cast<size_t>(count_) * 6);
        double* hostPosX = hostState.data();
        double* hostPosY = hostPosX + count_;
        double* hostPosZ = hostPosY + count_;
        double* hostVelX = hostPosZ + count_;
        double* hostVelY = hostVelX + count_;
        double* hostVelZ = hostVelY + count_;
        CUDA_CHECK(cudaMemcpyAsync(hostPosX, currentX_, componentBytes * 3,
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(hostVelX, velocityX_, componentBytes * 3,
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        for (int i = 0; i < count_; ++i) {
            bodies[i].pos.x = hostPosX[i];
            bodies[i].pos.y = hostPosY[i];
            bodies[i].pos.z = hostPosZ[i];
            bodies[i].vel.x = hostVelX[i];
            bodies[i].vel.y = hostVelY[i];
            bodies[i].vel.z = hostVelZ[i];
        }
    }

  private:
    static constexpr int BLOCK_SIZE = 128;
    static constexpr int GRAPH_BATCH_STEPS = 128;
    // Three 150,000-element position arrays occupy 3.6 MB. Below this point
    // they remain cache-friendly and a warp per body maximizes arithmetic
    // occupancy; beyond it, explicit shared-memory tiling limits DRAM traffic.
    static constexpr int COOPERATIVE_BODY_LIMIT = 150000;

    template <int LANES_PER_BODY>
    void launchCooperativeStep() {
        const int bodiesPerBlock = BLOCK_SIZE / LANES_PER_BODY;
        const int blocks = (count_ + bodiesPerBlock - 1) / bodiesPerBlock;
        advanceBodiesCooperative<LANES_PER_BODY, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream_>>>(
            currentX_, currentY_, currentZ_, nextX_, nextY_, nextZ_,
            velocityX_, velocityY_, velocityZ_, count_);
    }

    void launchStep() {
        if (count_ <= COOPERATIVE_BODY_LIMIT) {
            launchCooperativeStep<32>();
        } else {
            const int blocks = (count_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
            advanceBodiesTiled<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream_>>>(
                currentX_, currentY_, currentZ_, nextX_, nextY_, nextZ_,
                velocityX_, velocityY_, velocityZ_, count_);
        }
        CUDA_CHECK(cudaGetLastError());
        std::swap(currentX_, nextX_);
        std::swap(currentY_, nextY_);
        std::swap(currentZ_, nextZ_);
    }

    void captureSteps(const int graphSteps, cudaGraph_t* graph, cudaGraphExec_t* executable) {
        CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        for (int step = 0; step < graphSteps; ++step) {
            launchStep();
        }
        CUDA_CHECK(cudaStreamEndCapture(stream_, graph));
        CUDA_CHECK(cudaGraphInstantiate(executable, *graph, nullptr, nullptr, 0));
    }

    void buildExecutionGraph() {
        if (steps_ <= 0) {
            return;
        }

        if (steps_ <= GRAPH_BATCH_STEPS) {
            captureSteps(steps_, &batchGraph_, &batchGraphExecutable_);
            batchLaunches_ = 1;
        } else {
            // An even-sized graph returns the ping-pong position buffers to
            // their starting orientation, so it can be replayed without
            // constructing an unbounded graph.
            captureSteps(GRAPH_BATCH_STEPS, &batchGraph_, &batchGraphExecutable_);
            batchLaunches_ = steps_ / GRAPH_BATCH_STEPS;
            const int tailSteps = steps_ % GRAPH_BATCH_STEPS;
            if (tailSteps != 0) {
                captureSteps(tailSteps, &tailGraph_, &tailGraphExecutable_);
            }
        }

        // Upload graph metadata before the benchmark interval, just like the
        // device allocation and initial-state transfer above.
        CUDA_CHECK(cudaGraphUpload(batchGraphExecutable_, stream_));
        if (tailGraphExecutable_ != nullptr) {
            CUDA_CHECK(cudaGraphUpload(tailGraphExecutable_, stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    int count_ = 0;
    int steps_ = 0;
    double* deviceState_ = nullptr;
    double* currentX_ = nullptr;
    double* currentY_ = nullptr;
    double* currentZ_ = nullptr;
    double* nextX_ = nullptr;
    double* nextY_ = nullptr;
    double* nextZ_ = nullptr;
    double* velocityX_ = nullptr;
    double* velocityY_ = nullptr;
    double* velocityZ_ = nullptr;
    cudaStream_t stream_ = nullptr;
    int batchLaunches_ = 0;
    cudaGraph_t batchGraph_ = nullptr;
    cudaGraphExec_t batchGraphExecutable_ = nullptr;
    cudaGraph_t tailGraph_ = nullptr;
    cudaGraphExec_t tailGraphExecutable_ = nullptr;
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

    // Initialize bodies and upload the structure-of-arrays device state.
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    CudaSimulation simulation(bodies, numSteps);

    // Graph construction and transfers are setup costs; this interval measures
    // the same simulation-step work as the original CPU timer.
    const auto start = std::chrono::high_resolution_clock::now();
    simulation.advance();
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    if (printResults || validate) {
        simulation.copyToHost(bodies);
    }

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
            const double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            return 0;
        }

        printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
