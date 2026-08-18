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

namespace {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int STATE_COMPONENTS = 6;

void checkCuda(const cudaError_t error, const char* expression, const char* file, const int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", file, line,
                     expression, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

// One thread owns one destination body.  The source positions are tiled through
// shared memory so every loaded position is reused by all threads in a block.
// The source and destination states are separate to make the timestep boundary
// explicit and to allow the compiler to use the non-aliasing annotations.
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void advanceBodiesKernel(const double* __restrict__ source,
                         double* __restrict__ destination,
                         const int n) {
    extern __shared__ double sharedPositions[];
    double* tileX = sharedPositions;
    double* tileY = tileX + THREADS_PER_BLOCK;
    double* tileZ = tileY + THREADS_PER_BLOCK;

    const int thread = static_cast<int>(threadIdx.x);
    const int body = static_cast<int>(blockIdx.x) * THREADS_PER_BLOCK + thread;

    const double* sourceX = source;
    const double* sourceY = source + n;
    const double* sourceZ = source + 2 * n;
    const double* sourceVx = source + 3 * n;
    const double* sourceVy = source + 4 * n;
    const double* sourceVz = source + 5 * n;

    double* destinationX = destination;
    double* destinationY = destination + n;
    double* destinationZ = destination + 2 * n;
    double* destinationVx = destination + 3 * n;
    double* destinationVy = destination + 4 * n;
    double* destinationVz = destination + 5 * n;

    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
    if (body < n) {
        px = sourceX[body];
        py = sourceY[body];
        pz = sourceZ[body];
        vx = sourceVx[body];
        vy = sourceVy[body];
        vz = sourceVz[body];
    }

    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileStart = 0; tileStart < n; tileStart += THREADS_PER_BLOCK) {
        const int sourceBody = tileStart + thread;
        if (sourceBody < n) {
            tileX[thread] = sourceX[sourceBody];
            tileY[thread] = sourceY[sourceBody];
            tileZ[thread] = sourceZ[sourceBody];
        }
        __syncthreads();

        const int tileSize = min(THREADS_PER_BLOCK, n - tileStart);
        if (body < n) {
            #pragma unroll 4
            for (int j = 0; j < tileSize; ++j) {
                const double dx = tileX[j] - px;
                const double dy = tileY[j] - py;
                const double dz = tileZ[j] - pz;
                const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inverseDistance = 1.0 / sqrt(distanceSquared);
                const double inverseDistanceCubed = inverseDistance * inverseDistance * inverseDistance;

                forceX += dx * inverseDistanceCubed;
                forceY += dy * inverseDistanceCubed;
                forceZ += dz * inverseDistanceCubed;
            }
        }
        __syncthreads();
    }

    if (body < n) {
        const double newVx = vx + DT * forceX;
        const double newVy = vy + DT * forceY;
        const double newVz = vz + DT * forceZ;

        destinationVx[body] = newVx;
        destinationVy[body] = newVy;
        destinationVz[body] = newVz;
        destinationX[body] = px + newVx * DT;
        destinationY[body] = py + newVy * DT;
        destinationZ[body] = pz + newVz * DT;
    }
}

void packBodies(const std::vector<Body>& bodies, std::vector<double>& state) {
    const size_t n = bodies.size();
    state.resize(STATE_COMPONENTS * n);
    if (n == 0) {
        return;
    }

    double* posX = state.data();
    double* posY = posX + n;
    double* posZ = posX + 2 * n;
    double* velX = posX + 3 * n;
    double* velY = posX + 4 * n;
    double* velZ = posX + 5 * n;

    for (size_t i = 0; i < n; ++i) {
        posX[i] = bodies[i].pos.x;
        posY[i] = bodies[i].pos.y;
        posZ[i] = bodies[i].pos.z;
        velX[i] = bodies[i].vel.x;
        velY[i] = bodies[i].vel.y;
        velZ[i] = bodies[i].vel.z;
    }
}

void unpackBodies(const std::vector<double>& state, std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    const double* posX = state.data();
    const double* posY = posX + n;
    const double* posZ = posX + 2 * n;
    const double* velX = posX + 3 * n;
    const double* velY = posX + 4 * n;
    const double* velZ = posX + 5 * n;

    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos.x = posX[i];
        bodies[i].pos.y = posY[i];
        bodies[i].pos.z = posZ[i];
        bodies[i].vel.x = velX[i];
        bodies[i].vel.y = velY[i];
        bodies[i].vel.z = velZ[i];
    }
}

class GpuNBody {
  public:
    explicit GpuNBody(const std::vector<Body>& bodies)
        : bodyCount_(static_cast<int>(bodies.size())) {
        packBodies(bodies, hostState_);
        if (bodyCount_ == 0) {
            return;
        }

        const size_t stateBytes = STATE_COMPONENTS * bodies.size() * sizeof(double);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state_[0]), stateBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state_[1]), stateBytes));
        CUDA_CHECK(cudaMemcpy(state_[0], hostState_.data(), stateBytes, cudaMemcpyHostToDevice));
        current_ = 0;
        next_ = 1;
    }

    ~GpuNBody() {
        if (state_[0] != nullptr) {
            cudaFree(state_[0]);
        }
        if (state_[1] != nullptr) {
            cudaFree(state_[1]);
        }
    }

    GpuNBody(const GpuNBody&) = delete;
    GpuNBody& operator=(const GpuNBody&) = delete;

    float advance(const int steps) {
        if (bodyCount_ == 0 || steps <= 0) {
            return 0.0f;
        }

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        const size_t blockCount = (static_cast<size_t>(bodyCount_) + THREADS_PER_BLOCK - 1) /
                                  THREADS_PER_BLOCK;
        const dim3 grid(static_cast<unsigned int>(blockCount));
        const dim3 block(THREADS_PER_BLOCK);
        constexpr size_t sharedBytes = 3 * THREADS_PER_BLOCK * sizeof(double);

        CUDA_CHECK(cudaEventRecord(start));
        for (int step = 0; step < steps; ++step) {
            advanceBodiesKernel<<<grid, block, sharedBytes>>>(state_[current_], state_[next_], bodyCount_);
            CUDA_CHECK(cudaGetLastError());
            const int oldCurrent = current_;
            current_ = next_;
            next_ = oldCurrent;
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsedMilliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return elapsedMilliseconds;
    }

    void download(std::vector<Body>& bodies) {
        if (bodyCount_ == 0) {
            return;
        }

        const size_t stateBytes = STATE_COMPONENTS * bodies.size() * sizeof(double);
        CUDA_CHECK(cudaMemcpy(hostState_.data(), state_[current_], stateBytes, cudaMemcpyDeviceToHost));
        unpackBodies(hostState_, bodies);
    }

  private:
    int bodyCount_ = 0;
    int current_ = 0;
    int next_ = 1;
    double* state_[2] = {nullptr, nullptr};
    std::vector<double> hostState_;
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

    // Keep the complete simulation on the GPU between the initial upload and
    // the optional final download used for reporting or validation.
    GpuNBody simulation(bodies);
    
    // Run simulation
    const float simulationMilliseconds = simulation.advance(numSteps);

    printf("Simulation time: %ld ms\n", static_cast<long>(simulationMilliseconds));

    if (printResults || validate) {
        simulation.download(bodies);
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
