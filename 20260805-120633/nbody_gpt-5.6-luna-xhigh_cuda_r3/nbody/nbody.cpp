#include <algorithm>
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
constexpr int THREADS_PER_BLOCK = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

[[noreturn]] void cudaError(const cudaError_t status, const char* expression, const char* file, const int line) {
    fprintf(stderr, "CUDA error at %s:%d (%s): %s\n", file, line, expression, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) \
    do { \
        const cudaError_t status = (expression); \
        if (status != cudaSuccess) { \
            cudaError(status, #expression, __FILE__, __LINE__); \
        } \
    } while (false)

// The six arrays are kept separate on the device so every warp performs
// coalesced loads and stores.  The host-facing representation remains Body,
// preserving the original layout and output order.
struct DeviceBodies {
    double* storage = nullptr;
    double* posX = nullptr;
    double* posY = nullptr;
    double* posZ = nullptr;
    double* velX = nullptr;
    double* velY = nullptr;
    double* velZ = nullptr;

    void allocate(const size_t bodyCount) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&storage), 6 * bodyCount * sizeof(double)));
        posX = storage;
        posY = posX + bodyCount;
        posZ = posY + bodyCount;
        velX = posZ + bodyCount;
        velY = velX + bodyCount;
        velZ = velY + bodyCount;
    }

    void release() noexcept {
        if (storage != nullptr) {
            cudaFree(storage);
            storage = nullptr;
        }
        posX = nullptr;
        posY = nullptr;
        posZ = nullptr;
        velX = nullptr;
        velY = nullptr;
        velZ = nullptr;
    }

    ~DeviceBodies() {
        release();
    }

    DeviceBodies() = default;
    DeviceBodies(const DeviceBodies&) = delete;
    DeviceBodies& operator=(const DeviceBodies&) = delete;
};

// One thread owns one output body.  Every block cooperatively loads a tile of
// positions into shared memory, reducing the global position traffic from one
// load per interaction to one load per body per tile.
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void advanceBodies(const double* __restrict__ inputPosX,
                   const double* __restrict__ inputPosY,
                   const double* __restrict__ inputPosZ,
                   const double* __restrict__ inputVelX,
                   const double* __restrict__ inputVelY,
                   const double* __restrict__ inputVelZ,
                   double* __restrict__ outputPosX,
                   double* __restrict__ outputPosY,
                   double* __restrict__ outputPosZ,
                   double* __restrict__ outputVelX,
                   double* __restrict__ outputVelY,
                   double* __restrict__ outputVelZ,
                   const int bodyCount) {
    extern __shared__ double tilePositions[];
    double* tilePosX = tilePositions;
    double* tilePosY = tilePosX + blockDim.x;
    double* tilePosZ = tilePosY + blockDim.x;

    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = body < bodyCount;

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;

    if (active) {
        x = inputPosX[body];
        y = inputPosY[body];
        z = inputPosZ[body];
        vx = inputVelX[body];
        vy = inputVelY[body];
        vz = inputVelZ[body];
    }

    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileStart = 0; tileStart < bodyCount; tileStart += blockDim.x) {
        const int sourceBody = tileStart + threadIdx.x;
        if (sourceBody < bodyCount) {
            tilePosX[threadIdx.x] = inputPosX[sourceBody];
            tilePosY[threadIdx.x] = inputPosY[sourceBody];
            tilePosZ[threadIdx.x] = inputPosZ[sourceBody];
        } else {
            tilePosX[threadIdx.x] = 0.0;
            tilePosY[threadIdx.x] = 0.0;
            tilePosZ[threadIdx.x] = 0.0;
        }
        __syncthreads();

        if (active) {
            const int tileCount = min(blockDim.x, bodyCount - tileStart);
            #pragma unroll 4
            for (int j = 0; j < tileCount; ++j) {
                const double dx = tilePosX[j] - x;
                const double dy = tilePosY[j] - y;
                const double dz = tilePosZ[j] - z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                forceX += dx * invDist3;
                forceY += dy * invDist3;
                forceZ += dz * invDist3;
            }
        }
        // In the final, partially occupied block, inactive threads can reach
        // the next tile load early.  This barrier prevents them from
        // overwriting shared positions still being consumed by active threads.
        __syncthreads();
    }

    if (active) {
        const double newVx = vx + DT * forceX;
        const double newVy = vy + DT * forceY;
        const double newVz = vz + DT * forceZ;

        // This is the original velocity update followed by integration,
        // evaluated from the same input state for every body.
        outputVelX[body] = newVx;
        outputVelY[body] = newVy;
        outputVelZ[body] = newVz;
        outputPosX[body] = x + newVx * DT;
        outputPosY[body] = y + newVy * DT;
        outputPosZ[body] = z + newVz * DT;
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

class CudaSimulation {
public:
    explicit CudaSimulation(const std::vector<Body>& bodies)
        : bodyCount_(static_cast<int>(bodies.size())),
          allocationCount_(std::max<size_t>(bodies.size(), 1)),
          blocks_((bodyCount_ + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) {
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            fprintf(stderr, "CUDA n-body simulation requires an NVIDIA GPU\n");
            std::exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaSetDevice(0));

        first_.allocate(allocationCount_);
        second_.allocate(allocationCount_);

        if (bodyCount_ != 0) {
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

            upload(first_.posX, posX);
            upload(first_.posY, posY);
            upload(first_.posZ, posZ);
            upload(first_.velX, velX);
            upload(first_.velY, velY);
            upload(first_.velZ, velZ);
        }
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    void run(const int steps) {
        if (bodyCount_ == 0) {
            return;
        }

        constexpr size_t sharedBytes = 3 * THREADS_PER_BLOCK * sizeof(double);
        for (int step = 0; step < steps; ++step) {
            advanceBodies<<<blocks_, THREADS_PER_BLOCK, sharedBytes>>>(
                current_->posX, current_->posY, current_->posZ,
                current_->velX, current_->velY, current_->velZ,
                next_->posX, next_->posY, next_->posZ,
                next_->velX, next_->velY, next_->velZ,
                bodyCount_);
            CUDA_CHECK(cudaGetLastError());
            std::swap(current_, next_);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void download(std::vector<Body>& bodies) const {
        if (bodyCount_ == 0) {
            return;
        }

        std::vector<double> posX(bodies.size());
        std::vector<double> posY(bodies.size());
        std::vector<double> posZ(bodies.size());
        std::vector<double> velX(bodies.size());
        std::vector<double> velY(bodies.size());
        std::vector<double> velZ(bodies.size());

        download(current_->posX, posX);
        download(current_->posY, posY);
        download(current_->posZ, posZ);
        download(current_->velX, velX);
        download(current_->velY, velY);
        download(current_->velZ, velZ);

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
    template <typename T>
    static void upload(double* destination, const std::vector<T>& source) {
        CUDA_CHECK(cudaMemcpy(destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
    }

    static void download(const double* source, std::vector<double>& destination) {
        CUDA_CHECK(cudaMemcpy(destination.data(), source, destination.size() * sizeof(double), cudaMemcpyDeviceToHost));
    }

    int bodyCount_;
    size_t allocationCount_;
    int blocks_;
    DeviceBodies first_;
    DeviceBodies second_;
    DeviceBodies* current_ = &first_;
    DeviceBodies* next_ = &second_;
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

    if (numBodies < 0) {
        fprintf(stderr, "Number of bodies must be non-negative\n");
        return 1;
    }
    
    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Initialize bodies
    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    randomizeBodies(bodies);
    CudaSimulation simulation(bodies);
    
    // Run simulation on the GPU.  CUDA is intentionally required; there is
    // no serial fallback path.
    auto start = std::chrono::high_resolution_clock::now();
    simulation.run(numSteps);
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
