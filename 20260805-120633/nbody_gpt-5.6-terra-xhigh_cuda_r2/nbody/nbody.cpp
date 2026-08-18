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

[[noreturn]] void cudaCheckFailed(cudaError_t error, const char* expression, const char* file, int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file, line, expression,
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                                      \
    do {                                                                                            \
        const cudaError_t cudaError = (expression);                                                 \
        if (cudaError != cudaSuccess) {                                                             \
            cudaCheckFailed(cudaError, #expression, __FILE__, __LINE__);                           \
        }                                                                                           \
    } while (false)

// Each block cooperatively stages a tile of source positions in shared memory.  Positions remain
// read-only during this kernel, so updating each body's velocity in place is race-free.
__global__ void computeForcesKernel(const double* __restrict__ posX,
                                    const double* __restrict__ posY,
                                    const double* __restrict__ posZ,
                                    double* __restrict__ velX,
                                    double* __restrict__ velY,
                                    double* __restrict__ velZ,
                                    size_t numBodies) {
    __shared__ double tileX[THREADS_PER_BLOCK];
    __shared__ double tileY[THREADS_PER_BLOCK];
    __shared__ double tileZ[THREADS_PER_BLOCK];

    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = i < numBodies;
    const double xi = active ? posX[i] : 0.0;
    const double yi = active ? posY[i] : 0.0;
    const double zi = active ? posZ[i] : 0.0;
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;

    for (size_t tileStart = 0; tileStart < numBodies; tileStart += blockDim.x) {
        const size_t source = tileStart + threadIdx.x;
        if (source < numBodies) {
            tileX[threadIdx.x] = posX[source];
            tileY[threadIdx.x] = posY[source];
            tileZ[threadIdx.x] = posZ[source];
        }
        __syncthreads();

        const size_t tileSize = min(static_cast<size_t>(blockDim.x), numBodies - tileStart);
        if (active) {
            for (size_t j = 0; j < tileSize; ++j) {
                const double dx = tileX[j] - xi;
                const double dy = tileY[j] - yi;
                const double dz = tileZ[j] - zi;
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
        velX[i] += DT * fx;
        velY[i] += DT * fy;
        velZ[i] += DT * fz;
    }
}

__global__ void integrateBodiesKernel(double* __restrict__ posX,
                                      double* __restrict__ posY,
                                      double* __restrict__ posZ,
                                      const double* __restrict__ velX,
                                      const double* __restrict__ velY,
                                      const double* __restrict__ velZ,
                                      size_t numBodies) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < numBodies) {
        posX[i] += velX[i] * DT;
        posY[i] += velY[i] * DT;
        posZ[i] += velZ[i] * DT;
    }
}

class DeviceBodies {
public:
    explicit DeviceBodies(const std::vector<Body>& bodies) : numBodies_(bodies.size()) {
        if (numBodies_ == 0) {
            return;
        }

        const size_t bytes = numBodies_ * sizeof(double);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&posX_), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&posY_), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&posZ_), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&velX_), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&velY_), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&velZ_), bytes));

        std::vector<double> posX(numBodies_);
        std::vector<double> posY(numBodies_);
        std::vector<double> posZ(numBodies_);
        std::vector<double> velX(numBodies_);
        std::vector<double> velY(numBodies_);
        std::vector<double> velZ(numBodies_);
        for (size_t i = 0; i < numBodies_; ++i) {
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

    DeviceBodies(const DeviceBodies&) = delete;
    DeviceBodies& operator=(const DeviceBodies&) = delete;

    ~DeviceBodies() {
        cudaFree(posX_);
        cudaFree(posY_);
        cudaFree(posZ_);
        cudaFree(velX_);
        cudaFree(velY_);
        cudaFree(velZ_);
    }

    void simulate(int numSteps) {
        if (numBodies_ == 0 || numSteps <= 0) {
            return;
        }

        const unsigned int blocks = static_cast<unsigned int>(
            (numBodies_ + static_cast<size_t>(THREADS_PER_BLOCK) - 1) / THREADS_PER_BLOCK);
        for (int step = 0; step < numSteps; ++step) {
            computeForcesKernel<<<blocks, THREADS_PER_BLOCK>>>(posX_, posY_, posZ_, velX_, velY_, velZ_, numBodies_);
            integrateBodiesKernel<<<blocks, THREADS_PER_BLOCK>>>(posX_, posY_, posZ_, velX_, velY_, velZ_, numBodies_);
        }
        // Kernels in the default stream execute in launch order.  Check once after the batch so
        // long simulations do not pay host-side error-query overhead for every individual launch.
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void download(std::vector<Body>& bodies) const {
        if (numBodies_ == 0) {
            return;
        }

        const size_t bytes = numBodies_ * sizeof(double);
        std::vector<double> posX(numBodies_);
        std::vector<double> posY(numBodies_);
        std::vector<double> posZ(numBodies_);
        std::vector<double> velX(numBodies_);
        std::vector<double> velY(numBodies_);
        std::vector<double> velZ(numBodies_);
        CUDA_CHECK(cudaMemcpy(posX.data(), posX_, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(posY.data(), posY_, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(posZ.data(), posZ_, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velX.data(), velX_, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velY.data(), velY_, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velZ.data(), velZ_, bytes, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < numBodies_; ++i) {
            bodies[i].pos.x = posX[i];
            bodies[i].pos.y = posY[i];
            bodies[i].pos.z = posZ[i];
            bodies[i].vel.x = velX[i];
            bodies[i].vel.y = velY[i];
            bodies[i].vel.z = velZ[i];
        }
    }

private:
    size_t numBodies_ = 0;
    double* posX_ = nullptr;
    double* posY_ = nullptr;
    double* posZ_ = nullptr;
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

    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

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

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        constexpr double maxPos = 1e6;
        constexpr double maxVel = 1e6;
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

    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    DeviceBodies deviceBodies(bodies);

    const auto start = std::chrono::high_resolution_clock::now();
    deviceBodies.simulate(numSteps);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    if (printResults || validate) {
        deviceBodies.download(bodies);
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
        printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) {
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
