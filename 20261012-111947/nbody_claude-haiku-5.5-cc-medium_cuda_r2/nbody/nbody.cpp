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
constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
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

// One simulation step, one thread per body. Forces are computed from the positions at the
// start of the step (posIn) and the new positions are written to posOut, so the step never
// reads a position that another thread has already advanced. Each block stages a tile of
// bodies in shared memory so every position is read from global memory once per block
// instead of once per thread. Each thread accumulates over j in ascending order, the same
// order as the sequential reference.
__global__ void stepKernel(const Vec3* __restrict__ posIn, Vec3* __restrict__ posOut,
                           Vec3* __restrict__ vel, const int n) {
    __shared__ Vec3 tile[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const Vec3 pi = active ? posIn[i] : Vec3();

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? posIn[j] : Vec3();
        __syncthreads();

        const int count = min(BLOCK_SIZE, n - base);
        for (int k = 0; k < count; ++k) {
            const double dx = tile[k].x - pi.x;
            const double dy = tile[k].y - pi.y;
            const double dz = tile[k].z - pi.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    if (active) {
        Vec3 v = vel[i];
        v.x += DT * Fx;
        v.y += DT * Fy;
        v.z += DT * Fz;
        vel[i] = v;
        posOut[i] = Vec3(pi.x + v.x * DT, pi.y + v.y * DT, pi.z + v.z * DT);
    }
}

// Device-resident simulation state. Positions are double-buffered; velocities are updated in place.
// Bodies are uploaded once and downloaded once, so the per-step loop never touches host memory.
class DeviceBodies {
  public:
    explicit DeviceBodies(const std::vector<Body>& bodies) : n_(bodies.size()) {
        if (n_ == 0) return;

        std::vector<Vec3> pos(n_), vel(n_);
        for (size_t i = 0; i < n_; ++i) {
            pos[i] = bodies[i].pos;
            vel[i] = bodies[i].vel;
        }

        CUDA_CHECK(cudaMalloc(&pos_[0], n_ * sizeof(Vec3)));
        CUDA_CHECK(cudaMalloc(&pos_[1], n_ * sizeof(Vec3)));
        CUDA_CHECK(cudaMalloc(&vel_, n_ * sizeof(Vec3)));
        CUDA_CHECK(cudaMemcpy(pos_[0], pos.data(), n_ * sizeof(Vec3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(vel_, vel.data(), n_ * sizeof(Vec3), cudaMemcpyHostToDevice));
    }

    ~DeviceBodies() {
        if (n_ == 0) return;
        cudaFree(pos_[0]);
        cudaFree(pos_[1]);
        cudaFree(vel_);
    }

    DeviceBodies(const DeviceBodies&) = delete;
    DeviceBodies& operator=(const DeviceBodies&) = delete;

    void step(const int numSteps) {
        if (n_ == 0) return;

        const unsigned int blocks = static_cast<unsigned int>((n_ + BLOCK_SIZE - 1) / BLOCK_SIZE);
        for (int s = 0; s < numSteps; ++s) {
            stepKernel<<<blocks, BLOCK_SIZE>>>(pos_[cur_], pos_[cur_ ^ 1], vel_, static_cast<int>(n_));
            cur_ ^= 1;
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void download(std::vector<Body>& bodies) const {
        if (n_ == 0) return;

        std::vector<Vec3> pos(n_), vel(n_);
        CUDA_CHECK(cudaMemcpy(pos.data(), pos_[cur_], n_ * sizeof(Vec3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vel.data(), vel_, n_ * sizeof(Vec3), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n_; ++i) {
            bodies[i].pos = pos[i];
            bodies[i].vel = vel[i];
        }
    }

  private:
    size_t n_;
    Vec3* pos_[2] = {nullptr, nullptr};
    Vec3* vel_ = nullptr;
    int cur_ = 0;
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

    // Upload bodies to the GPU (outside the timed region, as host initialization is)
    DeviceBodies deviceBodies(bodies);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    deviceBodies.step(numSteps);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    deviceBodies.download(bodies);

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
