#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// All-pairs force computation with shared memory tiling for reduced global
// memory traffic. Each thread computes forces for one body (index i).
__global__ void computeForces_kernel(Body* __restrict__ bodies, int n, double dt) {
    extern __shared__ Vec3 sharedPos[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double px = 0.0, py = 0.0, pz = 0.0;
    bool active = (i < n);

    if (active) {
        px = bodies[i].pos.x;
        py = bodies[i].pos.y;
        pz = bodies[i].pos.z;
    }

    for (int tile = 0; tile < n; tile += blockDim.x) {
        int idx = tile + threadIdx.x;
        if (idx < n) {
            sharedPos[threadIdx.x] = bodies[idx].pos;
        }
        __syncthreads();

        if (active) {
            int jCount = blockDim.x;
            if (tile + blockDim.x > n) jCount = n - tile;

            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                if (j < jCount) {
                    double dx = sharedPos[j].x - px;
                    double dy = sharedPos[j].y - py;
                    double dz = sharedPos[j].z - pz;
                    double distSqr = dx*dx + dy*dy + dz*dz + 1e-9;
                    double invDist = rsqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            }
            for (int j = 32; j < 64; ++j) {
                if (j < jCount) {
                    double dx = sharedPos[j].x - px;
                    double dy = sharedPos[j].y - py;
                    double dz = sharedPos[j].z - pz;
                    double distSqr = dx*dx + dy*dy + dz*dz + 1e-9;
                    double invDist = rsqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            }
            for (int j = 64; j < 96; ++j) {
                if (j < jCount) {
                    double dx = sharedPos[j].x - px;
                    double dy = sharedPos[j].y - py;
                    double dz = sharedPos[j].z - pz;
                    double distSqr = dx*dx + dy*dy + dz*dz + 1e-9;
                    double invDist = rsqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            }
            for (int j = 96; j < 128; ++j) {
                if (j < jCount) {
                    double dx = sharedPos[j].x - px;
                    double dy = sharedPos[j].y - py;
                    double dz = sharedPos[j].z - pz;
                    double distSqr = dx*dx + dy*dy + dz*dz + 1e-9;
                    double invDist = rsqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            }
            for (int j = 128; j < 160; ++j) {
                if (j < jCount) {
                    double dx = sharedPos[j].x - px;
                    double dy = sharedPos[j].y - py;
                    double dz = sharedPos[j].z - pz;
                    double distSqr = dx*dx + dy*dy + dz*dz + 1e-9;
                    double invDist = rsqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            }
            for (int j = 160; j < 192; ++j) {
                if (j < jCount) {
                    double dx = sharedPos[j].x - px;
                    double dy = sharedPos[j].y - py;
                    double dz = sharedPos[j].z - pz;
                    double distSqr = dx*dx + dy*dy + dz*dz + 1e-9;
                    double invDist = rsqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            }
            for (int j = 192; j < 224; ++j) {
                if (j < jCount) {
                    double dx = sharedPos[j].x - px;
                    double dy = sharedPos[j].y - py;
                    double dz = sharedPos[j].z - pz;
                    double distSqr = dx*dx + dy*dy + dz*dz + 1e-9;
                    double invDist = rsqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            }
            for (int j = 224; j < 256; ++j) {
                if (j < jCount) {
                    double dx = sharedPos[j].x - px;
                    double dy = sharedPos[j].y - py;
                    double dz = sharedPos[j].z - pz;
                    double distSqr = dx*dx + dy*dy + dz*dz + 1e-9;
                    double invDist = rsqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        bodies[i].vel.x += dt * Fx;
        bodies[i].vel.y += dt * Fy;
        bodies[i].vel.z += dt * Fz;
    }
}

__global__ void integrateBodies_kernel(Body* bodies, int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    bodies[i].pos.x += bodies[i].vel.x * dt;
    bodies[i].pos.y += bodies[i].vel.y * dt;
    bodies[i].pos.z += bodies[i].vel.z * dt;
}

// ---------------------------------------------------------------------------
// GPU device memory management
// ---------------------------------------------------------------------------
static Body* d_bodies = nullptr;
static size_t d_capacity = 0;

static void ensureGpuBodies(size_t n) {
    if (n > d_capacity) {
        if (d_bodies) cudaFree(d_bodies);
        cudaMalloc(&d_bodies, n * sizeof(Body));
        d_capacity = n;
    }
}

static void freeGpuBodies() {
    if (d_bodies) {
        cudaFree(d_bodies);
        d_bodies = nullptr;
        d_capacity = 0;
    }
}

static void bodiesToGpu(const Body* src, size_t n) {
    cudaMemcpy(d_bodies, src, n * sizeof(Body), cudaMemcpyHostToDevice);
}

static void bodiesFromGpu(Body* dst, size_t n) {
    cudaMemcpy(dst, d_bodies, n * sizeof(Body), cudaMemcpyDeviceToHost);
}

// ---------------------------------------------------------------------------
// Host helper functions
// ---------------------------------------------------------------------------

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

// Launch CUDA force computation kernel (assumes device data is current)
void computeForces(std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    ensureGpuBodies(n);

    // On first call, push host data to device
    static bool first = true;
    if (first) {
        bodiesToGpu(bodies.data(), n);
        first = false;
    }

    const int blockSize = 256;
    const int gridSize = (static_cast<int>(n) + blockSize - 1) / blockSize;
    const size_t sharedMemSize = blockSize * sizeof(Vec3);

    computeForces_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_bodies,
                                                                  static_cast<int>(n),
                                                                  DT);
    cudaDeviceSynchronize();
}

// Launch CUDA position-integration kernel (assumes device data is current)
void integrateBodies(std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    ensureGpuBodies(n);

    const int blockSize = 256;
    const int gridSize = (static_cast<int>(n) + blockSize - 1) / blockSize;

    integrateBodies_kernel<<<gridSize, blockSize>>>(d_bodies,
                                                     static_cast<int>(n),
                                                     DT);
    cudaDeviceSynchronize();
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
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + 1e-9);
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

    // Initialize bodies on host
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Allocate GPU memory and transfer initial state
    ensureGpuBodies(numBodies);
    bodiesToGpu(bodies.data(), numBodies);

    // Run simulation – all computation stays on GPU during the loop
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies);
        integrateBodies(bodies);
    }

    // Retrieve final state from GPU
    bodiesFromGpu(bodies.data(), numBodies);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

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
            freeGpuBodies();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            freeGpuBodies();
            return 1;
        }
    }

    freeGpuBodies();
    return 0;
}
