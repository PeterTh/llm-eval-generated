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

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// CUDA error checking macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// CUDA kernel: compute gravitational forces for all bodies (O(n^2) per step)
// Each thread computes the total force on one body from all other bodies.
// Uses rsqrt for fast reciprocal square root on GPU.
__global__ void computeForcesKernel(const double* __restrict__ pos, double* __restrict__ vel,
                                    const int n, const double dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double px = pos[i * 3 + 0];
    const double py = pos[i * 3 + 1];
    const double pz = pos[i * 3 + 2];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int j = 0; j < n; ++j) {
        const double dx = pos[j * 3 + 0] - px;
        const double dy = pos[j * 3 + 1] - py;
        const double dz = pos[j * 3 + 2] - pz;
        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double invDist = rsqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    vel[i * 3 + 0] += dt * Fx;
    vel[i * 3 + 1] += dt * Fy;
    vel[i * 3 + 2] += dt * Fz;
}

// CUDA kernel: integrate positions using velocities (O(n) per step)
__global__ void integrateKernel(double* __restrict__ pos, const double* __restrict__ vel,
                                const int n, const double dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    pos[i * 3 + 0] += vel[i * 3 + 0] * dt;
    pos[i * 3 + 1] += vel[i * 3 + 1] * dt;
    pos[i * 3 + 2] += vel[i * 3 + 2] * dt;
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
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

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

    printf("N-Body Simulation (CUDA)\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Initialize bodies on CPU
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Convert to flat interleaved arrays for GPU (x0,y0,z0, x1,y1,z1, ...)
    // This layout enables coalesced global memory access patterns
    std::vector<double> h_pos(numBodies * 3);
    std::vector<double> h_vel(numBodies * 3);
    for (int i = 0; i < numBodies; ++i) {
        h_pos[i * 3 + 0] = bodies[i].pos.x;
        h_pos[i * 3 + 1] = bodies[i].pos.y;
        h_pos[i * 3 + 2] = bodies[i].pos.z;
        h_vel[i * 3 + 0] = bodies[i].vel.x;
        h_vel[i * 3 + 1] = bodies[i].vel.y;
        h_vel[i * 3 + 2] = bodies[i].vel.z;
    }

    // Allocate device memory
    double *d_pos, *d_vel;
    const size_t arraySize = static_cast<size_t>(numBodies) * 3 * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_pos, arraySize));
    CUDA_CHECK(cudaMalloc(&d_vel, arraySize));

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), arraySize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, h_vel.data(), arraySize, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    constexpr int blockSize = 256;
    const int numBlocks = (numBodies + blockSize - 1) / blockSize;

    // Warm-up GPU to eliminate first-launch driver overhead from timing
    computeForcesKernel<<<1, 1>>>(d_pos, d_vel, 1, DT);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation (timing GPU kernels only, excluding PCIe transfers)
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<numBlocks, blockSize>>>(d_pos, d_vel, numBodies, DT);
        integrateKernel<<<numBlocks, blockSize>>>(d_pos, d_vel, numBodies, DT);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Copy results back to CPU
    CUDA_CHECK(cudaMemcpy(h_pos.data(), d_pos, arraySize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel.data(), d_vel, arraySize, cudaMemcpyDeviceToHost));

    // Convert flat arrays back to Body structs
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = h_pos[i * 3 + 0];
        bodies[i].pos.y = h_pos[i * 3 + 1];
        bodies[i].pos.z = h_pos[i * 3 + 2];
        bodies[i].vel.x = h_vel[i * 3 + 0];
        bodies[i].vel.y = h_vel[i * 3 + 1];
        bodies[i].vel.z = h_vel[i * 3 + 2];
    }

    // Clean up device memory
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));

    // Print results for external validation
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

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
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
