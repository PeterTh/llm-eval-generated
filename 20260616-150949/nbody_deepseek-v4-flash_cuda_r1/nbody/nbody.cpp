#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <cuda_runtime.h>

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 256;

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
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ==================== CUDA Kernels ====================

// Force computation kernel with shared memory tiling for coalesced global memory access.
// Each thread computes the net force on one body i, iterating over all bodies j in tiles.
__global__ void computeForcesKernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    int n,
    double dt)
{
    extern __shared__ double shared[];
    double* sh_x = shared;
    double* sh_y = &shared[blockDim.x];
    double* sh_z = &shared[2 * blockDim.x];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    const double my_x = pos_x[i];
    const double my_y = pos_y[i];
    const double my_z = pos_z[i];

    // Iterate over tiles of bodies loaded into shared memory
    for (int tile = 0; tile * blockDim.x < n; ++tile) {
        const int idx = tile * blockDim.x + threadIdx.x;
        if (idx < n) {
            sh_x[threadIdx.x] = pos_x[idx];
            sh_y[threadIdx.x] = pos_y[idx];
            sh_z[threadIdx.x] = pos_z[idx];
        }
        __syncthreads();

        // Accumulate forces from all bodies in this tile
        for (int j = 0; j < blockDim.x; ++j) {
            const int j_idx = tile * blockDim.x + j;
            if (j_idx >= n) break;

            const double dx = sh_x[j] - my_x;
            const double dy = sh_y[j] - my_y;
            const double dz = sh_z[j] - my_z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = rsqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    // Update velocity with accumulated force
    vel_x[i] += dt * Fx;
    vel_y[i] += dt * Fy;
    vel_z[i] += dt * Fz;
}

// Position integration kernel: updates positions using current velocities
__global__ void integrateBodiesKernel(
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ vel_z,
    int n,
    double dt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    pos_x[i] += vel_x[i] * dt;
    pos_y[i] += vel_y[i] * dt;
    pos_z[i] += vel_z[i] * dt;
}

// ==================== Host Functions ====================

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

// Host wrapper for launching the force computation kernel
void computeForcesGPU(double* d_pos_x, double* d_pos_y, double* d_pos_z,
                      double* d_vel_x, double* d_vel_y, double* d_vel_z,
                      int n, double dt) {
    const int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t sharedMemSize = 3 * BLOCK_SIZE * sizeof(double);
    computeForcesKernel<<<gridSize, BLOCK_SIZE, sharedMemSize>>>(
        d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, n, dt);
    CUDA_CHECK(cudaGetLastError());
}

// Host wrapper for launching the integration kernel
void integrateBodiesGPU(double* d_pos_x, double* d_pos_y, double* d_pos_z,
                        double* d_vel_x, double* d_vel_y, double* d_vel_z,
                        int n, double dt) {
    const int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    integrateBodiesKernel<<<gridSize, BLOCK_SIZE>>>(
        d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, n, dt);
    CUDA_CHECK(cudaGetLastError());
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

    printf("N-Body Simulation (CUDA)\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Initialize bodies on CPU (AoS layout)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Allocate GPU device memory (SoA layout for coalesced access)
    double *d_pos_x, *d_pos_y, *d_pos_z;
    double *d_vel_x, *d_vel_y, *d_vel_z;

    CUDA_CHECK(cudaMalloc(&d_pos_x, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pos_y, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pos_z, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_x, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_y, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_z, numBodies * sizeof(double)));

    // Convert AoS (CPU) -> SoA (GPU) and copy to device
    std::vector<double> h_pos_x(numBodies), h_pos_y(numBodies), h_pos_z(numBodies);
    std::vector<double> h_vel_x(numBodies), h_vel_y(numBodies), h_vel_z(numBodies);

    for (int i = 0; i < numBodies; ++i) {
        h_pos_x[i] = bodies[i].pos.x;
        h_pos_y[i] = bodies[i].pos.y;
        h_pos_z[i] = bodies[i].pos.z;
        h_vel_x[i] = bodies[i].vel.x;
        h_vel_y[i] = bodies[i].vel.y;
        h_vel_z[i] = bodies[i].vel.z;
    }

    CUDA_CHECK(cudaMemcpy(d_pos_x, h_pos_x.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y, h_pos_y.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_z, h_pos_z.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_x, h_vel_x.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_y, h_vel_y.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_z, h_vel_z.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));

    // Run simulation — all computation stays on GPU
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesGPU(d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, numBodies, DT);
        integrateBodiesGPU(d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, numBodies, DT);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Copy results back from GPU (SoA -> AoS)
    CUDA_CHECK(cudaMemcpy(h_pos_x.data(), d_pos_x, numBodies * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_y.data(), d_pos_y, numBodies * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_z.data(), d_pos_z, numBodies * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel_x.data(), d_vel_x, numBodies * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel_y.data(), d_vel_y, numBodies * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel_z.data(), d_vel_z, numBodies * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = h_pos_x[i];
        bodies[i].pos.y = h_pos_y[i];
        bodies[i].pos.z = h_pos_z[i];
        bodies[i].vel.x = h_vel_x[i];
        bodies[i].vel.y = h_vel_y[i];
        bodies[i].vel.z = h_vel_z[i];
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

    // Cleanup GPU memory
    CUDA_CHECK(cudaFree(d_pos_x));
    CUDA_CHECK(cudaFree(d_pos_y));
    CUDA_CHECK(cudaFree(d_pos_z));
    CUDA_CHECK(cudaFree(d_vel_x));
    CUDA_CHECK(cudaFree(d_vel_y));
    CUDA_CHECK(cudaFree(d_vel_z));

    return 0;
}
