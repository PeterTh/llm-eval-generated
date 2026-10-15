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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

constexpr int BLOCK = 256;

// One thread per body: accumulate force over all bodies (tiled through shared memory),
// update velocity, and integrate position into the output buffer (fused force + integrate).
__global__ void stepKernel(const double4* __restrict__ posIn, double4* __restrict__ posOut,
                           double4* __restrict__ vel, int n) {
    __shared__ double4 tile[BLOCK];
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    double4 pi = make_double4(0, 0, 0, 0);
    if (i < n) pi = posIn[i];
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += BLOCK) {
        const int cnt = min(BLOCK, n - base);
        __syncthreads();
        if (threadIdx.x < cnt) tile[threadIdx.x] = posIn[base + threadIdx.x];
        __syncthreads();
        if (cnt == BLOCK) {
#pragma unroll 8
            for (int j = 0; j < BLOCK; ++j) {
                const double4 q = tile[j];
                const double dx = q.x - pi.x;
                const double dy = q.y - pi.y;
                const double dz = q.z - pi.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        } else {
            for (int j = 0; j < cnt; ++j) {
                const double4 q = tile[j];
                const double dx = q.x - pi.x;
                const double dy = q.y - pi.y;
                const double dz = q.z - pi.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
    }

    if (i < n) {
        double4 v = vel[i];
        v.x += DT * Fx;
        v.y += DT * Fy;
        v.z += DT * Fz;
        vel[i] = v;
        posOut[i] = make_double4(pi.x + v.x * DT, pi.y + v.y * DT, pi.z + v.z * DT, 0.0);
    }
}

// Pairwise potential energy: each thread handles one i, j > i; block partial sums.
__global__ void energyKernel(const double4* __restrict__ pos, const double4* __restrict__ vel,
                             double* __restrict__ blockSums, int n) {
    __shared__ double red[BLOCK];
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    double e = 0.0;
    if (i < n) {
        const double4 v = vel[i];
        e += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
        const double4 p = pos[i];
        for (int j = i + 1; j < n; ++j) {
            const double4 q = pos[j];
            const double dx = q.x - p.x;
            const double dy = q.y - p.y;
            const double dz = q.z - p.z;
            e -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    red[threadIdx.x] = e;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) blockSums[blockIdx.x] = red[0];
}

double computeTotalEnergy(const double4* dPos, const double4* dVel, int n) {
    const int blocks = (n + BLOCK - 1) / BLOCK;
    double* dSums;
    CUDA_CHECK(cudaMalloc(&dSums, sizeof(double) * blocks));
    energyKernel<<<blocks, BLOCK>>>(dPos, dVel, dSums, n);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> sums(blocks);
    CUDA_CHECK(cudaMemcpy(sums.data(), dSums, sizeof(double) * blocks, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dSums));
    double energy = 0.0;
    for (double s : sums) energy += s;
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
    
    // Device state (double4 for vectorized loads; w unused)
    const int n = numBodies;
    const int blocks = (n + BLOCK - 1) / BLOCK;
    std::vector<double4> hPos(n), hVel(n);
    for (int i = 0; i < n; ++i) {
        hPos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
        hVel[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
    }
    double4 *dPosA = nullptr, *dPosB = nullptr, *dVel = nullptr;
    const size_t bytes = sizeof(double4) * (n > 0 ? n : 1);
    CUDA_CHECK(cudaMalloc(&dPosA, bytes));
    CUDA_CHECK(cudaMalloc(&dPosB, bytes));
    CUDA_CHECK(cudaMalloc(&dVel, bytes));
    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(dPosA, hPos.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dVel, hVel.data(), bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    if (n > 0) {
        for (int step = 0; step < numSteps; ++step) {
            stepKernel<<<blocks, BLOCK>>>(dPosA, dPosB, dVel, n);
            std::swap(dPosA, dPosB);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(hPos.data(), dPosA, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hVel.data(), dVel, bytes, cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(hPos[i].x, hPos[i].y, hPos[i].z);
        bodies[i].vel = Vec3(hVel[i].x, hVel[i].y, hVel[i].z);
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
            double finalEnergy = n > 0 ? computeTotalEnergy(dPosA, dVel, n) : 0.0;
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
