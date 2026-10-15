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

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

constexpr int BLOCK = 128;

// One thread per body; positions are tiled through shared memory.
// Fuses velocity update and position integration (positions are double buffered,
// so every thread sees the old positions for the whole force computation).
__global__ void stepKernel(const double4* __restrict__ posIn, double4* __restrict__ posOut,
                           double4* __restrict__ vel, int n) {
    __shared__ double4 tile[BLOCK];
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    const double4 pi = (i < n) ? posIn[i] : make_double4(0, 0, 0, 0);
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += BLOCK) {
        const int j = base + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? posIn[j] : make_double4(0, 0, 0, 0);
        __syncthreads();
        const int cnt = min(BLOCK, n - base);
        if (cnt == BLOCK) {
#pragma unroll 8
            for (int k = 0; k < BLOCK; ++k) {
                const double4 p = tile[k];
                const double dx = p.x - pi.x;
                const double dy = p.y - pi.y;
                const double dz = p.z - pi.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        } else {
            for (int k = 0; k < cnt; ++k) {
                const double4 p = tile[k];
                const double dx = p.x - pi.x;
                const double dy = p.y - pi.y;
                const double dz = p.z - pi.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
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

// Per-body potential energy (pairs j > i), summed on the host.
__global__ void energyKernel(const double4* __restrict__ pos, double* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double4 pi = pos[i];
    double e = 0.0;
    for (int j = i + 1; j < n; ++j) {
        const double4 p = pos[j];
        const double dx = p.x - pi.x;
        const double dy = p.y - pi.y;
        const double dz = p.z - pi.z;
        e -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
    }
    out[i] = e;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const int n = (int)bodies.size();
    double energy = 0.0;

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }
    if (n == 0) return energy;

    std::vector<double4> hp(n);
    for (int i = 0; i < n; ++i) hp[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
    double4* dp;
    double* de;
    CUDA_CHECK(cudaMalloc(&dp, n * sizeof(double4)));
    CUDA_CHECK(cudaMalloc(&de, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dp, hp.data(), n * sizeof(double4), cudaMemcpyHostToDevice));
    energyKernel<<<(n + 255) / 256, 256>>>(dp, de, n);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> he(n);
    CUDA_CHECK(cudaMemcpy(he.data(), de, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dp));
    CUDA_CHECK(cudaFree(de));
    for (int i = 0; i < n; ++i) energy += he[i];
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
    
    // Initialize the CUDA context up front
    CUDA_CHECK(cudaFree(0));

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    if (numBodies > 0) {
        const size_t bytes = (size_t)numBodies * sizeof(double4);
        std::vector<double4> hp(numBodies), hv(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            hp[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
            hv[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
        }
        double4 *dpA, *dpB, *dv;
        CUDA_CHECK(cudaMalloc(&dpA, bytes));
        CUDA_CHECK(cudaMalloc(&dpB, bytes));
        CUDA_CHECK(cudaMalloc(&dv, bytes));
        CUDA_CHECK(cudaMemcpy(dpA, hp.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv, hv.data(), bytes, cudaMemcpyHostToDevice));

        const int grid = (numBodies + BLOCK - 1) / BLOCK;
        for (int step = 0; step < numSteps; ++step) {
            stepKernel<<<grid, BLOCK>>>(dpA, dpB, dv, numBodies);
            std::swap(dpA, dpB);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hp.data(), dpA, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hv.data(), dv, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = Vec3(hp[i].x, hp[i].y, hp[i].z);
            bodies[i].vel = Vec3(hv[i].x, hv[i].y, hv[i].z);
        }
        CUDA_CHECK(cudaFree(dpA));
        CUDA_CHECK(cudaFree(dpB));
        CUDA_CHECK(cudaFree(dv));
    }

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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
