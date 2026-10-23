#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;

struct Vec3 {
    double x, y, z;
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be tightly packed");

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

// One thread per body. Each thread accumulates the force from every other body
// in the same order as the sequential reference implementation, so results are
// numerically identical while the work for all bodies runs in parallel.
// Positions are staged through shared memory in tiles to cut global traffic.
constexpr int SMALL_BLOCK_SIZE = 32;
constexpr int LARGE_BLOCK_SIZE = 64;
constexpr int LARGE_BLOCK_THRESHOLD = 32768;

template <int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void nbodyStepKernel(
    const Body* __restrict__ src, Body* __restrict__ dst, const int n) {
    constexpr double kSoftening = 1e-9;
    constexpr double kDt = 0.01;

    __shared__ double tile[BLOCK_SIZE * 3];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * BLOCK_SIZE + tid;
    const bool active = i < n;

    double px = 0.0, py = 0.0, pz = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    if (active) {
        const Body b = src[i];
        px = b.pos.x;
        py = b.pos.y;
        pz = b.pos.z;
        vx = b.vel.x;
        vy = b.vel.y;
        vz = b.vel.z;
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int j = base + tid;
        if (j < n) {
            tile[3 * tid + 0] = __ldg(&src[j].pos.x);
            tile[3 * tid + 1] = __ldg(&src[j].pos.y);
            tile[3 * tid + 2] = __ldg(&src[j].pos.z);
        }
        __syncthreads();

        const int jmax = (n - base < BLOCK_SIZE) ? (n - base) : BLOCK_SIZE;
        #pragma unroll 4
        for (int k = 0; k < jmax; ++k) {
            const double dx = tile[3 * k + 0] - px;
            const double dy = tile[3 * k + 1] - py;
            const double dz = tile[3 * k + 2] - pz;
            // Match the reference compiler's FMA contraction exactly:
            // distSqr = ((dy*dy) + dx*dx) + dz*dz + SOFTENING
            double distSqr = __dmul_rn(dy, dy);
            distSqr = __fma_rn(dx, dx, distSqr);
            distSqr = __fma_rn(dz, dz, distSqr);
            distSqr += kSoftening;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist2 = __dmul_rn(invDist, invDist);
            const double invDist3 = __dmul_rn(invDist2, invDist);

            Fx = __fma_rn(dx, invDist3, Fx);
            Fy = __fma_rn(dy, invDist3, Fy);
            Fz = __fma_rn(dz, invDist3, Fz);
        }
        __syncthreads();
    }

    if (active) {
        Body out;
        const double nvx = vx + __dmul_rn(kDt, Fx);
        const double nvy = vy + __dmul_rn(kDt, Fy);
        const double nvz = vz + __dmul_rn(kDt, Fz);
        out.pos.x = __fma_rn(nvx, kDt, px);
        out.pos.y = __fma_rn(nvy, kDt, py);
        out.pos.z = __fma_rn(nvz, kDt, pz);
        out.vel.x = nvx;
        out.vel.y = nvy;
        out.vel.z = nvz;
        dst[i] = out;
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

    const size_t bytes = bodies.size() * sizeof(Body);
    Body* d_buf0 = nullptr;
    Body* d_buf1 = nullptr;
    if (bytes > 0) {
        const cudaError_t alloc0 = cudaMalloc(&d_buf0, bytes);
        const cudaError_t alloc1 = cudaMalloc(&d_buf1, bytes);
        if (alloc0 != cudaSuccess || alloc1 != cudaSuccess) {
            fprintf(stderr, "Failed to allocate device memory\n");
            cudaFree(d_buf0);
            cudaFree(d_buf1);
            return 1;
        }
        if (cudaMemcpy(d_buf0, bodies.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "Failed to copy bodies to device\n");
            cudaFree(d_buf0);
            cudaFree(d_buf1);
            return 1;
        }
    }

    Body* src = d_buf0;
    Body* dst = d_buf1;

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    if (bytes > 0 && numSteps > 0) {
        if (numBodies <= LARGE_BLOCK_THRESHOLD) {
            const int grid = (numBodies + SMALL_BLOCK_SIZE - 1) / SMALL_BLOCK_SIZE;
            for (int step = 0; step < numSteps; ++step) {
                nbodyStepKernel<SMALL_BLOCK_SIZE><<<grid, SMALL_BLOCK_SIZE>>>(src, dst, numBodies);
                Body* tmp = src;
                src = dst;
                dst = tmp;
            }
        } else {
            const int grid = (numBodies + LARGE_BLOCK_SIZE - 1) / LARGE_BLOCK_SIZE;
            for (int step = 0; step < numSteps; ++step) {
                nbodyStepKernel<LARGE_BLOCK_SIZE><<<grid, LARGE_BLOCK_SIZE>>>(src, dst, numBodies);
                Body* tmp = src;
                src = dst;
                dst = tmp;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            fprintf(stderr, "CUDA kernel execution failed\n");
            cudaFree(d_buf0);
            cudaFree(d_buf1);
            return 1;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "CUDA kernel execution failed\n");
        cudaFree(d_buf0);
        cudaFree(d_buf1);
        return 1;
    }

    if (bytes > 0) {
        if (cudaMemcpy(bodies.data(), src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Failed to copy bodies from device\n");
            cudaFree(d_buf0);
            cudaFree(d_buf1);
            return 1;
        }
    }

    if (d_buf0) cudaFree(d_buf0);
    if (d_buf1) cudaFree(d_buf1);

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
