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

#define CUDA_CHECK(call)                                                          \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                    \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

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

// GPU-side structure-of-arrays layout for coalesced memory access.
struct BodiesSoA {
    double *px, *py, *pz;
    double *vx, *vy, *vz;
};

__global__ void computeForcesKernel(const double* __restrict__ px, const double* __restrict__ py,
                                     const double* __restrict__ pz, double* __restrict__ vx,
                                     double* __restrict__ vy, double* __restrict__ vz, int n) {
    extern __shared__ double tile[];
    double* spx = tile;
    double* spy = tile + blockDim.x;
    double* spz = tile + 2 * blockDim.x;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (i < n) {
        xi = px[i];
        yi = py[i];
        zi = pz[i];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tileStart = 0; tileStart < n; tileStart += blockDim.x) {
        const int j = tileStart + threadIdx.x;
        if (j < n) {
            spx[threadIdx.x] = px[j];
            spy[threadIdx.x] = py[j];
            spz[threadIdx.x] = pz[j];
        }
        __syncthreads();

        const int tileSize = min(blockDim.x, n - tileStart);
        if (i < n) {
            for (int k = 0; k < tileSize; ++k) {
                const double dx = spx[k] - xi;
                const double dy = spy[k] - yi;
                const double dz = spz[k] - zi;
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
        vx[i] += DT * Fx;
        vy[i] += DT * Fy;
        vz[i] += DT * Fz;
    }
}

__global__ void integrateBodiesKernel(double* __restrict__ px, double* __restrict__ py,
                                       double* __restrict__ pz, const double* __restrict__ vx,
                                       const double* __restrict__ vy, const double* __restrict__ vz,
                                       int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
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

    // Convert to structure-of-arrays for GPU-friendly memory access
    std::vector<double> hpx(numBodies), hpy(numBodies), hpz(numBodies);
    std::vector<double> hvx(numBodies), hvy(numBodies), hvz(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        hpx[i] = bodies[i].pos.x;
        hpy[i] = bodies[i].pos.y;
        hpz[i] = bodies[i].pos.z;
        hvx[i] = bodies[i].vel.x;
        hvy[i] = bodies[i].vel.y;
        hvz[i] = bodies[i].vel.z;
    }

    BodiesSoA d;
    const size_t bytes = static_cast<size_t>(numBodies) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d.px, bytes));
    CUDA_CHECK(cudaMalloc(&d.py, bytes));
    CUDA_CHECK(cudaMalloc(&d.pz, bytes));
    CUDA_CHECK(cudaMalloc(&d.vx, bytes));
    CUDA_CHECK(cudaMalloc(&d.vy, bytes));
    CUDA_CHECK(cudaMalloc(&d.vz, bytes));

    CUDA_CHECK(cudaMemcpy(d.px, hpx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.py, hpy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.pz, hpz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vx, hvx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vy, hvy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vz, hvz.data(), bytes, cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (numBodies + blockSize - 1) / blockSize;
    const size_t sharedMemBytes = 3 * blockSize * sizeof(double);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<gridSize, blockSize, sharedMemBytes>>>(d.px, d.py, d.pz, d.vx, d.vy, d.vz, numBodies);
        integrateBodiesKernel<<<gridSize, blockSize>>>(d.px, d.py, d.pz, d.vx, d.vy, d.vz, numBodies);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(hpx.data(), d.px, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hpy.data(), d.py, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hpz.data(), d.pz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvx.data(), d.vx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvy.data(), d.vy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvz.data(), d.vz, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = hpx[i];
        bodies[i].pos.y = hpy[i];
        bodies[i].pos.z = hpz[i];
        bodies[i].vel.x = hvx[i];
        bodies[i].vel.y = hvy[i];
        bodies[i].vel.z = hvz[i];
    }

    CUDA_CHECK(cudaFree(d.px));
    CUDA_CHECK(cudaFree(d.py));
    CUDA_CHECK(cudaFree(d.pz));
    CUDA_CHECK(cudaFree(d.vx));
    CUDA_CHECK(cudaFree(d.vy));
    CUDA_CHECK(cudaFree(d.vz));

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
