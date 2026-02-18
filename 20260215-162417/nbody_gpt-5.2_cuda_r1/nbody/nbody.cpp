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

static inline void cudaCheck(cudaError_t err, const char* call, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d: %s failed with %s\n", file, line, call, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

__global__ void computeForcesKernel(const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
                                   double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz, int n) {
    extern __shared__ double sh[];
    double* sx = sh;
    double* sy = sh + blockDim.x;
    double* sz = sh + 2 * blockDim.x;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double xi = x[i];
    const double yi = y[i];
    const double zi = z[i];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < n; tile += blockDim.x) {
        const int j = tile + threadIdx.x;
        if (j < n) {
            sx[threadIdx.x] = x[j];
            sy[threadIdx.x] = y[j];
            sz[threadIdx.x] = z[j];
        }
        __syncthreads();

        const int tileSize = min(blockDim.x, n - tile);
        for (int k = 0; k < tileSize; ++k) {
            const double dx = sx[k] - xi;
            const double dy = sy[k] - yi;
            const double dz = sz[k] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx = fma(dx, invDist3, Fx);
            Fy = fma(dy, invDist3, Fy);
            Fz = fma(dz, invDist3, Fz);
        }
        __syncthreads();
    }

    vx[i] += DT * Fx;
    vy[i] += DT * Fy;
    vz[i] += DT * Fz;
}

__global__ void integrateKernel(double* __restrict__ x, double* __restrict__ y, double* __restrict__ z,
                               const double* __restrict__ vx, const double* __restrict__ vy, const double* __restrict__ vz,
                               int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += vx[i] * DT;
    y[i] += vy[i] * DT;
    z[i] += vz[i] * DT;
}

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

void computeForces(std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    
    for (size_t i = 0; i < n; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        bodies[i].vel.x += DT * Fx;
        bodies[i].vel.y += DT * Fy;
        bodies[i].vel.z += DT * Fz;
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    for (auto& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
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

    // Pack AoS -> SoA for coalesced GPU access
    std::vector<double> hx(numBodies), hy(numBodies), hz(numBodies);
    std::vector<double> hvx(numBodies), hvy(numBodies), hvz(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        hx[i] = bodies[i].pos.x;
        hy[i] = bodies[i].pos.y;
        hz[i] = bodies[i].pos.z;
        hvx[i] = bodies[i].vel.x;
        hvy[i] = bodies[i].vel.y;
        hvz[i] = bodies[i].vel.z;
    }

    double *dx = nullptr, *dy = nullptr, *dz = nullptr;
    double *dvx = nullptr, *dvy = nullptr, *dvz = nullptr;
    const size_t bytes = static_cast<size_t>(numBodies) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMalloc(&dz, bytes));
    CUDA_CHECK(cudaMalloc(&dvx, bytes));
    CUDA_CHECK(cudaMalloc(&dvy, bytes));
    CUDA_CHECK(cudaMalloc(&dvz, bytes));

    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dz, hz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvx, hvx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvy, hvy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvz, hvz.data(), bytes, cudaMemcpyHostToDevice));

    // Prefer shared memory (tiling) and 8-byte banks for doubles.
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

    const int block = 256;
    const int grid = (numBodies + block - 1) / block;
    const size_t shmemBytes = 3ull * static_cast<size_t>(block) * sizeof(double);

    // Run simulation (GPU)
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<grid, block, shmemBytes>>>(dx, dy, dz, dvx, dvy, dvz, numBodies);
        CUDA_CHECK(cudaGetLastError());
        integrateKernel<<<grid, block>>>(dx, dy, dz, dvx, dvy, dvz, numBodies);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(hx.data(), dx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hz.data(), dz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvx.data(), dvx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvy.data(), dvy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvz.data(), dvz, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dz));
    CUDA_CHECK(cudaFree(dvx));
    CUDA_CHECK(cudaFree(dvy));
    CUDA_CHECK(cudaFree(dvz));

    // Unpack SoA -> AoS
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = hx[i];
        bodies[i].pos.y = hy[i];
        bodies[i].pos.z = hz[i];
        bodies[i].vel.x = hvx[i];
        bodies[i].vel.y = hvy[i];
        bodies[i].vel.z = hvz[i];
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
    
    return 0;
}
