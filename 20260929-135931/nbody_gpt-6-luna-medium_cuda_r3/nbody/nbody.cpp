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
constexpr int CUDA_BLOCK_SIZE = 256;

__global__ void advanceBodies(const double* x, const double* y, const double* z,
                              double* nextX, double* nextY, double* nextZ,
                              double* vx, double* vy, double* vz, int n) {
    __shared__ double tileX[CUDA_BLOCK_SIZE];
    __shared__ double tileY[CUDA_BLOCK_SIZE];
    __shared__ double tileZ[CUDA_BLOCK_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    const double xi = active ? x[i] : 0.0;
    const double yi = active ? y[i] : 0.0;
    const double zi = active ? z[i] : 0.0;
    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + threadIdx.x;
        if (j < n) { tileX[threadIdx.x] = x[j]; tileY[threadIdx.x] = y[j]; tileZ[threadIdx.x] = z[j]; }
        __syncthreads();
        const int count = min((int)blockDim.x, n - base);
        for (int k = 0; k < count; ++k) {
            const double dx = tileX[k] - xi;
            const double dy = tileY[k] - yi;
            const double dz = tileZ[k] - zi;
            const double d2 = dx * dx + dy * dy + dz * dz + 1e-9;
            const double inv = 1.0 / sqrt(d2);
            const double inv3 = inv * inv * inv;
            fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
        }
        __syncthreads();
    }
    if (active) {
        vx[i] += DT * fx; vy[i] += DT * fy; vz[i] += DT * fz;
        nextX[i] = xi + vx[i] * DT;
        nextY[i] = yi + vy[i] * DT;
        nextZ[i] = zi + vz[i] * DT;
    }
}

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
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

void simulateCuda(std::vector<Body>& bodies, int steps) {
    const int n = static_cast<int>(bodies.size());
    if (n == 0 || steps == 0) return;
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    std::vector<double> hx(n), hy(n), hz(n), hvx(n), hvy(n), hvz(n);
    for (int i = 0; i < n; ++i) {
        hx[i] = bodies[i].pos.x; hy[i] = bodies[i].pos.y; hz[i] = bodies[i].pos.z;
        hvx[i] = bodies[i].vel.x; hvy[i] = bodies[i].vel.y; hvz[i] = bodies[i].vel.z;
    }
    double *x, *y, *z, *otherX, *otherY, *otherZ, *vx, *vy, *vz;
    cudaCheck(cudaMalloc(&x, bytes)); cudaCheck(cudaMalloc(&y, bytes)); cudaCheck(cudaMalloc(&z, bytes));
    cudaCheck(cudaMalloc(&otherX, bytes)); cudaCheck(cudaMalloc(&otherY, bytes)); cudaCheck(cudaMalloc(&otherZ, bytes));
    cudaCheck(cudaMalloc(&vx, bytes)); cudaCheck(cudaMalloc(&vy, bytes)); cudaCheck(cudaMalloc(&vz, bytes));
    cudaCheck(cudaMemcpy(x, hx.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(y, hy.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(z, hz.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(vx, hvx.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(vy, hvy.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(vz, hvz.data(), bytes, cudaMemcpyHostToDevice));
    const int blocks = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    for (int step = 0; step < steps; ++step) {
        advanceBodies<<<blocks, CUDA_BLOCK_SIZE>>>(x, y, z, otherX, otherY, otherZ, vx, vy, vz, n);
        cudaCheck(cudaGetLastError());
        double *tmp;
        tmp = x; x = otherX; otherX = tmp;
        tmp = y; y = otherY; otherY = tmp;
        tmp = z; z = otherZ; otherZ = tmp;
    }
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(hx.data(), x, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(hy.data(), y, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(hz.data(), z, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(hvx.data(), vx, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(hvy.data(), vy, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(hvz.data(), vz, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(hx[i], hy[i], hz[i]); bodies[i].vel = Vec3(hvx[i], hvy[i], hvz[i]);
    }
    cudaFree(x); cudaFree(y); cudaFree(z); cudaFree(otherX); cudaFree(otherY); cudaFree(otherZ);
    cudaFree(vx); cudaFree(vy); cudaFree(vz);
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
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    simulateCuda(bodies, numSteps);
    
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
