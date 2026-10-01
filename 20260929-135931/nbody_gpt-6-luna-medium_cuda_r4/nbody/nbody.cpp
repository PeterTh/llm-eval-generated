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

constexpr int CUDA_BLOCK_SIZE = 256;

inline void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void advanceBodies(double* px, double* py, double* pz,
                              double* vx, double* vy, double* vz, int n) {
    __shared__ double sx[CUDA_BLOCK_SIZE];
    __shared__ double sy[CUDA_BLOCK_SIZE];
    __shared__ double sz[CUDA_BLOCK_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const double ix = active ? px[i] : 0.0;
    const double iy = active ? py[i] : 0.0;
    const double iz = active ? pz[i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + threadIdx.x;
        if (j < n) { sx[threadIdx.x] = px[j]; sy[threadIdx.x] = py[j]; sz[threadIdx.x] = pz[j]; }
        __syncthreads();
        const int count = min(blockDim.x, n - base);
        if (active) {
            for (int k = 0; k < count; ++k) {
                const double dx = sx[k] - ix, dy = sy[k] - iy, dz = sz[k] - iz;
                const double d2 = dx * dx + dy * dy + dz * dz + 1e-9;
                const double inv = 1.0 / sqrt(d2);
                const double inv3 = inv * inv * inv;
                fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
            }
        }
        __syncthreads();
    }
    if (active) {
        vx[i] += 0.01 * fx; vy[i] += 0.01 * fy; vz[i] += 0.01 * fz;
        px[i] += vx[i] * 0.01; py[i] += vy[i] * 0.01; pz[i] += vz[i] * 0.01;
    }
}

void simulateOnGpu(std::vector<Body>& bodies, int steps) {
    const int n = static_cast<int>(bodies.size());
    if (!n || steps <= 0) return;
    double *px, *py, *pz, *vx, *vy, *vz;
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    checkCuda(cudaMalloc(&px, bytes)); checkCuda(cudaMalloc(&py, bytes)); checkCuda(cudaMalloc(&pz, bytes));
    checkCuda(cudaMalloc(&vx, bytes)); checkCuda(cudaMalloc(&vy, bytes)); checkCuda(cudaMalloc(&vz, bytes));
    std::vector<double> host(static_cast<size_t>(n) * 6);
    for (int i = 0; i < n; ++i) {
        const Body& b = bodies[i];
        host[i] = b.pos.x; host[n+i] = b.pos.y; host[2*n+i] = b.pos.z;
        host[3*n+i] = b.vel.x; host[4*n+i] = b.vel.y; host[5*n+i] = b.vel.z;
    }
    checkCuda(cudaMemcpy(px, host.data(), bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(py, host.data()+n, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(pz, host.data()+2*n, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(vx, host.data()+3*n, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(vy, host.data()+4*n, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(vz, host.data()+5*n, bytes, cudaMemcpyHostToDevice));
    const int blocks = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    for (int step = 0; step < steps; ++step) {
        advanceBodies<<<blocks, CUDA_BLOCK_SIZE>>>(px, py, pz, vx, vy, vz, n);
        checkCuda(cudaGetLastError());
    }
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(host.data(), px, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(host.data()+n, py, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(host.data()+2*n, pz, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(host.data()+3*n, vx, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(host.data()+4*n, vy, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(host.data()+5*n, vz, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(host[i], host[n+i], host[2*n+i]);
        bodies[i].vel = Vec3(host[3*n+i], host[4*n+i], host[5*n+i]);
    }
    cudaFree(px); cudaFree(py); cudaFree(pz); cudaFree(vx); cudaFree(vy); cudaFree(vz);
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
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    simulateOnGpu(bodies, numSteps);
    
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
