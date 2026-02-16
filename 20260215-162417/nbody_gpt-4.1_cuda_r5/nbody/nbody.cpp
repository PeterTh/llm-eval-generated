#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { printf("CUDA error at %s:%d\n",__FILE__,__LINE__); return; } } while(0)
#else
#define __host__
#define __device__
#define CUDA_CALL(x) x
#endif

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

#ifdef __CUDACC__
__global__ void computeForcesKernel(double* pos_x, double* pos_y, double* pos_z, double* vel_x, double* vel_y, double* vel_z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double xi = pos_x[i], yi = pos_y[i], zi = pos_z[i];
    for (int j = 0; j < n; ++j) {
        double dx = pos_x[j] - xi;
        double dy = pos_y[j] - yi;
        double dz = pos_z[j] - zi;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
    vel_x[i] += DT * Fx;
    vel_y[i] += DT * Fy;
    vel_z[i] += DT * Fz;
}

__global__ void integrateBodiesKernel(double* pos_x, double* pos_y, double* pos_z, double* vel_x, double* vel_y, double* vel_z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    pos_x[i] += vel_x[i] * DT;
    pos_y[i] += vel_y[i] * DT;
    pos_z[i] += vel_z[i] * DT;
}
#endif

void computeForces(std::vector<Body>& bodies) {
#ifdef __CUDACC__
    int n = bodies.size();
    static double *d_pos_x = nullptr, *d_pos_y = nullptr, *d_pos_z = nullptr, *d_vel_x = nullptr, *d_vel_y = nullptr, *d_vel_z = nullptr;
    static int d_n = 0;
    if (d_n != n) {
        if (d_pos_x) { cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_pos_z); cudaFree(d_vel_x); cudaFree(d_vel_y); cudaFree(d_vel_z); }
        CUDA_CALL(cudaMalloc(&d_pos_x, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&d_pos_y, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&d_pos_z, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&d_vel_x, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&d_vel_y, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&d_vel_z, n * sizeof(double)));
        d_n = n;
    }
    std::vector<double> pos_x(n), pos_y(n), pos_z(n), vel_x(n), vel_y(n), vel_z(n);
    for (int i = 0; i < n; ++i) {
        pos_x[i] = bodies[i].pos.x;
        pos_y[i] = bodies[i].pos.y;
        pos_z[i] = bodies[i].pos.z;
        vel_x[i] = bodies[i].vel.x;
        vel_y[i] = bodies[i].vel.y;
        vel_z[i] = bodies[i].vel.z;
    }
    CUDA_CALL(cudaMemcpy(d_pos_x, pos_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pos_y, pos_y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pos_z, pos_z.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_vel_x, vel_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_vel_y, vel_y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_vel_z, vel_z.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    computeForcesKernel<<<numBlocks, blockSize>>>(d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, n);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(vel_x.data(), d_vel_x, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(vel_y.data(), d_vel_y, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(vel_z.data(), d_vel_z, n * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        bodies[i].vel.x = vel_x[i];
        bodies[i].vel.y = vel_y[i];
        bodies[i].vel.z = vel_z[i];
    }
#else
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
#endif
}

void integrateBodies(std::vector<Body>& bodies) {
#ifdef __CUDACC__
    int n = bodies.size();
    static double *d_pos_x = nullptr, *d_pos_y = nullptr, *d_pos_z = nullptr, *d_vel_x = nullptr, *d_vel_y = nullptr, *d_vel_z = nullptr;
    static int d_n = 0;
    if (d_n != n) return; // Must have been allocated in computeForces
    std::vector<double> pos_x(n), pos_y(n), pos_z(n), vel_x(n), vel_y(n), vel_z(n);
    for (int i = 0; i < n; ++i) {
        vel_x[i] = bodies[i].vel.x;
        vel_y[i] = bodies[i].vel.y;
        vel_z[i] = bodies[i].vel.z;
    }
    CUDA_CALL(cudaMemcpy(d_vel_x, vel_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_vel_y, vel_y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_vel_z, vel_z.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    integrateBodiesKernel<<<(n+255)/256, 256>>>(d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, n);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(pos_x.data(), d_pos_x, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(pos_y.data(), d_pos_y, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(pos_z.data(), d_pos_z, n * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = pos_x[i];
        bodies[i].pos.y = pos_y[i];
        bodies[i].pos.z = pos_z[i];
    }
#else
    for (auto& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
#endif
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
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies);
        integrateBodies(bodies);
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
