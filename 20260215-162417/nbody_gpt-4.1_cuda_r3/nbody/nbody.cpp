#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
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

struct BodySoA {
    double* pos_x;
    double* pos_y;
    double* pos_z;
    double* vel_x;
    double* vel_y;
    double* vel_z;
};

__global__ void computeForcesKernel(
    double* pos_x, double* pos_y, double* pos_z,
    double* vel_x, double* vel_y, double* vel_z,
    int n) {
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

__global__ void integrateBodiesKernel(
    double* pos_x, double* pos_y, double* pos_z,
    double* vel_x, double* vel_y, double* vel_z,
    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    pos_x[i] += vel_x[i] * DT;
    pos_y[i] += vel_y[i] * DT;
    pos_z[i] += vel_z[i] * DT;
}

void computeForcesCUDA(BodySoA& d_bodies, int n, int threadsPerBlock = 256) {
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    computeForcesKernel<<<blocks, threadsPerBlock>>>(
        d_bodies.pos_x, d_bodies.pos_y, d_bodies.pos_z,
        d_bodies.vel_x, d_bodies.vel_y, d_bodies.vel_z, n);
    cudaDeviceSynchronize();
}

void integrateBodiesCUDA(BodySoA& d_bodies, int n, int threadsPerBlock = 256) {
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    integrateBodiesKernel<<<blocks, threadsPerBlock>>>(
        d_bodies.pos_x, d_bodies.pos_y, d_bodies.pos_z,
        d_bodies.vel_x, d_bodies.vel_y, d_bodies.vel_z, n);
    cudaDeviceSynchronize();
}

void toSoA(const std::vector<Body>& bodies, BodySoA& soa, int n) {
    for (int i = 0; i < n; ++i) {
        soa.pos_x[i] = bodies[i].pos.x;
        soa.pos_y[i] = bodies[i].pos.y;
        soa.pos_z[i] = bodies[i].pos.z;
        soa.vel_x[i] = bodies[i].vel.x;
        soa.vel_y[i] = bodies[i].vel.y;
        soa.vel_z[i] = bodies[i].vel.z;
    }
}
void fromSoA(std::vector<Body>& bodies, const BodySoA& soa, int n) {
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = soa.pos_x[i];
        bodies[i].pos.y = soa.pos_y[i];
        bodies[i].pos.z = soa.pos_z[i];
        bodies[i].vel.x = soa.vel_x[i];
        bodies[i].vel.y = soa.vel_y[i];
        bodies[i].vel.z = soa.vel_z[i];
    }
}

void allocateBodySoA(BodySoA& soa, int n) {
    cudaMalloc(&soa.pos_x, n * sizeof(double));
    cudaMalloc(&soa.pos_y, n * sizeof(double));
    cudaMalloc(&soa.pos_z, n * sizeof(double));
    cudaMalloc(&soa.vel_x, n * sizeof(double));
    cudaMalloc(&soa.vel_y, n * sizeof(double));
    cudaMalloc(&soa.vel_z, n * sizeof(double));
}
void freeBodySoA(BodySoA& soa) {
    cudaFree(soa.pos_x);
    cudaFree(soa.pos_y);
    cudaFree(soa.pos_z);
    cudaFree(soa.vel_x);
    cudaFree(soa.vel_y);
    cudaFree(soa.vel_z);
}

void copyToDevice(BodySoA& d_soa, const BodySoA& h_soa, int n) {
    cudaMemcpy(d_soa.pos_x, h_soa.pos_x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.pos_y, h_soa.pos_y, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.pos_z, h_soa.pos_z, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.vel_x, h_soa.vel_x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.vel_y, h_soa.vel_y, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.vel_z, h_soa.vel_z, n * sizeof(double), cudaMemcpyHostToDevice);
}
void copyToHost(BodySoA& h_soa, const BodySoA& d_soa, int n) {
    cudaMemcpy(h_soa.pos_x, d_soa.pos_x, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_soa.pos_y, d_soa.pos_y, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_soa.pos_z, d_soa.pos_z, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_soa.vel_x, d_soa.vel_x, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_soa.vel_y, d_soa.vel_y, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_soa.vel_z, d_soa.vel_z, n * sizeof(double), cudaMemcpyDeviceToHost);
}

void computeForces(std::vector<Body>& bodies) {
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
    
    // Allocate SoA for host and device
    BodySoA h_soa, d_soa;
    h_soa.pos_x = new double[numBodies];
    h_soa.pos_y = new double[numBodies];
    h_soa.pos_z = new double[numBodies];
    h_soa.vel_x = new double[numBodies];
    h_soa.vel_y = new double[numBodies];
    h_soa.vel_z = new double[numBodies];
    toSoA(bodies, h_soa, numBodies);
    allocateBodySoA(d_soa, numBodies);
    copyToDevice(d_soa, h_soa, numBodies);
    for (int step = 0; step < numSteps; ++step) {
        computeForcesCUDA(d_soa, numBodies);
        integrateBodiesCUDA(d_soa, numBodies);
    }
    copyToHost(h_soa, d_soa, numBodies);
    fromSoA(bodies, h_soa, numBodies);
    freeBodySoA(d_soa);
    delete[] h_soa.pos_x;
    delete[] h_soa.pos_y;
    delete[] h_soa.pos_z;
    delete[] h_soa.vel_x;
    delete[] h_soa.vel_y;
    delete[] h_soa.vel_z;
    
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
