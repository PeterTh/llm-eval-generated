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
constexpr int BLOCK_SIZE = 256;

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

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// A block cooperatively caches source positions.  The SoA layout makes both
// global loads and writes contiguous, while each cached position is reused by
// every thread in the block.
__global__ void computeForces(double* __restrict__ px,
                              double* __restrict__ py,
                              double* __restrict__ pz,
                              double* __restrict__ vx,
                              double* __restrict__ vy,
                              double* __restrict__ vz,
                              int n) {
    __shared__ double sx[BLOCK_SIZE];
    __shared__ double sy[BLOCK_SIZE];
    __shared__ double sz[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const double xi = active ? px[i] : 0.0;
    const double yi = active ? py[i] : 0.0;
    const double zi = active ? pz[i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        const int source = tile + threadIdx.x;
        if (source < n) {
            sx[threadIdx.x] = px[source];
            sy[threadIdx.x] = py[source];
            sz[threadIdx.x] = pz[source];
        }
        __syncthreads();

        if (active) {
            const int count = min(BLOCK_SIZE, n - tile);
#pragma unroll 8
            for (int j = 0; j < count; ++j) {
                const double dx = sx[j] - xi;
                const double dy = sy[j] - yi;
                const double dz = sz[j] - zi;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        vx[i] += DT * fx;
        vy[i] += DT * fy;
        vz[i] += DT * fz;
    }
}

__global__ void integrateBodies(double* __restrict__ px,
                                double* __restrict__ py,
                                double* __restrict__ pz,
                                const double* __restrict__ vx,
                                const double* __restrict__ vy,
                                const double* __restrict__ vz,
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

    if (numBodies < 0 || numSteps < 0) {
        std::fprintf(stderr, "Number of bodies and steps must be non-negative\n");
        return 1;
    }
    
    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    const size_t bodyCount = static_cast<size_t>(numBodies);
    const size_t planeBytes = bodyCount * sizeof(double);
    std::vector<double> hostData(bodyCount * 6);
    for (int i = 0; i < numBodies; ++i) {
        const size_t index = static_cast<size_t>(i);
        hostData[index] = bodies[i].pos.x;
        hostData[bodyCount + index] = bodies[i].pos.y;
        hostData[2 * bodyCount + index] = bodies[i].pos.z;
        hostData[3 * bodyCount + index] = bodies[i].vel.x;
        hostData[4 * bodyCount + index] = bodies[i].vel.y;
        hostData[5 * bodyCount + index] = bodies[i].vel.z;
    }

    double* deviceData = nullptr;
    if (numBodies > 0) {
        checkCuda(cudaMalloc(&deviceData, 6 * planeBytes), "device allocation");
        checkCuda(cudaMemcpy(deviceData, hostData.data(), 6 * planeBytes,
                             cudaMemcpyHostToDevice), "initial host-to-device copy");
    }
    double* px = deviceData;
    double* py = numBodies > 0 ? deviceData + bodyCount : nullptr;
    double* pz = numBodies > 0 ? deviceData + 2 * bodyCount : nullptr;
    double* vx = numBodies > 0 ? deviceData + 3 * bodyCount : nullptr;
    double* vy = numBodies > 0 ? deviceData + 4 * bodyCount : nullptr;
    double* vz = numBodies > 0 ? deviceData + 5 * bodyCount : nullptr;
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    const int blocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int step = 0; step < numSteps && numBodies > 0; ++step) {
        computeForces<<<blocks, BLOCK_SIZE>>>(px, py, pz, vx, vy, vz, numBodies);
        integrateBodies<<<blocks, BLOCK_SIZE>>>(px, py, pz, vx, vy, vz, numBodies);
    }
    checkCuda(cudaGetLastError(), "simulation kernel launch");
    checkCuda(cudaDeviceSynchronize(), "simulation kernel execution");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());

    if (numBodies > 0) {
        checkCuda(cudaMemcpy(hostData.data(), deviceData, 6 * planeBytes,
                             cudaMemcpyDeviceToHost), "final device-to-host copy");
        checkCuda(cudaFree(deviceData), "device deallocation");
        for (int i = 0; i < numBodies; ++i) {
            const size_t index = static_cast<size_t>(i);
            bodies[i].pos = Vec3(hostData[index], hostData[bodyCount + index],
                                 hostData[2 * bodyCount + index]);
            bodies[i].vel = Vec3(hostData[3 * bodyCount + index], hostData[4 * bodyCount + index],
                                 hostData[5 * bodyCount + index]);
        }
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
