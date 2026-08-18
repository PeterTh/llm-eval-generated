#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

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

namespace {

constexpr int BLOCK_SIZE = 256;

bool checkCuda(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                 cudaGetErrorString(result));
    return false;
}

// Positions are staged in shared memory a tile at a time.  Compared with the
// host Body layout, the six separate arrays make every global-memory access by
// a warp contiguous and let a position be reused by all threads in a block.
__global__ __launch_bounds__(BLOCK_SIZE)
void computeForcesKernel(const double* __restrict__ px,
                         const double* __restrict__ py,
                         const double* __restrict__ pz,
                         double* __restrict__ vx,
                         double* __restrict__ vy,
                         double* __restrict__ vz, int n) {
    __shared__ double tileX[BLOCK_SIZE];
    __shared__ double tileY[BLOCK_SIZE];
    __shared__ double tileZ[BLOCK_SIZE];

    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool active = i < n;
    const double xi = active ? px[i] : 0.0;
    const double yi = active ? py[i] : 0.0;
    const double zi = active ? pz[i] : 0.0;
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;

    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int source = base + threadIdx.x;
        if (source < n) {
            tileX[threadIdx.x] = px[source];
            tileY[threadIdx.x] = py[source];
            tileZ[threadIdx.x] = pz[source];
        }
        __syncthreads();

        const int count = min(BLOCK_SIZE, n - base);
        if (active) {
            for (int j = 0; j < count; ++j) {
                const double dx = tileX[j] - xi;
                const double dy = tileY[j] - yi;
                const double dz = tileZ[j] - zi;
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

__global__ __launch_bounds__(BLOCK_SIZE)
void integrateBodiesKernel(double* __restrict__ px,
                           double* __restrict__ py,
                           double* __restrict__ pz,
                           const double* __restrict__ vx,
                           const double* __restrict__ vy,
                           const double* __restrict__ vz, int n) {
    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
    }
}

}  // namespace

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
    
    if (numBodies < 0 || numSteps < 0) {
        std::fprintf(stderr, "Number of bodies and steps must be non-negative\n");
        return 1;
    }

    // Initialize bodies on the host, preserving the original random sequence.
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    double* deviceData = nullptr;
    const size_t arrayBytes = static_cast<size_t>(numBodies) * sizeof(double);
    std::vector<double> hostData(static_cast<size_t>(numBodies) * 6);
    for (int i = 0; i < numBodies; ++i) {
        hostData[i] = bodies[i].pos.x;
        hostData[numBodies + i] = bodies[i].pos.y;
        hostData[2 * numBodies + i] = bodies[i].pos.z;
        hostData[3 * numBodies + i] = bodies[i].vel.x;
        hostData[4 * numBodies + i] = bodies[i].vel.y;
        hostData[5 * numBodies + i] = bodies[i].vel.z;
    }

    if (numBodies > 0 &&
        (!checkCuda(cudaMalloc(&deviceData, arrayBytes * 6), "allocation") ||
         !checkCuda(cudaMemcpy(deviceData, hostData.data(), arrayBytes * 6,
                               cudaMemcpyHostToDevice), "initial upload"))) {
        cudaFree(deviceData);
        return 1;
    }

    double* px = deviceData;
    double* py = numBodies > 0 ? px + numBodies : nullptr;
    double* pz = numBodies > 0 ? py + numBodies : nullptr;
    double* vx = numBodies > 0 ? pz + numBodies : nullptr;
    double* vy = numBodies > 0 ? vx + numBodies : nullptr;
    double* vz = numBodies > 0 ? vy + numBodies : nullptr;
    const int blocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Run the complete simulation on the GPU. Kernel launches in the default
    // stream provide the global synchronization required between phases.
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps && numBodies > 0; ++step) {
        computeForcesKernel<<<blocks, BLOCK_SIZE>>>(px, py, pz, vx, vy, vz,
                                                    numBodies);
        integrateBodiesKernel<<<blocks, BLOCK_SIZE>>>(px, py, pz, vx, vy, vz,
                                                      numBodies);
    }
    if (numBodies > 0 && numSteps > 0 &&
        (!checkCuda(cudaGetLastError(), "kernel launch") ||
         !checkCuda(cudaDeviceSynchronize(), "simulation"))) {
        cudaFree(deviceData);
        return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    if (numBodies > 0 && (printResults || validate)) {
        if (!checkCuda(cudaMemcpy(hostData.data(), deviceData, arrayBytes * 6,
                                  cudaMemcpyDeviceToHost), "result download")) {
            cudaFree(deviceData);
            return 1;
        }
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos.x = hostData[i];
            bodies[i].pos.y = hostData[numBodies + i];
            bodies[i].pos.z = hostData[2 * numBodies + i];
            bodies[i].vel.x = hostData[3 * numBodies + i];
            bodies[i].vel.y = hostData[4 * numBodies + i];
            bodies[i].vel.z = hostData[5 * numBodies + i];
        }
    }
    cudaFree(deviceData);

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
