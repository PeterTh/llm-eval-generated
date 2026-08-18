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

constexpr int BLOCK_SIZE = 256;

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

struct DeviceBodies {
    double *px, *py, *pz;
    double *vx, *vy, *vz;
};

static DeviceBodies allocateDeviceBodies(size_t n) {
    DeviceBodies d{};
    const size_t bytes = n * sizeof(double);
    cudaCheck(cudaMalloc(&d.px, bytes), "position allocation");
    cudaCheck(cudaMalloc(&d.py, bytes), "position allocation");
    cudaCheck(cudaMalloc(&d.pz, bytes), "position allocation");
    cudaCheck(cudaMalloc(&d.vx, bytes), "velocity allocation");
    cudaCheck(cudaMalloc(&d.vy, bytes), "velocity allocation");
    cudaCheck(cudaMalloc(&d.vz, bytes), "velocity allocation");
    return d;
}

static void freeDeviceBodies(DeviceBodies& d) {
    cudaFree(d.px); cudaFree(d.py); cudaFree(d.pz);
    cudaFree(d.vx); cudaFree(d.vy); cudaFree(d.vz);
}

__global__ __launch_bounds__(BLOCK_SIZE)
void advanceBodies(const DeviceBodies in, DeviceBodies out, const int n) {
    __shared__ double tileX[BLOCK_SIZE];
    __shared__ double tileY[BLOCK_SIZE];
    __shared__ double tileZ[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    double x = 0.0, y = 0.0, z = 0.0;
    if (active) {
        x = in.px[i]; y = in.py[i]; z = in.pz[i];
    }
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        if (j < n) {
            tileX[threadIdx.x] = in.px[j];
            tileY[threadIdx.x] = in.py[j];
            tileZ[threadIdx.x] = in.pz[j];
        }
        __syncthreads();

        if (active) {
            const int count = min(BLOCK_SIZE, n - base);
#pragma unroll 8
            for (int k = 0; k < count; ++k) {
                const double dx = tileX[k] - x;
                const double dy = tileY[k] - y;
                const double dz = tileZ[k] - z;
                const double invDist = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        const double vx = in.vx[i] + DT * fx;
        const double vy = in.vy[i] + DT * fy;
        const double vz = in.vz[i] + DT * fz;
        out.vx[i] = vx; out.vy[i] = vy; out.vz[i] = vz;
        out.px[i] = x + vx * DT;
        out.py[i] = y + vy * DT;
        out.pz[i] = z + vz * DT;
    }
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
    if (numBodies <= 0 || numSteps < 0) {
        std::fprintf(stderr, "Number of bodies must be positive and steps non-negative\n");
        return 1;
    }
    
    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // A structure-of-arrays layout gives coalesced global-memory accesses on
    // the GPU.  Keep two states so every timestep reads one immutable snapshot.
    std::vector<double> px(numBodies), py(numBodies), pz(numBodies);
    std::vector<double> vx(numBodies), vy(numBodies), vz(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        px[i] = bodies[i].pos.x; py[i] = bodies[i].pos.y; pz[i] = bodies[i].pos.z;
        vx[i] = bodies[i].vel.x; vy[i] = bodies[i].vel.y; vz[i] = bodies[i].vel.z;
    }

    DeviceBodies current = allocateDeviceBodies(numBodies);
    DeviceBodies next = allocateDeviceBodies(numBodies);
    const size_t bytes = static_cast<size_t>(numBodies) * sizeof(double);
    cudaCheck(cudaMemcpy(current.px, px.data(), bytes, cudaMemcpyHostToDevice), "position upload");
    cudaCheck(cudaMemcpy(current.py, py.data(), bytes, cudaMemcpyHostToDevice), "position upload");
    cudaCheck(cudaMemcpy(current.pz, pz.data(), bytes, cudaMemcpyHostToDevice), "position upload");
    cudaCheck(cudaMemcpy(current.vx, vx.data(), bytes, cudaMemcpyHostToDevice), "velocity upload");
    cudaCheck(cudaMemcpy(current.vy, vy.data(), bytes, cudaMemcpyHostToDevice), "velocity upload");
    cudaCheck(cudaMemcpy(current.vz, vz.data(), bytes, cudaMemcpyHostToDevice), "velocity upload");
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    const int blocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int step = 0; step < numSteps; ++step) {
        advanceBodies<<<blocks, BLOCK_SIZE>>>(current, next, numBodies);
        std::swap(current, next);
    }
    cudaCheck(cudaGetLastError(), "simulation kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "simulation");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());

    cudaCheck(cudaMemcpy(px.data(), current.px, bytes, cudaMemcpyDeviceToHost), "position download");
    cudaCheck(cudaMemcpy(py.data(), current.py, bytes, cudaMemcpyDeviceToHost), "position download");
    cudaCheck(cudaMemcpy(pz.data(), current.pz, bytes, cudaMemcpyDeviceToHost), "position download");
    cudaCheck(cudaMemcpy(vx.data(), current.vx, bytes, cudaMemcpyDeviceToHost), "velocity download");
    cudaCheck(cudaMemcpy(vy.data(), current.vy, bytes, cudaMemcpyDeviceToHost), "velocity download");
    cudaCheck(cudaMemcpy(vz.data(), current.vz, bytes, cudaMemcpyDeviceToHost), "velocity download");
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos = Vec3(px[i], py[i], pz[i]);
        bodies[i].vel = Vec3(vx[i], vy[i], vz[i]);
    }
    freeDeviceBodies(current);
    freeDeviceBodies(next);
    
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
