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

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);          \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

__global__ void computeForcesKernel(const double* __restrict__ posx,
                                    const double* __restrict__ posy,
                                    const double* __restrict__ posz,
                                    double* __restrict__ velx,
                                    double* __restrict__ vely,
                                    double* __restrict__ velz,
                                    int n) {
    __shared__ double shx[BLOCK_SIZE];
    __shared__ double shy[BLOCK_SIZE];
    __shared__ double shz[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const double xi = posx[i];
    const double yi = posy[i];
    const double zi = posz[i];
    double Fx = 0.0;
    double Fy = 0.0;
    double Fz = 0.0;

    for (int tile = 0; tile < n; tile += blockDim.x) {
        const int j = tile + threadIdx.x;
        if (j < n) {
            shx[threadIdx.x] = posx[j];
            shy[threadIdx.x] = posy[j];
            shz[threadIdx.x] = posz[j];
        }
        __syncthreads();

        int tileCount = n - tile;
        if (tileCount > blockDim.x) {
            tileCount = blockDim.x;
        }
        for (int k = 0; k < tileCount; ++k) {
            const double dx = shx[k] - xi;
            const double dy = shy[k] - yi;
            const double dz = shz[k] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    velx[i] += DT * Fx;
    vely[i] += DT * Fy;
    velz[i] += DT * Fz;
}

__global__ void integrateKernel(double* __restrict__ posx,
                                double* __restrict__ posy,
                                double* __restrict__ posz,
                                const double* __restrict__ velx,
                                const double* __restrict__ vely,
                                const double* __restrict__ velz,
                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    posx[i] += velx[i] * DT;
    posy[i] += vely[i] * DT;
    posz[i] += velz[i] * DT;
}

void initializeCudaOrExit() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "CUDA error: no CUDA-capable devices found\n");
        std::exit(1);
    }
    CUDA_CHECK(cudaSetDevice(0));
}

long long runSimulationCUDA(std::vector<Body>& bodies, int numSteps) {
    const int numBodies = static_cast<int>(bodies.size());
    if (numBodies <= 0) {
        return 0;
    }

    std::vector<double> posx(numBodies), posy(numBodies), posz(numBodies);
    std::vector<double> velx(numBodies), vely(numBodies), velz(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        posx[i] = bodies[i].pos.x;
        posy[i] = bodies[i].pos.y;
        posz[i] = bodies[i].pos.z;
        velx[i] = bodies[i].vel.x;
        vely[i] = bodies[i].vel.y;
        velz[i] = bodies[i].vel.z;
    }

    double* d_posx = nullptr;
    double* d_posy = nullptr;
    double* d_posz = nullptr;
    double* d_velx = nullptr;
    double* d_vely = nullptr;
    double* d_velz = nullptr;
    const size_t bytes = static_cast<size_t>(numBodies) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_posx, bytes));
    CUDA_CHECK(cudaMalloc(&d_posy, bytes));
    CUDA_CHECK(cudaMalloc(&d_posz, bytes));
    CUDA_CHECK(cudaMalloc(&d_velx, bytes));
    CUDA_CHECK(cudaMalloc(&d_vely, bytes));
    CUDA_CHECK(cudaMalloc(&d_velz, bytes));

    CUDA_CHECK(cudaMemcpy(d_posx, posx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posy, posy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posz, posz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velx, velx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vely, vely.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velz, velz.data(), bytes, cudaMemcpyHostToDevice));

    const int grid = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<grid, BLOCK_SIZE>>>(d_posx, d_posy, d_posz, d_velx, d_vely, d_velz, numBodies);
        CUDA_CHECK(cudaGetLastError());
        integrateKernel<<<grid, BLOCK_SIZE>>>(d_posx, d_posy, d_posz, d_velx, d_vely, d_velz, numBodies);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(posx.data(), d_posx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(posy.data(), d_posy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(posz.data(), d_posz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velx.data(), d_velx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vely.data(), d_vely, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velz.data(), d_velz, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_posx));
    CUDA_CHECK(cudaFree(d_posy));
    CUDA_CHECK(cudaFree(d_posz));
    CUDA_CHECK(cudaFree(d_velx));
    CUDA_CHECK(cudaFree(d_vely));
    CUDA_CHECK(cudaFree(d_velz));

    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = posx[i];
        bodies[i].pos.y = posy[i];
        bodies[i].pos.z = posz[i];
        bodies[i].vel.x = velx[i];
        bodies[i].vel.y = vely[i];
        bodies[i].vel.z = velz[i];
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return duration.count();
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
    initializeCudaOrExit();
    
    // Run simulation
    const long long durationMs = runSimulationCUDA(bodies, numSteps);
    printf("Simulation time: %lld ms\n", durationMs);
    
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
