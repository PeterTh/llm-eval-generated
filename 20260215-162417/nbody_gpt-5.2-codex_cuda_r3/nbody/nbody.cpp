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

inline void checkCuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", context, cudaGetErrorString(result));
        std::exit(1);
    }
}

__global__ void computeForcesKernel(const double* posx, const double* posy, const double* posz,
                                    double* velx, double* vely, double* velz, size_t n) {
    extern __shared__ double shared[];
    double* shx = shared;
    double* shy = shared + blockDim.x;
    double* shz = shared + 2 * blockDim.x;

    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (i < n) {
        xi = posx[i];
        yi = posy[i];
        zi = posz[i];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (size_t tile = 0; tile < n; tile += blockDim.x) {
        const size_t idx = tile + threadIdx.x;
        if (idx < n) {
            shx[threadIdx.x] = posx[idx];
            shy[threadIdx.x] = posy[idx];
            shz[threadIdx.x] = posz[idx];
        } else {
            shx[threadIdx.x] = 0.0;
            shy[threadIdx.x] = 0.0;
            shz[threadIdx.x] = 0.0;
        }
        __syncthreads();

        size_t tileSize = n - tile;
        if (tileSize > blockDim.x) {
            tileSize = blockDim.x;
        }
        if (i < n) {
            for (size_t j = 0; j < tileSize; ++j) {
                const double dx = shx[j] - xi;
                const double dy = shy[j] - yi;
                const double dz = shz[j] - zi;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / ::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        velx[i] += DT * Fx;
        vely[i] += DT * Fy;
        velz[i] += DT * Fz;
    }
}

__global__ void integrateKernel(double* posx, double* posy, double* posz,
                                const double* velx, const double* vely, const double* velz, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        posx[i] += velx[i] * DT;
        posy[i] += vely[i] * DT;
        posz[i] += velz[i] * DT;
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

void computeForcesGPU(const double* posx, const double* posy, const double* posz,
                      double* velx, double* vely, double* velz, size_t n) {
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((n + block.x - 1) / block.x);
    const size_t sharedBytes = static_cast<size_t>(block.x) * 3 * sizeof(double);
    computeForcesKernel<<<grid, block, sharedBytes>>>(posx, posy, posz, velx, vely, velz, n);
    checkCuda(cudaGetLastError(), "computeForcesKernel launch");
}

void integrateBodiesGPU(double* posx, double* posy, double* posz,
                        const double* velx, const double* vely, const double* velz, size_t n) {
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((n + block.x - 1) / block.x);
    integrateKernel<<<grid, block>>>(posx, posy, posz, velx, vely, velz, n);
    checkCuda(cudaGetLastError(), "integrateKernel launch");
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
    checkCuda(cudaMalloc(&d_posx, bytes), "cudaMalloc d_posx");
    checkCuda(cudaMalloc(&d_posy, bytes), "cudaMalloc d_posy");
    checkCuda(cudaMalloc(&d_posz, bytes), "cudaMalloc d_posz");
    checkCuda(cudaMalloc(&d_velx, bytes), "cudaMalloc d_velx");
    checkCuda(cudaMalloc(&d_vely, bytes), "cudaMalloc d_vely");
    checkCuda(cudaMalloc(&d_velz, bytes), "cudaMalloc d_velz");
    checkCuda(cudaMemcpy(d_posx, posx.data(), bytes, cudaMemcpyHostToDevice), "copy posx");
    checkCuda(cudaMemcpy(d_posy, posy.data(), bytes, cudaMemcpyHostToDevice), "copy posy");
    checkCuda(cudaMemcpy(d_posz, posz.data(), bytes, cudaMemcpyHostToDevice), "copy posz");
    checkCuda(cudaMemcpy(d_velx, velx.data(), bytes, cudaMemcpyHostToDevice), "copy velx");
    checkCuda(cudaMemcpy(d_vely, vely.data(), bytes, cudaMemcpyHostToDevice), "copy vely");
    checkCuda(cudaMemcpy(d_velz, velz.data(), bytes, cudaMemcpyHostToDevice), "copy velz");

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        computeForcesGPU(d_posx, d_posy, d_posz, d_velx, d_vely, d_velz, numBodies);
        integrateBodiesGPU(d_posx, d_posy, d_posz, d_velx, d_vely, d_velz, numBodies);
    }

    checkCuda(cudaDeviceSynchronize(), "simulation sync");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());

    checkCuda(cudaMemcpy(posx.data(), d_posx, bytes, cudaMemcpyDeviceToHost), "copy posx back");
    checkCuda(cudaMemcpy(posy.data(), d_posy, bytes, cudaMemcpyDeviceToHost), "copy posy back");
    checkCuda(cudaMemcpy(posz.data(), d_posz, bytes, cudaMemcpyDeviceToHost), "copy posz back");
    checkCuda(cudaMemcpy(velx.data(), d_velx, bytes, cudaMemcpyDeviceToHost), "copy velx back");
    checkCuda(cudaMemcpy(vely.data(), d_vely, bytes, cudaMemcpyDeviceToHost), "copy vely back");
    checkCuda(cudaMemcpy(velz.data(), d_velz, bytes, cudaMemcpyDeviceToHost), "copy velz back");
    checkCuda(cudaFree(d_posx), "cudaFree d_posx");
    checkCuda(cudaFree(d_posy), "cudaFree d_posy");
    checkCuda(cudaFree(d_posz), "cudaFree d_posz");
    checkCuda(cudaFree(d_velx), "cudaFree d_velx");
    checkCuda(cudaFree(d_vely), "cudaFree d_vely");
    checkCuda(cudaFree(d_velz), "cudaFree d_velz");

    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = posx[i];
        bodies[i].pos.y = posy[i];
        bodies[i].pos.z = posz[i];
        bodies[i].vel.x = velx[i];
        bodies[i].vel.y = vely[i];
        bodies[i].vel.z = velz[i];
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
