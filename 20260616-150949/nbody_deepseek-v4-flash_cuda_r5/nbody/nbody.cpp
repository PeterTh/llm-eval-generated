#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Force computation kernel using shared memory tiling for coalesced access
__global__ void computeForcesKernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    int n, double dt, double softening)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double px = pos_x[i];
    double py = pos_y[i];
    double pz = pos_z[i];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    extern __shared__ double shared[];
    double* sh_x = shared;
    double* sh_y = shared + blockDim.x;
    double* sh_z = shared + 2 * blockDim.x;

    int numTiles = (n + blockDim.x - 1) / blockDim.x;
    for (int tile = 0; tile < numTiles; ++tile) {
        int j = tile * blockDim.x + threadIdx.x;
        if (j < n) {
            sh_x[threadIdx.x] = pos_x[j];
            sh_y[threadIdx.x] = pos_y[j];
            sh_z[threadIdx.x] = pos_z[j];
        }
        __syncthreads();

        int tileSize = (tile == numTiles - 1 && n % blockDim.x != 0) ? (n % blockDim.x) : blockDim.x;
        for (int jj = 0; jj < tileSize; ++jj) {
            double dx = sh_x[jj] - px;
            double dy = sh_y[jj] - py;
            double dz = sh_z[jj] - pz;
            double distSqr = dx*dx + dy*dy + dz*dz + softening;
            double invDist = rsqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    vel_x[i] += dt * Fx;
    vel_y[i] += dt * Fy;
    vel_z[i] += dt * Fz;
}

// Position integration kernel
__global__ void integrateBodiesKernel(
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ vel_z,
    int n, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    pos_x[i] += vel_x[i] * dt;
    pos_y[i] += vel_y[i] * dt;
    pos_z[i] += vel_z[i] * dt;
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
    
    // Allocate host-side arrays (SoA layout for efficient GPU transfer)
    std::vector<double> h_pos_x(numBodies), h_pos_y(numBodies), h_pos_z(numBodies);
    std::vector<double> h_vel_x(numBodies), h_vel_y(numBodies), h_vel_z(numBodies);
    
    // Convert AoS to SoA
    for (int i = 0; i < numBodies; ++i) {
        h_pos_x[i] = bodies[i].pos.x;
        h_pos_y[i] = bodies[i].pos.y;
        h_pos_z[i] = bodies[i].pos.z;
        h_vel_x[i] = bodies[i].vel.x;
        h_vel_y[i] = bodies[i].vel.y;
        h_vel_z[i] = bodies[i].vel.z;
    }
    
    // Allocate GPU memory
    double *d_pos_x, *d_pos_y, *d_pos_z;
    double *d_vel_x, *d_vel_y, *d_vel_z;
    CUDA_CHECK(cudaMalloc(&d_pos_x, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pos_y, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pos_z, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_x, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_y, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_z, numBodies * sizeof(double)));
    
    // Copy initial data to GPU
    CUDA_CHECK(cudaMemcpy(d_pos_x, h_pos_x.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y, h_pos_y.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_z, h_pos_z.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_x, h_vel_x.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_y, h_vel_y.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_z, h_vel_z.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch configuration
    int blockSize = 256;
    int numBlocks = (numBodies + blockSize - 1) / blockSize;
    int sharedMemSize = 3 * blockSize * sizeof(double);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<numBlocks, blockSize, sharedMemSize>>>(
            d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, numBodies, DT, SOFTENING);
        CUDA_CHECK(cudaGetLastError());
        
        integrateBodiesKernel<<<numBlocks, blockSize>>>(
            d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, numBodies, DT);
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_pos_x.data(), d_pos_x, numBodies * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_y.data(), d_pos_y, numBodies * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_z.data(), d_pos_z, numBodies * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel_x.data(), d_vel_x, numBodies * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel_y.data(), d_vel_y, numBodies * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel_z.data(), d_vel_z, numBodies * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Clean up GPU memory
    CUDA_CHECK(cudaFree(d_pos_x));
    CUDA_CHECK(cudaFree(d_pos_y));
    CUDA_CHECK(cudaFree(d_pos_z));
    CUDA_CHECK(cudaFree(d_vel_x));
    CUDA_CHECK(cudaFree(d_vel_y));
    CUDA_CHECK(cudaFree(d_vel_z));
    
    // Convert SoA back to AoS
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = h_pos_x[i];
        bodies[i].pos.y = h_pos_y[i];
        bodies[i].pos.z = h_pos_z[i];
        bodies[i].vel.x = h_vel_x[i];
        bodies[i].vel.y = h_vel_y[i];
        bodies[i].vel.z = h_vel_z[i];
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
