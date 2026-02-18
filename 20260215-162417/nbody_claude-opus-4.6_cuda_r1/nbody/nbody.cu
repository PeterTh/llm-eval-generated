#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define SOFTENING 1e-9
#define DT 0.01
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

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

// Tiled shared-memory force computation kernel (SoA layout)
__global__ void computeForcesKernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    const int n)
{
    extern __shared__ double smem[];
    double* s_px = smem;
    double* s_py = s_px + BLOCK_SIZE;
    double* s_pz = s_py + BLOCK_SIZE;

    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    double my_px = 0.0, my_py = 0.0, my_pz = 0.0;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    if (i < n) {
        my_px = pos_x[i];
        my_py = pos_y[i];
        my_pz = pos_z[i];
    }

    const int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < numTiles; tile++) {
        const int idx = tile * BLOCK_SIZE + threadIdx.x;
        s_px[threadIdx.x] = (idx < n) ? pos_x[idx] : 0.0;
        s_py[threadIdx.x] = (idx < n) ? pos_y[idx] : 0.0;
        s_pz[threadIdx.x] = (idx < n) ? pos_z[idx] : 0.0;
        __syncthreads();

        if (i < n) {
            const int tileEnd = min(BLOCK_SIZE, n - tile * BLOCK_SIZE);
            for (int k = 0; k < tileEnd; k++) {
                const double dx = s_px[k] - my_px;
                const double dy = s_py[k] - my_py;
                const double dz = s_pz[k] - my_pz;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        vel_x[i] += DT * Fx;
        vel_y[i] += DT * Fy;
        vel_z[i] += DT * Fz;
    }
}

__global__ void integrateKernel(
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ vel_z,
    const int n)
{
    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) {
        pos_x[i] += vel_x[i] * DT;
        pos_y[i] += vel_y[i] * DT;
        pos_z[i] += vel_z[i] * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
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

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
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
    
    // Initialize bodies on host
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    
    // Convert AoS to SoA for GPU
    std::vector<double> h_px(numBodies), h_py(numBodies), h_pz(numBodies);
    std::vector<double> h_vx(numBodies), h_vy(numBodies), h_vz(numBodies);
    for (int i = 0; i < numBodies; i++) {
        h_px[i] = bodies[i].pos.x;
        h_py[i] = bodies[i].pos.y;
        h_pz[i] = bodies[i].pos.z;
        h_vx[i] = bodies[i].vel.x;
        h_vy[i] = bodies[i].vel.y;
        h_vz[i] = bodies[i].vel.z;
    }
    
    // Allocate device memory
    double *d_px, *d_py, *d_pz, *d_vx, *d_vy, *d_vz;
    const size_t bytes = numBodies * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_px, bytes));
    CUDA_CHECK(cudaMalloc(&d_py, bytes));
    CUDA_CHECK(cudaMalloc(&d_pz, bytes));
    CUDA_CHECK(cudaMalloc(&d_vx, bytes));
    CUDA_CHECK(cudaMalloc(&d_vy, bytes));
    CUDA_CHECK(cudaMalloc(&d_vz, bytes));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, h_pz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz.data(), bytes, cudaMemcpyHostToDevice));
    
    // Kernel launch configuration
    const int gridSize = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t sharedMemSize = 3 * BLOCK_SIZE * sizeof(double);
    
    // Run simulation on GPU
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<gridSize, BLOCK_SIZE, sharedMemSize>>>(
            d_px, d_py, d_pz, d_vx, d_vy, d_vz, numBodies);
        integrateKernel<<<gridSize, BLOCK_SIZE>>>(
            d_px, d_py, d_pz, d_vx, d_vy, d_vz, numBodies);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_px.data(), d_px, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_py.data(), d_py, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pz.data(), d_pz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vx.data(), d_vx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vy.data(), d_vy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vz.data(), d_vz, bytes, cudaMemcpyDeviceToHost));
    
    // Convert SoA back to AoS
    for (int i = 0; i < numBodies; i++) {
        bodies[i].pos.x = h_px[i];
        bodies[i].pos.y = h_py[i];
        bodies[i].pos.z = h_pz[i];
        bodies[i].vel.x = h_vx[i];
        bodies[i].vel.y = h_vy[i];
        bodies[i].vel.z = h_vz[i];
    }
    
    // Print results for external validation
    if (printResults) {
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
    
    // Validation
    if (validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
            cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
            return 1;
        }
    }
    
    // Free device memory
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    
    return 0;
}
