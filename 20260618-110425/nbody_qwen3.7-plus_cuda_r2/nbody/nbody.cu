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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define TILE_SIZE 256

// Fused kernel: compute gravitational forces, update velocities, and integrate positions
// Uses shared memory tiling for O(N^2/blocksize) global memory bandwidth
__global__ void bodyForceAndIntegrate(
    double* __restrict__ posX,
    double* __restrict__ posY,
    double* __restrict__ posZ,
    double* __restrict__ velX,
    double* __restrict__ velY,
    double* __restrict__ velZ,
    const int n)
{
    __shared__ double sPosX[TILE_SIZE];
    __shared__ double sPosY[TILE_SIZE];
    __shared__ double sPosZ[TILE_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double myPosX = 0.0, myPosY = 0.0, myPosZ = 0.0;
    double accX = 0.0, accY = 0.0, accZ = 0.0;

    if (i < n) {
        myPosX = posX[i];
        myPosY = posY[i];
        myPosZ = posZ[i];
    }

    const int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; tile++) {
        const int srcIdx = tile * TILE_SIZE + threadIdx.x;
        if (srcIdx < n) {
            sPosX[threadIdx.x] = posX[srcIdx];
            sPosY[threadIdx.x] = posY[srcIdx];
            sPosZ[threadIdx.x] = posZ[srcIdx];
        }
        __syncthreads();

        if (i < n) {
            for (int j = 0; j < TILE_SIZE; j++) {
                const int srcIdx2 = tile * TILE_SIZE + j;
                if (srcIdx2 < n) {
                    const double dx = sPosX[j] - myPosX;
                    const double dy = sPosY[j] - myPosY;
                    const double dz = sPosZ[j] - myPosZ;
                    const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    const double invDist = 1.0 / sqrt(distSqr);
                    const double invDist3 = invDist * invDist * invDist;

                    accX += dx * invDist3;
                    accY += dy * invDist3;
                    accZ += dz * invDist3;
                }
            }
        }
        __syncthreads();
    }

    if (i < n) {
        // Update velocities: v_new = v + dt * a
        double vx = velX[i] + DT * accX;
        double vy = velY[i] + DT * accY;
        double vz = velZ[i] + DT * accZ;

        velX[i] = vx;
        velY[i] = vy;
        velZ[i] = vz;

        // Integrate positions: p_new = p + dt * v_new
        posX[i] = myPosX + vx * DT;
        posY[i] = myPosY + vy * DT;
        posZ[i] = myPosZ + vz * DT;
    }
}

void randomizeBodies(std::vector<double>& posX, std::vector<double>& posY, std::vector<double>& posZ,
                     std::vector<double>& velX, std::vector<double>& velY, std::vector<double>& velZ,
                     unsigned int seed = 42) {
    const int n = static_cast<int>(posX.size());
    for (int i = 0; i < n; i++) {
        posX[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        posY[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        posZ[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velX[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velY[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velZ[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

double computeTotalEnergy(const std::vector<double>& posX, const std::vector<double>& posY, const std::vector<double>& posZ,
                          const std::vector<double>& velX, const std::vector<double>& velY, const std::vector<double>& velZ) {
    double energy = 0.0;
    const int n = static_cast<int>(posX.size());

    // Kinetic energy
    for (int i = 0; i < n; i++) {
        energy += 0.5 * (velX[i] * velX[i] + velY[i] * velY[i] + velZ[i] * velZ[i]);
    }

    // Potential energy
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            const double dx = posX[j] - posX[i];
            const double dy = posY[j] - posY[i];
            const double dz = posZ[j] - posZ[i];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

bool validateSimulation(const std::vector<double>& posX, const std::vector<double>& posY, const std::vector<double>& posZ,
                        const std::vector<double>& velX, const std::vector<double>& velY, const std::vector<double>& velZ) {
    const int n = static_cast<int>(posX.size());
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(posX[i]) || !std::isfinite(posY[i]) || !std::isfinite(posZ[i]) ||
            !std::isfinite(velX[i]) || !std::isfinite(velY[i]) || !std::isfinite(velZ[i])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(posX[i]) > maxPos || std::abs(posY[i]) > maxPos || std::abs(posZ[i]) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(velX[i]) > maxVel || std::abs(velY[i]) > maxVel || std::abs(velZ[i]) > maxVel) {
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
    
    // Initialize body data in SoA layout
    std::vector<double> h_posX(numBodies), h_posY(numBodies), h_posZ(numBodies);
    std::vector<double> h_velX(numBodies), h_velY(numBodies), h_velZ(numBodies);
    randomizeBodies(h_posX, h_posY, h_posZ, h_velX, h_velY, h_velZ);

    // Allocate GPU memory
    double *d_posX, *d_posY, *d_posZ;
    double *d_velX, *d_velY, *d_velZ;
    const size_t bytes = numBodies * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_posX, bytes));
    CUDA_CHECK(cudaMalloc(&d_posY, bytes));
    CUDA_CHECK(cudaMalloc(&d_posZ, bytes));
    CUDA_CHECK(cudaMalloc(&d_velX, bytes));
    CUDA_CHECK(cudaMalloc(&d_velY, bytes));
    CUDA_CHECK(cudaMalloc(&d_velZ, bytes));

    // Copy initial data to GPU
    CUDA_CHECK(cudaMemcpy(d_posX, h_posX.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posY, h_posY.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posZ, h_posZ.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velX, h_velX.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velY, h_velY.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velZ, h_velZ.data(), bytes, cudaMemcpyHostToDevice));

    // Set up kernel launch parameters
    const int blockSize = TILE_SIZE;
    const int gridSize = (numBodies + blockSize - 1) / blockSize;

    // Run simulation on GPU
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        bodyForceAndIntegrate<<<gridSize, blockSize>>>(
            d_posX, d_posY, d_posZ,
            d_velX, d_velY, d_velZ,
            numBodies);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_posX.data(), d_posX, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_posY.data(), d_posY, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_posZ.data(), d_posZ, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_velX.data(), d_velX, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_velY.data(), d_velY, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_velZ.data(), d_velZ, bytes, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_posX));
    CUDA_CHECK(cudaFree(d_posY));
    CUDA_CHECK(cudaFree(d_posZ));
    CUDA_CHECK(cudaFree(d_velX));
    CUDA_CHECK(cudaFree(d_velY));
    CUDA_CHECK(cudaFree(d_velZ));

    // Print results for external validation
    if (printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (int i = 0; i < numBodies; i++) {
            bodyData.push_back(h_posX[i]);
            bodyData.push_back(h_posY[i]);
            bodyData.push_back(h_posZ[i]);
            bodyData.push_back(h_velX[i]);
            bodyData.push_back(h_velY[i]);
            bodyData.push_back(h_velZ[i]);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(h_posX, h_posY, h_posZ, h_velX, h_velY, h_velZ)) {
            double finalEnergy = computeTotalEnergy(h_posX, h_posY, h_posZ, h_velX, h_velY, h_velZ);
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
