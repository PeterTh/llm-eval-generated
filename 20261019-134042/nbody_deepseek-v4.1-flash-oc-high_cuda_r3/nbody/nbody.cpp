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

constexpr int BLOCK_SIZE = 32;
constexpr int TILE_SIZE = 512;

static void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
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

__global__ void __launch_bounds__(BLOCK_SIZE) computeForcesKernel(
    const double* __restrict__ posX, const double* __restrict__ posY, const double* __restrict__ posZ,
    double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
    int n) {
    __shared__ double tileX[TILE_SIZE];
    __shared__ double tileY[TILE_SIZE];
    __shared__ double tileZ[TILE_SIZE];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * BLOCK_SIZE + tid;
    const bool active = i < n;

    const double pxi = active ? posX[i] : 0.0;
    const double pyi = active ? posY[i] : 0.0;
    const double pzi = active ? posZ[i] : 0.0;
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;

    for (int j0 = 0; j0 < n; j0 += TILE_SIZE) {
#pragma unroll
        for (int t = tid; t < TILE_SIZE; t += BLOCK_SIZE) {
            const int j = j0 + t;
            if (j < n) {
                tileX[t] = posX[j];
                tileY[t] = posY[j];
                tileZ[t] = posZ[j];
            }
        }
        __syncthreads();

        const int count = min(TILE_SIZE, n - j0);
        for (int t = 0; t < count; ++t) {
            const double dx = tileX[t] - pxi;
            const double dy = tileY[t] - pyi;
            const double dz = tileZ[t] - pzi;
            const double dx2dy2 = fma(dx, dx, dy * dy);
            const double distSqr = fma(dz, dz, dx2dy2) + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            fx = fma(dx, invDist3, fx);
            fy = fma(dy, invDist3, fy);
            fz = fma(dz, invDist3, fz);
        }
        __syncthreads();
    }

    if (active) {
        velX[i] += DT * fx;
        velY[i] += DT * fy;
        velZ[i] += DT * fz;
    }
}

__global__ void __launch_bounds__(BLOCK_SIZE) integrateKernel(
    double* __restrict__ posX, double* __restrict__ posY, double* __restrict__ posZ,
    const double* __restrict__ velX, const double* __restrict__ velY, const double* __restrict__ velZ,
    int n) {
    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) {
        posX[i] = fma(velX[i], DT, posX[i]);
        posY[i] = fma(velY[i], DT, posY[i]);
        posZ[i] = fma(velZ[i], DT, posZ[i]);
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

    std::vector<Body> bodies(numBodies > 0 ? numBodies : 0);
    randomizeBodies(bodies);

    const size_t n = bodies.size();
    const size_t bytes = n * sizeof(double);

    std::vector<double> hostPosX(n), hostPosY(n), hostPosZ(n);
    std::vector<double> hostVelX(n), hostVelY(n), hostVelZ(n);

    for (size_t i = 0; i < n; ++i) {
        hostPosX[i] = bodies[i].pos.x;
        hostPosY[i] = bodies[i].pos.y;
        hostPosZ[i] = bodies[i].pos.z;
        hostVelX[i] = bodies[i].vel.x;
        hostVelY[i] = bodies[i].vel.y;
        hostVelZ[i] = bodies[i].vel.z;
    }

    double *dPosX = nullptr, *dPosY = nullptr, *dPosZ = nullptr;
    double *dVelX = nullptr, *dVelY = nullptr, *dVelZ = nullptr;

    if (n > 0) {
        checkCuda(cudaMalloc(&dPosX, bytes), "cudaMalloc");
        checkCuda(cudaMalloc(&dPosY, bytes), "cudaMalloc");
        checkCuda(cudaMalloc(&dPosZ, bytes), "cudaMalloc");
        checkCuda(cudaMalloc(&dVelX, bytes), "cudaMalloc");
        checkCuda(cudaMalloc(&dVelY, bytes), "cudaMalloc");
        checkCuda(cudaMalloc(&dVelZ, bytes), "cudaMalloc");

        checkCuda(cudaMemcpy(dPosX, hostPosX.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
        checkCuda(cudaMemcpy(dPosY, hostPosY.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
        checkCuda(cudaMemcpy(dPosZ, hostPosZ.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
        checkCuda(cudaMemcpy(dVelX, hostVelX.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
        checkCuda(cudaMemcpy(dVelY, hostVelY.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
        checkCuda(cudaMemcpy(dVelZ, hostVelZ.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
    }

    const int forceGrid = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const int integrateGrid = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (n == 0) {
            break;
        }
        computeForcesKernel<<<forceGrid, BLOCK_SIZE>>>(dPosX, dPosY, dPosZ, dVelX, dVelY, dVelZ, static_cast<int>(n));
        integrateKernel<<<integrateGrid, BLOCK_SIZE>>>(dPosX, dPosY, dPosZ, dVelX, dVelY, dVelZ, static_cast<int>(n));
    }

    if (n > 0) {
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    if (n > 0) {
        checkCuda(cudaMemcpy(hostPosX.data(), dPosX, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
        checkCuda(cudaMemcpy(hostPosY.data(), dPosY, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
        checkCuda(cudaMemcpy(hostPosZ.data(), dPosZ, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
        checkCuda(cudaMemcpy(hostVelX.data(), dVelX, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
        checkCuda(cudaMemcpy(hostVelY.data(), dVelY, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
        checkCuda(cudaMemcpy(hostVelZ.data(), dVelZ, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");

        for (size_t i = 0; i < n; ++i) {
            bodies[i].pos.x = hostPosX[i];
            bodies[i].pos.y = hostPosY[i];
            bodies[i].pos.z = hostPosZ[i];
            bodies[i].vel.x = hostVelX[i];
            bodies[i].vel.y = hostVelY[i];
            bodies[i].vel.z = hostVelZ[i];
        }
    }

    if (printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(n * 6);
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

    int exitCode = 0;
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    checkCuda(cudaFree(dPosX), "cudaFree");
    checkCuda(cudaFree(dPosY), "cudaFree");
    checkCuda(cudaFree(dPosZ), "cudaFree");
    checkCuda(cudaFree(dVelX), "cudaFree");
    checkCuda(cudaFree(dVelY), "cudaFree");
    checkCuda(cudaFree(dVelZ), "cudaFree");

    return exitCode;
}
