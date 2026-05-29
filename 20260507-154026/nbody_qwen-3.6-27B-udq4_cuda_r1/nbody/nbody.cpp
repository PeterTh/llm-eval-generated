#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Body {
    double pos_x, pos_y, pos_z;
    double vel_x, vel_y, vel_z;
};

// CUDA kernel: compute gravitational forces with shared memory tiling
// Each thread computes forces on one body; bodies are loaded in tiles into shared memory
__global__ void computeForcesKernel(Body* bodies, size_t n) {
    constexpr int TILE_SIZE = 256;
    __shared__ double sx[TILE_SIZE];
    __shared__ double sy[TILE_SIZE];
    __shared__ double sz[TILE_SIZE];

    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    const bool valid = i < n;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    Body bodyI = {0, 0, 0, 0, 0, 0};
    if (valid) bodyI = bodies[i];

    for (size_t tileStart = 0; tileStart < n; tileStart += TILE_SIZE) {
        // Load tile into shared memory
        size_t tileIdx = tileStart + tid;
        if (tileIdx < n) {
            sx[tid] = bodies[tileIdx].pos_x;
            sy[tid] = bodies[tileIdx].pos_y;
            sz[tid] = bodies[tileIdx].pos_z;
        }
        __syncthreads();

        // Compute forces from this tile
        size_t tileEnd = (tileStart + TILE_SIZE < n) ? tileStart + TILE_SIZE : n;
        for (size_t j = tileStart; j < tileEnd; ++j) {
            const double dx = sx[j - tileStart] - bodyI.pos_x;
            const double dy = sy[j - tileStart] - bodyI.pos_y;
            const double dz = sz[j - tileStart] - bodyI.pos_z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    if (valid) {
        bodies[i].vel_x += DT * Fx;
        bodies[i].vel_y += DT * Fy;
        bodies[i].vel_z += DT * Fz;
    }
}

// CUDA kernel: integrate positions from velocities
__global__ void integrateBodiesKernel(Body* bodies, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    bodies[idx].pos_x += bodies[idx].vel_x * DT;
    bodies[idx].pos_y += bodies[idx].vel_y * DT;
    bodies[idx].pos_z += bodies[idx].vel_z * DT;
}

// CUDA kernel: compute kinetic energy with parallel reduction
__global__ void computeKineticEnergyKernel(const Body* bodies, size_t n, double* energy) {
    extern __shared__ double sdata[];

    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    double e = 0.0;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        e += 0.5 * (bodies[i].vel_x * bodies[i].vel_x +
                    bodies[i].vel_y * bodies[i].vel_y +
                    bodies[i].vel_z * bodies[i].vel_z);
    }

    sdata[tid] = e;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(energy, sdata[0]);
    }
}

// CUDA kernel: compute potential energy (pairwise, i < j)
__global__ void computePotentialEnergyKernel(const Body* bodies, size_t n, double* energy) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    double e = 0.0;
    for (size_t i = idx; i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos_x - bodies[i].pos_x;
            const double dy = bodies[j].pos_y - bodies[i].pos_y;
            const double dz = bodies[j].pos_z - bodies[i].pos_z;
            const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            e -= 1.0 / dist;
        }
    }

    if (e != 0.0) {
        atomicAdd(energy, e);
    }
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos_x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos_y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos_z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel_x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel_y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel_z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos_x) || !std::isfinite(body.pos_y) || !std::isfinite(body.pos_z) ||
            !std::isfinite(body.vel_x) || !std::isfinite(body.vel_y) || !std::isfinite(body.vel_z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos_x) > maxPos || std::abs(body.pos_y) > maxPos || std::abs(body.pos_z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel_x) > maxVel || std::abs(body.vel_y) > maxVel || std::abs(body.vel_z) > maxVel) {
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
    std::vector<Body> h_bodies(numBodies);
    randomizeBodies(h_bodies);

    // Allocate device memory
    Body* d_bodies = nullptr;
    cudaMalloc(&d_bodies, numBodies * sizeof(Body));
    cudaMemcpy(d_bodies, h_bodies.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice);

    // Launch configuration
    constexpr int BLOCK_SIZE = 256;
    int numBlocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<numBlocks, BLOCK_SIZE>>>(d_bodies, numBodies);
        integrateBodiesKernel<<<numBlocks, BLOCK_SIZE>>>(d_bodies, numBodies);
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Copy results back to host
    cudaMemcpy(h_bodies.data(), d_bodies, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);

    // Print results for external validation
    if (printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : h_bodies) {
            bodyData.push_back(body.pos_x);
            bodyData.push_back(body.pos_y);
            bodyData.push_back(body.pos_z);
            bodyData.push_back(body.vel_x);
            bodyData.push_back(body.vel_y);
            bodyData.push_back(body.vel_z);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(h_bodies)) {
            // Compute total energy on GPU
            double h_energy = 0.0;
            double* d_energy = nullptr;
            cudaMalloc(&d_energy, sizeof(double));
            cudaMemset(d_energy, 0, sizeof(double));

            size_t sharedMemSize = BLOCK_SIZE * sizeof(double);
            computeKineticEnergyKernel<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_bodies, numBodies, d_energy);
            computePotentialEnergyKernel<<<numBlocks, BLOCK_SIZE>>>(d_bodies, numBodies, d_energy);
            cudaDeviceSynchronize();

            cudaMemcpy(&h_energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(d_energy);

            printf("Final energy: %.6f\n", h_energy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    cudaFree(d_bodies);

    return 0;
}
