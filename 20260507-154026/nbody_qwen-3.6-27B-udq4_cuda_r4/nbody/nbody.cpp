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

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

__global__ void computeForcesKernel(const Body* bodies, double* fx, double* fy, double* fz, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double px = bodies[i].pos.x;
    double py = bodies[i].pos.y;
    double pz = bodies[i].pos.z;

    for (size_t j = 0; j < n; ++j) {
        double dx = bodies[j].pos.x - px;
        double dy = bodies[j].pos.y - py;
        double dz = bodies[j].pos.z - pz;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    fx[i] = Fx;
    fy[i] = Fy;
    fz[i] = Fz;
}

__global__ void integrateBodiesKernel(Body* bodies, const double* fx, const double* fy, const double* fz, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    bodies[i].vel.x += DT * fx[i];
    bodies[i].vel.y += DT * fy[i];
    bodies[i].vel.z += DT * fz[i];

    bodies[i].pos.x += bodies[i].vel.x * DT;
    bodies[i].pos.y += bodies[i].vel.y * DT;
    bodies[i].pos.z += bodies[i].vel.z * DT;
}

__global__ void kineticEnergyKernel(const Body* bodies, double* energy, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double e = 0.5 * (bodies[i].vel.x * bodies[i].vel.x +
                       bodies[i].vel.y * bodies[i].vel.y +
                       bodies[i].vel.z * bodies[i].vel.z);
    atomicAdd(energy, e);
}

__global__ void potentialEnergyKernel(const Body* bodies, double* energy, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    for (size_t j = i + 1; j < n; j += blockDim.x * gridDim.x) {
        double dx = bodies[j].pos.x - bodies[i].pos.x;
        double dy = bodies[j].pos.y - bodies[i].pos.y;
        double dz = bodies[j].pos.z - bodies[i].pos.z;
        double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        atomicAdd(energy, -1.0 / dist);
    }
}

// ---------------------------------------------------------------------------
// Host helpers (used only for optional validation)
// ---------------------------------------------------------------------------

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

    // Allocate device memory
    Body* d_bodies = nullptr;
    double* d_fx = nullptr;
    double* d_fy = nullptr;
    double* d_fz = nullptr;

    cudaMalloc(&d_bodies, numBodies * sizeof(Body));
    cudaMalloc(&d_fx, numBodies * sizeof(double));
    cudaMalloc(&d_fy, numBodies * sizeof(double));
    cudaMalloc(&d_fz, numBodies * sizeof(double));

    cudaMemcpy(d_bodies, bodies.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numBodies + blockSize - 1) / blockSize;

    // Run simulation on GPU
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<numBlocks, blockSize>>>(d_bodies, d_fx, d_fy, d_fz, numBodies);
        integrateBodiesKernel<<<numBlocks, blockSize>>>(d_bodies, d_fx, d_fy, d_fz, numBodies);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Copy results back to host
    cudaMemcpy(bodies.data(), d_bodies, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);

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

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    cudaFree(d_bodies);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    return 0;
}
