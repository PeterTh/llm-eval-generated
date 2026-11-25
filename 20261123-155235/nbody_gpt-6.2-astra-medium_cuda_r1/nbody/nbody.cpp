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

using Vec3 = double3;

struct Bodies {
    std::vector<Vec3> pos;
    std::vector<Vec3> vel;

    explicit Bodies(size_t n) : pos(n), vel(n) {}
};

void randomizeBodies(Bodies& bodies, unsigned int seed = 42) {
    for (size_t i = 0; i < bodies.pos.size(); ++i) {
        bodies.pos[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.pos[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.pos[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vel[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vel[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vel[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__device__ double inverseDistance(double dx, double dy, double dz) {
    const double squared = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
    return 1.0 / sqrt(squared);
}

__device__ double inverseCube(double dx, double dy, double dz) {
    const double inv = inverseDistance(dx, dy, dz);
    return inv * inv * inv;
}

// For small systems, expose both pair indices to the GPU. Each tile computes
// an unordered pair of body groups and stores the weights in both directions.
constexpr int pairTile = 16;
__global__ void computeWeights(const Vec3* pos, double* weights, int n) {
    if (blockIdx.y > blockIdx.x) return;
    const int i = blockIdx.x * pairTile + threadIdx.x;
    const int j = blockIdx.y * pairTile + threadIdx.y;
    if (i >= n || j >= n) return;
    const Vec3 a = pos[i], b = pos[j];
    const double w = inverseCube(b.x - a.x, b.y - a.y, b.z - a.z);
    weights[j * n + i] = w;
    if (blockIdx.x != blockIdx.y) weights[i * n + j] = w;
}

constexpr int bodyTile = 64;
template<bool cachedWeights>
__global__ void advance(const Vec3* __restrict__ pos, Vec3* __restrict__ next,
                        Vec3* vel, const double* weights, int n) {
    __shared__ Vec3 sources[bodyTile];
    const unsigned i = blockIdx.x * bodyTile + threadIdx.x;
    const bool active = i < static_cast<unsigned>(n);
    const Vec3 target = active ? pos[i] : Vec3{};
    Vec3 force{};
    for (unsigned base = 0; base < static_cast<unsigned>(n); base += bodyTile) {
        const int count = min(static_cast<unsigned>(bodyTile), n - base);
        if (threadIdx.x < count) sources[threadIdx.x] = pos[base + threadIdx.x];
        __syncthreads();
        if (active) {
            #pragma unroll 8
            for (int k = 0; k < count; ++k) {
                const double dx = sources[k].x - target.x;
                const double dy = sources[k].y - target.y;
                const double dz = sources[k].z - target.z;
                const double w = cachedWeights ? weights[(base + k) * n + i]
                                               : inverseCube(dx, dy, dz);
                force.x = fma(dx, w, force.x);
                force.y = fma(dy, w, force.y);
                force.z = fma(dz, w, force.z);
            }
        }
        __syncthreads();
    }
    if (active) {
        Vec3 v = vel[i];
        v.x += __dmul_rn(DT, force.x);
        v.y += __dmul_rn(DT, force.y);
        v.z += __dmul_rn(DT, force.z);
        vel[i] = v;
        next[i] = {fma(DT, v.x, target.x), fma(DT, v.y, target.y), fma(DT, v.z, target.z)};
    }
}

// One warp sums the unique potential-energy pairs for each target body.
__global__ void energyRows(const Vec3* pos, const Vec3* vel, double* rows, unsigned n) {
    const unsigned i = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    if (i >= n) return;
    const unsigned lane = threadIdx.x % warpSize;
    const Vec3 a = pos[i];
    double energy = 0.0;
    for (unsigned j = i + 1 + lane; j < n; j += warpSize) {
        const Vec3 b = pos[j];
        energy -= inverseDistance(b.x - a.x, b.y - a.y, b.z - a.z);
    }
    for (int offset = warpSize / 2; offset; offset /= 2)
        energy += __shfl_down_sync(0xffffffff, energy, offset);
    if (lane == 0) {
        const Vec3 v = vel[i];
        rows[i] = energy + 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const Bodies& bodies) {
    for (size_t i = 0; i < bodies.pos.size(); ++i) {
        // Check for NaN or Inf values
        if (!std::isfinite(bodies.pos[i].x) || !std::isfinite(bodies.pos[i].y) || !std::isfinite(bodies.pos[i].z) ||
            !std::isfinite(bodies.vel[i].x) || !std::isfinite(bodies.vel[i].y) || !std::isfinite(bodies.vel[i].z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(bodies.pos[i].x) > maxPos || std::abs(bodies.pos[i].y) > maxPos || std::abs(bodies.pos[i].z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(bodies.vel[i].x) > maxVel || std::abs(bodies.vel[i].y) > maxVel || std::abs(bodies.vel[i].z) > maxVel) {
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
    Bodies bodies(numBodies);
    randomizeBodies(bodies);
    
    const size_t bytes = bodies.pos.size() * sizeof(Vec3);
    Vec3 *storage, *pos, *next, *vel;
    checkCuda(cudaMalloc(&storage, std::max(size_t{1}, 3 * bytes)));
    pos = storage;
    next = pos + bodies.pos.size();
    vel = next + bodies.pos.size();
    double* weights = nullptr;
    if (numBodies > 0) {
        checkCuda(cudaMemcpy(pos, bodies.pos.data(), bytes, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(vel, bodies.vel.data(), bytes, cudaMemcpyHostToDevice));
        // Bound temporary storage. Large systems compute weights directly
        // from shared-memory position tiles and use only linear device memory.
        if (numBodies <= 2048 && numSteps > 0)
            checkCuda(cudaMalloc(&weights, bodies.pos.size() * bodies.pos.size() * sizeof(double)));
    }
    const unsigned blocks = (bodies.pos.size() + bodyTile - 1) / bodyTile;
    const unsigned tiles = (bodies.pos.size() + pairTile - 1) / pairTile;
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps && numBodies > 0; ++step) {
        if (weights) {
            computeWeights<<<dim3(tiles, tiles), dim3(pairTile, pairTile)>>>(pos, weights, numBodies);
            checkCuda(cudaGetLastError());
            advance<true><<<blocks, bodyTile>>>(pos, next, vel, weights, numBodies);
        } else {
            advance<false><<<blocks, bodyTile>>>(pos, next, vel, nullptr, numBodies);
        }
        checkCuda(cudaGetLastError());
        std::swap(pos, next);
    }
    checkCuda(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    if ((printResults || validate) && numBodies > 0) {
        checkCuda(cudaMemcpy(bodies.pos.data(), pos, bytes, cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(bodies.vel.data(), vel, bytes, cudaMemcpyDeviceToHost));
    }

    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(bodies.pos.size() * 6);
        for (size_t i = 0; i < bodies.pos.size(); ++i) {
            bodyData.push_back(bodies.pos[i].x);
            bodyData.push_back(bodies.pos[i].y);
            bodyData.push_back(bodies.pos[i].z);
            bodyData.push_back(bodies.vel[i].x);
            bodyData.push_back(bodies.vel[i].y);
            bodyData.push_back(bodies.vel[i].z);
        }
        print_results(bodyData, "Bodies");
    }
    
    int status = 0;
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = 0;
            if (numBodies > 0) {
                // The inactive position buffer is available for the row sums.
                double* rows = reinterpret_cast<double*>(next);
                energyRows<<<(bodies.pos.size() + 3) / 4, 128>>>(pos, vel, rows, numBodies);
                checkCuda(cudaGetLastError());
                std::vector<double> energies(numBodies);
                checkCuda(cudaMemcpy(energies.data(), rows, numBodies * sizeof(double),
                                     cudaMemcpyDeviceToHost));
                for (double energy : energies) finalEnergy += energy;
            }
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }
    
    checkCuda(cudaFree(weights));
    checkCuda(cudaFree(storage));
    return status;
}
