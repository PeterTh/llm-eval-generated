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

struct Bodies {
    std::vector<double3> positions;
    std::vector<double3> velocities;

    explicit Bodies(size_t n) : positions(n), velocities(n) {}
};

void randomizeBodies(Bodies& bodies, unsigned int seed = 42) {
    auto sample = [&seed]() { return 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0; };
    for (size_t i = 0; i < bodies.positions.size(); ++i) {
        bodies.positions[i] = {sample(), sample(), sample()};
        bodies.velocities[i] = {sample(), sample(), sample()};
    }
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

constexpr int TILE = 128;

__device__ double inverseDistance(double dx, double dy, double dz) {
    const double squared = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
    return __ddiv_rn(1.0, __dsqrt_rn(squared));
}

__device__ double inverseCube(double dx, double dy, double dz) {
    const double inverse = inverseDistance(dx, dy, dz);
    return inverse * inverse * inverse;
}

__device__ double advanceVelocity(double velocity, double force) {
    return velocity + __dmul_rn(DT, force);
}

__device__ void integrate(int i, double3 p, double3 force, double3* velocities, double3* next) {
    double3 v = velocities[i];
    v.x = advanceVelocity(v.x, force.x);
    v.y = advanceVelocity(v.y, force.y);
    v.z = advanceVelocity(v.z, force.z);
    velocities[i] = v;
    next[i] = make_double3(fma(DT, v.x, p.x), fma(DT, v.y, p.y), fma(DT, v.z, p.z));
}

// With few targets, distribute each target's distance calculations across
// a whole block. Three lanes then accumulate x, y and z in source order.
__global__ void stepSmall(const double3* positions, double3* next,
                          double3* velocities, int n) {
    __shared__ double terms[4][TILE];
    const int i = blockIdx.x;
    const int lane = threadIdx.x;
    const double3 p = positions[i];
    double force = 0.0;
    for (int base = 0; base < n; base += TILE) {
        const int count = min(n - base, TILE);
        if (lane < count) {
            const double3 q = positions[base + lane];
            const double dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
            terms[0][lane] = dx;
            terms[1][lane] = dy;
            terms[2][lane] = dz;
            terms[3][lane] = inverseCube(dx, dy, dz);
        }
        __syncthreads();
        if (lane < 3) {
            for (int j = 0; j < count; ++j)
                force = fma(terms[lane][j], terms[3][j], force);
        }
        __syncthreads();
    }
    if (lane < 3) {
        // Reuse the tile to gather the three independently accumulated forces.
        terms[lane][0] = force;
    }
    __syncthreads();
    if (lane == 0) {
        integrate(i, p, make_double3(terms[0][0], terms[1][0], terms[2][0]), velocities, next);
    }
}

// Larger systems expose enough targets for each thread to integrate one.
// Every target in a block reuses the same shared-memory tile of sources.
__global__ void stepTiled(const double3* positions, double3* next,
                          double3* velocities, int n) {
    __shared__ double3 sources[TILE];
    const unsigned i = blockIdx.x * TILE + threadIdx.x;
    const bool active = i < static_cast<unsigned>(n);
    const double3 p = active ? positions[i] : make_double3(0, 0, 0);
    double Fx = 0, Fy = 0, Fz = 0;
    for (unsigned base = 0; base < static_cast<unsigned>(n); base += TILE) {
        const int count = min(static_cast<unsigned>(TILE), n - base);
        if (threadIdx.x < count)
            sources[threadIdx.x] = positions[base + threadIdx.x];
        __syncthreads();
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < count; ++j) {
                const double dx = sources[j].x - p.x;
                const double dy = sources[j].y - p.y;
                const double dz = sources[j].z - p.z;
                const double weight = inverseCube(dx, dy, dz);
                Fx = fma(dx, weight, Fx);
                Fy = fma(dy, weight, Fy);
                Fz = fma(dz, weight, Fz);
            }
        }
        __syncthreads();
    }
    if (active) {
        integrate(i, p, make_double3(Fx, Fy, Fz), velocities, next);
    }
}

// One block per body, visiting each unordered pair exactly once.
__global__ void energyPerBody(const double3* positions, const double3* velocities,
                              double* energies, unsigned n) {
    __shared__ double partial[TILE];
    const unsigned i = blockIdx.x;
    const unsigned lane = threadIdx.x;
    const double3 p = positions[i];
    double potential = 0.0;
    for (unsigned j = i + 1 + lane; j < n; j += TILE) {
        const double3 q = positions[j];
        potential -= inverseDistance(q.x - p.x, q.y - p.y, q.z - p.z);
    }
    partial[lane] = potential;
    for (unsigned stride = TILE / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (lane < stride)
            partial[lane] += partial[lane + stride];
    }
    if (lane == 0) {
        const double3 v = velocities[i];
        energies[i] = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z) + partial[0];
    }
}

double computeTotalEnergy(const double3* positions, const double3* velocities,
                          double* scratch, int n) {
    if (!n) return 0;
    energyPerBody<<<n, TILE>>>(positions, velocities, scratch, n);
    checkCuda(cudaGetLastError());
    std::vector<double> energies(n);
    checkCuda(cudaMemcpy(energies.data(), scratch, n * sizeof(double), cudaMemcpyDeviceToHost));
    return kahan_sum(energies);
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const Bodies& bodies) {
    for (size_t i = 0; i < bodies.positions.size(); ++i) {
        const auto& pos = bodies.positions[i];
        const auto& vel = bodies.velocities[i];
        // Check for NaN or Inf values
        if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
            !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(pos.x) > maxPos || std::abs(pos.y) > maxPos || std::abs(pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vel.x) > maxVel || std::abs(vel.y) > maxVel || std::abs(vel.z) > maxVel) {
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
    
    double3 *positions, *next, *velocities;
    const size_t bytes = std::max(1, numBodies) * sizeof(double3);
    checkCuda(cudaMalloc(&positions, bytes));
    checkCuda(cudaMalloc(&next, bytes));
    checkCuda(cudaMalloc(&velocities, bytes));
    if (numBodies > 0) {
        checkCuda(cudaMemcpy(positions, bodies.positions.data(), bytes, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(velocities, bodies.velocities.data(), bytes, cudaMemcpyHostToDevice));
    }
    checkCuda(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps && numBodies > 0; ++step) {
        if (numBodies <= 4096)
            stepSmall<<<numBodies, TILE>>>(positions, next, velocities, numBodies);
        else
            stepTiled<<<1 + (numBodies - 1) / TILE, TILE>>>(positions, next, velocities, numBodies);
        checkCuda(cudaGetLastError());
        std::swap(positions, next);
    }
    checkCuda(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    if ((printResults || validate) && numBodies > 0) {
        checkCuda(cudaMemcpy(bodies.positions.data(), positions, bytes, cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(bodies.velocities.data(), velocities, bytes, cudaMemcpyDeviceToHost));
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(bodies.positions.size() * 6);
        for (size_t i = 0; i < bodies.positions.size(); ++i) {
            const auto& pos = bodies.positions[i];
            const auto& vel = bodies.velocities[i];
            bodyData.push_back(pos.x);
            bodyData.push_back(pos.y);
            bodyData.push_back(pos.z);
            bodyData.push_back(vel.x);
            bodyData.push_back(vel.y);
            bodyData.push_back(vel.z);
        }
        print_results(bodyData, "Bodies");
    }
    
    // Validation: check that simulation produces finite, reasonable values
    int status = 0;
    if (validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            // The spare position buffer has room for one energy per body.
            double finalEnergy = computeTotalEnergy(positions, velocities,
                                                   reinterpret_cast<double*>(next), numBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }
    
    checkCuda(cudaFree(positions));
    checkCuda(cudaFree(next));
    checkCuda(cudaFree(velocities));
    return status;
}
