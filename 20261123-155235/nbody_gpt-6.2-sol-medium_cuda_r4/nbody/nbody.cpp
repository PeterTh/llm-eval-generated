#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
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

namespace {

constexpr unsigned BLOCK_SIZE = 64;
constexpr unsigned TILE_SIZE = 128;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size) {
        if (size != 0) {
            checkCuda(cudaMalloc(&data_, size * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

__device__ double inverseDistance(double dx, double dy, double dz) {
    const double xy = fma(dx, dx, dy * dy);
    const double distSqr = fma(dz, dz, xy) + SOFTENING;
    // Round the square root and reciprocal separately.
    return __drcp_rn(__dsqrt_rn(distSqr));
}

template <unsigned BlockSize, bool ComputeEnergy>
__global__ void interact(const Body* __restrict__ current, Body* __restrict__ next,
                        double* __restrict__ potential, unsigned n) {
    __shared__ Vec3 sources[TILE_SIZE];
    const unsigned firstTarget = blockIdx.x * BlockSize;
    const unsigned i = firstTarget + threadIdx.x;
    const Vec3 pos = i < n ? current[i].pos : Vec3{};
    Vec3 force{};
    double energy = 0.0;

    // Each tile is loaded once per block and reused by its target bodies.
    const unsigned firstTile = ComputeEnergy ? firstTarget / TILE_SIZE * TILE_SIZE : 0;
    for (unsigned base = firstTile; base < n; base += TILE_SIZE) {
        const unsigned count = min(TILE_SIZE, n - base);
        for (unsigned j = threadIdx.x; j < TILE_SIZE; j += BlockSize) {
            if (j < count) {
                sources[j] = current[base + j].pos;
            }
        }
        __syncthreads();

        if (i < n) {
            // The potential counts each unordered pair once.
            const unsigned first = ComputeEnergy && i >= base ? i - base + 1 : 0;
#pragma unroll 4
            for (unsigned j = first; j < count; ++j) {
                const double dx = sources[j].x - pos.x;
                const double dy = sources[j].y - pos.y;
                const double dz = sources[j].z - pos.z;
                const double invDist = inverseDistance(dx, dy, dz);
                if constexpr (ComputeEnergy) {
                    energy -= invDist;
                } else {
                    const double invDist3 = invDist * invDist * invDist;
                    force.x = fma(dx, invDist3, force.x);
                    force.y = fma(dy, invDist3, force.y);
                    force.z = fma(dz, invDist3, force.z);
                }
            }
        }
        __syncthreads();
    }

    if (i < n) {
        if constexpr (ComputeEnergy) {
            potential[i] = energy;
        } else {
            const Vec3 oldVel = current[i].vel;
            const Vec3 vel{oldVel.x + __dmul_rn(DT, force.x),
                           oldVel.y + __dmul_rn(DT, force.y),
                           oldVel.z + __dmul_rn(DT, force.z)};
            // Update into a separate buffer so all forces see the same positions.
            next[i] = {{fma(vel.x, DT, pos.x),
                        fma(vel.y, DT, pos.y),
                        fma(vel.z, DT, pos.z)}, vel};
        }
    }
}

template <unsigned BlockSize>
void launchSimulation(Body*& current, Body*& next, unsigned n, int steps) {
    const unsigned blocks = (n + BlockSize - 1) / BlockSize;
    for (int step = 0; step < steps; ++step) {
        interact<BlockSize, false><<<blocks, BlockSize>>>(current, next, nullptr, n);
        std::swap(current, next);
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies, const Body* deviceBodies) {
    const unsigned n = static_cast<unsigned>(bodies.size());
    double energy = 0.0;
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    if (n != 0) {
        DeviceBuffer<double> devicePotential(n);
        interact<BLOCK_SIZE, true><<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            deviceBodies, nullptr, devicePotential.get(), n);
        checkCuda(cudaGetLastError());

        std::vector<double> potential(n);
        checkCuda(cudaMemcpy(potential.data(), devicePotential.get(), n * sizeof(double),
                             cudaMemcpyDeviceToHost));
        // Combine the potential sums computed independently for each target.
        for (double rowEnergy : potential) {
            energy += rowEnergy;
        }
    }
    return energy;
}

} // namespace

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

int main(int argc, char** argv) try {
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
    
    const unsigned n = static_cast<unsigned>(bodies.size());
    const size_t bodyBytes = bodies.size() * sizeof(Body);
    const bool simulate = n != 0 && numSteps > 0;
    const bool upload = simulate || (n != 0 && validate);

    DeviceBuffer<Body> initial(upload ? n : 0);
    DeviceBuffer<Body> updated(simulate ? n : 0);
    Body* current = initial.get();
    Body* next = updated.get();
    if (upload) {
        checkCuda(cudaMemcpy(current, bodies.data(), bodyBytes, cudaMemcpyHostToDevice));
    }

    // Time the simulation kernels, excluding allocation and host transfers.
    auto start = std::chrono::high_resolution_clock::now();
    if (simulate) {
        // Use more blocks to keep more multiprocessors busy on small inputs.
        if (n <= 4096) {
            launchSimulation<32>(current, next, n, numSteps);
        } else {
            launchSimulation<BLOCK_SIZE>(current, next, n, numSteps);
        }
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Simulation time: %ld ms\n", duration.count());

    if (simulate && (printResults || validate)) {
        checkCuda(cudaMemcpy(bodies.data(), current, bodyBytes, cudaMemcpyDeviceToHost));
    }

    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(bodies.size() * 6);
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
            double finalEnergy = computeTotalEnergy(bodies, current);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
} catch (const std::runtime_error& error) {
    fprintf(stderr, "CUDA error: %s\n", error.what());
    return 1;
}
