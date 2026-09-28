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

// Threads per block for the force kernel; also the tile size used for the
// shared-memory staging of body positions.
constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK(call)                                                                                     \
    do {                                                                                                     \
        const cudaError_t err_ = (call);                                                                      \
        if (err_ != cudaSuccess) {                                                                            \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__);                 \
            exit(1);                                                                                          \
        }                                                                                                     \
    } while (0)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// Structure-of-arrays device state, for fully coalesced accesses.
struct DeviceBodies {
    double* pos = nullptr; // 3 * stride doubles: x, y, z planes
    double* vel = nullptr; // 3 * stride doubles: x, y, z planes
    double* scratch = nullptr;
    int n = 0;
    int stride = 0;
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

// One thread per body i; bodies j are streamed through shared memory in tiles.
// Tiles are visited in increasing j order so the accumulation order (and hence
// the floating-point result) matches the sequential reference implementation.
__global__ __launch_bounds__(BLOCK_SIZE) void computeForcesKernel(double* __restrict__ pos, double* __restrict__ vel,
    const int n, const int stride) {
    __shared__ double sx[BLOCK_SIZE];
    __shared__ double sy[BLOCK_SIZE];
    __shared__ double sz[BLOCK_SIZE];

    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool active = i < n;

    const double* __restrict__ px = pos;
    const double* __restrict__ py = pos + stride;
    const double* __restrict__ pz = pos + 2 * stride;

    const double xi = active ? px[i] : 0.0;
    const double yi = active ? py[i] : 0.0;
    const double zi = active ? pz[i] : 0.0;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        const int count = min(BLOCK_SIZE, n - tile);
        const int j = tile + threadIdx.x;
        if (static_cast<int>(threadIdx.x) < count) {
            sx[threadIdx.x] = px[j];
            sy[threadIdx.x] = py[j];
            sz[threadIdx.x] = pz[j];
        }
        __syncthreads();

        if (active) {
#pragma unroll 4
            for (int k = 0; k < count; ++k) {
                const double dx = sx[k] - xi;
                const double dy = sy[k] - yi;
                const double dz = sz[k] - zi;
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

    if (active) {
        vel[i] += DT * Fx;
        vel[stride + i] += DT * Fy;
        vel[2 * stride + i] += DT * Fz;
    }
}

__global__ void integrateBodiesKernel(double* __restrict__ pos, const double* __restrict__ vel, const int n,
    const int stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    pos[i] += vel[i] * DT;
    pos[stride + i] += vel[stride + i] * DT;
    pos[2 * stride + i] += vel[2 * stride + i] * DT;
}

// Partial potential energy of body i against all bodies j > i, matching the
// inner-loop accumulation order of the reference implementation.
__global__ __launch_bounds__(BLOCK_SIZE) void potentialEnergyKernel(const double* __restrict__ pos,
    double* __restrict__ partial, const int n, const int stride) {
    __shared__ double sx[BLOCK_SIZE];
    __shared__ double sy[BLOCK_SIZE];
    __shared__ double sz[BLOCK_SIZE];

    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool active = i < n;

    const double* __restrict__ px = pos;
    const double* __restrict__ py = pos + stride;
    const double* __restrict__ pz = pos + 2 * stride;

    const double xi = active ? px[i] : 0.0;
    const double yi = active ? py[i] : 0.0;
    const double zi = active ? pz[i] : 0.0;

    double e = 0.0;

    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        const int count = min(BLOCK_SIZE, n - tile);
        const int j = tile + threadIdx.x;
        if (static_cast<int>(threadIdx.x) < count) {
            sx[threadIdx.x] = px[j];
            sy[threadIdx.x] = py[j];
            sz[threadIdx.x] = pz[j];
        }
        __syncthreads();

        if (active && tile + count > i + 1) {
            const int kBegin = max(0, i + 1 - tile);
            for (int k = kBegin; k < count; ++k) {
                const double dx = sx[k] - xi;
                const double dy = sy[k] - yi;
                const double dz = sz[k] - zi;
                const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                e -= 1.0 / dist;
            }
        }
        __syncthreads();
    }

    if (active) partial[i] = e;
}

void allocateDevice(DeviceBodies& d, const std::vector<Body>& bodies) {
    d.n = static_cast<int>(bodies.size());
    d.stride = d.n;
    if (d.n == 0) return;

    CUDA_CHECK(cudaMalloc(&d.pos, sizeof(double) * 3 * d.stride));
    CUDA_CHECK(cudaMalloc(&d.vel, sizeof(double) * 3 * d.stride));
    CUDA_CHECK(cudaMalloc(&d.scratch, sizeof(double) * d.stride));

    std::vector<double> host(6 * static_cast<size_t>(d.stride));
    for (int i = 0; i < d.n; ++i) {
        host[i] = bodies[i].pos.x;
        host[d.stride + i] = bodies[i].pos.y;
        host[2 * d.stride + i] = bodies[i].pos.z;
        host[3 * d.stride + i] = bodies[i].vel.x;
        host[4 * d.stride + i] = bodies[i].vel.y;
        host[5 * d.stride + i] = bodies[i].vel.z;
    }
    CUDA_CHECK(cudaMemcpy(d.pos, host.data(), sizeof(double) * 3 * d.stride, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vel, host.data() + 3 * d.stride, sizeof(double) * 3 * d.stride, cudaMemcpyHostToDevice));
}

void downloadBodies(const DeviceBodies& d, std::vector<Body>& bodies) {
    if (d.n == 0) return;
    std::vector<double> host(6 * static_cast<size_t>(d.stride));
    CUDA_CHECK(cudaMemcpy(host.data(), d.pos, sizeof(double) * 3 * d.stride, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host.data() + 3 * d.stride, d.vel, sizeof(double) * 3 * d.stride, cudaMemcpyDeviceToHost));
    for (int i = 0; i < d.n; ++i) {
        bodies[i].pos.x = host[i];
        bodies[i].pos.y = host[d.stride + i];
        bodies[i].pos.z = host[2 * d.stride + i];
        bodies[i].vel.x = host[3 * d.stride + i];
        bodies[i].vel.y = host[4 * d.stride + i];
        bodies[i].vel.z = host[5 * d.stride + i];
    }
}

void freeDevice(DeviceBodies& d) {
    cudaFree(d.pos);
    cudaFree(d.vel);
    cudaFree(d.scratch);
    d.pos = d.vel = d.scratch = nullptr;
}

void computeForces(const DeviceBodies& d) {
    if (d.n == 0) return;
    const int blocks = (d.n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computeForcesKernel<<<blocks, BLOCK_SIZE>>>(d.pos, d.vel, d.n, d.stride);
}

void integrateBodies(const DeviceBodies& d) {
    if (d.n == 0) return;
    const int blocks = (d.n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    integrateBodiesKernel<<<blocks, BLOCK_SIZE>>>(d.pos, d.vel, d.n, d.stride);
}

double computeTotalEnergy(const std::vector<Body>& bodies, const DeviceBodies& d) {
    double energy = 0.0;

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    if (d.n > 0) {
        const int blocks = (d.n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        potentialEnergyKernel<<<blocks, BLOCK_SIZE>>>(d.pos, d.scratch, d.n, d.stride);
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> partial(d.n);
        CUDA_CHECK(cudaMemcpy(partial.data(), d.scratch, sizeof(double) * d.n, cudaMemcpyDeviceToHost));
        for (const double p : partial) {
            energy += p;
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

    // Initialize the GPU (and upload the initial state) before timing.
    CUDA_CHECK(cudaFree(nullptr));
    DeviceBodies device;
    allocateDevice(device, bodies);
    // Warm up: force module loading of both kernels (no-op launches with n = 0).
    computeForcesKernel<<<1, BLOCK_SIZE>>>(device.pos, device.vel, 0, device.stride);
    integrateBodiesKernel<<<1, BLOCK_SIZE>>>(device.pos, device.vel, 0, device.stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(device);
        integrateBodies(device);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    downloadBodies(device, bodies);

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
            double finalEnergy = computeTotalEnergy(bodies, device);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            freeDevice(device);
            return 0;
        } else {
            printf("Validation: FAILED\n");
            freeDevice(device);
            return 1;
        }
    }

    freeDevice(device);
    return 0;
}
