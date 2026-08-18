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
constexpr int BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

struct DeviceBodies {
    double *storage = nullptr;
    double *x = nullptr, *y = nullptr, *z = nullptr;
    double *vx = nullptr, *vy = nullptr, *vz = nullptr;

    explicit DeviceBodies(size_t n) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&storage), 6 * n * sizeof(double)), "device allocation");
        x = storage;
        y = x + n;
        z = y + n;
        vx = z + n;
        vy = vx + n;
        vz = vy + n;
    }

    ~DeviceBodies() { cudaFree(storage); }
    DeviceBodies(const DeviceBodies&) = delete;
    DeviceBodies& operator=(const DeviceBodies&) = delete;
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

// Each position is fetched from global memory once per block rather than once
// per interacting body. Separate structure-of-arrays storage keeps those loads
// and the final velocity stores fully coalesced.
__global__ __launch_bounds__(BLOCK_SIZE)
void computeForcesKernel(const double* __restrict__ x,
                         const double* __restrict__ y,
                         const double* __restrict__ z,
                         double* __restrict__ vx,
                         double* __restrict__ vy,
                         double* __restrict__ vz,
                         int n) {
    __shared__ double tileX[BLOCK_SIZE];
    __shared__ double tileY[BLOCK_SIZE];
    __shared__ double tileZ[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const double xi = active ? x[i] : 0.0;
    const double yi = active ? y[i] : 0.0;
    const double zi = active ? z[i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int source = base + threadIdx.x;
        if (source < n) {
            tileX[threadIdx.x] = x[source];
            tileY[threadIdx.x] = y[source];
            tileZ[threadIdx.x] = z[source];
        }
        __syncthreads();

        if (active) {
            const int count = min(BLOCK_SIZE, n - base);
#pragma unroll 8
            for (int j = 0; j < count; ++j) {
                const double dx = tileX[j] - xi;
                const double dy = tileY[j] - yi;
                const double dz = tileZ[j] - zi;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = rsqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        vx[i] += DT * fx;
        vy[i] += DT * fy;
        vz[i] += DT * fz;
    }
}

__global__ __launch_bounds__(BLOCK_SIZE)
void integrateBodiesKernel(double* __restrict__ x,
                           double* __restrict__ y,
                           double* __restrict__ z,
                           const double* __restrict__ vx,
                           const double* __restrict__ vy,
                           const double* __restrict__ vz,
                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += vx[i] * DT;
        y[i] += vy[i] * DT;
        z[i] += vz[i] * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
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
        constexpr double maxPos = 1e6;
        constexpr double maxVel = 1e6;
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
    if (numBodies <= 0 || numSteps < 0) {
        std::fprintf(stderr, "Number of bodies must be positive and number of steps must be non-negative.\n");
        return 1;
    }

    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    randomizeBodies(bodies);

    const size_t n = static_cast<size_t>(numBodies);
    std::vector<double> hostStorage(6 * n);
    double* hx = hostStorage.data();
    double* hy = hx + n;
    double* hz = hy + n;
    double* hvx = hz + n;
    double* hvy = hvx + n;
    double* hvz = hvy + n;
    for (size_t i = 0; i < n; ++i) {
        hx[i] = bodies[i].pos.x; hy[i] = bodies[i].pos.y; hz[i] = bodies[i].pos.z;
        hvx[i] = bodies[i].vel.x; hvy[i] = bodies[i].vel.y; hvz[i] = bodies[i].vel.z;
    }

    DeviceBodies device(n);
    checkCuda(cudaMemcpy(device.storage, hostStorage.data(), 6 * n * sizeof(double), cudaMemcpyHostToDevice),
              "copying initial bodies to GPU");

    const int blocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // CUDA 12 loads kernels lazily. Resolve them before timing so the reported
    // simulation time measures the simulation rather than one-time setup.
    cudaFuncAttributes attributes{};
    checkCuda(cudaFuncGetAttributes(&attributes, computeForcesKernel), "loading force kernel");
    checkCuda(cudaFuncGetAttributes(&attributes, integrateBodiesKernel), "loading integration kernel");
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<blocks, BLOCK_SIZE>>>(device.x, device.y, device.z,
                                                    device.vx, device.vy, device.vz, numBodies);
        integrateBodiesKernel<<<blocks, BLOCK_SIZE>>>(device.x, device.y, device.z,
                                                      device.vx, device.vy, device.vz, numBodies);
    }
    checkCuda(cudaGetLastError(), "launching simulation kernels");
    checkCuda(cudaDeviceSynchronize(), "running simulation kernels");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Simulation time: %ld ms\n", static_cast<long>(duration.count()));

    checkCuda(cudaMemcpy(hostStorage.data(), device.storage, 6 * n * sizeof(double), cudaMemcpyDeviceToHost),
              "copying final bodies from GPU");
    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(hx[i], hy[i], hz[i]);
        bodies[i].vel = Vec3(hvx[i], hvy[i], hvz[i]);
    }

    if (printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(6 * n);
        for (const auto& body : bodies) {
            bodyData.push_back(body.pos.x); bodyData.push_back(body.pos.y); bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x); bodyData.push_back(body.vel.y); bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    if (validate) {
        printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) {
            printf("Final energy: %.6f\n", computeTotalEnergy(bodies));
            printf("Validation: PASSED\n");
            return 0;
        }
        printf("Validation: FAILED\n");
        return 1;
    }
    return 0;
}
