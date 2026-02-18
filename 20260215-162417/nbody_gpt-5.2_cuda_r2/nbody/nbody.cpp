#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        const cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
                         cudaGetErrorString(err__));                                      \
            std::exit(1);                                                                 \
        }                                                                                \
    } while (0)

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

void computeForces(std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    
    for (size_t i = 0; i < n; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        bodies[i].vel.x += DT * Fx;
        bodies[i].vel.y += DT * Fy;
        bodies[i].vel.z += DT * Fz;
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    for (auto& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

struct DeviceBodies {
    int n = 0;
    double *px = nullptr, *py = nullptr, *pz = nullptr;
    double *vx = nullptr, *vy = nullptr, *vz = nullptr;
};

static void allocDeviceBodies(DeviceBodies& d, int n) {
    d.n = n;
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d.px, bytes));
    CUDA_CHECK(cudaMalloc(&d.py, bytes));
    CUDA_CHECK(cudaMalloc(&d.pz, bytes));
    CUDA_CHECK(cudaMalloc(&d.vx, bytes));
    CUDA_CHECK(cudaMalloc(&d.vy, bytes));
    CUDA_CHECK(cudaMalloc(&d.vz, bytes));
}

static void freeDeviceBodies(DeviceBodies& d) {
    if (d.px) CUDA_CHECK(cudaFree(d.px));
    if (d.py) CUDA_CHECK(cudaFree(d.py));
    if (d.pz) CUDA_CHECK(cudaFree(d.pz));
    if (d.vx) CUDA_CHECK(cudaFree(d.vx));
    if (d.vy) CUDA_CHECK(cudaFree(d.vy));
    if (d.vz) CUDA_CHECK(cudaFree(d.vz));
    d = DeviceBodies{};
}

static void copyHostToDevice(const std::vector<Body>& bodies, DeviceBodies& d) {
    const int n = static_cast<int>(bodies.size());
    std::vector<double> hpx(n), hpy(n), hpz(n), hvx(n), hvy(n), hvz(n);
    for (int i = 0; i < n; ++i) {
        hpx[i] = bodies[i].pos.x;
        hpy[i] = bodies[i].pos.y;
        hpz[i] = bodies[i].pos.z;
        hvx[i] = bodies[i].vel.x;
        hvy[i] = bodies[i].vel.y;
        hvz[i] = bodies[i].vel.z;
    }
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d.px, hpx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.py, hpy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.pz, hpz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vx, hvx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vy, hvy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vz, hvz.data(), bytes, cudaMemcpyHostToDevice));
}

static void copyDeviceToHost(std::vector<Body>& bodies, const DeviceBodies& d) {
    const int n = static_cast<int>(bodies.size());
    std::vector<double> hpx(n), hpy(n), hpz(n), hvx(n), hvy(n), hvz(n);
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    CUDA_CHECK(cudaMemcpy(hpx.data(), d.px, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hpy.data(), d.py, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hpz.data(), d.pz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvx.data(), d.vx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvy.data(), d.vy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvz.data(), d.vz, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = hpx[i];
        bodies[i].pos.y = hpy[i];
        bodies[i].pos.z = hpz[i];
        bodies[i].vel.x = hvx[i];
        bodies[i].vel.y = hvy[i];
        bodies[i].vel.z = hvz[i];
    }
}

__global__ void nbodyStepKernel(const int n,
                               double* __restrict__ px,
                               double* __restrict__ py,
                               double* __restrict__ pz,
                               double* __restrict__ vx,
                               double* __restrict__ vy,
                               double* __restrict__ vz) {
    extern __shared__ double sh[];
    double* spx = sh;
    double* spy = sh + blockDim.x;
    double* spz = sh + 2 * blockDim.x;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double xi = px[i];
    const double yi = py[i];
    const double zi = pz[i];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < n; tile += blockDim.x) {
        const int j = tile + threadIdx.x;
        if (j < n) {
            spx[threadIdx.x] = px[j];
            spy[threadIdx.x] = py[j];
            spz[threadIdx.x] = pz[j];
        } else {
            spx[threadIdx.x] = 0.0;
            spy[threadIdx.x] = 0.0;
            spz[threadIdx.x] = 0.0;
        }
        __syncthreads();

        const int remaining = n - tile;
        const int tileCount = (remaining < static_cast<int>(blockDim.x)) ? remaining : static_cast<int>(blockDim.x);
        for (int k = 0; k < tileCount; ++k) {
            const double dx = spx[k] - xi;
            const double dy = spy[k] - yi;
            const double dz = spz[k] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        __syncthreads();
    }

    const double vxi = vx[i] + DT * Fx;
    const double vyi = vy[i] + DT * Fy;
    const double vzi = vz[i] + DT * Fz;
    vx[i] = vxi;
    vy[i] = vyi;
    vz[i] = vzi;

    px[i] = xi + vxi * DT;
    py[i] = yi + vyi * DT;
    pz[i] = zi + vzi * DT;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
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
    
    // Run simulation on GPU (CUDA)
    CUDA_CHECK(cudaSetDevice(0));

    DeviceBodies db;
    allocDeviceBodies(db, numBodies);
    copyHostToDevice(bodies, db);
    CUDA_CHECK(cudaFuncSetCacheConfig(nbodyStepKernel, cudaFuncCachePreferShared));

    constexpr int BLOCK_SIZE = 256;
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const size_t shmemBytes = 3ull * BLOCK_SIZE * sizeof(double);

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        nbodyStepKernel<<<grid, block, shmemBytes>>>(db.n, db.px, db.py, db.pz, db.vx, db.vy, db.vz);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    if (printResults || validate) {
        copyDeviceToHost(bodies, db);
    }
    freeDeviceBodies(db);
    
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
            double finalEnergy = computeTotalEnergy(bodies);
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
