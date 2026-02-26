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

static inline void cudaCheck(cudaError_t err, const char* stmt, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", stmt, file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CUDA_CHECK(stmt) cudaCheck((stmt), #stmt, __FILE__, __LINE__)

struct DeviceState {
    int n = 0;
    double* px = nullptr;
    double* py = nullptr;
    double* pz = nullptr;
    double* vx = nullptr;
    double* vy = nullptr;
    double* vz = nullptr;
};

static DeviceState createDeviceState(const int n) {
    DeviceState d;
    d.n = n;
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d.px, bytes));
    CUDA_CHECK(cudaMalloc(&d.py, bytes));
    CUDA_CHECK(cudaMalloc(&d.pz, bytes));
    CUDA_CHECK(cudaMalloc(&d.vx, bytes));
    CUDA_CHECK(cudaMalloc(&d.vy, bytes));
    CUDA_CHECK(cudaMalloc(&d.vz, bytes));
    return d;
}

static void destroyDeviceState(DeviceState& d) {
    if (d.px) CUDA_CHECK(cudaFree(d.px));
    if (d.py) CUDA_CHECK(cudaFree(d.py));
    if (d.pz) CUDA_CHECK(cudaFree(d.pz));
    if (d.vx) CUDA_CHECK(cudaFree(d.vx));
    if (d.vy) CUDA_CHECK(cudaFree(d.vy));
    if (d.vz) CUDA_CHECK(cudaFree(d.vz));
    d = {};
}

static void uploadToDevice(const DeviceState& d, const std::vector<Body>& bodies) {
    const int n = d.n;
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

static void downloadFromDevice(std::vector<Body>& bodies, const DeviceState& d) {
    const int n = d.n;
    std::vector<double> hpx(n), hpy(n), hpz(n), hvx(n), hvy(n), hvz(n);
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    CUDA_CHECK(cudaMemcpy(hpx.data(), d.px, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hpy.data(), d.py, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hpz.data(), d.pz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvx.data(), d.vx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvy.data(), d.vy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hvz.data(), d.vz, bytes, cudaMemcpyDeviceToHost));

    bodies.resize(n);
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = hpx[i];
        bodies[i].pos.y = hpy[i];
        bodies[i].pos.z = hpz[i];
        bodies[i].vel.x = hvx[i];
        bodies[i].vel.y = hvy[i];
        bodies[i].vel.z = hvz[i];
    }
}

constexpr int BLOCK_SIZE = 256;

__global__ __launch_bounds__(BLOCK_SIZE) void nbodyStepKernel(double* __restrict__ px,
                                                             double* __restrict__ py,
                                                             double* __restrict__ pz,
                                                             double* __restrict__ vx,
                                                             double* __restrict__ vy,
                                                             double* __restrict__ vz,
                                                             const int n) {
    __shared__ double shx[BLOCK_SIZE];
    __shared__ double shy[BLOCK_SIZE];
    __shared__ double shz[BLOCK_SIZE];

    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (i >= n) return;

    const double xi = px[i];
    const double yi = py[i];
    const double zi = pz[i];

    double Fx = 0.0;
    double Fy = 0.0;
    double Fz = 0.0;

    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        const int idx = tile + static_cast<int>(threadIdx.x);
        if (idx < n) {
            shx[threadIdx.x] = px[idx];
            shy[threadIdx.x] = py[idx];
            shz[threadIdx.x] = pz[idx];
        } else {
            shx[threadIdx.x] = 0.0;
            shy[threadIdx.x] = 0.0;
            shz[threadIdx.x] = 0.0;
        }
        __syncthreads();

        const int tileLen = (n - tile < BLOCK_SIZE) ? (n - tile) : BLOCK_SIZE;
#pragma unroll 8
        for (int k = 0; k < tileLen; ++k) {
            const double dx = shx[k] - xi;
            const double dy = shy[k] - yi;
            const double dz = shz[k] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / ::sqrt(distSqr);
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

static void runSimulationCUDA(const DeviceState& d, const int numSteps) {
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((d.n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int step = 0; step < numSteps; ++step) {
        nbodyStepKernel<<<grid, block>>>(d.px, d.py, d.pz, d.vx, d.vy, d.vz, d.n);
        CUDA_CHECK(cudaGetLastError());
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

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    DeviceState d = createDeviceState(numBodies);
    uploadToDevice(d, bodies);

    // Run simulation (GPU)
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    runSimulationCUDA(d, numSteps);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    if (printResults || validate) {
        downloadFromDevice(bodies, d);
    }

    destroyDeviceState(d);

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
