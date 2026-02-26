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

static inline void cudaCheck(cudaError_t err, const char* call, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s at %s:%d: %s\n", call, file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CUDA_CHECK(x) cudaCheck((x), #x, __FILE__, __LINE__)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

struct BodiesSoA {
    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;
};

static BodiesSoA toSoA(const std::vector<Body>& bodies) {
    BodiesSoA s;
    const size_t n = bodies.size();
    s.x.resize(n);
    s.y.resize(n);
    s.z.resize(n);
    s.vx.resize(n);
    s.vy.resize(n);
    s.vz.resize(n);
    for (size_t i = 0; i < n; ++i) {
        s.x[i] = bodies[i].pos.x;
        s.y[i] = bodies[i].pos.y;
        s.z[i] = bodies[i].pos.z;
        s.vx[i] = bodies[i].vel.x;
        s.vy[i] = bodies[i].vel.y;
        s.vz[i] = bodies[i].vel.z;
    }
    return s;
}

static void fromSoA(std::vector<Body>& bodies, const BodiesSoA& s) {
    const size_t n = bodies.size();
    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos.x = s.x[i];
        bodies[i].pos.y = s.y[i];
        bodies[i].pos.z = s.z[i];
        bodies[i].vel.x = s.vx[i];
        bodies[i].vel.y = s.vy[i];
        bodies[i].vel.z = s.vz[i];
    }
}

constexpr int BLOCK_SIZE = 256;

__global__ void compute_forces_kernel(const double* __restrict__ x,
                                     const double* __restrict__ y,
                                     const double* __restrict__ z,
                                     double* __restrict__ vx,
                                     double* __restrict__ vy,
                                     double* __restrict__ vz,
                                     int n) {
    __shared__ double shx[BLOCK_SIZE];
    __shared__ double shy[BLOCK_SIZE];
    __shared__ double shz[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * BLOCK_SIZE + tid;
    const bool active = (i < n);

    const double xi = active ? x[i] : 0.0;
    const double yi = active ? y[i] : 0.0;
    const double zi = active ? z[i] : 0.0;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        const int j = tile + tid;
        if (j < n) {
            shx[tid] = x[j];
            shy[tid] = y[j];
            shz[tid] = z[j];
        } else {
            shx[tid] = 0.0;
            shy[tid] = 0.0;
            shz[tid] = 0.0;
        }
        __syncthreads();

        const int remain = n - tile;
        const int tileSize = (remain < BLOCK_SIZE) ? remain : BLOCK_SIZE;

        #pragma unroll 4
        for (int k = 0; k < tileSize; ++k) {
            const double dx = shx[k] - xi;
            const double dy = shy[k] - yi;
            const double dz = shz[k] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    if (active) {
        vx[i] += DT * Fx;
        vy[i] += DT * Fy;
        vz[i] += DT * Fz;
    }
}

__global__ void integrate_kernel(double* __restrict__ x,
                                double* __restrict__ y,
                                double* __restrict__ z,
                                const double* __restrict__ vx,
                                const double* __restrict__ vy,
                                const double* __restrict__ vz,
                                int n) {
    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) {
        x[i] += vx[i] * DT;
        y[i] += vy[i] * DT;
        z[i] += vz[i] * DT;
    }
}

static long runCudaSimulation(BodiesSoA& h, int numSteps) {
    const int n = static_cast<int>(h.x.size());
    if (n <= 0 || numSteps <= 0) {
        return 0;
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        std::fprintf(stderr, "No CUDA devices found.\n");
        std::exit(1);
    }
    CUDA_CHECK(cudaSetDevice(0));

    double *dx = nullptr, *dy = nullptr, *dz = nullptr;
    double *dvx = nullptr, *dvy = nullptr, *dvz = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);

    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMalloc(&dz, bytes));
    CUDA_CHECK(cudaMalloc(&dvx, bytes));
    CUDA_CHECK(cudaMalloc(&dvy, bytes));
    CUDA_CHECK(cudaMalloc(&dvz, bytes));

    CUDA_CHECK(cudaMemcpy(dx, h.x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, h.y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dz, h.z.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvx, h.vx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvy, h.vy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvz, h.vz.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(BLOCK_SIZE);
    const dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Ensure CUDA context is fully initialized before timing.
    compute_forces_kernel<<<1, 1>>>(dx, dy, dz, dvx, dvy, dvz, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int step = 0; step < numSteps; ++step) {
        compute_forces_kernel<<<grid, block>>>(dx, dy, dz, dvx, dvy, dvz, n);
        integrate_kernel<<<grid, block>>>(dx, dy, dz, dvx, dvy, dvz, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h.x.data(), dx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.y.data(), dy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.z.data(), dz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.vx.data(), dvx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.vy.data(), dvy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.vz.data(), dvz, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dz));
    CUDA_CHECK(cudaFree(dvx));
    CUDA_CHECK(cudaFree(dvy));
    CUDA_CHECK(cudaFree(dvz));

    return static_cast<long>(ms + 0.5f);
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
    
    // Run simulation (CUDA)
    BodiesSoA soa = toSoA(bodies);
    const long simMs = runCudaSimulation(soa, numSteps);
    fromSoA(bodies, soa);

    printf("Simulation time: %ld ms\n", simMs);
    
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
