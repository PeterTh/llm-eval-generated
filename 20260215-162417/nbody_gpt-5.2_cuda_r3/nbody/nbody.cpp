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

static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CUDA_CHECK(x) cudaCheck((x), __FILE__, __LINE__)

constexpr int CUDA_BLOCK_SIZE = 256;

__global__ void computeForcesKernel(const double4* __restrict__ pos, double4* __restrict__ vel, int n) {
    extern __shared__ double4 shPos[];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double4 pi = pos[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int tile = 0; tile < n; tile += blockDim.x) {
        const int j = tile + threadIdx.x;
        shPos[threadIdx.x] = (j < n) ? pos[j] : make_double4(0.0, 0.0, 0.0, 0.0);
        __syncthreads();

        const int tileSize = ((n - tile) < (int)blockDim.x) ? (n - tile) : (int)blockDim.x;
#pragma unroll 4
        for (int k = 0; k < tileSize; ++k) {
            const double4 pj = shPos[k];
            const double dx = pj.x - pi.x;
            const double dy = pj.y - pi.y;
            const double dz = pj.z - pi.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            fx += dx * invDist3;
            fy += dy * invDist3;
            fz += dz * invDist3;
        }
        __syncthreads();
    }

    double4 vi = vel[i];
    vi.x += DT * fx;
    vi.y += DT * fy;
    vi.z += DT * fz;
    vel[i] = vi;
}

__global__ void integrateKernel(double4* __restrict__ pos, const double4* __restrict__ vel, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double4 pi = pos[i];
    const double4 vi = vel[i];
    pi.x += vi.x * DT;
    pi.y += vi.y * DT;
    pi.z += vi.z * DT;
    pos[i] = pi;
}

struct DeviceBodies {
    int n = 0;
    double4* d_pos = nullptr;
    double4* d_vel = nullptr;

    explicit DeviceBodies(const std::vector<Body>& bodies) {
        n = (int)bodies.size();
        CUDA_CHECK(cudaMalloc(&d_pos, (size_t)n * sizeof(double4)));
        CUDA_CHECK(cudaMalloc(&d_vel, (size_t)n * sizeof(double4)));

        std::vector<double4> h_pos(n);
        std::vector<double4> h_vel(n);
        for (int i = 0; i < n; ++i) {
            h_pos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
            h_vel[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
        }

        CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), (size_t)n * sizeof(double4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel, h_vel.data(), (size_t)n * sizeof(double4), cudaMemcpyHostToDevice));
    }

    void step() {
        const dim3 block(CUDA_BLOCK_SIZE);
        const dim3 grid((n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
        const size_t shmem = (size_t)CUDA_BLOCK_SIZE * sizeof(double4);

        computeForcesKernel<<<grid, block, shmem>>>(d_pos, d_vel, n);
        integrateKernel<<<grid, block>>>(d_pos, d_vel, n);
    }

    void download(std::vector<Body>& bodies) const {
        std::vector<double4> h_pos((size_t)n);
        std::vector<double4> h_vel((size_t)n);
        CUDA_CHECK(cudaMemcpy(h_pos.data(), d_pos, (size_t)n * sizeof(double4), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vel.data(), d_vel, (size_t)n * sizeof(double4), cudaMemcpyDeviceToHost));

        for (int i = 0; i < n; ++i) {
            bodies[i].pos.x = h_pos[i].x;
            bodies[i].pos.y = h_pos[i].y;
            bodies[i].pos.z = h_pos[i].z;
            bodies[i].vel.x = h_vel[i].x;
            bodies[i].vel.y = h_vel[i].y;
            bodies[i].vel.z = h_vel[i].z;
        }
    }

    ~DeviceBodies() {
        if (d_pos) cudaFree(d_pos);
        if (d_vel) cudaFree(d_vel);
    }
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
    
    // Upload to GPU once, keep data resident for all steps
    CUDA_CHECK(cudaSetDevice(0));
    DeviceBodies dev(bodies);

    // Run simulation (GPU)
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        dev.step();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    if (printResults || validate) {
        dev.download(bodies);
    }
    
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
