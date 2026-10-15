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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

constexpr int BX = 64;     // bodies per block
constexpr int TILE = 256;  // j-bodies staged in shared memory per iteration

// Each block handles BX bodies i; SLICES thread-rows split the j-range of every tile.
template <int SLICES>
__global__ void __launch_bounds__(BX * SLICES)
forcesKernel(const double4* __restrict__ pos, double4* __restrict__ vel, int n) {
    __shared__ double4 tile[TILE];
    __shared__ double3 red[SLICES > 1 ? SLICES : 1][BX];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * BX + tx;
    const int i = blockIdx.x * BX + tx;
    const double4 pi = (i < n) ? pos[i] : make_double4(0, 0, 0, 0);
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += TILE) {
        const int cnt = min(TILE, n - base);
        __syncthreads();
        for (int k = tid; k < cnt; k += BX * SLICES) tile[k] = pos[base + k];
        __syncthreads();
        const int lo = (cnt * ty) / SLICES, hi = (cnt * (ty + 1)) / SLICES;
#pragma unroll 8
        for (int j = lo; j < hi; ++j) {
            const double4 pj = tile[j];
            const double dx = pj.x - pi.x;
            const double dy = pj.y - pi.y;
            const double dz = pj.z - pi.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
    }

    if (SLICES > 1) {
        red[ty][tx] = make_double3(Fx, Fy, Fz);
        __syncthreads();
        if (ty != 0) return;
        Fx = Fy = Fz = 0.0;
        for (int s = 0; s < SLICES; ++s) {
            Fx += red[s][tx].x;
            Fy += red[s][tx].y;
            Fz += red[s][tx].z;
        }
    }
    if (i < n) {
        double4 v = vel[i];
        v.x += DT * Fx;
        v.y += DT * Fy;
        v.z += DT * Fz;
        vel[i] = v;
    }
}

__global__ void integrateKernel(double4* __restrict__ pos, const double4* __restrict__ vel, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double4 p = pos[i];
        const double4 v = vel[i];
        p.x += v.x * DT;
        p.y += v.y * DT;
        p.z += v.z * DT;
        pos[i] = p;
    }
}

// Per-body potential energy contribution (pairs j > i), reduced per block.
__global__ void energyKernel(const double4* __restrict__ pos, double* __restrict__ out, int n) {
    __shared__ double4 tile[256];
    __shared__ double warpSum[8];
    const int i = blockIdx.x * 256 + threadIdx.x;
    const double4 pi = (i < n) ? pos[i] : make_double4(0, 0, 0, 0);
    double e = 0.0;
    for (int base = blockIdx.x * 256; base < n; base += 256) {
        __syncthreads();
        if (base + (int)threadIdx.x < n) tile[threadIdx.x] = pos[base + threadIdx.x];
        __syncthreads();
        const int cnt = min(256, n - base);
        if (i < n) {
            for (int k = 0; k < cnt; ++k) {
                const int j = base + k;
                if (j <= i) continue;
                const double dx = tile[k].x - pi.x;
                const double dy = tile[k].y - pi.y;
                const double dz = tile[k].z - pi.z;
                e -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            }
        }
    }
    for (int o = 16; o > 0; o >>= 1) e += __shfl_down_sync(0xffffffffu, e, o);
    if ((threadIdx.x & 31) == 0) warpSum[threadIdx.x >> 5] = e;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int w = 0; w < 8; ++w) t += warpSum[w];
        out[blockIdx.x] = t;
    }
}

static void launchForces(const double4* pos, double4* vel, int n) {
    const int blocks = (n + BX - 1) / BX;
    // Split the j-range across more threads when there are too few bodies to fill the GPU.
    if (blocks >= 1024) {
        forcesKernel<1><<<blocks, dim3(BX, 1)>>>(pos, vel, n);
    } else if (blocks >= 512) {
        forcesKernel<2><<<blocks, dim3(BX, 2)>>>(pos, vel, n);
    } else if (blocks >= 128) {
        forcesKernel<4><<<blocks, dim3(BX, 4)>>>(pos, vel, n);
    } else {
        forcesKernel<8><<<blocks, dim3(BX, 8)>>>(pos, vel, n);
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies, const double4* dPos) {
    double energy = 0.0;
    const int n = (int)bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies), computed on the GPU
    const int blocks = (n + 255) / 256;
    double* dOut;
    CUDA_CHECK(cudaMalloc(&dOut, sizeof(double) * blocks));
    energyKernel<<<blocks, 256>>>(dPos, dOut, n);
    std::vector<double> part(blocks);
    CUDA_CHECK(cudaMemcpy(part.data(), dOut, sizeof(double) * blocks, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dOut));
    for (double p : part) energy += p;
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
    
    // Run simulation
    const int n = numBodies;
    CUDA_CHECK(cudaFree(0));  // create context before timing
    std::vector<double4> hPos(n), hVel(n);
    for (int i = 0; i < n; ++i) {
        hPos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
        hVel[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
    }
    double4 *dPos = nullptr, *dVel = nullptr;
    CUDA_CHECK(cudaMalloc(&dPos, sizeof(double4) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMalloc(&dVel, sizeof(double4) * (n > 0 ? n : 1)));

    auto start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(dPos, hPos.data(), sizeof(double4) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, hVel.data(), sizeof(double4) * n, cudaMemcpyHostToDevice));
    if (n > 0) {
        for (int step = 0; step < numSteps; ++step) {
            launchForces(dPos, dVel, n);
            integrateKernel<<<(n + 255) / 256, 256>>>(dPos, dVel, n);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hPos.data(), dPos, sizeof(double4) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hVel.data(), dVel, sizeof(double4) * n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(hPos[i].x, hPos[i].y, hPos[i].z);
        bodies[i].vel = Vec3(hVel[i].x, hVel[i].y, hVel[i].z);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
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
            double finalEnergy = computeTotalEnergy(bodies, dPos);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            cudaFree(dPos);
            cudaFree(dVel);
            return 0;
        } else {
            printf("Validation: FAILED\n");
            cudaFree(dPos);
            cudaFree(dVel);
            return 1;
        }
    }

    cudaFree(dPos);
    cudaFree(dVel);
    return 0;
}
