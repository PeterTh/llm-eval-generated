// N-Body benchmark, CUDA implementation.
//
// All O(n^2) work (force computation, potential energy) runs on the GPU in
// double precision; integration runs in a separate kernel after all forces
// for a step are known, preserving the original (synchronous) semantics.
// Body data lives on the device for the whole simulation; only the final
// state is copied back.
//
// This file is compiled as CUDA (see CMakeLists.txt).
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

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err__ = (call);                                                           \
        if (err__ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,       \
                    cudaGetErrorString(err__));                                               \
            exit(EXIT_FAILURE);                                                               \
        }                                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------
constexpr int BLOCK_SIZE = 256;   // threads per block (= bodies per shared-memory tile)
constexpr int UNROLL = 4;         // inner-loop unroll factor (BLOCK_SIZE % UNROLL == 0)

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ void accumulateForce(const double4 pi, const double4 pj,
                                                double& Fx, double& Fy, double& Fz) {
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

// One thread per body i; all-pairs over j using shared-memory tiles.
// Positions are read-only during this kernel, velocities are updated in place
// (each thread owns exactly one velocity), matching computeForces().
__global__ void __launch_bounds__(BLOCK_SIZE)
computeForcesKernel(const double4* __restrict__ pos, double4* __restrict__ vel, const int n) {
    __shared__ double4 tile[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Threads beyond n still participate in tile loads; they use a dummy position.
    const double4 pi = (i < n) ? pos[i] : make_double4(0.0, 0.0, 0.0, 0.0);

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    const int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        const int j = t * BLOCK_SIZE + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? pos[j] : make_double4(0.0, 0.0, 0.0, 0.0);
        __syncthreads();

        const int base = t * BLOCK_SIZE;
        const int count = min(BLOCK_SIZE, n - base);
        if (count == BLOCK_SIZE) {
            // Full tile: fully unrolled, no bounds checks.
#pragma unroll UNROLL
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                accumulateForce(pi, tile[k], Fx, Fy, Fz);
            }
        } else {
            // Last partial tile.
            for (int k = 0; k < count; ++k) {
                accumulateForce(pi, tile[k], Fx, Fy, Fz);
            }
        }
        __syncthreads();
    }

    if (i < n) {
        double4 v = vel[i];
        v.x += DT * Fx;
        v.y += DT * Fy;
        v.z += DT * Fz;
        vel[i] = v;
    }
}

__global__ void integrateBodiesKernel(double4* __restrict__ pos, const double4* __restrict__ vel,
                                      const int n) {
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

// Block-wide sum reduction (BLOCK_SIZE threads), result valid in thread 0.
__device__ __forceinline__ double blockReduceSum(double val) {
    __shared__ double shared[BLOCK_SIZE / 32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : 0.0;
    if (wid == 0) {
#pragma unroll
        for (int offset = (BLOCK_SIZE / 32) / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffffu, val, offset);
        }
    }
    return val;
}

// Total energy: per-block partial sums of kinetic energy plus potential energy
// over pairs (i, j>i). Each thread handles body i and iterates over j > i via
// shared-memory tiles. Partial results are written to blockSums[blockIdx.x].
__global__ void __launch_bounds__(BLOCK_SIZE)
totalEnergyKernel(const double4* __restrict__ pos, const double4* __restrict__ vel, const int n,
                  double* __restrict__ blockSums) {
    __shared__ double4 tile[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const double4 pi = active ? pos[i] : make_double4(0.0, 0.0, 0.0, 0.0);

    double e = 0.0;
    if (active) {
        const double4 v = vel[i];
        e += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    // Only tiles starting at or after this block's first body contain j > i.
    const int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = blockIdx.x; t < numTiles; ++t) {
        const int j = t * BLOCK_SIZE + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? pos[j] : make_double4(0.0, 0.0, 0.0, 0.0);
        __syncthreads();

        if (active) {
            const int base = t * BLOCK_SIZE;
            const int count = min(BLOCK_SIZE, n - base);
            // Start after i within this tile (only matters for the diagonal tile).
            const int kStart = max(0, i + 1 - base);
            for (int k = kStart; k < count; ++k) {
                const double4 pj = tile[k];
                const double dx = pj.x - pi.x;
                const double dy = pj.y - pi.y;
                const double dz = pj.z - pi.z;
                const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                e -= 1.0 / dist;
            }
        }
        __syncthreads();
    }

    const double total = blockReduceSum(e);
    if (threadIdx.x == 0) blockSums[blockIdx.x] = total;
}

// ---------------------------------------------------------------------------
// Host-side simulation state
// ---------------------------------------------------------------------------
struct DeviceBodies {
    double4* pos = nullptr;
    double4* vel = nullptr;
    int n = 0;
    int numBlocks = 0;

    explicit DeviceBodies(const std::vector<Body>& bodies) {
        n = static_cast<int>(bodies.size());
        numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const size_t bytes = static_cast<size_t>(n) * sizeof(double4);
        CUDA_CHECK(cudaMalloc(&pos, bytes > 0 ? bytes : sizeof(double4)));
        CUDA_CHECK(cudaMalloc(&vel, bytes > 0 ? bytes : sizeof(double4)));
        upload(bodies);
    }
    ~DeviceBodies() {
        cudaFree(pos);
        cudaFree(vel);
    }
    DeviceBodies(const DeviceBodies&) = delete;
    DeviceBodies& operator=(const DeviceBodies&) = delete;

    void upload(const std::vector<Body>& bodies) {
        std::vector<double4> hp(n), hv(n);
        for (int i = 0; i < n; ++i) {
            hp[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
            hv[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
        }
        if (n > 0) {
            CUDA_CHECK(cudaMemcpy(pos, hp.data(), n * sizeof(double4), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(vel, hv.data(), n * sizeof(double4), cudaMemcpyHostToDevice));
        }
    }

    void download(std::vector<Body>& bodies) const {
        std::vector<double4> hp(n), hv(n);
        if (n > 0) {
            CUDA_CHECK(cudaMemcpy(hp.data(), pos, n * sizeof(double4), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hv.data(), vel, n * sizeof(double4), cudaMemcpyDeviceToHost));
        }
        for (int i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(hp[i].x, hp[i].y, hp[i].z);
            bodies[i].vel = Vec3(hv[i].x, hv[i].y, hv[i].z);
        }
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

void computeForces(DeviceBodies& d) {
    if (d.n == 0) return;
    computeForcesKernel<<<d.numBlocks, BLOCK_SIZE>>>(d.pos, d.vel, d.n);
    CUDA_CHECK(cudaGetLastError());
}

void integrateBodies(DeviceBodies& d) {
    if (d.n == 0) return;
    integrateBodiesKernel<<<d.numBlocks, BLOCK_SIZE>>>(d.pos, d.vel, d.n);
    CUDA_CHECK(cudaGetLastError());
}

double computeTotalEnergy(const DeviceBodies& d) {
    if (d.n == 0) return 0.0;
    double* dBlockSums = nullptr;
    CUDA_CHECK(cudaMalloc(&dBlockSums, d.numBlocks * sizeof(double)));
    totalEnergyKernel<<<d.numBlocks, BLOCK_SIZE>>>(d.pos, d.vel, d.n, dBlockSums);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> blockSums(d.numBlocks);
    CUDA_CHECK(cudaMemcpy(blockSums.data(), dBlockSums, d.numBlocks * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dBlockSums));
    double energy = 0.0;
    for (const double s : blockSums) energy += s;
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

    // Initialize CUDA context and upload bodies (outside the timed region).
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    {
        // Force kernel module loading now so lazy loading does not land in the timed loop.
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, computeForcesKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, integrateBodiesKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, totalEnergyKernel));
    }
    DeviceBodies device(bodies);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(device);
        integrateBodies(device);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());

    // Copy final state back to host.
    device.download(bodies);
    
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
            double finalEnergy = computeTotalEnergy(device);
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
