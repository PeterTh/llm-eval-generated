// N-Body benchmark, CUDA parallelization.
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

// Thread block size of the elementwise (integrate / reduce / validate) kernels.
constexpr int BLOCK_SIZE = 256;

// Thread block size of the force kernel, which also determines the number of
// bodies staged in shared memory per tile. 64 measures fastest: the kernel is
// limited by double precision throughput rather than by shared memory traffic,
// and small blocks schedule better when the body count is low.
constexpr int TILE_SIZE = 64;

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        const cudaError_t err_ = (call);                                                                               \
        if (err_ != cudaSuccess) {                                                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__, __LINE__,                \
                    cudaGetErrorString(err_));                                                                         \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// Structure-of-arrays mirror of the body state, used on the device so that all
// global memory accesses are fully coalesced.
struct DeviceBodies {
    double* pos[3] = {nullptr, nullptr, nullptr};
    double* vel[3] = {nullptr, nullptr, nullptr};
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

// 1/sqrt(x) in double precision, built from the hardware single precision
// reciprocal square root plus two Newton-Raphson steps. The starting value is
// accurate to ~22 bits and each step doubles that, so the result is accurate to
// the last bit or two of a double while avoiding the very expensive
// double-precision sqrt + divide sequence (~40% of the kernel's arithmetic).
__device__ __forceinline__ double rsqrtNewton(const double x) {
    double y = (double)rsqrtf((float)x);
    y = y * (1.5 - 0.5 * x * y * y);
    y = y * (1.5 - 0.5 * x * y * y);
    return y;
}

// Each thread owns one body i and accumulates the interactions with a
// contiguous range of j in increasing order. Positions are staged through
// shared memory in tiles, which turns the n^2 global loads into n^2/TILE_SIZE
// loads and lets every j value be broadcast to the whole warp.
//
// blockIdx.y splits the j range into gridDim.y chunks so that the GPU can be
// filled even when the body count alone does not provide enough blocks. With a
// single chunk (partial == nullptr) the force is applied to the velocity
// directly; otherwise each chunk writes its partial force and a second kernel
// combines them in chunk order.
__global__ void computeForcesKernel(const double* __restrict__ posX, const double* __restrict__ posY,
                                    const double* __restrict__ posZ, double* __restrict__ velX,
                                    double* __restrict__ velY, double* __restrict__ velZ,
                                    double* __restrict__ partial, const int n, const int chunkTiles) {
    __shared__ double sx[TILE_SIZE];
    __shared__ double sy[TILE_SIZE];
    __shared__ double sz[TILE_SIZE];

    const int i = blockIdx.x * TILE_SIZE + threadIdx.x;
    const bool active = (i < n);

    // Inactive threads still participate in the tile loads and barriers; they
    // just read body 0 to stay in bounds and discard their result.
    const int iSafe = active ? i : 0;
    const double xi = posX[iSafe];
    const double yi = posY[iSafe];
    const double zi = posZ[iSafe];

    const int tileBegin = blockIdx.y * chunkTiles * TILE_SIZE;
    int tileEndAll = tileBegin + chunkTiles * TILE_SIZE;
    if (tileEndAll > n) tileEndAll = n;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = tileBegin; tile < tileEndAll; tile += TILE_SIZE) {
        const int idx = tile + threadIdx.x;
        if (idx < n) {
            sx[threadIdx.x] = posX[idx];
            sy[threadIdx.x] = posY[idx];
            sz[threadIdx.x] = posZ[idx];
        }
        __syncthreads();

        const int tileEnd = min(TILE_SIZE, tileEndAll - tile);
        if (tileEnd == TILE_SIZE) {
#pragma unroll 4
            for (int j = 0; j < TILE_SIZE; ++j) {
                const double dx = sx[j] - xi;
                const double dy = sy[j] - yi;
                const double dz = sz[j] - zi;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = rsqrtNewton(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        } else {
            for (int j = 0; j < tileEnd; ++j) {
                const double dx = sx[j] - xi;
                const double dy = sy[j] - yi;
                const double dz = sz[j] - zi;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = rsqrtNewton(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (!active) return;

    if (partial == nullptr) {
        velX[i] += DT * Fx;
        velY[i] += DT * Fy;
        velZ[i] += DT * Fz;
    } else {
        const size_t base = (size_t)blockIdx.y * 3 * n + i;
        partial[base] = Fx;
        partial[base + n] = Fy;
        partial[base + 2 * n] = Fz;
    }
}

// Sums the per-chunk partial forces in increasing chunk order and applies them.
__global__ void applyForcesKernel(double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
                                  const double* __restrict__ partial, const int n, const int chunks) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int c = 0; c < chunks; ++c) {
        const size_t base = (size_t)c * 3 * n + i;
        Fx += partial[base];
        Fy += partial[base + n];
        Fz += partial[base + 2 * n];
    }

    velX[i] += DT * Fx;
    velY[i] += DT * Fy;
    velZ[i] += DT * Fz;
}

__global__ void integrateBodiesKernel(double* __restrict__ posX, double* __restrict__ posY,
                                      double* __restrict__ posZ, const double* __restrict__ velX,
                                      const double* __restrict__ velY, const double* __restrict__ velZ,
                                      const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    posX[i] += velX[i] * DT;
    posY[i] += velY[i] * DT;
    posZ[i] += velZ[i] * DT;
}

// Per-body partial potential energy: body i sums the terms for j = i+1..n-1 in
// the original order. The partials are combined on the host, preserving the
// outer iteration order.
__global__ void potentialEnergyKernel(const double* __restrict__ posX, const double* __restrict__ posY,
                                      const double* __restrict__ posZ, double* __restrict__ partial,
                                      const int n) {
    __shared__ double sx[BLOCK_SIZE];
    __shared__ double sy[BLOCK_SIZE];
    __shared__ double sz[BLOCK_SIZE];

    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool active = (i < n);
    const int iSafe = active ? i : 0;
    const double xi = posX[iSafe];
    const double yi = posY[iSafe];
    const double zi = posZ[iSafe];

    double e = 0.0;

    // Only tiles at or beyond this thread block can contain j > i.
    const int firstTile = blockIdx.x * BLOCK_SIZE;
    for (int tile = firstTile; tile < n; tile += BLOCK_SIZE) {
        const int idx = tile + threadIdx.x;
        if (idx < n) {
            sx[threadIdx.x] = posX[idx];
            sy[threadIdx.x] = posY[idx];
            sz[threadIdx.x] = posZ[idx];
        }
        __syncthreads();

        const int tileEnd = min(BLOCK_SIZE, n - tile);
        const int jStart = (tile <= i) ? (i - tile + 1) : 0;
        for (int j = jStart; j < tileEnd; ++j) {
            const double dx = sx[j] - xi;
            const double dy = sy[j] - yi;
            const double dz = sz[j] - zi;
            const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            e -= 1.0 / dist;
        }
        __syncthreads();
    }

    if (active) partial[i] = e;
}

__global__ void validateKernel(const double* __restrict__ posX, const double* __restrict__ posY,
                               const double* __restrict__ posZ, const double* __restrict__ velX,
                               const double* __restrict__ velY, const double* __restrict__ velZ,
                               int* __restrict__ failure, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double px = posX[i], py = posY[i], pz = posZ[i];
    const double vx = velX[i], vy = velY[i], vz = velZ[i];

    // Bit 0: non-finite value, bit 1: position out of bounds, bit 2: velocity out of bounds.
    int flags = 0;
    if (!isfinite(px) || !isfinite(py) || !isfinite(pz) || !isfinite(vx) || !isfinite(vy) || !isfinite(vz)) {
        flags |= 1;
    } else {
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (fabs(px) > maxPos || fabs(py) > maxPos || fabs(pz) > maxPos) flags |= 2;
        if (fabs(vx) > maxVel || fabs(vy) > maxVel || fabs(vz) > maxVel) flags |= 4;
    }
    if (flags != 0) atomicOr(failure, flags);
}

double computeTotalEnergy(const std::vector<Body>& bodies, const DeviceBodies& d) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    if (n > 1) {
        const int blocks = (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        double* dPartial = nullptr;
        CUDA_CHECK(cudaMalloc(&dPartial, n * sizeof(double)));
        potentialEnergyKernel<<<blocks, BLOCK_SIZE>>>(d.pos[0], d.pos[1], d.pos[2], dPartial, (int)n);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> partial(n);
        CUDA_CHECK(cudaMemcpy(partial.data(), dPartial, n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dPartial));
        for (size_t i = 0; i < n; ++i) {
            energy += partial[i];
        }
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const DeviceBodies& d, int n, int* dFailure) {
    CUDA_CHECK(cudaMemset(dFailure, 0, sizeof(int)));
    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    validateKernel<<<blocks, BLOCK_SIZE>>>(d.pos[0], d.pos[1], d.pos[2], d.vel[0], d.vel[1], d.vel[2], dFailure, n);
    CUDA_CHECK(cudaGetLastError());

    int flags = 0;
    CUDA_CHECK(cudaMemcpy(&flags, dFailure, sizeof(int), cudaMemcpyDeviceToHost));

    if (flags & 1) {
        printf("Validation failed: found NaN or Inf value in body state\n");
        return false;
    }
    if (flags & 2) {
        printf("Validation failed: body position exceeds reasonable bounds\n");
        return false;
    }
    if (flags & 4) {
        printf("Validation failed: body velocity exceeds reasonable bounds\n");
        return false;
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

    if (numBodies <= 0) {
        printf("Simulation time: 0 ms\n");
        if (printResults) {
            std::vector<double> empty;
            print_results(empty, "Bodies");
        }
        if (validate) {
            printf("Validating simulation results...\n");
            printf("Final energy: %.6f\n", 0.0);
            printf("Validation: PASSED\n");
        }
        return 0;
    }

    // Set up the device and warm up the runtime before timing.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    const size_t bytes = (size_t)numBodies * sizeof(double);
    DeviceBodies d;
    for (int c = 0; c < 3; ++c) {
        CUDA_CHECK(cudaMalloc(&d.pos[c], bytes));
        CUDA_CHECK(cudaMalloc(&d.vel[c], bytes));
    }
    int* dFailure = nullptr;
    CUDA_CHECK(cudaMalloc(&dFailure, sizeof(int)));

    // AoS -> SoA transfer of the initial state.
    {
        std::vector<double> host((size_t)numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            host[(size_t)0 * numBodies + i] = bodies[i].pos.x;
            host[(size_t)1 * numBodies + i] = bodies[i].pos.y;
            host[(size_t)2 * numBodies + i] = bodies[i].pos.z;
            host[(size_t)3 * numBodies + i] = bodies[i].vel.x;
            host[(size_t)4 * numBodies + i] = bodies[i].vel.y;
            host[(size_t)5 * numBodies + i] = bodies[i].vel.z;
        }
        for (int c = 0; c < 3; ++c) {
            CUDA_CHECK(cudaMemcpy(d.pos[c], host.data() + (size_t)c * numBodies, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d.vel[c], host.data() + (size_t)(c + 3) * numBodies, bytes, cudaMemcpyHostToDevice));
        }
    }

    int smCount = 1;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));

    const int targetBlocks = smCount * 8;
    const int blocks = (numBodies + TILE_SIZE - 1) / TILE_SIZE;
    const int integrateBlocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Split the interaction (j) loop across several blocks per body when the
    // body count alone yields too few blocks; the per-chunk partial forces are
    // summed by a second kernel in chunk order.
    int chunks = 1;
    int chunkTiles = blocks;
    if (blocks < targetBlocks) {
        const int wanted = (targetBlocks + blocks - 1) / blocks;
        chunkTiles = (blocks + wanted - 1) / wanted;   // tiles of j per chunk
        if (chunkTiles < 1) chunkTiles = 1;
        chunks = (blocks + chunkTiles - 1) / chunkTiles;
    }

    double* dPartialForce = nullptr;
    if (chunks > 1) {
        CUDA_CHECK(cudaMalloc(&dPartialForce, (size_t)chunks * 3 * bytes));
    }

    const dim3 forceGrid(blocks, chunks);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<forceGrid, TILE_SIZE>>>(d.pos[0], d.pos[1], d.pos[2], d.vel[0], d.vel[1], d.vel[2],
                                                      dPartialForce, numBodies, chunkTiles);
        if (chunks > 1) {
            applyForcesKernel<<<integrateBlocks, BLOCK_SIZE>>>(d.vel[0], d.vel[1], d.vel[2], dPartialForce, numBodies,
                                                               chunks);
        }
        integrateBodiesKernel<<<integrateBlocks, BLOCK_SIZE>>>(d.pos[0], d.pos[1], d.pos[2], d.vel[0], d.vel[1],
                                                               d.vel[2], numBodies);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Copy the final state back into the AoS layout.
    {
        std::vector<double> host((size_t)numBodies * 6);
        for (int c = 0; c < 3; ++c) {
            CUDA_CHECK(cudaMemcpy(host.data() + (size_t)c * numBodies, d.pos[c], bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(host.data() + (size_t)(c + 3) * numBodies, d.vel[c], bytes, cudaMemcpyDeviceToHost));
        }
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos.x = host[(size_t)0 * numBodies + i];
            bodies[i].pos.y = host[(size_t)1 * numBodies + i];
            bodies[i].pos.z = host[(size_t)2 * numBodies + i];
            bodies[i].vel.x = host[(size_t)3 * numBodies + i];
            bodies[i].vel.y = host[(size_t)4 * numBodies + i];
            bodies[i].vel.z = host[(size_t)5 * numBodies + i];
        }
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

    int exitCode = 0;

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(d, numBodies, dFailure)) {
            // Report final energy for reference.
            double finalEnergy = computeTotalEnergy(bodies, d);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    for (int c = 0; c < 3; ++c) {
        CUDA_CHECK(cudaFree(d.pos[c]));
        CUDA_CHECK(cudaFree(d.vel[c]));
    }
    CUDA_CHECK(cudaFree(dFailure));
    if (dPartialForce != nullptr) CUDA_CHECK(cudaFree(dPartialForce));

    return exitCode;
}
