#include <algorithm>
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

// CUDA execution configuration
constexpr int TILE = 128;     // threads per block == shared-memory tile of bodies
constexpr int BLOCK_1D = 256; // threads per block for the elementwise/reduction kernels

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        const cudaError_t err_ = (call);                                                                               \
        if (err_ != cudaSuccess) {                                                                                     \
            printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__, __LINE__,                         \
                   cudaGetErrorString(err_));                                                                          \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

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

// Structure-of-arrays body state living in device memory.
struct DeviceBodies {
    double* px = nullptr;
    double* py = nullptr;
    double* pz = nullptr;
    double* vx = nullptr;
    double* vy = nullptr;
    double* vz = nullptr;
};

// One thread computes the total force acting on one body. Body positions are streamed through shared
// memory in tiles so that each position is read from global memory only once per block, which removes
// almost all global-memory traffic from the O(n^2) inner loop.
//
// The inner loop visits j in ascending order, i.e. forces are accumulated in exactly the same order as
// in the sequential reference implementation.
//
// SPLIT == false: every block covers the whole j range and directly updates the velocities.
// SPLIT == true:  the j range is additionally partitioned over blockIdx.y (needed to fill the GPU when
//                 there are fewer bodies than desired threads); each block stores the partial force of
//                 its j chunk, which reduceForcesKernel then sums up in ascending chunk order.
template <bool SPLIT>
__global__ __launch_bounds__(TILE) void computeForcesKernel(const double* __restrict__ px,
                                                            const double* __restrict__ py,
                                                            const double* __restrict__ pz, double* __restrict__ vx,
                                                            double* __restrict__ vy, double* __restrict__ vz,
                                                            const int n, const int chunkTiles) {
    __shared__ double sx[TILE];
    __shared__ double sy[TILE];
    __shared__ double sz[TILE];

    const int tid = static_cast<int>(threadIdx.x);
    const int i = static_cast<int>(blockIdx.x) * TILE + tid;

    const int jBegin = SPLIT ? static_cast<int>(blockIdx.y) * chunkTiles * TILE : 0;
    const int jEnd = SPLIT ? min(n, jBegin + chunkTiles * TILE) : n;

    const bool active = i < n;
    const double xi = active ? px[i] : 0.0;
    const double yi = active ? py[i] : 0.0;
    const double zi = active ? pz[i] : 0.0;

    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int tileStart = jBegin; tileStart < jEnd; tileStart += TILE) {
        const int count = min(TILE, jEnd - tileStart);

        __syncthreads();
        if (tid < count) {
            sx[tid] = px[tileStart + tid];
            sy[tid] = py[tileStart + tid];
            sz[tid] = pz[tileStart + tid];
        }
        __syncthreads();

#pragma unroll 4
        for (int jj = 0; jj < count; ++jj) {
            const double dx = sx[jj] - xi;
            const double dy = sy[jj] - yi;
            const double dz = sz[jj] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            fx += dx * invDist3;
            fy += dy * invDist3;
            fz += dz * invDist3;
        }
    }

    if (active) {
        if (SPLIT) {
            // vx points at the chunk-partial buffer, laid out as [chunk][component][body]
            const size_t off = (static_cast<size_t>(blockIdx.y) * 3) * static_cast<size_t>(n) + i;
            vx[off] = fx;
            vx[off + n] = fy;
            vx[off + 2 * n] = fz;
        } else {
            vx[i] += DT * fx;
            vy[i] += DT * fy;
            vz[i] += DT * fz;
        }
    }
}

// Sums the per-chunk partial forces in ascending chunk (i.e. ascending j) order and advances the
// velocities.
__global__ void reduceForcesKernel(const double* __restrict__ partial, double* __restrict__ vx,
                                  double* __restrict__ vy, double* __restrict__ vz, const int n,
                                  const int numChunks) {
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (i >= n) {
        return;
    }

    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int c = 0; c < numChunks; ++c) {
        const size_t off = (static_cast<size_t>(c) * 3) * static_cast<size_t>(n) + i;
        fx += partial[off];
        fy += partial[off + n];
        fz += partial[off + 2 * n];
    }

    vx[i] += DT * fx;
    vy[i] += DT * fy;
    vz[i] += DT * fz;
}

__global__ void integrateBodiesKernel(double* __restrict__ px, double* __restrict__ py, double* __restrict__ pz,
                                      const double* __restrict__ vx, const double* __restrict__ vy,
                                      const double* __restrict__ vz, const int n) {
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (i < n) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
    }
}

// Potential energy contribution of body row i (all j > i), summed in ascending j order. Row i has
// n-1-i terms, so each thread handles the short row n-1-i together with the long row i to keep the
// work per thread constant.
__global__ __launch_bounds__(BLOCK_1D) void potentialEnergyKernel(const double* __restrict__ px,
                                                                      const double* __restrict__ py,
                                                                      const double* __restrict__ pz,
                                                                      double* __restrict__ partial, const int n) {
    const int row = static_cast<int>(blockIdx.x) * BLOCK_1D + static_cast<int>(threadIdx.x);
    if (row >= (n + 1) / 2) {
        return;
    }

    for (int i = row, k = 0; k < 2; ++k, i = n - 1 - row) {
        if (k == 1 && i == row) {
            break; // odd n: the middle row is handled only once
        }

        const double xi = px[i];
        const double yi = py[i];
        const double zi = pz[i];

        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            const double dx = px[j] - xi;
            const double dy = py[j] - yi;
            const double dz = pz[j] - zi;
            const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            sum += 1.0 / dist;
        }
        partial[i] = sum;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies, const DeviceBodies& dev) {
    double energy = 0.0;
    const int n = static_cast<int>(bodies.size());

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    if (n == 0) {
        return energy;
    }

    // Potential energy (assuming unit mass for all bodies), row sums computed on the GPU
    double* dPartial = nullptr;
    CUDA_CHECK(cudaMalloc(&dPartial, static_cast<size_t>(n) * sizeof(double)));
    const int energyRows = (n + 1) / 2;
    potentialEnergyKernel<<<(energyRows + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D>>>(dev.px, dev.py, dev.pz,
                                                                                            dPartial, n);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> partial(n);
    CUDA_CHECK(cudaMemcpy(partial.data(), dPartial, static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dPartial));

    for (int i = 0; i < n; ++i) {
        energy -= partial[i];
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

// Launch configuration for the force computation: one thread per body, plus - if that alone does not
// produce enough blocks to fill the GPU - a partitioning of the j loop over a second grid dimension.
struct ForcePlan {
    int iBlocks = 0;
    int numChunks = 1; // number of j chunks; > 1 selects the split kernel
    int chunkTiles = 0;
};

static ForcePlan makeForcePlan(const int n, const int numSMs) {
    ForcePlan plan;
    if (n <= 0) {
        return plan;
    }

    const int totalTiles = (n + TILE - 1) / TILE;
    plan.iBlocks = totalTiles;

    // Aim for a few blocks per SM so that the FP64 pipelines stay saturated.
    const int desiredBlocks = 4 * numSMs;
    if (plan.iBlocks < desiredBlocks) {
        const int wantedChunks = (desiredBlocks + plan.iBlocks - 1) / plan.iBlocks;
        plan.chunkTiles = std::max(1, (totalTiles + wantedChunks - 1) / wantedChunks);
        plan.numChunks = (totalTiles + plan.chunkTiles - 1) / plan.chunkTiles;
    }
    return plan;
}

static void launchComputeForces(const ForcePlan& plan, const DeviceBodies& dev, double* partial, const int n) {
    if (plan.numChunks > 1) {
        const dim3 grid(plan.iBlocks, plan.numChunks);
        computeForcesKernel<true><<<grid, TILE>>>(dev.px, dev.py, dev.pz, partial, nullptr, nullptr, n,
                                                  plan.chunkTiles);
        reduceForcesKernel<<<(n + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D>>>(partial, dev.vx, dev.vy, dev.vz, n,
                                                                                    plan.numChunks);
    } else {
        computeForcesKernel<false><<<plan.iBlocks, TILE>>>(dev.px, dev.py, dev.pz, dev.vx, dev.vy, dev.vz, n, 0);
    }
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

    // Set up the GPU (device selection and context creation are one-off setup costs)
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaGetDevice(&device));
    int numSMs = 1;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaFree(nullptr));

    // Empty launches of every kernel: this pays the one-off module loading cost up front instead of
    // inside the timed region (the kernels return immediately for n == 0).
    computeForcesKernel<false><<<1, TILE>>>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0);
    computeForcesKernel<true><<<dim3(1, 1), TILE>>>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 1);
    reduceForcesKernel<<<1, BLOCK_1D>>>(nullptr, nullptr, nullptr, nullptr, 0, 0);
    integrateBodiesKernel<<<1, BLOCK_1D>>>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
    potentialEnergyKernel<<<1, BLOCK_1D>>>(nullptr, nullptr, nullptr, nullptr, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Host-side structure-of-arrays staging buffers (pinned for fast transfers)
    const size_t bytes = static_cast<size_t>(numBodies) * sizeof(double);
    double* host = nullptr;
    double* dPartial = nullptr;
    DeviceBodies dev;
    const ForcePlan plan = makeForcePlan(numBodies, numSMs);
    if (numBodies > 0) {
        CUDA_CHECK(cudaHostAlloc(&host, 6 * bytes, cudaHostAllocDefault));
        double* devMem = nullptr;
        CUDA_CHECK(cudaMalloc(&devMem, 6 * bytes));
        if (plan.numChunks > 1) {
            CUDA_CHECK(cudaMalloc(&dPartial, static_cast<size_t>(plan.numChunks) * 3 * bytes));
        }
        dev.px = devMem;
        dev.py = devMem + numBodies;
        dev.pz = devMem + 2 * numBodies;
        dev.vx = devMem + 3 * numBodies;
        dev.vy = devMem + 4 * numBodies;
        dev.vz = devMem + 5 * numBodies;

        for (int i = 0; i < numBodies; ++i) {
            host[i] = bodies[i].pos.x;
            host[numBodies + i] = bodies[i].pos.y;
            host[2 * numBodies + i] = bodies[i].pos.z;
            host[3 * numBodies + i] = bodies[i].vel.x;
            host[4 * numBodies + i] = bodies[i].vel.y;
            host[5 * numBodies + i] = bodies[i].vel.z;
        }
    }

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    if (numBodies > 0) {
        CUDA_CHECK(cudaMemcpy(dev.px, host, 6 * bytes, cudaMemcpyHostToDevice));

        const int integrateBlocks = (numBodies + BLOCK_1D - 1) / BLOCK_1D;
        for (int step = 0; step < numSteps; ++step) {
            launchComputeForces(plan, dev, dPartial, numBodies);
            integrateBodiesKernel<<<integrateBlocks, BLOCK_1D>>>(dev.px, dev.py, dev.pz, dev.vx, dev.vy, dev.vz,
                                                                     numBodies);
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(host, dev.px, 6 * bytes, cudaMemcpyDeviceToHost));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = host[i];
        bodies[i].pos.y = host[numBodies + i];
        bodies[i].pos.z = host[2 * numBodies + i];
        bodies[i].vel.x = host[3 * numBodies + i];
        bodies[i].vel.y = host[4 * numBodies + i];
        bodies[i].vel.z = host[5 * numBodies + i];
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
            double finalEnergy = computeTotalEnergy(bodies, dev);
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
