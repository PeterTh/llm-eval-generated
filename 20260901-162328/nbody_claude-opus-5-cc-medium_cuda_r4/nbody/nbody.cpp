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

constexpr int WARP_SIZE = 32;

// Threads per block for the force kernel.
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

static_assert(BLOCK_SIZE % 32 == 0 || BLOCK_SIZE == 32, "BLOCK_SIZE must be a multiple of the warp size");

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                                    \
    do {                                                                                                    \
        const cudaError_t err__ = (call);                                                                   \
        if (err__ != cudaSuccess) {                                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,                      \
                    cudaGetErrorString(err__));                                                             \
            exit(EXIT_FAILURE);                                                                             \
        }                                                                                                   \
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

// Bodies live on the device in structure-of-arrays layout so that all global
// loads/stores are fully coalesced.
struct DeviceBodies {
    double* pos[3] = {nullptr, nullptr, nullptr};
    double* vel[3] = {nullptr, nullptr, nullptr};
};

// One thread per body i. Each thread keeps its own force accumulators in
// registers and walks all j in ascending order, so the summation order (and
// hence the floating-point result) matches the serial version exactly.
__global__ __launch_bounds__(BLOCK_SIZE) void computeForcesKernel(const double* __restrict__ px,
                                                                  const double* __restrict__ py,
                                                                  const double* __restrict__ pz, double* __restrict__ vx,
                                                                  double* __restrict__ vy, double* __restrict__ vz,
                                                                  const int n) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Threads past the last body mirror it: they compute a throwaway result but
    // keep the warp converged for the shuffle broadcasts below.
    const int iClamped = min(i, n - 1);
    const double xi = px[iClamped];
    const double yi = py[iClamped];
    const double zi = pz[iClamped];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    // The j-loop walks 0..n-1 in ascending order in warp-sized tiles. Each lane
    // loads one j-position and the tile is replayed via __shfl_sync, so no
    // shared memory or barriers are needed.
    const int fullTiles = (n / WARP_SIZE) * WARP_SIZE;

    for (int tile = 0; tile < n; tile += WARP_SIZE) {
        const int jIdx = min(tile + lane, n - 1);
        const double jx = px[jIdx];
        const double jy = py[jIdx];
        const double jz = pz[jIdx];

        const int lim = (tile < fullTiles) ? WARP_SIZE : (n - tile);

#pragma unroll 8
        for (int k = 0; k < lim; ++k) {
            const double dx = __shfl_sync(0xffffffffu, jx, k, WARP_SIZE) - xi;
            const double dy = __shfl_sync(0xffffffffu, jy, k, WARP_SIZE) - yi;
            const double dz = __shfl_sync(0xffffffffu, jz, k, WARP_SIZE) - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
    }

    if (i < n) {
        vx[i] += DT * Fx;
        vy[i] += DT * Fy;
        vz[i] += DT * Fz;
    }
}

__global__ void integrateBodiesKernel(double* __restrict__ px, double* __restrict__ py, double* __restrict__ pz,
                                      const double* __restrict__ vx, const double* __restrict__ vy,
                                      const double* __restrict__ vz, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
    }
}

// Per-body partial potential energy: body i accumulates the pairs (i, j) with
// j > i, in ascending j order, exactly as the serial reference does.
__global__ __launch_bounds__(BLOCK_SIZE) void potentialEnergyKernel(const double* __restrict__ px,
                                                                    const double* __restrict__ py,
                                                                    const double* __restrict__ pz,
                                                                    double* __restrict__ partial, const int n) {
    __shared__ double sx[BLOCK_SIZE];
    __shared__ double sy[BLOCK_SIZE];
    __shared__ double sz[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * BLOCK_SIZE + tid;
    const int iClamped = (i < n) ? i : (n - 1);
    const double xi = px[iClamped];
    const double yi = py[iClamped];
    const double zi = pz[iClamped];

    double e = 0.0;

    // Only tiles that can contain j > i need to be visited.
    const int firstTile = blockIdx.x * BLOCK_SIZE;
    for (int tile = firstTile; tile < n; tile += BLOCK_SIZE) {
        const int rem = min(BLOCK_SIZE, n - tile);
        if (tid < rem) {
            sx[tid] = px[tile + tid];
            sy[tid] = py[tile + tid];
            sz[tid] = pz[tile + tid];
        }
        __syncthreads();

        const int kStart = (tile <= i) ? (i - tile + 1) : 0;
        for (int k = kStart; k < rem; ++k) {
            const double dx = sx[k] - xi;
            const double dy = sy[k] - yi;
            const double dz = sz[k] - zi;
            const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            e -= 1.0 / dist;
        }
        __syncthreads();
    }

    if (i < n) {
        partial[i] = e;
    }
}

double computeTotalEnergy(const DeviceBodies& d, const std::vector<Body>& bodies, double* dPartial) {
    const size_t n = bodies.size();
    if (n == 0) {
        return 0.0;
    }

    const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    potentialEnergyKernel<<<blocks, BLOCK_SIZE>>>(d.pos[0], d.pos[1], d.pos[2], dPartial, static_cast<int>(n));
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> terms;
    terms.reserve(2 * n);

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        terms.push_back(0.5 * (body.vel.x * body.vel.x +
                              body.vel.y * body.vel.y +
                              body.vel.z * body.vel.z));
    }

    // Potential energy (assuming unit mass for all bodies)
    std::vector<double> partial(n);
    CUDA_CHECK(cudaMemcpy(partial.data(), dPartial, n * sizeof(double), cudaMemcpyDeviceToHost));
    terms.insert(terms.end(), partial.begin(), partial.end());

    return kahan_sum(terms);
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

    const size_t n = static_cast<size_t>(numBodies);
    const size_t bytes = n * sizeof(double);

    // Set up the device (context creation is not part of the measured region).
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));

    DeviceBodies d;
    for (int c = 0; c < 3; ++c) {
        CUDA_CHECK(cudaMalloc(&d.pos[c], bytes));
        CUDA_CHECK(cudaMalloc(&d.vel[c], bytes));
    }
    double* dPartial = nullptr;
    CUDA_CHECK(cudaMalloc(&dPartial, bytes));

    // Host-side SoA staging buffers (pinned for fast transfers).
    double* hSoA = nullptr;
    CUDA_CHECK(cudaMallocHost(&hSoA, 6 * bytes));
    for (size_t i = 0; i < n; ++i) {
        hSoA[0 * n + i] = bodies[i].pos.x;
        hSoA[1 * n + i] = bodies[i].pos.y;
        hSoA[2 * n + i] = bodies[i].pos.z;
        hSoA[3 * n + i] = bodies[i].vel.x;
        hSoA[4 * n + i] = bodies[i].vel.y;
        hSoA[5 * n + i] = bodies[i].vel.z;
    }

    const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int c = 0; c < 3; ++c) {
        CUDA_CHECK(cudaMemcpyAsync(d.pos[c], hSoA + c * n, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d.vel[c], hSoA + (3 + c) * n, bytes, cudaMemcpyHostToDevice));
    }

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<blocks, BLOCK_SIZE>>>(d.pos[0], d.pos[1], d.pos[2], d.vel[0], d.vel[1], d.vel[2],
                                                    numBodies);
        integrateBodiesKernel<<<blocks, BLOCK_SIZE>>>(d.pos[0], d.pos[1], d.pos[2], d.vel[0], d.vel[1], d.vel[2],
                                                      numBodies);
    }
    CUDA_CHECK(cudaGetLastError());

    for (int c = 0; c < 3; ++c) {
        CUDA_CHECK(cudaMemcpyAsync(hSoA + c * n, d.pos[c], bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpyAsync(hSoA + (3 + c) * n, d.vel[c], bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos.x = hSoA[0 * n + i];
        bodies[i].pos.y = hSoA[1 * n + i];
        bodies[i].pos.z = hSoA[2 * n + i];
        bodies[i].vel.x = hSoA[3 * n + i];
        bodies[i].vel.y = hSoA[4 * n + i];
        bodies[i].vel.z = hSoA[5 * n + i];
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
    int exitCode = 0;
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(d, bodies, dPartial);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    CUDA_CHECK(cudaFreeHost(hSoA));
    for (int c = 0; c < 3; ++c) {
        CUDA_CHECK(cudaFree(d.pos[c]));
        CUDA_CHECK(cudaFree(d.vel[c]));
    }
    CUDA_CHECK(cudaFree(dPartial));

    return exitCode;
}
