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
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

constexpr int TILE = 128;     // bodies (threads) per block along i
constexpr int MAX_SPLIT = 8;  // max j-range splits per body

// Each block handles TILE bodies i; `split` thread groups divide each j-tile.
// Block shape: (TILE, split). Updates velocity in place (positions are read-only here).
__global__ void forceKernel(const double4* __restrict__ pos, double4* __restrict__ vel, int n) {
    __shared__ double3 tile[TILE];
    __shared__ double3 red[MAX_SPLIT][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int split = blockDim.y;
    const int tid = ty * TILE + tx;
    const int nthreads = TILE * split;
    const int i = blockIdx.x * TILE + tx;

    double px = 0, py = 0, pz = 0;
    if (i < n) {
        const double4 p = pos[i];
        px = p.x; py = p.y; pz = p.z;
    }
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    const int chunk = TILE / split;

    for (int base = 0; base < n; base += TILE) {
        __syncthreads();
        for (int t = tid; t < TILE; t += nthreads) {
            const int j = base + t;
            if (j < n) {
                const double4 p = pos[j];
                tile[t] = make_double3(p.x, p.y, p.z);
            }
        }
        __syncthreads();
        int cnt = n - base;
        if (cnt > TILE) cnt = TILE;
        int lo = ty * chunk;
        int hi = lo + chunk;
        if (hi > cnt) hi = cnt;
#pragma unroll 4
        for (int k = lo; k < hi; ++k) {
            const double3 q = tile[k];
            const double dx = q.x - px;
            const double dy = q.y - py;
            const double dz = q.z - pz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
    }

    if (split > 1) {
        red[ty][tx] = make_double3(Fx, Fy, Fz);
        __syncthreads();
        if (ty != 0) return;
        for (int s = 1; s < split; ++s) {
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

// Per-body potential energy contribution (pairs j > i) and kinetic energy.
__global__ void energyKernel(const double4* __restrict__ pos, const double4* __restrict__ vel,
                             double* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double4 pi = pos[i];
    const double4 v = vel[i];
    double e = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    for (int j = i + 1; j < n; ++j) {
        const double4 q = pos[j];
        const double dx = q.x - pi.x;
        const double dy = q.y - pi.y;
        const double dz = q.z - pi.z;
        e -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
    }
    out[i] = e;
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
    
    // Upload state to the device
    const int n = numBodies;
    double4 *dPos = nullptr, *dVel = nullptr;
    double* dEnergy = nullptr;
    {
        std::vector<double4> hPos(n), hVel(n);
        for (int i = 0; i < n; ++i) {
            hPos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
            hVel[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
        }
        CUDA_CHECK(cudaMalloc(&dPos, sizeof(double4) * (n > 0 ? n : 1)));
        CUDA_CHECK(cudaMalloc(&dVel, sizeof(double4) * (n > 0 ? n : 1)));
        CUDA_CHECK(cudaMalloc(&dEnergy, sizeof(double) * (n > 0 ? n : 1)));
        if (n > 0) {
            CUDA_CHECK(cudaMemcpy(dPos, hPos.data(), sizeof(double4) * n, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dVel, hVel.data(), sizeof(double4) * n, cudaMemcpyHostToDevice));
        }
    }

    // Pick j-split so that small problems still fill the GPU
    int dev = 0, numSMs = 1;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
    const int blocks = (n + TILE - 1) / TILE;
    int split = 1;
    while (split < MAX_SPLIT && (long long)blocks * split < 4LL * numSMs) split *= 2;
    const dim3 fBlock(TILE, split);
    const int iBlock = 256;
    const int iGrid = (n + iBlock - 1) / iBlock;

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    if (n > 0) {
        for (int step = 0; step < numSteps; ++step) {
            forceKernel<<<blocks, fBlock>>>(dPos, dVel, n);
            integrateKernel<<<iGrid, iBlock>>>(dPos, dVel, n);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Download results
    {
        std::vector<double4> hPos(n), hVel(n);
        if (n > 0) {
            CUDA_CHECK(cudaMemcpy(hPos.data(), dPos, sizeof(double4) * n, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hVel.data(), dVel, sizeof(double4) * n, cudaMemcpyDeviceToHost));
        }
        for (int i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(hPos[i].x, hPos[i].y, hPos[i].z);
            bodies[i].vel = Vec3(hVel[i].x, hVel[i].y, hVel[i].z);
        }
    }

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
            energyKernel<<<iGrid, iBlock>>>(dPos, dVel, dEnergy, n);
            CUDA_CHECK(cudaGetLastError());
            std::vector<double> hE(n);
            CUDA_CHECK(cudaMemcpy(hE.data(), dEnergy, sizeof(double) * n, cudaMemcpyDeviceToHost));
            double finalEnergy = 0.0;
            for (int i = 0; i < n; ++i) finalEnergy += hE[i];
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            cudaFree(dPos); cudaFree(dVel); cudaFree(dEnergy);
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
