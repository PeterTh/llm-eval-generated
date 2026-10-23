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

// ---------------------------------------------------------------------------
// CUDA kernel configuration
// ---------------------------------------------------------------------------
// One thread per body gives the best SM occupancy for this FP64, latency-bound
// force calculation: many small blocks spread evenly over the GPU and each
// thread still exposes three independent accumulation chains.
constexpr int kThreads = 32;               // threads (one body each) per block
constexpr int kJTileSmall = 256;           // interaction partners staged per tile
constexpr int kJTileLarge = 512;           // larger tiles amortize staging for big N
constexpr int kLargeBodyThreshold = 32768; // switch to larger tiles above this N

// Match the floating-point contraction of the host reference build. On targets
// with hardware FMA the compiler fuses these expressions; elsewhere it keeps
// the individual rounding steps. Device-side automatic contraction is disabled
// in CMake so the explicit choice below is the only one that takes effect.
#if defined(__FP_FAST_FMA)
#define NBODY_FMA(a, b, c) __fma_rn((a), (b), (c))
#else
#define NBODY_FMA(a, b, c) ((a) * (b) + (c))
#endif

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

// ---------------------------------------------------------------------------
// Device helpers / kernels
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        const cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err__));                                          \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

// One block owns kThreads bodies. The interaction partner positions are staged
// through shared memory one tile at a time, so global traffic is O(n^2 / tile)
// while each thread keeps a single, ordered force accumulation chain. The
// explicit fused multiply-adds mirror the contraction performed by the host
// reference build, keeping the floating-point results bit-identical.
template <int JTile>
__global__ __launch_bounds__(kThreads)
void computeForcesKernel(const double* __restrict__ posX,
                         const double* __restrict__ posY,
                         const double* __restrict__ posZ,
                         double* __restrict__ velX,
                         double* __restrict__ velY,
                         double* __restrict__ velZ,
                         const int n) {
    __shared__ double sX[JTile];
    __shared__ double sY[JTile];
    __shared__ double sZ[JTile];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * kThreads + tid;
    const bool valid = i < n;

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (valid) {
        xi = posX[i];
        yi = posY[i];
        zi = posZ[i];
    }

    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int jBase = 0; jBase < n; jBase += JTile) {
        const int remaining = n - jBase;
        const int jCount = JTile < remaining ? JTile : remaining;

        for (int t = tid; t < JTile; t += kThreads) {
            const int j = jBase + t;
            if (j < n) {
                sX[t] = posX[j];
                sY[t] = posY[j];
                sZ[t] = posZ[j];
            } else {
                sX[t] = 0.0;
                sY[t] = 0.0;
                sZ[t] = 0.0;
            }
        }
        __syncthreads();

        #pragma unroll 4
        for (int jj = 0; jj < jCount; ++jj) {
            const double dx = sX[jj] - xi;
            const double dy = sY[jj] - yi;
            const double dz = sZ[jj] - zi;
            const double distSqr = NBODY_FMA(dz, dz, NBODY_FMA(dx, dx, dy * dy)) + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            fx = NBODY_FMA(dx, invDist3, fx);
            fy = NBODY_FMA(dy, invDist3, fy);
            fz = NBODY_FMA(dz, invDist3, fz);
        }
        __syncthreads();
    }

    if (valid) {
        velX[i] = velX[i] + DT * fx;
        velY[i] = velY[i] + DT * fy;
        velZ[i] = velZ[i] + DT * fz;
    }
}

__global__ void integrateKernel(double* __restrict__ posX,
                                double* __restrict__ posY,
                                double* __restrict__ posZ,
                                const double* __restrict__ velX,
                                const double* __restrict__ velY,
                                const double* __restrict__ velZ,
                                const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        posX[i] = NBODY_FMA(velX[i], DT, posX[i]);
        posY[i] = NBODY_FMA(velY[i], DT, posY[i]);
        posZ[i] = NBODY_FMA(velZ[i], DT, posZ[i]);
    }
}

// ---------------------------------------------------------------------------
// Host-side GPU state
// ---------------------------------------------------------------------------
class NBodyGpu {
  public:
    explicit NBodyGpu(const std::vector<Body>& bodies) : n_(static_cast<int>(bodies.size())) {
        CUDA_CHECK(cudaMalloc(&posX_, sizeof(double) * n_));
        CUDA_CHECK(cudaMalloc(&posY_, sizeof(double) * n_));
        CUDA_CHECK(cudaMalloc(&posZ_, sizeof(double) * n_));
        CUDA_CHECK(cudaMalloc(&velX_, sizeof(double) * n_));
        CUDA_CHECK(cudaMalloc(&velY_, sizeof(double) * n_));
        CUDA_CHECK(cudaMalloc(&velZ_, sizeof(double) * n_));

        std::vector<double> hx(n_), hy(n_), hz(n_), ux(n_), uy(n_), uz(n_);
        for (int i = 0; i < n_; ++i) {
            hx[i] = bodies[i].pos.x;
            hy[i] = bodies[i].pos.y;
            hz[i] = bodies[i].pos.z;
            ux[i] = bodies[i].vel.x;
            uy[i] = bodies[i].vel.y;
            uz[i] = bodies[i].vel.z;
        }
        CUDA_CHECK(cudaMemcpy(posX_, hx.data(), sizeof(double) * n_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(posY_, hy.data(), sizeof(double) * n_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(posZ_, hz.data(), sizeof(double) * n_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(velX_, ux.data(), sizeof(double) * n_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(velY_, uy.data(), sizeof(double) * n_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(velZ_, uz.data(), sizeof(double) * n_, cudaMemcpyHostToDevice));
    }

    ~NBodyGpu() {
        cudaFree(posX_);
        cudaFree(posY_);
        cudaFree(posZ_);
        cudaFree(velX_);
        cudaFree(velY_);
        cudaFree(velZ_);
    }

    NBodyGpu(const NBodyGpu&) = delete;
    NBodyGpu& operator=(const NBodyGpu&) = delete;

    void step() {
        if (n_ <= 0) {
            return;
        }
        const int grid = (n_ + kThreads - 1) / kThreads;
        if (n_ <= kLargeBodyThreshold) {
            computeForcesKernel<kJTileSmall>
                <<<grid, kThreads>>>(posX_, posY_, posZ_, velX_, velY_, velZ_, n_);
        } else {
            computeForcesKernel<kJTileLarge>
                <<<grid, kThreads>>>(posX_, posY_, posZ_, velX_, velY_, velZ_, n_);
        }

        const int integrateGrid = (n_ + kThreads - 1) / kThreads;
        integrateKernel<<<integrateGrid, kThreads>>>(posX_, posY_, posZ_, velX_, velY_, velZ_, n_);
        CUDA_CHECK(cudaGetLastError());
    }

    void synchronize() const {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void download(std::vector<Body>& bodies) const {
        bodies.resize(n_);
        if (n_ == 0) {
            return;
        }
        std::vector<double> hx(n_), hy(n_), hz(n_), ux(n_), uy(n_), uz(n_);
        CUDA_CHECK(cudaMemcpy(hx.data(), posX_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), posY_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hz.data(), posZ_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ux.data(), velX_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uy.data(), velY_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uz.data(), velZ_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n_; ++i) {
            bodies[i].pos.x = hx[i];
            bodies[i].pos.y = hy[i];
            bodies[i].pos.z = hz[i];
            bodies[i].vel.x = ux[i];
            bodies[i].vel.y = uy[i];
            bodies[i].vel.z = uz[i];
        }
    }

  private:
    int n_;
    double* posX_ = nullptr;
    double* posY_ = nullptr;
    double* posZ_ = nullptr;
    double* velX_ = nullptr;
    double* velY_ = nullptr;
    double* velZ_ = nullptr;
};

// ---------------------------------------------------------------------------
// Host-side reference routines (validation only)
// ---------------------------------------------------------------------------
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
    std::vector<Body> bodies(numBodies > 0 ? numBodies : 0);
    randomizeBodies(bodies);

    // Upload initial state and run simulation entirely on the GPU.
    NBodyGpu gpu(bodies);

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        gpu.step();
    }

    gpu.synchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Print results for external validation
    if (printResults) {
        gpu.download(bodies);
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(bodies.size() * 6);
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

        gpu.download(bodies);
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
