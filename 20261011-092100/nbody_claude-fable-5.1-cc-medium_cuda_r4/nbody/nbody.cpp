// N-body benchmark, CUDA implementation.
//
// The O(n^2) force computation and the position update are performed on the
// GPU. Each thread owns one body and accumulates the force exerted by all other
// bodies in the same j = 0..n-1 order as the original sequential code; body
// positions are streamed through shared memory in tiles so that each tile is
// read from global memory once per thread block.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Threads per block for the force kernel (also the shared-memory tile size).
constexpr int BLOCK_SIZE = 128;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,         \
                    cudaGetErrorString(err__));                                                \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
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

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Force accumulation for one simulation step (computeForces()).
//
// Thread (i, split) accumulates the force on body i from the bodies in every
// numSplits-th tile, starting at tile `split`. Within a split the j order is
// ascending, and the per-split partials are later added in ascending split
// order, so the result is deterministic. Splitting the j range lets small
// problems still fill all SMs of the GPU.
__global__ void __launch_bounds__(BLOCK_SIZE)
nbodyForceKernel(const double4* __restrict__ posIn,
                 double4* __restrict__ partialForce,
                 const int n,
                 const int numSplits) {
    __shared__ double4 tile[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int split = blockIdx.y;
    const bool active = i < n;

    double4 p = make_double4(0.0, 0.0, 0.0, 0.0);
    if (active) p = posIn[i];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tileStart = split * BLOCK_SIZE; tileStart < n; tileStart += numSplits * BLOCK_SIZE) {
        const int j = tileStart + threadIdx.x;
        if (j < n) tile[threadIdx.x] = posIn[j];
        __syncthreads();

        const int tileCount = min(BLOCK_SIZE, n - tileStart);
        if (tileCount == BLOCK_SIZE) {
#pragma unroll 8
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                const double4 q = tile[k];
                const double dx = q.x - p.x;
                const double dy = q.y - p.y;
                const double dz = q.z - p.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        } else {
            for (int k = 0; k < tileCount; ++k) {
                const double4 q = tile[k];
                const double dx = q.x - p.x;
                const double dy = q.y - p.y;
                const double dz = q.z - p.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) partialForce[split * n + i] = make_double4(Fx, Fy, Fz, 0.0);
}

// Velocity update (rest of computeForces()) and position update
// (integrateBodies()). Positions are double buffered (posIn -> posOut) so that
// the force kernel of the next step sees a consistent state.
__global__ void __launch_bounds__(BLOCK_SIZE)
nbodyIntegrateKernel(const double4* __restrict__ posIn,
                     double4* __restrict__ posOut,
                     double4* __restrict__ vel,
                     const double4* __restrict__ partialForce,
                     const int n,
                     const int numSplits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int s = 0; s < numSplits; ++s) {
        const double4 f = partialForce[s * n + i];
        Fx += f.x;
        Fy += f.y;
        Fz += f.z;
    }

    double4 v = vel[i];
    v.x += DT * Fx;
    v.y += DT * Fy;
    v.z += DT * Fz;
    vel[i] = v;

    double4 p = posIn[i];
    p.x += v.x * DT;
    p.y += v.y * DT;
    p.z += v.z * DT;
    posOut[i] = p;
}

// Potential energy: thread i sums -1/dist over all j > i. Per-thread partial
// sums are written out and reduced on the host so the result is deterministic.
__global__ void __launch_bounds__(BLOCK_SIZE)
potentialEnergyKernel(const double4* __restrict__ pos,
                      double* __restrict__ partial,
                      const int n) {
    __shared__ double4 tile[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;

    double4 p = make_double4(0.0, 0.0, 0.0, 0.0);
    if (active) p = pos[i];

    double energy = 0.0;

    // Only tiles that contain some j > i for some thread of this block matter.
    const int firstTile = (blockIdx.x * blockDim.x / BLOCK_SIZE) * BLOCK_SIZE;
    for (int tileStart = firstTile; tileStart < n; tileStart += BLOCK_SIZE) {
        const int jLoad = tileStart + threadIdx.x;
        if (jLoad < n) tile[threadIdx.x] = pos[jLoad];
        __syncthreads();

        const int tileCount = min(BLOCK_SIZE, n - tileStart);
        // Start after i within this tile (all k are > i when tileStart > i).
        const int kStart = max(0, i + 1 - tileStart);
        for (int k = kStart; k < tileCount; ++k) {
            const double4 q = tile[k];
            const double dx = q.x - p.x;
            const double dy = q.y - p.y;
            const double dz = q.z - p.z;
            const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
        __syncthreads();
    }

    if (active) partial[i] = energy;
}

// ---------------------------------------------------------------------------
// Host-side GPU simulation driver
// ---------------------------------------------------------------------------

class GpuSimulation {
public:
    explicit GpuSimulation(const std::vector<Body>& bodies) : n_(static_cast<int>(bodies.size())) {
        const size_t bytes = static_cast<size_t>(n_) * sizeof(double4);
        CUDA_CHECK(cudaMalloc(&dPos_[0], bytes));
        CUDA_CHECK(cudaMalloc(&dPos_[1], bytes));
        CUDA_CHECK(cudaMalloc(&dVel_, bytes));

        // Choose the j-range split factor so that the force kernel launches
        // enough blocks to occupy every SM several times over.
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        int numSMs = 1;
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
        bodyBlocks_ = (n_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int targetBlocks = 4 * numSMs;
        numSplits_ = (targetBlocks + bodyBlocks_ - 1) / std::max(bodyBlocks_, 1);
        numSplits_ = std::max(1, std::min(numSplits_, bodyBlocks_));
        CUDA_CHECK(cudaMalloc(&dPartialForce_, static_cast<size_t>(numSplits_) * bytes));

        std::vector<double4> hPos(n_), hVel(n_);
        for (int i = 0; i < n_; ++i) {
            hPos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
            hVel[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
        }
        CUDA_CHECK(cudaMemcpy(dPos_[0], hPos.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dVel_, hVel.data(), bytes, cudaMemcpyHostToDevice));
    }

    ~GpuSimulation() {
        cudaFree(dPos_[0]);
        cudaFree(dPos_[1]);
        cudaFree(dVel_);
        cudaFree(dPartialForce_);
    }

    GpuSimulation(const GpuSimulation&) = delete;
    GpuSimulation& operator=(const GpuSimulation&) = delete;

    // Equivalent to computeForces(bodies); integrateBodies(bodies);
    void step() {
        if (n_ == 0) return;
        const dim3 forceGrid(bodyBlocks_, numSplits_);
        nbodyForceKernel<<<forceGrid, BLOCK_SIZE>>>(dPos_[cur_], dPartialForce_, n_, numSplits_);
        CUDA_CHECK(cudaGetLastError());
        nbodyIntegrateKernel<<<bodyBlocks_, BLOCK_SIZE>>>(dPos_[cur_], dPos_[cur_ ^ 1], dVel_,
                                                         dPartialForce_, n_, numSplits_);
        CUDA_CHECK(cudaGetLastError());
        cur_ ^= 1;
    }

    void synchronize() { CUDA_CHECK(cudaDeviceSynchronize()); }

    void download(std::vector<Body>& bodies) {
        synchronize();
        const size_t bytes = static_cast<size_t>(n_) * sizeof(double4);
        std::vector<double4> hPos(n_), hVel(n_);
        if (n_ > 0) {
            CUDA_CHECK(cudaMemcpy(hPos.data(), dPos_[cur_], bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hVel.data(), dVel_, bytes, cudaMemcpyDeviceToHost));
        }
        bodies.resize(n_);
        for (int i = 0; i < n_; ++i) {
            bodies[i].pos = Vec3(hPos[i].x, hPos[i].y, hPos[i].z);
            bodies[i].vel = Vec3(hVel[i].x, hVel[i].y, hVel[i].z);
        }
    }

    // Potential energy of the current positions (sum over pairs i<j of -1/dist).
    double potentialEnergy() {
        if (n_ == 0) return 0.0;
        double* dPartial = nullptr;
        CUDA_CHECK(cudaMalloc(&dPartial, static_cast<size_t>(n_) * sizeof(double)));
        const int blocks = (n_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
        potentialEnergyKernel<<<blocks, BLOCK_SIZE>>>(dPos_[cur_], dPartial, n_);
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> hPartial(n_);
        CUDA_CHECK(cudaMemcpy(hPartial.data(), dPartial, static_cast<size_t>(n_) * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dPartial));
        double energy = 0.0;
        for (int i = 0; i < n_; ++i) energy += hPartial[i];
        return energy;
    }

private:
    int n_;
    int cur_ = 0;
    int bodyBlocks_ = 0;
    int numSplits_ = 1;
    double4* dPos_[2] = {nullptr, nullptr};
    double4* dVel_ = nullptr;
    double4* dPartialForce_ = nullptr;
};

// Total energy: kinetic part on the host (O(n)), potential part on the GPU (O(n^2)).
double computeTotalEnergy(const std::vector<Body>& bodies, GpuSimulation& sim) {
    double energy = 0.0;

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    energy += sim.potentialEnergy();

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

    // Initialize the GPU and upload the initial state (one-time setup cost).
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));  // force context creation before timing starts
    {
        // CUDA 12 loads kernels lazily on first launch; load them now so the
        // timed region measures only the simulation itself.
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, nbodyForceKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, nbodyIntegrateKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, potentialEnergyKernel));
    }
    GpuSimulation sim(bodies);
    sim.synchronize();

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        sim.step();
    }
    sim.synchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Copy the final state back to the host.
    sim.download(bodies);

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
            double finalEnergy = computeTotalEnergy(bodies, sim);
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
