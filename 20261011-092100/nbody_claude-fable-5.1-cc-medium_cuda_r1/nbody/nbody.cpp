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

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// ---------------------------------------------------------------------------
// CUDA helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        const cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                             \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

// Threads per block; also the size of the shared-memory tile of j-bodies.
constexpr int TILE = 256;
// Inner-loop unroll factor for full tiles.
constexpr int UNROLL = 8;

// All-pairs force kernel.
//
// Grid: x = block of TILE i-bodies, y = slice of the j-range ("split").  Each
// block accumulates the force on its i-bodies from the j-bodies in its slice
// and writes the partial sum to partial[split][i].  Splitting the j-range
// across blocks keeps the whole GPU busy for small N; the split partials are
// reduced in a fixed order by the integrate kernel, so results are
// deterministic run to run.
__global__ void __launch_bounds__(TILE)
forceKernel(const double4* __restrict__ pos, double3* __restrict__ partial, const int n, const int chunk,
            const int iBegin, const int iEnd) {
    __shared__ double4 tile[TILE];

    const int i = iBegin + blockIdx.x * TILE + threadIdx.x;
    const int jBegin = blockIdx.y * chunk;
    const int jEnd = min(jBegin + chunk, n);

    // Out-of-range threads still participate in tile loads / barriers.
    double4 p = make_double4(0.0, 0.0, 0.0, 0.0);
    if (i < iEnd) p = pos[i];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = jBegin; base < jEnd; base += TILE) {
        const int j = base + threadIdx.x;
        if (j < jEnd) tile[threadIdx.x] = pos[j];
        __syncthreads();

        const int cnt = min(TILE, jEnd - base);
        if (cnt == TILE) {
#pragma unroll UNROLL
            for (int k = 0; k < TILE; ++k) {
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
#pragma unroll 4
            for (int k = 0; k < cnt; ++k) {
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

    if (i < iEnd) partial[blockIdx.y * n + i] = make_double3(Fx, Fy, Fz);
}

// Reduce the per-split partial forces (fixed order), kick velocities, then
// drift positions into the other position buffer.  Positions are only written
// here, after every force has been computed from the old positions, matching
// the original computeForces() -> integrateBodies() ordering.
__global__ void integrateKernel(const double4* __restrict__ posIn, double4* __restrict__ posOut,
                                double3* __restrict__ vel, const double3* __restrict__ partial, const int n,
                                const int nSplit, const int iBegin, const int iEnd) {
    const int i = iBegin + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= iEnd) return;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int s = 0; s < nSplit; ++s) {
        const double3 f = partial[s * n + i];
        Fx += f.x;
        Fy += f.y;
        Fz += f.z;
    }

    double3 v = vel[i - iBegin];
    v.x += DT * Fx;
    v.y += DT * Fy;
    v.z += DT * Fz;
    vel[i - iBegin] = v;

    double4 p = posIn[i];
    p.x += v.x * DT;
    p.y += v.y * DT;
    p.z += v.z * DT;
    posOut[i] = p;
}

// ---------------------------------------------------------------------------
// GPU simulation driver (single or multiple GPUs)
//
// Each GPU owns a contiguous slice of the i-bodies.  Every GPU keeps a full
// copy of all positions (double buffered).  Per step, GPU g computes forces on
// its slice from the complete position array, integrates its slice into the
// other buffer, and pushes the updated slice into every other GPU's other
// buffer.  Events ensure every GPU has received all slices before it starts
// the next step.  The all-gather is O(N) per step and negligible against the
// O(N^2) force computation.
// ---------------------------------------------------------------------------

class GpuSimulation {
public:
    explicit GpuSimulation(const std::vector<Body>& bodies) : n_(static_cast<int>(bodies.size())) {
        if (n_ == 0) return;

        int devCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devCount));
        if (devCount < 1) {
            fprintf(stderr, "No CUDA device found\n");
            exit(EXIT_FAILURE);
        }

        // For small problems inter-GPU exchange latency outweighs the gain, so
        // only spread across GPUs once every GPU gets a reasonable amount of
        // work (at least ~8 tiles of i-bodies each).
        const int minBodiesPerGpu = 8 * TILE;
        int numGpus = std::min(devCount, std::max(1, n_ / minBodiesPerGpu));
        numGpus = std::max(1, numGpus);

        // Common j-split layout.  Derived from N and the SM count only, so the
        // summation order (and therefore the result) does not depend on how
        // many GPUs are used.  Chunks are a multiple of the warp size; for
        // small N they may be shorter than a tile so that enough blocks exist
        // to fill the machine.
        const int numTiles = (n_ + TILE - 1) / TILE;
        int smCount = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));
        const int targetBlocks = 8 * smCount;
        const int nSplitTarget = std::max(1, (targetBlocks + numTiles - 1) / numTiles);
        chunk_ = (n_ + nSplitTarget - 1) / nSplitTarget;
        chunk_ = std::max(32, ((chunk_ + 31) / 32) * 32);
        if (chunk_ > TILE) chunk_ = ((chunk_ + TILE - 1) / TILE) * TILE;  // whole tiles when large
        nSplit_ = (n_ + chunk_ - 1) / chunk_;
        const int iBlocksPerGpu = (numTiles + numGpus - 1) / numGpus;

        // Assign i-slices in whole tiles.
        gpus_.resize(numGpus);
        int tileCursor = 0;
        for (int g = 0; g < numGpus; ++g) {
            Gpu& G = gpus_[g];
            G.dev = g;
            const int tiles = std::min(iBlocksPerGpu, numTiles - tileCursor);
            G.iBegin = std::min(n_, tileCursor * TILE);
            G.iEnd = std::min(n_, (tileCursor + tiles) * TILE);
            G.numIBlocks = std::max(0, tiles);
            tileCursor += tiles;

            CUDA_CHECK(cudaSetDevice(G.dev));
            // Force the kernels to be loaded now (CUDA lazy loading would
            // otherwise charge the module load to the first timed step).
            cudaFuncAttributes attr{};
            CUDA_CHECK(cudaFuncGetAttributes(&attr, forceKernel));
            CUDA_CHECK(cudaFuncGetAttributes(&attr, integrateKernel));
            CUDA_CHECK(cudaStreamCreateWithFlags(&G.stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&G.done[0], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&G.done[1], cudaEventDisableTiming));
            CUDA_CHECK(cudaMalloc(&G.pos[0], sizeof(double4) * n_));
            CUDA_CHECK(cudaMalloc(&G.pos[1], sizeof(double4) * n_));
            const int local = std::max(1, G.iEnd - G.iBegin);
            CUDA_CHECK(cudaMalloc(&G.vel, sizeof(double3) * local));
            CUDA_CHECK(cudaMalloc(&G.partial, sizeof(double3) * static_cast<size_t>(n_) * nSplit_));
        }

        // Enable peer access where available (falls back to staging through
        // host memory otherwise; cudaMemcpyPeerAsync handles both).
        for (int g = 0; g < numGpus; ++g) {
            CUDA_CHECK(cudaSetDevice(gpus_[g].dev));
            for (int h = 0; h < numGpus; ++h) {
                if (h == g) continue;
                int can = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can, gpus_[g].dev, gpus_[h].dev));
                if (can) {
                    const cudaError_t e = cudaDeviceEnablePeerAccess(gpus_[h].dev, 0);
                    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
                    (void)cudaGetLastError();
                }
            }
        }

        upload(bodies);
    }

    ~GpuSimulation() {
        for (Gpu& G : gpus_) {
            cudaSetDevice(G.dev);
            cudaFree(G.pos[0]);
            cudaFree(G.pos[1]);
            cudaFree(G.vel);
            cudaFree(G.partial);
            cudaEventDestroy(G.done[0]);
            cudaEventDestroy(G.done[1]);
            cudaStreamDestroy(G.stream);
        }
    }

    GpuSimulation(const GpuSimulation&) = delete;
    GpuSimulation& operator=(const GpuSimulation&) = delete;

    void step() {
        if (n_ == 0) return;
        const int cur = cur_;
        const int nxt = cur ^ 1;
        const int numGpus = static_cast<int>(gpus_.size());

        for (int g = 0; g < numGpus; ++g) {
            Gpu& G = gpus_[g];
            CUDA_CHECK(cudaSetDevice(G.dev));

            // Wait until all slices from the previous step have landed in
            // pos[cur] on this GPU (and all previous-step readers of pos[nxt]
            // are finished before anyone writes into it).  The previous step
            // recorded its events under parity `cur`; this step records under
            // `nxt`, so no event is overwritten before everyone has waited on it.
            for (int h = 0; h < numGpus; ++h) {
                if (h != g) CUDA_CHECK(cudaStreamWaitEvent(G.stream, gpus_[h].done[cur], 0));
            }

            const int local = G.iEnd - G.iBegin;
            if (local > 0) {
                const dim3 grid(G.numIBlocks, nSplit_);
                forceKernel<<<grid, TILE, 0, G.stream>>>(G.pos[cur], G.partial, n_, chunk_, G.iBegin, G.iEnd);
                integrateKernel<<<G.numIBlocks, TILE, 0, G.stream>>>(G.pos[cur], G.pos[nxt], G.vel, G.partial,
                                                                       n_, nSplit_, G.iBegin, G.iEnd);
                // Broadcast the updated slice to the other GPUs' next buffer.
                for (int h = 0; h < numGpus; ++h) {
                    if (h == g) continue;
                    CUDA_CHECK(cudaMemcpyPeerAsync(gpus_[h].pos[nxt] + G.iBegin, gpus_[h].dev,
                                                   G.pos[nxt] + G.iBegin, G.dev,
                                                   sizeof(double4) * local, G.stream));
                }
            }
            CUDA_CHECK(cudaEventRecord(G.done[nxt], G.stream));
        }
        cur_ = nxt;
    }

    void synchronize() {
        for (Gpu& G : gpus_) {
            CUDA_CHECK(cudaSetDevice(G.dev));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(G.stream));
        }
    }

    void download(std::vector<Body>& bodies) {
        if (n_ == 0) return;
        std::vector<double4> hPos(n_);
        std::vector<double3> hVel(n_);
        for (Gpu& G : gpus_) {
            const int local = G.iEnd - G.iBegin;
            if (local <= 0) continue;
            CUDA_CHECK(cudaSetDevice(G.dev));
            CUDA_CHECK(cudaMemcpy(hPos.data() + G.iBegin, G.pos[cur_] + G.iBegin, sizeof(double4) * local,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hVel.data() + G.iBegin, G.vel, sizeof(double3) * local, cudaMemcpyDeviceToHost));
        }
        for (int i = 0; i < n_; ++i) {
            bodies[i].pos = Vec3(hPos[i].x, hPos[i].y, hPos[i].z);
            bodies[i].vel = Vec3(hVel[i].x, hVel[i].y, hVel[i].z);
        }
    }

private:
    struct Gpu {
        int dev = 0;
        int iBegin = 0, iEnd = 0;
        int numIBlocks = 0;
        cudaStream_t stream = nullptr;
        cudaEvent_t done[2] = {nullptr, nullptr};  // indexed by step parity
        double4* pos[2] = {nullptr, nullptr};  // full position arrays (double buffered)
        double3* vel = nullptr;                // velocities of the local slice only
        double3* partial = nullptr;            // [nSplit][n] partial forces (local slice used)
    };

    void upload(const std::vector<Body>& bodies) {
        std::vector<double4> hPos(n_);
        std::vector<double3> hVel(n_);
        for (int i = 0; i < n_; ++i) {
            hPos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
            hVel[i] = make_double3(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z);
        }
        for (Gpu& G : gpus_) {
            CUDA_CHECK(cudaSetDevice(G.dev));
            CUDA_CHECK(cudaMemcpy(G.pos[0], hPos.data(), sizeof(double4) * n_, cudaMemcpyHostToDevice));
            const int local = G.iEnd - G.iBegin;
            if (local > 0) {
                CUDA_CHECK(cudaMemcpy(G.vel, hVel.data() + G.iBegin, sizeof(double3) * local,
                                      cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaEventRecord(G.done[0], G.stream));
            CUDA_CHECK(cudaEventRecord(G.done[1], G.stream));
        }
    }

    int n_ = 0;
    int nSplit_ = 1;
    int chunk_ = 0;
    int cur_ = 0;
    std::vector<Gpu> gpus_;
};

// ---------------------------------------------------------------------------
// Host-side code (initialization, validation, I/O)
// ---------------------------------------------------------------------------

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

    // Create the CUDA context and upload initial state before timing so the
    // measured time covers the simulation steps themselves.
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
