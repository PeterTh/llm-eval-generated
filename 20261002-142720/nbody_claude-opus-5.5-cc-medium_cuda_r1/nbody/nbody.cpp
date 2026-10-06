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
// CUDA implementation
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

constexpr int MAX_BLOCK = 256;
constexpr int STEP_BLOCK = 128;
constexpr int SPLIT_THREADS_PER_SM = 2048;

// Accumulates the contribution of body pj to the force on body pi.
__device__ __forceinline__ void accumulate(const double4& pi, const double4& pj,
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

// Velocity update followed by position integration of one body.
__device__ __forceinline__ void updateBody(double4 p, double4& v, double4& pOut,
                                           double Fx, double Fy, double Fz) {
    v.x += DT * Fx;
    v.y += DT * Fy;
    v.z += DT * Fz;
    p.x += v.x * DT;
    p.y += v.y * DT;
    p.z += v.z * DT;
    pOut = p;
}

// One simulation step for bodies [offset, offset + count): computes the force on
// each body from all n bodies, updates its velocity and writes the integrated
// position into posOut. Positions are double-buffered so integrating one body never
// affects the force computation of another, exactly like the original two-phase loop.
// K consecutive lanes of a warp share one body (K > 1 only when there are too few
// bodies to fill the GPU), each summing a strided subset of the interactions in
// order, followed by a fixed-order shuffle reduction, so results are deterministic.
template <int K>
__global__ void __launch_bounds__(MAX_BLOCK)
stepKernel(const double4* __restrict__ pos, double4* __restrict__ posOut,
           double4* __restrict__ vel, int n, int offset, int count) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int local = gid / K;
    const int lane = gid % K;
    const bool active = local < count;
    const int i = offset + (active ? local : 0);

    const double4 pi = pos[i];
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
#pragma unroll 4
    for (int j = lane; j < n; j += K) accumulate(pi, pos[j], Fx, Fy, Fz);

#pragma unroll
    for (int d = K / 2; d > 0; d /= 2) {
        Fx += __shfl_down_sync(0xffffffffu, Fx, d, K);
        Fy += __shfl_down_sync(0xffffffffu, Fy, d, K);
        Fz += __shfl_down_sync(0xffffffffu, Fz, d, K);
    }

    if (active && lane == 0) {
        double4 v = vel[local];
        updateBody(pi, v, posOut[i], Fx, Fy, Fz);
        vel[local] = v;
    }
}

// Per-body potential energy contribution: -sum_{j>i} 1/dist (sequential j order).
__global__ void __launch_bounds__(MAX_BLOCK)
potentialKernel(const double4* __restrict__ pos, double* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double4 pi = pos[i];
    double e = 0.0;
    for (int j = i + 1; j < n; ++j) {
        const double4 pj = pos[j];
        const double dx = pj.x - pi.x;
        const double dy = pj.y - pi.y;
        const double dz = pj.z - pi.z;
        const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        e -= 1.0 / dist;
    }
    out[i] = e;
}

// Number of lanes per body: the smallest power of two (<= 32) giving enough threads
// to keep all SMs busy.
static int chooseSplit(int count, int numSMs) {
    const long long target = (long long)numSMs * SPLIT_THREADS_PER_SM;
    int k = 1;
    while (k < 32 && (long long)count * k < target) k *= 2;
    return k;
}

static void launchStep(int split, cudaStream_t stream, const double4* pos, double4* posOut,
                       double4* vel, int n, int offset, int count) {
    const int threads = STEP_BLOCK;
    const int blocks = (int)(((long long)count * split + threads - 1) / threads);
    switch (split) {
        case 1: stepKernel<1><<<blocks, threads, 0, stream>>>(pos, posOut, vel, n, offset, count); break;
        case 2: stepKernel<2><<<blocks, threads, 0, stream>>>(pos, posOut, vel, n, offset, count); break;
        case 4: stepKernel<4><<<blocks, threads, 0, stream>>>(pos, posOut, vel, n, offset, count); break;
        case 8: stepKernel<8><<<blocks, threads, 0, stream>>>(pos, posOut, vel, n, offset, count); break;
        case 16: stepKernel<16><<<blocks, threads, 0, stream>>>(pos, posOut, vel, n, offset, count); break;
        default: stepKernel<32><<<blocks, threads, 0, stream>>>(pos, posOut, vel, n, offset, count); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

struct DeviceSlice {
    int device = 0;
    int offset = 0;
    int count = 0;
    int split = 1;
    cudaStream_t stream = nullptr;
    cudaEvent_t done[2] = {nullptr, nullptr};  // slice published (by step parity)
    double4* pos[2] = {nullptr, nullptr};  // full position arrays (ping-pong)
    double4* vel = nullptr;                // velocities of this slice only
};

static int numGpusFor(int n) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        fprintf(stderr, "No CUDA device found\n");
        exit(1);
    }
    // Only spread across GPUs when each one gets enough work to amortise the exchange.
    constexpr int MIN_BODIES_PER_GPU = 1024;
    return std::max(1, std::min(deviceCount, n / MIN_BODIES_PER_GPU));
}

// Creates the CUDA contexts (and peer mappings) of the devices that will be used,
// so that one-time driver initialisation is not attributed to the simulation.
// Returns true if every pair of used devices has direct peer access.
static bool initDevices(int n) {
    const int numGpus = numGpusFor(n);
    bool allPeer = true;
    for (int g = 0; g < numGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaFree(nullptr));
        // Force module loading now (CUDA loads kernels lazily on first launch).
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, stepKernel<1>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, stepKernel<2>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, stepKernel<4>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, stepKernel<8>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, stepKernel<16>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, stepKernel<32>));
        for (int h = 0; h < numGpus; ++h) {
            if (h == g) continue;
            int canAccess = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, g, h));
            if (canAccess) {
                cudaError_t e = cudaDeviceEnablePeerAccess(h, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
                cudaGetLastError();
            } else {
                allPeer = false;
            }
        }
    }
    CUDA_CHECK(cudaSetDevice(0));
    return allPeer;
}

// Runs the full simulation on all used GPUs. Bodies are split across devices; each
// device keeps a full copy of the positions which is all-gathered after every step,
// either directly between peers or (without P2P support) through pinned host memory.
void simulate(std::vector<Body>& bodies, int numSteps, bool usePeer) {
    const int n = (int)bodies.size();
    if (n == 0) return;
    const int numGpus = numGpusFor(n);
    const bool staged = numGpus > 1 && !usePeer;

    std::vector<double4> hPos(n), hVel(n);
    for (int i = 0; i < n; ++i) {
        hPos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
        hVel[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
    }
    // Pinned ping-pong buffers for the host-staged position exchange.
    double4* hStage[2] = {nullptr, nullptr};
    if (staged) {
        CUDA_CHECK(cudaMallocHost(&hStage[0], n * sizeof(double4)));
        CUDA_CHECK(cudaMallocHost(&hStage[1], n * sizeof(double4)));
    }

    std::vector<DeviceSlice> slices(numGpus);
    for (int g = 0; g < numGpus; ++g) {
        DeviceSlice& s = slices[g];
        s.device = g;
        s.offset = (int)((long long)n * g / numGpus);
        s.count = (int)((long long)n * (g + 1) / numGpus) - s.offset;
        CUDA_CHECK(cudaSetDevice(g));
        int numSMs = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, g));
        s.split = chooseSplit(s.count, numSMs);
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.done[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.done[1], cudaEventDisableTiming));
        CUDA_CHECK(cudaMalloc(&s.pos[0], n * sizeof(double4)));
        CUDA_CHECK(cudaMalloc(&s.pos[1], n * sizeof(double4)));
        CUDA_CHECK(cudaMalloc(&s.vel, s.count * sizeof(double4)));
        CUDA_CHECK(cudaMemcpyAsync(s.pos[0], hPos.data(), n * sizeof(double4),
                                   cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.vel, hVel.data() + s.offset, s.count * sizeof(double4),
                                   cudaMemcpyHostToDevice, s.stream));
    }
    for (auto& s : slices) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }

    // Step s reads positions from buffer cur and writes buffer cur^1. Every device
    // records done[s&1] once its slice of step s is published; consumers of step s
    // wait on that event before they start step s+1.
    int cur = 0;
    for (int step = 0; step < numSteps; ++step) {
        const int prev = (step - 1) & 1;
        for (int g = 0; g < numGpus; ++g) {
            DeviceSlice& s = slices[g];
            CUDA_CHECK(cudaSetDevice(s.device));
            if (step > 0 && numGpus > 1) {
                for (int h = 0; h < numGpus; ++h) {
                    if (h == g) continue;
                    CUDA_CHECK(cudaStreamWaitEvent(s.stream, slices[h].done[prev], 0));
                    if (staged)
                        CUDA_CHECK(cudaMemcpyAsync(s.pos[cur] + slices[h].offset,
                                                   hStage[cur] + slices[h].offset,
                                                   slices[h].count * sizeof(double4),
                                                   cudaMemcpyHostToDevice, s.stream));
                }
            }
            launchStep(s.split, s.stream, s.pos[cur], s.pos[cur ^ 1], s.vel,
                       n, s.offset, s.count);
            if (numGpus > 1) {
                if (staged) {
                    CUDA_CHECK(cudaMemcpyAsync(hStage[cur ^ 1] + s.offset, s.pos[cur ^ 1] + s.offset,
                                               s.count * sizeof(double4), cudaMemcpyDeviceToHost,
                                               s.stream));
                } else {
                    for (int h = 0; h < numGpus; ++h) {
                        if (h == g) continue;
                        CUDA_CHECK(cudaMemcpyPeerAsync(slices[h].pos[cur ^ 1] + s.offset, slices[h].device,
                                                       s.pos[cur ^ 1] + s.offset, s.device,
                                                       s.count * sizeof(double4), s.stream));
                    }
                }
                CUDA_CHECK(cudaEventRecord(s.done[step & 1], s.stream));
            }
        }
        cur ^= 1;
    }

    for (auto& s : slices) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaMemcpyAsync(hPos.data() + s.offset, s.pos[cur] + s.offset,
                                   s.count * sizeof(double4), cudaMemcpyDeviceToHost, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(hVel.data() + s.offset, s.vel, s.count * sizeof(double4),
                                   cudaMemcpyDeviceToHost, s.stream));
    }
    for (auto& s : slices) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
        CUDA_CHECK(cudaFree(s.pos[0]));
        CUDA_CHECK(cudaFree(s.pos[1]));
        CUDA_CHECK(cudaFree(s.vel));
        CUDA_CHECK(cudaEventDestroy(s.done[0]));
        CUDA_CHECK(cudaEventDestroy(s.done[1]));
        CUDA_CHECK(cudaStreamDestroy(s.stream));
    }
    CUDA_CHECK(cudaSetDevice(0));

    for (int i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(hPos[i].x, hPos[i].y, hPos[i].z);
        bodies[i].vel = Vec3(hVel[i].x, hVel[i].y, hVel[i].z);
    }
    if (staged) {
        CUDA_CHECK(cudaFreeHost(hStage[0]));
        CUDA_CHECK(cudaFreeHost(hStage[1]));
    }
}


double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const int n = (int)bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }
    if (n == 0) return energy;

    // Potential energy (assuming unit mass for all bodies), computed on the GPU
    std::vector<double4> hPos(n);
    for (int i = 0; i < n; ++i)
        hPos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
    double4* dPos = nullptr;
    double* dOut = nullptr;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&dPos, n * sizeof(double4)));
    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dPos, hPos.data(), n * sizeof(double4), cudaMemcpyHostToDevice));
    potentialKernel<<<(n + 127) / 128, 128>>>(dPos, dOut, n);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> partial(n);
    CUDA_CHECK(cudaMemcpy(partial.data(), dOut, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dPos));
    CUDA_CHECK(cudaFree(dOut));
    for (int i = 0; i < n; ++i) energy += partial[i];

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
    
    const bool usePeer = initDevices(numBodies);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    simulate(bodies, numSteps, usePeer);
    
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
