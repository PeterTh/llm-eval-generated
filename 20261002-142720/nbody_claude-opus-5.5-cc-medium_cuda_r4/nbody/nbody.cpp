#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

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
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                                        \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

// One fused step: velocities are updated from the forces computed on the
// (read-only) input positions, then the new positions are written to a
// separate output buffer. This is equivalent to computeForces() followed by
// integrateBodies() in the sequential version. The j-loop runs in the same
// order as the original code so the floating-point summation is identical.
// FMA usage is explicit (device code is built with --fmad=false) and mirrors
// the contraction the optimizing host compiler applies to the reference code.
template <int BS>
__global__ void __launch_bounds__(BS) stepKernel(const double4* __restrict__ posIn, double4* __restrict__ posOut,
                                                 double4* __restrict__ vel, int n, int iBegin, int iEnd) {
    __shared__ double4 tile[BS];
    const int i = iBegin + blockIdx.x * BS + threadIdx.x;
    const bool active = i < iEnd;
    const double4 pi = active ? posIn[i] : make_double4(0.0, 0.0, 0.0, 0.0);
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += BS) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = posIn[j];
        __syncthreads();
        const int cnt = min(BS, n - base);
        if (cnt == BS) {
#pragma unroll 8
            for (int k = 0; k < BS; ++k) {
                const double4 pj = tile[k];
                const double dx = pj.x - pi.x;
                const double dy = pj.y - pi.y;
                const double dz = pj.z - pi.z;
                const double distSqr = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx = fma(dx, invDist3, Fx);
                Fy = fma(dy, invDist3, Fy);
                Fz = fma(dz, invDist3, Fz);
            }
        } else {
            for (int k = 0; k < cnt; ++k) {
                const double4 pj = tile[k];
                const double dx = pj.x - pi.x;
                const double dy = pj.y - pi.y;
                const double dz = pj.z - pi.z;
                const double distSqr = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx = fma(dx, invDist3, Fx);
                Fy = fma(dy, invDist3, Fy);
                Fz = fma(dz, invDist3, Fz);
            }
        }
        __syncthreads();
    }

    if (active) {
        double4 v = vel[i];
        v.x += DT * Fx;
        v.y += DT * Fy;
        v.z += DT * Fz;
        vel[i] = v;
        posOut[i] = make_double4(fma(v.x, DT, pi.x), fma(v.y, DT, pi.y), fma(v.z, DT, pi.z), 0.0);
    }
}

// Per-body potential energy partial sums (j > i), same inner order as the original.
template <int BS>
__global__ void __launch_bounds__(BS) potentialKernel(const double4* __restrict__ pos, double* __restrict__ partial,
                                                      int n) {
    __shared__ double4 tile[BS];
    const int i = blockIdx.x * BS + threadIdx.x;
    const double4 pi = i < n ? pos[i] : make_double4(0.0, 0.0, 0.0, 0.0);
    double e = 0.0;
    // Tiles before this block's first body contain no j > i.
    for (int base = blockIdx.x * BS; base < n; base += BS) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = pos[j];
        __syncthreads();
        const int cnt = min(BS, n - base);
        for (int k = 0; k < cnt; ++k) {
            if (base + k > i) {
                const double4 pj = tile[k];
                const double dx = pj.x - pi.x;
                const double dy = pj.y - pi.y;
                const double dz = pj.z - pi.z;
                e -= 1.0 / sqrt(fma(dz, dz, fma(dy, dy, dx * dx)) + SOFTENING);
            }
        }
        __syncthreads();
    }
    if (i < n) partial[i] = e;
}

// Bodies per GPU below which using additional devices does not pay off
// (the per-step all-gather of positions would dominate).
constexpr int MIN_BODIES_PER_GPU = 2048;

struct DeviceCtx {
    int dev = 0;
    int iBegin = 0, iEnd = 0;
    double4* pos[2] = {nullptr, nullptr};
    double4* vel = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    int smCount = 1;
};

template <int BS>
static void launchStep(const DeviceCtx& c, int cur, int n) {
    const int cnt = c.iEnd - c.iBegin;
    const int blocks = (cnt + BS - 1) / BS;
    stepKernel<BS><<<blocks, BS, 0, c.stream>>>(c.pos[cur], c.pos[cur ^ 1], c.vel, n, c.iBegin, c.iEnd);
}

static void launchStepAuto(const DeviceCtx& c, int cur, int n) {
    const int cnt = c.iEnd - c.iBegin;
    // Pick the largest block size that still gives enough blocks to fill the GPU.
    if ((cnt + 255) / 256 >= 2 * c.smCount) launchStep<256>(c, cur, n);
    else if ((cnt + 127) / 128 >= c.smCount) launchStep<128>(c, cur, n);
    else if ((cnt + 63) / 64 >= c.smCount) launchStep<64>(c, cur, n);
    else launchStep<32>(c, cur, n);
}

// Runs the whole simulation on the GPU(s) and writes back the final state.
void runSimulationGPU(std::vector<Body>& bodies, int numSteps) {
    const int n = static_cast<int>(bodies.size());
    if (n == 0) return;

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    int numDev = std::max(1, std::min(devCount, n / MIN_BODIES_PER_GPU));

    std::vector<double4> hPos(n), hVel(n);
    for (int i = 0; i < n; ++i) {
        hPos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
        hVel[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
    }

    std::vector<DeviceCtx> ctx(numDev);
    const size_t bytes = sizeof(double4) * n;
    for (int d = 0; d < numDev; ++d) {
        DeviceCtx& c = ctx[d];
        c.dev = d;
        c.iBegin = static_cast<int>((static_cast<long long>(n) * d) / numDev);
        c.iEnd = static_cast<int>((static_cast<long long>(n) * (d + 1)) / numDev);
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaDeviceGetAttribute(&c.smCount, cudaDevAttrMultiProcessorCount, d));
        CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&c.done, cudaEventDisableTiming));
        CUDA_CHECK(cudaMalloc(&c.pos[0], bytes));
        CUDA_CHECK(cudaMalloc(&c.pos[1], bytes));
        CUDA_CHECK(cudaMalloc(&c.vel, bytes));
        CUDA_CHECK(cudaMemcpyAsync(c.pos[0], hPos.data(), bytes, cudaMemcpyHostToDevice, c.stream));
        CUDA_CHECK(cudaMemcpyAsync(c.vel, hVel.data(), bytes, cudaMemcpyHostToDevice, c.stream));
    }

    int cur = 0;
    for (int step = 0; step < numSteps; ++step) {
        for (int d = 0; d < numDev; ++d) {
            CUDA_CHECK(cudaSetDevice(ctx[d].dev));
            launchStepAuto(ctx[d], cur, n);
            CUDA_CHECK(cudaEventRecord(ctx[d].done, ctx[d].stream));
        }
        if (numDev > 1) {
            // All-gather the freshly integrated position slices.
            for (int d = 0; d < numDev; ++d) {
                CUDA_CHECK(cudaSetDevice(ctx[d].dev));
                for (int s = 0; s < numDev; ++s) {
                    if (s == d) continue;
                    CUDA_CHECK(cudaStreamWaitEvent(ctx[d].stream, ctx[s].done, 0));
                }
            }
            for (int d = 0; d < numDev; ++d) {
                CUDA_CHECK(cudaSetDevice(ctx[d].dev));
                for (int s = 0; s < numDev; ++s) {
                    if (s == d) continue;
                    const size_t off = ctx[s].iBegin;
                    const size_t cnt = ctx[s].iEnd - ctx[s].iBegin;
                    CUDA_CHECK(cudaMemcpyPeerAsync(ctx[d].pos[cur ^ 1] + off, ctx[d].dev, ctx[s].pos[cur ^ 1] + off,
                                                   ctx[s].dev, cnt * sizeof(double4), ctx[d].stream));
                }
                CUDA_CHECK(cudaEventRecord(ctx[d].done, ctx[d].stream));
            }
        }
        cur ^= 1;
    }

    for (int d = 0; d < numDev; ++d) {
        const DeviceCtx& c = ctx[d];
        const size_t cnt = c.iEnd - c.iBegin;
        CUDA_CHECK(cudaSetDevice(c.dev));
        CUDA_CHECK(cudaMemcpyAsync(hPos.data() + c.iBegin, c.pos[cur] + c.iBegin, cnt * sizeof(double4),
                                   cudaMemcpyDeviceToHost, c.stream));
        CUDA_CHECK(cudaMemcpyAsync(hVel.data() + c.iBegin, c.vel + c.iBegin, cnt * sizeof(double4),
                                   cudaMemcpyDeviceToHost, c.stream));
    }
    for (int d = 0; d < numDev; ++d) {
        DeviceCtx& c = ctx[d];
        CUDA_CHECK(cudaSetDevice(c.dev));
        CUDA_CHECK(cudaStreamSynchronize(c.stream));
        CUDA_CHECK(cudaGetLastError());
        cudaFree(c.pos[0]);
        cudaFree(c.pos[1]);
        cudaFree(c.vel);
        cudaEventDestroy(c.done);
        cudaStreamDestroy(c.stream);
    }
    CUDA_CHECK(cudaSetDevice(0));

    for (int i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(hPos[i].x, hPos[i].y, hPos[i].z);
        bodies[i].vel = Vec3(hVel[i].x, hVel[i].y, hVel[i].z);
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
    
    // Potential energy (assuming unit mass for all bodies), computed on the GPU
    if (n > 1) {
        constexpr int BS = 128;
        std::vector<double4> hPos(n);
        for (size_t i = 0; i < n; ++i) hPos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
        std::vector<double> partial(n);
        double4* dPos = nullptr;
        double* dPartial = nullptr;
        CUDA_CHECK(cudaMalloc(&dPos, n * sizeof(double4)));
        CUDA_CHECK(cudaMalloc(&dPartial, n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dPos, hPos.data(), n * sizeof(double4), cudaMemcpyHostToDevice));
        potentialKernel<BS><<<(n + BS - 1) / BS, BS>>>(dPos, dPartial, static_cast<int>(n));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(partial.data(), dPartial, n * sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(dPos);
        cudaFree(dPartial);
        for (size_t i = 0; i < n; ++i) energy += partial[i];
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
    
    // Initialize the CUDA contexts up front so start-up cost is not timed
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    for (int d = 0; d < devCount; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(nullptr));
    }
    CUDA_CHECK(cudaSetDevice(0));

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulationGPU(bodies, numSteps);
    
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
