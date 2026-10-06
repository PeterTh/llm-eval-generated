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

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                             \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

// Kernel configuration: bodies per block, threads per block, and j-tile length
constexpr int BODIES_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = 128;
constexpr int TILE_J = 32;

// Minimum number of bodies per GPU before another GPU is added
constexpr int MIN_ROWS_PER_GPU = 512;

// Pair term for bodies i <- j: returns (dx, dy, dz, invDist3). Rounding is pinned with
// explicit intrinsics so the result is deterministic and bitwise identical to the optimized
// (FMA-contracted) host build of the original code.
__device__ __forceinline__ double4 pairTerm(const double4& pi, const double4& pj) {
    const double dx = __dsub_rn(pj.x, pi.x);
    const double dy = __dsub_rn(pj.y, pi.y);
    const double dz = __dsub_rn(pj.z, pi.z);
    const double distSqr = __dadd_rn(__fma_rn(dz, dz, __fma_rn(dx, dx, __dmul_rn(dy, dy))), SOFTENING);
    const double invDist = __drcp_rn(__dsqrt_rn(distSqr));
    const double invDist3 = __dmul_rn(__dmul_rn(invDist, invDist), invDist);
    return make_double4(dx, dy, dz, invDist3);
}

// One block advances B bodies [i0, i0 + B) of the slice [lo, hi) by one step.
//
// The force on body i is a sequential sum over j = 0..n-1, which we must keep in order to
// stay bitwise identical to the original. The expensive part of each term (sqrt + division)
// is independent across j, so for each tile of J bodies:
//   phase 1: all T threads compute the B x J pair terms in parallel into shared memory;
//   phase 2: 3*B lanes (one per body and component) accumulate them in j order with FMA.
// Small B gives many blocks and therefore an even load over all SMs, for any n.
// Positions are double-buffered so every thread sees the old positions, which is equivalent
// to the original computeForces + integrateBodies sequence.
template <int B, int T, int J>
__global__ void __launch_bounds__(T)
stepKernel(const double4* __restrict__ posIn, double4* __restrict__ posOut,
           double* __restrict__ vel, int lo, int hi, int n) {
    static_assert(T % B == 0 && J % (T / B) == 0 && 3 * B <= T, "bad tile configuration");
    constexpr int ROWS = T / B;  // pair terms computed per thread per tile: J / ROWS

    __shared__ double4 tile[J];
    __shared__ double4 terms[J][B];

    const int i0 = lo + blockIdx.x * B;
    const int b = threadIdx.x % B;
    const int r = threadIdx.x / B;
    const bool bodyActive = i0 + b < hi;
    const double4 pi = bodyActive ? posIn[i0 + b] : make_double4(0.0, 0.0, 0.0, 0.0);

    // Phase-2 lane mapping: lane -> (body, component)
    const int accBody = threadIdx.x / 3;
    const int accComp = threadIdx.x % 3;
    const bool accActive = threadIdx.x < 3 * B && i0 + accBody < hi;
    double F = 0.0;

    for (int base = 0; base < n; base += J) {
        const int cnt = min(J, n - base);
        for (int k = threadIdx.x; k < cnt; k += T) tile[k] = posIn[base + k];
        __syncthreads();

        if (bodyActive) {
#pragma unroll
            for (int k = r; k < J; k += ROWS) {
                if (k < cnt) terms[k][b] = pairTerm(pi, tile[k]);
            }
        }
        __syncthreads();

        if (accActive) {
            const double* t = reinterpret_cast<const double*>(&terms[0][accBody]);
            constexpr int STRIDE = 4 * B;  // doubles between terms[k][b] and terms[k+1][b]
            if (cnt == J) {
#pragma unroll 16
                for (int k = 0; k < J; ++k) F = __fma_rn(t[k * STRIDE + accComp], t[k * STRIDE + 3], F);
            } else {
                for (int k = 0; k < cnt; ++k) F = __fma_rn(t[k * STRIDE + accComp], t[k * STRIDE + 3], F);
            }
        }
        __syncthreads();
    }

    if (accActive) {
        const int i = i0 + accBody;
        double* v = vel + 4 * (i - lo) + accComp;
        const double vNew = __dadd_rn(*v, __dmul_rn(DT, F));
        *v = vNew;
        const double p = reinterpret_cast<const double*>(&posIn[i])[accComp];
        reinterpret_cast<double*>(&posOut[i])[accComp] = __fma_rn(vNew, DT, p);
    }
}

// Per-GPU state: each device owns a contiguous slice [lo, hi) of the bodies (it integrates
// their velocities) but keeps a full, double-buffered copy of all positions.
struct DeviceCtx {
    int dev = 0;
    int lo = 0, hi = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready[2] = {nullptr, nullptr};  // slice of step parity p is on the host
    double4* pos[2] = {nullptr, nullptr};
    double4* vel = nullptr;
};

static void launchStep(const DeviceCtx& c, const double4* in, double4* out, int n) {
    const int rows = c.hi - c.lo;
    if (rows <= 0) return;
    stepKernel<BODIES_PER_BLOCK, THREADS_PER_BLOCK, TILE_J>
        <<<(rows + BODIES_PER_BLOCK - 1) / BODIES_PER_BLOCK, THREADS_PER_BLOCK, 0, c.stream>>>(
            in, out, reinterpret_cast<double*>(c.vel), c.lo, c.hi, n);
}

// Advances the system by numSteps on all GPUs in ctx. Each step, every device computes its
// slice, publishes it to a pinned host staging buffer, and pulls the other slices back in.
// Two staging buffers (by step parity) are enough: a device can only overwrite parity p
// after all devices have finished their reads of parity p from two steps earlier.
static void runSimulation(std::vector<DeviceCtx>& ctx, double4* stage[2], int n, int numSteps) {
    const int G = static_cast<int>(ctx.size());
    for (int step = 0; step < numSteps; ++step) {
        const int in = step & 1, out = in ^ 1;
        for (auto& c : ctx) {
            CUDA_CHECK(cudaSetDevice(c.dev));
            launchStep(c, c.pos[in], c.pos[out], n);
            if (G > 1 && c.hi > c.lo) {
                CUDA_CHECK(cudaMemcpyAsync(stage[out] + c.lo, c.pos[out] + c.lo,
                                           sizeof(double4) * (c.hi - c.lo),
                                           cudaMemcpyDeviceToHost, c.stream));
            }
            if (G > 1) CUDA_CHECK(cudaEventRecord(c.ready[out], c.stream));
        }
        if (G == 1) continue;
        for (auto& c : ctx) {
            CUDA_CHECK(cudaSetDevice(c.dev));
            for (auto& o : ctx) {
                if (&o != &c) CUDA_CHECK(cudaStreamWaitEvent(c.stream, o.ready[out], 0));
            }
            if (c.lo > 0) {
                CUDA_CHECK(cudaMemcpyAsync(c.pos[out], stage[out], sizeof(double4) * c.lo,
                                           cudaMemcpyHostToDevice, c.stream));
            }
            if (c.hi < n) {
                CUDA_CHECK(cudaMemcpyAsync(c.pos[out] + c.hi, stage[out] + c.hi,
                                           sizeof(double4) * (n - c.hi),
                                           cudaMemcpyHostToDevice, c.stream));
            }
        }
    }
    for (auto& c : ctx) {
        CUDA_CHECK(cudaSetDevice(c.dev));
        CUDA_CHECK(cudaGetLastError());
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
    
    // Select GPUs: split the bodies across all devices, but keep enough rows per device
    // to fill it (small problems run faster on a single GPU than with per-step exchanges).
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    const int n = numBodies > 0 ? numBodies : 0;
    int numGpus = std::max(1, std::min(devCount, n / MIN_ROWS_PER_GPU));

    std::vector<DeviceCtx> ctx(numGpus);
    for (int g = 0; g < numGpus; ++g) {
        DeviceCtx& c = ctx[g];
        c.dev = g;
        c.lo = static_cast<int>(static_cast<long long>(n) * g / numGpus);
        c.hi = static_cast<int>(static_cast<long long>(n) * (g + 1) / numGpus);
        CUDA_CHECK(cudaSetDevice(c.dev));
        const int rows = c.hi - c.lo;
        CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
        for (int p = 0; p < 2; ++p) {
            CUDA_CHECK(cudaEventCreateWithFlags(&c.ready[p], cudaEventDisableTiming));
            CUDA_CHECK(cudaMalloc(&c.pos[p], sizeof(double4) * std::max(n, 1)));
            CUDA_CHECK(cudaMemset(c.pos[p], 0, sizeof(double4) * std::max(n, 1)));
        }
        CUDA_CHECK(cudaMalloc(&c.vel, sizeof(double4) * std::max(rows, 1)));
    }

    double4* stage[2] = {nullptr, nullptr};  // pinned host staging (positions + velocities)
    for (int p = 0; p < 2; ++p) {
        CUDA_CHECK(cudaMallocHost(&stage[p], sizeof(double4) * std::max(n, 1)));
    }
    double4* hPos = stage[0];
    double4* hVel = stage[1];
    for (int i = 0; i < n; ++i) {
        hPos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
        hVel[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
    }
    for (auto& c : ctx) {
        CUDA_CHECK(cudaSetDevice(c.dev));
        CUDA_CHECK(cudaMemcpy(c.pos[0], hPos, sizeof(double4) * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(c.vel, hVel + c.lo, sizeof(double4) * (c.hi - c.lo),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    if (n > 0) runSimulation(ctx, stage, n, numSteps);

    // Gather final state (positions from each device's own slice, velocities likewise)
    const int fin = numSteps & 1;
    std::vector<double4> finPos(n), finVel(n);
    for (auto& c : ctx) {
        CUDA_CHECK(cudaSetDevice(c.dev));
        const size_t rows = c.hi - c.lo;
        CUDA_CHECK(cudaMemcpyAsync(finPos.data() + c.lo, c.pos[fin] + c.lo, sizeof(double4) * rows,
                                   cudaMemcpyDeviceToHost, c.stream));
        CUDA_CHECK(cudaMemcpyAsync(finVel.data() + c.lo, c.vel, sizeof(double4) * rows,
                                   cudaMemcpyDeviceToHost, c.stream));
    }
    for (auto& c : ctx) {
        CUDA_CHECK(cudaSetDevice(c.dev));
        CUDA_CHECK(cudaStreamSynchronize(c.stream));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    for (int i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(finPos[i].x, finPos[i].y, finPos[i].z);
        bodies[i].vel = Vec3(finVel[i].x, finVel[i].y, finVel[i].z);
    }
    for (auto& c : ctx) {
        CUDA_CHECK(cudaSetDevice(c.dev));
        for (int p = 0; p < 2; ++p) {
            CUDA_CHECK(cudaFree(c.pos[p]));
            CUDA_CHECK(cudaEventDestroy(c.ready[p]));
        }
        CUDA_CHECK(cudaFree(c.vel));
        CUDA_CHECK(cudaStreamDestroy(c.stream));
    }
    for (int p = 0; p < 2; ++p) CUDA_CHECK(cudaFreeHost(stage[p]));

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
