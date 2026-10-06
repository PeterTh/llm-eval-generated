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

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err_ = (call);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,        \
                    cudaGetErrorString(err_));                                                \
            exit(EXIT_FAILURE);                                                               \
        }                                                                                     \
    } while (0)

// Threads per block along the body (i) dimension. The device arrays are padded to a
// multiple of this with far-away sentinel bodies whose contribution underflows to exactly 0.
constexpr int BLOCK_BODIES = 64;
constexpr double SENTINEL_POS = 1e150;

// Body-body interaction, identical arithmetic to the original sequential code.
__device__ __forceinline__ void interact(const double4 pi, const double4 pj, double& Fx, double& Fy, double& Fz) {
    const double dx = pj.x - pi.x;
    const double dy = pj.y - pi.y;
    const double dz = pj.z - pi.z;
    const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
    const double invDist = rsqrt(distSqr);
    const double invDist3 = invDist * invDist * invDist;
    Fx += dx * invDist3;
    Fy += dy * invDist3;
    Fz += dz * invDist3;
}

// One full time step (computeForces + integrateBodies) for bodies [lo, hi).
// Positions are double-buffered (posIn -> posOut), so force evaluation for every body
// sees the positions from the start of the step exactly as in the original code.
// The j loop is split across P thread slices (blockDim.y) to expose more parallelism for
// small N; with P == 1 each body sums its interactions in the original j order.
template <int P>
__global__ void __launch_bounds__(BLOCK_BODIES * P)
stepKernel(const double4* __restrict__ posIn, double4* __restrict__ posOut, double4* __restrict__ vel,
           int lo, int hi, int nPad) {
    __shared__ double4 tile[BLOCK_BODIES * P];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_BODIES + tx;
    const int i = lo + blockIdx.x * BLOCK_BODIES + tx;
    const double4 pi = posIn[min(i, nPad - 1)];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    constexpr int TILE = BLOCK_BODIES * P;

    for (int base = 0; base < nPad; base += TILE) {
        if (base + tid < nPad) tile[tid] = posIn[base + tid];
        __syncthreads();
        const int chunk = base + ty * BLOCK_BODIES;
        if (chunk < nPad) {
            const double4* t = &tile[ty * BLOCK_BODIES];
#pragma unroll 16
            for (int k = 0; k < BLOCK_BODIES; ++k) interact(pi, t[k], Fx, Fy, Fz);
        }
        __syncthreads();
    }

    if constexpr (P > 1) {
        // Reuse the position tile (no longer needed) for the per-slice partial forces.
        double3 (*partial)[BLOCK_BODIES] = reinterpret_cast<double3 (*)[BLOCK_BODIES]>(tile);
        partial[ty][tx] = make_double3(Fx, Fy, Fz);
        __syncthreads();
        if (ty != 0) return;
        Fx = Fy = Fz = 0.0;
#pragma unroll
        for (int s = 0; s < P; ++s) {
            Fx += partial[s][tx].x;
            Fy += partial[s][tx].y;
            Fz += partial[s][tx].z;
        }
    }

    if (i < hi) {
        double4 v = vel[i];
        v.x += DT * Fx;
        v.y += DT * Fy;
        v.z += DT * Fz;
        vel[i] = v;
        double4 p = pi;
        p.x += v.x * DT;
        p.y += v.y * DT;
        p.z += v.z * DT;
        posOut[i] = p;
    }
}

// Potential energy contribution of body i: -sum_{j>i} 1/dist, summed in original j order.
__global__ void potentialKernel(const double4* __restrict__ pos, double* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double4 pi = pos[i];
    double e = 0.0;
    for (int j = i + 1; j < n; ++j) {
        const double4 pj = pos[j];
        const double dx = pj.x - pi.x;
        const double dy = pj.y - pi.y;
        const double dz = pj.z - pi.z;
        e -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
    }
    out[i] = e;
}

// Multi-GPU (or single-GPU) simulation driver. Each device owns a contiguous range of
// bodies, keeps a full copy of all positions, and broadcasts its updated slice to its
// peers after every step.
class GpuSimulation {
  public:
    GpuSimulation(const std::vector<Body>& bodies) : n(static_cast<int>(bodies.size())) {
        nPad = std::max(BLOCK_BODIES, (n + BLOCK_BODIES - 1) / BLOCK_BODIES * BLOCK_BODIES);

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount < 1) {
            fprintf(stderr, "No CUDA device found\n");
            exit(EXIT_FAILURE);
        }
        // Use more GPUs only when each one gets enough work to amortize the exchange.
        constexpr long long MIN_BODIES_PER_GPU = 4096;
        int numDev = static_cast<int>(std::min<long long>(deviceCount, std::max(1LL, n / MIN_BODIES_PER_GPU)));
        devs.resize(numDev);

        hostPos.resize(nPad, make_double4(SENTINEL_POS, SENTINEL_POS, SENTINEL_POS, 0.0));
        hostVel.resize(nPad, make_double4(0.0, 0.0, 0.0, 0.0));
        for (int i = 0; i < n; ++i) {
            hostPos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
            hostVel[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
        }

        // Partition bodies in BLOCK_BODIES-aligned chunks.
        const int blocks = nPad / BLOCK_BODIES;
        for (int d = 0; d < numDev; ++d) {
            Device& dev = devs[d];
            dev.id = d;
            dev.lo = std::min(n, static_cast<int>(static_cast<long long>(blocks) * d / numDev) * BLOCK_BODIES);
            dev.hi = std::min(n, static_cast<int>(static_cast<long long>(blocks) * (d + 1) / numDev) * BLOCK_BODIES);
            CUDA_CHECK(cudaSetDevice(d));
            CUDA_CHECK(cudaFree(nullptr));  // force context creation outside of timing
            // Force (lazy) kernel module loading outside of timing.
            cudaFuncAttributes attr;
            CUDA_CHECK(cudaFuncGetAttributes(&attr, stepKernel<1>));
            CUDA_CHECK(cudaFuncGetAttributes(&attr, stepKernel<2>));
            CUDA_CHECK(cudaFuncGetAttributes(&attr, stepKernel<4>));
            CUDA_CHECK(cudaFuncGetAttributes(&attr, stepKernel<8>));
            CUDA_CHECK(cudaFuncGetAttributes(&attr, stepKernel<16>));
            for (int p = 0; p < numDev; ++p) {
                int canAccess = 0;
                if (p != d && cudaDeviceCanAccessPeer(&canAccess, d, p) == cudaSuccess && canAccess) {
                    cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
                    if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                    else CUDA_CHECK(e);
                }
            }
            CUDA_CHECK(cudaStreamCreateWithFlags(&dev.stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&dev.done, cudaEventDisableTiming));
            for (int b = 0; b < 2; ++b) CUDA_CHECK(cudaMalloc(&dev.pos[b], nPad * sizeof(double4)));
            CUDA_CHECK(cudaMalloc(&dev.vel, nPad * sizeof(double4)));

            int sms = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d));
            // Split the j loop until there are enough threads to fill the device.
            const long long targetThreads = static_cast<long long>(sms) * 1024;
            const long long localBodies = static_cast<long long>(dev.hi - dev.lo + BLOCK_BODIES - 1) / BLOCK_BODIES * BLOCK_BODIES;
            dev.slices = 1;
            while (dev.slices < 16 && localBodies * dev.slices < targetThreads) dev.slices *= 2;
        }
    }

    ~GpuSimulation() {
        for (Device& dev : devs) {
            cudaSetDevice(dev.id);
            cudaFree(dev.pos[0]);
            cudaFree(dev.pos[1]);
            cudaFree(dev.vel);
            cudaEventDestroy(dev.done);
            cudaStreamDestroy(dev.stream);
        }
    }

    void upload() {
        for (Device& dev : devs) {
            CUDA_CHECK(cudaSetDevice(dev.id));
            // Both position buffers hold sentinels in the padding region.
            for (int b = 0; b < 2; ++b)
                CUDA_CHECK(cudaMemcpyAsync(dev.pos[b], hostPos.data(), nPad * sizeof(double4), cudaMemcpyHostToDevice, dev.stream));
            CUDA_CHECK(cudaMemcpyAsync(dev.vel, hostVel.data(), nPad * sizeof(double4), cudaMemcpyHostToDevice, dev.stream));
        }
        // Peers write into each other's buffers from step 0 on, so all uploads must be complete.
        for (Device& dev : devs) {
            CUDA_CHECK(cudaSetDevice(dev.id));
            CUDA_CHECK(cudaStreamSynchronize(dev.stream));
        }
        cur = 0;
    }

    void run(int numSteps) {
        const int numDev = static_cast<int>(devs.size());
        for (int step = 0; step < numSteps; ++step) {
            for (Device& dev : devs) {
                CUDA_CHECK(cudaSetDevice(dev.id));
                // Wait until all peers have delivered their slices of the current positions.
                if (step > 0)
                    for (Device& peer : devs)
                        if (peer.id != dev.id) CUDA_CHECK(cudaStreamWaitEvent(dev.stream, peer.done, 0));
                launch(dev);
            }
            if (numDev > 1) {
                // Broadcast each device's new positions into the peers' output buffer. That
                // buffer is not read by any kernel of this step, and the peers' reads of it
                // from the previous step completed before this step's kernels started
                // (enforced by the event waits above).
                for (Device& dev : devs) {
                    CUDA_CHECK(cudaSetDevice(dev.id));
                    if (dev.hi <= dev.lo) continue;
                    for (Device& peer : devs) {
                        if (peer.id == dev.id) continue;
                        CUDA_CHECK(cudaMemcpyPeerAsync(peer.pos[cur ^ 1] + dev.lo, peer.id, dev.pos[cur ^ 1] + dev.lo, dev.id,
                                                       (dev.hi - dev.lo) * sizeof(double4), dev.stream));
                    }
                }
                for (Device& dev : devs) {
                    CUDA_CHECK(cudaSetDevice(dev.id));
                    CUDA_CHECK(cudaEventRecord(dev.done, dev.stream));
                }
            }
            cur ^= 1;
        }
    }

    void download(std::vector<Body>& bodies) {
        for (Device& dev : devs) {
            if (dev.hi <= dev.lo) continue;
            CUDA_CHECK(cudaSetDevice(dev.id));
            CUDA_CHECK(cudaMemcpyAsync(hostPos.data() + dev.lo, dev.pos[cur] + dev.lo, (dev.hi - dev.lo) * sizeof(double4),
                                       cudaMemcpyDeviceToHost, dev.stream));
            CUDA_CHECK(cudaMemcpyAsync(hostVel.data() + dev.lo, dev.vel + dev.lo, (dev.hi - dev.lo) * sizeof(double4),
                                       cudaMemcpyDeviceToHost, dev.stream));
        }
        for (Device& dev : devs) {
            CUDA_CHECK(cudaSetDevice(dev.id));
            CUDA_CHECK(cudaStreamSynchronize(dev.stream));
        }
        for (int i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(hostPos[i].x, hostPos[i].y, hostPos[i].z);
            bodies[i].vel = Vec3(hostVel[i].x, hostVel[i].y, hostVel[i].z);
        }
    }

  private:
    struct Device {
        int id = 0;
        int lo = 0, hi = 0;
        int slices = 1;
        cudaStream_t stream = nullptr;
        cudaEvent_t done = nullptr;
        double4* pos[2] = {nullptr, nullptr};
        double4* vel = nullptr;
    };

    void launch(Device& dev) {
        if (dev.hi <= dev.lo) return;
        const dim3 grid((dev.hi - dev.lo + BLOCK_BODIES - 1) / BLOCK_BODIES);
        const dim3 block(BLOCK_BODIES, dev.slices);
        const double4* in = dev.pos[cur];
        double4* out = dev.pos[cur ^ 1];
        switch (dev.slices) {
            case 1: stepKernel<1><<<grid, block, 0, dev.stream>>>(in, out, dev.vel, dev.lo, dev.hi, nPad); break;
            case 2: stepKernel<2><<<grid, block, 0, dev.stream>>>(in, out, dev.vel, dev.lo, dev.hi, nPad); break;
            case 4: stepKernel<4><<<grid, block, 0, dev.stream>>>(in, out, dev.vel, dev.lo, dev.hi, nPad); break;
            case 8: stepKernel<8><<<grid, block, 0, dev.stream>>>(in, out, dev.vel, dev.lo, dev.hi, nPad); break;
            default: stepKernel<16><<<grid, block, 0, dev.stream>>>(in, out, dev.vel, dev.lo, dev.hi, nPad); break;
        }
        CUDA_CHECK(cudaGetLastError());
    }

    int n;
    int nPad;
    int cur = 0;
    std::vector<Device> devs;
    std::vector<double4> hostPos;
    std::vector<double4> hostVel;
};

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies), per-body partial sums on the GPU
    if (n > 1) {
        CUDA_CHECK(cudaSetDevice(0));
        std::vector<double4> hostPos(n);
        for (size_t i = 0; i < n; ++i) hostPos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
        double4* dPos = nullptr;
        double* dOut = nullptr;
        CUDA_CHECK(cudaMalloc(&dPos, n * sizeof(double4)));
        CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dPos, hostPos.data(), n * sizeof(double4), cudaMemcpyHostToDevice));
        constexpr int threads = 128;
        potentialKernel<<<static_cast<unsigned>((n + threads - 1) / threads), threads>>>(dPos, dOut, static_cast<int>(n));
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> partial(n);
        CUDA_CHECK(cudaMemcpy(partial.data(), dOut, n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dPos));
        CUDA_CHECK(cudaFree(dOut));
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
    
    GpuSimulation sim(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    sim.upload();
    sim.run(numSteps);
    sim.download(bodies);
    
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
