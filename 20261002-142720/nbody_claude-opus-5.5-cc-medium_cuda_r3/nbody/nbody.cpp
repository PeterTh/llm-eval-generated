#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

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

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                             \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

constexpr int FORCE_BLOCK = 128;
constexpr int ENERGY_BLOCK = 256;
// Minimum number of bodies per GPU before work is split across several GPUs
constexpr int MIN_BODIES_PER_GPU = 2048;
// Threads per SM needed to saturate a GPU with the force kernel
constexpr int MIN_THREADS_PER_SM = 2048;

// Positions and velocities are stored packed as [x0,y0,z0,x1,y1,z1,...] so that a
// contiguous range of bodies is a contiguous range of memory.
//
// One simulation step (force computation + integration) for bodies [lo, hi).
// Forces only depend on positions, which are read from posIn and the integrated
// positions are written to posOut, so the step is equivalent to the sequential
// computeForces() followed by integrateBodies().
//
// P threads cooperate on one body (P == 1 for large N, where the j-summation order
// is identical to the sequential code). For small N, P > 1 splits the j-range into
// P interleaved partial sums that are combined with a fixed shuffle tree so that
// enough threads exist to fill the GPU.
template <int BS, int P>
__global__ void __launch_bounds__(BS) stepKernel(const double* __restrict__ posIn, double* __restrict__ posOut,
                                                 double* __restrict__ vel, int n, int lo, int hi) {
    __shared__ double sh[3 * BS];

    const int tid = threadIdx.x;
    const int lane = tid % P;
    const int i = lo + blockIdx.x * (BS / P) + tid / P;
    const bool active = i < hi;

    double px = 0.0, py = 0.0, pz = 0.0;
    if (active) {
        px = posIn[3 * (size_t)i + 0];
        py = posIn[3 * (size_t)i + 1];
        pz = posIn[3 * (size_t)i + 2];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += BS) {
        const int cnt = min(BS, n - base);
        const double* src = posIn + 3 * (size_t)base;
        for (int k = tid; k < 3 * cnt; k += BS) sh[k] = src[k];
        __syncthreads();

        if (cnt == BS) {
#pragma unroll 16
            for (int j = lane; j < BS; j += P) {
                const double dx = sh[3 * j + 0] - px;
                const double dy = sh[3 * j + 1] - py;
                const double dz = sh[3 * j + 2] - pz;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        } else {
            for (int j = lane; j < cnt; j += P) {
                const double dx = sh[3 * j + 0] - px;
                const double dy = sh[3 * j + 1] - py;
                const double dz = sh[3 * j + 2] - pz;
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

    if constexpr (P > 1) {
#pragma unroll
        for (int off = P / 2; off > 0; off >>= 1) {
            Fx += __shfl_down_sync(0xffffffffu, Fx, off, P);
            Fy += __shfl_down_sync(0xffffffffu, Fy, off, P);
            Fz += __shfl_down_sync(0xffffffffu, Fz, off, P);
        }
    }

    if (active && lane == 0) {
        double vx = vel[3 * (size_t)i + 0];
        double vy = vel[3 * (size_t)i + 1];
        double vz = vel[3 * (size_t)i + 2];
        vx += DT * Fx;
        vy += DT * Fy;
        vz += DT * Fz;
        vel[3 * (size_t)i + 0] = vx;
        vel[3 * (size_t)i + 1] = vy;
        vel[3 * (size_t)i + 2] = vz;
        px += vx * DT;
        py += vy * DT;
        pz += vz * DT;
        posOut[3 * (size_t)i + 0] = px;
        posOut[3 * (size_t)i + 1] = py;
        posOut[3 * (size_t)i + 2] = pz;
    }
}

template <int P>
void launchStep(const double* posIn, double* posOut, double* vel, int n, int lo, int hi, cudaStream_t stream) {
    constexpr int bodiesPerBlock = FORCE_BLOCK / P;
    const int grid = (hi - lo + bodiesPerBlock - 1) / bodiesPerBlock;
    stepKernel<FORCE_BLOCK, P><<<grid, FORCE_BLOCK, 0, stream>>>(posIn, posOut, vel, n, lo, hi);
}

void launchStep(int p, const double* posIn, double* posOut, double* vel, int n, int lo, int hi,
                cudaStream_t stream) {
    switch (p) {
        case 1: launchStep<1>(posIn, posOut, vel, n, lo, hi, stream); break;
        case 2: launchStep<2>(posIn, posOut, vel, n, lo, hi, stream); break;
        case 4: launchStep<4>(posIn, posOut, vel, n, lo, hi, stream); break;
        case 8: launchStep<8>(posIn, posOut, vel, n, lo, hi, stream); break;
        case 16: launchStep<16>(posIn, posOut, vel, n, lo, hi, stream); break;
        default: launchStep<32>(posIn, posOut, vel, n, lo, hi, stream); break;
    }
}

// Per-block partial sums of total energy (kinetic + potential).
template <int BS>
__global__ void __launch_bounds__(BS) energyKernel(const double* __restrict__ pos, const double* __restrict__ vel,
                                                   int n, double* __restrict__ blockSums) {
    __shared__ double sh[3 * BS];
    __shared__ double red[BS / 32];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * BS + tid;
    const bool active = i < n;

    double px = 0.0, py = 0.0, pz = 0.0;
    double e = 0.0;
    if (active) {
        px = pos[3 * (size_t)i + 0];
        py = pos[3 * (size_t)i + 1];
        pz = pos[3 * (size_t)i + 2];
        const double vx = vel[3 * (size_t)i + 0];
        const double vy = vel[3 * (size_t)i + 1];
        const double vz = vel[3 * (size_t)i + 2];
        e = 0.5 * (vx * vx + vy * vy + vz * vz);
    }

    // Only tiles containing some j > i for this block are needed
    const int firstBase = blockIdx.x * BS;
    double pot = 0.0;
    for (int base = firstBase; base < n; base += BS) {
        const int cnt = min(BS, n - base);
        const double* src = pos + 3 * (size_t)base;
        for (int k = tid; k < 3 * cnt; k += BS) sh[k] = src[k];
        __syncthreads();
        if (active) {
            const int jStart = max(0, i + 1 - base);
            for (int j = jStart; j < cnt; ++j) {
                const double dx = sh[3 * j + 0] - px;
                const double dy = sh[3 * j + 1] - py;
                const double dz = sh[3 * j + 2] - pz;
                const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                pot += 1.0 / dist;
            }
        }
        __syncthreads();
    }
    e -= pot;

    for (int off = 16; off > 0; off >>= 1) e += __shfl_down_sync(0xffffffffu, e, off);
    if ((tid & 31) == 0) red[tid >> 5] = e;
    __syncthreads();
    if (tid < 32) {
        e = (tid < BS / 32) ? red[tid] : 0.0;
        for (int off = 16; off > 0; off >>= 1) e += __shfl_down_sync(0xffffffffu, e, off);
        if (tid == 0) blockSums[blockIdx.x] = e;
    }
}

struct GpuContext {
    int device;
    int lo, hi;              // owned body range
    int threadsPerBody;      // cooperating threads per body (power of two)
    double* pos[2];          // full position arrays (double buffered)
    double* vel;             // full velocity array (only [lo,hi) is updated)
    cudaStream_t stream;
    cudaEvent_t done;        // signals local slice of new positions is on the host
};

class NBodyGpu {
  public:
    explicit NBodyGpu(int n) : n_(n) {
        int devCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devCount));
        if (devCount < 1) {
            fprintf(stderr, "No CUDA device found\n");
            exit(EXIT_FAILURE);
        }
        int numGpus = std::max(1, std::min(devCount, n / MIN_BODIES_PER_GPU));
        const size_t bytes = std::max<size_t>(1, 3 * (size_t)n) * sizeof(double);

        // Distribute bodies in multiples of the block size
        const int blocks = (n + FORCE_BLOCK - 1) / FORCE_BLOCK;
        numGpus = std::max(1, std::min(numGpus, blocks));
        gpus_.resize(numGpus);
        for (int g = 0; g < numGpus; ++g) {
            GpuContext& c = gpus_[g];
            c.device = g;
            c.lo = std::min(n, (int)((long long)blocks * g / numGpus) * FORCE_BLOCK);
            c.hi = std::min(n, (int)((long long)blocks * (g + 1) / numGpus) * FORCE_BLOCK);
            CUDA_CHECK(cudaSetDevice(g));
            int numSMs = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, g));
            // Use more threads per body until the device has enough threads to be saturated
            c.threadsPerBody = 1;
            const long long targetThreads = (long long)numSMs * MIN_THREADS_PER_SM;
            while (c.threadsPerBody < 32 && (long long)(c.hi - c.lo) * c.threadsPerBody < targetThreads)
                c.threadsPerBody *= 2;
            CUDA_CHECK(cudaMalloc(&c.pos[0], bytes));
            CUDA_CHECK(cudaMalloc(&c.pos[1], bytes));
            CUDA_CHECK(cudaMalloc(&c.vel, bytes));
            CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&c.done, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaMallocHost(&hPos_, bytes));
        CUDA_CHECK(cudaMallocHost(&hVel_, bytes));
    }

    ~NBodyGpu() {
        for (auto& c : gpus_) {
            cudaSetDevice(c.device);
            cudaFree(c.pos[0]);
            cudaFree(c.pos[1]);
            cudaFree(c.vel);
            cudaStreamDestroy(c.stream);
            cudaEventDestroy(c.done);
        }
        cudaFreeHost(hPos_);
        cudaFreeHost(hVel_);
    }

    int numGpus() const { return (int)gpus_.size(); }

    void upload(const std::vector<Body>& bodies) {
        for (int i = 0; i < n_; ++i) {
            hPos_[3 * (size_t)i + 0] = bodies[i].pos.x;
            hPos_[3 * (size_t)i + 1] = bodies[i].pos.y;
            hPos_[3 * (size_t)i + 2] = bodies[i].pos.z;
            hVel_[3 * (size_t)i + 0] = bodies[i].vel.x;
            hVel_[3 * (size_t)i + 1] = bodies[i].vel.y;
            hVel_[3 * (size_t)i + 2] = bodies[i].vel.z;
        }
        const size_t bytes = 3 * (size_t)n_ * sizeof(double);
        for (auto& c : gpus_) {
            CUDA_CHECK(cudaSetDevice(c.device));
            CUDA_CHECK(cudaMemcpyAsync(c.pos[cur_], hPos_, bytes, cudaMemcpyHostToDevice, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(c.vel, hVel_, bytes, cudaMemcpyHostToDevice, c.stream));
        }
    }

    void step() {
        const int nxt = cur_ ^ 1;
        for (auto& c : gpus_) {
            if (c.hi <= c.lo) continue;
            CUDA_CHECK(cudaSetDevice(c.device));
            launchStep(c.threadsPerBody, c.pos[cur_], c.pos[nxt], c.vel, n_, c.lo, c.hi, c.stream);
            CUDA_CHECK(cudaGetLastError());
            if (gpus_.size() > 1) {
                // Publish own slice of the new positions through pinned host memory
                CUDA_CHECK(cudaMemcpyAsync(hPos_ + 3 * (size_t)c.lo, c.pos[nxt] + 3 * (size_t)c.lo,
                                           3 * (size_t)(c.hi - c.lo) * sizeof(double), cudaMemcpyDeviceToHost,
                                           c.stream));
                CUDA_CHECK(cudaEventRecord(c.done, c.stream));
            }
        }
        if (gpus_.size() > 1) {
            // All-gather: fetch the other GPUs' slices
            for (auto& c : gpus_) {
                CUDA_CHECK(cudaSetDevice(c.device));
                for (auto& o : gpus_) {
                    if (&o == &c || o.hi <= o.lo) continue;
                    CUDA_CHECK(cudaStreamWaitEvent(c.stream, o.done, 0));
                    CUDA_CHECK(cudaMemcpyAsync(c.pos[nxt] + 3 * (size_t)o.lo, hPos_ + 3 * (size_t)o.lo,
                                               3 * (size_t)(o.hi - o.lo) * sizeof(double), cudaMemcpyHostToDevice,
                                               c.stream));
                }
            }
        }
        cur_ = nxt;
    }

    void download(std::vector<Body>& bodies) {
        for (auto& c : gpus_) {
            if (c.hi <= c.lo) continue;
            CUDA_CHECK(cudaSetDevice(c.device));
            const size_t off = 3 * (size_t)c.lo;
            const size_t bytes = 3 * (size_t)(c.hi - c.lo) * sizeof(double);
            CUDA_CHECK(cudaMemcpyAsync(hPos_ + off, c.pos[cur_] + off, bytes, cudaMemcpyDeviceToHost, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(hVel_ + off, c.vel + off, bytes, cudaMemcpyDeviceToHost, c.stream));
        }
        synchronize();
        for (int i = 0; i < n_; ++i) {
            bodies[i].pos = Vec3(hPos_[3 * (size_t)i + 0], hPos_[3 * (size_t)i + 1], hPos_[3 * (size_t)i + 2]);
            bodies[i].vel = Vec3(hVel_[3 * (size_t)i + 0], hVel_[3 * (size_t)i + 1], hVel_[3 * (size_t)i + 2]);
        }
    }

    void synchronize() {
        for (auto& c : gpus_) {
            CUDA_CHECK(cudaSetDevice(c.device));
            CUDA_CHECK(cudaStreamSynchronize(c.stream));
        }
    }

  private:
    int n_;
    int cur_ = 0;
    std::vector<GpuContext> gpus_;
    double* hPos_ = nullptr;
    double* hVel_ = nullptr;
};

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const int n = (int)bodies.size();
    if (n == 0) return 0.0;
    std::vector<double> hPos(3 * (size_t)n), hVel(3 * (size_t)n);
    for (int i = 0; i < n; ++i) {
        hPos[3 * (size_t)i + 0] = bodies[i].pos.x;
        hPos[3 * (size_t)i + 1] = bodies[i].pos.y;
        hPos[3 * (size_t)i + 2] = bodies[i].pos.z;
        hVel[3 * (size_t)i + 0] = bodies[i].vel.x;
        hVel[3 * (size_t)i + 1] = bodies[i].vel.y;
        hVel[3 * (size_t)i + 2] = bodies[i].vel.z;
    }
    const int grid = (n + ENERGY_BLOCK - 1) / ENERGY_BLOCK;
    const size_t bytes = 3 * (size_t)n * sizeof(double);
    double *dPos, *dVel, *dSums;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&dPos, bytes));
    CUDA_CHECK(cudaMalloc(&dVel, bytes));
    CUDA_CHECK(cudaMalloc(&dSums, grid * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dPos, hPos.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, hVel.data(), bytes, cudaMemcpyHostToDevice));
    energyKernel<ENERGY_BLOCK><<<grid, ENERGY_BLOCK>>>(dPos, dVel, n, dSums);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> sums(grid);
    CUDA_CHECK(cudaMemcpy(sums.data(), dSums, grid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dPos));
    CUDA_CHECK(cudaFree(dVel));
    CUDA_CHECK(cudaFree(dSums));
    double energy = 0.0;
    for (double s : sums) energy += s;
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
    
    // Set up GPU(s) (context creation and allocation are not part of the timed region)
    NBodyGpu sim(numBodies);
    sim.synchronize();

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    sim.upload(bodies);
    for (int step = 0; step < numSteps; ++step) {
        sim.step();
    }
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
