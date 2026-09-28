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

// CUDA tuning parameters.
//
// A thread block owns one body ("i") per thread and a shared-memory tile of the
// same number of source bodies ("j"), so the block size sets both. Because the
// kernel is limited purely by FP64 throughput, an SM running two blocks takes
// twice as long as one running a single block: the runtime is essentially
// ceil(bodies / (SMs * blockSize)) waves, and a wave that is only fractionally
// filled is wasted. Smaller blocks quantise more finely, so the block size is
// picked per run from these candidates to minimise the wasted fraction.
constexpr int BLOCK_CANDIDATES[] = {32, 64, 128};
// Per-body cost of each candidate relative to the others, measured on this
// class of GPU: 32-thread blocks lose a little throughput to their smaller
// shared-memory tiles and higher barrier density.
constexpr double BLOCK_OVERHEAD[] = {1.08, 1.0, 1.0};
constexpr int MAX_BLOCK = 128;
// Body slices are aligned to this many bodies so that any candidate block size
// produces fully coalesced accesses.
constexpr int SLICE_ALIGN = MAX_BLOCK;
// Block size of the energy kernel, which runs once outside the timed region.
constexpr int ENERGY_BLOCK = 128;
// Below this body count the per-step cross-device position exchange costs more
// than the extra compute throughput buys, so the simulation stays on one GPU.
constexpr int MULTI_GPU_MIN_BODIES = 12288;

#define CUDA_CHECK(expr)                                                                                               \
    do {                                                                                                               \
        const cudaError_t err_ = (expr);                                                                               \
        if (err_ != cudaSuccess) {                                                                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_));                \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (false)

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
// Device kernels
// ---------------------------------------------------------------------------

// One full simulation step for the bodies in [iStart, iEnd): force
// accumulation followed by integration.
//
// Thread t of block b owns body iStart + b*B + t. Every block walks the
// j-bodies in ascending order, one B-sized tile at a time, so each force sum is
// accumulated in exactly the same order as the sequential reference
// implementation - the split across threads, blocks and GPUs reassociates
// nothing.
template <int B>
__global__ void __launch_bounds__(B) nbodyStepKernel(const double* __restrict__ pxIn, const double* __restrict__ pyIn,
    const double* __restrict__ pzIn, double* __restrict__ pxOut, double* __restrict__ pyOut,
    double* __restrict__ pzOut, double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    const int n, const int iStart, const int iEnd) {
    __shared__ double sx[B];
    __shared__ double sy[B];
    __shared__ double sz[B];

    const int tid = static_cast<int>(threadIdx.x);
    const int i = iStart + static_cast<int>(blockIdx.x) * B + tid;
    // Out-of-range lanes still take part in the tile loads and barriers, but
    // their results are discarded, so any in-bounds position will do.
    const int src = (i < iEnd) ? i : 0;

    const double pxi = pxIn[src], pyi = pyIn[src], pzi = pzIn[src];
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int tile = 0; tile < n; tile += B) {
        const int tileN = min(B, n - tile);
        if (tid < tileN) {
            sx[tid] = pxIn[tile + tid];
            sy[tid] = pyIn[tile + tid];
            sz[tid] = pzIn[tile + tid];
        }
        __syncthreads();

        // The full-tile path has a compile-time trip count and no bounds test.
        const int inner = (tileN == B) ? B : tileN;
#pragma unroll 4
        for (int j = 0; j < inner; ++j) {
            const double dx = sx[j] - pxi;
            const double dy = sy[j] - pyi;
            const double dz = sz[j] - pzi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            fx += dx * invDist3;
            fy += dy * invDist3;
            fz += dz * invDist3;
        }
        __syncthreads();
    }

    if (i < iEnd) {
        const double nvx = vx[i] + DT * fx;
        const double nvy = vy[i] + DT * fy;
        const double nvz = vz[i] + DT * fz;
        vx[i] = nvx;
        vy[i] = nvy;
        vz[i] = nvz;
        pxOut[i] = pxi + nvx * DT;
        pyOut[i] = pyi + nvy * DT;
        pzOut[i] = pzi + nvz * DT;
    }
}

// Per-body contribution to the total energy: the kinetic energy of body i plus
// the potential energy of all pairs (i, j) with j > i, matching the reference
// implementation's pair enumeration and inner summation order.
__global__ void __launch_bounds__(ENERGY_BLOCK) energyKernel(const double* __restrict__ px, const double* __restrict__ py,
    const double* __restrict__ pz, const double* __restrict__ vx, const double* __restrict__ vy,
    const double* __restrict__ vz, double* __restrict__ kinetic, double* __restrict__ potential, const int n) {
    __shared__ double sx[ENERGY_BLOCK];
    __shared__ double sy[ENERGY_BLOCK];
    __shared__ double sz[ENERGY_BLOCK];

    const int tid = static_cast<int>(threadIdx.x);
    const int blockStart = static_cast<int>(blockIdx.x) * ENERGY_BLOCK;
    const int i = blockStart + tid;
    const int src = (i < n) ? i : 0;

    const double ix = px[src], iy = py[src], iz = pz[src];
    double pot = 0.0;

    // Tiles below blockStart contain only j <= i for every thread in the block,
    // so they contribute nothing and can be skipped outright.
    for (int tile = blockStart; tile < n; tile += ENERGY_BLOCK) {
        const int tileN = min(ENERGY_BLOCK, n - tile);
        if (tid < tileN) {
            sx[tid] = px[tile + tid];
            sy[tid] = py[tile + tid];
            sz[tid] = pz[tile + tid];
        }
        __syncthreads();

        for (int j = 0; j < tileN; ++j) {
            if (tile + j > i) {
                const double dx = sx[j] - ix;
                const double dy = sy[j] - iy;
                const double dz = sz[j] - iz;
                pot += 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            }
        }
        __syncthreads();
    }

    if (i < n) {
        kinetic[i] = 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
        potential[i] = pot;
    }
}

// ---------------------------------------------------------------------------
// Multi-GPU simulation driver
// ---------------------------------------------------------------------------

// State owned by one GPU. Every GPU keeps a full copy of the position arrays
// (it needs all of them to accumulate forces) but only the velocities of its
// own body slice are ever read or written.
struct GpuState {
    int device = 0;
    int iStart = 0;
    int iEnd = 0;
    // Positions are double buffered so a single kernel can read the previous
    // positions and write the integrated ones without a global barrier.
    double* pos[2][3] = {{nullptr, nullptr, nullptr}, {nullptr, nullptr, nullptr}};
    double* vel[3] = {nullptr, nullptr, nullptr};
    cudaStream_t stream = nullptr;
    // Signals that this GPU's freshly integrated position slice has landed in
    // the host staging buffer.
    cudaEvent_t stageDone = nullptr;
};

class Simulation {
  public:
    Simulation(const int n, const int numDevices) : n_(n), gpus_(numDevices) {
        const size_t bytes = static_cast<size_t>(n_) * sizeof(double);

        // Partition the bodies into aligned, near-equal contiguous slices so
        // that every GPU issues fully coalesced accesses.
        const int totalChunks = (n_ + SLICE_ALIGN - 1) / SLICE_ALIGN;
        int assigned = 0;
        for (int g = 0; g < numDevices; ++g) {
            const int myChunks = totalChunks / numDevices + (g < totalChunks % numDevices ? 1 : 0);
            GpuState& s = gpus_[g];
            s.device = g;
            s.iStart = assigned;
            assigned = std::min(n_, assigned + myChunks * SLICE_ALIGN);
            s.iEnd = assigned;

            CUDA_CHECK(cudaSetDevice(g));
            for (int b = 0; b < 2; ++b) {
                for (int c = 0; c < 3; ++c) {
                    CUDA_CHECK(cudaMalloc(&s.pos[b][c], bytes));
                }
            }
            for (int c = 0; c < 3; ++c) {
                CUDA_CHECK(cudaMalloc(&s.vel[c], bytes));
            }
            CUDA_CHECK(cudaStreamCreate(&s.stream));
            CUDA_CHECK(cudaEventCreateWithFlags(&s.stageDone, cudaEventDisableTiming));
        }

        // Pinned, portable staging buffers for the per-step position exchange.
        // These GeForce-class GPUs sit on separate PCIe root complexes, where
        // cudaMemcpyPeer falls back to a *synchronous* staged copy that stalls
        // the issuing thread; routing the exchange through pinned host memory
        // with ordinary async copies keeps every GPU's stream running ahead.
        // Two buffers are used in alternation so that a GPU staging step s+1
        // cannot overwrite data another GPU is still reading for step s.
        if (numDevices > 1) {
            CUDA_CHECK(cudaSetDevice(0));
            for (int b = 0; b < 2; ++b) {
                for (int c = 0; c < 3; ++c) {
                    CUDA_CHECK(cudaHostAlloc(&stageBuf_[b][c], bytes, cudaHostAllocPortable));
                }
            }
        }

        // Pick the block size that leaves the least of the final wave idle for
        // the largest slice any GPU has to process.
        int smCount = 1;
        CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));
        int maxRows = 0;
        for (const GpuState& s : gpus_) {
            maxRows = std::max(maxRows, s.iEnd - s.iStart);
        }
        double bestCost = -1.0;
        for (size_t c = 0; c < std::size(BLOCK_CANDIDATES); ++c) {
            const int b = BLOCK_CANDIDATES[c];
            const int waves = (maxRows + smCount * b - 1) / (smCount * b);
            const double cost = static_cast<double>(waves) * b * BLOCK_OVERHEAD[c];
            if (bestCost < 0.0 || cost < bestCost) {
                bestCost = cost;
                blockSize_ = b;
            }
        }

        scratch_.resize(static_cast<size_t>(n_) * 6);
    }

    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    ~Simulation() {
        for (GpuState& s : gpus_) {
            cudaSetDevice(s.device);
            for (int b = 0; b < 2; ++b) {
                for (int c = 0; c < 3; ++c) {
                    cudaFree(s.pos[b][c]);
                }
            }
            for (int c = 0; c < 3; ++c) {
                cudaFree(s.vel[c]);
            }
            cudaStreamDestroy(s.stream);
            cudaEventDestroy(s.stageDone);
        }
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < 3; ++c) {
                if (stageBuf_[b][c] != nullptr) {
                    cudaFreeHost(stageBuf_[b][c]);
                }
            }
        }
    }

    void upload(const std::vector<Body>& bodies) {
        for (int i = 0; i < n_; ++i) {
            scratch_[i] = bodies[i].pos.x;
            scratch_[n_ + i] = bodies[i].pos.y;
            scratch_[2 * n_ + i] = bodies[i].pos.z;
            scratch_[3 * n_ + i] = bodies[i].vel.x;
            scratch_[4 * n_ + i] = bodies[i].vel.y;
            scratch_[5 * n_ + i] = bodies[i].vel.z;
        }
        const size_t bytes = static_cast<size_t>(n_) * sizeof(double);
        for (GpuState& s : gpus_) {
            CUDA_CHECK(cudaSetDevice(s.device));
            for (int c = 0; c < 3; ++c) {
                CUDA_CHECK(cudaMemcpyAsync(
                    s.pos[0][c], &scratch_[static_cast<size_t>(c) * n_], bytes, cudaMemcpyHostToDevice, s.stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    s.vel[c], &scratch_[static_cast<size_t>(3 + c) * n_], bytes, cudaMemcpyHostToDevice, s.stream));
            }
        }
        cur_ = 0;
        stageIdx_ = 0;
    }

    void step() {
        const int in = cur_;
        const int out = 1 - cur_;
        const int numDevices = static_cast<int>(gpus_.size());
        const size_t fullBytes = static_cast<size_t>(n_) * sizeof(double);

        // Every GPU integrates its own slice from the complete previous
        // positions. Each stream already carries the dependency on the previous
        // step's exchange, so all GPUs are launched back to back here.
        for (GpuState& s : gpus_) {
            CUDA_CHECK(cudaSetDevice(s.device));
            const int count = s.iEnd - s.iStart;
            if (count == 0) {
                continue;
            }
            launchStep(s, in, out, count);
        }

        if (numDevices == 1) {
            cur_ = out;
            return;
        }

        // Exchange: every GPU pushes its own slice into the staging buffer ...
        const int buf = stageIdx_;
        for (GpuState& s : gpus_) {
            CUDA_CHECK(cudaSetDevice(s.device));
            const size_t sliceBytes = static_cast<size_t>(s.iEnd - s.iStart) * sizeof(double);
            for (int c = 0; c < 3; ++c) {
                CUDA_CHECK(cudaMemcpyAsync(stageBuf_[buf][c] + s.iStart, s.pos[out][c] + s.iStart, sliceBytes,
                    cudaMemcpyDeviceToHost, s.stream));
            }
            CUDA_CHECK(cudaEventRecord(s.stageDone, s.stream));
        }

        // ... and, once all slices have arrived, pulls the complete array back.
        // A GPU's own slice is rewritten with the value it just staged, so the
        // round trip is bit-preserving.
        for (GpuState& s : gpus_) {
            CUDA_CHECK(cudaSetDevice(s.device));
            for (const GpuState& other : gpus_) {
                if (other.device != s.device) {
                    CUDA_CHECK(cudaStreamWaitEvent(s.stream, other.stageDone, 0));
                }
            }
            for (int c = 0; c < 3; ++c) {
                CUDA_CHECK(
                    cudaMemcpyAsync(s.pos[out][c], stageBuf_[buf][c], fullBytes, cudaMemcpyHostToDevice, s.stream));
            }
        }

        stageIdx_ = 1 - buf;
        cur_ = out;
    }

    void download(std::vector<Body>& bodies) {
        for (GpuState& s : gpus_) {
            CUDA_CHECK(cudaSetDevice(s.device));
            const size_t sliceBytes = static_cast<size_t>(s.iEnd - s.iStart) * sizeof(double);
            if (sliceBytes == 0) {
                continue;
            }
            for (int c = 0; c < 3; ++c) {
                CUDA_CHECK(cudaMemcpyAsync(&scratch_[static_cast<size_t>(c) * n_ + s.iStart],
                    s.pos[cur_][c] + s.iStart, sliceBytes, cudaMemcpyDeviceToHost, s.stream));
                CUDA_CHECK(cudaMemcpyAsync(&scratch_[static_cast<size_t>(3 + c) * n_ + s.iStart],
                    s.vel[c] + s.iStart, sliceBytes, cudaMemcpyDeviceToHost, s.stream));
            }
        }
        for (GpuState& s : gpus_) {
            CUDA_CHECK(cudaSetDevice(s.device));
            CUDA_CHECK(cudaStreamSynchronize(s.stream));
        }
        for (int i = 0; i < n_; ++i) {
            bodies[i].pos.x = scratch_[i];
            bodies[i].pos.y = scratch_[n_ + i];
            bodies[i].pos.z = scratch_[2 * n_ + i];
            bodies[i].vel.x = scratch_[3 * n_ + i];
            bodies[i].vel.y = scratch_[4 * n_ + i];
            bodies[i].vel.z = scratch_[5 * n_ + i];
        }
    }

  private:
    int n_;
    std::vector<GpuState> gpus_;
    std::vector<double> scratch_;
    double* stageBuf_[2][3] = {{nullptr, nullptr, nullptr}, {nullptr, nullptr, nullptr}};
    int cur_ = 0;
    int stageIdx_ = 0;
    int blockSize_ = MAX_BLOCK;

    // Dispatch to the instantiation matching the chosen block size.
    void launchStep(const GpuState& s, const int in, const int out, const int count) {
        const int blocks = (count + blockSize_ - 1) / blockSize_;
        switch (blockSize_) {
        case 32:
            nbodyStepKernel<32><<<blocks, 32, 0, s.stream>>>(s.pos[in][0], s.pos[in][1], s.pos[in][2], s.pos[out][0],
                s.pos[out][1], s.pos[out][2], s.vel[0], s.vel[1], s.vel[2], n_, s.iStart, s.iEnd);
            break;
        case 64:
            nbodyStepKernel<64><<<blocks, 64, 0, s.stream>>>(s.pos[in][0], s.pos[in][1], s.pos[in][2], s.pos[out][0],
                s.pos[out][1], s.pos[out][2], s.vel[0], s.vel[1], s.vel[2], n_, s.iStart, s.iEnd);
            break;
        default:
            nbodyStepKernel<128><<<blocks, 128, 0, s.stream>>>(s.pos[in][0], s.pos[in][1], s.pos[in][2], s.pos[out][0],
                s.pos[out][1], s.pos[out][2], s.vel[0], s.vel[1], s.vel[2], n_, s.iStart, s.iEnd);
            break;
        }
    }
};

// Total energy of the system, evaluated on a single GPU.
double computeTotalEnergy(const std::vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);

    std::vector<double> host(static_cast<size_t>(n) * 6);
    for (int i = 0; i < n; ++i) {
        host[i] = bodies[i].pos.x;
        host[n + i] = bodies[i].pos.y;
        host[2 * n + i] = bodies[i].pos.z;
        host[3 * n + i] = bodies[i].vel.x;
        host[4 * n + i] = bodies[i].vel.y;
        host[5 * n + i] = bodies[i].vel.z;
    }

    CUDA_CHECK(cudaSetDevice(0));
    double* buf[8] = {};
    for (double*& p : buf) {
        CUDA_CHECK(cudaMalloc(&p, bytes));
    }
    for (int c = 0; c < 6; ++c) {
        CUDA_CHECK(cudaMemcpy(buf[c], &host[static_cast<size_t>(c) * n], bytes, cudaMemcpyHostToDevice));
    }

    const int blocks = (n + ENERGY_BLOCK - 1) / ENERGY_BLOCK;
    energyKernel<<<blocks, ENERGY_BLOCK>>>(buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], n);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> kinetic(n), potential(n);
    CUDA_CHECK(cudaMemcpy(kinetic.data(), buf[6], bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(potential.data(), buf[7], bytes, cudaMemcpyDeviceToHost));
    for (double* p : buf) {
        CUDA_CHECK(cudaFree(p));
    }

    // Accumulate in the reference implementation's order: every kinetic term
    // first, then the pair potentials in ascending body order.
    double energy = 0.0;
    for (int i = 0; i < n; ++i) {
        energy += kinetic[i];
    }
    for (int i = 0; i < n; ++i) {
        energy -= potential[i];
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

    if (numBodies > 0) {
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount < 1) {
            fprintf(stderr, "No CUDA device available\n");
            return 1;
        }
        int numDevices = numBodies >= MULTI_GPU_MIN_BODIES ? deviceCount : 1;
        // Keep at least one aligned slice of work per GPU.
        numDevices = std::min(numDevices, (numBodies + SLICE_ALIGN - 1) / SLICE_ALIGN);

        // Device setup happens before timing, mirroring the reference
        // implementation which excludes allocation from the measured region.
        Simulation sim(numBodies, numDevices);
        for (int g = 0; g < numDevices; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

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
    } else {
        printf("Simulation time: 0 ms\n");
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
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = numBodies > 0 ? computeTotalEnergy(bodies) : 0.0;
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
