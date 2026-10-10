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

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

constexpr int TILE_SIZE = 128;

// Use explicit rounding at the update boundaries, and retain the ascending-j
// accumulation order. This also matches the original optimized host arithmetic.
__device__ __forceinline__ double squaredDistance(double dx, double dy, double dz) {
    return __fma_rn(dz, dz, __fma_rn(dx, dx, __dmul_rn(dy, dy))) + SOFTENING;
}

template <int Threads>
__global__ __launch_bounds__(Threads) void advanceBodies(const double* __restrict__ positions,
                              double* __restrict__ nextPositions,
                              double* __restrict__ velocities, size_t n) {
    __shared__ double tileX[TILE_SIZE], tileY[TILE_SIZE], tileZ[TILE_SIZE];
    const size_t i = static_cast<size_t>(blockIdx.x) * Threads + threadIdx.x;
    const double x = i < n ? positions[i] : 0.0;
    const double y = i < n ? positions[n + i] : 0.0;
    const double z = i < n ? positions[2 * n + i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (size_t base = 0; base < n; base += TILE_SIZE) {
        #pragma unroll
        for (int offset = 0; offset < TILE_SIZE; offset += Threads) {
            const int k = offset + threadIdx.x;
            const size_t j = base + k;
            tileX[k] = j < n ? positions[j] : 0.0;
            tileY[k] = j < n ? positions[n + j] : 0.0;
            tileZ[k] = j < n ? positions[2 * n + j] : 0.0;
        }
        if constexpr (Threads == 32) __syncwarp();
        else __syncthreads();

        const int count = static_cast<int>(n - base < TILE_SIZE ? n - base : TILE_SIZE);
        if (i < n) {
            #pragma unroll 8
            for (int j = 0; j < count; ++j) {
                const double dx = tileX[j] - x;
                const double dy = tileY[j] - y;
                const double dz = tileZ[j] - z;
                const double invDist = 1.0 / sqrt(squaredDistance(dx, dy, dz));
                const double invDist3 = invDist * invDist * invDist;
                fx = __fma_rn(dx, invDist3, fx);
                fy = __fma_rn(dy, invDist3, fy);
                fz = __fma_rn(dz, invDist3, fz);
            }
        }
        // All threads, including inactive threads in the final block, must wait
        // before the shared tile is overwritten.
        if constexpr (Threads == 32) __syncwarp();
        else __syncthreads();
    }

    if (i < n) {
        const double vx = __dadd_rn(velocities[i], __dmul_rn(DT, fx));
        const double vy = __dadd_rn(velocities[n + i], __dmul_rn(DT, fy));
        const double vz = __dadd_rn(velocities[2 * n + i], __dmul_rn(DT, fz));
        velocities[i] = vx;
        velocities[n + i] = vy;
        velocities[2 * n + i] = vz;
        nextPositions[i] = __fma_rn(vx, DT, x);
        nextPositions[n + i] = __fma_rn(vy, DT, y);
        nextPositions[2 * n + i] = __fma_rn(vz, DT, z);
    }
}

template <int Threads>
__global__ __launch_bounds__(Threads) void potentialEnergy(const double* __restrict__ positions,
                                double* __restrict__ potential, size_t n) {
    __shared__ double tileX[TILE_SIZE], tileY[TILE_SIZE], tileZ[TILE_SIZE];
    const size_t first = static_cast<size_t>(blockIdx.x) * Threads;
    const size_t i = first + threadIdx.x;
    const double x = i < n ? positions[i] : 0.0;
    const double y = i < n ? positions[n + i] : 0.0;
    const double z = i < n ? positions[2 * n + i] : 0.0;
    double energy = 0.0;

    // Only visit the upper triangle: each unordered pair contributes once.
    for (size_t base = (first / TILE_SIZE) * TILE_SIZE; base < n; base += TILE_SIZE) {
        #pragma unroll
        for (int offset = 0; offset < TILE_SIZE; offset += Threads) {
            const int k = offset + threadIdx.x;
            const size_t j = base + k;
            tileX[k] = j < n ? positions[j] : 0.0;
            tileY[k] = j < n ? positions[n + j] : 0.0;
            tileZ[k] = j < n ? positions[2 * n + j] : 0.0;
        }
        if constexpr (Threads == 32) __syncwarp();
        else __syncthreads();
        const int count = static_cast<int>(n - base < TILE_SIZE ? n - base : TILE_SIZE);
        if (i < n) {
            const int begin = i >= base ? static_cast<int>(i - base + 1) : 0;
            #pragma unroll 8
            for (int j = begin; j < count; ++j) {
                const double dx = tileX[j] - x;
                const double dy = tileY[j] - y;
                const double dz = tileZ[j] - z;
                energy -= 1.0 / sqrt(squaredDistance(dx, dy, dz));
            }
        }
        if constexpr (Threads == 32) __syncwarp();
        else __syncthreads();
    }
    if (i < n) potential[i] = energy;
}

class CudaSimulation {
public:
    explicit CudaSimulation(const std::vector<Body>& bodies, int steps) : n_(bodies.size()) {
        // CUDA is required even for empty or zero-step runs; there is no CPU fallback.
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        if (n_ == 0) return;

        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        // Small blocks distribute work evenly across SMs on GPUs with limited
        // FP64 throughput. Wider blocks improve tile reuse on FP64-rich GPUs,
        // once there are enough bodies to occupy every SM.
        wideBlocks_ = properties.singleToDoublePrecisionPerfRatio <= 8 &&
                      n_ >= static_cast<size_t>(properties.multiProcessorCount) * 128;

        CUDA_CHECK(cudaMalloc(&state_, 9 * n_ * sizeof(double)));
        positions_[0] = state_;
        positions_[1] = state_ + 3 * n_;
        velocities_ = state_ + 6 * n_;
        std::vector<double> packed(6 * n_);
        for (size_t i = 0; i < n_; ++i) {
            packed[i] = bodies[i].pos.x;
            packed[n_ + i] = bodies[i].pos.y;
            packed[2 * n_ + i] = bodies[i].pos.z;
            packed[3 * n_ + i] = bodies[i].vel.x;
            packed[4 * n_ + i] = bodies[i].vel.y;
            packed[5 * n_ + i] = bodies[i].vel.z;
        }
        CUDA_CHECK(cudaMemcpyAsync(positions_[0], packed.data(), 3 * n_ * sizeof(double),
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(velocities_, packed.data() + 3 * n_, 3 * n_ * sizeof(double),
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        // A two-step graph returns to the same position buffer and can be replayed
        // without parameter updates or intermediate host/device synchronization.
        if (steps >= 2) {
            CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            launchStep(0);
            launchStep(1);
            CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
            CUDA_CHECK(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphUpload(graphExec_, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    ~CudaSimulation() {
        if (graphExec_) cudaGraphExecDestroy(graphExec_);
        if (graph_) cudaGraphDestroy(graph_);
        if (state_) cudaFree(state_);
        cudaStreamDestroy(stream_);
    }

    void run(int steps) {
        if (n_ == 0 || steps <= 0) return;
        for (int pair = 0; pair < steps / 2; ++pair) {
            CUDA_CHECK(cudaGraphLaunch(graphExec_, stream_));
        }
        current_ = steps % 2;
        if (current_) launchStep(0);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void copyBodies(std::vector<Body>& bodies) const {
        if (n_ == 0) return;
        std::vector<double> packed(6 * n_);
        CUDA_CHECK(cudaMemcpyAsync(packed.data(), positions_[current_], 3 * n_ * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(packed.data() + 3 * n_, velocities_, 3 * n_ * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        for (size_t i = 0; i < n_; ++i) {
            bodies[i].pos = Vec3(packed[i], packed[n_ + i], packed[2 * n_ + i]);
            bodies[i].vel = Vec3(packed[3 * n_ + i], packed[4 * n_ + i], packed[5 * n_ + i]);
        }
    }

    double computeTotalEnergy(const std::vector<Body>& bodies) const {
        double energy = 0.0;
        for (const auto& body : bodies) {
            energy += 0.5 * (body.vel.x * body.vel.x +
                            body.vel.y * body.vel.y + body.vel.z * body.vel.z);
        }
        if (n_ == 0) return energy;

        double* devicePotential = nullptr;
        CUDA_CHECK(cudaMalloc(&devicePotential, n_ * sizeof(double)));
        if (!wideBlocks_) {
            potentialEnergy<32><<<blocks(32), 32, 0, stream_>>>(positions_[current_], devicePotential, n_);
        } else {
            potentialEnergy<128><<<blocks(128), 128, 0, stream_>>>(positions_[current_], devicePotential, n_);
        }
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> potential(n_);
        CUDA_CHECK(cudaMemcpyAsync(potential.data(), devicePotential, n_ * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaFree(devicePotential));
        for (double value : potential) energy += value;
        return energy;
    }

private:
    unsigned int blocks(int threads) const {
        return static_cast<unsigned int>((n_ + threads - 1) / threads);
    }

    void launchStep(int input) {
        if (!wideBlocks_) {
            advanceBodies<32><<<blocks(32), 32, 0, stream_>>>(positions_[input], positions_[1 - input],
                                                           velocities_, n_);
        } else {
            advanceBodies<128><<<blocks(128), 128, 0, stream_>>>(positions_[input], positions_[1 - input],
                                                             velocities_, n_);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    size_t n_;
    double* state_ = nullptr;
    double* positions_[2] = {};
    double* velocities_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graphExec_ = nullptr;
    int current_ = 0;
    bool wideBlocks_ = false;
};

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
    
    if (numBodies < 0) {
        fprintf(stderr, "Number of bodies must be nonnegative\n");
        return 1;
    }

    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    CudaSimulation simulation(bodies, numSteps);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    simulation.run(numSteps);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %lld ms\n", static_cast<long long>(duration.count()));
    if (printResults || validate) simulation.copyBodies(bodies);
    
    // Print results for external validation
    if (printResults) {
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
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = simulation.computeTotalEnergy(bodies);
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
