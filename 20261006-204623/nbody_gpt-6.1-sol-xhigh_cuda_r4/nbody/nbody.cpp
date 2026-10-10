#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
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

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// Each thread owns one target body. Source positions are loaded cooperatively
// into shared memory and reused by all targets in the block.
template <int BlockSize, int Unroll>
__global__ void advanceBodies(const double* __restrict__ positions,
                              double* __restrict__ nextPositions,
                              double* __restrict__ velocities,
                              size_t n, size_t stride) {
    __shared__ double tile[3][BlockSize];
    const unsigned int lane = threadIdx.x;
    const size_t i = static_cast<size_t>(blockIdx.x) * BlockSize + lane;
    const bool active = i < n;
    const double px = active ? positions[i] : 0.0;
    const double py = active ? positions[stride + i] : 0.0;
    const double pz = active ? positions[2 * stride + i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (size_t base = 0; base < n; base += BlockSize) {
        const size_t j = base + lane;
        tile[0][lane] = j < n ? positions[j] : 0.0;
        tile[1][lane] = j < n ? positions[stride + j] : 0.0;
        tile[2][lane] = j < n ? positions[2 * stride + j] : 0.0;
        __syncthreads();

        const int count = static_cast<int>(n - base < BlockSize ? n - base : BlockSize);
        if (active) {
            // Preserve increasing source-index order; no parallel force reduction
            // or reduced-precision/approximate reciprocal square root is used.
            #pragma unroll Unroll
            for (int k = 0; k < count; ++k) {
                const double dx = tile[0][k] - px;
                const double dy = tile[1][k] - py;
                const double dz = tile[2][k] - pz;
                const double distSqr = __fma_rn(dz, dz, __fma_rn(dx, dx, dy * dy)) + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx = __fma_rn(dx, invDist3, fx);
                fy = __fma_rn(dy, invDist3, fy);
                fz = __fma_rn(dz, invDist3, fz);
            }
        }
        // Inactive threads in a partial block must also reach both barriers.
        __syncthreads();
    }

    if (active) {
        const double vx = __dadd_rn(velocities[i], __dmul_rn(DT, fx));
        const double vy = __dadd_rn(velocities[stride + i], __dmul_rn(DT, fy));
        const double vz = __dadd_rn(velocities[2 * stride + i], __dmul_rn(DT, fz));
        velocities[i] = vx;
        velocities[stride + i] = vy;
        velocities[2 * stride + i] = vz;
        // A separate output buffer keeps every force calculation on the same
        // position snapshot, even though integration is fused into this kernel.
        nextPositions[i] = __fma_rn(vx, DT, px);
        nextPositions[stride + i] = __fma_rn(vy, DT, py);
        nextPositions[2 * stride + i] = __fma_rn(vz, DT, pz);
    }
}

class CudaSimulation {
public:
    CudaSimulation(const std::vector<Body>& bodies, int steps)
        : n_(bodies.size()), stride_((n_ + 31) / 32 * 32), state_(6 * stride_) {
        // CUDA is required even for an empty simulation; there is no CPU fallback.
        checkCuda(cudaFree(nullptr), "initializing the CUDA runtime");
        if (n_ == 0) return;

        for (size_t i = 0; i < n_; ++i) {
            state_[i] = bodies[i].pos.x;
            state_[stride_ + i] = bodies[i].pos.y;
            state_[2 * stride_ + i] = bodies[i].pos.z;
            state_[3 * stride_ + i] = bodies[i].vel.x;
            state_[4 * stride_ + i] = bodies[i].vel.y;
            state_[5 * stride_ + i] = bodies[i].vel.z;
        }

        checkCuda(cudaMalloc(reinterpret_cast<void**>(&storage_), 9 * stride_ * sizeof(double)),
                  "allocating body state");
        positions_ = storage_;
        nextPositions_ = storage_ + 3 * stride_;
        velocities_ = storage_ + 6 * stride_;
        checkCuda(cudaMemcpy(positions_, state_.data(), 3 * stride_ * sizeof(double),
                             cudaMemcpyHostToDevice), "uploading positions");
        checkCuda(cudaMemcpy(velocities_, state_.data() + 3 * stride_,
                             3 * stride_ * sizeof(double), cudaMemcpyHostToDevice),
                  "uploading velocities");
        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                  "creating the simulation stream");

        if (steps >= 4) {
            // An even batch returns to the same position buffer, so this graph
            // can be replayed without updating any device pointers or nodes.
            graphSteps_ = steps < 32 ? steps / 2 * 2 : 32;
            checkCuda(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
                      "capturing the simulation graph");
            for (int step = 0; step < graphSteps_; ++step) advance();
            checkCuda(cudaGetLastError(), "capturing simulation kernels");
            cudaGraph_t graph = nullptr;
            checkCuda(cudaStreamEndCapture(stream_, &graph), "finishing the simulation graph");
            checkCuda(cudaGraphInstantiate(&graph_, graph, nullptr, nullptr, 0),
                      "instantiating the simulation graph");
            checkCuda(cudaGraphDestroy(graph), "freeing the captured graph");
        }
    }

    ~CudaSimulation() {
        if (graph_) checkCuda(cudaGraphExecDestroy(graph_), "freeing the simulation graph");
        if (stream_) checkCuda(cudaStreamDestroy(stream_), "freeing the simulation stream");
        if (storage_) checkCuda(cudaFree(storage_), "freeing body state");
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    void run(int steps) {
        if (n_ == 0 || steps <= 0) return;
        while (graph_ && positions_ == storage_ && steps >= graphSteps_) {
            checkCuda(cudaGraphLaunch(graph_, stream_), "launching the simulation graph");
            steps -= graphSteps_;
        }
        for (int step = 0; step < steps; ++step) advance();
        checkCuda(cudaGetLastError(), "launching the simulation kernels");
        checkCuda(cudaStreamSynchronize(stream_), "running the simulation");
    }

    void download(std::vector<Body>& bodies) {
        if (n_ == 0) return;
        checkCuda(cudaMemcpy(state_.data(), positions_, 3 * stride_ * sizeof(double),
                             cudaMemcpyDeviceToHost), "downloading positions");
        checkCuda(cudaMemcpy(state_.data() + 3 * stride_, velocities_,
                             3 * stride_ * sizeof(double), cudaMemcpyDeviceToHost),
                  "downloading velocities");
        for (size_t i = 0; i < n_; ++i) {
            bodies[i].pos = Vec3(state_[i], state_[stride_ + i], state_[2 * stride_ + i]);
            bodies[i].vel = Vec3(state_[3 * stride_ + i], state_[4 * stride_ + i],
                                 state_[5 * stride_ + i]);
        }
    }

private:
    void advance() {
        // One warp per block exposes many independent targets and avoids
        // waiting for other warps at each shared-memory tile barrier.
        constexpr int BlockSize = 32;
        const unsigned int blocks = static_cast<unsigned int>((n_ + BlockSize - 1) / BlockSize);
        if (n_ < 4096) {
            advanceBodies<BlockSize, 16><<<blocks, BlockSize, 0, stream_>>>(positions_, nextPositions_, velocities_, n_, stride_);
        } else {
            advanceBodies<BlockSize, 4><<<blocks, BlockSize, 0, stream_>>>(positions_, nextPositions_, velocities_, n_, stride_);
        }
        std::swap(positions_, nextPositions_);
    }

    const size_t n_, stride_;
    std::vector<double> state_;
    double* storage_ = nullptr;
    double* positions_ = nullptr;
    double* nextPositions_ = nullptr;
    double* velocities_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaGraphExec_t graph_ = nullptr;
    int graphSteps_ = 0;
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
    CudaSimulation simulation(bodies, numSteps);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    simulation.run(numSteps);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    simulation.download(bodies);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
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
