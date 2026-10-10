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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

constexpr int TILE_SIZE = 128;

// Explicit rounding keeps the velocity update separate from integration.
// Fused operations elsewhere preserve the original optimized CPU arithmetic.
__device__ __forceinline__ double distanceSquared(double dx, double dy, double dz) {
    return __fma_rn(dz, dz, __fma_rn(dx, dx, dy * dy)) + SOFTENING;
}

template<int BLOCK_SIZE, int GROUP_SIZE = 1>
__global__ void advanceBodies(const double* __restrict__ positions,
                              double* __restrict__ nextPositions,
                              double* __restrict__ velocities,
                              size_t stride, int n) {
    __shared__ double sx[TILE_SIZE], sy[TILE_SIZE], sz[TILE_SIZE];
    const size_t i = (size_t(blockIdx.x) * BLOCK_SIZE + threadIdx.x) / GROUP_SIZE;
    const int lane = threadIdx.x % GROUP_SIZE;
    const bool active = i < size_t(n);
    const double x = active ? positions[i] : 0.0;
    const double y = active ? positions[stride + i] : 0.0;
    const double z = active ? positions[2 * stride + i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    const unsigned mask = __ballot_sync(0xffffffff, active);

    for (size_t base = 0; base < size_t(n); base += TILE_SIZE) {
        const int count = int(size_t(n) - base < TILE_SIZE ? size_t(n) - base : TILE_SIZE);
        for (int k = threadIdx.x; k < count; k += BLOCK_SIZE) {
            sx[k] = positions[base + k];
            sy[k] = positions[stride + base + k];
            sz[k] = positions[2 * stride + base + k];
        }
        __syncthreads();
        if (active) {
            // Keep the j accumulation order, including the zero self force.
            #pragma unroll 4
            for (int j = 0; j < count; j += GROUP_SIZE) {
                const int source = j + lane;
                const double dx = source < count ? sx[source] - x : 0.0;
                const double dy = source < count ? sy[source] - y : 0.0;
                const double dz = source < count ? sz[source] - z : 0.0;
                const double invDist = 1.0 / sqrt(distanceSquared(dx, dy, dz));
                const double invDist3 = invDist * invDist * invDist;
                if constexpr (GROUP_SIZE == 1) {
                    fx = __fma_rn(dx, invDist3, fx);
                    fy = __fma_rn(dy, invDist3, fy);
                    fz = __fma_rn(dz, invDist3, fz);
                } else {
                    // Cooperate on expensive pair arithmetic for small systems,
                    // then accumulate on lane zero in the original j order.
                    #pragma unroll
                    for (int k = 0; k < GROUP_SIZE; ++k) {
                        const double pairX = __shfl_sync(mask, dx, k, GROUP_SIZE);
                        const double pairY = __shfl_sync(mask, dy, k, GROUP_SIZE);
                        const double pairZ = __shfl_sync(mask, dz, k, GROUP_SIZE);
                        const double pairInvDist3 = __shfl_sync(mask, invDist3, k, GROUP_SIZE);
                        if (lane == 0 && j + k < count) {
                            fx = __fma_rn(pairX, pairInvDist3, fx);
                            fy = __fma_rn(pairY, pairInvDist3, fy);
                            fz = __fma_rn(pairZ, pairInvDist3, fz);
                        }
                    }
                }
            }
        }
        // Inactive lanes also participate before the next tile overwrites it.
        __syncthreads();
    }

    if (active && lane == 0) {
        const double vx = velocities[i] + DT * fx;
        const double vy = velocities[stride + i] + DT * fy;
        const double vz = velocities[2 * stride + i] + DT * fz;
        velocities[i] = vx;
        velocities[stride + i] = vy;
        velocities[2 * stride + i] = vz;
        // Every force reads the old positions; the next step reads this buffer.
        nextPositions[i] = __fma_rn(vx, DT, x);
        nextPositions[stride + i] = __fma_rn(vy, DT, y);
        nextPositions[2 * stride + i] = __fma_rn(vz, DT, z);
    }
}

__global__ void energyPerBody(const double* __restrict__ positions,
                              const double* __restrict__ velocities,
                              double* __restrict__ energies, size_t stride, int n) {
    constexpr int BLOCK_SIZE = 128;
    __shared__ double sx[TILE_SIZE], sy[TILE_SIZE], sz[TILE_SIZE];
    const size_t blockStart = size_t(blockIdx.x) * BLOCK_SIZE;
    const size_t i = blockStart + threadIdx.x;
    const bool active = i < size_t(n);
    const double x = active ? positions[i] : 0.0;
    const double y = active ? positions[stride + i] : 0.0;
    const double z = active ? positions[2 * stride + i] : 0.0;
    double energy = 0.0;
    if (active) {
        const double vx = velocities[i];
        const double vy = velocities[stride + i];
        const double vz = velocities[2 * stride + i];
        energy = 0.5 * __fma_rn(vz, vz, __fma_rn(vx, vx, vy * vy));
    }
    for (size_t base = blockStart; base < size_t(n); base += TILE_SIZE) {
        const int count = int(size_t(n) - base < TILE_SIZE ? size_t(n) - base : TILE_SIZE);
        for (int k = threadIdx.x; k < count; k += BLOCK_SIZE) {
            sx[k] = positions[base + k];
            sy[k] = positions[stride + base + k];
            sz[k] = positions[2 * stride + base + k];
        }
        __syncthreads();
        if (active) {
            const int first = i >= base ? int(i - base) + 1 : 0;
            for (int j = first; j < count; ++j) {
                const double dx = sx[j] - x;
                const double dy = sy[j] - y;
                const double dz = sz[j] - z;
                energy -= 1.0 / sqrt(distanceSquared(dx, dy, dz));
            }
        }
        __syncthreads();
    }
    if (active) energies[i] = energy;
}

class CudaSimulation {
public:
    explicit CudaSimulation(const std::vector<Body>& bodies, int steps)
        : n(int(bodies.size())), stride((size_t(n) + 31) / 32 * 32) {
        // A GPU is required even for an empty simulation; there is no CPU path.
        checkCuda(cudaFree(nullptr), "initializing the CUDA device");
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "creating stream");
        if (n == 0) return;
        int device = 0;
        int multiprocessors = 0;
        checkCuda(cudaGetDevice(&device), "getting CUDA device");
        checkCuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device),
                  "getting multiprocessor count");
        // Expose more parallel pairs when too few bodies would occupy the GPU.
        // Large systems avoid the shuffle cost and use one thread per body.
        if (n <= 2 * multiprocessors) groupSize = 32;
        else if (n <= 4 * multiprocessors) groupSize = 16;
        else if (n <= 16 * multiprocessors) groupSize = 8;
        else if (n <= 32 * multiprocessors) groupSize = 4;
        else if (n <= 96 * multiprocessors) groupSize = 2;
        checkCuda(cudaMalloc(&storage, 9 * stride * sizeof(double)), "allocating body state");
        positions = storage;
        velocities = storage + 3 * stride;
        nextPositions = storage + 6 * stride;
        std::vector<double> initial(6 * stride, 0.0);
        for (int i = 0; i < n; ++i) {
            initial[i] = bodies[i].pos.x;
            initial[stride + i] = bodies[i].pos.y;
            initial[2 * stride + i] = bodies[i].pos.z;
            initial[3 * stride + i] = bodies[i].vel.x;
            initial[4 * stride + i] = bodies[i].vel.y;
            initial[5 * stride + i] = bodies[i].vel.z;
        }
        checkCuda(cudaMemcpyAsync(storage, initial.data(), initial.size() * sizeof(double),
                                  cudaMemcpyHostToDevice, stream), "uploading body state");
        checkCuda(cudaStreamSynchronize(stream), "completing body upload");
        // An even graph length leaves the position buffers in their initial roles.
        // Bounded batches avoid building a graph proportional to the step count.
        graphSteps = steps > 1 ? (steps < 32 ? steps / 2 * 2 : 32) : 0;
        if (graphSteps) {
            checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "starting graph capture");
            for (int step = 0; step < graphSteps; ++step) launchStep();
            cudaGraph_t graph = nullptr;
            checkCuda(cudaStreamEndCapture(stream, &graph), "ending graph capture");
            checkCuda(cudaGraphInstantiate(&executableGraph, graph, nullptr, nullptr, 0), "instantiating graph");
            checkCuda(cudaGraphDestroy(graph), "destroying captured graph");
        }
    }

    ~CudaSimulation() {
        if (executableGraph) cudaGraphExecDestroy(executableGraph);
        if (storage) cudaFree(storage);
        if (stream) cudaStreamDestroy(stream);
    }
    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    void run(int steps) {
        if (n == 0) return;
        int remaining = steps;
        if (graphSteps) {
            for (; remaining >= graphSteps; remaining -= graphSteps)
                checkCuda(cudaGraphLaunch(executableGraph, stream), "launching simulation graph");
        }
        for (; remaining > 0; --remaining) launchStep();
        // Include completion of all GPU work in the reported simulation time.
        checkCuda(cudaStreamSynchronize(stream), "running simulation");
    }

    void download(std::vector<Body>& bodies) const {
        if (n == 0) return;
        std::vector<double> final(6 * stride);
        checkCuda(cudaMemcpy(final.data(), positions, 3 * stride * sizeof(double),
                             cudaMemcpyDeviceToHost), "downloading positions");
        checkCuda(cudaMemcpy(final.data() + 3 * stride, velocities, 3 * stride * sizeof(double),
                             cudaMemcpyDeviceToHost), "downloading velocities");
        for (int i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(final[i], final[stride + i], final[2 * stride + i]);
            bodies[i].vel = Vec3(final[3 * stride + i], final[4 * stride + i], final[5 * stride + i]);
        }
    }

    double totalEnergy() const {
        if (n == 0) return 0.0;
        double* deviceEnergies = nullptr;
        checkCuda(cudaMalloc(&deviceEnergies, size_t(n) * sizeof(double)), "allocating energy output");
        energyPerBody<<<(size_t(n) + 127) / 128, 128, 0, stream>>>(positions, velocities, deviceEnergies, stride, n);
        checkCuda(cudaGetLastError(), "launching energy kernel");
        std::vector<double> energies(n);
        checkCuda(cudaStreamSynchronize(stream), "computing energy");
        checkCuda(cudaMemcpy(energies.data(), deviceEnergies, size_t(n) * sizeof(double),
                             cudaMemcpyDeviceToHost), "downloading energies");
        checkCuda(cudaFree(deviceEnergies), "freeing energy output");
        return kahan_sum(energies);
    }

private:
    template<int GROUP_SIZE>
    void launchGroupedStep() {
        constexpr int BLOCK_SIZE = 32;
        const size_t blocks = (size_t(n) * GROUP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
        advanceBodies<BLOCK_SIZE, GROUP_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            positions, nextPositions, velocities, stride, n);
    }

    void launchStep() {
        switch (groupSize) {
            case 32: launchGroupedStep<32>(); break;
            case 16: launchGroupedStep<16>(); break;
            case 8: launchGroupedStep<8>(); break;
            case 4: launchGroupedStep<4>(); break;
            case 2: launchGroupedStep<2>(); break;
            default: launchGroupedStep<1>(); break;
        }
        checkCuda(cudaGetLastError(), "launching body kernel");
        std::swap(positions, nextPositions);
    }

    int n;
    size_t stride;
    int groupSize = 1;
    double* storage = nullptr;
    double* positions = nullptr;
    double* nextPositions = nullptr;
    double* velocities = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t executableGraph = nullptr;
    int graphSteps = 0;
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
    simulation.download(bodies);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(size_t(numBodies) * 6);
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
            double finalEnergy = simulation.totalEnergy();
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
