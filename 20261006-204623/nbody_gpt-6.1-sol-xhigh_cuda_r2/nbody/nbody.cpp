#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

namespace {

constexpr unsigned int BLOCK_SIZE = 128;

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

// A thread or a small group of threads owns each target body. All threads,
// including inactive targets, load and synchronize the shared source tiles.
// Separate input/output positions allow the velocity and position updates to
// share one kernel without exposing new positions to other force calculations.
template <unsigned int THREADS_PER_BODY>
__global__ void advanceBodies(const double* __restrict__ positions,
                              double* __restrict__ nextPositions,
                              double* __restrict__ velocities,
                              unsigned int n, size_t stride) {
    __shared__ double tileX[BLOCK_SIZE];
    __shared__ double tileY[BLOCK_SIZE];
    __shared__ double tileZ[BLOCK_SIZE];
    const unsigned int lane = threadIdx.x;
    const unsigned int i = blockIdx.x * (BLOCK_SIZE / THREADS_PER_BODY) + lane / THREADS_PER_BODY;
    const unsigned int sourceLane = lane % THREADS_PER_BODY;
    const bool active = i < n;
    const double x = active ? positions[i] : 0.0;
    const double y = active ? positions[stride + i] : 0.0;
    const double z = active ? positions[2 * stride + i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    const unsigned int activeMask = __ballot_sync(0xffffffff, active);

    for (unsigned int base = 0; base < n; base += BLOCK_SIZE) {
        const unsigned int j = base + lane;
        tileX[lane] = j < n ? positions[j] : 0.0;
        tileY[lane] = j < n ? positions[stride + j] : 0.0;
        tileZ[lane] = j < n ? positions[2 * stride + j] : 0.0;
        __syncthreads();

        const unsigned int count = min(BLOCK_SIZE, n - base);
        if (active) {
            #pragma unroll 8
            for (unsigned int k = 0; k < count; k += THREADS_PER_BODY) {
                const unsigned int source = k + sourceLane;
                const double dx = source < count ? tileX[source] - x : 0.0;
                const double dy = source < count ? tileY[source] - y : 0.0;
                const double dz = source < count ? tileZ[source] - z : 0.0;
                const double distSqr = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                if constexpr (THREADS_PER_BODY == 1) {
                    fx = fma(dx, invDist3, fx);
                    fy = fma(dy, invDist3, fy);
                    fz = fma(dz, invDist3, fz);
                } else {
                    // Parallelize the expensive distance calculations, then
                    // accumulate in source order to retain reproducibility.
                    const unsigned int remaining = min(THREADS_PER_BODY, count - k);
                    #pragma unroll
                    for (unsigned int s = 0; s < remaining; ++s) {
                        const double sx = __shfl_sync(activeMask, dx, s, THREADS_PER_BODY);
                        const double sy = __shfl_sync(activeMask, dy, s, THREADS_PER_BODY);
                        const double sz = __shfl_sync(activeMask, dz, s, THREADS_PER_BODY);
                        const double inverse = __shfl_sync(activeMask, invDist3, s, THREADS_PER_BODY);
                        if (sourceLane == 0) {
                            fx = fma(sx, inverse, fx);
                            fy = fma(sy, inverse, fy);
                            fz = fma(sz, inverse, fz);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (active && sourceLane == 0) {
        const double vx = velocities[i] + DT * fx;
        const double vy = velocities[stride + i] + DT * fy;
        const double vz = velocities[2 * stride + i] + DT * fz;
        velocities[i] = vx;
        velocities[stride + i] = vy;
        velocities[2 * stride + i] = vz;
        nextPositions[i] = fma(vx, DT, x);
        nextPositions[stride + i] = fma(vy, DT, y);
        nextPositions[2 * stride + i] = fma(vz, DT, z);
    }
}

__device__ double blockSum(double value) {
    __shared__ double warpSums[BLOCK_SIZE / 32];
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (threadIdx.x % 32 == 0) {
        warpSums[threadIdx.x / 32] = value;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < BLOCK_SIZE / 32 ? warpSums[threadIdx.x] : 0.0;
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
    }
    return value;
}

__global__ void energyPartials(const double* __restrict__ positions,
                               const double* __restrict__ velocities,
                               double* __restrict__ partials,
                               unsigned int n, size_t stride) {
    __shared__ double tileX[BLOCK_SIZE];
    __shared__ double tileY[BLOCK_SIZE];
    __shared__ double tileZ[BLOCK_SIZE];
    const unsigned int lane = threadIdx.x;
    const unsigned int i = blockIdx.x * BLOCK_SIZE + lane;
    const bool active = i < n;
    const double x = active ? positions[i] : 0.0;
    const double y = active ? positions[stride + i] : 0.0;
    const double z = active ? positions[2 * stride + i] : 0.0;
    double energy = 0.0;
    if (active) {
        const double vx = velocities[i];
        const double vy = velocities[stride + i];
        const double vz = velocities[2 * stride + i];
        energy = 0.5 * (vx * vx + vy * vy + vz * vz);
    }

    // Each unordered pair contributes once, just as in the original code.
    for (unsigned int base = blockIdx.x * BLOCK_SIZE; base < n; base += BLOCK_SIZE) {
        const unsigned int j = base + lane;
        tileX[lane] = j < n ? positions[j] : 0.0;
        tileY[lane] = j < n ? positions[stride + j] : 0.0;
        tileZ[lane] = j < n ? positions[2 * stride + j] : 0.0;
        __syncthreads();
        const unsigned int count = min(BLOCK_SIZE, n - base);
        const unsigned int first = base <= i ? i - base + 1 : 0;
        if (active) {
            #pragma unroll 8
            for (unsigned int k = first; k < count; ++k) {
                const double dx = tileX[k] - x;
                const double dy = tileY[k] - y;
                const double dz = tileZ[k] - z;
                energy -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            }
        }
        __syncthreads();
    }
    energy = blockSum(energy);
    if (lane == 0) {
        partials[blockIdx.x] = energy;
    }
}

__global__ void reduceEnergy(const double* partials, double* result, unsigned int count) {
    double energy = 0.0;
    for (unsigned int i = threadIdx.x; i < count; i += BLOCK_SIZE) {
        energy += partials[i];
    }
    energy = blockSum(energy);
    if (threadIdx.x == 0) {
        *result = energy;
    }
}

class CudaSimulation {
public:
    CudaSimulation(const std::vector<Body>& bodies, int steps)
        : n(static_cast<unsigned int>(bodies.size())),
          blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE),
          stride(static_cast<size_t>(blocks) * BLOCK_SIZE), totalSteps(steps) {
        // Even an empty simulation requires CUDA; there is no CPU fallback.
        checkCuda(cudaFree(nullptr), "initializing CUDA");
        int device;
        cudaDeviceProp properties;
        checkCuda(cudaGetDevice(&device), "selecting CUDA device");
        checkCuda(cudaGetDeviceProperties(&properties, device), "querying CUDA device");
        if (n == 0) {
            return;
        }
        const unsigned int targetCount = static_cast<unsigned int>(properties.multiProcessorCount) * 32;
        threadsPerBody = n < targetCount / 2 ? 8 : n < targetCount ? 4 : n < targetCount * 2 ? 2 : 1;
        forceBlocks = (n + BLOCK_SIZE / threadsPerBody - 1) / (BLOCK_SIZE / threadsPerBody);
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "creating simulation stream");
        checkCuda(cudaMalloc(&storage, (9 * stride + blocks + 1) * sizeof(double)),
                  "allocating simulation state");
        positions = storage;
        nextPositions = storage + 3 * stride;
        velocities = storage + 6 * stride;
        partials = storage + 9 * stride;
        energyResult = partials + blocks;

        // Structure-of-arrays storage provides coalesced global memory access.
        std::vector<double> packed(6 * stride);
        for (unsigned int i = 0; i < n; ++i) {
            packed[i] = bodies[i].pos.x;
            packed[stride + i] = bodies[i].pos.y;
            packed[2 * stride + i] = bodies[i].pos.z;
            packed[3 * stride + i] = bodies[i].vel.x;
            packed[4 * stride + i] = bodies[i].vel.y;
            packed[5 * stride + i] = bodies[i].vel.z;
        }
        checkCuda(cudaMemcpy(positions, packed.data(), 3 * stride * sizeof(double),
                             cudaMemcpyHostToDevice), "uploading positions");
        checkCuda(cudaMemcpy(velocities, packed.data() + 3 * stride,
                             3 * stride * sizeof(double), cudaMemcpyHostToDevice),
                  "uploading velocities");

        if (steps > 0) {
            // Bounded graph size keeps setup/storage independent of the number
            // of timesteps. An even batch restores the input/output buffer
            // assignment, so the same graph can advance successive batches.
            graphSteps = std::min(steps, 64);
            checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                      "capturing simulation graph");
            for (int step = 0; step < graphSteps; ++step) {
                enqueueStep();
            }
            checkCuda(cudaGetLastError(), "capturing simulation kernels");
            checkCuda(cudaStreamEndCapture(stream, &graph), "finishing simulation graph");
            checkCuda(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0),
                      "instantiating simulation graph");
            checkCuda(cudaGraphUpload(graphExec, stream), "uploading simulation graph");
            checkCuda(cudaStreamSynchronize(stream), "preparing simulation graph");
        }
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    ~CudaSimulation() {
        if (graphExec) {
            cudaGraphExecDestroy(graphExec);
        }
        if (graph) {
            cudaGraphDestroy(graph);
        }
        if (stream) {
            cudaStreamDestroy(stream);
        }
        if (storage) {
            cudaFree(storage);
        }
    }

    void run() {
        if (graphSteps != 0) {
            for (int batch = 0; batch < totalSteps / graphSteps; ++batch) {
                checkCuda(cudaGraphLaunch(graphExec, stream), "launching simulation graph");
            }
            for (int step = 0; step < totalSteps % graphSteps; ++step) {
                enqueueStep();
            }
            checkCuda(cudaGetLastError(), "launching simulation kernels");
        }
        checkCuda(cudaStreamSynchronize(stream), "completing simulation");
    }

    void download(std::vector<Body>& bodies) const {
        if (n == 0) {
            return;
        }
        std::vector<double> packed(6 * stride);
        checkCuda(cudaMemcpy(packed.data(), positions, 3 * stride * sizeof(double),
                             cudaMemcpyDeviceToHost), "downloading positions");
        checkCuda(cudaMemcpy(packed.data() + 3 * stride, velocities,
                             3 * stride * sizeof(double), cudaMemcpyDeviceToHost),
                  "downloading velocities");
        for (unsigned int i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(packed[i], packed[stride + i], packed[2 * stride + i]);
            bodies[i].vel = Vec3(packed[3 * stride + i], packed[4 * stride + i], packed[5 * stride + i]);
        }
    }

    double totalEnergy() const {
        if (n == 0) {
            return 0.0;
        }
        energyPartials<<<blocks, BLOCK_SIZE>>>(positions, velocities, partials, n, stride);
        reduceEnergy<<<1, BLOCK_SIZE>>>(partials, energyResult, blocks);
        checkCuda(cudaGetLastError(), "launching energy kernels");
        double energy;
        checkCuda(cudaMemcpy(&energy, energyResult, sizeof(double), cudaMemcpyDeviceToHost),
                  "downloading total energy");
        return energy;
    }

private:
    void enqueueStep() {
        switch (threadsPerBody) {
            case 8:
                advanceBodies<8><<<forceBlocks, BLOCK_SIZE, 0, stream>>>(positions, nextPositions, velocities, n, stride);
                break;
            case 4:
                advanceBodies<4><<<forceBlocks, BLOCK_SIZE, 0, stream>>>(positions, nextPositions, velocities, n, stride);
                break;
            case 2:
                advanceBodies<2><<<forceBlocks, BLOCK_SIZE, 0, stream>>>(positions, nextPositions, velocities, n, stride);
                break;
            default:
                advanceBodies<1><<<forceBlocks, BLOCK_SIZE, 0, stream>>>(positions, nextPositions, velocities, n, stride);
                break;
        }
        std::swap(positions, nextPositions);
    }

    unsigned int n, blocks;
    size_t stride;
    int totalSteps;
    int graphSteps = 0;
    unsigned int threadsPerBody = 1, forceBlocks = 0;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    double* storage = nullptr;
    double* positions = nullptr;
    double* nextPositions = nullptr;
    double* velocities = nullptr;
    double* partials = nullptr;
    double* energyResult = nullptr;
};

} // namespace

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
    
    simulation.run();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    if (printResults || validate) {
        simulation.download(bodies);
    }
    
    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
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
