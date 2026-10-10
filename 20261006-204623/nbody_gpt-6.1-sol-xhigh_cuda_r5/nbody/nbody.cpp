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

// Each thread owns one target body. All threads, including those past n in
// the last block, participate in loading tiles and in both barriers.
template <int TileSize>
__global__ void advanceBodies(const double* __restrict__ positions,
                              double* __restrict__ nextPositions,
                              double* __restrict__ velocities,
                              size_t n, size_t stride) {
    __shared__ double tileX[TileSize];
    __shared__ double tileY[TileSize];
    __shared__ double tileZ[TileSize];
    const size_t i = static_cast<size_t>(blockIdx.x) * TileSize + threadIdx.x;
    const bool active = i < n;
    const double x = active ? positions[i] : 0.0;
    const double y = active ? positions[stride + i] : 0.0;
    const double z = active ? positions[2 * stride + i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (size_t base = 0; base < n; base += TileSize) {
        const size_t j = base + threadIdx.x;
        tileX[threadIdx.x] = j < n ? positions[j] : 0.0;
        tileY[threadIdx.x] = j < n ? positions[stride + j] : 0.0;
        tileZ[threadIdx.x] = j < n ? positions[2 * stride + j] : 0.0;
        __syncthreads();

        const int count = n - base < TileSize ? static_cast<int>(n - base) : TileSize;
        if (active) {
            // Keep source bodies in their original order, including self
            // interactions. No approximate reciprocal square root is used.
            #pragma unroll 4
            for (int k = 0; k < count; ++k) {
                const double dx = tileX[k] - x;
                const double dy = tileY[k] - y;
                const double dz = tileZ[k] - z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        const double vx = velocities[i] + DT * fx;
        const double vy = velocities[stride + i] + DT * fy;
        const double vz = velocities[2 * stride + i] + DT * fz;
        velocities[i] = vx;
        velocities[stride + i] = vy;
        velocities[2 * stride + i] = vz;
        // Writing a separate position buffer allows integration to be fused
        // with force calculation without changing positions read by any block.
        nextPositions[i] = x + vx * DT;
        nextPositions[stride + i] = y + vy * DT;
        nextPositions[2 * stride + i] = z + vz * DT;
    }
}

class CudaSimulation {
public:
    explicit CudaSimulation(const std::vector<Body>& bodies)
        : n(bodies.size()), stride((n + 127) / 128 * 128), hostState(6 * stride) {
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        if (n == 0) return;
        configureLaunch();
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&storage), 9 * stride * sizeof(double)),
                  "allocate body state");
        positions = storage;
        velocities = storage + 3 * stride;
        nextPositions = storage + 6 * stride;
        for (size_t i = 0; i < n; ++i) {
            hostState[i] = bodies[i].pos.x;
            hostState[stride + i] = bodies[i].pos.y;
            hostState[2 * stride + i] = bodies[i].pos.z;
            hostState[3 * stride + i] = bodies[i].vel.x;
            hostState[4 * stride + i] = bodies[i].vel.y;
            hostState[5 * stride + i] = bodies[i].vel.z;
        }
        checkCuda(cudaMemcpyAsync(storage, hostState.data(), hostState.size() * sizeof(double),
                                  cudaMemcpyHostToDevice, stream), "upload body state");
        checkCuda(cudaStreamSynchronize(stream), "finish upload");
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    ~CudaSimulation() {
        if (graphExec) checkCuda(cudaGraphExecDestroy(graphExec), "destroy simulation graph");
        if (storage) checkCuda(cudaFree(storage), "free body state");
        checkCuda(cudaStreamDestroy(stream), "destroy stream");
    }

    void prepare(int steps) {
        if (n == 0 || steps < GraphSteps) return;
        // An even-sized graph returns to the same input buffer, so it can be
        // replayed without changing kernel arguments or synchronizing the CPU.
        cudaGraph_t graph;
        checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture simulation");
        for (int step = 0; step < GraphSteps; ++step) launchStep();
        checkCuda(cudaStreamEndCapture(stream, &graph), "finish simulation capture");
        checkCuda(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), "instantiate simulation graph");
        checkCuda(cudaGraphDestroy(graph), "destroy captured graph");
    }

    void run(int steps) {
        if (n == 0) return;
        int step = 0;
        if (graphExec) {
            for (; steps - step >= GraphSteps; step += GraphSteps) {
                checkCuda(cudaGraphLaunch(graphExec, stream), "launch simulation graph");
            }
        }
        for (; step < steps; ++step) launchStep();
        // Include GPU execution, not just asynchronous launches, in timing.
        checkCuda(cudaStreamSynchronize(stream), "finish simulation");
    }

    void download(std::vector<Body>& bodies) {
        if (n == 0) return;
        checkCuda(cudaMemcpyAsync(hostState.data(), positions, 3 * stride * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream), "download positions");
        checkCuda(cudaMemcpyAsync(hostState.data() + 3 * stride, velocities, 3 * stride * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream), "download velocities");
        checkCuda(cudaStreamSynchronize(stream), "finish download");
        for (size_t i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(hostState[i], hostState[stride + i], hostState[2 * stride + i]);
            bodies[i].vel = Vec3(hostState[3 * stride + i], hostState[4 * stride + i], hostState[5 * stride + i]);
        }
    }

private:
    void configureLaunch() {
        int device, multiprocessors, blocks32, blocks64;
        checkCuda(cudaGetDevice(&device), "get CUDA device");
        checkCuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device),
                  "get multiprocessor count");
        checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks32, advanceBodies<32>, 32, 0),
                  "get small-tile occupancy");
        checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks64, advanceBodies<64>, 64, 0),
                  "get medium-tile occupancy");
        // Small tiles spread modest inputs across more multiprocessors. Grow
        // the tile when the grid exceeds their resident-thread capacity, to
        // improve occupancy and reuse on larger inputs.
        const size_t capacity32 = static_cast<size_t>(multiprocessors) * blocks32 * 32;
        const size_t capacity64 = static_cast<size_t>(multiprocessors) * blocks64 * 64;
        tileSize = n <= capacity32 ? 32 : (n <= capacity64 ? 64 : 128);
    }

    void launchStep() {
        const unsigned int blocks = static_cast<unsigned int>((n + tileSize - 1) / tileSize);
        if (tileSize == 32) {
            advanceBodies<32><<<blocks, 32, 0, stream>>>(positions, nextPositions, velocities, n, stride);
        } else if (tileSize == 64) {
            advanceBodies<64><<<blocks, 64, 0, stream>>>(positions, nextPositions, velocities, n, stride);
        } else {
            advanceBodies<128><<<blocks, 128, 0, stream>>>(positions, nextPositions, velocities, n, stride);
        }
        checkCuda(cudaGetLastError(), "launch body update");
        std::swap(positions, nextPositions);
    }

    static constexpr int GraphSteps = 16;
    size_t n, stride;
    int tileSize = 32;
    std::vector<double> hostState;
    double* storage = nullptr;
    double* positions = nullptr;
    double* nextPositions = nullptr;
    double* velocities = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t graphExec = nullptr;
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
    
    if (numBodies < 0) {
        fprintf(stderr, "Number of bodies must be nonnegative\n");
        return 1;
    }

    // Initialize bodies on the host to preserve the original random sequence.
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    CudaSimulation simulation(bodies);
    simulation.prepare(numSteps);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    simulation.run(numSteps);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (printResults || validate) simulation.download(bodies);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
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
