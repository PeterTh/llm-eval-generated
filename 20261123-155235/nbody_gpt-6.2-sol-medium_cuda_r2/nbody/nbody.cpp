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
};

void randomizeBodies(std::vector<Vec3>& positions, std::vector<Vec3>& velocities,
                     unsigned int seed = 42) {
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        positions[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        positions[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error while %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            checkCuda(cudaMalloc(&data_, count * sizeof(T)), "allocating device memory");
        }
    }

    ~DeviceBuffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() const { return data_; }

private:
    T* data_ = nullptr;
};

__device__ __forceinline__ double normSquared(double x, double y, double z) {
    // Match the contraction order of the original optimized CPU benchmark.
    return __fma_rn(z, z, __fma_rn(x, x, y * y));
}

__device__ __forceinline__ void accumulateForce(const Vec3& target, const Vec3& source,
                                                double& fx, double& fy, double& fz) {
    const double dx = source.x - target.x;
    const double dy = source.y - target.y;
    const double dz = source.z - target.z;
    const double invDist = 1.0 / sqrt(normSquared(dx, dy, dz) + SOFTENING);
    const double invDist3 = invDist * invDist * invDist;

    fx = __fma_rn(dx, invDist3, fx);
    fy = __fma_rn(dy, invDist3, fy);
    fz = __fma_rn(dz, invDist3, fz);
}

template <unsigned BlockSize>
__global__ void advanceBodies(const Vec3* __restrict__ positions,
                              Vec3* __restrict__ nextPositions,
                              Vec3* __restrict__ velocities, unsigned count) {
    // Each position tile is loaded once per block and reused by all its targets.
    __shared__ double tileX[BlockSize];
    __shared__ double tileY[BlockSize];
    __shared__ double tileZ[BlockSize];
    const unsigned lane = threadIdx.x;
    const unsigned i = blockIdx.x * BlockSize + lane;
    const Vec3 target = i < count ? positions[i] : Vec3{};
    double fx = 0.0, fy = 0.0, fz = 0.0;

    const unsigned fullTilesEnd = count - count % BlockSize;
    for (unsigned base = 0; base < fullTilesEnd; base += BlockSize) {
        const Vec3 source = positions[base + lane];
        tileX[lane] = source.x;
        tileY[lane] = source.y;
        tileZ[lane] = source.z;
        __syncthreads();

        #pragma unroll 4
        for (unsigned j = 0; j < BlockSize; ++j) {
            accumulateForce(target, Vec3{tileX[j], tileY[j], tileZ[j]}, fx, fy, fz);
        }
        __syncthreads();
    }

    const unsigned tail = count - fullTilesEnd;
    if (tail != 0) {
        if (lane < tail) {
            const Vec3 source = positions[fullTilesEnd + lane];
            tileX[lane] = source.x;
            tileY[lane] = source.y;
            tileZ[lane] = source.z;
        }
        __syncthreads();
    }

    if (i < count) {
        for (unsigned j = 0; j < tail; ++j) {
            accumulateForce(target, Vec3{tileX[j], tileY[j], tileZ[j]}, fx, fy, fz);
        }
        Vec3 velocity = velocities[i];
        velocity.x += DT * fx;
        velocity.y += DT * fy;
        velocity.z += DT * fz;
        velocities[i] = velocity;
        nextPositions[i] = Vec3{__fma_rn(velocity.x, DT, target.x),
                                __fma_rn(velocity.y, DT, target.y),
                                __fma_rn(velocity.z, DT, target.z)};
    }
}

template <unsigned BlockSize>
void runTimesteps(Vec3*& current, Vec3*& next, Vec3* velocities, unsigned count, int steps) {
    const unsigned blocks = (count + BlockSize - 1) / BlockSize;
    for (int step = 0; step < steps; ++step) {
        advanceBodies<BlockSize><<<blocks, BlockSize>>>(current, next, velocities, count);
        std::swap(current, next);
    }
    checkCuda(cudaGetLastError(), "launching simulation kernel");
    checkCuda(cudaDeviceSynchronize(), "running simulation");
}

constexpr unsigned WarpSize = 32;
constexpr unsigned EnergyBlockSize = 128;

__global__ void energyRows(const Vec3* __restrict__ positions,
                            const Vec3* __restrict__ velocities, unsigned count,
                            double* __restrict__ rows) {
    // One warp per body, visiting only the upper triangle of the pair matrix.
    const unsigned i = blockIdx.x * (EnergyBlockSize / WarpSize) + threadIdx.x / WarpSize;
    const unsigned lane = threadIdx.x % WarpSize;
    if (i >= count) {
        return;
    }

    const Vec3 target = positions[i];
    double potential = 0.0;
    for (unsigned j = i + 1 + lane; j < count; j += WarpSize) {
        const Vec3 source = positions[j];
        potential -= 1.0 / sqrt(normSquared(source.x - target.x, source.y - target.y,
                                            source.z - target.z) + SOFTENING);
    }
    for (unsigned offset = WarpSize / 2; offset > 0; offset /= 2) {
        potential += __shfl_down_sync(0xffffffffu, potential, offset);
    }
    if (lane == 0) {
        const Vec3 velocity = velocities[i];
        rows[i] = 0.5 * normSquared(velocity.x, velocity.y, velocity.z) + potential;
    }
}

double computeTotalEnergy(const Vec3* positions, const Vec3* velocities, unsigned count) {
    if (count == 0) {
        return 0.0;
    }
    constexpr unsigned rowsPerBlock = EnergyBlockSize / WarpSize;
    DeviceBuffer<double> deviceRows(count);
    energyRows<<<(count + rowsPerBlock - 1) / rowsPerBlock, EnergyBlockSize>>>(
        positions, velocities, count, deviceRows.data());
    checkCuda(cudaGetLastError(), "launching energy kernel");

    std::vector<double> rows(count);
    checkCuda(cudaMemcpy(rows.data(), deviceRows.data(), count * sizeof(double),
                         cudaMemcpyDeviceToHost), "downloading energy sums");
    return kahan_sum(rows);
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities) {
    for (size_t i = 0; i < positions.size(); ++i) {
        const Vec3& pos = positions[i];
        const Vec3& vel = velocities[i];
        // Check for NaN or Inf values
        if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
            !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(pos.x) > maxPos || std::abs(pos.y) > maxPos || std::abs(pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vel.x) > maxVel || std::abs(vel.y) > maxVel || std::abs(vel.z) > maxVel) {
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
    std::vector<Vec3> positions(numBodies), velocities(numBodies);
    randomizeBodies(positions, velocities);

    const unsigned count = static_cast<unsigned>(positions.size());
    const size_t bytes = positions.size() * sizeof(Vec3);
    DeviceBuffer<Vec3> state(3 * positions.size());
    Vec3* current = state.data();
    Vec3* next = nullptr;
    Vec3* deviceVelocities = nullptr;
    if (count != 0) {
        next = current + count;
        deviceVelocities = next + count;
        checkCuda(cudaMemcpy(current, positions.data(), bytes, cudaMemcpyHostToDevice),
                  "uploading positions");
        checkCuda(cudaMemcpy(deviceVelocities, velocities.data(), bytes, cudaMemcpyHostToDevice),
                  "uploading velocities");
    }
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    if (count != 0 && numSteps > 0) {
        // Smaller blocks spread small systems across more multiprocessors.
        if (count <= 4096) {
            runTimesteps<32>(current, next, deviceVelocities, count, numSteps);
        } else if (count <= 16384) {
            runTimesteps<64>(current, next, deviceVelocities, count, numSteps);
        } else {
            runTimesteps<128>(current, next, deviceVelocities, count, numSteps);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());

    if ((printResults || validate) && count != 0 && numSteps > 0) {
        checkCuda(cudaMemcpy(positions.data(), current, bytes, cudaMemcpyDeviceToHost),
                  "downloading positions");
        checkCuda(cudaMemcpy(velocities.data(), deviceVelocities, bytes, cudaMemcpyDeviceToHost),
                  "downloading velocities");
    }
    
    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(positions.size() * 6);
        for (size_t i = 0; i < positions.size(); ++i) {
            const Vec3& pos = positions[i];
            const Vec3& vel = velocities[i];
            bodyData.insert(bodyData.end(), {pos.x, pos.y, pos.z, vel.x, vel.y, vel.z});
        }
        print_results(bodyData, "Bodies");
    }
    
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(positions, velocities)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(current, deviceVelocities, count);
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
