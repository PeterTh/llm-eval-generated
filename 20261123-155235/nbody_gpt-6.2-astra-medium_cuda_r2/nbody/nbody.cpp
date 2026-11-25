#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
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
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

constexpr unsigned TILE = 32;

__device__ double inverseCube(double dx, double dy, double dz) {
    // Match the rounded square root and division used by the reference.
    const double xy = fma(dx, dx, __dmul_rn(dy, dy));
    const double distSqr = fma(dz, dz, xy) + SOFTENING;
    const double invDist = __ddiv_rn(1.0, __dsqrt_rn(distSqr));
    return invDist * invDist * invDist;
}

// For small systems, distribute the expensive distance calculations over the
// pair grid. Consecutive threads own consecutive targets in both kernels.
__global__ void computeWeights(const Vec3* positions, double* weights, unsigned n) {
    if (blockIdx.x > blockIdx.y) return;
    __shared__ double tile[TILE][TILE + 1];
    const unsigned x = blockIdx.x * TILE;
    const unsigned y = blockIdx.y * TILE;
    const unsigned i = x + threadIdx.x;
    const Vec3 p = i < n ? positions[i] : Vec3{};
    for (unsigned row = threadIdx.y; row < TILE; row += blockDim.y) {
        const unsigned j = y + row;
        if (i < n && j < n) {
            const Vec3 q = positions[j];
            const double w = inverseCube(q.x - p.x, q.y - p.y, q.z - p.z);
            weights[static_cast<size_t>(x) * n + j * TILE + threadIdx.x] = w;
            tile[row][threadIdx.x] = w;
        }
    }
    if (blockIdx.x == blockIdx.y) return;
    __syncthreads();
    // Transpose the tile so the symmetric weights also have coalesced stores.
    for (unsigned row = threadIdx.y; row < TILE; row += blockDim.y) {
        if (x + row < n && y + threadIdx.x < n)
            weights[static_cast<size_t>(y) * n + (x + row) * TILE + threadIdx.x] =
                tile[threadIdx.x][row];
    }
}

template<bool cached>
__global__ void advance(const Vec3* __restrict__ positions, Vec3* __restrict__ next,
                        Vec3* __restrict__ velocities, unsigned n, unsigned first,
                        const double* weights) {
    __shared__ Vec3 sources[TILE];
    const unsigned local = blockIdx.x * TILE + threadIdx.x;
    const unsigned i = first + local;
    const Vec3 p = i < n ? positions[i] : Vec3{};
    // Weights for a warp are contiguous, with one row of TILE targets per source.
    const double* warpWeights = cached ? weights + static_cast<size_t>(i - threadIdx.x) * n
                                       : nullptr;
    Vec3 force{};
    // One warp shares each source tile and sums its interactions in body order.
    for (unsigned base = 0; base < n; base += TILE) {
        const unsigned count = min(TILE, n - base);
        if (threadIdx.x < count) sources[threadIdx.x] = positions[base + threadIdx.x];
        __syncwarp();
        if (i < n) {
            #pragma unroll 8
            for (unsigned k = 0; k < count; ++k) {
                const double dx = sources[k].x - p.x;
                const double dy = sources[k].y - p.y;
                const double dz = sources[k].z - p.z;
                const double w = cached ? warpWeights[(base + k) * TILE + threadIdx.x]
                                        : inverseCube(dx, dy, dz);
                force.x = fma(dx, w, force.x);
                force.y = fma(dy, w, force.y);
                force.z = fma(dz, w, force.z);
            }
        }
        __syncwarp();
    }
    if (i < n) {
        Vec3 v = velocities[local];
        v.x += __dmul_rn(DT, force.x);
        v.y += __dmul_rn(DT, force.y);
        v.z += __dmul_rn(DT, force.z);
        velocities[local] = v;
        next[i] = {fma(DT, v.x, p.x), fma(DT, v.y, p.y), fma(DT, v.z, p.z)};
    }
}

struct Partition {
    int device;
    unsigned first, count, blocks;
    Vec3 *positions, *next, *velocities;
    cudaEvent_t advanced;
};

long simulate(std::vector<Body>& bodies, int steps, bool readBack) {
    const unsigned n = bodies.size();
    if (!n || steps <= 0) return 0;
    int available;
    checkCuda(cudaGetDeviceCount(&available), "Count CUDA devices");
    if (available == 0) {
        std::fprintf(stderr, "Simulation requires a CUDA device\n");
        std::exit(EXIT_FAILURE);
    }
    // Amortize exchanging positions by assigning at least 8192 targets per GPU.
    const unsigned devices = std::min<unsigned>(available, std::max(1u, n / 8192));
    const unsigned blocks = (n + TILE - 1) / TILE;
    const size_t bytes = n * sizeof(Vec3);
    std::vector<Vec3> positions(n), velocities(n);
    for (unsigned i = 0; i < n; ++i) {
        positions[i] = bodies[i].pos;
        velocities[i] = bodies[i].vel;
    }
    std::vector<Partition> partitions(devices);
    for (unsigned d = 0; d < devices; ++d) {
        Partition& part = partitions[d];
        part.device = d;
        const unsigned firstBlock = static_cast<size_t>(blocks) * d / devices;
        const unsigned endBlock = static_cast<size_t>(blocks) * (d + 1) / devices;
        part.first = TILE * firstBlock;
        part.blocks = endBlock - firstBlock;
        part.count = std::min(n, TILE * endBlock) - part.first;
        const size_t localBytes = part.count * sizeof(Vec3);
        checkCuda(cudaSetDevice(d), "Select CUDA device");
        checkCuda(cudaMalloc(&part.positions, bytes), "Allocate positions");
        checkCuda(cudaMalloc(&part.next, bytes), "Allocate next positions");
        checkCuda(cudaMalloc(&part.velocities, localBytes), "Allocate velocities");
        checkCuda(cudaMemcpy(part.positions, positions.data(), bytes, cudaMemcpyHostToDevice),
                  "Upload positions");
        checkCuda(cudaMemcpy(part.velocities, velocities.data() + part.first, localBytes,
                             cudaMemcpyHostToDevice), "Upload velocities");
        if (devices > 1)
            checkCuda(cudaEventCreateWithFlags(&part.advanced, cudaEventDisableTiming), "Create step event");
    }

    double* weights = nullptr;
    // Bound scratch memory. Larger systems compute weights directly from each tile.
    if (n <= 4096) {
        const size_t entries = static_cast<size_t>(blocks) * TILE * n;
        checkCuda(cudaMalloc(&weights, entries * sizeof(double)), "Allocate pair weights");
    }

    auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        for (auto& part : partitions) {
            if (devices > 1) checkCuda(cudaSetDevice(part.device), "Select CUDA device");
            if (weights) {
                const dim3 threads(TILE, 8);
                const dim3 grid(blocks, blocks);
                computeWeights<<<grid, threads>>>(part.positions, weights, n);
                checkCuda(cudaGetLastError(), "Launch pair weights");
                advance<true><<<part.blocks, TILE>>>(part.positions, part.next, part.velocities,
                                                      n, part.first, weights);
            } else {
                advance<false><<<part.blocks, TILE>>>(part.positions, part.next, part.velocities,
                                                       n, part.first, nullptr);
            }
            checkCuda(cudaGetLastError(), "Launch simulation step");
            if (devices > 1)
                checkCuda(cudaEventRecord(part.advanced), "Record completed partition");
        }

        // Each destination stream gathers the new positions before its next step.
        if (devices > 1 && step + 1 < steps) {
            for (const auto& dst : partitions) {
                checkCuda(cudaSetDevice(dst.device), "Select position destination");
                for (const auto& src : partitions) {
                    if (src.device == dst.device) continue;
                    checkCuda(cudaStreamWaitEvent(nullptr, src.advanced), "Wait for source positions");
                    checkCuda(cudaMemcpyPeerAsync(dst.next + src.first, dst.device,
                                                  src.next + src.first, src.device,
                                                  src.count * sizeof(Vec3)), "Exchange positions");
                }
            }
        }
        for (auto& part : partitions) std::swap(part.positions, part.next);
    }
    for (const auto& part : partitions) {
        checkCuda(cudaSetDevice(part.device), "Select CUDA device");
        checkCuda(cudaDeviceSynchronize(), "Complete simulation");
    }
    auto end = std::chrono::steady_clock::now();
    if (weights) checkCuda(cudaFree(weights), "Free pair weights");
    for (const auto& part : partitions) {
        checkCuda(cudaSetDevice(part.device), "Select CUDA device");
        if (readBack) {
            const size_t localBytes = part.count * sizeof(Vec3);
            checkCuda(cudaMemcpy(positions.data() + part.first, part.positions + part.first,
                                 localBytes, cudaMemcpyDeviceToHost), "Download positions");
            checkCuda(cudaMemcpy(velocities.data() + part.first, part.velocities,
                                 localBytes, cudaMemcpyDeviceToHost), "Download velocities");
        }
        if (devices > 1) checkCuda(cudaEventDestroy(part.advanced), "Free step event");
        checkCuda(cudaFree(part.velocities), "Free velocities");
        checkCuda(cudaFree(part.next), "Free next positions");
        checkCuda(cudaFree(part.positions), "Free positions");
    }
    if (readBack) {
        for (unsigned i = 0; i < n; ++i) bodies[i] = {positions[i], velocities[i]};
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
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
    
    // Run simulation
    const long elapsed = simulate(bodies, numSteps, validate || printResults);
    printf("Simulation time: %ld ms\n", elapsed);

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
