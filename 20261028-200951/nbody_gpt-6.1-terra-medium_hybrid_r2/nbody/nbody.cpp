#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <climits>
#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
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

// Each thread owns one target body; positions remain unchanged until the
// entire force kernel has completed. Shared tiles reduce global memory traffic.
__global__ void advanceBodies(Body* local, const Vec3* positions, int n, int offset, int count) {
    extern __shared__ Vec3 tile[];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < count;
    Vec3 p;
    if (active) p = positions[offset + i];
    double fx = 0, fy = 0, fz = 0;
    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        const int limit = min((int)blockDim.x, n - base);
        if (active) {
            for (int k = 0; k < limit; ++k) {
                const double dx = tile[k].x - p.x;
                const double dy = tile[k].y - p.y;
                const double dz = tile[k].z - p.z;
                const double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                const double inv3 = inv * inv * inv;
                fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
            }
        }
        __syncthreads();
    }
    if (active) {
        Body b = local[i];
        b.vel.x += DT * fx; b.vel.y += DT * fy; b.vel.z += DT * fz;
        b.pos.x += DT * b.vel.x; b.pos.y += DT * b.vel.y; b.pos.z += DT * b.vel.z;
        local[i] = b;
    }
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    #pragma omp parallel for reduction(+:energy)
    for (size_t i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    #pragma omp parallel for reduction(+:energy) schedule(dynamic, 16)
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
    
    if (numBodies < 0 || numSteps < 0) {
        fprintf(stderr, "Number of bodies and steps must be non-negative\n");
        return 1;
    }
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA device available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
               numBodies, numSteps, validate ? "enabled" : "disabled");
    }
    // All ranks initialize identically, preserving the original random sequence.
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    std::vector<int> counts(ranks), offsets(ranks), byteCounts(ranks), byteOffsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
        // MPI's count arguments are in bytes for these plain data structures.
        if ((size_t)counts[r] * sizeof(Body) > INT_MAX || (size_t)offsets[r] * sizeof(Body) > INT_MAX) {
            if (rank == 0) fprintf(stderr, "Problem size exceeds MPI count limits\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        byteCounts[r] = counts[r] * sizeof(Vec3);
        byteOffsets[r] = offsets[r] * sizeof(Vec3);
    }
    const int count = counts[rank], offset = offsets[rank];
    std::vector<Vec3> positions(numBodies), localPositions(count);
    for (int i = 0; i < numBodies; ++i) positions[i] = bodies[i].pos;
    Body* deviceBodies = nullptr;
    Vec3* devicePositions = nullptr;
    if (count) {
        cudaCheck(cudaMalloc(&deviceBodies, (size_t)count * sizeof(Body)), "allocate bodies");
        cudaCheck(cudaMalloc(&devicePositions, (size_t)numBodies * sizeof(Vec3)), "allocate positions");
        cudaCheck(cudaMemcpy(deviceBodies, bodies.data() + offset, (size_t)count * sizeof(Body), cudaMemcpyHostToDevice), "copy bodies");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (count) {
            cudaCheck(cudaMemcpy(devicePositions, positions.data(), (size_t)numBodies * sizeof(Vec3), cudaMemcpyHostToDevice), "copy positions");
            constexpr int block = 256;
            advanceBodies<<<(count + block - 1) / block, block, block * sizeof(Vec3)>>>(deviceBodies, devicePositions, numBodies, offset, count);
            cudaCheck(cudaGetLastError(), "launch force kernel");
            cudaCheck(cudaMemcpy(bodies.data() + offset, deviceBodies, (size_t)count * sizeof(Body), cudaMemcpyDeviceToHost), "copy updated bodies");
            #pragma omp parallel for
            for (int i = 0; i < count; ++i) localPositions[i] = bodies[offset + i].pos;
        }
        if (step + 1 < numSteps)
            MPI_Allgatherv(localPositions.data(), byteCounts[rank], MPI_BYTE, positions.data(), byteCounts.data(), byteOffsets.data(), MPI_BYTE, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count(), max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    for (int r = 0; r < ranks; ++r) {
        byteCounts[r] = counts[r] * sizeof(Body);
        byteOffsets[r] = offsets[r] * sizeof(Body);
    }
    // The local segment is already in its receive location on the root.
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : (count ? bodies.data() + offset : nullptr), byteCounts[rank], MPI_BYTE,
                bodies.data(), byteCounts.data(), byteOffsets.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (deviceBodies) cudaCheck(cudaFree(deviceBodies), "free bodies");
    if (devicePositions) cudaCheck(cudaFree(devicePositions), "free positions");
    if (rank != 0) { MPI_Finalize(); return 0; }
    printf("Simulation time: %ld ms\n", max_ms);

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
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
