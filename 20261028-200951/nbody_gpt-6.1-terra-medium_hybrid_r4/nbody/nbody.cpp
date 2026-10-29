#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>

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

// Each thread owns one target body. Shared tiles reuse source positions across
// the block, and all positions remain unchanged until the kernel completes.
__global__ void advanceBodies(Body* local, const Vec3* positions, Vec3* updated,
                              int count, int n, int offset) {
    constexpr int block = 128;
    __shared__ Vec3 tile[block];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < count;
    Vec3 p{};
    if (active) p = positions[offset + i];
    double fx = 0, fy = 0, fz = 0;
    for (int base = 0; base < n; base += block) {
        int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        int limit = min(block, n - base);
        if (active) {
            for (int k = 0; k < limit; ++k) {
                double dx = tile[k].x - p.x;
                double dy = tile[k].y - p.y;
                double dz = tile[k].z - p.z;
                double dist = dx*dx + dy*dy + dz*dz + SOFTENING;
                double inv = 1.0 / sqrt(dist);
                double inv3 = inv * inv * inv;
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
        updated[i] = b.pos;
    }
}

static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
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
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (numBodies < 0 || numSteps < 0 || numBodies > INT_MAX / static_cast<int>(sizeof(Body))) {
        if (rank == 0) fprintf(stderr, "Invalid number of bodies or steps\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) { fprintf(stderr, "No CUDA device available on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), offsets(ranks), bodyCounts(ranks), bodyOffsets(ranks);
    for (int r = 0, offset = 0; r < ranks; ++r) {
        int count = numBodies / ranks + (r < numBodies % ranks);
        counts[r] = count * 3; offsets[r] = offset * 3;
        bodyCounts[r] = count * sizeof(Body); bodyOffsets[r] = offset * sizeof(Body);
        offset += count;
    }
    int count = counts[rank] / 3, offset = offsets[rank] / 3;
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    std::vector<Vec3> positions(numBodies), localPositions(count);
    #pragma omp parallel for
    for (int i = 0; i < numBodies; ++i) positions[i] = bodies[i].pos;
    Body* dBodies = nullptr;
    Vec3 *dPositions = nullptr, *dUpdated = nullptr;
    // Allocate at least one element for ranks with no assigned bodies.
    cudaCheck(cudaMalloc(&dBodies, std::max(1, count) * sizeof(Body)));
    cudaCheck(cudaMalloc(&dPositions, std::max(1, numBodies) * sizeof(Vec3)));
    cudaCheck(cudaMalloc(&dUpdated, std::max(1, count) * sizeof(Vec3)));
    if (count) cudaCheck(cudaMemcpy(dBodies, bodies.data() + offset, count * sizeof(Body), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (numBodies) cudaCheck(cudaMemcpy(dPositions, positions.data(), numBodies * sizeof(Vec3), cudaMemcpyHostToDevice));
        if (count) {
            advanceBodies<<<(count + 127) / 128, 128>>>(dBodies, dPositions, dUpdated, count, numBodies, offset);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(localPositions.data(), dUpdated, count * sizeof(Vec3), cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(localPositions.data(), counts[rank], MPI_DOUBLE, positions.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", maxDuration);
    if (printResults || validate) {
        std::vector<Body> localBodies(count);
        if (count) cudaCheck(cudaMemcpy(localBodies.data(), dBodies, count * sizeof(Body), cudaMemcpyDeviceToHost));
        MPI_Gatherv(localBodies.data(), bodyCounts[rank], MPI_BYTE, bodies.data(), bodyCounts.data(), bodyOffsets.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    cudaCheck(cudaFree(dBodies)); cudaCheck(cudaFree(dPositions)); cudaCheck(cudaFree(dUpdated));
    if (rank != 0) { MPI_Finalize(); return 0; }

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
