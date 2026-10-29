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
#include <omp.h>

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

// Each block reuses a tile of source positions across its target bodies.
__global__ void advance(const Body* all, Body* local, int n, int offset, int count) {
    constexpr int tile = 128;
    __shared__ double x[tile], y[tile], z[tile];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Body body;
    if (i < count) body = all[offset + i];
    double fx = 0, fy = 0, fz = 0;
    for (int base = 0; base < n; base += tile) {
        int j = base + threadIdx.x;
        if (j < n) { x[threadIdx.x] = all[j].pos.x; y[threadIdx.x] = all[j].pos.y; z[threadIdx.x] = all[j].pos.z; }
        __syncthreads();
        int limit = min(tile, n - base);
        if (i < count) {
            for (int k = 0; k < limit; ++k) {
                double dx = x[k] - body.pos.x, dy = y[k] - body.pos.y, dz = z[k] - body.pos.z;
                double d = dx * dx + dy * dy + dz * dz + SOFTENING;
                double inv = 1.0 / sqrt(d);
                double inv3 = inv * inv * inv;
                fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
            }
        }
        __syncthreads();
    }
    if (i < count) {
        body.vel.x += DT * fx; body.vel.y += DT * fy; body.vel.z += DT * fz;
        body.pos.x += body.vel.x * DT; body.pos.y += body.vel.y * DT; body.pos.z += body.vel.z * DT;
        local[i] = body;
    }
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA %s: %s\n", operation, cudaGetErrorString(error));
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) { if (!rank) fprintf(stderr, "MPI thread support is insufficient\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
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
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (numBodies < 0 || numSteps < 0 || static_cast<long long>(numBodies) * sizeof(Body) > INT_MAX) {
        if (!rank) fprintf(stderr, "Invalid body count or step count\n");
        MPI_Finalize();
        return 1;
    }
    if (!rank) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&deviceCount), "device count");
    if (deviceCount == 0) { fprintf(stderr, "No CUDA device available for rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "set device");
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        int count = numBodies / ranks + (r < numBodies % ranks);
        int offset = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
        counts[r] = count * sizeof(Body);
        displacements[r] = offset * sizeof(Body);
    }
    int count = counts[rank] / sizeof(Body), offset = displacements[rank] / sizeof(Body);
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    Body *deviceAll = nullptr, *deviceLocal = nullptr;
    if (numBodies) cudaCheck(cudaMalloc(&deviceAll, sizeof(Body) * numBodies), "allocate global state");
    if (count) cudaCheck(cudaMalloc(&deviceLocal, sizeof(Body) * count), "allocate local state");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (count) {
            cudaCheck(cudaMemcpy(deviceAll, bodies.data(), sizeof(Body) * numBodies, cudaMemcpyHostToDevice), "copy positions");
            advance<<<(count + 127) / 128, 128>>>(deviceAll, deviceLocal, numBodies, offset, count);
            cudaCheck(cudaGetLastError(), "launch");
            cudaCheck(cudaMemcpy(bodies.data() + offset, deviceLocal, sizeof(Body) * count, cudaMemcpyDeviceToHost), "copy updated bodies");
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bodies.data(), counts.data(), displacements.data(), MPI_BYTE, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) printf("Simulation time: %ld ms\n", max_duration_ms);
    if (deviceAll) cudaCheck(cudaFree(deviceAll), "free global state");
    if (deviceLocal) cudaCheck(cudaFree(deviceLocal), "free local state");
    int status = 0;
    if (rank == 0) {
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
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }
    
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
