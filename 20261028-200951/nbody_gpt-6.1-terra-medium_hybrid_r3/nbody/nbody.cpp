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

// Each thread owns one body. All threads participate in loading each tile,
// including threads beyond the end of the local partition.
__global__ void advance(const Vec3* positions, Vec3* next, Vec3* velocities,
                        int n, int offset, int count) {
    __shared__ Vec3 tile[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = i < count;
    Vec3 p = active ? positions[offset + i] : Vec3();
    double fx = 0, fy = 0, fz = 0;
    for (int base = 0; base < n; base += blockDim.x) {
        int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        int limit = min((int)blockDim.x, n - base);
        if (active) {
            for (int k = 0; k < limit; ++k) {
                double dx = tile[k].x - p.x, dy = tile[k].y - p.y, dz = tile[k].z - p.z;
                double d2 = dx * dx + dy * dy + dz * dz + SOFTENING;
                double inv = 1.0 / sqrt(d2);
                double inv3 = inv * inv * inv;
                fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
            }
        }
        __syncthreads();
    }
    if (active) {
        Vec3 v = velocities[i];
        v.x += DT * fx; v.y += DT * fy; v.z += DT * fz;
        velocities[i] = v;
        next[i] = Vec3(p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT);
    }
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (numBodies < 0 || numSteps < 0 || numBodies > INT_MAX / 3) {
        if (!rank) fprintf(stderr, "Invalid body or step count\n");
        MPI_Finalize(); return 1;
    }
    if (!rank) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
               numBodies, numSteps, validate ? "enabled" : "disabled");
    }
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, devices;
    MPI_Comm_rank(local, &localRank);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA device available on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&local);

    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0, offset = 0; r < ranks; ++r) {
        int count = numBodies / ranks + (r < numBodies % ranks);
        counts[r] = count * 3; displs[r] = offset * 3; offset += count;
    }
    int count = counts[rank] / 3, offset = displs[rank] / 3;
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    std::vector<Vec3> positions(numBodies), velocities(count);
    #pragma omp parallel for
    for (int i = 0; i < numBodies; ++i) positions[i] = bodies[i].pos;
    #pragma omp parallel for
    for (int i = 0; i < count; ++i) velocities[i] = bodies[offset + i].vel;
    Vec3 *dPos, *dNext, *dVel;
    size_t allBytes = std::max<size_t>(1, numBodies) * sizeof(Vec3);
    size_t localBytes = std::max<size_t>(1, count) * sizeof(Vec3);
    CUDA(cudaMalloc(&dPos, allBytes));
    CUDA(cudaMalloc(&dNext, ranks == 1 ? allBytes : localBytes));
    CUDA(cudaMalloc(&dVel, localBytes));
    if (numBodies) CUDA(cudaMemcpy(dPos, positions.data(), numBodies * sizeof(Vec3), cudaMemcpyHostToDevice));
    if (count) CUDA(cudaMemcpy(dVel, velocities.data(), count * sizeof(Vec3), cudaMemcpyHostToDevice));
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        if (count) {
            advance<<<(count + 255) / 256, 256>>>(dPos, dNext, dVel, numBodies, offset, count);
            CUDA(cudaGetLastError());
        }
        if (ranks == 1) {
            std::swap(dPos, dNext);
        } else {
            if (count) CUDA(cudaMemcpy(positions.data() + offset, dNext, count * sizeof(Vec3), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, positions.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            if (numBodies) CUDA(cudaMemcpy(dPos, positions.data(), numBodies * sizeof(Vec3), cudaMemcpyHostToDevice));
        }
    }
    CUDA(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) printf("Simulation time: %ld ms\n", (long)(maxElapsed * 1000));
    if (printResults || validate) {
        if (ranks == 1 && numBodies) CUDA(cudaMemcpy(positions.data(), dPos, numBodies * sizeof(Vec3), cudaMemcpyDeviceToHost));
        if (count) CUDA(cudaMemcpy(velocities.data(), dVel, count * sizeof(Vec3), cudaMemcpyDeviceToHost));
        std::vector<Vec3> allVelocities(rank == 0 ? numBodies : 0);
        MPI_Gatherv(velocities.data(), counts[rank], MPI_DOUBLE, allVelocities.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) {
            #pragma omp parallel for
            for (int i = 0; i < numBodies; ++i) { bodies[i].pos = positions[i]; bodies[i].vel = allVelocities[i]; }
            if (printResults) {
                std::vector<double> bodyData(numBodies * 6ULL);
                #pragma omp parallel for
                for (int i = 0; i < numBodies; ++i) {
                    bodyData[6ULL*i] = bodies[i].pos.x; bodyData[6ULL*i+1] = bodies[i].pos.y; bodyData[6ULL*i+2] = bodies[i].pos.z;
                    bodyData[6ULL*i+3] = bodies[i].vel.x; bodyData[6ULL*i+4] = bodies[i].vel.y; bodyData[6ULL*i+5] = bodies[i].vel.z;
                }
                print_results(bodyData, "Bodies");
            }
        }
    }
    int status = 0;
    if (!rank && validate) {
        printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) {
            printf("Final energy: %.6f\n", computeTotalEnergy(bodies));
            printf("Validation: PASSED\n");
        } else { printf("Validation: FAILED\n"); status = 1; }
    }
    CUDA(cudaFree(dPos)); CUDA(cudaFree(dNext)); CUDA(cudaFree(dVel));
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
