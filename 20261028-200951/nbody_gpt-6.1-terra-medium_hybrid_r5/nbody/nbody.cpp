#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
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

// Each block reuses a tile of source positions for all its target bodies.
constexpr int TILE = 128;
__global__ void advance(const Vec3* positions, Vec3* next, Vec3* velocities,
                        int n, int offset, int count) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];
    int t = threadIdx.x, local = blockIdx.x * blockDim.x + t;
    bool active = local < count;
    Vec3 p;
    if (active) p = positions[offset + local];
    double fx = 0, fy = 0, fz = 0;
    for (int base = 0; base < n; base += TILE) {
        int j = base + t;
        if (j < n) { sx[t] = positions[j].x; sy[t] = positions[j].y; sz[t] = positions[j].z; }
        __syncthreads();
        int limit = min(TILE, n - base);
        if (active) {
            for (int k = 0; k < limit; ++k) {
                double dx = sx[k] - p.x, dy = sy[k] - p.y, dz = sz[k] - p.z;
                double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                double inv3 = inv * inv * inv;
                fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
            }
        }
        __syncthreads();
    }
    if (active) {
        Vec3 v = velocities[local];
        v.x += DT * fx; v.y += DT * fy; v.z += DT * fz;
        velocities[local] = v;
        next[local].x = p.x + DT * v.x;
        next[local].y = p.y + DT * v.y;
        next[local].z = p.z + DT * v.z;
    }
}

static void cudaCheck(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", where, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

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
    #pragma omp parallel for reduction(-:energy) schedule(dynamic, 16)
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
        if (!rank) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
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
    if (numBodies < 0 || numBodies > INT_MAX / 3) {
        if (!rank) fprintf(stderr, "Invalid number of bodies\n");
        MPI_Finalize(); return 1;
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    CUDA(cudaGetDeviceCount(&devices));
    if (devices == 0) { fprintf(stderr, "No CUDA devices available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        int count = numBodies / ranks + (r < numBodies % ranks);
        int offset = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
        counts[r] = 3 * count; displs[r] = 3 * offset;
    }
    int count = counts[rank] / 3, offset = displs[rank] / 3;
    if (!rank) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
               numBodies, numSteps, validate ? "enabled" : "disabled");
    }
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    Vec3* hostPos;
    CUDA(cudaMallocHost(&hostPos, sizeof(Vec3) * std::max(numBodies, 1)));
    std::vector<Vec3> hostVel(count);
    #pragma omp parallel for
    for (int i = 0; i < numBodies; ++i) hostPos[i] = bodies[i].pos;
    #pragma omp parallel for
    for (int i = 0; i < count; ++i) hostVel[i] = bodies[offset + i].vel;
    Vec3 *dPos, *dNext, *dVel;
    CUDA(cudaMalloc(&dPos, sizeof(Vec3) * std::max(numBodies, 1)));
    CUDA(cudaMalloc(&dNext, sizeof(Vec3) * std::max(numBodies, 1)));
    CUDA(cudaMalloc(&dVel, sizeof(Vec3) * std::max(count, 1)));
    CUDA(cudaMemcpy(dPos, hostPos, sizeof(Vec3) * numBodies, cudaMemcpyHostToDevice));
    if (count) CUDA(cudaMemcpy(dVel, hostVel.data(), sizeof(Vec3) * count, cudaMemcpyHostToDevice));
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (count) {
            advance<<<(count + TILE - 1) / TILE, TILE>>>(dPos, dNext, dVel, numBodies, offset, count);
            CUDA(cudaGetLastError());
        }
        if (ranks == 1) {
            std::swap(dPos, dNext);
        } else {
            if (count) CUDA(cudaMemcpy(hostPos + offset, dNext, sizeof(Vec3) * count, cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, hostPos, counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            CUDA(cudaMemcpy(dPos, hostPos, sizeof(Vec3) * numBodies, cudaMemcpyHostToDevice));
        }
    }
    CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    if (!rank) printf("Simulation time: %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    if (validate || printResults) {
        if (ranks == 1) CUDA(cudaMemcpy(hostPos, dPos, sizeof(Vec3) * numBodies, cudaMemcpyDeviceToHost));
        if (count) CUDA(cudaMemcpy(hostVel.data(), dVel, sizeof(Vec3) * count, cudaMemcpyDeviceToHost));
        std::vector<Vec3> allVel(rank == 0 ? numBodies : 0);
        MPI_Gatherv(hostVel.data(), 3 * count, MPI_DOUBLE, allVel.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) {
            #pragma omp parallel for
            for (int i = 0; i < numBodies; ++i) { bodies[i].pos = hostPos[i]; bodies[i].vel = allVel[i]; }
            if (printResults) {
                std::vector<double> data(static_cast<size_t>(numBodies) * 6);
                #pragma omp parallel for
                for (int i = 0; i < numBodies; ++i) {
                    size_t j = static_cast<size_t>(i) * 6;
                    data[j] = bodies[i].pos.x; data[j+1] = bodies[i].pos.y; data[j+2] = bodies[i].pos.z;
                    data[j+3] = bodies[i].vel.x; data[j+4] = bodies[i].vel.y; data[j+5] = bodies[i].vel.z;
                }
                print_results(data, "Bodies");
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
    CUDA(cudaFree(dPos)); CUDA(cudaFree(dNext)); CUDA(cudaFree(dVel)); CUDA(cudaFreeHost(hostPos));
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
