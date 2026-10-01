#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <chrono>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int TILE = 256;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(double x = 0, double y = 0, double z = 0) noexcept : x(x), y(y), z(z) {}
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

void checkCuda(cudaError_t err, int rank) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Each thread owns one body. All threads read the old positions before any
// position is exchanged, preserving the original step ordering.
__global__ void advance(const Vec3* positions, Vec3* nextLocal,
                        Vec3* velocities, int n, int offset, int localCount) {
    __shared__ Vec3 tile[TILE];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < localCount;
    Vec3 self;
    Vec3 vel;
    if (active) {
        self = positions[offset + local];
        vel = velocities[local];
    }
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += TILE) {
        const int source = base + threadIdx.x;
        if (source < n) tile[threadIdx.x] = positions[source];
        __syncthreads();
        if (active) {
            const int count = min(TILE, n - base);
            for (int j = 0; j < count; ++j) {
                const double dx = tile[j].x - self.x;
                const double dy = tile[j].y - self.y;
                const double dz = tile[j].z - self.z;
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
        vel.x += DT * fx;
        vel.y += DT * fy;
        vel.z += DT * fz;
        velocities[local] = vel;
        nextLocal[local] = Vec3(self.x + vel.x * DT,
                                self.y + vel.y * DT,
                                self.z + vel.z * DT);
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());
    double kinetic = 0.0, potential = 0.0;
#pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (int i = 0; i < n; ++i) {
        const Vec3 v = bodies[i].vel;
        kinetic += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }
#pragma omp parallel for reduction(+:potential) schedule(dynamic, 8)
    for (int i = 0; i < n; ++i) {
        const Vec3 p = bodies[i].pos;
        double partial = 0.0;
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - p.x;
            const double dy = bodies[j].pos.y - p.y;
            const double dz = bodies[j].pos.z - p.z;
            partial -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
        potential += partial;
    }
    return kinetic + potential;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int invalid = 0;
#pragma omp parallel for reduction(|:invalid) schedule(static)
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        const Body& b = bodies[i];
        if (!std::isfinite(b.pos.x) || !std::isfinite(b.pos.y) || !std::isfinite(b.pos.z) ||
            !std::isfinite(b.vel.x) || !std::isfinite(b.vel.y) || !std::isfinite(b.vel.z) ||
            std::abs(b.pos.x) > 1e6 || std::abs(b.pos.y) > 1e6 || std::abs(b.pos.z) > 1e6 ||
            std::abs(b.vel.x) > 1e6 || std::abs(b.vel.y) > 1e6 || std::abs(b.vel.z) > 1e6)
            invalid = 1;
    }
    if (invalid) std::printf("Validation failed: invalid body state\n");
    return !invalid;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (numBodies < 0 || numBodies > INT_MAX / static_cast<int>(sizeof(Body)) || numSteps < 0) {
        if (rank == 0) std::fprintf(stderr, "Invalid body or step count\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
                    numBodies, numSteps, validate ? "enabled" : "disabled");
    }

    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank;
    MPI_Comm_rank(shared, &localRank);
    MPI_Comm_free(&shared);
    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices), rank);
    if (devices == 0) {
        std::fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % devices), rank);

    std::vector<Body> bodies(numBodies);
    if (rank == 0) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), numBodies * static_cast<int>(sizeof(Body)), MPI_BYTE, 0, MPI_COMM_WORLD);

    const int offset = static_cast<int>((static_cast<long long>(numBodies) * rank) / ranks);
    const int end = static_cast<int>((static_cast<long long>(numBodies) * (rank + 1)) / ranks);
    const int localCount = end - offset;
    std::vector<Vec3> positions(numBodies), localPositions(localCount), localVelocities(localCount);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) positions[i] = bodies[i].pos;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) localVelocities[i] = bodies[offset + i].vel;
    bodies.clear();
    if (rank != 0) bodies.shrink_to_fit();

    std::vector<int> counts(ranks), displacements(ranks), bodyCounts(ranks), bodyDisplacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const int first = static_cast<int>((static_cast<long long>(numBodies) * r) / ranks);
        const int last = static_cast<int>((static_cast<long long>(numBodies) * (r + 1)) / ranks);
        counts[r] = (last - first) * static_cast<int>(sizeof(Vec3));
        displacements[r] = first * static_cast<int>(sizeof(Vec3));
        bodyCounts[r] = (last - first) * static_cast<int>(sizeof(Body));
        bodyDisplacements[r] = first * static_cast<int>(sizeof(Body));
    }

    Vec3 *devicePositions = nullptr, *deviceNext = nullptr, *deviceVelocities = nullptr;
    checkCuda(cudaMalloc(&devicePositions, std::max(size_t(1), positions.size() * sizeof(Vec3))), rank);
    checkCuda(cudaMalloc(&deviceNext, std::max(size_t(1), localPositions.size() * sizeof(Vec3))), rank);
    checkCuda(cudaMalloc(&deviceVelocities, std::max(size_t(1), localVelocities.size() * sizeof(Vec3))), rank);
    checkCuda(cudaMemcpy(devicePositions, positions.data(), positions.size() * sizeof(Vec3), cudaMemcpyHostToDevice), rank);
    checkCuda(cudaMemcpy(deviceVelocities, localVelocities.data(), localVelocities.size() * sizeof(Vec3), cudaMemcpyHostToDevice), rank);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            advance<<<(localCount + TILE - 1) / TILE, TILE>>>(devicePositions, deviceNext,
                                                                 deviceVelocities, numBodies, offset, localCount);
            checkCuda(cudaGetLastError(), rank);
            checkCuda(cudaMemcpy(localPositions.data(), deviceNext, localPositions.size() * sizeof(Vec3),
                                 cudaMemcpyDeviceToHost), rank);
        }
        MPI_Allgatherv(localPositions.data(), counts[rank], MPI_BYTE, positions.data(),
                       counts.data(), displacements.data(), MPI_BYTE, MPI_COMM_WORLD);
        if (step + 1 < numSteps)
            checkCuda(cudaMemcpy(devicePositions, positions.data(), positions.size() * sizeof(Vec3),
                                 cudaMemcpyHostToDevice), rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto endTime = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - start);
    const long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Simulation time: %ld ms\n", maxDuration);

    if (printResults || validate) {
        checkCuda(cudaMemcpy(localVelocities.data(), deviceVelocities, localVelocities.size() * sizeof(Vec3),
                             cudaMemcpyDeviceToHost), rank);
        std::vector<Body> localBodies(localCount);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < localCount; ++i) {
            localBodies[i].pos = positions[offset + i];
            localBodies[i].vel = localVelocities[i];
        }
        if (rank == 0) bodies.resize(numBodies);
        MPI_Gatherv(localBodies.data(), bodyCounts[rank], MPI_BYTE, bodies.data(),
                    bodyCounts.data(), bodyDisplacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    checkCuda(cudaFree(devicePositions), rank);
    checkCuda(cudaFree(deviceNext), rank);
    checkCuda(cudaFree(deviceVelocities), rank);

    int result = 0;
    if (rank == 0) {
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(static_cast<size_t>(numBodies) * 6);
            for (const auto& body : bodies) {
                bodyData.push_back(body.pos.x); bodyData.push_back(body.pos.y); bodyData.push_back(body.pos.z);
                bodyData.push_back(body.vel.x); bodyData.push_back(body.vel.y); bodyData.push_back(body.vel.z);
            }
            print_results(bodyData, "Bodies");
        }
        if (validate) {
            std::printf("Validating simulation results...\n");
            if (validateSimulation(bodies)) {
                std::printf("Final energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(bodies));
            } else {
                std::printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
