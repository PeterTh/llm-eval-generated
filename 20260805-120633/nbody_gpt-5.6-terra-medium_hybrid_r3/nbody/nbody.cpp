#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int THREADS_PER_BLOCK = 256;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(sizeof(Body) == 6 * sizeof(double));

[[noreturn]] void fail(const char* message, int rank) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s: %s", operation, cudaGetErrorString(status));
        fail(message, rank);
    }
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (Body& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

__global__ void computeForcesKernel(const Vec3* __restrict__ positions, Body* __restrict__ localBodies,
                                    int globalCount, int localCount) {
    extern __shared__ double shared[];
    double* sx = shared;
    double* sy = sx + blockDim.x;
    double* sz = sy + blockDim.x;
    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIndex < localCount;
    Vec3 target;
    if (active) target = localBodies[localIndex].pos;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int tile = 0; tile < globalCount; tile += blockDim.x) {
        const int source = tile + threadIdx.x;
        if (source < globalCount) {
            const Vec3 p = positions[source];
            sx[threadIdx.x] = p.x;
            sy[threadIdx.x] = p.y;
            sz[threadIdx.x] = p.z;
        }
        __syncthreads();
        const int count = min(static_cast<int>(blockDim.x), globalCount - tile);
        if (active) {
#pragma unroll 4
            for (int j = 0; j < count; ++j) {
                const double dx = sx[j] - target.x;
                const double dy = sy[j] - target.y;
                const double dz = sz[j] - target.z;
                const double invDist = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }
    if (active) {
        localBodies[localIndex].vel.x += DT * fx;
        localBodies[localIndex].vel.y += DT * fy;
        localBodies[localIndex].vel.z += DT * fz;
    }
}

__global__ void integrateBodiesKernel(Body* bodies, Vec3* __restrict__ positions, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
        positions[i] = bodies[i].pos;
    }
}

void refreshGlobalPositions(const std::vector<Vec3>& localPositions, std::vector<Vec3>& globalPositions,
                            const std::vector<int>& counts, const std::vector<int>& offsets) {
    MPI_Allgatherv(localPositions.data(), static_cast<int>(localPositions.size() * 3), MPI_DOUBLE,
                   globalPositions.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
}

double computeTotalEnergy(const std::vector<Body>& localBodies, const std::vector<Vec3>& globalPositions,
                          int globalOffset, int rank) {
    double localEnergy = 0.0;
#pragma omp parallel for reduction(+ : localEnergy) schedule(static)
    for (long long localI = 0; localI < static_cast<long long>(localBodies.size()); ++localI) {
        const Body& body = localBodies[static_cast<size_t>(localI)];
        double energy = 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
        const int globalI = globalOffset + static_cast<int>(localI);
        for (size_t j = static_cast<size_t>(globalI + 1); j < globalPositions.size(); ++j) {
            const double dx = globalPositions[j].x - body.pos.x;
            const double dy = globalPositions[j].y - body.pos.y;
            const double dz = globalPositions[j].z - body.pos.z;
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
        localEnergy += energy;
    }
    double totalEnergy = 0.0;
    MPI_Reduce(&localEnergy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return rank == 0 ? totalEnergy : 0.0;
}

bool validateSimulation(const std::vector<Body>& localBodies) {
    int valid = 1;
#pragma omp parallel for reduction(& : valid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(localBodies.size()); ++i) {
        const Body& b = localBodies[static_cast<size_t>(i)];
        const bool finite = std::isfinite(b.pos.x) && std::isfinite(b.pos.y) && std::isfinite(b.pos.z) &&
                            std::isfinite(b.vel.x) && std::isfinite(b.vel.y) && std::isfinite(b.vel.z);
        const bool bounded = std::abs(b.pos.x) <= 1e6 && std::abs(b.pos.y) <= 1e6 && std::abs(b.pos.z) <= 1e6 &&
                             std::abs(b.vel.x) <= 1e6 && std::abs(b.vel.y) <= 1e6 && std::abs(b.vel.z) <= 1e6;
        valid &= finite && bounded;
    }
    int globalValid = 0;
    MPI_Allreduce(&valid, &globalValid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    return globalValid != 0;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -n <num>     Number of bodies (default: 1024)\n"
                "  -s <num>     Number of simulation steps (default: 10)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (numBodies < 1 || numSteps < 0) fail("number of bodies must be positive and steps must be non-negative", rank);
    if (ranks > numBodies) fail("MPI rank count cannot exceed number of bodies", rank);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) fail("no CUDA device is available", rank);
    checkCuda(cudaSetDevice(rank % deviceCount), "cudaSetDevice", rank);

    std::vector<int> bodyCounts(ranks), bodyOffsets(ranks), positionCounts(ranks), positionOffsets(ranks);
    const int base = numBodies / ranks, remainder = numBodies % ranks;
    for (int r = 0, offset = 0; r < ranks; ++r) {
        const int count = base + (r < remainder ? 1 : 0);
        bodyCounts[r] = count * 6; bodyOffsets[r] = offset * 6;
        positionCounts[r] = count * 3; positionOffsets[r] = offset * 3;
        offset += count;
    }
    const int localCount = bodyCounts[rank] / 6;
    const int globalOffset = bodyOffsets[rank] / 6;
    std::vector<Body> initialBodies;
    if (rank == 0) { initialBodies.resize(numBodies); randomizeBodies(initialBodies); }
    std::vector<Body> localBodies(localCount);
    MPI_Scatterv(rank == 0 ? reinterpret_cast<double*>(initialBodies.data()) : nullptr, bodyCounts.data(), bodyOffsets.data(), MPI_DOUBLE,
                 reinterpret_cast<double*>(localBodies.data()), localCount * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<Vec3> globalPositions(numBodies);
    std::vector<Vec3> localPositions(localCount);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) localPositions[i] = localBodies[i].pos;
    refreshGlobalPositions(localPositions, globalPositions, positionCounts, positionOffsets);

    if (rank == 0) {
        std::printf("N-Body Simulation (MPI ranks: %d, CUDA GPUs: %d, OpenMP threads/rank: %d)\n", ranks, deviceCount, omp_get_max_threads());
        std::printf("Number of bodies: %d\nNumber of steps: %d\nValidation: %s\n", numBodies, numSteps, validate ? "enabled" : "disabled");
    }
    Body* deviceBodies = nullptr; Vec3* devicePositions = nullptr; Vec3* deviceLocalPositions = nullptr;
    checkCuda(cudaMalloc(&deviceBodies, static_cast<size_t>(localCount) * sizeof(Body)), "allocate device bodies", rank);
    checkCuda(cudaMalloc(&devicePositions, static_cast<size_t>(numBodies) * sizeof(Vec3)), "allocate device positions", rank);
    checkCuda(cudaMalloc(&deviceLocalPositions, static_cast<size_t>(localCount) * sizeof(Vec3)), "allocate device local positions", rank);
    checkCuda(cudaMemcpy(deviceBodies, localBodies.data(), static_cast<size_t>(localCount) * sizeof(Body), cudaMemcpyHostToDevice), "copy initial bodies to device", rank);
    const int localBlocks = (localCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const size_t sharedBytes = 3 * THREADS_PER_BLOCK * sizeof(double);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        checkCuda(cudaMemcpy(devicePositions, globalPositions.data(), static_cast<size_t>(numBodies) * sizeof(Vec3), cudaMemcpyHostToDevice), "copy positions to device", rank);
        computeForcesKernel<<<localBlocks, THREADS_PER_BLOCK, sharedBytes>>>(devicePositions, deviceBodies, numBodies, localCount);
        integrateBodiesKernel<<<localBlocks, THREADS_PER_BLOCK>>>(deviceBodies, deviceLocalPositions, localCount);
        checkCuda(cudaGetLastError(), "launch simulation kernels", rank);
        checkCuda(cudaMemcpy(localPositions.data(), deviceLocalPositions, static_cast<size_t>(localCount) * sizeof(Vec3), cudaMemcpyDeviceToHost), "copy local positions from device", rank);
        refreshGlobalPositions(localPositions, globalPositions, positionCounts, positionOffsets);
    }
    checkCuda(cudaDeviceSynchronize(), "synchronize CUDA device", rank);
    checkCuda(cudaMemcpy(localBodies.data(), deviceBodies, static_cast<size_t>(localCount) * sizeof(Body), cudaMemcpyDeviceToHost), "copy final bodies from device", rank);
    const auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count(), maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Simulation time: %.3f ms\n", maxElapsed);

    if (printResults) {
        std::vector<Body> allBodies(rank == 0 ? numBodies : 0);
        MPI_Gatherv(localBodies.data(), localCount * 6, MPI_DOUBLE, rank == 0 ? reinterpret_cast<double*>(allBodies.data()) : nullptr,
                    bodyCounts.data(), bodyOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<double> bodyData; bodyData.reserve(static_cast<size_t>(numBodies) * 6);
            for (const Body& b : allBodies) { bodyData.insert(bodyData.end(), {b.pos.x, b.pos.y, b.pos.z, b.vel.x, b.vel.y, b.vel.z}); }
            print_results(bodyData, "Bodies");
        }
    }
    int result = 0;
    if (validate) {
        const bool valid = validateSimulation(localBodies);
        const double energy = computeTotalEnergy(localBodies, globalPositions, globalOffset, rank);
        if (rank == 0) {
            if (valid) std::printf("Final energy: %.6f\nValidation: PASSED\n", energy);
            else { std::printf("Validation: FAILED\n"); result = 1; }
        }
        MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    cudaFree(deviceLocalPositions); cudaFree(devicePositions); cudaFree(deviceBodies);
    MPI_Finalize();
    return result;
}
