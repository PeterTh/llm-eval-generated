#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int THREADS_PER_BLOCK = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static void checkCuda(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, rank)

// The global positions are laid out as x,y,z triples.  Each rank updates only
// its contiguous local range, so no atomic operations are required.
__global__ void advanceBodies(const double* __restrict__ globalPositions,
                              double* __restrict__ localPositions,
                              double* __restrict__ localVelocities,
                              int localCount, int globalCount) {
    extern __shared__ double tile[];
    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIndex < localCount;

    double px = 0.0, py = 0.0, pz = 0.0;
    if (active) {
        px = localPositions[3 * localIndex];
        py = localPositions[3 * localIndex + 1];
        pz = localPositions[3 * localIndex + 2];
    }
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int base = 0; base < globalCount; base += blockDim.x) {
        const int source = base + threadIdx.x;
        if (source < globalCount) {
            tile[3 * threadIdx.x] = globalPositions[3 * source];
            tile[3 * threadIdx.x + 1] = globalPositions[3 * source + 1];
            tile[3 * threadIdx.x + 2] = globalPositions[3 * source + 2];
        }
        __syncthreads();

        const int tileCount = (globalCount - base < blockDim.x) ? globalCount - base : blockDim.x;
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < tileCount; ++j) {
                const double dx = tile[3 * j] - px;
                const double dy = tile[3 * j + 1] - py;
                const double dz = tile[3 * j + 2] - pz;
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
        double vx = localVelocities[3 * localIndex] + DT * fx;
        double vy = localVelocities[3 * localIndex + 1] + DT * fy;
        double vz = localVelocities[3 * localIndex + 2] + DT * fz;
        localVelocities[3 * localIndex] = vx;
        localVelocities[3 * localIndex + 1] = vy;
        localVelocities[3 * localIndex + 2] = vz;
        localPositions[3 * localIndex] = px + vx * DT;
        localPositions[3 * localIndex + 1] = py + vy * DT;
        localPositions[3 * localIndex + 2] = pz + vz * DT;
    }
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double kinetic = 0.0;
    const int n = static_cast<int>(bodies.size());
    #pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (int i = 0; i < n; ++i) {
        kinetic += 0.5 * (bodies[i].vel.x * bodies[i].vel.x + bodies[i].vel.y * bodies[i].vel.y + bodies[i].vel.z * bodies[i].vel.z);
    }
    double potential = 0.0;
    #pragma omp parallel for reduction(+:potential) schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            potential -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return kinetic + potential;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        const Body& body = bodies[i];
        const bool finite = std::isfinite(body.pos.x) && std::isfinite(body.pos.y) && std::isfinite(body.pos.z) &&
                            std::isfinite(body.vel.x) && std::isfinite(body.vel.y) && std::isfinite(body.vel.z);
        const bool bounded = std::abs(body.pos.x) <= 1e6 && std::abs(body.pos.y) <= 1e6 && std::abs(body.pos.z) <= 1e6 &&
                             std::abs(body.vel.x) <= 1e6 && std::abs(body.vel.y) <= 1e6 && std::abs(body.vel.z) <= 1e6;
        valid &= finite && bounded;
    }
    return valid != 0;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -n <num>     Number of bodies (default: 1024)\n  -s <num>     Number of simulation steps (default: 10)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    int argumentError = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else argumentError = 1;
    }
    if (argumentError || numBodies <= 0 || numSteps < 0) {
        if (rank == 0) { std::printf("Invalid command line arguments\n"); printUsage(argv[0]); }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n", numBodies, numSteps, validate ? "enabled" : "disabled");
    }

    const int baseCount = numBodies / ranks;
    const int remainder = numBodies % ranks;
    const int localCount = baseCount + (rank < remainder ? 1 : 0);
    std::vector<int> bodyCounts(ranks), bodyDisplacements(ranks), scalarCounts(ranks), scalarDisplacements(ranks);
    std::vector<int> byteCounts(ranks), byteDisplacements(ranks);
    for (int r = 0, offset = 0; r < ranks; ++r) {
        bodyCounts[r] = baseCount + (r < remainder ? 1 : 0);
        bodyDisplacements[r] = offset;
        scalarCounts[r] = 3 * bodyCounts[r];
        scalarDisplacements[r] = 3 * offset;
        byteCounts[r] = bodyCounts[r] * static_cast<int>(sizeof(Body));
        byteDisplacements[r] = offset * static_cast<int>(sizeof(Body));
        offset += bodyCounts[r];
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    std::vector<Body> initialBodies;
    if (rank == 0) { initialBodies.resize(numBodies); randomizeBodies(initialBodies); }
    std::vector<Body> localBodies(localCount);
    MPI_Scatterv(rank == 0 ? initialBodies.data() : nullptr, byteCounts.data(), byteDisplacements.data(), MPI_BYTE,
                 localBodies.data(), localCount * static_cast<int>(sizeof(Body)), MPI_BYTE, 0, MPI_COMM_WORLD);

    std::vector<double> hostLocalPositions(3 * localCount), hostLocalVelocities(3 * localCount), hostGlobalPositions(3 * numBodies);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) {
        hostLocalPositions[3 * i] = localBodies[i].pos.x; hostLocalPositions[3 * i + 1] = localBodies[i].pos.y; hostLocalPositions[3 * i + 2] = localBodies[i].pos.z;
        hostLocalVelocities[3 * i] = localBodies[i].vel.x; hostLocalVelocities[3 * i + 1] = localBodies[i].vel.y; hostLocalVelocities[3 * i + 2] = localBodies[i].vel.z;
    }
    MPI_Allgatherv(hostLocalPositions.data(), 3 * localCount, MPI_DOUBLE, hostGlobalPositions.data(), scalarCounts.data(), scalarDisplacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    double *deviceGlobalPositions = nullptr, *deviceLocalPositions = nullptr, *deviceLocalVelocities = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceGlobalPositions, sizeof(double) * hostGlobalPositions.size()));
    CUDA_CHECK(cudaMalloc(&deviceLocalPositions, sizeof(double) * (hostLocalPositions.empty() ? 1 : hostLocalPositions.size())));
    CUDA_CHECK(cudaMalloc(&deviceLocalVelocities, sizeof(double) * (hostLocalVelocities.empty() ? 1 : hostLocalVelocities.size())));
    CUDA_CHECK(cudaMemcpy(deviceGlobalPositions, hostGlobalPositions.data(), sizeof(double) * hostGlobalPositions.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceLocalPositions, hostLocalPositions.data(), sizeof(double) * hostLocalPositions.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceLocalVelocities, hostLocalVelocities.data(), sizeof(double) * hostLocalVelocities.size(), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const int blocks = (localCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            advanceBodies<<<blocks, THREADS_PER_BLOCK, 3 * THREADS_PER_BLOCK * sizeof(double)>>>(deviceGlobalPositions, deviceLocalPositions, deviceLocalVelocities, localCount, numBodies);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemcpy(hostLocalPositions.data(), deviceLocalPositions, sizeof(double) * hostLocalPositions.size(), cudaMemcpyDeviceToHost));
        MPI_Allgatherv(hostLocalPositions.data(), 3 * localCount, MPI_DOUBLE, hostGlobalPositions.data(), scalarCounts.data(), scalarDisplacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(deviceGlobalPositions, hostGlobalPositions.data(), sizeof(double) * hostGlobalPositions.size(), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();
    const long long localElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long globalElapsedMs = 0;
    MPI_Reduce(&localElapsedMs, &globalElapsedMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Simulation time: %lld ms\n", globalElapsedMs);

    CUDA_CHECK(cudaMemcpy(hostLocalVelocities.data(), deviceLocalVelocities, sizeof(double) * hostLocalVelocities.size(), cudaMemcpyDeviceToHost));
    std::vector<Body> finalBodies;
    if (rank == 0) finalBodies.resize(numBodies);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) {
        localBodies[i].pos = Vec3(hostLocalPositions[3 * i], hostLocalPositions[3 * i + 1], hostLocalPositions[3 * i + 2]);
        localBodies[i].vel = Vec3(hostLocalVelocities[3 * i], hostLocalVelocities[3 * i + 1], hostLocalVelocities[3 * i + 2]);
    }
    MPI_Gatherv(localBodies.data(), localCount * static_cast<int>(sizeof(Body)), MPI_BYTE,
                rank == 0 ? finalBodies.data() : nullptr, byteCounts.data(), byteDisplacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(deviceGlobalPositions)); CUDA_CHECK(cudaFree(deviceLocalPositions)); CUDA_CHECK(cudaFree(deviceLocalVelocities));
    int exitCode = 0;
    if (rank == 0 && printResults) {
        std::vector<double> data; data.reserve(6 * numBodies);
        for (const Body& body : finalBodies) { data.insert(data.end(), {body.pos.x, body.pos.y, body.pos.z, body.vel.x, body.vel.y, body.vel.z}); }
        print_results(data, "Bodies");
    }
    if (rank == 0 && validate) {
        std::printf("Validating simulation results...\n");
        if (validateSimulation(finalBodies)) std::printf("Final energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(finalBodies));
        else { std::printf("Validation: FAILED\n"); exitCode = 1; }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
