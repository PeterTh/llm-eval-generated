#include <mpi.h>
#include <cuda_runtime.h>
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

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0,
                                       const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be MPI-packable");

__global__ void computeForcesKernel(const Body* bodies, Body* result,
                                    int first, int count, int n) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;

    const int i = first + local;
    const Body self = bodies[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;

    // Keep the same j traversal and interaction equation as the reference code.
    for (int j = 0; j < n; ++j) {
        const double dx = bodies[j].pos.x - self.pos.x;
        const double dy = bodies[j].pos.y - self.pos.y;
        const double dz = bodies[j].pos.z - self.pos.z;
        const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double inverseDistance = 1.0 / sqrt(distanceSquared);
        const double inverseDistanceCubed = inverseDistance * inverseDistance * inverseDistance;
        fx += dx * inverseDistanceCubed;
        fy += dy * inverseDistanceCubed;
        fz += dz * inverseDistanceCubed;
    }

    Body updated = self;
    updated.vel.x += DT * fx;
    updated.vel.y += DT * fy;
    updated.vel.z += DT * fz;
    result[local] = updated;
}

[[noreturn]] void mpiAbort(const char* message, int rank) {
    if (rank == 0) std::fprintf(stderr, "%s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(cudaError_t error, const char* operation, int rank) {
    if (error != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "CUDA error in %s: %s", operation,
                      cudaGetErrorString(error));
        mpiAbort(message, rank);
    }
}

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

void integrateBodies(std::vector<Body>& bodies, int first, int count) {
    #pragma omp parallel for schedule(static)
    for (int i = first; i < first + count; ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());
    double energy = 0.0;
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < n; ++i) {
        const Body& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int invalid = 0;
    #pragma omp parallel for reduction(|:invalid) schedule(static)
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        const Body& body = bodies[i];
        const bool finite = std::isfinite(body.pos.x) && std::isfinite(body.pos.y) &&
                            std::isfinite(body.pos.z) && std::isfinite(body.vel.x) &&
                            std::isfinite(body.vel.y) && std::isfinite(body.vel.z);
        const bool bounded = std::abs(body.pos.x) <= 1e6 && std::abs(body.pos.y) <= 1e6 &&
                             std::abs(body.pos.z) <= 1e6 && std::abs(body.vel.x) <= 1e6 &&
                             std::abs(body.vel.y) <= 1e6 && std::abs(body.vel.z) <= 1e6;
        invalid |= !(finite && bounded);
    }
    if (invalid) std::printf("Validation failed: invalid body state\n");
    return invalid == 0;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -n <num>     Number of bodies (default: 1024)\n"
                "  -s <num>     Number of simulation steps (default: 10)\n"
                "  -v           Enable validation (checks energy conservation)\n"
                "  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (numBodies < 0 || numSteps < 0) mpiAbort("Number of bodies and steps must be non-negative", rank);

    if (rank == 0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
                    numBodies, numSteps, validate ? "enabled" : "disabled");
    }
    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    if (rank == 0) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), numBodies * static_cast<int>(sizeof(Body)), MPI_BYTE, 0, MPI_COMM_WORLD);

    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount", rank);
    if (devices == 0) mpiAbort("No CUDA device is available", rank);
    checkCuda(cudaSetDevice(rank % devices), "cudaSetDevice", rank);

    const int first = (numBodies * rank) / ranks;
    const int last = (numBodies * (rank + 1)) / ranks;
    const int localCount = last - first;
    Body* deviceBodies = nullptr;
    Body* deviceResult = nullptr;
    checkCuda(cudaMalloc(&deviceBodies, static_cast<size_t>(numBodies) * sizeof(Body)), "cudaMalloc", rank);
    checkCuda(cudaMalloc(&deviceResult, static_cast<size_t>(localCount) * sizeof(Body)), "cudaMalloc", rank);
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = ((numBodies * (r + 1)) / ranks - (numBodies * r) / ranks) * sizeof(Body);
        displacements[r] = (numBodies * r / ranks) * sizeof(Body);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        checkCuda(cudaMemcpy(deviceBodies, bodies.data(), static_cast<size_t>(numBodies) * sizeof(Body),
                             cudaMemcpyHostToDevice), "cudaMemcpy bodies", rank);
        const int blockSize = 256;
        const int gridSize = (localCount + blockSize - 1) / blockSize;
        if (localCount > 0) {
            computeForcesKernel<<<gridSize, blockSize>>>(deviceBodies, deviceResult, first, localCount, numBodies);
            checkCuda(cudaGetLastError(), "computeForcesKernel launch", rank);
            checkCuda(cudaMemcpy(bodies.data() + first, deviceResult, static_cast<size_t>(localCount) * sizeof(Body),
                                 cudaMemcpyDeviceToHost), "cudaMemcpy results", rank);
        }
        integrateBodies(bodies, first, localCount);
        MPI_Allgatherv(MPI_IN_PLACE, counts[rank], MPI_BYTE, bodies.data(), counts.data(),
                       displacements.data(), MPI_BYTE, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::printf("Simulation time: %ld ms\n", duration.count());
    }

    if (printResults && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (const auto& body : bodies) {
            bodyData.insert(bodyData.end(), {body.pos.x, body.pos.y, body.pos.z,
                                             body.vel.x, body.vel.y, body.vel.z});
        }
        print_results(bodyData, "Bodies");
    }
    if (validate && rank == 0) {
        std::printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) {
            std::printf("Final energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(bodies));
        } else {
            std::printf("Validation: FAILED\n");
            cudaFree(deviceResult); cudaFree(deviceBodies); MPI_Finalize(); return 1;
        }
    }
    cudaFree(deviceResult);
    cudaFree(deviceBodies);
    MPI_Finalize();
    return 0;
}
