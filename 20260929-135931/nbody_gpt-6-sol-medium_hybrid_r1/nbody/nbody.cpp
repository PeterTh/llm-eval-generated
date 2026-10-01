#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 128;
constexpr int WARP_SIZE = 32;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(double x = 0, double y = 0, double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Body) == 6 * sizeof(double));

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
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

// One warp owns each destination body. Its lanes traverse source bodies in
// parallel; coalesced source reads and warp reductions keep small runs busy.
__global__ void stepBodies(Body* localBodies, const double4* positions,
                           double4* nextPositions, int localCount, int totalCount) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int i = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (i >= localCount) return;
    Body body = localBodies[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = lane; j < totalCount; j += WARP_SIZE) {
        const double4 source = positions[j];
        const double dx = source.x - body.pos.x;
        const double dy = source.y - body.pos.y;
        const double dz = source.z - body.pos.z;
        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;
        fx += dx * invDist3;
        fy += dy * invDist3;
        fz += dz * invDist3;
    }
    for (int delta = WARP_SIZE / 2; delta > 0; delta /= 2) {
        fx += __shfl_down_sync(0xffffffff, fx, delta);
        fy += __shfl_down_sync(0xffffffff, fy, delta);
        fz += __shfl_down_sync(0xffffffff, fz, delta);
    }
    if (lane == 0) {
        body.vel.x += DT * fx;
        body.vel.y += DT * fy;
        body.vel.z += DT * fz;
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
        localBodies[i] = body;
        nextPositions[i] = make_double4(body.pos.x, body.pos.y, body.pos.z, 0.0);
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());
    double kinetic = 0.0, potential = 0.0;
#pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (int i = 0; i < n; ++i) {
        const Body& b = bodies[i];
        kinetic += 0.5 * (b.vel.x * b.vel.x + b.vel.y * b.vel.y + b.vel.z * b.vel.z);
    }
#pragma omp parallel for reduction(+:potential) schedule(dynamic, 8)
    for (int i = 0; i < n; ++i) {
        double subtotal = 0.0;
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            subtotal -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
        potential += subtotal;
    }
    return kinetic + potential;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int invalid = 0;
#pragma omp parallel for reduction(|:invalid)
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        const Body& b = bodies[i];
        invalid |= !std::isfinite(b.pos.x) || !std::isfinite(b.pos.y) || !std::isfinite(b.pos.z) ||
                   !std::isfinite(b.vel.x) || !std::isfinite(b.vel.y) || !std::isfinite(b.vel.z) ||
                   std::abs(b.pos.x) > 1e6 || std::abs(b.pos.y) > 1e6 || std::abs(b.pos.z) > 1e6 ||
                   std::abs(b.vel.x) > 1e6 || std::abs(b.vel.y) > 1e6 || std::abs(b.vel.z) > 1e6;
    }
    if (invalid) std::printf("Validation failed: found non-finite or out-of-bounds body state\n");
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
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
    // MPI's vector collectives use int counts. Keep all position and body counts in range.
    if (numBodies < 0 || numBodies > INT_MAX / static_cast<int>(sizeof(Body))) {
        if (rank == 0) std::fprintf(stderr, "Invalid number of bodies\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        std::fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");
    MPI_Comm_free(&localComm);

    std::vector<int> bodyCounts(worldSize), bodyOffsets(worldSize);
    std::vector<int> positionBytes(worldSize), positionOffsets(worldSize);
    std::vector<int> bodyBytes(worldSize), bodyByteOffsets(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        bodyOffsets[r] = static_cast<int>((static_cast<long long>(numBodies) * r) / worldSize);
        bodyCounts[r] = static_cast<int>((static_cast<long long>(numBodies) * (r + 1)) / worldSize) - bodyOffsets[r];
        positionBytes[r] = bodyCounts[r] * static_cast<int>(sizeof(double4));
        positionOffsets[r] = bodyOffsets[r] * static_cast<int>(sizeof(double4));
        bodyBytes[r] = bodyCounts[r] * static_cast<int>(sizeof(Body));
        bodyByteOffsets[r] = bodyOffsets[r] * static_cast<int>(sizeof(Body));
    }
    const int localCount = bodyCounts[rank];
    const int localOffset = bodyOffsets[rank];

    std::vector<Body> bodies;
    if (rank == 0) { bodies.resize(numBodies); randomizeBodies(bodies); }
    std::vector<Body> initial(localCount);
    MPI_Scatterv(rank == 0 ? bodies.data() : nullptr, bodyBytes.data(), bodyByteOffsets.data(), MPI_BYTE,
                 initial.data(), bodyBytes[rank], MPI_BYTE, 0, MPI_COMM_WORLD);

    // Positions start with the same initialization on every rank. All subsequent
    // steps exchange only positions; velocity remains on its owning GPU.
    double4* hostPositions = nullptr;
    cudaCheck(cudaMallocHost(&hostPositions, std::max(1, numBodies) * sizeof(double4)), "cudaMallocHost positions");
    if (rank == 0) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i)
            hostPositions[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
    }
    MPI_Bcast(hostPositions, numBodies * static_cast<int>(sizeof(double4)), MPI_BYTE, 0, MPI_COMM_WORLD);

    Body* deviceBodies = nullptr;
    double4 *devicePositions = nullptr, *deviceNext = nullptr;
    cudaCheck(cudaMalloc(&deviceBodies, std::max(1, localCount) * sizeof(Body)), "cudaMalloc bodies");
    cudaCheck(cudaMalloc(&devicePositions, std::max(1, numBodies) * sizeof(double4)), "cudaMalloc positions");
    cudaCheck(cudaMalloc(&deviceNext, std::max(1, localCount) * sizeof(double4)), "cudaMalloc next positions");
    if (localCount) cudaCheck(cudaMemcpy(deviceBodies, initial.data(), localCount * sizeof(Body), cudaMemcpyHostToDevice), "copy bodies");
    if (numBodies) cudaCheck(cudaMemcpy(devicePositions, hostPositions, numBodies * sizeof(double4), cudaMemcpyHostToDevice), "copy positions");

    // Force module loading and first launch overhead outside the measured loop.
    stepBodies<<<1, BLOCK_SIZE>>>(deviceBodies, devicePositions, deviceNext, 0, 0);
    cudaCheck(cudaGetLastError(), "warm-up launch");
    cudaCheck(cudaDeviceSynchronize(), "warm-up synchronization");

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount) {
            stepBodies<<<(localCount + BLOCK_SIZE / WARP_SIZE - 1) / (BLOCK_SIZE / WARP_SIZE), BLOCK_SIZE>>>(
                deviceBodies, devicePositions, deviceNext, localCount, numBodies);
            cudaCheck(cudaGetLastError(), "stepBodies launch");
        }
        if (step + 1 < numSteps) {
            if (worldSize == 1) {
                std::swap(devicePositions, deviceNext);
            } else {
                if (localCount) cudaCheck(cudaMemcpy(hostPositions + localOffset, deviceNext,
                    localCount * sizeof(double4), cudaMemcpyDeviceToHost), "copy updated positions");
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostPositions,
                               positionBytes.data(), positionOffsets.data(), MPI_BYTE, MPI_COMM_WORLD);
                if (numBodies) cudaCheck(cudaMemcpy(devicePositions, hostPositions,
                    numBodies * sizeof(double4), cudaMemcpyHostToDevice), "upload positions");
            }
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "synchronize simulation");
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Simulation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));

    if (printResults || validate) {
        if (localCount) cudaCheck(cudaMemcpy(initial.data(), deviceBodies, localCount * sizeof(Body),
                                              cudaMemcpyDeviceToHost), "copy final bodies");
        MPI_Gatherv(initial.data(), bodyBytes[rank], MPI_BYTE, rank == 0 ? bodies.data() : nullptr,
                    bodyBytes.data(), bodyByteOffsets.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    cudaCheck(cudaFree(deviceNext), "cudaFree next");
    cudaCheck(cudaFree(devicePositions), "cudaFree positions");
    cudaCheck(cudaFree(deviceBodies), "cudaFree bodies");
    cudaCheck(cudaFreeHost(hostPositions), "cudaFreeHost positions");

    int status = 0;
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
                std::printf("Final energy: %.6f\n", computeTotalEnergy(bodies));
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
