#include <algorithm>
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

constexpr int CUDA_BLOCK_SIZE = 256;

[[noreturn]] void cudaFail(cudaError_t error, const char* expression, const char* file, int line) {
    std::fprintf(stderr, "CUDA failure at %s:%d (%s): %s\n", file, line, expression,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define CUDA_CHECK(expression) do { const cudaError_t error_ = (expression); \
    if (error_ != cudaSuccess) cudaFail(error_, #expression, __FILE__, __LINE__); } while (0)

// Each rank owns a contiguous range of bodies.  The complete position array is
// replicated on its GPU, while only the owned bodies' position and velocity are updated.
__global__ void advanceBodies(Body* __restrict__ localBodies,
                              const Vec3* __restrict__ globalPositions,
                              int localCount, int globalCount) {
    __shared__ Vec3 positionTile[CUDA_BLOCK_SIZE];
    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIndex < localCount;

    Vec3 position{};
    double fx = 0.0, fy = 0.0, fz = 0.0;
    if (active) position = localBodies[localIndex].pos;

    for (int tileStart = 0; tileStart < globalCount; tileStart += blockDim.x) {
        const int source = tileStart + threadIdx.x;
        if (source < globalCount) positionTile[threadIdx.x] = globalPositions[source];
        __syncthreads();

        const int tileCount = min(blockDim.x, globalCount - tileStart);
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < tileCount; ++j) {
                const double dx = positionTile[j].x - position.x;
                const double dy = positionTile[j].y - position.y;
                const double dz = positionTile[j].z - position.z;
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
        Body body = localBodies[localIndex];
        body.vel.x += DT * fx;
        body.vel.y += DT * fy;
        body.vel.z += DT * fz;
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
        localBodies[localIndex] = body;
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

void makeDistribution(int total, int ranks, std::vector<int>& counts, std::vector<int>& offsets) {
    counts.resize(ranks);
    offsets.resize(ranks);
    const int quotient = total / ranks;
    const int remainder = total % ranks;
    int offset = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        counts[rank] = quotient + (rank < remainder ? 1 : 0);
        offsets[rank] = offset;
        offset += counts[rank];
    }
}

bool validateLocalBodies(const std::vector<Body>& bodies) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (size_t i = 0; i < bodies.size(); ++i) {
        const Body& body = bodies[i];
        const bool finite = std::isfinite(body.pos.x) && std::isfinite(body.pos.y) && std::isfinite(body.pos.z) &&
                            std::isfinite(body.vel.x) && std::isfinite(body.vel.y) && std::isfinite(body.vel.z);
        const bool bounded = std::abs(body.pos.x) <= 1e6 && std::abs(body.pos.y) <= 1e6 && std::abs(body.pos.z) <= 1e6 &&
                             std::abs(body.vel.x) <= 1e6 && std::abs(body.vel.y) <= 1e6 && std::abs(body.vel.z) <= 1e6;
        valid &= static_cast<int>(finite && bounded);
    }
    return valid != 0;
}

double computeLocalEnergy(const std::vector<Body>& localBodies, const std::vector<Vec3>& positions,
                          int globalOffset) {
    double energy = 0.0;
    const int n = static_cast<int>(positions.size());
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t local = 0; local < localBodies.size(); ++local) {
        const Body& body = localBodies[local];
        double contribution = 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
        const int global = globalOffset + static_cast<int>(local);
        for (int j = global + 1; j < n; ++j) {
            const double dx = positions[j].x - body.pos.x;
            const double dy = positions[j].y - body.pos.y;
            const double dz = positions[j].z - body.pos.z;
            contribution -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
        energy += contribution;
    }
    return energy;
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
    int mpiProvided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiProvided);
    int rank = 0, ranks = 1;
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (numBodies <= 0 || numSteps < 0) {
        if (rank == 0) std::fprintf(stderr, "Number of bodies must be positive and steps must be non-negative\n");
        MPI_Finalize();
        return 1;
    }

    std::vector<int> counts, offsets, elementCounts, elementOffsets, bodyElementCounts, bodyElementOffsets;
    makeDistribution(numBodies, ranks, counts, offsets);
    elementCounts.resize(ranks);
    elementOffsets.resize(ranks);
    bodyElementCounts.resize(ranks);
    bodyElementOffsets.resize(ranks);
    for (int i = 0; i < ranks; ++i) {
        elementCounts[i] = counts[i] * 3;
        elementOffsets[i] = offsets[i] * 3;
        bodyElementCounts[i] = counts[i] * 6;
        bodyElementOffsets[i] = offsets[i] * 6;
    }
    const int localCount = counts[rank];

    // Assign ranks to GPUs by their node-local rank, allowing every node to use all its accelerators.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device is available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        printf("N-Body Simulation (MPI + OpenMP + CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI ranks: %d\n", ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Body> initialBodies;
    if (rank == 0) {
        initialBodies.resize(numBodies);
        randomizeBodies(initialBodies);
    }
    std::vector<Body> localBodies(localCount);
    MPI_Scatterv(rank == 0 ? initialBodies.data() : nullptr, bodyElementCounts.data(), bodyElementOffsets.data(), MPI_DOUBLE,
                 localBodies.data(), localCount * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<Vec3> localPositions(localCount), globalPositions(numBodies);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) localPositions[i] = localBodies[i].pos;
    MPI_Allgatherv(localPositions.data(), localCount * 3, MPI_DOUBLE, globalPositions.data(),
                   elementCounts.data(), elementOffsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    Body* deviceLocalBodies = nullptr;
    Vec3* deviceGlobalPositions = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceLocalBodies, static_cast<size_t>(std::max(localCount, 1)) * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&deviceGlobalPositions, static_cast<size_t>(numBodies) * sizeof(Vec3)));
    if (localCount > 0) CUDA_CHECK(cudaMemcpy(deviceLocalBodies, localBodies.data(), localCount * sizeof(Body), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceGlobalPositions, globalPositions.data(), numBodies * sizeof(Vec3), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            const int blocks = (localCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            advanceBodies<<<blocks, CUDA_BLOCK_SIZE>>>(deviceLocalBodies, deviceGlobalPositions, localCount, numBodies);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(localBodies.data(), deviceLocalBodies, localCount * sizeof(Body), cudaMemcpyDeviceToHost));
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < localCount; ++i) localPositions[i] = localBodies[i].pos;
        MPI_Allgatherv(localPositions.data(), localCount * 3, MPI_DOUBLE, globalPositions.data(),
                       elementCounts.data(), elementOffsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(deviceGlobalPositions, globalPositions.data(), numBodies * sizeof(Vec3), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double localSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %.3f ms\n", elapsedSeconds * 1000.0);

    if (printResults) {
        std::vector<Body> allBodies;
        if (rank == 0) allBodies.resize(numBodies);
        MPI_Gatherv(localBodies.data(), localCount * 6, MPI_DOUBLE, rank == 0 ? allBodies.data() : nullptr,
                    bodyElementCounts.data(), bodyElementOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                bodyData[6 * i] = allBodies[i].pos.x; bodyData[6 * i + 1] = allBodies[i].pos.y; bodyData[6 * i + 2] = allBodies[i].pos.z;
                bodyData[6 * i + 3] = allBodies[i].vel.x; bodyData[6 * i + 4] = allBodies[i].vel.y; bodyData[6 * i + 5] = allBodies[i].vel.z;
            }
            print_results(bodyData, "Bodies");
        }
    }

    int localValid = validateLocalBodies(localBodies) ? 1 : 0;
    int globallyValid = 0;
    if (validate) MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (validate) {
        const double localEnergy = computeLocalEnergy(localBodies, globalPositions, offsets[rank]);
        double finalEnergy = 0.0;
        MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating simulation results...\n");
            if (globallyValid) printf("Final energy: %.6f\nValidation: PASSED\n", finalEnergy);
            else printf("Validation: FAILED\n");
        }
    }

    CUDA_CHECK(cudaFree(deviceGlobalPositions));
    CUDA_CHECK(cudaFree(deviceLocalBodies));
    MPI_Finalize();
    return validate && !globallyValid;
}
