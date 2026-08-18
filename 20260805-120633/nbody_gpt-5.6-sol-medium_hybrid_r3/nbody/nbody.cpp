#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int CUDA_BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0,
                                      const double z = 0) noexcept
        : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(sizeof(Body) == 6 * sizeof(double));

[[noreturn]] void abortWithMessage(const char* message, int rank) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(call, rank)                                                   \
    do {                                                                         \
        const cudaError_t cuda_status_ = (call);                                  \
        if (cuda_status_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "Rank %d: %s failed at %s:%d: %s\n", (rank),   \
                         #call, __FILE__, __LINE__,                               \
                         cudaGetErrorString(cuda_status_));                       \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                              \
        }                                                                        \
    } while (false)

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

// Every block walks the source bodies in the same increasing order as the
// scalar implementation.  Shared-memory tiling makes each source position serve
// up to CUDA_BLOCK_SIZE independent target bodies.
__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void advanceBodies(const Vec3* __restrict__ positions,
                   Vec3* __restrict__ nextLocalPositions,
                   Vec3* __restrict__ localVelocities, int numBodies,
                   int firstBody, int localBodies) {
    __shared__ Vec3 tile[CUDA_BLOCK_SIZE];

    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIndex < localBodies;
    Vec3 myPosition;
    Vec3 myVelocity;
    if (active) {
        myPosition = positions[firstBody + localIndex];
        myVelocity = localVelocities[localIndex];
    }

    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    for (int base = 0; base < numBodies; base += blockDim.x) {
        const int source = base + threadIdx.x;
        if (source < numBodies) {
            tile[threadIdx.x] = positions[source];
        }
        __syncthreads();

        if (active) {
            const int tileSize = min(blockDim.x, numBodies - base);
#pragma unroll 8
            for (int j = 0; j < tileSize; ++j) {
                const double dx = tile[j].x - myPosition.x;
                const double dy = tile[j].y - myPosition.y;
                const double dz = tile[j].z - myPosition.z;
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
        myVelocity.x += DT * fx;
        myVelocity.y += DT * fy;
        myVelocity.z += DT * fz;
        localVelocities[localIndex] = myVelocity;
        nextLocalPositions[localIndex] =
            Vec3(myPosition.x + myVelocity.x * DT,
                 myPosition.y + myVelocity.y * DT,
                 myPosition.z + myVelocity.z * DT);
    }
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int valid = 1;
#pragma omp parallel for reduction(&:valid) schedule(static)
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const Body& body = bodies[i];
        const bool finite = std::isfinite(body.pos.x) && std::isfinite(body.pos.y) &&
                            std::isfinite(body.pos.z) && std::isfinite(body.vel.x) &&
                            std::isfinite(body.vel.y) && std::isfinite(body.vel.z);
        const bool bounded = std::abs(body.pos.x) <= 1e6 &&
                             std::abs(body.pos.y) <= 1e6 &&
                             std::abs(body.pos.z) <= 1e6 &&
                             std::abs(body.vel.x) <= 1e6 &&
                             std::abs(body.vel.y) <= 1e6 &&
                             std::abs(body.vel.z) <= 1e6;
        valid &= finite && bounded;
    }
    if (!valid) {
        std::printf("Validation failed: found a non-finite or out-of-bounds body state\n");
    }
    return valid != 0;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
#pragma omp parallel for reduction(+:energy) schedule(static)
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const Vec3 v = bodies[i].vel;
        energy += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }
#pragma omp parallel for reduction(+:energy) schedule(dynamic, 8)
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        double localEnergy = 0.0;
        for (std::size_t j = i + 1; j < bodies.size(); ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            localEnergy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
        energy += localEnergy;
    }
    return energy;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            argumentsValid = false;
        }
    }
    if (numBodies <= 0 || numSteps < 0 ||
        static_cast<unsigned long long>(numBodies) * sizeof(Body) >
            static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Body count must be positive and MPI-addressable; steps cannot be negative\n");
        argumentsValid = false;
    }
    if (showHelp || !argumentsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    // Map ranks on each node round-robin onto the visible accelerators.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount), rank);
    if (deviceCount == 0) abortWithMessage("No CUDA accelerator is visible", rank);
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount), rank);

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        const int begin = static_cast<int>((static_cast<long long>(numBodies) * r) / ranks);
        const int end = static_cast<int>((static_cast<long long>(numBodies) * (r + 1)) / ranks);
        offsets[r] = begin;
        counts[r] = end - begin;
    }
    const int localCount = counts[rank];
    const int firstBody = offsets[rank];

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Body> bodies(numBodies);
    if (rank == 0) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), numBodies * static_cast<int>(sizeof(Body)), MPI_BYTE, 0,
              MPI_COMM_WORLD);

    Vec3* hostPositions = nullptr;
    CUDA_CHECK(cudaMallocHost(&hostPositions, static_cast<std::size_t>(numBodies) * sizeof(Vec3)), rank);
    std::vector<Vec3> localVelocities(localCount);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        hostPositions[i] = bodies[i].pos;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) {
        localVelocities[i] = bodies[firstBody + i].vel;
    }
    std::vector<Body>().swap(bodies);

    Vec3* devicePositions = nullptr;
    Vec3* deviceNextLocal = nullptr;
    Vec3* deviceLocalVelocities = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePositions, static_cast<std::size_t>(numBodies) * sizeof(Vec3)), rank);
    // CUDA permits neither zero-byte allocations nor launches with zero blocks.
    const std::size_t localBytes = static_cast<std::size_t>(localCount) * sizeof(Vec3);
    if (localCount > 0) {
        CUDA_CHECK(cudaMalloc(&deviceNextLocal, localBytes), rank);
        CUDA_CHECK(cudaMalloc(&deviceLocalVelocities, localBytes), rank);
        CUDA_CHECK(cudaMemcpy(deviceLocalVelocities, localVelocities.data(), localBytes,
                              cudaMemcpyHostToDevice), rank);
    }
    CUDA_CHECK(cudaMemcpy(devicePositions, hostPositions,
                          static_cast<std::size_t>(numBodies) * sizeof(Vec3),
                          cudaMemcpyHostToDevice), rank);

    std::vector<int> byteCounts(ranks), byteOffsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        byteCounts[r] = counts[r] * static_cast<int>(sizeof(Vec3));
        byteOffsets[r] = offsets[r] * static_cast<int>(sizeof(Vec3));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            const int blocks = (localCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            advanceBodies<<<blocks, CUDA_BLOCK_SIZE>>>(
                devicePositions, deviceNextLocal, deviceLocalVelocities,
                numBodies, firstBody, localCount);
            CUDA_CHECK(cudaGetLastError(), rank);
            CUDA_CHECK(cudaMemcpy(hostPositions + firstBody, deviceNextLocal, localBytes,
                                  cudaMemcpyDeviceToHost), rank);
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostPositions,
                       byteCounts.data(), byteOffsets.data(), MPI_BYTE, MPI_COMM_WORLD);
        // The final state is already in pinned host memory; upload only when a
        // subsequent force evaluation will consume it.
        if (step + 1 < numSteps) {
            CUDA_CHECK(cudaMemcpy(devicePositions, hostPositions,
                                  static_cast<std::size_t>(numBodies) * sizeof(Vec3),
                                  cudaMemcpyHostToDevice), rank);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize(), rank);
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (localCount > 0) {
        CUDA_CHECK(cudaMemcpy(localVelocities.data(), deviceLocalVelocities, localBytes,
                              cudaMemcpyDeviceToHost), rank);
    }
    std::vector<Vec3> allVelocities(rank == 0 ? numBodies : 0);
    MPI_Gatherv(localVelocities.data(), localCount * static_cast<int>(sizeof(Vec3)), MPI_BYTE,
                rank == 0 ? allVelocities.data() : nullptr, byteCounts.data(),
                byteOffsets.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(devicePositions), rank);
    if (localCount > 0) {
        CUDA_CHECK(cudaFree(deviceNextLocal), rank);
        CUDA_CHECK(cudaFree(deviceLocalVelocities), rank);
    }

    int exitCode = EXIT_SUCCESS;
    if (rank == 0) {
        std::printf("Simulation time: %.3f ms\n", elapsedSeconds * 1000.0);
        bodies.resize(numBodies);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = hostPositions[i];
            bodies[i].vel = allVelocities[i];
        }
        if (printResults) {
            std::vector<double> bodyData(static_cast<std::size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                bodyData[6 * i + 0] = bodies[i].pos.x;
                bodyData[6 * i + 1] = bodies[i].pos.y;
                bodyData[6 * i + 2] = bodies[i].pos.z;
                bodyData[6 * i + 3] = bodies[i].vel.x;
                bodyData[6 * i + 4] = bodies[i].vel.y;
                bodyData[6 * i + 5] = bodies[i].vel.z;
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
                exitCode = EXIT_FAILURE;
            }
        }
    }
    CUDA_CHECK(cudaFreeHost(hostPositions), rank);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
