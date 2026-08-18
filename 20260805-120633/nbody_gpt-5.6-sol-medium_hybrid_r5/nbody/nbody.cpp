#include <mpi.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int CUDA_BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(double x_ = 0, double y_ = 0, double z_ = 0) noexcept
        : x(x_), y(y_), z(z_) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(std::is_trivially_copyable_v<Vec3> && sizeof(Vec3) == 3 * sizeof(double));

static void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA error in %s: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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

// One CUDA thread advances one rank-local body. Global positions are staged
// through shared memory, so each position is fetched from device memory once
// per thread block rather than once per interaction.
__global__ void advanceBodies(const Vec3* __restrict__ positions,
                              Vec3* __restrict__ velocities,
                              Vec3* __restrict__ newPositions,
                              int globalBegin, int localCount, int bodyCount) {
    extern __shared__ Vec3 positionTile[];
    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIndex < localCount;

    Vec3 position;
    Vec3 velocity;
    if (active) {
        position = positions[globalBegin + localIndex];
        velocity = velocities[localIndex];
    }

    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    for (int tileBegin = 0; tileBegin < bodyCount; tileBegin += blockDim.x) {
        const int source = tileBegin + threadIdx.x;
        if (source < bodyCount) {
            positionTile[threadIdx.x] = positions[source];
        }
        __syncthreads();

        if (active) {
            const int tileCount = min(static_cast<int>(blockDim.x), bodyCount - tileBegin);
#pragma unroll 4
            for (int j = 0; j < tileCount; ++j) {
                const double dx = positionTile[j].x - position.x;
                const double dy = positionTile[j].y - position.y;
                const double dz = positionTile[j].z - position.z;
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
        velocity.x += DT * fx;
        velocity.y += DT * fy;
        velocity.z += DT * fz;
        position.x += velocity.x * DT;
        position.y += velocity.y * DT;
        position.z += velocity.z * DT;
        velocities[localIndex] = velocity;
        newPositions[localIndex] = position;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const long long n = static_cast<long long>(bodies.size());
    double kinetic = 0.0;
#pragma omp parallel for reduction(+ : kinetic) schedule(static)
    for (long long i = 0; i < n; ++i) {
        const Vec3 v = bodies[static_cast<size_t>(i)].vel;
        kinetic += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    double potential = 0.0;
#pragma omp parallel for reduction(+ : potential) schedule(dynamic, 8)
    for (long long i = 0; i < n; ++i) {
        const Vec3 pi = bodies[static_cast<size_t>(i)].pos;
        for (long long j = i + 1; j < n; ++j) {
            const Vec3 pj = bodies[static_cast<size_t>(j)].pos;
            const double dx = pj.x - pi.x;
            const double dy = pj.y - pi.y;
            const double dz = pj.z - pi.z;
            potential -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return kinetic + potential;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int nonFinite = 0;
    int badPosition = 0;
    int badVelocity = 0;
    const long long n = static_cast<long long>(bodies.size());
#pragma omp parallel for reduction(| : nonFinite, badPosition, badVelocity) schedule(static)
    for (long long i = 0; i < n; ++i) {
        const Body& body = bodies[static_cast<size_t>(i)];
        nonFinite |= !std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) ||
                     !std::isfinite(body.pos.z) || !std::isfinite(body.vel.x) ||
                     !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z);
        badPosition |= std::abs(body.pos.x) > 1e6 || std::abs(body.pos.y) > 1e6 ||
                       std::abs(body.pos.z) > 1e6;
        badVelocity |= std::abs(body.vel.x) > 1e6 || std::abs(body.vel.y) > 1e6 ||
                       std::abs(body.vel.z) > 1e6;
    }
    if (nonFinite) std::printf("Validation failed: found NaN or Inf value in body state\n");
    if (badPosition) std::printf("Validation failed: body position exceeds reasonable bounds\n");
    if (badVelocity) std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
    return !(nonFinite || badPosition || badVelocity);
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
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
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
    if (numBodies <= 0 || numSteps < 0) {
        if (rank == 0) std::fprintf(stderr, "Body count must be positive and step count non-negative\n");
        argumentsValid = false;
    }
    if (showHelp || !argumentsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    // Assign ranks to GPUs by their node-local rank. Multiple ranks per GPU
    // remain supported for schedulers that intentionally oversubscribe GPUs.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices are available\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);
    cudaCheck(cudaFree(nullptr), "CUDA context initialization", rank);

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", worldSize);
    }

    std::vector<int> counts(static_cast<size_t>(worldSize));
    std::vector<int> offsets(static_cast<size_t>(worldSize));
    const int quotient = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    for (int r = 0, offset = 0; r < worldSize; ++r) {
        counts[static_cast<size_t>(r)] = quotient + (r < remainder ? 1 : 0);
        offsets[static_cast<size_t>(r)] = offset;
        offset += counts[static_cast<size_t>(r)];
    }
    const int localCount = counts[static_cast<size_t>(rank)];
    const int globalBegin = offsets[static_cast<size_t>(rank)];
    const size_t localAllocation = static_cast<size_t>(localCount > 0 ? localCount : 1);

    MPI_Datatype mpiVec3;
    MPI_Type_contiguous(3, MPI_DOUBLE, &mpiVec3);
    MPI_Type_commit(&mpiVec3);

    Vec3* globalPositions = nullptr;
    Vec3* localPositions = nullptr;
    Vec3* localVelocities = nullptr;
    cudaCheck(cudaHostAlloc(&globalPositions, static_cast<size_t>(numBodies) * sizeof(Vec3),
                            cudaHostAllocPortable), "allocating global host positions", rank);
    cudaCheck(cudaHostAlloc(&localPositions, localAllocation * sizeof(Vec3), cudaHostAllocPortable),
              "allocating local host positions", rank);
    cudaCheck(cudaHostAlloc(&localVelocities, localAllocation * sizeof(Vec3), cudaHostAllocPortable),
              "allocating local host velocities", rank);

    std::vector<Body> bodies;
    std::vector<Vec3> initialVelocities;
    if (rank == 0) {
        bodies.resize(static_cast<size_t>(numBodies));
        initialVelocities.resize(static_cast<size_t>(numBodies));
        randomizeBodies(bodies);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            globalPositions[i] = bodies[static_cast<size_t>(i)].pos;
            initialVelocities[static_cast<size_t>(i)] = bodies[static_cast<size_t>(i)].vel;
        }
    }
    MPI_Bcast(globalPositions, numBodies, mpiVec3, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? initialVelocities.data() : nullptr, counts.data(), offsets.data(), mpiVec3,
                 localVelocities, localCount, mpiVec3, 0, MPI_COMM_WORLD);

    Vec3* devicePositions = nullptr;
    Vec3* deviceVelocities = nullptr;
    Vec3* deviceNewPositions = nullptr;
    cudaCheck(cudaMalloc(&devicePositions, static_cast<size_t>(numBodies) * sizeof(Vec3)),
              "allocating device positions", rank);
    cudaCheck(cudaMalloc(&deviceVelocities, localAllocation * sizeof(Vec3)),
              "allocating device velocities", rank);
    cudaCheck(cudaMalloc(&deviceNewPositions, localAllocation * sizeof(Vec3)),
              "allocating new device positions", rank);
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "creating CUDA stream", rank);
    cudaCheck(cudaMemcpyAsync(devicePositions, globalPositions,
                              static_cast<size_t>(numBodies) * sizeof(Vec3), cudaMemcpyHostToDevice, stream),
              "copying initial positions", rank);
    if (localCount > 0) {
        cudaCheck(cudaMemcpyAsync(deviceVelocities, localVelocities,
                                  static_cast<size_t>(localCount) * sizeof(Vec3), cudaMemcpyHostToDevice, stream),
                  "copying initial velocities", rank);
    }
    cudaCheck(cudaStreamSynchronize(stream), "initial data transfer", rank);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            const int blocks = (localCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            advanceBodies<<<blocks, CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE * sizeof(Vec3), stream>>>(
                devicePositions, deviceVelocities, deviceNewPositions,
                globalBegin, localCount, numBodies);
            cudaCheck(cudaGetLastError(), "launching advanceBodies", rank);
            cudaCheck(cudaMemcpyAsync(localPositions, deviceNewPositions,
                                      static_cast<size_t>(localCount) * sizeof(Vec3),
                                      cudaMemcpyDeviceToHost, stream), "copying updated positions", rank);
        }
        cudaCheck(cudaStreamSynchronize(stream), "advancing bodies", rank);

        MPI_Allgatherv(localPositions, localCount, mpiVec3, globalPositions,
                       counts.data(), offsets.data(), mpiVec3, MPI_COMM_WORLD);
        if (step + 1 < numSteps) {
            cudaCheck(cudaMemcpyAsync(devicePositions, globalPositions,
                                      static_cast<size_t>(numBodies) * sizeof(Vec3),
                                      cudaMemcpyHostToDevice, stream), "broadcasting positions to GPU", rank);
            cudaCheck(cudaStreamSynchronize(stream), "position transfer", rank);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (localCount > 0) {
        cudaCheck(cudaMemcpyAsync(localVelocities, deviceVelocities,
                                  static_cast<size_t>(localCount) * sizeof(Vec3),
                                  cudaMemcpyDeviceToHost, stream), "copying final velocities", rank);
        cudaCheck(cudaStreamSynchronize(stream), "final velocity transfer", rank);
    }
    std::vector<Vec3> finalVelocities;
    if (rank == 0) finalVelocities.resize(static_cast<size_t>(numBodies));
    MPI_Gatherv(localVelocities, localCount, mpiVec3,
                rank == 0 ? finalVelocities.data() : nullptr, counts.data(), offsets.data(), mpiVec3,
                0, MPI_COMM_WORLD);

    int returnCode = 0;
    if (rank == 0) {
        std::printf("Simulation time: %ld ms\n", static_cast<long>(elapsedSeconds * 1000.0));
        bodies.resize(static_cast<size_t>(numBodies));
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            bodies[static_cast<size_t>(i)].pos = globalPositions[i];
            bodies[static_cast<size_t>(i)].vel = finalVelocities[static_cast<size_t>(i)];
        }

        if (printResults) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                const Body& body = bodies[static_cast<size_t>(i)];
                const size_t base = static_cast<size_t>(i) * 6;
                bodyData[base] = body.pos.x;
                bodyData[base + 1] = body.pos.y;
                bodyData[base + 2] = body.pos.z;
                bodyData[base + 3] = body.vel.x;
                bodyData[base + 4] = body.vel.y;
                bodyData[base + 5] = body.vel.z;
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
                returnCode = 1;
            }
        }
    }
    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    cudaStreamDestroy(stream);
    cudaFree(deviceNewPositions);
    cudaFree(deviceVelocities);
    cudaFree(devicePositions);
    cudaFreeHost(localVelocities);
    cudaFreeHost(localPositions);
    cudaFreeHost(globalPositions);
    MPI_Type_free(&mpiVec3);
    MPI_Finalize();
    return returnCode;
}
