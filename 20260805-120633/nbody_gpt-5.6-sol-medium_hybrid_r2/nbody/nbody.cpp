#include <mpi.h>
#if defined(HAVE_MPIX_QUERY_CUDA_SUPPORT)
#include <mpi-ext.h>
#endif
#include <cuda_runtime.h>
#include <omp.h>

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
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept
        : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static void mpiCheck(int status, const char* operation, MPI_Comm comm) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    std::fprintf(stderr, "MPI error in %s: %.*s\n", operation, length, message);
    MPI_Abort(comm, status);
}

static void cudaCheck(cudaError_t status, const char* operation, MPI_Comm comm) {
    if (status == cudaSuccess) return;
    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    std::fprintf(stderr, "Rank %d CUDA error in %s: %s\n", rank, operation,
                 cudaGetErrorString(status));
    MPI_Abort(comm, static_cast<int>(status));
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

// Every block reuses a tile of source positions from shared memory. Targets are
// partitioned across MPI ranks, so no force reduction or atomics are needed.
__global__ void computeForcesKernel(const Vec3* __restrict__ positions,
                                    Vec3* __restrict__ velocities,
                                    int numBodies, int localBodies,
                                    int globalStart) {
    __shared__ Vec3 tile[CUDA_BLOCK_SIZE];
    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIndex < localBodies;
    Vec3 target;
    if (active) target = positions[globalStart + localIndex];
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int base = 0; base < numBodies; base += blockDim.x) {
        const int source = base + threadIdx.x;
        if (source < numBodies) tile[threadIdx.x] = positions[source];
        __syncthreads();

        const int tileSize = min(static_cast<int>(blockDim.x), numBodies - base);
        if (active) {
#pragma unroll 8
            for (int j = 0; j < tileSize; ++j) {
                const double dx = tile[j].x - target.x;
                const double dy = tile[j].y - target.y;
                const double dz = tile[j].z - target.z;
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
        Vec3 velocity = velocities[localIndex];
        velocity.x += DT * fx;
        velocity.y += DT * fy;
        velocity.z += DT * fz;
        velocities[localIndex] = velocity;
    }
}

__global__ void integrateKernel(Vec3* positions, const Vec3* __restrict__ velocities,
                                int localBodies, int globalStart) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < localBodies) {
        Vec3 position = positions[globalStart + i];
        const Vec3 velocity = velocities[i];
        position.x += velocity.x * DT;
        position.y += velocity.y * DT;
        position.z += velocity.z * DT;
        positions[globalStart + i] = position;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const long long n = static_cast<long long>(bodies.size());
    double kinetic = 0.0;
#pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (long long i = 0; i < n; ++i) {
        const Vec3& v = bodies[static_cast<size_t>(i)].vel;
        kinetic += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    double potential = 0.0;
#pragma omp parallel for reduction(+:potential) schedule(dynamic, 8)
    for (long long i = 0; i < n; ++i) {
        const Vec3& a = bodies[static_cast<size_t>(i)].pos;
        for (long long j = i + 1; j < n; ++j) {
            const Vec3& b = bodies[static_cast<size_t>(j)].pos;
            const double dx = b.x - a.x;
            const double dy = b.y - a.y;
            const double dz = b.z - a.z;
            potential -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return kinetic + potential;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int invalid = 0;
#pragma omp parallel for reduction(|:invalid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(bodies.size()); ++i) {
        const Body& body = bodies[static_cast<size_t>(i)];
        const bool finite = std::isfinite(body.pos.x) && std::isfinite(body.pos.y) &&
                            std::isfinite(body.pos.z) && std::isfinite(body.vel.x) &&
                            std::isfinite(body.vel.y) && std::isfinite(body.vel.z);
        const bool bounded = std::abs(body.pos.x) <= 1e6 && std::abs(body.pos.y) <= 1e6 &&
                             std::abs(body.pos.z) <= 1e6 && std::abs(body.vel.x) <= 1e6 &&
                             std::abs(body.vel.y) <= 1e6 && std::abs(body.vel.z) <= 1e6;
        invalid |= !(finite && bounded);
    }
    if (invalid) std::printf("Validation failed: found non-finite or out-of-bounds body state\n");
    return invalid == 0;
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
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread",
             MPI_COMM_WORLD);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, ranks = 1;
    mpiCheck(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank", comm);
    mpiCheck(MPI_Comm_size(comm, &ranks), "MPI_Comm_size", comm);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI implementation lacks MPI_THREAD_FUNNELED support\n");
        MPI_Abort(comm, 1);
    }

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) parseStatus = 2;
        else parseStatus = 1;
    }
    if (numBodies <= 0 || numSteps < 0) parseStatus = 1;
    if (parseStatus != 0) {
        if (rank == 0) {
            if (parseStatus == 1) std::fprintf(stderr, "Invalid command line arguments\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    // Bind one rank to one GPU using its node-local rank.
    MPI_Comm localComm;
    mpiCheck(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm),
             "MPI_Comm_split_type", comm);
    int localRank = 0;
    mpiCheck(MPI_Comm_rank(localComm, &localRank), "MPI_Comm_rank(local)", comm);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", comm);
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices are available\n");
        MPI_Abort(comm, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", comm);
    mpiCheck(MPI_Comm_free(&localComm), "MPI_Comm_free", comm);

    if (rank == 0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\n", numBodies,
                    numSteps);
        std::printf("Validation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n", validate ? "enabled" : "disabled",
                    ranks, omp_get_max_threads());
    }

    std::vector<int> bodyCounts(static_cast<size_t>(ranks));
    std::vector<int> bodyOffsets(static_cast<size_t>(ranks));
    std::vector<int> byteCounts(static_cast<size_t>(ranks));
    std::vector<int> byteOffsets(static_cast<size_t>(ranks));
    for (int r = 0; r < ranks; ++r) {
        bodyCounts[r] = numBodies / ranks + (r < numBodies % ranks ? 1 : 0);
        bodyOffsets[r] = r * (numBodies / ranks) + (r < numBodies % ranks ? r : numBodies % ranks);
        const long long bytes = static_cast<long long>(bodyCounts[r]) * sizeof(Vec3);
        const long long offset = static_cast<long long>(bodyOffsets[r]) * sizeof(Vec3);
        if (bytes > std::numeric_limits<int>::max() || offset > std::numeric_limits<int>::max()) {
            if (rank == 0) std::fprintf(stderr, "Problem is too large for MPI_Allgatherv counts\n");
            MPI_Abort(comm, 1);
        }
        byteCounts[r] = static_cast<int>(bytes);
        byteOffsets[r] = static_cast<int>(offset);
    }
    if (static_cast<long long>(numBodies) * sizeof(Vec3) > std::numeric_limits<int>::max()) {
        if (rank == 0) std::fprintf(stderr, "Problem is too large for MPI byte counts\n");
        MPI_Abort(comm, 1);
    }
    const int localBodies = bodyCounts[rank];
    const int globalStart = bodyOffsets[rank];

    // Initialization is generated once to preserve the original rand_r sequence.
    std::vector<Body> bodies;
    std::vector<Vec3> hostPositions(static_cast<size_t>(numBodies));
    std::vector<Vec3> hostLocalVelocities(static_cast<size_t>(localBodies));
    if (rank == 0) {
        bodies.resize(static_cast<size_t>(numBodies));
        randomizeBodies(bodies);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) hostPositions[i] = bodies[i].pos;
    }
    mpiCheck(MPI_Bcast(hostPositions.data(), numBodies * static_cast<int>(sizeof(Vec3)), MPI_BYTE, 0, comm),
             "MPI_Bcast(initial positions)", comm);
    std::vector<Vec3> rootVelocities;
    if (rank == 0) {
        rootVelocities.resize(static_cast<size_t>(numBodies));
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) rootVelocities[i] = bodies[i].vel;
    }
    mpiCheck(MPI_Scatterv(rank == 0 ? rootVelocities.data() : nullptr, byteCounts.data(),
                          byteOffsets.data(), MPI_BYTE, hostLocalVelocities.data(), byteCounts[rank],
                          MPI_BYTE, 0, comm), "MPI_Scatterv(initial velocities)", comm);

    Vec3* devicePositions = nullptr;
    Vec3* deviceVelocities = nullptr;
    Vec3* communicationPositions = nullptr;
    cudaCheck(cudaMalloc(&devicePositions, static_cast<size_t>(numBodies) * sizeof(Vec3)),
              "cudaMalloc(positions)", comm);
    cudaCheck(cudaMalloc(&deviceVelocities,
                         static_cast<size_t>(localBodies > 0 ? localBodies : 1) * sizeof(Vec3)),
              "cudaMalloc(velocities)", comm);
    cudaCheck(cudaMemcpy(devicePositions, hostPositions.data(), static_cast<size_t>(numBodies) * sizeof(Vec3),
                         cudaMemcpyHostToDevice), "copy initial positions", comm);
    cudaCheck(cudaMemcpy(deviceVelocities, hostLocalVelocities.data(),
                         static_cast<size_t>(localBodies) * sizeof(Vec3), cudaMemcpyHostToDevice),
              "copy initial velocities", comm);

    bool cudaAwareMpi = false;
#if defined(HAVE_MPIX_QUERY_CUDA_SUPPORT)
    cudaAwareMpi = MPIX_Query_cuda_support() != 0;
#endif
    if (!cudaAwareMpi) {
        cudaCheck(cudaMallocHost(&communicationPositions,
                                 static_cast<size_t>(numBodies) * sizeof(Vec3)),
                  "cudaMallocHost(communication positions)", comm);
        std::memcpy(communicationPositions, hostPositions.data(),
                    static_cast<size_t>(numBodies) * sizeof(Vec3));
    }
    if (rank == 0)
        std::printf("CUDA-aware MPI: %s\n", cudaAwareMpi ? "enabled" : "unavailable (pinned staging)");

    mpiCheck(MPI_Barrier(comm), "MPI_Barrier(start)", comm);
    const auto start = std::chrono::high_resolution_clock::now();
    const int blocks = (localBodies + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    for (int step = 0; step < numSteps; ++step) {
        if (blocks > 0) {
            computeForcesKernel<<<blocks, CUDA_BLOCK_SIZE>>>(devicePositions, deviceVelocities,
                                                             numBodies, localBodies, globalStart);
            integrateKernel<<<blocks, CUDA_BLOCK_SIZE>>>(devicePositions, deviceVelocities,
                                                          localBodies, globalStart);
            cudaCheck(cudaGetLastError(), "simulation kernel launch", comm);
            cudaCheck(cudaDeviceSynchronize(), "simulation kernels", comm);
        }
        if (cudaAwareMpi) {
            // Fast path: positions remain resident on the GPU throughout the run.
            mpiCheck(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, devicePositions,
                                    byteCounts.data(), byteOffsets.data(), MPI_BYTE, comm),
                     "MPI_Allgatherv(device positions)", comm);
        } else {
            cudaCheck(cudaMemcpy(communicationPositions + globalStart,
                                 devicePositions + globalStart,
                                 static_cast<size_t>(localBodies) * sizeof(Vec3),
                                 cudaMemcpyDeviceToHost), "stage local positions", comm);
            mpiCheck(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, communicationPositions,
                                    byteCounts.data(), byteOffsets.data(), MPI_BYTE, comm),
                     "MPI_Allgatherv(staged positions)", comm);
            cudaCheck(cudaMemcpy(devicePositions, communicationPositions,
                                 static_cast<size_t>(numBodies) * sizeof(Vec3),
                                 cudaMemcpyHostToDevice), "restore global positions", comm);
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "final synchronization", comm);
    mpiCheck(MPI_Barrier(comm), "MPI_Barrier(stop)", comm);
    const auto end = std::chrono::high_resolution_clock::now();
    const double localMilliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    double milliseconds = 0.0;
    mpiCheck(MPI_Reduce(&localMilliseconds, &milliseconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm),
             "MPI_Reduce(time)", comm);
    if (rank == 0) std::printf("Simulation time: %.3f ms\n", milliseconds);

    cudaCheck(cudaMemcpy(hostPositions.data(), devicePositions, static_cast<size_t>(numBodies) * sizeof(Vec3),
                         cudaMemcpyDeviceToHost), "copy final positions", comm);
    cudaCheck(cudaMemcpy(hostLocalVelocities.data(), deviceVelocities,
                         static_cast<size_t>(localBodies) * sizeof(Vec3), cudaMemcpyDeviceToHost),
              "copy final velocities", comm);
    std::vector<Vec3> allVelocities;
    if (rank == 0) allVelocities.resize(static_cast<size_t>(numBodies));
    mpiCheck(MPI_Gatherv(hostLocalVelocities.data(), byteCounts[rank], MPI_BYTE,
                         rank == 0 ? allVelocities.data() : nullptr, byteCounts.data(), byteOffsets.data(),
                         MPI_BYTE, 0, comm), "MPI_Gatherv(final velocities)", comm);

    int result = 0;
    if (rank == 0) {
        bodies.resize(static_cast<size_t>(numBodies));
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = hostPositions[i];
            bodies[i].vel = allVelocities[i];
        }
        if (printResults) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                bodyData[6 * i] = bodies[i].pos.x;
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
                std::printf("Final energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(bodies));
            } else {
                std::printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }
    mpiCheck(MPI_Bcast(&result, 1, MPI_INT, 0, comm), "MPI_Bcast(result)", comm);
    if (communicationPositions) cudaFreeHost(communicationPositions);
    cudaFree(deviceVelocities);
    cudaFree(devicePositions);
    MPI_Finalize();
    return result;
}
