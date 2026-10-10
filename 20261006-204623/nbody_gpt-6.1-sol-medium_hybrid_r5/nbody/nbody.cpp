#include <algorithm>
#include <climits>
#include <cstddef>
#include <exception>
#include <type_traits>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
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
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// One MPI rank per GPU is recommended. MPI is called only by the main thread.
static int worldRank = 0;

void checkMPI(int error, const char* operation) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "Rank %d: %s: %.*s\n", worldRank, operation, length, message);
        MPI_Abort(MPI_COMM_WORLD, error);
    }
}

void checkCUDA(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", worldRank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define MPI_CHECK(call) checkMPI((call), #call)
#define CUDA_CHECK(call) checkCUDA((call), #call)

static_assert(std::is_standard_layout<Body>::value && sizeof(Body) == 6 * sizeof(double),
              "MPI body layout must contain six contiguous doubles");
static_assert(sizeof(Vec3) == 3 * sizeof(double), "MPI position layout must contain three doubles");

// Fixed allocations avoid per-step allocation and use pinned MPI staging memory.
template <typename T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename T> struct PinnedBuffer {
    T* data = nullptr;
    explicit PinnedBuffer(size_t count) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~PinnedBuffer() { if (data) cudaFreeHost(data); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

constexpr int TILE = 128;

// Each thread owns a target body and visits sources in their original order.
// All threads (including inactive tail lanes) participate in both tile barriers.
// Input and output positions are separate: integration must not affect forces
// for other bodies during this time step.
__global__ void advanceBodies(const Vec3* positions, Vec3* nextPositions,
                              Vec3* velocities, int n, int offset, int localCount) {
    __shared__ Vec3 sources[TILE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < localCount;
    Vec3 p;
    if (active) p = positions[offset + i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += TILE) {
        const int count = min(TILE, n - base);
        if (threadIdx.x < count) sources[threadIdx.x] = positions[base + threadIdx.x];
        __syncthreads();
        if (active) {
            #pragma unroll 8
            for (int j = 0; j < count; ++j) {
                const double dx = sources[j].x - p.x;
                const double dy = sources[j].y - p.y;
                const double dz = sources[j].z - p.z;
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
        Vec3 v = velocities[i];
        v.x += DT * fx;
        v.y += DT * fy;
        v.z += DT * fz;
        velocities[i] = v;
        p.x += v.x * DT;
        p.y += v.y * DT;
        p.z += v.z * DT;
        nextPositions[i] = p;
    }
}

// Optional diagnostic remains distributed and accelerated instead of running
// an O(N^2) serial energy calculation on rank zero.
__global__ void bodyEnergies(const Vec3* positions, const Vec3* velocities,
                             double* energies, int n, int offset, int localCount) {
    __shared__ Vec3 sources[TILE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalIndex = offset + i;
    const bool active = i < localCount;
    Vec3 p;
    double energy = 0.0;
    if (active) {
        p = positions[globalIndex];
        const Vec3 v = velocities[i];
        energy = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }
    // Skip source tiles that precede every target in this block.
    for (int base = (offset + int(blockIdx.x) * TILE) / TILE * TILE; base < n; base += TILE) {
        const int count = min(TILE, n - base);
        if (threadIdx.x < count) sources[threadIdx.x] = positions[base + threadIdx.x];
        __syncthreads();
        if (active) {
            for (int j = max(0, globalIndex + 1 - base); j < count; ++j) {
                const double dx = sources[j].x - p.x;
                const double dy = sources[j].y - p.y;
                const double dz = sources[j].z - p.z;
                energy -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            }
        }
        __syncthreads();
    }
    if (active) energies[i] = energy;
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks state and reports energy)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv) {
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-s") == 0) && i + 1 < argc) {
            const bool bodiesOption = strcmp(argv[i], "-n") == 0;
            char* end = nullptr;
            const long value = strtol(argv[++i], &end, 10);
            // MPI position counts use three doubles per body.
            if (*end || end == argv[i] || value < 0 ||
                value > (bodiesOption ? INT_MAX / 3 : INT_MAX)) {
                if (worldRank == 0) fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
                return 1;
            }
            if (bodiesOption) numBodies = static_cast<int>(value);
            else numSteps = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Honors CUDA_VISIBLE_DEVICES, including launchers exposing one GPU/rank.
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    if (worldRank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts(worldSize), offsets(worldSize), positionCounts(worldSize), positionOffsets(worldSize);
    for (int rank = 0; rank < worldSize; ++rank) {
        counts[rank] = numBodies / worldSize + (rank < numBodies % worldSize);
        offsets[rank] = rank * (numBodies / worldSize) + std::min(rank, numBodies % worldSize);
        positionCounts[rank] = 3 * counts[rank];
        positionOffsets[rank] = 3 * offsets[rank];
    }
    const int localCount = counts[worldRank];
    const int offset = offsets[worldRank];
    const size_t localBytes = size_t(localCount) * sizeof(Vec3);
    const size_t globalBytes = size_t(numBodies) * sizeof(Vec3);
    std::vector<Body> bodies(worldRank == 0 ? numBodies : 0);
    // Retain the original seed and serial rand_r draw order exactly.
    if (worldRank == 0) randomizeBodies(bodies);
    std::vector<Body> localBodies(localCount);
    MPI_Datatype bodyType;
    MPI_CHECK(MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType));
    MPI_CHECK(MPI_Type_commit(&bodyType));
    MPI_CHECK(MPI_Scatterv(bodies.data(), counts.data(), offsets.data(), bodyType,
                         localBodies.data(), localCount, bodyType, 0, MPI_COMM_WORLD));

    PinnedBuffer<Vec3> hostPositions(numBodies), hostVelocities(localCount);
    Vec3* localPositions = hostPositions.data + offset;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) {
        localPositions[i] = localBodies[i].pos;
        hostVelocities.data[i] = localBodies[i].vel;
    }
    MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, hostPositions.data,
                            positionCounts.data(), positionOffsets.data(), MPI_DOUBLE, MPI_COMM_WORLD));
    DeviceBuffer<Vec3> allPositions(numBodies), nextPositions(localCount), velocities(localCount);
    CUDA_CHECK(cudaMemcpy(allPositions.data, hostPositions.data, globalBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(velocities.data, hostVelocities.data, localBytes, cudaMemcpyHostToDevice));
    Vec3* current = allPositions.data;
    Vec3* next = nextPositions.data;
    const int blocks = (localCount + TILE - 1) / TILE;
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount) {
            advanceBodies<<<blocks, TILE>>>(current, next, velocities.data, numBodies, offset, localCount);
            CUDA_CHECK(cudaGetLastError());
        }
        if (worldSize == 1) {
            // No PCIe round trips or MPI copies in the single-GPU step loop.
            std::swap(current, next);
        } else if (step + 1 < numSteps || validate) {
            CUDA_CHECK(cudaMemcpy(localPositions, next, localBytes, cudaMemcpyDeviceToHost));
            // Only positions are communicated; velocities stay on their owner.
            // Final positions need replication only for the energy diagnostic.
            MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, hostPositions.data,
                                    positionCounts.data(), positionOffsets.data(), MPI_DOUBLE, MPI_COMM_WORLD));
            CUDA_CHECK(cudaMemcpy(current, hostPositions.data, globalBytes, cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (worldRank == 0) printf("Simulation time: %lld ms\n", static_cast<long long>(maxElapsed * 1000.0));

    if (printResults || validate) {
        if (worldSize == 1)
            CUDA_CHECK(cudaMemcpy(localPositions, current, localBytes, cudaMemcpyDeviceToHost));
        else if (numSteps > 0 && !validate)
            CUDA_CHECK(cudaMemcpy(localPositions, next, localBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostVelocities.data, velocities.data, localBytes, cudaMemcpyDeviceToHost));
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < localCount; ++i) {
            localBodies[i].pos = localPositions[i];
            localBodies[i].vel = hostVelocities.data[i];
        }
    }
    if (printResults) {
        MPI_CHECK(MPI_Gatherv(localBodies.data(), localCount, bodyType, bodies.data(),
                             counts.data(), offsets.data(), bodyType, 0, MPI_COMM_WORLD));
        if (worldRank == 0) {
            std::vector<double> bodyData(size_t(numBodies) * 6);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                bodyData[size_t(i) * 6] = bodies[i].pos.x;
                bodyData[size_t(i) * 6 + 1] = bodies[i].pos.y;
                bodyData[size_t(i) * 6 + 2] = bodies[i].pos.z;
                bodyData[size_t(i) * 6 + 3] = bodies[i].vel.x;
                bodyData[size_t(i) * 6 + 4] = bodies[i].vel.y;
                bodyData[size_t(i) * 6 + 5] = bodies[i].vel.z;
            }
            print_results(bodyData, "Bodies");
        }
    }

    int failed = 0;
    if (validate) {
        int localFailure = 0;
        #pragma omp parallel for schedule(static) reduction(|:localFailure)
        for (int i = 0; i < localCount; ++i) {
            const Body& b = localBodies[i];
            if (!std::isfinite(b.pos.x) || !std::isfinite(b.pos.y) || !std::isfinite(b.pos.z) ||
                !std::isfinite(b.vel.x) || !std::isfinite(b.vel.y) || !std::isfinite(b.vel.z)) localFailure |= 1;
            if (std::abs(b.pos.x) > 1e6 || std::abs(b.pos.y) > 1e6 || std::abs(b.pos.z) > 1e6) localFailure |= 2;
            if (std::abs(b.vel.x) > 1e6 || std::abs(b.vel.y) > 1e6 || std::abs(b.vel.z) > 1e6) localFailure |= 4;
        }
        MPI_CHECK(MPI_Allreduce(&localFailure, &failed, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD));
        if (worldRank == 0) printf("Validating simulation results...\n");
        if (!failed) {
            DeviceBuffer<double> energies(localCount);
            PinnedBuffer<double> hostEnergies(localCount);
            if (localCount) {
                bodyEnergies<<<blocks, TILE>>>(current, velocities.data, energies.data, numBodies, offset, localCount);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(hostEnergies.data, energies.data, size_t(localCount) * sizeof(double), cudaMemcpyDeviceToHost));
            }
            double localEnergy = 0.0, finalEnergy = 0.0;
            #pragma omp parallel for schedule(static) reduction(+:localEnergy)
            for (int i = 0; i < localCount; ++i) localEnergy += hostEnergies.data[i];
            MPI_CHECK(MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
            if (worldRank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else if (worldRank == 0) {
            if (failed & 1) printf("Validation failed: found NaN or Inf value in body state\n");
            if (failed & 2) printf("Validation failed: body position exceeds reasonable bounds\n");
            if (failed & 4) printf("Validation failed: body velocity exceeds reasonable bounds\n");
            printf("Validation: FAILED\n");
        }
    }
    MPI_CHECK(MPI_Type_free(&bodyType));
    return failed ? 1 : 0;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int initError = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (initError != MPI_SUCCESS) return 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int result = 1;
    try {
        result = run(argc, argv);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Finalize());
    return result;
}
