#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int CUDA_BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be a contiguous six-double MPI record");

[[noreturn]] void abortWithCudaError(cudaError_t error, const char* expression, int rank) {
    std::fprintf(stderr, "Rank %d: CUDA error in %s: %s\n", rank, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(rank, call)                                                           \
    do {                                                                                  \
        const cudaError_t cudaStatus = (call);                                            \
        if (cudaStatus != cudaSuccess) {                                                  \
            abortWithCudaError(cudaStatus, #call, (rank));                                \
        }                                                                                 \
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

// Each CUDA thread owns one target body.  Source positions are staged in shared
// memory so that every global load supplies an entire block of target bodies.
// Velocity and position updates deliberately use separate kernels: this keeps
// the force calculation dependent only on positions from the preceding step.
__global__ void computeVelocityKernel(Body* __restrict__ bodies, int numBodies, int firstBody, int localCount) {
    __shared__ double sourceX[CUDA_BLOCK_SIZE];
    __shared__ double sourceY[CUDA_BLOCK_SIZE];
    __shared__ double sourceZ[CUDA_BLOCK_SIZE];

    const int lane = threadIdx.x;
    const int localIndex = blockIdx.x * blockDim.x + lane;
    const bool active = localIndex < localCount;
    const int bodyIndex = firstBody + localIndex;

    const double x = active ? bodies[bodyIndex].pos.x : 0.0;
    const double y = active ? bodies[bodyIndex].pos.y : 0.0;
    const double z = active ? bodies[bodyIndex].pos.z : 0.0;
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileStart = 0; tileStart < numBodies; tileStart += CUDA_BLOCK_SIZE) {
        const int sourceIndex = tileStart + lane;
        if (sourceIndex < numBodies) {
            sourceX[lane] = bodies[sourceIndex].pos.x;
            sourceY[lane] = bodies[sourceIndex].pos.y;
            sourceZ[lane] = bodies[sourceIndex].pos.z;
        }
        __syncthreads();

        const int tileSize = min(CUDA_BLOCK_SIZE, numBodies - tileStart);
        if (active) {
            for (int source = 0; source < tileSize; ++source) {
                const double dx = sourceX[source] - x;
                const double dy = sourceY[source] - y;
                const double dz = sourceZ[source] - z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                forceX += dx * invDist3;
                forceY += dy * invDist3;
                forceZ += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        bodies[bodyIndex].vel.x += DT * forceX;
        bodies[bodyIndex].vel.y += DT * forceY;
        bodies[bodyIndex].vel.z += DT * forceZ;
    }
}

__global__ void integrateKernel(Body* __restrict__ bodies, int firstBody, int localCount) {
    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIndex >= localCount) {
        return;
    }

    Body& body = bodies[firstBody + localIndex];
    body.pos.x += body.vel.x * DT;
    body.pos.y += body.vel.y * DT;
    body.pos.z += body.vel.z * DT;
}

// This CPU portion runs concurrently with the GPU portion.  It is useful on
// GPU clusters with substantial host CPU capacity and is load-balanced from
// measured rates after each step.
void computeVelocitiesCpu(std::vector<Body>& bodies, int firstBody, int count) {
    if (count == 0) {
        return;
    }

    const int lastBody = firstBody + count;
    const int threadCount = std::min(omp_get_max_threads(), count);
#pragma omp parallel for schedule(static) num_threads(threadCount)
    for (int i = firstBody; i < lastBody; ++i) {
        const double x = bodies[i].pos.x;
        const double y = bodies[i].pos.y;
        const double z = bodies[i].pos.z;
        double forceX = 0.0;
        double forceY = 0.0;
        double forceZ = 0.0;

        for (int j = 0; j < static_cast<int>(bodies.size()); ++j) {
            const double dx = bodies[j].pos.x - x;
            const double dy = bodies[j].pos.y - y;
            const double dz = bodies[j].pos.z - z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            forceX += dx * invDist3;
            forceY += dy * invDist3;
            forceZ += dz * invDist3;
        }

        bodies[i].vel.x += DT * forceX;
        bodies[i].vel.y += DT * forceY;
        bodies[i].vel.z += DT * forceZ;
    }
}

void integrateBodiesCpu(std::vector<Body>& bodies, int firstBody, int count) {
    if (count == 0) {
        return;
    }

    const int lastBody = firstBody + count;
    const int threadCount = std::min(omp_get_max_threads(), count);
#pragma omp parallel for schedule(static) num_threads(threadCount)
    for (int i = firstBody; i < lastBody; ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            std::printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        constexpr double maxPos = 1e6;
        constexpr double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            std::printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
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

int initialCpuTargetCount(int localCount, int numBodies) {
    if (localCount < 2 || numBodies == 0 || omp_get_max_threads() < 2) {
        return 0;
    }

    // A small initial slice is measured and then adjusted.  Avoid scheduling a
    // CPU team for tiny kernels, where launch and OpenMP setup costs dominate.
    const int initialCount = std::max(1, localCount / 64);
    constexpr long long minimumCpuInteractions = 50LL * 1000LL * 1000LL;
    if (static_cast<long long>(initialCount) * numBodies < minimumCpuInteractions) {
        return 0;
    }
    return std::min(initialCount, localCount - 1);
}

int rebalanceCpuTargetCount(int localCount, int previousCpuCount, float cpuMilliseconds, float gpuMilliseconds) {
    const int gpuCount = localCount - previousCpuCount;
    if (previousCpuCount == 0 || gpuCount == 0 || cpuMilliseconds <= 0.0F || gpuMilliseconds <= 0.0F) {
        return previousCpuCount;
    }

    const double cpuRate = previousCpuCount / static_cast<double>(cpuMilliseconds);
    const double gpuRate = gpuCount / static_cast<double>(gpuMilliseconds);
    const int balancedCount = static_cast<int>(std::lround(localCount * cpuRate / (cpuRate + gpuRate)));
    return std::clamp(balancedCount, 1, localCount - 1);
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide the MPI_THREAD_FUNNELED thread level required by nbody\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    int localRank = 0;
    int localSize = 1;
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);

    // Honour an explicit OpenMP setting; otherwise avoid oversubscribing CPUs
    // when one MPI rank is launched for each GPU on a node.
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    int deviceCount = 0;
    CUDA_CHECK(rank, cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "nbody requires at least one CUDA device per MPI node\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    CUDA_CHECK(rank, cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return EXIT_SUCCESS;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) {
            std::fprintf(stderr, "Number of bodies and steps must be non-negative\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", worldSize, omp_get_max_threads());
    }

    std::vector<int> receiveCounts(worldSize);
    std::vector<int> displacements(worldSize);
    const int baseCount = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    int displacement = 0;
    for (int process = 0; process < worldSize; ++process) {
        receiveCounts[process] = baseCount + (process < remainder ? 1 : 0);
        displacements[process] = displacement;
        displacement += receiveCounts[process];
    }
    const int localCount = receiveCounts[rank];
    const int firstBody = displacements[rank];

    MPI_Datatype mpiBody;
    MPI_Type_contiguous(6, MPI_DOUBLE, &mpiBody);
    MPI_Type_commit(&mpiBody);

    // Only positions are needed as force sources at the next step.  The
    // resized datatype leaves each rank's locally-owned velocity components
    // untouched and cuts per-step network traffic in half.
    MPI_Datatype packedPosition;
    MPI_Datatype mpiPosition;
    MPI_Type_contiguous(3, MPI_DOUBLE, &packedPosition);
    MPI_Type_create_resized(packedPosition, 0, static_cast<MPI_Aint>(sizeof(Body)), &mpiPosition);
    MPI_Type_commit(&mpiPosition);
    MPI_Type_free(&packedPosition);

    std::vector<Body> bodies(numBodies);
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    if (numBodies > 0) {
        MPI_Bcast(bodies.data(), numBodies, mpiBody, 0, MPI_COMM_WORLD);
    }

    Body* deviceBodies = nullptr;
    if (numBodies > 0) {
        CUDA_CHECK(rank, cudaMalloc(&deviceBodies, static_cast<size_t>(numBodies) * sizeof(Body)));
        CUDA_CHECK(rank, cudaMemcpy(deviceBodies, bodies.data(), static_cast<size_t>(numBodies) * sizeof(Body),
                                    cudaMemcpyHostToDevice));
    }

    cudaEvent_t gpuStart;
    cudaEvent_t gpuStop;
    CUDA_CHECK(rank, cudaEventCreate(&gpuStart));
    CUDA_CHECK(rank, cudaEventCreate(&gpuStop));

    int cpuCount = initialCpuTargetCount(localCount, numBodies);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        const int gpuCount = localCount - cpuCount;
        CUDA_CHECK(rank, cudaEventRecord(gpuStart));
        if (gpuCount > 0) {
            const int blocks = (gpuCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            computeVelocityKernel<<<blocks, CUDA_BLOCK_SIZE>>>(deviceBodies, numBodies, firstBody + cpuCount, gpuCount);
            CUDA_CHECK(rank, cudaGetLastError());
            integrateKernel<<<blocks, CUDA_BLOCK_SIZE>>>(deviceBodies, firstBody + cpuCount, gpuCount);
            CUDA_CHECK(rank, cudaGetLastError());
        }
        CUDA_CHECK(rank, cudaEventRecord(gpuStop));

        const auto cpuStart = std::chrono::steady_clock::now();
        computeVelocitiesCpu(bodies, firstBody, cpuCount);
        integrateBodiesCpu(bodies, firstBody, cpuCount);
        const auto cpuStop = std::chrono::steady_clock::now();

        CUDA_CHECK(rank, cudaEventSynchronize(gpuStop));
        float gpuMilliseconds = 0.0F;
        CUDA_CHECK(rank, cudaEventElapsedTime(&gpuMilliseconds, gpuStart, gpuStop));
        const float cpuMilliseconds =
            std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(cpuStop - cpuStart).count();

        if (gpuCount > 0) {
            CUDA_CHECK(rank, cudaMemcpy(bodies.data() + firstBody + cpuCount, deviceBodies + firstBody + cpuCount,
                                        static_cast<size_t>(gpuCount) * sizeof(Body), cudaMemcpyDeviceToHost));
        }

        if (numBodies > 0) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bodies.data(), receiveCounts.data(),
                           displacements.data(), mpiPosition, MPI_COMM_WORLD);
            CUDA_CHECK(rank, cudaMemcpy(deviceBodies, bodies.data(), static_cast<size_t>(numBodies) * sizeof(Body),
                                        cudaMemcpyHostToDevice));
        }

        cpuCount = rebalanceCpuTargetCount(localCount, cpuCount, cpuMilliseconds, gpuMilliseconds);
    }

    const auto end = std::chrono::steady_clock::now();
    const long long localMilliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long elapsedMilliseconds = 0;
    MPI_Reduce(&localMilliseconds, &elapsedMilliseconds, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Simulation time: %lld ms\n", elapsedMilliseconds);
    }

    // Velocities stay local during the timestep exchange.  Gather the complete
    // state once, outside the timed region, only for requested output or
    // root-side validation.
    if ((printResults || validate) && numBodies > 0) {
        if (rank == 0) {
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bodies.data(), receiveCounts.data(), displacements.data(),
                        mpiBody, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(bodies.data() + firstBody, localCount, mpiBody, bodies.data(), receiveCounts.data(),
                        displacements.data(), mpiBody, 0, MPI_COMM_WORLD);
        }
    }

    if (printResults && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (const auto& body : bodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    int validationPassed = 1;
    if (validate && rank == 0) {
        std::printf("Validating simulation results...\n");
        validationPassed = validateSimulation(bodies) ? 1 : 0;
        if (validationPassed != 0) {
            const double finalEnergy = computeTotalEnergy(bodies);
            std::printf("Final energy: %.6f\n", finalEnergy);
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
        }
    }
    if (validate) {
        MPI_Bcast(&validationPassed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(rank, cudaEventDestroy(gpuStart));
    CUDA_CHECK(rank, cudaEventDestroy(gpuStop));
    if (deviceBodies != nullptr) {
        CUDA_CHECK(rank, cudaFree(deviceBodies));
    }
    MPI_Type_free(&mpiPosition);
    MPI_Type_free(&mpiBody);
    MPI_Finalize();
    return validationPassed != 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
