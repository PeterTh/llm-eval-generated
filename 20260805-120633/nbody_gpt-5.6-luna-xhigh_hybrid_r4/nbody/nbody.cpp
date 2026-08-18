#include <mpi.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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

struct Position {
    double x, y, z;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be tightly packed");
static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be tightly packed");
static_assert(sizeof(Position) == 3 * sizeof(double), "Position must be tightly packed");

// The original rand_r sequence is retained.  Values are generated serially,
// then assigned in parallel so changing the OpenMP team size does not change
// the benchmark's initial condition.
void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    const size_t n = bodies.size();
    std::vector<unsigned int> randomValues(n * 6);
    unsigned int state = seed;
    for (auto& value : randomValues) {
        value = rand_r(&state);
    }

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        const size_t offset = static_cast<size_t>(i) * 6;
        Body& body = bodies[static_cast<size_t>(i)];
        body.pos.x = 2.0 * (randomValues[offset + 0] / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (randomValues[offset + 1] / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (randomValues[offset + 2] / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (randomValues[offset + 3] / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (randomValues[offset + 4] / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (randomValues[offset + 5] / (double)RAND_MAX) - 1.0;
    }
}

void bodiesToPositions(const std::vector<Body>& bodies, std::vector<Position>& positions) {
    const long long n = static_cast<long long>(bodies.size());
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        positions[static_cast<size_t>(i)] = {
            bodies[static_cast<size_t>(i)].pos.x,
            bodies[static_cast<size_t>(i)].pos.y,
            bodies[static_cast<size_t>(i)].pos.z};
    }
}

void positionsToSoA(const std::vector<Position>& positions,
                    std::vector<double>& x,
                    std::vector<double>& y,
                    std::vector<double>& z) {
    const long long n = static_cast<long long>(positions.size());
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        const Position& position = positions[static_cast<size_t>(i)];
        x[static_cast<size_t>(i)] = position.x;
        y[static_cast<size_t>(i)] = position.y;
        z[static_cast<size_t>(i)] = position.z;
    }
}

void positionsToBodies(const std::vector<Position>& positions, std::vector<Body>& bodies) {
    const long long n = static_cast<long long>(positions.size());
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        const Position& position = positions[static_cast<size_t>(i)];
        Body& body = bodies[static_cast<size_t>(i)];
        body.pos.x = position.x;
        body.pos.y = position.y;
        body.pos.z = position.z;
    }
}

__global__ void computeForcesAndIntegrateKernel(const double* __restrict__ x,
                                                 const double* __restrict__ y,
                                                 const double* __restrict__ z,
                                                 Body* __restrict__ localBodies,
                                                 const int bodyCount,
                                                 const int firstBody,
                                                 const int localBodyCount) {
    extern __shared__ double sharedPositions[];
    double* tileX = sharedPositions;
    double* tileY = tileX + blockDim.x;
    double* tileZ = tileY + blockDim.x;

    const int lane = static_cast<int>(threadIdx.x);
    const int localIndex = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + lane;
    const bool active = localIndex < localBodyCount;

    double xi = 0.0;
    double yi = 0.0;
    double zi = 0.0;
    if (active) {
        const Body& body = localBodies[localIndex];
        xi = body.pos.x;
        yi = body.pos.y;
        zi = body.pos.z;
    }

    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    // Every thread walks j in increasing order, matching the scalar
    // reference algorithm while the target-body dimension is parallel.
    for (int tileStart = 0; tileStart < bodyCount; tileStart += blockDim.x) {
        const int sourceIndex = tileStart + lane;
        if (sourceIndex < bodyCount) {
            tileX[lane] = x[sourceIndex];
            tileY[lane] = y[sourceIndex];
            tileZ[lane] = z[sourceIndex];
        } else {
            tileX[lane] = 0.0;
            tileY[lane] = 0.0;
            tileZ[lane] = 0.0;
        }
        __syncthreads();

        const int tileCount = min(static_cast<int>(blockDim.x), bodyCount - tileStart);
        if (active) {
            for (int k = 0; k < tileCount; ++k) {
                const double dx = tileX[k] - xi;
                const double dy = tileY[k] - yi;
                const double dz = tileZ[k] - zi;
                const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inverseDistance = __ddiv_rn(1.0, sqrt(distanceSquared));
                const double inverseDistanceCubed = inverseDistance * inverseDistance * inverseDistance;

                forceX += dx * inverseDistanceCubed;
                forceY += dy * inverseDistanceCubed;
                forceZ += dz * inverseDistanceCubed;
            }
        }
        __syncthreads();
    }

    if (active) {
        Body& body = localBodies[localIndex];
        body.vel.x += DT * forceX;
        body.vel.y += DT * forceY;
        body.vel.z += DT * forceZ;
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

[[noreturn]] void abortCuda(MPI_Comm communicator, int rank, const char* operation, cudaError_t error) {
    std::fprintf(stderr, "Rank %d: CUDA operation '%s' failed: %s\n", rank, operation, cudaGetErrorString(error));
    MPI_Abort(communicator, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(comm, rank, operation) \
    do { \
        const cudaError_t cudaStatus = (operation); \
        if (cudaStatus != cudaSuccess) { \
            abortCuda((comm), (rank), #operation, cudaStatus); \
        } \
    } while (false)

class CudaNBodySolver {
  public:
    CudaNBodySolver(MPI_Comm communicator, int rank, int bodyCount, int firstBody, int localBodyCount)
        : communicator_(communicator), rank_(rank), bodyCount_(bodyCount), firstBody_(firstBody), localBodyCount_(localBodyCount) {
        int deviceCount = 0;
        CUDA_CHECK(communicator_, rank_, cudaGetDeviceCount(&deviceCount));
        if (deviceCount <= 0) {
            std::fprintf(stderr, "Rank %d: no CUDA device is available\n", rank_);
            MPI_Abort(communicator_, EXIT_FAILURE);
        }

        int localRank = 0;
        MPI_Comm nodeCommunicator = MPI_COMM_NULL;
        if (MPI_Comm_split_type(communicator_, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &nodeCommunicator) != MPI_SUCCESS ||
            MPI_Comm_rank(nodeCommunicator, &localRank) != MPI_SUCCESS) {
            std::fprintf(stderr, "Rank %d: unable to determine MPI local rank\n", rank_);
            MPI_Abort(communicator_, EXIT_FAILURE);
        }
        MPI_Comm_free(&nodeCommunicator);

        device_ = localRank % deviceCount;
        CUDA_CHECK(communicator_, rank_, cudaSetDevice(device_));
        CUDA_CHECK(communicator_, rank_, cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

        const size_t globalAllocation = std::max<size_t>(1, static_cast<size_t>(bodyCount_));
        const size_t localAllocation = std::max<size_t>(1, static_cast<size_t>(localBodyCount_));
        CUDA_CHECK(communicator_, rank_, cudaMalloc(reinterpret_cast<void**>(&deviceX_), globalAllocation * sizeof(double)));
        CUDA_CHECK(communicator_, rank_, cudaMalloc(reinterpret_cast<void**>(&deviceY_), globalAllocation * sizeof(double)));
        CUDA_CHECK(communicator_, rank_, cudaMalloc(reinterpret_cast<void**>(&deviceZ_), globalAllocation * sizeof(double)));
        CUDA_CHECK(communicator_, rank_, cudaMalloc(reinterpret_cast<void**>(&deviceLocalBodies_), localAllocation * sizeof(Body)));
    }

    ~CudaNBodySolver() {
        cudaFree(deviceLocalBodies_);
        cudaFree(deviceZ_);
        cudaFree(deviceY_);
        cudaFree(deviceX_);
        cudaStreamDestroy(stream_);
    }

    CudaNBodySolver(const CudaNBodySolver&) = delete;
    CudaNBodySolver& operator=(const CudaNBodySolver&) = delete;

    void initialize(const std::vector<double>& x,
                    const std::vector<double>& y,
                    const std::vector<double>& z,
                    const Body* localBodies) {
        if (bodyCount_ > 0) {
            CUDA_CHECK(communicator_, rank_, cudaMemcpy(deviceX_, x.data(), static_cast<size_t>(bodyCount_) * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(communicator_, rank_, cudaMemcpy(deviceY_, y.data(), static_cast<size_t>(bodyCount_) * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(communicator_, rank_, cudaMemcpy(deviceZ_, z.data(), static_cast<size_t>(bodyCount_) * sizeof(double), cudaMemcpyHostToDevice));
        }
        if (localBodyCount_ > 0) {
            CUDA_CHECK(communicator_, rank_, cudaMemcpy(deviceLocalBodies_, localBodies,
                                                        static_cast<size_t>(localBodyCount_) * sizeof(Body), cudaMemcpyHostToDevice));
        }
    }

    void step(const std::vector<double>& x,
              const std::vector<double>& y,
              const std::vector<double>& z,
              Body* localBodies) {
        if (localBodyCount_ == 0) {
            return;
        }

        const size_t globalBytes = static_cast<size_t>(bodyCount_) * sizeof(double);
        CUDA_CHECK(communicator_, rank_, cudaMemcpyAsync(deviceX_, x.data(), globalBytes, cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(communicator_, rank_, cudaMemcpyAsync(deviceY_, y.data(), globalBytes, cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(communicator_, rank_, cudaMemcpyAsync(deviceZ_, z.data(), globalBytes, cudaMemcpyHostToDevice, stream_));

        const int blockCount = (localBodyCount_ + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        const size_t sharedBytes = 3 * static_cast<size_t>(CUDA_BLOCK_SIZE) * sizeof(double);
        computeForcesAndIntegrateKernel<<<blockCount, CUDA_BLOCK_SIZE, sharedBytes, stream_>>>(
            deviceX_, deviceY_, deviceZ_, deviceLocalBodies_, bodyCount_, firstBody_, localBodyCount_);
        CUDA_CHECK(communicator_, rank_, cudaGetLastError());

        CUDA_CHECK(communicator_, rank_, cudaMemcpyAsync(localBodies, deviceLocalBodies_,
                                                        static_cast<size_t>(localBodyCount_) * sizeof(Body),
                                                        cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(communicator_, rank_, cudaStreamSynchronize(stream_));
    }

    int device() const noexcept { return device_; }

  private:
    MPI_Comm communicator_;
    int rank_;
    int bodyCount_;
    int firstBody_;
    int localBodyCount_;
    int device_ = 0;
    cudaStream_t stream_ = nullptr;
    double* deviceX_ = nullptr;
    double* deviceY_ = nullptr;
    double* deviceZ_ = nullptr;
    Body* deviceLocalBodies_ = nullptr;
};

void mpiAbortIf(bool condition, MPI_Comm communicator, int rank, const char* message) {
    if (condition) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, message);
        MPI_Abort(communicator, EXIT_FAILURE);
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const long long n = static_cast<long long>(bodies.size());
    double kineticEnergy = 0.0;
    #pragma omp parallel for reduction(+:kineticEnergy) schedule(static)
    for (long long i = 0; i < n; ++i) {
        const Body& body = bodies[static_cast<size_t>(i)];
        kineticEnergy += 0.5 * (body.vel.x * body.vel.x +
                                body.vel.y * body.vel.y +
                                body.vel.z * body.vel.z);
    }

    double potentialEnergy = 0.0;
    #pragma omp parallel for reduction(+:potentialEnergy) schedule(static)
    for (long long i = 0; i < n; ++i) {
        for (long long j = i + 1; j < n; ++j) {
            const Body& first = bodies[static_cast<size_t>(i)];
            const Body& second = bodies[static_cast<size_t>(j)];
            const double dx = second.pos.x - first.pos.x;
            const double dy = second.pos.y - first.pos.y;
            const double dz = second.pos.z - first.pos.z;
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potentialEnergy -= 1.0 / distance;
        }
    }
    return kineticEnergy + potentialEnergy;
}

// Validate that simulation produces finite, reasonable values.
bool validateSimulation(const std::vector<Body>& bodies) {
    int nonFinite = 0;
    int extremePosition = 0;
    int extremeVelocity = 0;
    const long long n = static_cast<long long>(bodies.size());

    #pragma omp parallel for reduction(max:nonFinite, extremePosition, extremeVelocity) schedule(static)
    for (long long i = 0; i < n; ++i) {
        const Body& body = bodies[static_cast<size_t>(i)];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            nonFinite = 1;
        }
        if (std::abs(body.pos.x) > 1e6 || std::abs(body.pos.y) > 1e6 || std::abs(body.pos.z) > 1e6) {
            extremePosition = 1;
        }
        if (std::abs(body.vel.x) > 1e6 || std::abs(body.vel.y) > 1e6 || std::abs(body.vel.z) > 1e6) {
            extremeVelocity = 1;
        }
    }

    if (nonFinite != 0) {
        std::printf("Validation failed: found NaN or Inf value in body state\n");
        return false;
    }
    if (extremePosition != 0) {
        std::printf("Validation failed: body position exceeds reasonable bounds\n");
        return false;
    }
    if (extremeVelocity != 0) {
        std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
        return false;
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

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    mpiAbortIf(providedThreadLevel < MPI_THREAD_FUNNELED, MPI_COMM_WORLD, rank,
               "MPI implementation does not provide MPI_THREAD_FUNNELED");

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
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    mpiAbortIf(numBodies < 0, MPI_COMM_WORLD, rank, "number of bodies must be non-negative");
    mpiAbortIf(numSteps < 0, MPI_COMM_WORLD, rank, "number of steps must be non-negative");
    mpiAbortIf(static_cast<unsigned long long>(numBodies) * sizeof(Body) >
                   static_cast<unsigned long long>(std::numeric_limits<int>::max()),
               MPI_COMM_WORLD, rank, "number of bodies exceeds MPI count limits");

    const int baseBodyCount = numBodies / worldSize;
    const int extraBodies = numBodies % worldSize;
    const int localBodyCount = baseBodyCount + (rank < extraBodies ? 1 : 0);
    const int firstBody = rank * baseBodyCount + std::min(rank, extraBodies);

    std::vector<int> positionCounts(worldSize);
    std::vector<int> positionDisplacements(worldSize);
    std::vector<int> bodyCounts(worldSize);
    std::vector<int> bodyDisplacements(worldSize);
    for (int process = 0; process < worldSize; ++process) {
        const int processCount = baseBodyCount + (process < extraBodies ? 1 : 0);
        const int processFirst = process * baseBodyCount + std::min(process, extraBodies);
        positionCounts[process] = processCount * static_cast<int>(sizeof(Position));
        positionDisplacements[process] = processFirst * static_cast<int>(sizeof(Position));
        bodyCounts[process] = processCount * static_cast<int>(sizeof(Body));
        bodyDisplacements[process] = processFirst * static_cast<int>(sizeof(Body));
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", worldSize);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
    }

    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    MPI_Bcast(bodies.data(), numBodies * static_cast<int>(sizeof(Body)), MPI_BYTE, 0, MPI_COMM_WORLD);

    std::vector<Position> positions(static_cast<size_t>(numBodies));
    std::vector<Position> localPositions(static_cast<size_t>(localBodyCount));
    std::vector<double> positionX(static_cast<size_t>(numBodies));
    std::vector<double> positionY(static_cast<size_t>(numBodies));
    std::vector<double> positionZ(static_cast<size_t>(numBodies));
    bodiesToPositions(bodies, positions);
    positionsToSoA(positions, positionX, positionY, positionZ);

    Body* localBodies = localBodyCount > 0 ? bodies.data() + firstBody : nullptr;
    CudaNBodySolver* solver = new CudaNBodySolver(MPI_COMM_WORLD, rank, numBodies, firstBody, localBodyCount);
    solver->initialize(positionX, positionY, positionZ, localBodies);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        solver->step(positionX, positionY, positionZ, localBodies);

        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(localBodyCount); ++i) {
            const Body& body = bodies[static_cast<size_t>(firstBody) + static_cast<size_t>(i)];
            localPositions[static_cast<size_t>(i)] = {body.pos.x, body.pos.y, body.pos.z};
        }

        MPI_Allgatherv(localPositions.data(), localBodyCount * static_cast<int>(sizeof(Position)), MPI_BYTE,
                       positions.data(), positionCounts.data(), positionDisplacements.data(), MPI_BYTE,
                       MPI_COMM_WORLD);
        positionsToBodies(positions, bodies);
        positionsToSoA(positions, positionX, positionY, positionZ);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    delete solver;

    long long localMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long elapsedMilliseconds = 0;
    MPI_Reduce(&localMilliseconds, &elapsedMilliseconds, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Only rank 0 needs the final velocities.  Positions have already been
    // replicated for the final force step and are present on every rank.
    if (rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_BYTE,
                    bodies.data(), bodyCounts.data(), bodyDisplacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(localBodies, localBodyCount * static_cast<int>(sizeof(Body)), MPI_BYTE,
                    bodies.data(), bodyCounts.data(), bodyDisplacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    int returnCode = 0;
    if (rank == 0) {
        std::printf("Simulation time: %lld ms\n", elapsedMilliseconds);

        if (printResults) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < static_cast<long long>(numBodies); ++i) {
                const Body& body = bodies[static_cast<size_t>(i)];
                const size_t offset = static_cast<size_t>(i) * 6;
                bodyData[offset + 0] = body.pos.x;
                bodyData[offset + 1] = body.pos.y;
                bodyData[offset + 2] = body.pos.z;
                bodyData[offset + 3] = body.vel.x;
                bodyData[offset + 4] = body.vel.y;
                bodyData[offset + 5] = body.vel.z;
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            std::printf("Validating simulation results...\n");
            const bool valid = validateSimulation(bodies);
            if (valid) {
                const double finalEnergy = computeTotalEnergy(bodies);
                std::printf("Final energy: %.6f\n", finalEnergy);
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
            }
            returnCode = valid ? 0 : 1;
        }
    }

    if (validate) {
        MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return returnCode;
}

#undef CUDA_CHECK
