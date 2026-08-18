#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

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

static void mpiFailure(const char* expression, const char* file, int line, int error) {
    char message[MPI_MAX_ERROR_STRING];
    int messageLength = 0;
    MPI_Error_string(error, message, &messageLength);
    fprintf(stderr, "MPI call failed at %s:%d (%s): %.*s\n", file, line, expression, messageLength, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

#define MPI_CHECK(expression)                                                               \
    do {                                                                                     \
        const int mpiError = (expression);                                                   \
        if (mpiError != MPI_SUCCESS) {                                                       \
            mpiFailure(#expression, __FILE__, __LINE__, mpiError);                           \
        }                                                                                    \
    } while (false)

static void cudaFailure(const char* expression, const char* file, int line, cudaError_t error) {
    fprintf(stderr, "CUDA call failed at %s:%d (%s): %s\n", file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(expression)                                                               \
    do {                                                                                     \
        const cudaError_t cudaError = (expression);                                          \
        if (cudaError != cudaSuccess) {                                                      \
            cudaFailure(#expression, __FILE__, __LINE__, cudaError);                         \
        }                                                                                    \
    } while (false)

// One CUDA block processes a consecutive group of target bodies. The source
// positions are tiled through shared memory, reducing global loads from the
// all-pairs interaction while retaining the original j-loop order.
__global__ void computeForcesKernel(const double* __restrict__ posX,
                                    const double* __restrict__ posY,
                                    const double* __restrict__ posZ,
                                    double* __restrict__ velX,
                                    double* __restrict__ velY,
                                    double* __restrict__ velZ,
                                    const int numBodies,
                                    const int localBegin,
                                    const int localCount) {
    extern __shared__ double sharedPositions[];
    double* tileX = sharedPositions;
    double* tileY = tileX + blockDim.x;
    double* tileZ = tileY + blockDim.x;

    const int localIndex = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const bool active = localIndex < localCount;
    const int bodyIndex = localBegin + localIndex;

    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    double ix = 0.0;
    double iy = 0.0;
    double iz = 0.0;
    if (active) {
        ix = posX[bodyIndex];
        iy = posY[bodyIndex];
        iz = posZ[bodyIndex];
    }

    for (int tileStart = 0; tileStart < numBodies; tileStart += blockDim.x) {
        const int sourceIndex = tileStart + static_cast<int>(threadIdx.x);
        if (sourceIndex < numBodies) {
            tileX[threadIdx.x] = posX[sourceIndex];
            tileY[threadIdx.x] = posY[sourceIndex];
            tileZ[threadIdx.x] = posZ[sourceIndex];
        }
        __syncthreads();

        const int tileCount = (numBodies - tileStart < blockDim.x) ? numBodies - tileStart : blockDim.x;
        if (active) {
            for (int k = 0; k < tileCount; ++k) {
                const double dx = tileX[k] - ix;
                const double dy = tileY[k] - iy;
                const double dz = tileZ[k] - iz;
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
        velX[localIndex] += DT * fx;
        velY[localIndex] += DT * fy;
        velZ[localIndex] += DT * fz;
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

class CudaNBody {
public:
    CudaNBody(const int numBodies, const int localBegin, const int localCount)
        : numBodies_(numBodies), localBegin_(localBegin), localCount_(localCount) {
        // A one-element allocation keeps the zero-body/zero-local-body case
        // valid without ever launching a zero-sized CUDA grid.
        const size_t positionAllocation = std::max<size_t>(1, static_cast<size_t>(numBodies_)) * sizeof(double);
        const size_t velocityAllocation = std::max<size_t>(1, static_cast<size_t>(localCount_)) * sizeof(double);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&posX_), positionAllocation));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&posY_), positionAllocation));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&posZ_), positionAllocation));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&velX_), velocityAllocation));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&velY_), velocityAllocation));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&velZ_), velocityAllocation));
    }

    CudaNBody(const CudaNBody&) = delete;
    CudaNBody& operator=(const CudaNBody&) = delete;

    ~CudaNBody() {
        cudaFree(posX_);
        cudaFree(posY_);
        cudaFree(posZ_);
        cudaFree(velX_);
        cudaFree(velY_);
        cudaFree(velZ_);
    }

    void uploadPositions(const std::vector<double>& posX,
                         const std::vector<double>& posY,
                         const std::vector<double>& posZ) {
        if (numBodies_ == 0) {
            return;
        }
        CUDA_CHECK(cudaMemcpy(posX_, posX.data(), static_cast<size_t>(numBodies_) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(posY_, posY.data(), static_cast<size_t>(numBodies_) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(posZ_, posZ.data(), static_cast<size_t>(numBodies_) * sizeof(double), cudaMemcpyHostToDevice));
    }

    void uploadLocalVelocities(const std::vector<double>& velX,
                               const std::vector<double>& velY,
                               const std::vector<double>& velZ) {
        if (localCount_ == 0) {
            return;
        }
        CUDA_CHECK(cudaMemcpy(velX_, velX.data() + localBegin_, static_cast<size_t>(localCount_) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(velY_, velY.data() + localBegin_, static_cast<size_t>(localCount_) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(velZ_, velZ.data() + localBegin_, static_cast<size_t>(localCount_) * sizeof(double), cudaMemcpyHostToDevice));
    }

    void computeForces() {
        if (localCount_ == 0) {
            return;
        }
        const int gridSize = (localCount_ + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        const size_t sharedMemoryBytes = 3 * CUDA_BLOCK_SIZE * sizeof(double);
        computeForcesKernel<<<gridSize, CUDA_BLOCK_SIZE, sharedMemoryBytes>>>(
            posX_, posY_, posZ_, velX_, velY_, velZ_, numBodies_, localBegin_, localCount_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void downloadLocalVelocities(std::vector<double>& velX,
                                  std::vector<double>& velY,
                                  std::vector<double>& velZ) const {
        if (localCount_ == 0) {
            return;
        }
        CUDA_CHECK(cudaMemcpy(velX.data(), velX_, static_cast<size_t>(localCount_) * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velY.data(), velY_, static_cast<size_t>(localCount_) * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velZ.data(), velZ_, static_cast<size_t>(localCount_) * sizeof(double), cudaMemcpyDeviceToHost));
    }

private:
    int numBodies_;
    int localBegin_;
    int localCount_;
    double* posX_ = nullptr;
    double* posY_ = nullptr;
    double* posZ_ = nullptr;
    double* velX_ = nullptr;
    double* velY_ = nullptr;
    double* velZ_ = nullptr;
};

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const long long n = static_cast<long long>(bodies.size());

    // The reduction is independent per outer-loop body and is safe to run on
    // all OpenMP threads. It retains the original pairwise potential exactly.
#pragma omp parallel for reduction(+ : energy) schedule(static)
    for (long long i = 0; i < n; ++i) {
        const Body& body = bodies[static_cast<size_t>(i)];
        double bodyEnergy = 0.5 * (body.vel.x * body.vel.x +
                                   body.vel.y * body.vel.y +
                                   body.vel.z * body.vel.z);
        for (long long j = i + 1; j < n; ++j) {
            const Body& other = bodies[static_cast<size_t>(j)];
            const double dx = other.pos.x - body.pos.x;
            const double dy = other.pos.y - body.pos.y;
            const double dz = other.pos.z - body.pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            bodyEnergy -= 1.0 / dist;
        }
        energy += bodyEnergy;
    }
    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    int invalid = 0;
#pragma omp parallel for reduction(| : invalid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(bodies.size()); ++i) {
        const Body& body = bodies[static_cast<size_t>(i)];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z) ||
            std::abs(body.pos.x) > 1e6 || std::abs(body.pos.y) > 1e6 || std::abs(body.pos.z) > 1e6 ||
            std::abs(body.vel.x) > 1e6 || std::abs(body.vel.y) > 1e6 || std::abs(body.vel.z) > 1e6) {
            invalid = 1;
        }
    }
    if (invalid != 0) {
        printf("Validation failed: body state contains NaN, Inf, or unreasonable values\n");
        return false;
    }
    return true;
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

static int runSimulation(const int numBodies,
                         const int numSteps,
                         const bool validate,
                         const bool printResults,
                         const int rank,
                         const int worldSize) {
    const int baseCount = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    const int localBegin = rank * baseCount + std::min(rank, remainder);
    const int localCount = baseCount + (rank < remainder ? 1 : 0);

    std::vector<int> bodyCounts(static_cast<size_t>(worldSize));
    std::vector<int> bodyDisplacements(static_cast<size_t>(worldSize));
    for (int process = 0; process < worldSize; ++process) {
        bodyCounts[static_cast<size_t>(process)] = baseCount + (process < remainder ? 1 : 0);
        bodyDisplacements[static_cast<size_t>(process)] = process * baseCount + std::min(process, remainder);
    }

    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    MPI_CHECK(MPI_Bcast(bodies.data(), numBodies * static_cast<int>(sizeof(Body)), MPI_BYTE, 0, MPI_COMM_WORLD));

    std::vector<double> posX(static_cast<size_t>(numBodies));
    std::vector<double> posY(static_cast<size_t>(numBodies));
    std::vector<double> posZ(static_cast<size_t>(numBodies));
    std::vector<double> velX(static_cast<size_t>(numBodies));
    std::vector<double> velY(static_cast<size_t>(numBodies));
    std::vector<double> velZ(static_cast<size_t>(numBodies));

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(numBodies); ++i) {
        const Body& body = bodies[static_cast<size_t>(i)];
        posX[static_cast<size_t>(i)] = body.pos.x;
        posY[static_cast<size_t>(i)] = body.pos.y;
        posZ[static_cast<size_t>(i)] = body.pos.z;
        velX[static_cast<size_t>(i)] = body.vel.x;
        velY[static_cast<size_t>(i)] = body.vel.y;
        velZ[static_cast<size_t>(i)] = body.vel.z;
    }

    CudaNBody accelerator(numBodies, localBegin, localCount);
    accelerator.uploadPositions(posX, posY, posZ);
    accelerator.uploadLocalVelocities(velX, velY, velZ);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();

    std::vector<double> localVelX(static_cast<size_t>(localCount));
    std::vector<double> localVelY(static_cast<size_t>(localCount));
    std::vector<double> localVelZ(static_cast<size_t>(localCount));
    for (int step = 0; step < numSteps; ++step) {
        accelerator.computeForces();
        accelerator.downloadLocalVelocities(localVelX, localVelY, localVelZ);

        MPI_CHECK(MPI_Allgatherv(localVelX.data(), localCount, MPI_DOUBLE,
                                 velX.data(), bodyCounts.data(), bodyDisplacements.data(), MPI_DOUBLE,
                                 MPI_COMM_WORLD));
        MPI_CHECK(MPI_Allgatherv(localVelY.data(), localCount, MPI_DOUBLE,
                                 velY.data(), bodyCounts.data(), bodyDisplacements.data(), MPI_DOUBLE,
                                 MPI_COMM_WORLD));
        MPI_CHECK(MPI_Allgatherv(localVelZ.data(), localCount, MPI_DOUBLE,
                                 velZ.data(), bodyCounts.data(), bodyDisplacements.data(), MPI_DOUBLE,
                                 MPI_COMM_WORLD));

        // Every rank integrates its complete replicated state. This keeps the
        // next CUDA force evaluation local and requires only velocity exchange.
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(numBodies); ++i) {
            const size_t index = static_cast<size_t>(i);
            posX[index] += velX[index] * DT;
            posY[index] += velY[index] * DT;
            posZ[index] += velZ[index] * DT;
        }
        accelerator.uploadPositions(posX, posY, posZ);
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(numBodies); ++i) {
        const size_t index = static_cast<size_t>(i);
        bodies[index].pos.x = posX[index];
        bodies[index].pos.y = posY[index];
        bodies[index].pos.z = posZ[index];
        bodies[index].vel.x = velX[index];
        bodies[index].vel.y = velY[index];
        bodies[index].vel.z = velZ[index];
    }

    int result = 0;
    if (rank == 0) {
        const long simulationMilliseconds = static_cast<long>(elapsed * 1000.0);
        printf("Simulation time: %ld ms\n", simulationMilliseconds);

        if (printResults) {
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

        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(bodies)) {
                printf("Final energy: %.6f\n", computeTotalEnergy(bodies));
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return result;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int initError = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (initError != MPI_SUCCESS) {
        return 1;
    }
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    int rank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

    int localRank = 0;
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localCommunicator));
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices are available for MPI rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseError = true;
        }
    }

    if (parseError || numBodies < 0 || numSteps < 0) {
        if (rank == 0 && !parseError) {
            printf("Number of bodies and steps must be non-negative\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int result = runSimulation(numBodies, numSteps, validate, printResults, rank, worldSize);
    MPI_CHECK(MPI_Finalize());
    return result;
}
