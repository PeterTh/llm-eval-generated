#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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

static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be tightly packed");
static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be tightly packed");

[[noreturn]] void cudaError(const cudaError_t status, const char* expression, const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n", file, line, expression, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    std::abort();
}

#define CUDA_CHECK(expression) \
    do { \
        const cudaError_t status = (expression); \
        if (status != cudaSuccess) { \
            cudaError(status, #expression, __FILE__, __LINE__); \
        } \
    } while (false)

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Keep the original serial rand_r stream so every MPI rank starts with the
    // exact same state, independent of the OpenMP thread count.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// One CUDA thread owns one target body. All ranks hold the complete position
// arrays, while MPI distributes the target-body range among ranks.
__global__ void nbodyStepKernel(const double* __restrict__ posX,
                                const double* __restrict__ posY,
                                const double* __restrict__ posZ,
                                double* __restrict__ localPosX,
                                double* __restrict__ localPosY,
                                double* __restrict__ localPosZ,
                                double* __restrict__ localVelX,
                                double* __restrict__ localVelY,
                                double* __restrict__ localVelZ,
                                const int numBodies,
                                const int firstBody,
                                const int localBodies) {
    __shared__ double tileX[CUDA_BLOCK_SIZE];
    __shared__ double tileY[CUDA_BLOCK_SIZE];
    __shared__ double tileZ[CUDA_BLOCK_SIZE];

    const int localIndex = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = localIndex < localBodies;
    const int bodyIndex = firstBody + localIndex;

    // Inactive threads still participate in every barrier. This allows the
    // last, partially occupied block to use the same tiled loop safely.
    const double xi = active ? posX[bodyIndex] : 0.0;
    const double yi = active ? posY[bodyIndex] : 0.0;
    const double zi = active ? posZ[bodyIndex] : 0.0;
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileStart = 0; tileStart < numBodies; tileStart += CUDA_BLOCK_SIZE) {
        const int sourceIndex = tileStart + static_cast<int>(threadIdx.x);
        if (sourceIndex < numBodies) {
            tileX[threadIdx.x] = posX[sourceIndex];
            tileY[threadIdx.x] = posY[sourceIndex];
            tileZ[threadIdx.x] = posZ[sourceIndex];
        } else {
            tileX[threadIdx.x] = 0.0;
            tileY[threadIdx.x] = 0.0;
            tileZ[threadIdx.x] = 0.0;
        }
        __syncthreads();

        const int tileBodies = min(CUDA_BLOCK_SIZE, numBodies - tileStart);
        if (active) {
            // The order of this loop matches the original j=0..N-1 force
            // accumulation, while shared memory removes redundant global loads.
            #pragma unroll 4
            for (int j = 0; j < tileBodies; ++j) {
                const double dx = tileX[j] - xi;
                const double dy = tileY[j] - yi;
                const double dz = tileZ[j] - zi;
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
        const double newVelX = localVelX[localIndex] + DT * forceX;
        const double newVelY = localVelY[localIndex] + DT * forceY;
        const double newVelZ = localVelZ[localIndex] + DT * forceZ;

        // Force update and integration are deliberately fused, but retain the
        // original ordering: velocity is updated before position is advanced.
        localVelX[localIndex] = newVelX;
        localVelY[localIndex] = newVelY;
        localVelZ[localIndex] = newVelZ;
        localPosX[localIndex] = xi + newVelX * DT;
        localPosY[localIndex] = yi + newVelY * DT;
        localPosZ[localIndex] = zi + newVelZ * DT;
    }
}

class CudaNBody {
public:
    CudaNBody(const int numBodies, const int localBodies)
        : numBodies_(numBodies), localBodies_(localBodies) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        if (numBodies_ > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_posX_), sizeof(double) * numBodies_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_posY_), sizeof(double) * numBodies_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_posZ_), sizeof(double) * numBodies_));
        }
        if (localBodies_ > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_localPosX_), sizeof(double) * localBodies_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_localPosY_), sizeof(double) * localBodies_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_localPosZ_), sizeof(double) * localBodies_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_localVelX_), sizeof(double) * localBodies_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_localVelY_), sizeof(double) * localBodies_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_localVelZ_), sizeof(double) * localBodies_));
        }
    }

    CudaNBody(const CudaNBody&) = delete;
    CudaNBody& operator=(const CudaNBody&) = delete;

    ~CudaNBody() {
        cudaFree(d_posX_);
        cudaFree(d_posY_);
        cudaFree(d_posZ_);
        cudaFree(d_localPosX_);
        cudaFree(d_localPosY_);
        cudaFree(d_localPosZ_);
        cudaFree(d_localVelX_);
        cudaFree(d_localVelY_);
        cudaFree(d_localVelZ_);
        cudaStreamDestroy(stream_);
    }

    void step(const double* posX,
              const double* posY,
              const double* posZ,
              double* localPosX,
              double* localPosY,
              double* localPosZ,
              double* localVelX,
              double* localVelY,
              double* localVelZ,
              const int firstBody) {
        if (numBodies_ > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_posX_, posX, sizeof(double) * numBodies_, cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_posY_, posY, sizeof(double) * numBodies_, cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_posZ_, posZ, sizeof(double) * numBodies_, cudaMemcpyHostToDevice, stream_));
        }

        if (localBodies_ > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_localVelX_, localVelX, sizeof(double) * localBodies_, cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_localVelY_, localVelY, sizeof(double) * localBodies_, cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_localVelZ_, localVelZ, sizeof(double) * localBodies_, cudaMemcpyHostToDevice, stream_));

            const int blocks = (localBodies_ + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            nbodyStepKernel<<<blocks, CUDA_BLOCK_SIZE, 0, stream_>>>(
                d_posX_, d_posY_, d_posZ_,
                d_localPosX_, d_localPosY_, d_localPosZ_,
                d_localVelX_, d_localVelY_, d_localVelZ_,
                numBodies_, firstBody, localBodies_);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(localPosX, d_localPosX_, sizeof(double) * localBodies_, cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaMemcpyAsync(localPosY, d_localPosY_, sizeof(double) * localBodies_, cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaMemcpyAsync(localPosZ, d_localPosZ_, sizeof(double) * localBodies_, cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaMemcpyAsync(localVelX, d_localVelX_, sizeof(double) * localBodies_, cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaMemcpyAsync(localVelY, d_localVelY_, sizeof(double) * localBodies_, cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaMemcpyAsync(localVelZ, d_localVelZ_, sizeof(double) * localBodies_, cudaMemcpyDeviceToHost, stream_));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    int numBodies_ = 0;
    int localBodies_ = 0;
    cudaStream_t stream_ = nullptr;
    double* d_posX_ = nullptr;
    double* d_posY_ = nullptr;
    double* d_posZ_ = nullptr;
    double* d_localPosX_ = nullptr;
    double* d_localPosY_ = nullptr;
    double* d_localPosZ_ = nullptr;
    double* d_localVelX_ = nullptr;
    double* d_localVelY_ = nullptr;
    double* d_localVelZ_ = nullptr;
};

// Host-side staging is parallelized with OpenMP. Keeping the MPI calls out of
// these regions lets MPI_Init_thread use the cheaper FUNNELED thread level.
void prepareStep(const std::vector<Body>& bodies,
                 const int firstBody,
                 std::vector<double>& posX,
                 std::vector<double>& posY,
                 std::vector<double>& posZ,
                 std::vector<double>& localVelX,
                 std::vector<double>& localVelY,
                 std::vector<double>& localVelZ) {
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(bodies.size()); ++i) {
            posX[static_cast<std::size_t>(i)] = bodies[static_cast<std::size_t>(i)].pos.x;
            posY[static_cast<std::size_t>(i)] = bodies[static_cast<std::size_t>(i)].pos.y;
            posZ[static_cast<std::size_t>(i)] = bodies[static_cast<std::size_t>(i)].pos.z;
        }

        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(localVelX.size()); ++i) {
            const Body& body = bodies[static_cast<std::size_t>(firstBody) + static_cast<std::size_t>(i)];
            localVelX[static_cast<std::size_t>(i)] = body.vel.x;
            localVelY[static_cast<std::size_t>(i)] = body.vel.y;
            localVelZ[static_cast<std::size_t>(i)] = body.vel.z;
        }
    }
}

void finishStep(std::vector<Body>& localBodies,
                const std::vector<double>& localPosX,
                const std::vector<double>& localPosY,
                const std::vector<double>& localPosZ,
                const std::vector<double>& localVelX,
                const std::vector<double>& localVelY,
                const std::vector<double>& localVelZ) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(localBodies.size()); ++i) {
        Body& body = localBodies[static_cast<std::size_t>(i)];
        const std::size_t index = static_cast<std::size_t>(i);
        body.pos.x = localPosX[index];
        body.pos.y = localPosY[index];
        body.pos.z = localPosZ[index];
        body.vel.x = localVelX[index];
        body.vel.y = localVelY[index];
        body.vel.z = localVelZ[index];
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const std::size_t n = bodies.size();
    std::vector<double> kinetic(n, 0.0);
    std::vector<double> potential(n, 0.0);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        const Body& body = bodies[static_cast<std::size_t>(i)];
        kinetic[static_cast<std::size_t>(i)] = 0.5 * (body.vel.x * body.vel.x +
                                                       body.vel.y * body.vel.y +
                                                       body.vel.z * body.vel.z);
    }

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        double rowEnergy = 0.0;
        const Body& bodyI = bodies[static_cast<std::size_t>(i)];
        for (std::size_t j = static_cast<std::size_t>(i) + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodyI.pos.x;
            const double dy = bodies[j].pos.y - bodyI.pos.y;
            const double dz = bodies[j].pos.z - bodyI.pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            rowEnergy -= 1.0 / dist;
        }
        potential[static_cast<std::size_t>(i)] = rowEnergy;
    }

    // Sum rows in the original order to retain the original reduction order
    // while allowing both O(N^2) row work and kinetic terms to run in parallel.
    double energy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        energy += kinetic[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        energy += potential[i];
    }
    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(bodies.size()); ++i) {
        const Body& body = bodies[static_cast<std::size_t>(i)];
        const bool finite = std::isfinite(body.pos.x) && std::isfinite(body.pos.y) && std::isfinite(body.pos.z) &&
                            std::isfinite(body.vel.x) && std::isfinite(body.vel.y) && std::isfinite(body.vel.z);
        const bool bounded = std::abs(body.pos.x) <= 1e6 && std::abs(body.pos.y) <= 1e6 && std::abs(body.pos.z) <= 1e6 &&
                             std::abs(body.vel.x) <= 1e6 && std::abs(body.vel.y) <= 1e6 && std::abs(body.vel.z) <= 1e6;
        valid &= finite && bounded;
    }
    return valid != 0;
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

int localRankOnNode() {
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);
    MPI_Comm_free(&localCommunicator);
    return localRank;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;

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
            parseError = true;
        }
    }

    if (showHelp || parseError || numBodies < 0 || numSteps < 0) {
        if (rank == 0) {
            if (parseError || numBodies < 0 || numSteps < 0) {
                std::printf("Invalid command line arguments\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return (parseError || numBodies < 0 || numSteps < 0) ? 1 : 0;
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Datatype mpiBody;
    MPI_Type_contiguous(6, MPI_DOUBLE, &mpiBody);
    MPI_Type_commit(&mpiBody);

    std::vector<Body> bodies(static_cast<std::size_t>(numBodies));
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    MPI_Bcast(bodies.data(), numBodies, mpiBody, 0, MPI_COMM_WORLD);

    const int baseBodies = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    const int localBodies = baseBodies + (rank < remainder ? 1 : 0);
    const int firstBody = rank * baseBodies + std::min(rank, remainder);

    std::vector<int> bodyCounts(worldSize);
    std::vector<int> bodyDisplacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        bodyCounts[r] = baseBodies + (r < remainder ? 1 : 0);
        bodyDisplacements[r] = r * baseBodies + std::min(r, remainder);
    }

    const int localRank = localRankOnNode();
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA device is available for the MPI rank set\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    std::vector<Body> localState(static_cast<std::size_t>(localBodies));
    std::vector<double> posX(static_cast<std::size_t>(numBodies));
    std::vector<double> posY(static_cast<std::size_t>(numBodies));
    std::vector<double> posZ(static_cast<std::size_t>(numBodies));
    std::vector<double> localPosX(static_cast<std::size_t>(localBodies));
    std::vector<double> localPosY(static_cast<std::size_t>(localBodies));
    std::vector<double> localPosZ(static_cast<std::size_t>(localBodies));
    std::vector<double> localVelX(static_cast<std::size_t>(localBodies));
    std::vector<double> localVelY(static_cast<std::size_t>(localBodies));
    std::vector<double> localVelZ(static_cast<std::size_t>(localBodies));

    CudaNBody cudaNBody(numBodies, localBodies);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        prepareStep(bodies, firstBody, posX, posY, posZ, localVelX, localVelY, localVelZ);
        cudaNBody.step(posX.data(), posY.data(), posZ.data(),
                       localPosX.data(), localPosY.data(), localPosZ.data(),
                       localVelX.data(), localVelY.data(), localVelZ.data(), firstBody);
        finishStep(localState, localPosX, localPosY, localPosZ, localVelX, localVelY, localVelZ);

        // Every rank needs all positions for the next all-pairs force step.
        MPI_Allgatherv(localState.data(), localBodies, mpiBody,
                       bodies.data(), bodyCounts.data(), bodyDisplacements.data(), mpiBody,
                       MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    if (rank == 0) {
        const long long elapsedMilliseconds = static_cast<long long>((end - start) * 1000.0);
        std::printf("Simulation time: %lld ms\n", elapsedMilliseconds);
    }

    if (printResults && rank == 0) {
        std::vector<double> bodyData(static_cast<std::size_t>(numBodies) * 6);
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(bodies.size()); ++i) {
            const Body& body = bodies[static_cast<std::size_t>(i)];
            const std::size_t offset = static_cast<std::size_t>(i) * 6;
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
        const int localValid = validateSimulation(bodies) ? 1 : 0;
        int globallyValid = 0;
        MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (rank == 0) {
            std::printf("Validating simulation results...\n");
            if (globallyValid != 0) {
                std::printf("Final energy: %.6f\n", computeTotalEnergy(bodies));
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation failed: found an invalid body state\n");
                std::printf("Validation: FAILED\n");
            }
        }

        MPI_Type_free(&mpiBody);
        MPI_Finalize();
        return globallyValid != 0 ? 0 : 1;
    }

    MPI_Type_free(&mpiBody);
    MPI_Finalize();
    return 0;
}
