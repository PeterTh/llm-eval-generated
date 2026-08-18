#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

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

namespace {

int g_rank = 0;

[[noreturn]] void abortWithMessage(const char* kind, const char* expression, const char* detail,
                                   const char* file, int line) {
    std::fprintf(stderr, "[rank %d] %s failure at %s:%d while executing %s: %s\n",
                 g_rank, kind, file, line, expression, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        abortWithMessage("CUDA", expression, cudaGetErrorString(status), file, line);
    }
}

void checkMpi(int status, const char* expression, const char* file, int line) {
    if (status != MPI_SUCCESS) {
        char error[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, error, &length);
        error[length] = '\0';
        abortWithMessage("MPI", expression, error, file, line);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)

template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t count) : count_(count) {
        if (count_ != 0) {
            void* allocation = nullptr;
            CUDA_CHECK(cudaMallocHost(&allocation, count_ * sizeof(T)));
            data_ = static_cast<T*>(allocation);
        }
    }

    ~PinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count_ != 0) {
            void* allocation = nullptr;
            CUDA_CHECK(cudaMalloc(&allocation, count_ * sizeof(T)));
            data_ = static_cast<T*>(allocation);
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

struct Partition {
    std::vector<int> counts;
    std::vector<int> offsets;
    std::vector<int> positionCounts;
    std::vector<int> positionOffsets;
    int localCount = 0;
    int localOffset = 0;
};

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

Partition makePartition(int numBodies, int rank, int worldSize) {
    Partition partition;
    partition.counts.resize(worldSize);
    partition.offsets.resize(worldSize);
    partition.positionCounts.resize(worldSize);
    partition.positionOffsets.resize(worldSize);

    const int baseCount = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    int offset = 0;
    for (int process = 0; process < worldSize; ++process) {
        const int count = baseCount + (process < remainder ? 1 : 0);
        partition.counts[process] = count;
        partition.offsets[process] = offset;
        partition.positionCounts[process] = 3 * count;
        partition.positionOffsets[process] = 3 * offset;
        offset += count;
    }

    partition.localCount = partition.counts[rank];
    partition.localOffset = partition.offsets[rank];
    return partition;
}

__global__ void computeForcesKernel(const double* __restrict__ x, const double* __restrict__ y,
                                    const double* __restrict__ z, double* __restrict__ vx,
                                    double* __restrict__ vy, double* __restrict__ vz,
                                    int numBodies, int localOffset, int localCount) {
    extern __shared__ double sharedPositions[];
    double* const sharedX = sharedPositions;
    double* const sharedY = sharedX + blockDim.x;
    double* const sharedZ = sharedY + blockDim.x;

    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIndex < localCount;

    double xi = 0.0;
    double yi = 0.0;
    double zi = 0.0;
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    if (active) {
        const int globalIndex = localOffset + localIndex;
        xi = x[globalIndex];
        yi = y[globalIndex];
        zi = z[globalIndex];
    }

    for (int tileStart = 0; tileStart < numBodies; tileStart += blockDim.x) {
        const int sourceIndex = tileStart + threadIdx.x;
        if (sourceIndex < numBodies) {
            sharedX[threadIdx.x] = x[sourceIndex];
            sharedY[threadIdx.x] = y[sourceIndex];
            sharedZ[threadIdx.x] = z[sourceIndex];
        }
        __syncthreads();

        const int tileCount = min(static_cast<int>(blockDim.x), numBodies - tileStart);
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < tileCount; ++j) {
                const double dx = sharedX[j] - xi;
                const double dy = sharedY[j] - yi;
                const double dz = sharedZ[j] - zi;
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
        vx[localIndex] += DT * fx;
        vy[localIndex] += DT * fy;
        vz[localIndex] += DT * fz;
    }
}

__global__ void integrateBodiesKernel(double* x, double* y, double* z,
                                      const double* __restrict__ vx, const double* __restrict__ vy,
                                      const double* __restrict__ vz, int localOffset, int localCount) {
    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIndex < localCount) {
        const int globalIndex = localOffset + localIndex;
        x[globalIndex] += vx[localIndex] * DT;
        y[globalIndex] += vy[localIndex] * DT;
        z[globalIndex] += vz[localIndex] * DT;
    }
}

void launchTimestep(DeviceBuffer<double>& dX, DeviceBuffer<double>& dY, DeviceBuffer<double>& dZ,
                    DeviceBuffer<double>& dVx, DeviceBuffer<double>& dVy, DeviceBuffer<double>& dVz,
                    const Partition& partition, int numBodies, cudaStream_t stream) {
    if (partition.localCount == 0) {
        return;
    }

    const int blocks = (partition.localCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    const size_t sharedBytes = 3ULL * CUDA_BLOCK_SIZE * sizeof(double);
    computeForcesKernel<<<blocks, CUDA_BLOCK_SIZE, sharedBytes, stream>>>(
        dX.data(), dY.data(), dZ.data(), dVx.data(), dVy.data(), dVz.data(),
        numBodies, partition.localOffset, partition.localCount);
    CUDA_CHECK(cudaGetLastError());

    integrateBodiesKernel<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
        dX.data(), dY.data(), dZ.data(), dVx.data(), dVy.data(), dVz.data(),
        partition.localOffset, partition.localCount);
    CUDA_CHECK(cudaGetLastError());
}

void copyInitialStateToDevice(DeviceBuffer<double>& dX, DeviceBuffer<double>& dY, DeviceBuffer<double>& dZ,
                              DeviceBuffer<double>& dVx, DeviceBuffer<double>& dVy, DeviceBuffer<double>& dVz,
                              const PinnedBuffer<double>& hX, const PinnedBuffer<double>& hY,
                              const PinnedBuffer<double>& hZ, const PinnedBuffer<double>& hVx,
                              const PinnedBuffer<double>& hVy, const PinnedBuffer<double>& hVz,
                              int numBodies, int localCount, cudaStream_t stream) {
    if (numBodies != 0) {
        const size_t positionBytes = static_cast<size_t>(numBodies) * sizeof(double);
        CUDA_CHECK(cudaMemcpyAsync(dX.data(), hX.data(), positionBytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dY.data(), hY.data(), positionBytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dZ.data(), hZ.data(), positionBytes, cudaMemcpyHostToDevice, stream));
    }
    if (localCount != 0) {
        const size_t velocityBytes = static_cast<size_t>(localCount) * sizeof(double);
        CUDA_CHECK(cudaMemcpyAsync(dVx.data(), hVx.data(), velocityBytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dVy.data(), hVy.data(), velocityBytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dVz.data(), hVz.data(), velocityBytes, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void exchangePositions(DeviceBuffer<double>& dX, DeviceBuffer<double>& dY, DeviceBuffer<double>& dZ,
                       PinnedBuffer<double>& hX, PinnedBuffer<double>& hY, PinnedBuffer<double>& hZ,
                       PinnedBuffer<double>& hPackedPositions, const Partition& partition,
                       int numBodies, cudaStream_t stream) {
    if (partition.localCount != 0) {
        const size_t localBytes = static_cast<size_t>(partition.localCount) * sizeof(double);
        CUDA_CHECK(cudaMemcpyAsync(hX.data() + partition.localOffset, dX.data() + partition.localOffset,
                                   localBytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hY.data() + partition.localOffset, dY.data() + partition.localOffset,
                                   localBytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hZ.data() + partition.localOffset, dZ.data() + partition.localOffset,
                                   localBytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    #pragma omp parallel for schedule(static)
    for (int localIndex = 0; localIndex < partition.localCount; ++localIndex) {
        const int globalIndex = partition.localOffset + localIndex;
        hPackedPositions.data()[3 * globalIndex] = hX.data()[globalIndex];
        hPackedPositions.data()[3 * globalIndex + 1] = hY.data()[globalIndex];
        hPackedPositions.data()[3 * globalIndex + 2] = hZ.data()[globalIndex];
    }

    MPI_Request request = MPI_REQUEST_NULL;
    MPI_CHECK(MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, hPackedPositions.data(),
                              partition.positionCounts.data(), partition.positionOffsets.data(),
                              MPI_DOUBLE, MPI_COMM_WORLD, &request));
    MPI_CHECK(MPI_Wait(&request, MPI_STATUS_IGNORE));

    #pragma omp parallel for schedule(static)
    for (int globalIndex = 0; globalIndex < numBodies; ++globalIndex) {
        hX.data()[globalIndex] = hPackedPositions.data()[3 * globalIndex];
        hY.data()[globalIndex] = hPackedPositions.data()[3 * globalIndex + 1];
        hZ.data()[globalIndex] = hPackedPositions.data()[3 * globalIndex + 2];
    }

    const int beforeLocal = partition.localOffset;
    const int afterLocal = numBodies - partition.localOffset - partition.localCount;
    if (beforeLocal != 0) {
        const size_t bytes = static_cast<size_t>(beforeLocal) * sizeof(double);
        CUDA_CHECK(cudaMemcpyAsync(dX.data(), hX.data(), bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dY.data(), hY.data(), bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dZ.data(), hZ.data(), bytes, cudaMemcpyHostToDevice, stream));
    }
    if (afterLocal != 0) {
        const size_t bytes = static_cast<size_t>(afterLocal) * sizeof(double);
        const int offset = partition.localOffset + partition.localCount;
        CUDA_CHECK(cudaMemcpyAsync(dX.data() + offset, hX.data() + offset, bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dY.data() + offset, hY.data() + offset, bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dZ.data() + offset, hZ.data() + offset, bytes, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void copyLocalVelocitiesToHost(const DeviceBuffer<double>& dVx, const DeviceBuffer<double>& dVy,
                               const DeviceBuffer<double>& dVz, PinnedBuffer<double>& hVx,
                               PinnedBuffer<double>& hVy, PinnedBuffer<double>& hVz,
                               int localCount, cudaStream_t stream) {
    if (localCount == 0) {
        return;
    }
    const size_t bytes = static_cast<size_t>(localCount) * sizeof(double);
    CUDA_CHECK(cudaMemcpyAsync(hVx.data(), dVx.data(), bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hVy.data(), dVy.data(), bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hVz.data(), dVz.data(), bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const int numBodies = static_cast<int>(bodies.size());
    double kineticEnergy = 0.0;
    #pragma omp parallel for reduction(+ : kineticEnergy) schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        const Body& body = bodies[static_cast<size_t>(i)];
        kineticEnergy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

    double potentialEnergy = 0.0;
    #pragma omp parallel for reduction(+ : potentialEnergy) schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        double rowEnergy = 0.0;
        for (int j = i + 1; j < numBodies; ++j) {
            const double dx = bodies[static_cast<size_t>(j)].pos.x - bodies[static_cast<size_t>(i)].pos.x;
            const double dy = bodies[static_cast<size_t>(j)].pos.y - bodies[static_cast<size_t>(i)].pos.y;
            const double dz = bodies[static_cast<size_t>(j)].pos.z - bodies[static_cast<size_t>(i)].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            rowEnergy -= 1.0 / dist;
        }
        potentialEnergy += rowEnergy;
    }
    return kineticEnergy + potentialEnergy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    const int numBodies = static_cast<int>(bodies.size());
    int invalid = 0;
    constexpr double maxPosition = 1e6;
    constexpr double maxVelocity = 1e6;

    #pragma omp parallel for reduction(| : invalid) schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        const Body& body = bodies[static_cast<size_t>(i)];
        const bool nonFinite = !std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
                               !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z);
        const bool outOfBounds = std::abs(body.pos.x) > maxPosition || std::abs(body.pos.y) > maxPosition ||
                                 std::abs(body.pos.z) > maxPosition || std::abs(body.vel.x) > maxVelocity ||
                                 std::abs(body.vel.y) > maxVelocity || std::abs(body.vel.z) > maxVelocity;
        invalid |= static_cast<int>(nonFinite || outOfBounds);
    }

    if (invalid != 0) {
        std::printf("Validation failed: found a non-finite or out-of-bounds body state\n");
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

bool parsePositiveOrZeroInt(const char* text, int* value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    const int initStatus = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    if (initStatus != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI initialization failed\n");
        return EXIT_FAILURE;
    }
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank));
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (g_rank == 0) {
            std::fprintf(stderr, "MPI does not provide the MPI_THREAD_FUNNELED support required by this program\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    bool parseFailed = false;
    bool showHelp = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            parseFailed |= !parsePositiveOrZeroInt(argv[++i], &numBodies);
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            parseFailed |= !parsePositiveOrZeroInt(argv[++i], &numSteps);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseFailed = true;
        }
    }

    if (showHelp || parseFailed || numBodies > INT_MAX / 3) {
        if (g_rank == 0) {
            if (numBodies > INT_MAX / 3) {
                std::printf("Number of bodies is too large for MPI count arguments\n");
            } else if (parseFailed) {
                std::printf("Invalid command line option or value\n");
            }
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return parseFailed || numBodies > INT_MAX / 3 ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &localComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortWithMessage("CUDA", "cudaGetDeviceCount", "no CUDA devices are visible", __FILE__, __LINE__);
    }
    const int deviceId = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(deviceId));
    CUDA_CHECK(cudaFree(nullptr));

    const Partition partition = makePartition(numBodies, g_rank, worldSize);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const size_t totalBodies = static_cast<size_t>(numBodies);
    PinnedBuffer<double> hX(totalBodies);
    PinnedBuffer<double> hY(totalBodies);
    PinnedBuffer<double> hZ(totalBodies);
    PinnedBuffer<double> hPackedPositions(3 * totalBodies);
    PinnedBuffer<double> hVx(static_cast<size_t>(partition.localCount));
    PinnedBuffer<double> hVy(static_cast<size_t>(partition.localCount));
    PinnedBuffer<double> hVz(static_cast<size_t>(partition.localCount));

    std::vector<double> rootVx;
    std::vector<double> rootVy;
    std::vector<double> rootVz;
    if (g_rank == 0) {
        std::vector<Body> bodies(totalBodies);
        randomizeBodies(bodies);
        rootVx.resize(totalBodies);
        rootVy.resize(totalBodies);
        rootVz.resize(totalBodies);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const Body& body = bodies[static_cast<size_t>(i)];
            hPackedPositions.data()[3 * i] = body.pos.x;
            hPackedPositions.data()[3 * i + 1] = body.pos.y;
            hPackedPositions.data()[3 * i + 2] = body.pos.z;
            rootVx[static_cast<size_t>(i)] = body.vel.x;
            rootVy[static_cast<size_t>(i)] = body.vel.y;
            rootVz[static_cast<size_t>(i)] = body.vel.z;
        }
    }

    MPI_CHECK(MPI_Bcast(hPackedPositions.data(), 3 * numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        hX.data()[i] = hPackedPositions.data()[3 * i];
        hY.data()[i] = hPackedPositions.data()[3 * i + 1];
        hZ.data()[i] = hPackedPositions.data()[3 * i + 2];
    }
    MPI_CHECK(MPI_Scatterv(g_rank == 0 ? rootVx.data() : nullptr, partition.counts.data(), partition.offsets.data(),
                           MPI_DOUBLE, hVx.data(), partition.localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatterv(g_rank == 0 ? rootVy.data() : nullptr, partition.counts.data(), partition.offsets.data(),
                           MPI_DOUBLE, hVy.data(), partition.localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatterv(g_rank == 0 ? rootVz.data() : nullptr, partition.counts.data(), partition.offsets.data(),
                           MPI_DOUBLE, hVz.data(), partition.localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    DeviceBuffer<double> dX(totalBodies);
    DeviceBuffer<double> dY(totalBodies);
    DeviceBuffer<double> dZ(totalBodies);
    DeviceBuffer<double> dVx(static_cast<size_t>(partition.localCount));
    DeviceBuffer<double> dVy(static_cast<size_t>(partition.localCount));
    DeviceBuffer<double> dVz(static_cast<size_t>(partition.localCount));
    copyInitialStateToDevice(dX, dY, dZ, dVx, dVy, dVz, hX, hY, hZ, hVx, hVy, hVz,
                             numBodies, partition.localCount, stream);

    if (g_rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA device assignment: local-rank modulo visible GPUs\n",
                    worldSize, omp_get_max_threads());
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        launchTimestep(dX, dY, dZ, dVx, dVy, dVz, partition, numBodies, stream);
        exchangePositions(dX, dY, dZ, hX, hY, hZ, hPackedPositions, partition, numBodies, stream);
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto end = std::chrono::steady_clock::now();

    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double maximumSeconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &maximumSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (g_rank == 0) {
        const long long milliseconds = static_cast<long long>(maximumSeconds * 1000.0);
        std::printf("Simulation time: %lld ms\n", milliseconds);
    }

    const bool needFinalBodies = printResults || validate;
    std::vector<double> globalVx;
    std::vector<double> globalVy;
    std::vector<double> globalVz;
    if (needFinalBodies) {
        copyLocalVelocitiesToHost(dVx, dVy, dVz, hVx, hVy, hVz, partition.localCount, stream);
        if (g_rank == 0) {
            globalVx.resize(totalBodies);
            globalVy.resize(totalBodies);
            globalVz.resize(totalBodies);
        }
        MPI_CHECK(MPI_Gatherv(hVx.data(), partition.localCount, MPI_DOUBLE,
                              g_rank == 0 ? globalVx.data() : nullptr, partition.counts.data(), partition.offsets.data(),
                              MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Gatherv(hVy.data(), partition.localCount, MPI_DOUBLE,
                              g_rank == 0 ? globalVy.data() : nullptr, partition.counts.data(), partition.offsets.data(),
                              MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Gatherv(hVz.data(), partition.localCount, MPI_DOUBLE,
                              g_rank == 0 ? globalVz.data() : nullptr, partition.counts.data(), partition.offsets.data(),
                              MPI_DOUBLE, 0, MPI_COMM_WORLD));
    }

    int exitCode = EXIT_SUCCESS;
    if (g_rank == 0 && needFinalBodies) {
        std::vector<Body> finalBodies(totalBodies);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            Body& body = finalBodies[static_cast<size_t>(i)];
            body.pos = Vec3{hX.data()[i], hY.data()[i], hZ.data()[i]};
            body.vel = Vec3{globalVx[static_cast<size_t>(i)], globalVy[static_cast<size_t>(i)], globalVz[static_cast<size_t>(i)]};
        }

        if (printResults) {
            std::vector<double> bodyData(6 * totalBodies);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                const Body& body = finalBodies[static_cast<size_t>(i)];
                bodyData[6 * i] = body.pos.x;
                bodyData[6 * i + 1] = body.pos.y;
                bodyData[6 * i + 2] = body.pos.z;
                bodyData[6 * i + 3] = body.vel.x;
                bodyData[6 * i + 4] = body.vel.y;
                bodyData[6 * i + 5] = body.vel.z;
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            std::printf("Validating simulation results...\n");
            if (validateSimulation(finalBodies)) {
                std::printf("Final energy: %.6f\n", computeTotalEnergy(finalBodies));
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = EXIT_FAILURE;
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_CHECK(MPI_Comm_free(&localComm));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
