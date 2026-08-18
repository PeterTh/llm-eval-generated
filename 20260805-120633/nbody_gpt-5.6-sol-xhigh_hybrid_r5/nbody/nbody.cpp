#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#endif

#include <algorithm>
#include <cerrno>
#include <climits>
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
constexpr int SMALL_FORCE_BLOCK_SIZE = 32;
constexpr int LARGE_FORCE_BLOCK_SIZE = 64;
constexpr int ENERGY_BLOCK_SIZE = 256;

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

static_assert(std::is_standard_layout_v<Vec3> && sizeof(Vec3) == 3 * sizeof(double));
static_assert(std::is_standard_layout_v<Body> && sizeof(Body) == 6 * sizeof(double));

[[noreturn]] void abortCuda(const cudaError_t error, const char* expression, const int rank) {
    std::fprintf(stderr, "MPI rank %d: CUDA call %s failed: %s\n", rank, expression,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

[[noreturn]] void abortMpi(const int error, const char* expression, const int rank) {
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "MPI rank %d: MPI call %s failed: %.*s\n", rank, expression, length,
                 message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        const cudaError_t cuda_check_error = (call);                                             \
        if (cuda_check_error != cudaSuccess) {                                                   \
            abortCuda(cuda_check_error, #call, rank);                                            \
        }                                                                                       \
    } while (false)

#define MPI_CHECK(call)                                                                         \
    do {                                                                                        \
        const int mpi_check_error = (call);                                                      \
        if (mpi_check_error != MPI_SUCCESS) {                                                    \
            abortMpi(mpi_check_error, #call, rank);                                              \
        }                                                                                       \
    } while (false)

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Keep the original rand_r call order so rank zero creates exactly the same initial state.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

template <int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void advanceBodies(const Vec3* __restrict__ globalPositions, Vec3* __restrict__ nextLocalPositions,
                   Vec3* __restrict__ localVelocities, const int numBodies,
                   const int localBodies, const int localOffset) {
    if (localBodies == 0) {
        return;
    }

    __shared__ double tileX[BLOCK_SIZE];
    __shared__ double tileY[BLOCK_SIZE];
    __shared__ double tileZ[BLOCK_SIZE];

    const int localIndex = static_cast<int>(blockIdx.x) * BLOCK_SIZE +
                           static_cast<int>(threadIdx.x);
    const bool active = localIndex < localBodies;

    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    if (active) {
        const Vec3 position = globalPositions[localOffset + localIndex];
        px = position.x;
        py = position.y;
        pz = position.z;
    }

    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    for (int tileBase = 0; tileBase < numBodies; tileBase += BLOCK_SIZE) {
        const int sourceIndex = tileBase + static_cast<int>(threadIdx.x);
        if (sourceIndex < numBodies) {
            const Vec3 source = globalPositions[sourceIndex];
            tileX[threadIdx.x] = source.x;
            tileY[threadIdx.x] = source.y;
            tileZ[threadIdx.x] = source.z;
        }
        __syncthreads();

        if (active) {
            const int tileSize = min(BLOCK_SIZE, numBodies - tileBase);
#pragma unroll 8
            for (int j = 0; j < tileSize; ++j) {
                const double dx = tileX[j] - px;
                const double dy = tileY[j] - py;
                const double dz = tileZ[j] - pz;
                const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inverseDistance = 1.0 / sqrt(distanceSquared);
                const double inverseDistanceCubed =
                    inverseDistance * inverseDistance * inverseDistance;
                forceX += dx * inverseDistanceCubed;
                forceY += dy * inverseDistanceCubed;
                forceZ += dz * inverseDistanceCubed;
            }
        }
        __syncthreads();
    }

    if (active) {
        Vec3 velocity = localVelocities[localIndex];
        velocity.x += DT * forceX;
        velocity.y += DT * forceY;
        velocity.z += DT * forceZ;
        localVelocities[localIndex] = velocity;
        nextLocalPositions[localIndex] =
            Vec3(px + velocity.x * DT, py + velocity.y * DT, pz + velocity.z * DT);
    }
}

__global__ __launch_bounds__(ENERGY_BLOCK_SIZE)
void computeEnergyPartials(const Vec3* __restrict__ globalPositions,
                           const Vec3* __restrict__ localVelocities,
                           double* __restrict__ blockEnergy, const int numBodies,
                           const int localBodies, const int rank, const int worldSize) {
    __shared__ double scratch[ENERGY_BLOCK_SIZE];

    double energy = 0.0;
    const int gridStride = static_cast<int>(gridDim.x) * ENERGY_BLOCK_SIZE;
    for (int localIndex = static_cast<int>(blockIdx.x) * ENERGY_BLOCK_SIZE +
                          static_cast<int>(threadIdx.x);
         localIndex < localBodies; localIndex += gridStride) {
        const Vec3 velocity = localVelocities[localIndex];
        energy += 0.5 * (velocity.x * velocity.x + velocity.y * velocity.y +
                         velocity.z * velocity.z);
    }

    // Cyclic ownership of potential-energy rows balances the triangular pair space across ranks.
    const int potentialRows =
        rank < numBodies ? 1 + (numBodies - 1 - rank) / worldSize : 0;
    for (int row = static_cast<int>(blockIdx.x) * ENERGY_BLOCK_SIZE +
                   static_cast<int>(threadIdx.x);
         row < potentialRows; row += gridStride) {
        const int globalIndex = rank + row * worldSize;
        const Vec3 position = globalPositions[globalIndex];
        for (int j = globalIndex + 1; j < numBodies; ++j) {
            const double dx = globalPositions[j].x - position.x;
            const double dy = globalPositions[j].y - position.y;
            const double dz = globalPositions[j].z - position.z;
            energy -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }

    scratch[threadIdx.x] = energy;
    __syncthreads();

    for (int stride = ENERGY_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        blockEnergy[blockIdx.x] = scratch[0];
    }
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseInteger(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool cudaAwareMpiAvailable() {
    const char* overrideValue = std::getenv("NBODY_CUDA_AWARE_MPI");
    if (overrideValue != nullptr) {
        return std::strcmp(overrideValue, "1") == 0 ||
               std::strcmp(overrideValue, "true") == 0 ||
               std::strcmp(overrideValue, "TRUE") == 0;
    }
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    const int initError = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    if (initError != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide the required MPI_THREAD_FUNNELED support\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool validArguments = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            validArguments = parseInteger(argv[++i], numBodies) && validArguments;
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            validArguments = parseInteger(argv[++i], numSteps) && validArguments;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                std::printf("Unknown or incomplete option: %s\n", argv[i]);
            }
            validArguments = false;
        }
    }

    // Counts are passed as MPI_DOUBLE, so keep all collective counts representable by int.
    if (numBodies <= 0 || numBodies > INT_MAX / 6 || numSteps < 0) {
        validArguments = false;
    }
    if (showHelp || !validArguments) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return showHelp && validArguments ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    omp_set_dynamic(0);

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &localCommunicator));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (localRank == 0) {
            std::fprintf(stderr, "No CUDA device is visible to the node-local MPI ranks\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    // This handles both common launch modes: all node GPUs visible to every rank, or one
    // distinct CUDA_VISIBLE_DEVICES entry presented as device zero to each rank.
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));

    int cudaAware = cudaAwareMpiAvailable() ? 1 : 0;
    int cudaAwareEverywhere = 0;
    MPI_CHECK(MPI_Allreduce(&cudaAware, &cudaAwareEverywhere, 1, MPI_INT, MPI_MIN,
                            MPI_COMM_WORLD));
    cudaAware = cudaAwareEverywhere;

    std::vector<int> bodyCounts(worldSize);
    std::vector<int> bodyOffsets(worldSize);
    std::vector<int> counts3(worldSize);
    std::vector<int> offsets3(worldSize);
    std::vector<int> counts6(worldSize);
    std::vector<int> offsets6(worldSize);
    for (int process = 0; process < worldSize; ++process) {
        const int begin = static_cast<int>((static_cast<long long>(numBodies) * process) /
                                           worldSize);
        const int end = static_cast<int>((static_cast<long long>(numBodies) * (process + 1)) /
                                         worldSize);
        bodyOffsets[process] = begin;
        bodyCounts[process] = end - begin;
        counts3[process] = 3 * bodyCounts[process];
        offsets3[process] = 3 * bodyOffsets[process];
        counts6[process] = 6 * bodyCounts[process];
        offsets6[process] = 6 * bodyOffsets[process];
    }
    const int localOffset = bodyOffsets[rank];
    const int localBodies = bodyCounts[rank];
    const bool needFinalState = printResults || validate;

    Vec3* hostGlobalPositions = nullptr;
    CUDA_CHECK(cudaMallocHost(&hostGlobalPositions,
                              static_cast<size_t>(numBodies) * sizeof(Vec3)));

    std::vector<Body> initialBodies;
    if (rank == 0) {
        initialBodies.resize(numBodies);
        randomizeBodies(initialBodies);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            hostGlobalPositions[i] = initialBodies[i].pos;
        }
    }

    std::vector<Body> localInitialBodies(localBodies);
    MPI_CHECK(MPI_Scatterv(rank == 0 ? static_cast<const void*>(initialBodies.data()) : nullptr,
                           counts6.data(), offsets6.data(), MPI_DOUBLE,
                           static_cast<void*>(localInitialBodies.data()), 6 * localBodies,
                           MPI_DOUBLE, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(hostGlobalPositions, 3 * numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    std::vector<Vec3> hostLocalVelocities(std::max(localBodies, 1));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < localBodies; ++i) {
        hostLocalVelocities[i] = localInitialBodies[i].vel;
    }
    initialBodies.clear();
    initialBodies.shrink_to_fit();
    localInitialBodies.clear();
    localInitialBodies.shrink_to_fit();

    Vec3* deviceGlobalPositions = nullptr;
    Vec3* deviceNextLocalPositions = nullptr;
    Vec3* deviceLocalVelocities = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceGlobalPositions,
                          static_cast<size_t>(numBodies) * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&deviceNextLocalPositions,
                          static_cast<size_t>(std::max(localBodies, 1)) * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&deviceLocalVelocities,
                          static_cast<size_t>(std::max(localBodies, 1)) * sizeof(Vec3)));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(deviceGlobalPositions, hostGlobalPositions,
                               static_cast<size_t>(numBodies) * sizeof(Vec3),
                               cudaMemcpyHostToDevice, stream));
    if (localBodies > 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceLocalVelocities, hostLocalVelocities.data(),
                                   static_cast<size_t>(localBodies) * sizeof(Vec3),
                                   cudaMemcpyHostToDevice, stream));
    }

    // Small target partitions need more blocks to occupy the whole GPU; larger partitions use
    // wider tiles to reduce synchronization and source-load overhead.
    const bool useSmallForceBlocks =
        static_cast<long long>(localBodies) <
        8LL * deviceProperties.multiProcessorCount * LARGE_FORCE_BLOCK_SIZE;
    const int forceBlockSize =
        useSmallForceBlocks ? SMALL_FORCE_BLOCK_SIZE : LARGE_FORCE_BLOCK_SIZE;
    const int gridSize = (localBodies + forceBlockSize - 1) / forceBlockSize;
    const auto launchAdvance = [&](const int bodies, const int blocks) {
        if (useSmallForceBlocks) {
            advanceBodies<SMALL_FORCE_BLOCK_SIZE><<<blocks, SMALL_FORCE_BLOCK_SIZE, 0, stream>>>(
                deviceGlobalPositions, deviceNextLocalPositions, deviceLocalVelocities, numBodies,
                bodies, localOffset);
        } else {
            advanceBodies<LARGE_FORCE_BLOCK_SIZE><<<blocks, LARGE_FORCE_BLOCK_SIZE, 0, stream>>>(
                deviceGlobalPositions, deviceNextLocalPositions, deviceLocalVelocities, numBodies,
                bodies, localOffset);
        }
        CUDA_CHECK(cudaGetLastError());
    };

    // Force CUDA module loading/JIT before the measured region without changing any body.
    launchAdvance(0, 1);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cudaAware && !needFinalState) {
        CUDA_CHECK(cudaFreeHost(hostGlobalPositions));
        hostGlobalPositions = nullptr;
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s)/rank, "
                    "one CUDA GPU/rank\n",
                    worldSize, omp_get_max_threads());
        std::printf("Position exchange: %s\n",
                    cudaAware ? "CUDA-aware MPI (device direct)" : "MPI with pinned-host staging");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        if (localBodies > 0) {
            launchAdvance(localBodies, gridSize);
        }

        if (step + 1 < numSteps) {
            if (cudaAware) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                MPI_CHECK(MPI_Allgatherv(deviceNextLocalPositions, 3 * localBodies, MPI_DOUBLE,
                                         deviceGlobalPositions, counts3.data(), offsets3.data(),
                                         MPI_DOUBLE, MPI_COMM_WORLD));
            } else {
                if (localBodies > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(hostGlobalPositions + localOffset,
                                               deviceNextLocalPositions,
                                               static_cast<size_t>(localBodies) * sizeof(Vec3),
                                               cudaMemcpyDeviceToHost, stream));
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));
                MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                         hostGlobalPositions, counts3.data(), offsets3.data(),
                                         MPI_DOUBLE, MPI_COMM_WORLD));
                CUDA_CHECK(cudaMemcpyAsync(deviceGlobalPositions, hostGlobalPositions,
                                           static_cast<size_t>(numBodies) * sizeof(Vec3),
                                           cudaMemcpyHostToDevice, stream));
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const double elapsed = MPI_Wtime() - start;
    double maximumElapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));
    if (rank == 0) {
        const auto elapsedMilliseconds = static_cast<long long>(maximumElapsed * 1000.0);
        std::printf("Simulation time: %lld ms\n", elapsedMilliseconds);
    }

    if (needFinalState) {
        if (numSteps > 0) {
            if (localBodies > 0) {
                CUDA_CHECK(cudaMemcpyAsync(hostGlobalPositions + localOffset,
                                           deviceNextLocalPositions,
                                           static_cast<size_t>(localBodies) * sizeof(Vec3),
                                           cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                     hostGlobalPositions, counts3.data(), offsets3.data(),
                                     MPI_DOUBLE, MPI_COMM_WORLD));
        }
        if (localBodies > 0) {
            CUDA_CHECK(cudaMemcpyAsync(hostLocalVelocities.data(), deviceLocalVelocities,
                                       static_cast<size_t>(localBodies) * sizeof(Vec3),
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (printResults) {
        std::vector<Vec3> globalVelocities;
        if (rank == 0) {
            globalVelocities.resize(numBodies);
        }
        MPI_CHECK(MPI_Gatherv(hostLocalVelocities.data(), 3 * localBodies, MPI_DOUBLE,
                              rank == 0 ? static_cast<void*>(globalVelocities.data()) : nullptr,
                              counts3.data(), offsets3.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));

        if (rank == 0) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                const size_t output = static_cast<size_t>(i) * 6;
                bodyData[output] = hostGlobalPositions[i].x;
                bodyData[output + 1] = hostGlobalPositions[i].y;
                bodyData[output + 2] = hostGlobalPositions[i].z;
                bodyData[output + 3] = globalVelocities[i].x;
                bodyData[output + 4] = globalVelocities[i].y;
                bodyData[output + 5] = globalVelocities[i].z;
            }
            print_results(bodyData, "Bodies");
        }
    }

    int exitCode = EXIT_SUCCESS;
    if (validate) {
        constexpr int NONFINITE_VALUE = 1;
        constexpr int POSITION_OUT_OF_RANGE = 2;
        constexpr int VELOCITY_OUT_OF_RANGE = 4;
        int localValidationMask = 0;
#pragma omp parallel for schedule(static) reduction(| : localValidationMask)
        for (int i = 0; i < localBodies; ++i) {
            const Vec3 position = hostGlobalPositions[localOffset + i];
            const Vec3 velocity = hostLocalVelocities[i];
            if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
                !std::isfinite(position.z) || !std::isfinite(velocity.x) ||
                !std::isfinite(velocity.y) || !std::isfinite(velocity.z)) {
                localValidationMask |= NONFINITE_VALUE;
            }
            if (std::abs(position.x) > 1e6 || std::abs(position.y) > 1e6 ||
                std::abs(position.z) > 1e6) {
                localValidationMask |= POSITION_OUT_OF_RANGE;
            }
            if (std::abs(velocity.x) > 1e6 || std::abs(velocity.y) > 1e6 ||
                std::abs(velocity.z) > 1e6) {
                localValidationMask |= VELOCITY_OUT_OF_RANGE;
            }
        }

        int globalValidationMask = 0;
        MPI_CHECK(MPI_Allreduce(&localValidationMask, &globalValidationMask, 1, MPI_INT, MPI_BOR,
                                MPI_COMM_WORLD));
        if (rank == 0) {
            std::printf("Validating simulation results...\n");
        }

        if (globalValidationMask == 0) {
            CUDA_CHECK(cudaMemcpyAsync(deviceGlobalPositions, hostGlobalPositions,
                                       static_cast<size_t>(numBodies) * sizeof(Vec3),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            double localEnergy = 0.0;
            const int potentialRows =
                rank < numBodies ? 1 + (numBodies - 1 - rank) / worldSize : 0;
            const int energyItems = std::max(localBodies, potentialRows);
            if (energyItems > 0) {
                const int energyBlocks = std::max(
                    1, std::min((energyItems + ENERGY_BLOCK_SIZE - 1) / ENERGY_BLOCK_SIZE,
                                4 * deviceProperties.multiProcessorCount));
                double* deviceEnergy = nullptr;
                CUDA_CHECK(cudaMalloc(&deviceEnergy,
                                      static_cast<size_t>(energyBlocks) * sizeof(double)));
                computeEnergyPartials<<<energyBlocks, ENERGY_BLOCK_SIZE, 0, stream>>>(
                    deviceGlobalPositions, deviceLocalVelocities, deviceEnergy, numBodies,
                    localBodies, rank, worldSize);
                CUDA_CHECK(cudaGetLastError());
                std::vector<double> energyParts(energyBlocks);
                CUDA_CHECK(cudaMemcpyAsync(energyParts.data(), deviceEnergy,
                                           static_cast<size_t>(energyBlocks) * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaFree(deviceEnergy));
#pragma omp parallel for schedule(static) reduction(+ : localEnergy)
                for (int i = 0; i < energyBlocks; ++i) {
                    localEnergy += energyParts[i];
                }
            }

            double totalEnergy = 0.0;
            MPI_CHECK(MPI_Reduce(&localEnergy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0,
                                 MPI_COMM_WORLD));
            if (rank == 0) {
                std::printf("Final energy: %.6f\n", totalEnergy);
                std::printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                if (globalValidationMask & NONFINITE_VALUE) {
                    std::printf("Validation failed: found NaN or Inf value in body state\n");
                }
                if (globalValidationMask & POSITION_OUT_OF_RANGE) {
                    std::printf("Validation failed: body position exceeds reasonable bounds\n");
                }
                if (globalValidationMask & VELOCITY_OUT_OF_RANGE) {
                    std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
                }
                std::printf("Validation: FAILED\n");
            }
            exitCode = EXIT_FAILURE;
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceLocalVelocities));
    CUDA_CHECK(cudaFree(deviceNextLocalPositions));
    CUDA_CHECK(cudaFree(deviceGlobalPositions));
    if (hostGlobalPositions != nullptr) {
        CUDA_CHECK(cudaFreeHost(hostGlobalPositions));
    }
    MPI_CHECK(MPI_Comm_free(&localCommunicator));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
