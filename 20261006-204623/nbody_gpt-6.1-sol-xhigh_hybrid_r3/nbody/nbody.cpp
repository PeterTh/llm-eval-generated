#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(sizeof(Body) == 6 * sizeof(double));
static_assert(std::is_trivially_copyable<Body>::value);

int worldRank = 0;

void checkMPI(int error, const char* operation) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, message, &length);
    fprintf(stderr, "Rank %d: %s: %.*s\n", worldRank, operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

void checkCUDA(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    fprintf(stderr, "Rank %d: %s: %s\n", worldRank, operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define MPI_CHECK(operation) checkMPI((operation), #operation)
#define CUDA_CHECK(operation) checkCUDA((operation), #operation)

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), std::max(count, size_t(1)) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename T>
struct PinnedBuffer {
    T* data = nullptr;
    explicit PinnedBuffer(size_t count) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), std::max(count, size_t(1)) * sizeof(T)));
    }
    ~PinnedBuffer() { cudaFreeHost(data); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

struct Stream {
    cudaStream_t stream;
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(stream); }
};

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

constexpr int TILE_SIZE = 128;

__device__ __forceinline__ double distanceSquared(double dx, double dy, double dz) {
    // Explicit rounding/contraction keeps the ordered force sum reproducible
    // across GPU architectures and MPI decompositions, without fast math.
    double distance = __fma_rn(dx, dx, __dmul_rn(dy, dy));
    distance = __fma_rn(dz, dz, distance);
    return __dadd_rn(distance, SOFTENING);
}

__global__ void computeStep(const Vec3* __restrict__ positions,
                            Vec3* __restrict__ nextPositions,
                            Vec3* __restrict__ velocities,
                            int n, int first, int count) {
    __shared__ Vec3 sources[TILE_SIZE];
    const size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = local < static_cast<size_t>(count);
    const Vec3 position = active ? positions[first + local] : Vec3{0.0, 0.0, 0.0};
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (size_t base = 0; base < static_cast<size_t>(n); base += TILE_SIZE) {
        for (int slot = threadIdx.x; slot < TILE_SIZE; slot += blockDim.x) {
            const size_t source = base + slot;
            if (source < static_cast<size_t>(n)) sources[slot] = positions[source];
        }
        __syncthreads();
        const size_t remaining = static_cast<size_t>(n) - base;
        const int width = remaining < TILE_SIZE ? static_cast<int>(remaining) : TILE_SIZE;
        if (active) {
            // Every target traverses sources in the original increasing order.
            #pragma unroll 8
            for (int j = 0; j < width; ++j) {
                const double dx = sources[j].x - position.x;
                const double dy = sources[j].y - position.y;
                const double dz = sources[j].z - position.z;
                const double invDist = __ddiv_rn(1.0, __dsqrt_rn(distanceSquared(dx, dy, dz)));
                const double invDist3 = __dmul_rn(__dmul_rn(invDist, invDist), invDist);
                fx = __fma_rn(dx, invDist3, fx);
                fy = __fma_rn(dy, invDist3, fy);
                fz = __fma_rn(dz, invDist3, fz);
            }
        }
        __syncthreads();
    }

    if (active) {
        Vec3 velocity = velocities[local];
        velocity.x = __dadd_rn(velocity.x, __dmul_rn(DT, fx));
        velocity.y = __dadd_rn(velocity.y, __dmul_rn(DT, fy));
        velocity.z = __dadd_rn(velocity.z, __dmul_rn(DT, fz));
        velocities[local] = velocity;
        // Separate input/output positions make force calculation and integration
        // safe to fuse: all forces still use positions from the preceding step.
        nextPositions[first + local] = Vec3{__fma_rn(velocity.x, DT, position.x),
                                            __fma_rn(velocity.y, DT, position.y),
                                            __fma_rn(velocity.z, DT, position.z)};
    }
}

// For small partitions, several lanes compute distances for each target. The
// leader still accumulates every source in order, so this has exactly the same
// rounding as the tiled kernel while exposing more parallel work to the GPU.
template <int LANES>
__global__ void computeStepCooperative(const Vec3* __restrict__ positions,
                                       Vec3* __restrict__ nextPositions,
                                       Vec3* __restrict__ velocities,
                                       int n, int first, int count) {
    const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t local = thread / LANES;
    const int lane = threadIdx.x % LANES;
    const bool active = local < static_cast<size_t>(count);
    const Vec3 position = active ? positions[first + local] : Vec3{0.0, 0.0, 0.0};
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (size_t base = 0; base < static_cast<size_t>(n); base += LANES) {
        double dx = 0.0, dy = 0.0, dz = 0.0, invDist3 = 0.0;
        if (active && base + lane < static_cast<size_t>(n)) {
            const Vec3 source = positions[base + lane];
            dx = source.x - position.x;
            dy = source.y - position.y;
            dz = source.z - position.z;
            const double invDist = __ddiv_rn(1.0, __dsqrt_rn(distanceSquared(dx, dy, dz)));
            invDist3 = __dmul_rn(__dmul_rn(invDist, invDist), invDist);
        }
        #pragma unroll
        for (int j = 0; j < LANES; ++j) {
            if (base + j < static_cast<size_t>(n)) {
                const double sx = __shfl_sync(0xffffffffu, dx, j, LANES);
                const double sy = __shfl_sync(0xffffffffu, dy, j, LANES);
                const double sz = __shfl_sync(0xffffffffu, dz, j, LANES);
                const double scale = __shfl_sync(0xffffffffu, invDist3, j, LANES);
                if (lane == 0) {
                    fx = __fma_rn(sx, scale, fx);
                    fy = __fma_rn(sy, scale, fy);
                    fz = __fma_rn(sz, scale, fz);
                }
            }
        }
    }
    if (active && lane == 0) {
        Vec3 velocity = velocities[local];
        velocity.x = __dadd_rn(velocity.x, __dmul_rn(DT, fx));
        velocity.y = __dadd_rn(velocity.y, __dmul_rn(DT, fy));
        velocity.z = __dadd_rn(velocity.z, __dmul_rn(DT, fz));
        velocities[local] = velocity;
        nextPositions[first + local] = Vec3{__fma_rn(velocity.x, DT, position.x),
                                            __fma_rn(velocity.y, DT, position.y),
                                            __fma_rn(velocity.z, DT, position.z)};
    }
}

__global__ void computeEnergy(const Vec3* __restrict__ positions,
                              const Vec3* __restrict__ velocities,
                              double* __restrict__ blockEnergy,
                              int n, int first, int count) {
    __shared__ Vec3 sources[TILE_SIZE];
    __shared__ double partials[TILE_SIZE];
    const size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = local < static_cast<size_t>(count);
    const size_t i = first + local;
    const Vec3 position = active ? positions[i] : Vec3{0.0, 0.0, 0.0};
    double energy = 0.0;
    if (active) {
        const Vec3 velocity = velocities[local];
        energy = 0.5 * (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
    }
    for (size_t base = 0; base < static_cast<size_t>(n); base += TILE_SIZE) {
        const size_t source = base + threadIdx.x;
        if (source < static_cast<size_t>(n)) sources[threadIdx.x] = positions[source];
        __syncthreads();
        const size_t remaining = static_cast<size_t>(n) - base;
        const int width = remaining < TILE_SIZE ? static_cast<int>(remaining) : TILE_SIZE;
        if (active) {
            for (int j = 0; j < width; ++j) {
                if (base + j > static_cast<size_t>(i)) {
                    const double dx = sources[j].x - position.x;
                    const double dy = sources[j].y - position.y;
                    const double dz = sources[j].z - position.z;
                    energy -= __ddiv_rn(1.0, __dsqrt_rn(distanceSquared(dx, dy, dz)));
                }
            }
        }
        __syncthreads();
    }
    partials[threadIdx.x] = energy;
    __syncthreads();
    for (int stride = TILE_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < static_cast<unsigned int>(stride))
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) blockEnergy[blockIdx.x] = partials[0];
}

// MPI implementations without device-buffer support use pinned staging memory.
// Only positions travel between ranks; velocities remain local to their owner.
void exchangePositions(Vec3* devicePositions, Vec3* hostPositions, int first, int count,
                       int n, const std::vector<int>& counts, const std::vector<int>& offsets,
                       MPI_Datatype positionType, bool deviceMPI, cudaStream_t stream) {
    if (!deviceMPI && count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(hostPositions + first, devicePositions + first,
                                   static_cast<size_t>(count) * sizeof(Vec3), cudaMemcpyDeviceToHost, stream));
    }
    // Standard MPI collectives have no CUDA stream dependency mechanism.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                            deviceMPI ? devicePositions : hostPositions,
                            counts.data(), offsets.data(), positionType, MPI_COMM_WORLD));
    if (!deviceMPI && n > 0) {
        CUDA_CHECK(cudaMemcpyAsync(devicePositions, hostPositions, static_cast<size_t>(n) * sizeof(Vec3),
                                   cudaMemcpyHostToDevice, stream));
    }
}

// Return failure categories so rank zero alone prints validation diagnostics.
int validateSimulation(const std::vector<Body>& bodies) {
    int failures = 0;
    #pragma omp parallel for schedule(static) reduction(|:failures)
    for (size_t i = 0; i < bodies.size(); ++i) {
        const auto& body = bodies[i];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            failures |= 1;
        } else if (std::abs(body.pos.x) > 1e6 || std::abs(body.pos.y) > 1e6 || std::abs(body.pos.z) > 1e6) {
            failures |= 2;
        } else if (std::abs(body.vel.x) > 1e6 || std::abs(body.vel.y) > 1e6 || std::abs(body.vel.z) > 1e6) {
            failures |= 4;
        }
    }
    return failures;
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

bool parseCount(const char* argument, int& value) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(argument, &end, 10);
    if (errno != 0 || end == argument || *end != '\0' || parsed < 0 || parsed > INT_MAX) return false;
    value = static_cast<int>(parsed);
    return true;
}

int runSimulation(int argc, char** argv) {
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseCount(argv[++i], numBodies)) {
                if (worldRank == 0) fprintf(stderr, "Invalid number of bodies: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            if (!parseCount(argv[++i], numSteps)) {
                if (worldRank == 0) fprintf(stderr, "Invalid number of steps: %s\n", argv[i]);
                return 1;
            }
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

    int ranks = 1;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    // Respects per-rank CUDA_VISIBLE_DEVICES as well as a shared node GPU list.
    const int device = localRank % devices;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));

    // Avoid spawning a whole node's CPU threads for each GPU rank by default.
    // An explicit OMP_NUM_THREADS always takes precedence.
    if (std::getenv("OMP_NUM_THREADS") == nullptr)
        omp_set_num_threads(std::min(8, omp_get_max_threads()));

    std::vector<int> counts(ranks), offsets(ranks);
    for (int rank = 0; rank < ranks; ++rank) {
        counts[rank] = numBodies / ranks + (rank < numBodies % ranks ? 1 : 0);
        offsets[rank] = rank * (numBodies / ranks) + std::min(rank, numBodies % ranks);
    }
    const int count = counts[worldRank];
    const int first = offsets[worldRank];
    // Small per-rank partitions need more blocks to spread work across the SMs.
    // Source tiles stay large even when there is only one target warp per block.
    const int forceThreads = static_cast<size_t>(count) < static_cast<size_t>(deviceProperties.multiProcessorCount) * TILE_SIZE * 4
                                 ? 32 : TILE_SIZE;
    const int forceLanes = count < deviceProperties.multiProcessorCount * 8 ? 8
                             : count < deviceProperties.multiProcessorCount * 32 ? 4 : 1;
    const int targetsPerBlock = forceThreads / forceLanes;
    const int forceBlocks = count / targetsPerBlock + (count % targetsPerBlock != 0);
    const int energyBlocks = count / TILE_SIZE + (count % TILE_SIZE != 0);

    MPI_Datatype positionType, bodyType;
    MPI_CHECK(MPI_Type_contiguous(3, MPI_DOUBLE, &positionType));
    MPI_CHECK(MPI_Type_commit(&positionType));
    MPI_CHECK(MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType));
    MPI_CHECK(MPI_Type_commit(&bodyType));

    int deviceMPI = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    deviceMPI = MPIX_Query_cuda_support() != 0;
#endif
    // All participants must select the same collective buffer location.
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &deviceMPI, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

    if (worldRank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Body> localBodies(count);
    {
        // Preserve exactly the original rand_r stream; rank ownership must not
        // change the initial state. The serial initialization is only O(N).
        std::vector<Body> bodies(worldRank == 0 ? numBodies : 0);
        if (worldRank == 0) randomizeBodies(bodies);
        Body dummy{};
        MPI_CHECK(MPI_Scatterv(bodies.empty() ? &dummy : bodies.data(), counts.data(), offsets.data(), bodyType,
                              localBodies.empty() ? &dummy : localBodies.data(), count, bodyType, 0, MPI_COMM_WORLD));
    }

    PinnedBuffer<Vec3> hostPositions(numBodies), hostVelocities(count);
    DeviceBuffer<Vec3> positionStorage(numBodies), nextStorage(numBodies), velocities(count);
    Vec3* positions = positionStorage.data;
    Vec3* nextPositions = nextStorage.data;
    Stream stream;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) {
        hostPositions.data[first + i] = localBodies[i].pos;
        hostVelocities.data[i] = localBodies[i].vel;
    }
    MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostPositions.data,
                            counts.data(), offsets.data(), positionType, MPI_COMM_WORLD));
    if (numBodies > 0)
        CUDA_CHECK(cudaMemcpyAsync(positions, hostPositions.data, static_cast<size_t>(numBodies) * sizeof(Vec3),
                                   cudaMemcpyHostToDevice, stream.stream));
    if (count > 0)
        CUDA_CHECK(cudaMemcpyAsync(velocities.data, hostVelocities.data, static_cast<size_t>(count) * sizeof(Vec3),
                                   cudaMemcpyHostToDevice, stream.stream));
    cudaFuncAttributes attributes;
    if (forceLanes == 8)
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, computeStepCooperative<8>));
    else if (forceLanes == 4)
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, computeStepCooperative<4>));
    else
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, computeStep));
    CUDA_CHECK(cudaStreamSynchronize(stream.stream));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        if (count > 0) {
            if (forceLanes == 8)
                computeStepCooperative<8><<<forceBlocks, forceThreads, 0, stream.stream>>>(
                    positions, nextPositions, velocities.data, numBodies, first, count);
            else if (forceLanes == 4)
                computeStepCooperative<4><<<forceBlocks, forceThreads, 0, stream.stream>>>(
                    positions, nextPositions, velocities.data, numBodies, first, count);
            else
                computeStep<<<forceBlocks, forceThreads, 0, stream.stream>>>(positions, nextPositions, velocities.data,
                                                                            numBodies, first, count);
            CUDA_CHECK(cudaGetLastError());
        }
        if (ranks > 1 && step + 1 < numSteps) {
            exchangePositions(nextPositions, hostPositions.data, first, count, numBodies,
                              counts, offsets, positionType, deviceMPI != 0, stream.stream);
        }
        std::swap(positions, nextPositions);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream.stream));
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (worldRank == 0) printf("Simulation time: %lld ms\n", static_cast<long long>(duration * 1000.0));

    if ((printResults || validate) && count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(hostPositions.data + first, positions + first, static_cast<size_t>(count) * sizeof(Vec3),
                                   cudaMemcpyDeviceToHost, stream.stream));
        CUDA_CHECK(cudaMemcpyAsync(hostVelocities.data, velocities.data, static_cast<size_t>(count) * sizeof(Vec3),
                                   cudaMemcpyDeviceToHost, stream.stream));
        CUDA_CHECK(cudaStreamSynchronize(stream.stream));
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < count; ++i) {
            localBodies[i].pos = hostPositions.data[first + i];
            localBodies[i].vel = hostVelocities.data[i];
        }
    }

    if (printResults) {
        // Both MPI datatypes describe doubles in the original serialization
        // order. Counts remain in bodies, avoiding 6*N MPI count overflow.
        std::vector<double> bodyData(worldRank == 0 ? static_cast<size_t>(numBodies) * 6 : 0);
        Body dummy{};
        MPI_CHECK(MPI_Gatherv(localBodies.empty() ? &dummy : localBodies.data(), count, bodyType,
                             bodyData.empty() ? static_cast<void*>(&dummy) : static_cast<void*>(bodyData.data()),
                             counts.data(), offsets.data(), bodyType, 0, MPI_COMM_WORLD));
        if (worldRank == 0) print_results(bodyData, "Bodies");
    }

    int exitCode = 0;
    if (validate) {
        if (worldRank == 0) printf("Validating simulation results...\n");
        int failures = validateSimulation(localBodies);
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD));
        if (failures == 0) {
            // The last step deliberately skipped its exchange. Validation's
            // all-pairs energy needs the complete final positions, outside the
            // simulation timer. Zero-step runs already have complete positions.
            if (ranks > 1 && numSteps > 0)
                exchangePositions(positions, hostPositions.data, first, count, numBodies,
                                  counts, offsets, positionType, deviceMPI != 0, stream.stream);
            DeviceBuffer<double> deviceEnergy(energyBlocks);
            PinnedBuffer<double> hostEnergy(energyBlocks);
            if (count > 0) {
                computeEnergy<<<energyBlocks, TILE_SIZE, 0, stream.stream>>>(positions, velocities.data, deviceEnergy.data,
                                                                            numBodies, first, count);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(hostEnergy.data, deviceEnergy.data, static_cast<size_t>(energyBlocks) * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream.stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream.stream));
            double localEnergy = 0.0;
            #pragma omp parallel for schedule(static) reduction(+:localEnergy)
            for (int i = 0; i < energyBlocks; ++i) localEnergy += hostEnergy.data[i];
            double finalEnergy = 0.0;
            MPI_CHECK(MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
            if (worldRank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (worldRank == 0) {
                if (failures & 1) printf("Validation failed: found NaN or Inf value in body state\n");
                else if (failures & 2) printf("Validation failed: body position exceeds reasonable bounds\n");
                else printf("Validation failed: body velocity exceeds reasonable bounds\n");
                printf("Validation: FAILED\n");
            }
            exitCode = 1;
        }
    }

    MPI_CHECK(MPI_Type_free(&bodyType));
    MPI_CHECK(MPI_Type_free(&positionType));
    return exitCode;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int error = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (error != MPI_SUCCESS) {
        fprintf(stderr, "MPI initialization failed\n");
        return 1;
    }
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    if (provided < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    int exitCode = 0;
    try {
        exitCode = runSimulation(argc, argv);
    } catch (const std::exception& exception) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, exception.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
