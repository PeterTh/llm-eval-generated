#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#ifdef OMPI_MAJOR_VERSION
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int VALUES_PER_BODY = 6;
constexpr int ENERGY_BLOCK_SIZE = 128;
static_assert(sizeof(double3) == 3 * sizeof(double));

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

template <typename T, bool Pinned = false>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t count) {
        if (count == 0) {
            return;
        }
        if constexpr (Pinned) {
            CUDA_CHECK(cudaMallocHost(&data, count * sizeof(T)));
        } else {
            CUDA_CHECK(cudaMalloc(&data, count * sizeof(T)));
        }
    }

    ~CudaBuffer() {
        if (data) {
            if constexpr (Pinned) {
                CUDA_CHECK(cudaFreeHost(data));
            } else {
                CUDA_CHECK(cudaFree(data));
            }
        }
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    T* get() const { return data; }

private:
    T* data = nullptr;
};

__device__ __forceinline__ double squaredDistance(double dx, double dy, double dz) {
    // Match the contraction and rounding of the original force calculation.
    const double xy = __fma_rn(dx, dx, dy * dy);
    return __fma_rn(dz, dz, xy) + SOFTENING;
}

__device__ __forceinline__ double inverseDistanceCubed(double dx, double dy, double dz) {
    const double invDist = __drcp_rn(__dsqrt_rn(squaredDistance(dx, dy, dz)));
    return invDist * invDist * invDist;
}

__device__ __forceinline__ void updateBody(double3 position, double3 force,
                                           double3& velocity, double3& updated) {
    double3 next = velocity;
    next.x += DT * force.x;
    next.y += DT * force.y;
    next.z += DT * force.z;
    velocity = next;
    updated = make_double3(__fma_rn(next.x, DT, position.x),
                           __fma_rn(next.y, DT, position.y),
                           __fma_rn(next.z, DT, position.z));
}

template <int BlockSize>
__global__ void advanceBodies(const double3* __restrict__ positions,
                              double3* __restrict__ velocities,
                              double3* __restrict__ updated,
                              int total, int offset, int count) {
    __shared__ double3 tile[BlockSize];
    const int local = blockIdx.x * BlockSize + threadIdx.x;
    const bool active = local < count;
    const double3 position = active ? positions[offset + local] : double3{};
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (size_t base = 0; base < static_cast<size_t>(total); base += BlockSize) {
        const int tileSize = min(BlockSize, static_cast<int>(total - base));
        if (threadIdx.x < tileSize) {
            tile[threadIdx.x] = positions[base + threadIdx.x];
        }
        __syncthreads();

        if (active) {
            #pragma unroll 4
            for (int source = 0; source < tileSize; ++source) {
                const double3 other = tile[source];
                const double dx = other.x - position.x;
                const double dy = other.y - position.y;
                const double dz = other.z - position.z;
                const double invDist3 = inverseDistanceCubed(dx, dy, dz);
                fx = __fma_rn(dx, invDist3, fx);
                fy = __fma_rn(dy, invDist3, fy);
                fz = __fma_rn(dz, invDist3, fz);
            }
        }
        __syncthreads();
    }

    if (active) {
        updateBody(position, make_double3(fx, fy, fz), velocities[local], updated[local]);
    }
}

constexpr int SMALL_BLOCK_SIZE = 64;
constexpr int LANES_PER_BODY = 8;
constexpr int BODIES_PER_SMALL_BLOCK = SMALL_BLOCK_SIZE / LANES_PER_BODY;

__global__ void advanceSmallPartition(const double3* __restrict__ positions,
                                      double3* __restrict__ velocities,
                                      double3* __restrict__ updated,
                                      int total, int offset, int count) {
    __shared__ double3 tile[SMALL_BLOCK_SIZE];
    __shared__ double inverseCube[BODIES_PER_SMALL_BLOCK][SMALL_BLOCK_SIZE + LANES_PER_BODY];
    const int target = threadIdx.x / LANES_PER_BODY;
    const int lane = threadIdx.x % LANES_PER_BODY;
    const int local = blockIdx.x * BODIES_PER_SMALL_BLOCK + target;
    const bool active = local < count;
    const double3 position = active ? positions[offset + local] : double3{};
    double fx = 0.0, fy = 0.0, fz = 0.0;

    // For small partitions, use more lanes to evaluate the distances while
    // one lane per body still accumulates all forces in source order.
    for (size_t base = 0; base < static_cast<size_t>(total); base += SMALL_BLOCK_SIZE) {
        const int tileSize = min(SMALL_BLOCK_SIZE, static_cast<int>(total - base));
        if (threadIdx.x < tileSize) {
            tile[threadIdx.x] = positions[base + threadIdx.x];
        }
        __syncthreads();
        if (active) {
            for (int source = lane; source < tileSize; source += LANES_PER_BODY) {
                const double3 other = tile[source];
                inverseCube[target][source] = inverseDistanceCubed(other.x - position.x,
                                                                   other.y - position.y,
                                                                   other.z - position.z);
            }
        }
        __syncwarp();
        if (active && lane == 0) {
            #pragma unroll 4
            for (int source = 0; source < tileSize; ++source) {
                const double3 other = tile[source];
                const double factor = inverseCube[target][source];
                fx = __fma_rn(other.x - position.x, factor, fx);
                fy = __fma_rn(other.y - position.y, factor, fy);
                fz = __fma_rn(other.z - position.z, factor, fz);
            }
        }
        __syncthreads();
    }

    if (active && lane == 0) {
        updateBody(position, make_double3(fx, fy, fz), velocities[local], updated[local]);
    }
}

__global__ void potentialEnergy(const double3* __restrict__ positions,
                                double* __restrict__ partial, int total,
                                int rank, int ranks) {
    __shared__ double warpSums[ENERGY_BLOCK_SIZE / 32];
    // Cyclic rows balance the upper triangle across ranks.
    const size_t numBodies = total;
    const size_t first = rank + static_cast<size_t>(blockIdx.x) * ENERGY_BLOCK_SIZE * ranks;
    const size_t i = first + static_cast<size_t>(threadIdx.x) * ranks;
    double energy = 0.0;
    if (i < numBodies) {
        const double3 position = positions[i];
        for (size_t j = first + 1; j < numBodies; ++j) {
            if (j > i) {
                const double3 other = positions[j];
                const double dx = other.x - position.x;
                const double dy = other.y - position.y;
                const double dz = other.z - position.z;
                energy -= __drcp_rn(__dsqrt_rn(squaredDistance(dx, dy, dz)));
            }
        }
    }

    for (int delta = 16; delta > 0; delta /= 2) {
        energy += __shfl_down_sync(0xffffffff, energy, delta);
    }
    const int lane = threadIdx.x % 32;
    if (lane == 0) {
        warpSums[threadIdx.x / 32] = energy;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        energy = lane < ENERGY_BLOCK_SIZE / 32 ? warpSums[lane] : 0.0;
        for (int delta = ENERGY_BLOCK_SIZE / 64; delta > 0; delta /= 2) {
            energy += __shfl_down_sync(0xffffffff, energy, delta);
        }
        if (lane == 0) {
            partial[blockIdx.x] = energy;
        }
    }
}

bool deviceCollectivesAvailable() {
    #if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    return MPIX_Query_cuda_support() != 0;
    #elif defined(MPIX_GPU_SUPPORT_CUDA)
    int supported = 0;
    return MPIX_GPU_query_support(MPIX_GPU_SUPPORT_CUDA, &supported) == MPI_SUCCESS && supported;
    #else
    return false;
    #endif
}

void allGatherPositions(const double3* send, double3* receive,
                        const std::vector<int>& counts, const std::vector<int>& offsets,
                        MPI_Datatype positionType, int rank, MPI_Comm comm) {
    const void* source = send ? static_cast<const void*>(send) : MPI_IN_PLACE;
    if (counts.front() == counts.back()) {
        MPI_Allgather(source, counts[rank], positionType, receive, counts.front(), positionType, comm);
    } else {
        MPI_Allgatherv(source, counts[rank], positionType, receive,
                       counts.data(), offsets.data(), positionType, comm);
    }
}

int blocksFor(int count, int blockSize) {
    return count / blockSize + (count % blockSize != 0);
}

int hostThreads(int elements) {
    const int usefulThreads = std::max(2, blocksFor(elements, 4096));
    return std::min(omp_get_max_threads(), std::max(1, std::min(elements, usefulThreads)));
}

enum class InvalidState : uint64_t { Nonfinite, Position, Velocity };
constexpr uint64_t VALID_STATE = std::numeric_limits<uint64_t>::max();

uint64_t validateBody(double3 pos, double3 vel, uint64_t index) {
    InvalidState reason;
    if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
        !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) {
        reason = InvalidState::Nonfinite;
    } else if (std::abs(pos.x) > 1e6 || std::abs(pos.y) > 1e6 || std::abs(pos.z) > 1e6) {
        reason = InvalidState::Position;
    } else if (std::abs(vel.x) > 1e6 || std::abs(vel.y) > 1e6 || std::abs(vel.z) > 1e6) {
        reason = InvalidState::Velocity;
    } else {
        return VALID_STATE;
    }
    // MPI_MIN then selects the same first invalid body as the serial scan.
    return index * 3 + static_cast<uint64_t>(reason);
}

int runSimulation(int numBodies, int numSteps, bool validate, bool printResults, MPI_Comm comm) {
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    int devices;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        if (localRank == 0) {
            fprintf(stderr, "No CUDA devices are visible on this node\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % devices;
    CUDA_CHECK(cudaSetDevice(device));
    int multiprocessors;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));

    std::vector<int> counts(ranks), offsets(ranks);
    for (int peer = 0, offset = 0; peer < ranks; ++peer) {
        counts[peer] = numBodies / ranks + (peer < numBodies % ranks);
        offsets[peer] = offset;
        offset += counts[peer];
    }
    int count = counts[rank];
    int offset = offsets[rank];

    MPI_Datatype positionType, randomBodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &positionType);
    MPI_Type_commit(&positionType);
    MPI_Type_contiguous(VALUES_PER_BODY, MPI_INT, &randomBodyType);
    MPI_Type_commit(&randomBodyType);

    int localDeviceCollectives = ranks > 1 && deviceCollectivesAvailable();
    int deviceCollectives = 0;
    if (ranks > 1) {
        MPI_Allreduce(&localDeviceCollectives, &deviceCollectives, 1, MPI_INT, MPI_MIN, comm);
    }

    CudaBuffer<double3> positionBuffer(numBodies), updatedBuffer(count), velocities(count);
    CudaBuffer<double3, true> stagedPositions(ranks > 1 && !deviceCollectives ? numBodies : 0);
    CudaBuffer<double3, true> hostPositions(printResults || validate || !stagedPositions.get() ? count : 0);
    CudaBuffer<double3, true> hostVelocities(count);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    double3* positions = positionBuffer.get();
    double3* updated = updatedBuffer.get();
    const size_t localBytes = static_cast<size_t>(count) * sizeof(double3);
    const size_t globalBytes = static_cast<size_t>(numBodies) * sizeof(double3);

    {
        // Preserve the original rand_r stream, then decode each rank's share in parallel.
        std::vector<int> localRandom(static_cast<size_t>(count) * VALUES_PER_BODY);
        {
            std::vector<int> randomValues(rank == 0 ? static_cast<size_t>(numBodies) * VALUES_PER_BODY : 0);
            if (rank == 0) {
                unsigned int seed = 42;
                for (int& value : randomValues) {
                    value = rand_r(&seed);
                }
            }
            MPI_Scatterv(randomValues.data(), counts.data(), offsets.data(), randomBodyType,
                         localRandom.data(), count, randomBodyType, 0, comm);
        }

        double3* initialPositions = stagedPositions.get() ? stagedPositions.get() + offset : hostPositions.get();
        #pragma omp parallel for schedule(static) num_threads(hostThreads(count))
        for (int i = 0; i < count; ++i) {
            const int* values = localRandom.data() + static_cast<size_t>(i) * VALUES_PER_BODY;
            double coordinates[VALUES_PER_BODY];
            for (int component = 0; component < VALUES_PER_BODY; ++component) {
                coordinates[component] = 2.0 * (values[component] / static_cast<double>(RAND_MAX)) - 1.0;
            }
            initialPositions[i] = make_double3(coordinates[0], coordinates[1], coordinates[2]);
            hostVelocities.get()[i] = make_double3(coordinates[3], coordinates[4], coordinates[5]);
        }
        if (count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(velocities.get(), hostVelocities.get(), localBytes, cudaMemcpyHostToDevice, stream));
            if (ranks == 1) {
                CUDA_CHECK(cudaMemcpyAsync(positions, initialPositions, localBytes, cudaMemcpyHostToDevice, stream));
            } else if (deviceCollectives) {
                CUDA_CHECK(cudaMemcpyAsync(updated, initialPositions, localBytes, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                allGatherPositions(updated, positions, counts, offsets, positionType, rank, comm);
            } else {
                allGatherPositions(nullptr, stagedPositions.get(), counts, offsets, positionType, rank, comm);
                CUDA_CHECK(cudaMemcpyAsync(positions, stagedPositions.get(), globalBytes, cudaMemcpyHostToDevice, stream));
            }
        }
    }
    MPI_Type_free(&randomBodyType);

    const auto exchangePositions = [&] {
        if (deviceCollectives) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            allGatherPositions(updated, positions, counts, offsets, positionType, rank, comm);
        } else {
            double3* staging = stagedPositions.get();
            CUDA_CHECK(cudaMemcpyAsync(staging + offset, updated, localBytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            allGatherPositions(nullptr, staging, counts, offsets, positionType, rank, comm);
            CUDA_CHECK(cudaMemcpyAsync(positions, staging, globalBytes, cudaMemcpyHostToDevice, stream));
        }
    };

    int blockSize = 128;
    int blocks = blocksFor(count, blockSize);
    auto advance = advanceBodies<128>;
    if (count <= 16 * multiprocessors) {
        blockSize = SMALL_BLOCK_SIZE;
        blocks = blocksFor(count, BODIES_PER_SMALL_BLOCK);
        advance = advanceSmallPartition;
    } else if (count <= 64 * multiprocessors) {
        blockSize = 32;
        blocks = blocksFor(count, blockSize);
        advance = advanceBodies<32>;
    } else if (count <= 256 * multiprocessors) {
        blockSize = 64;
        blocks = blocksFor(count, blockSize);
        advance = advanceBodies<64>;
    }
    auto* localVelocities = velocities.get();
    void* kernelArgs[] = {&positions, &localVelocities, &updated, &numBodies, &offset, &count};
    if (numSteps > 0 && count > 0) {
        // Load the selected kernel before measuring the simulation.
        cudaFuncAttributes attributes;
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, advance));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(comm);
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < numSteps && count > 0; ++step) {
        CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(advance), blocks, blockSize, kernelArgs, 0, stream));
        if (ranks == 1) {
            std::swap(positions, updated);
        } else if (step + 1 < numSteps) {
            exchangePositions();
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    double slowest = 0.0;
    MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(slowest));
    }

    const bool distributedUpdates = ranks > 1 && numSteps > 0;
    if ((printResults || validate) && count > 0) {
        const double3* localPositions = distributedUpdates ? updated : positions + offset;
        CUDA_CHECK(cudaMemcpyAsync(hostPositions.get(), localPositions, localBytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hostVelocities.get(), velocities.get(), localBytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (printResults) {
        std::vector<double> localData(static_cast<size_t>(count) * VALUES_PER_BODY);
        #pragma omp parallel for schedule(static) num_threads(hostThreads(count))
        for (int i = 0; i < count; ++i) {
            const double3 pos = hostPositions.get()[i];
            const double3 vel = hostVelocities.get()[i];
            double* body = localData.data() + static_cast<size_t>(i) * VALUES_PER_BODY;
            body[0] = pos.x;
            body[1] = pos.y;
            body[2] = pos.z;
            body[3] = vel.x;
            body[4] = vel.y;
            body[5] = vel.z;
        }
        std::vector<double> data(rank == 0 ? static_cast<size_t>(numBodies) * VALUES_PER_BODY : 0);
        MPI_Datatype bodyType;
        MPI_Type_contiguous(VALUES_PER_BODY, MPI_DOUBLE, &bodyType);
        MPI_Type_commit(&bodyType);
        MPI_Gatherv(localData.data(), count, bodyType, data.data(), counts.data(), offsets.data(), bodyType, 0, comm);
        MPI_Type_free(&bodyType);
        if (rank == 0) {
            print_results(data, "Bodies");
        }
    }

    int status = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }
        uint64_t localInvalid = VALID_STATE;
        double kinetic = 0.0;
        #pragma omp parallel for schedule(static) num_threads(hostThreads(count)) reduction(min:localInvalid) reduction(+:kinetic)
        for (int i = 0; i < count; ++i) {
            const double3 pos = hostPositions.get()[i];
            const double3 vel = hostVelocities.get()[i];
            localInvalid = std::min(localInvalid, validateBody(pos, vel, static_cast<uint64_t>(offset) + i));
            kinetic += 0.5 * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
        }

        uint64_t invalid;
        MPI_Allreduce(&localInvalid, &invalid, 1, MPI_UINT64_T, MPI_MIN, comm);
        if (invalid == VALID_STATE) {
            double localEnergy = kinetic;
            if (numBodies > 1) {
                if (distributedUpdates) {
                    exchangePositions();
                }
                const int blocks = blocksFor(count, ENERGY_BLOCK_SIZE);
                CudaBuffer<double> partial(blocks);
                CudaBuffer<double, true> hostPartial(blocks);
                potentialEnergy<<<blocks, ENERGY_BLOCK_SIZE, 0, stream>>>(positions, partial.get(), numBodies, rank, ranks);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(hostPartial.get(), partial.get(), blocks * sizeof(double), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                for (int block = 0; block < blocks; ++block) {
                    localEnergy += hostPartial.get()[block];
                }
            }
            double finalEnergy = 0.0;
            MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            status = 1;
            if (rank == 0) {
                switch (static_cast<InvalidState>(invalid % 3)) {
                    case InvalidState::Nonfinite:
                        printf("Validation failed: found NaN or Inf value in body state\n");
                        break;
                    case InvalidState::Position:
                        printf("Validation failed: body position exceeds reasonable bounds\n");
                        break;
                    case InvalidState::Velocity:
                        printf("Validation failed: body velocity exceeds reasonable bounds\n");
                        break;
                }
                printf("Validation: FAILED\n");
            }
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Type_free(&positionType);
    return status;
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

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return 1;
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else {
            const bool help = strcmp(argv[i], "-h") == 0;
            if (rank == 0) {
                if (!help) {
                    printf("Unknown option: %s\n", argv[i]);
                }
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return help ? 0 : 1;
        }
    }

    if (numBodies < 0) {
        if (rank == 0) {
            fprintf(stderr, "Number of bodies must be nonnegative\n");
        }
        MPI_Finalize();
        return 1;
    }

    int status;
    if (ranks > std::max(1, numBodies)) {
        // Exclude empty ranks from the GPU work and position exchange.
        MPI_Comm active;
        MPI_Comm_split(MPI_COMM_WORLD, rank < std::max(1, numBodies) ? 0 : MPI_UNDEFINED, rank, &active);
        status = 0;
        if (active != MPI_COMM_NULL) {
            status = runSimulation(numBodies, numSteps, validate, printResults, active);
            MPI_Comm_free(&active);
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        status = runSimulation(numBodies, numSteps, validate, printResults, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return status;
}
