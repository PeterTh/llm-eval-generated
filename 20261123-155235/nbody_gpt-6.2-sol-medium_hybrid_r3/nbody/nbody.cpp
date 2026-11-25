#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 64;
constexpr int BODIES_PER_HOST_THREAD = 4096;

struct Vec3 {
    double x, y, z;
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(sizeof(Body) == 6 * sizeof(double));

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

template <typename T>
auto deviceBuffer(size_t count) {
    T* data = nullptr;
    checkCuda(cudaMalloc(&data, std::max(count, size_t{1}) * sizeof(T)), "allocate GPU buffer");
    return std::unique_ptr<T, decltype(&cudaFree)>(data, cudaFree);
}

template <typename T>
void copyToDevice(T* device, const T* host, size_t count) {
    if (count != 0) {
        checkCuda(cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice),
                  "copy to GPU");
    }
}

template <typename T>
void copyToHost(T* host, const T* device, size_t count) {
    if (count != 0) {
        checkCuda(cudaMemcpy(host, device, count * sizeof(T), cudaMemcpyDeviceToHost),
                  "copy from GPU");
    }
}

void randomizeBodies(std::vector<Body>& bodies) {
    unsigned int seed = 42;
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

__device__ double distanceSquared(double dx, double dy, double dz) {
    // Spell out the contractions used by the original optimized CPU loop.
    return fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
}

template <int TileSize>
__global__ void advanceBodies(const Vec3* __restrict__ positions,
                              Vec3* __restrict__ nextPositions,
                              Vec3* __restrict__ velocities,
                              size_t total, size_t offset, size_t count) {
    __shared__ Vec3 tile[TileSize];
    const size_t i = size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const bool active = i < count;
    const Vec3 position = active ? positions[offset + i] : Vec3{};
    Vec3 force{};

    for (size_t base = 0; base < total; base += TileSize) {
        const size_t source = base + threadIdx.x;
        if (source < total) {
            tile[threadIdx.x] = positions[source];
        }
        __syncthreads();

        if (active) {
            const int tileCount = min(size_t{TileSize}, total - base);
            #pragma unroll 4
            for (int j = 0; j < tileCount; ++j) {
                const double dx = tile[j].x - position.x;
                const double dy = tile[j].y - position.y;
                const double dz = tile[j].z - position.z;
                const double invDist = __drcp_rn(__dsqrt_rn(distanceSquared(dx, dy, dz)));
                const double invDist3 = invDist * invDist * invDist;
                force.x = fma(dx, invDist3, force.x);
                force.y = fma(dy, invDist3, force.y);
                force.z = fma(dz, invDist3, force.z);
            }
        }
        __syncthreads();
    }

    // Write to a separate position buffer so all targets use the old positions.
    if (active) {
        Vec3 velocity = velocities[i];
        velocity.x += DT * force.x;
        velocity.y += DT * force.y;
        velocity.z += DT * force.z;
        velocities[i] = velocity;
        nextPositions[i] = {fma(velocity.x, DT, position.x),
                            fma(velocity.y, DT, position.y),
                            fma(velocity.z, DT, position.z)};
    }
}

__device__ double potentialEnergy(Vec3 a, Vec3 b) {
    return -rsqrt(distanceSquared(b.x - a.x, b.y - a.y, b.z - a.z));
}

__global__ void bodyEnergies(const Vec3* __restrict__ positions,
                             const Vec3* __restrict__ velocities,
                             double* __restrict__ energies,
                             size_t total, size_t offset, size_t count) {
    const size_t i = size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const size_t globalIndex = offset + i;
    const Vec3 position = positions[globalIndex];
    const Vec3 velocity = velocities[i];
    double energy = 0.5 * (velocity.x * velocity.x + velocity.y * velocity.y +
                           velocity.z * velocity.z);

    // Each body visits the next half of the circle. For even totals, assign
    // the diametrically opposite pair to its lower index.
    const size_t end = globalIndex + 1 + (total - 1) / 2 +
                       (total % 2 == 0 && globalIndex < total / 2);
    const size_t firstEnd = min(end, total);
    for (size_t j = globalIndex + 1; j < firstEnd; ++j) {
        energy += potentialEnergy(position, positions[j]);
    }
    for (size_t j = total; j < end; ++j) {
        energy += potentialEnergy(position, positions[j - total]);
    }
    energies[i] = energy;
}

int bodyError(const Vec3& pos, const Vec3& vel) {
    if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
        !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) {
        return 0;
    }
    constexpr double limit = 1e6;
    if (std::abs(pos.x) > limit || std::abs(pos.y) > limit || std::abs(pos.z) > limit) {
        return 1;
    }
    if (std::abs(vel.x) > limit || std::abs(vel.y) > limit || std::abs(vel.z) > limit) {
        return 2;
    }
    return -1;
}

int selectDevice() {
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int localRank;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);

    int devices;
    checkCuda(cudaGetDeviceCount(&devices), "enumerate CUDA devices");
    if (devices == 0) {
        checkCuda(cudaErrorNoDevice, "select CUDA device");
    }
    // A scheduler may expose just the GPU assigned to the rank.
    const int device = localRank % devices;
    checkCuda(cudaSetDevice(device), "select CUDA device");
    return device;
}

int simulate(int numBodies, int numSteps, bool validate, bool printResults,
             int rank, int ranks) {
    const int device = selectDevice();

    MPI_Datatype positionType, bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &positionType);
    MPI_Type_commit(&positionType);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    const int bodiesPerRank = numBodies / ranks;
    const int remainder = numBodies % ranks;
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = bodiesPerRank + (r < remainder);
        offsets[r] = r * bodiesPerRank + std::min(r, remainder);
    }
    const int count = counts[rank];
    const int offset = offsets[rank];
    const int hostThreads = std::min({omp_get_max_threads(), std::max(count, 1),
                                      std::max(2, count / BODIES_PER_HOST_THREAD)});

    std::vector<Body> allBodies(rank == 0 ? numBodies : 0);
    if (rank == 0) {
        randomizeBodies(allBodies);
    }
    // Leave a valid buffer on ranks with no owned bodies, including for MPI.
    std::vector<Body> localBodies(std::max(count, 1));
    MPI_Scatterv(allBodies.data(), counts.data(), offsets.data(), bodyType,
                 localBodies.data(), count, bodyType, 0, MPI_COMM_WORLD);

    Vec3* stagingBuffer = nullptr;
    checkCuda(cudaMallocHost(&stagingBuffer, std::max(numBodies, 1) * sizeof(Vec3)),
              "allocate pinned position buffer");
    const std::unique_ptr<Vec3, decltype(&cudaFreeHost)> staging(stagingBuffer, cudaFreeHost);
    Vec3* const localPositions = staging.get() + offset;
    std::vector<Vec3> localVelocities(count);
    #pragma omp parallel for schedule(static) num_threads(hostThreads)
    for (int i = 0; i < count; ++i) {
        localPositions[i] = localBodies[i].pos;
        localVelocities[i] = localBodies[i].vel;
    }
    if (!printResults) {
        std::vector<Body>().swap(allBodies);
        std::vector<Body>().swap(localBodies);
    }

    auto positions = deviceBuffer<Vec3>(numBodies);
    auto nextPositions = deviceBuffer<Vec3>(count);
    auto velocities = deviceBuffer<Vec3>(count);
    copyToDevice(velocities.get(), localVelocities.data(), count);
    if (!printResults && !validate) {
        std::vector<Vec3>().swap(localVelocities);
    }

    const auto exchangePositions = [&] {
        if (ranks > 1 && remainder == 0) {
            MPI_Allgather(MPI_IN_PLACE, 0, positionType, staging.get(), count, positionType,
                          MPI_COMM_WORLD);
        } else if (ranks > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, staging.get(), counts.data(),
                           offsets.data(), positionType, MPI_COMM_WORLD);
        }
        copyToDevice(positions.get(), staging.get(), numBodies);
    };
    exchangePositions();

    int multiprocessors;
    checkCuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device),
              "query GPU multiprocessor count");
    // Smaller blocks give short target ranges more blocks to schedule on the GPU.
    constexpr int smallBlock = 32;
    const bool smallRange = count < 2 * multiprocessors * BLOCK_SIZE;
    const unsigned blocks = (size_t(count) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const unsigned smallBlocks = (size_t(count) + smallBlock - 1) / smallBlock;
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps && numBodies > 0; ++step) {
        if (count > 0) {
            if (smallRange) {
                advanceBodies<smallBlock><<<smallBlocks, smallBlock>>>(
                    positions.get(), nextPositions.get(), velocities.get(), numBodies, offset, count);
            } else {
                advanceBodies<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
                    positions.get(), nextPositions.get(), velocities.get(), numBodies, offset, count);
            }
            checkCuda(cudaGetLastError(), "launch force and integration kernel");
        }
        if (ranks == 1) {
            positions.swap(nextPositions);
        } else if (step + 1 < numSteps) {
            copyToHost(localPositions, nextPositions.get(), count);
            exchangePositions();
        }
    }
    checkCuda(cudaDeviceSynchronize(), "complete simulation");
    const double elapsed = MPI_Wtime() - start;
    double longest = 0.0;
    MPI_Reduce(&elapsed, &longest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Simulation time: %lld ms\n", static_cast<long long>(longest * 1000));
    }

    if (printResults || validate) {
        const Vec3* finalPositions = numSteps > 0 && ranks > 1
                                    ? nextPositions.get() : positions.get() + offset;
        copyToHost(localPositions, finalPositions, count);
        copyToHost(localVelocities.data(), velocities.get(), count);
    }

    if (printResults) {
        #pragma omp parallel for schedule(static) num_threads(hostThreads)
        for (int i = 0; i < count; ++i) {
            localBodies[i] = {localPositions[i], localVelocities[i]};
        }
        MPI_Gatherv(localBodies.data(), count, bodyType, allBodies.data(), counts.data(),
                    offsets.data(), bodyType, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<double> bodyData(size_t(numBodies) * 6);
            if (numBodies > 0) {
                std::memcpy(bodyData.data(), allBodies.data(), allBodies.size() * sizeof(Body));
            }
            print_results(bodyData, "Bodies");
        }
    }
    MPI_Type_free(&bodyType);

    int result = 0;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating simulation results...\n");
        }
        constexpr long long noError = std::numeric_limits<long long>::max();
        long long localError = noError;
        #pragma omp parallel for schedule(static) num_threads(hostThreads) reduction(min:localError)
        for (int i = 0; i < count; ++i) {
            const int error = bodyError(localPositions[i], localVelocities[i]);
            if (error >= 0) {
                localError = std::min(localError, 3LL * (offset + i) + error);
            }
        }

        long long firstError;
        MPI_Allreduce(&localError, &firstError, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
        if (firstError == noError) {
            if (ranks > 1 && numSteps > 0) {
                exchangePositions();
            }
            double localEnergy = 0.0;
            if (count > 0) {
                auto energies = deviceBuffer<double>(count);
                bodyEnergies<<<blocks, BLOCK_SIZE>>>(positions.get(), velocities.get(),
                                                     energies.get(), numBodies, offset, count);
                checkCuda(cudaGetLastError(), "launch energy kernel");
                std::vector<double> hostEnergies(count);
                copyToHost(hostEnergies.data(), energies.get(), count);
                #pragma omp parallel for schedule(static) num_threads(hostThreads) reduction(+:localEnergy)
                for (int i = 0; i < count; ++i) {
                    localEnergy += hostEnergies[i];
                }
            }
            double finalEnergy = 0.0;
            MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                std::printf("Final energy: %.6f\n", finalEnergy);
                std::printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                const char* errors[] = {"found NaN or Inf value in body state",
                                        "body position exceeds reasonable bounds",
                                        "body velocity exceeds reasonable bounds"};
                std::printf("Validation failed: %s\n", errors[firstError % 3]);
                std::printf("Validation: FAILED\n");
            }
            result = 1;
        }
    }

    MPI_Type_free(&positionType);
    return result;
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

int run(int argc, char** argv, int rank, int ranks) {
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
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (numBodies < 0) {
        if (rank == 0) {
            std::fprintf(stderr, "Number of bodies must be nonnegative\n");
        }
        return 1;
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    return simulate(numBodies, numSteps, validate, printResults, rank, ranks);
}

} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return 1;
    }
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int result = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return result;
}
