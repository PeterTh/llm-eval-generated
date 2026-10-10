#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int TILE_SIZE = 128;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(double x = 0, double y = 0, double z = 0) noexcept
        : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double), "MPI positions must be packed");
static int processRank = 0;

void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        std::fprintf(stderr, "Rank %d: MPI error: %.*s\n", processRank, length, message);
        MPI_Abort(MPI_COMM_WORLD, error);
        std::abort();
    }
}

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA error: %s\n", processRank,
                     cudaGetErrorString(error));
        // CUDA calls also run on OpenMP workers. Abort locally so that the MPI
        // launcher terminates the job without making MPI calls off the main thread.
        std::abort();
    }
}

template <class T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t count) {
        cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&data_),
                                std::max(size_t{1}, count) * sizeof(T),
                                cudaHostAllocPortable));
    }
    ~PinnedBuffer() { cudaFreeHost(data_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    T* data() const { return data_; }
    T& operator[](size_t i) const { return data_[i]; }
private:
    T* data_ = nullptr;
};

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Keep the original rand_r stream, including the interleaving of position
    // and velocity draws. Independent per-thread seeds would change the input.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Each thread owns one target body. Source tiles are loaded cooperatively and
// reused by the entire block; each target sums j in the original ascending order.
// The output positions are separate from the snapshot read by every thread, so
// force evaluation and integration can share a launch without a global race.
template <int BlockSize>
__global__ void advanceBodies(const Vec3* __restrict__ positions,
                              Vec3* __restrict__ velocities,
                              Vec3* __restrict__ nextPositions,
                              int first, int count, int n) {
    __shared__ double sourceX[TILE_SIZE];
    __shared__ double sourceY[TILE_SIZE];
    __shared__ double sourceZ[TILE_SIZE];
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = i < static_cast<size_t>(count);
    const Vec3 p = active ? positions[first + i] : Vec3{};
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (size_t base = 0; base < static_cast<size_t>(n); base += TILE_SIZE) {
        #pragma unroll
        for (int k = threadIdx.x; k < TILE_SIZE; k += BlockSize) {
            const size_t j = base + k;
            if (j < static_cast<size_t>(n)) {
                const Vec3 q = positions[j];
                sourceX[k] = q.x;
                sourceY[k] = q.y;
                sourceZ[k] = q.z;
            }
        }
        __syncthreads();
        const int tileCount = static_cast<int>(
            min(size_t(TILE_SIZE), static_cast<size_t>(n) - base));
        if (active) {
            #pragma unroll 8
            for (int k = 0; k < tileCount; ++k) {
                const double dx = sourceX[k] - p.x;
                const double dy = sourceY[k] - p.y;
                const double dz = sourceZ[k] - p.z;
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
        nextPositions[i] = Vec3(p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT);
    }
}

// Small target partitions cannot fill a GPU with one thread per body. In that
// case, expose the independent pairs to the whole device, then sum the stored
// contributions in exactly the same order. Scratch space is strictly bounded;
// large problems use the shared-memory tiled kernel above.
__global__ void computeInteractions(const Vec3* __restrict__ positions,
                                    Vec3* __restrict__ interactions,
                                    int first, int n) {
    const size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y;
    if (j >= static_cast<size_t>(n)) return;
    const Vec3 p = positions[first + i], q = positions[j];
    const double dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
    const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
    const double invDist = 1.0 / sqrt(distSqr);
    const double invDist3 = invDist * invDist * invDist;
    interactions[i * n + j] = Vec3(dx * invDist3, dy * invDist3, dz * invDist3);
}

__global__ void integrateInteractions(const Vec3* __restrict__ positions,
                                      const Vec3* __restrict__ interactions,
                                      Vec3* __restrict__ velocities,
                                      Vec3* __restrict__ nextPositions,
                                      int first, int count, int n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(count)) return;
    const Vec3* row = interactions + i * n;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < n; ++j) {
        const Vec3 force = row[j];
        fx += force.x;
        fy += force.y;
        fz += force.z;
    }
    Vec3 v = velocities[i];
    v.x += DT * fx;
    v.y += DT * fy;
    v.z += DT * fz;
    velocities[i] = v;
    const Vec3 p = positions[first + i];
    nextPositions[i] = Vec3(p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT);
}

// Each unordered pair contributes exactly once. Energy is computed on all GPUs
// and reduced on the host and across MPI ranks only when validation is requested.
__global__ void bodyEnergies(const Vec3* __restrict__ positions,
                             const Vec3* __restrict__ velocities,
                             double* __restrict__ energies,
                             int first, int count, int n) {
    __shared__ double sourceX[TILE_SIZE];
    __shared__ double sourceY[TILE_SIZE];
    __shared__ double sourceZ[TILE_SIZE];
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = i < static_cast<size_t>(count);
    const size_t globalI = static_cast<size_t>(first) + i;
    const Vec3 p = active ? positions[globalI] : Vec3{};
    double potential = 0.0;
    // All targets in a block can skip complete tiles preceding their indices.
    const size_t start = (static_cast<size_t>(first) + size_t(blockIdx.x) * blockDim.x)
                       / TILE_SIZE * TILE_SIZE;
    for (size_t base = start; base < static_cast<size_t>(n); base += TILE_SIZE) {
        const size_t j = base + threadIdx.x;
        if (j < static_cast<size_t>(n)) {
            const Vec3 q = positions[j];
            sourceX[threadIdx.x] = q.x;
            sourceY[threadIdx.x] = q.y;
            sourceZ[threadIdx.x] = q.z;
        }
        __syncthreads();
        const int tileCount = static_cast<int>(
            min(size_t(TILE_SIZE), static_cast<size_t>(n) - base));
        if (active) {
            #pragma unroll 8
            for (int k = 0; k < tileCount; ++k) {
                if (base + k > globalI) {
                    const double dx = sourceX[k] - p.x;
                    const double dy = sourceY[k] - p.y;
                    const double dz = sourceZ[k] - p.z;
                    potential -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                }
            }
        }
        __syncthreads();
    }
    if (active) {
        const Vec3 v = velocities[i];
        energies[i] = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z) + potential;
    }
}

struct Accelerator {
    int device = 0;
    int first = 0;
    int count = 0;
    Vec3* positions = nullptr;
    Vec3* velocities = nullptr;
    Vec3* nextPositions = nullptr;
    Vec3* latestPositions = nullptr;
    Vec3* interactions = nullptr;
    double* energies = nullptr;
    cudaStream_t stream = nullptr;
    int forceBlockSize = TILE_SIZE;

    void initialize(int n, int rankFirst, const Vec3* hostPositions,
                    const Vec3* hostVelocities, bool validate) {
        cudaCheck(cudaSetDevice(device));
        if (count == 0) return;
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cudaCheck(cudaMalloc(&positions, size_t(n) * sizeof(Vec3)));
        cudaCheck(cudaMalloc(&velocities, size_t(count) * sizeof(Vec3)));
        cudaCheck(cudaMalloc(&nextPositions, size_t(count) * sizeof(Vec3)));
        cudaDeviceProp properties;
        cudaCheck(cudaGetDeviceProperties(&properties, device));
        // Smaller blocks spread sparse target partitions over more SMs while
        // still reusing a full source tile within every block.
        if (count < properties.multiProcessorCount * TILE_SIZE) forceBlockSize = 32;
        const size_t pairs = size_t(n) * count;
        if (count < properties.multiProcessorCount * TILE_SIZE && pairs <= 4 * 1024 * 1024) {
            cudaCheck(cudaMalloc(&interactions, pairs * sizeof(Vec3)));
        }
        if (validate) cudaCheck(cudaMalloc(&energies, size_t(count) * sizeof(double)));
        cudaCheck(cudaMemcpyAsync(positions, hostPositions, size_t(n) * sizeof(Vec3),
                                  cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaMemcpyAsync(velocities, hostVelocities + first - rankFirst,
                                  size_t(count) * sizeof(Vec3), cudaMemcpyHostToDevice, stream));
        latestPositions = positions + first;
        cudaCheck(cudaStreamSynchronize(stream));
    }

    void advance(int n, int step, bool distributed, bool exchange, Vec3* hostPositions) {
        cudaCheck(cudaSetDevice(device));
        if (count == 0) return;
        if (distributed && step != 0) {
            cudaCheck(cudaMemcpyAsync(positions, hostPositions, size_t(n) * sizeof(Vec3),
                                      cudaMemcpyHostToDevice, stream));
        }
        const unsigned blocks = static_cast<unsigned>((size_t(count) + TILE_SIZE - 1) / TILE_SIZE);
        if (interactions) {
            const dim3 grid(static_cast<unsigned>((size_t(n) + TILE_SIZE - 1) / TILE_SIZE),
                            static_cast<unsigned>(count));
            computeInteractions<<<grid, TILE_SIZE, 0, stream>>>(positions, interactions, first, n);
            cudaCheck(cudaGetLastError());
            integrateInteractions<<<blocks, TILE_SIZE, 0, stream>>>(positions, interactions, velocities,
                                                                    nextPositions, first, count, n);
        } else {
            if (forceBlockSize == 32) {
                const unsigned sparseBlocks = static_cast<unsigned>((size_t(count) + 31) / 32);
                advanceBodies<32><<<sparseBlocks, 32, 0, stream>>>(positions, velocities, nextPositions,
                                                                   first, count, n);
            } else {
                advanceBodies<TILE_SIZE><<<blocks, TILE_SIZE, 0, stream>>>(positions, velocities,
                                                                          nextPositions, first, count, n);
            }
        }
        cudaCheck(cudaGetLastError());
        latestPositions = nextPositions;
        if (exchange) {
            cudaCheck(cudaMemcpyAsync(hostPositions + first, latestPositions,
                                      size_t(count) * sizeof(Vec3), cudaMemcpyDeviceToHost, stream));
            cudaCheck(cudaStreamSynchronize(stream));
        }
        if (!distributed) {
            // A single GPU retains the complete state on device across steps.
            std::swap(positions, nextPositions);
        }
    }

    void synchronize() const {
        cudaCheck(cudaSetDevice(device));
        if (count) cudaCheck(cudaStreamSynchronize(stream));
    }

    void download(int rankFirst, Vec3* hostPositions, Vec3* hostVelocities) const {
        cudaCheck(cudaSetDevice(device));
        if (count == 0) return;
        cudaCheck(cudaMemcpyAsync(hostPositions + first, latestPositions,
                                  size_t(count) * sizeof(Vec3), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaMemcpyAsync(hostVelocities + first - rankFirst, velocities,
                                  size_t(count) * sizeof(Vec3), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
    }

    void computeEnergy(int n, int rankFirst, bool distributed,
                       const Vec3* hostPositions, double* hostEnergies) const {
        cudaCheck(cudaSetDevice(device));
        if (count == 0) return;
        if (distributed) {
            cudaCheck(cudaMemcpyAsync(positions, hostPositions, size_t(n) * sizeof(Vec3),
                                      cudaMemcpyHostToDevice, stream));
        }
        const unsigned blocks = static_cast<unsigned>((size_t(count) + TILE_SIZE - 1) / TILE_SIZE);
        bodyEnergies<<<blocks, TILE_SIZE, 0, stream>>>(positions, velocities, energies, first, count, n);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpyAsync(hostEnergies + first - rankFirst, energies,
                                  size_t(count) * sizeof(double), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
    }

    void release() {
        cudaCheck(cudaSetDevice(device));
        if (count == 0) return;
        cudaCheck(cudaFree(energies));
        cudaCheck(cudaFree(interactions));
        cudaCheck(cudaFree(positions));
        cudaCheck(cudaFree(velocities));
        cudaCheck(cudaFree(nextPositions));
        cudaCheck(cudaStreamDestroy(stream));
    }
};

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks state and reports energy)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

bool parseCount(const char* argument, int& value) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(argument, &end, 10);
    if (errno || end == argument || *end || parsed < 0 || parsed > INT_MAX) return false;
    value = static_cast<int>(parsed);
    return true;
}

int runSimulation(int argc, char** argv) {
    int ranks = 1;
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-s") == 0) && i + 1 < argc) {
            int& value = strcmp(argv[i], "-n") == 0 ? numBodies : numSteps;
            if (!parseCount(argv[++i], value)) {
                if (processRank == 0) std::fprintf(stderr, "Invalid nonnegative count: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (processRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (processRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    MPI_Comm node;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, processRank,
                                 MPI_INFO_NULL, &node));
    int localRank = 0, localRanks = 1;
    mpiCheck(MPI_Comm_rank(node, &localRank));
    mpiCheck(MPI_Comm_size(node, &localRanks));
    int visibleDevices = 0;
    cudaCheck(cudaGetDeviceCount(&visibleDevices));
    if (visibleDevices == 0) {
        std::fprintf(stderr, "Rank %d: a CUDA device is required\n", processRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Use all visible GPUs when there are fewer node-local ranks than GPUs.
    // With one GPU exposed per rank by the scheduler, device zero is used.
    // Oversubscribed ranks share a GPU, but still own disjoint target bodies.
    std::vector<Accelerator> accelerators;
    for (int device = localRank; device < visibleDevices; device += localRanks) {
        Accelerator accelerator;
        accelerator.device = device;
        accelerators.push_back(accelerator);
    }
    if (accelerators.empty()) {
        Accelerator accelerator;
        accelerator.device = localRank % visibleDevices;
        accelerators.push_back(accelerator);
    }
    mpiCheck(MPI_Comm_free(&node));
    const int workers = static_cast<int>(accelerators.size());
    omp_set_dynamic(0);
    // Pinned allocations initialize a CUDA context. Keep that context on an
    // owned GPU instead of creating an unused device-zero context on every rank.
    cudaCheck(cudaSetDevice(accelerators.front().device));

    // Weight rank partitions by GPU count, so nodes with different numbers of
    // accelerators receive proportional work. Counts use a three-double MPI
    // datatype rather than multiplying int counts by three.
    std::vector<int> workerCounts(ranks), counts(ranks), offsets(ranks);
    mpiCheck(MPI_Allgather(&workers, 1, MPI_INT, workerCounts.data(), 1, MPI_INT, MPI_COMM_WORLD));
    long long totalWorkers = 0;
    for (int count : workerCounts) totalWorkers += count;
    long long prefix = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        offsets[rank] = static_cast<int>(static_cast<long long>(numBodies) * prefix / totalWorkers);
        prefix += workerCounts[rank];
        counts[rank] = static_cast<int>(static_cast<long long>(numBodies) * prefix / totalWorkers)
                     - offsets[rank];
    }
    const int first = offsets[processRank];
    const int count = counts[processRank];
    const bool distributed = totalWorkers > 1;
    for (int a = 0; a < workers; ++a) {
        accelerators[a].first = first + static_cast<int>(static_cast<long long>(count) * a / workers);
        accelerators[a].count = first + static_cast<int>(static_cast<long long>(count) * (a + 1) / workers)
                              - accelerators[a].first;
    }
    MPI_Datatype positionType;
    mpiCheck(MPI_Type_contiguous(3, MPI_DOUBLE, &positionType));
    mpiCheck(MPI_Type_commit(&positionType));
    PinnedBuffer<Vec3> positions(numBodies);
    PinnedBuffer<Vec3> velocities(count);
    std::vector<Vec3> allVelocities;
    if (processRank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);
        allVelocities.resize(numBodies);
        #pragma omp parallel for schedule(static) num_threads(workers)
        for (int i = 0; i < numBodies; ++i) {
            positions[i] = bodies[i].pos;
            allVelocities[i] = bodies[i].vel;
        }
    }
    mpiCheck(MPI_Bcast(positions.data(), numBodies, positionType, 0, MPI_COMM_WORLD));
    mpiCheck(MPI_Scatterv(allVelocities.data(), counts.data(), offsets.data(), positionType,
                          velocities.data(), count, positionType, 0, MPI_COMM_WORLD));
    if (processRank == 0 && !printResults) std::vector<Vec3>().swap(allVelocities);

    #pragma omp parallel for schedule(static) num_threads(workers)
    for (int a = 0; a < workers; ++a) {
        accelerators[a].initialize(numBodies, first, positions.data(), velocities.data(), validate);
    }
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    // Retain the worker team across time steps. MPI collectives are made only
    // by the initial thread, after all device-to-host copies have completed.
    #pragma omp parallel num_threads(workers)
    {
        for (int step = 0; step < numSteps; ++step) {
            const bool exchange = distributed && step + 1 < numSteps;
            #pragma omp for schedule(static)
            for (int a = 0; a < workers; ++a) {
                accelerators[a].advance(numBodies, step, distributed, exchange, positions.data());
            }
            #pragma omp master
            {
                if (exchange && ranks > 1) {
                    mpiCheck(MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(),
                                            counts.data(), offsets.data(), positionType, MPI_COMM_WORLD));
                }
            }
            #pragma omp barrier
        }
        #pragma omp for schedule(static)
        for (int a = 0; a < workers; ++a) accelerators[a].synchronize();
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    mpiCheck(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (processRank == 0) printf("Simulation time: %lld ms\n", static_cast<long long>(duration * 1000.0));

    if (printResults || validate) {
        #pragma omp parallel for schedule(static) num_threads(workers)
        for (int a = 0; a < workers; ++a) {
            accelerators[a].download(first, positions.data(), velocities.data());
        }
        if (validate) {
            mpiCheck(MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(),
                                    counts.data(), offsets.data(), positionType, MPI_COMM_WORLD));
        } else {
            mpiCheck(MPI_Gatherv(processRank == 0 ? MPI_IN_PLACE : positions.data() + first,
                                 count, positionType, positions.data(), counts.data(), offsets.data(),
                                 positionType, 0, MPI_COMM_WORLD));
        }
    }
    if (printResults) {
        mpiCheck(MPI_Gatherv(velocities.data(), count, positionType, allVelocities.data(),
                             counts.data(), offsets.data(), positionType, 0, MPI_COMM_WORLD));
        if (processRank == 0) {
            std::vector<double> bodyData(size_t(numBodies) * 6);
            #pragma omp parallel for schedule(static) num_threads(workers)
            for (int i = 0; i < numBodies; ++i) {
                const size_t k = size_t(i) * 6;
                bodyData[k] = positions[i].x;
                bodyData[k + 1] = positions[i].y;
                bodyData[k + 2] = positions[i].z;
                bodyData[k + 3] = allVelocities[i].x;
                bodyData[k + 4] = allVelocities[i].y;
                bodyData[k + 5] = allVelocities[i].z;
            }
            print_results(bodyData, "Bodies");
        }
    }

    int status = 0;
    if (validate) {
        if (processRank == 0) printf("Validating simulation results...\n");
        int invalid = 0;
        #pragma omp parallel for schedule(static) num_threads(workers) reduction(|:invalid)
        for (int i = 0; i < count; ++i) {
            const Vec3 p = positions[first + i], v = velocities[i];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) invalid |= 1;
            if (std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6) invalid |= 2;
            if (std::abs(v.x) > 1e6 || std::abs(v.y) > 1e6 || std::abs(v.z) > 1e6) invalid |= 4;
        }
        int globalInvalid = 0;
        mpiCheck(MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD));
        if (globalInvalid) {
            status = 1;
            if (processRank == 0) {
                if (globalInvalid & 1) printf("Validation failed: found NaN or Inf value in body state\n");
                else if (globalInvalid & 2) printf("Validation failed: body position exceeds reasonable bounds\n");
                else printf("Validation failed: body velocity exceeds reasonable bounds\n");
                printf("Validation: FAILED\n");
            }
        } else {
            PinnedBuffer<double> energies(count);
            #pragma omp parallel for schedule(static) num_threads(workers)
            for (int a = 0; a < workers; ++a) {
                accelerators[a].computeEnergy(numBodies, first, distributed, positions.data(), energies.data());
            }
            double localEnergy = 0.0;
            #pragma omp parallel for schedule(static) num_threads(workers) reduction(+:localEnergy)
            for (int i = 0; i < count; ++i) localEnergy += energies[i];
            double finalEnergy = 0.0;
            mpiCheck(MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
            if (processRank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        }
    }
    #pragma omp parallel for schedule(static) num_threads(workers)
    for (int a = 0; a < workers; ++a) accelerators[a].release();
    mpiCheck(MPI_Type_free(&positionType));
    return status;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int initError = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (initError != MPI_SUCCESS) {
        std::fprintf(stderr, "Could not initialize MPI\n");
        return 1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "Rank %d: MPI_THREAD_FUNNELED support is required\n", processRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int status = 1;
    try {
        status = runSimulation(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d: %s\n", processRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Finalize());
    return status;
}
