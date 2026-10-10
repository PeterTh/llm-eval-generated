#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

// Use device collectives when the MPI implementation can certify support.
// Other MPI implementations use pinned staging buffers, still computing on CUDA.
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int TILE = 128;
static_assert(sizeof(double3) == 3 * sizeof(double));

struct Vec3 {
    double x, y, z;
    constexpr Vec3(double x = 0, double y = 0, double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Preserve the original rand_r sequence, independently of the rank count.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void checkMPI(int error) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        throw std::runtime_error(std::string(message, length));
    }
}

void checkCUDA(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

template <typename T> class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        checkCUDA(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename T> class PinnedBuffer {
public:
    T* data = nullptr;
    explicit PinnedBuffer(size_t count) {
        checkCUDA(cudaMallocHost(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~PinnedBuffer() { cudaFreeHost(data); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

class Stream {
public:
    cudaStream_t value = nullptr;
    Stream() { checkCUDA(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};

class PositionType {
public:
    MPI_Datatype value = MPI_DATATYPE_NULL;
    PositionType() {
        checkMPI(MPI_Type_contiguous(3, MPI_DOUBLE, &value));
        checkMPI(MPI_Type_commit(&value));
    }
    ~PositionType() { MPI_Type_free(&value); }
};

// Tiles are read once per block and broadcast from shared memory to targets.
// Small rank-local partitions use several lanes per target to expose enough
// independent square roots/divisions to occupy the GPU. The owning lane still
// accumulates contributions in ascending j order, without a reordered reduction.
// Separate old/new position arrays allow force, velocity, and position updates
// to be fused without exposing updated positions to any force calculation.
template <int LANES>
__global__ void advanceBodies(const double3* __restrict__ positions,
                              double3* __restrict__ nextPositions,
                              double3* __restrict__ velocities,
                              int n, int first, int count) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];
    const int lane = threadIdx.x % LANES;
    const int local = static_cast<int>(blockIdx.x) * (TILE / LANES) + threadIdx.x / LANES;
    const bool active = local < count;
    double3 position = make_double3(0.0, 0.0, 0.0);
    if (active) position = positions[first + local];
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (size_t base = 0; base < static_cast<size_t>(n); base += TILE) {
        const size_t source = base + threadIdx.x;
        if (source < static_cast<size_t>(n)) {
            const double3 p = positions[source];
            sx[threadIdx.x] = p.x;
            sy[threadIdx.x] = p.y;
            sz[threadIdx.x] = p.z;
        }
        __syncthreads();
        const int length = static_cast<int>(min(static_cast<size_t>(TILE), static_cast<size_t>(n) - base));
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < length; j += LANES) {
                double cx = 0.0, cy = 0.0, cz = 0.0;
                if (j + lane < length) {
                    const double dx = sx[j + lane] - position.x;
                    const double dy = sy[j + lane] - position.y;
                    const double dz = sz[j + lane] - position.z;
                    const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
                    const double inverseDistance = 1.0 / sqrt(distanceSquared);
                    const double inverseCube = inverseDistance * inverseDistance * inverseDistance;
                    cx = dx * inverseCube;
                    cy = dy * inverseCube;
                    cz = dz * inverseCube;
                }
                if constexpr (LANES == 1) {
                    fx += cx;
                    fy += cy;
                    fz += cz;
                } else {
                    // Only this complete subgroup participates in its shuffles;
                    // the final warp can contain inactive target subgroups.
                    const unsigned mask = (0xffffffffu >> (32 - LANES)) <<
                                          ((threadIdx.x % 32) / LANES * LANES);
                    #pragma unroll
                    for (int k = 0; k < LANES; ++k) {
                        const double x = __shfl_sync(mask, cx, k, LANES);
                        const double y = __shfl_sync(mask, cy, k, LANES);
                        const double z = __shfl_sync(mask, cz, k, LANES);
                        if (lane == 0 && j + k < length) {
                            fx += x;
                            fy += y;
                            fz += z;
                        }
                    }
                }
            }
        }
        // Inactive threads must also reach both tile barriers.
        __syncthreads();
    }
    if (active && lane == 0) {
        double3 velocity = velocities[local];
        velocity.x += DT * fx;
        velocity.y += DT * fy;
        velocity.z += DT * fz;
        velocities[local] = velocity;
        position.x += velocity.x * DT;
        position.y += velocity.y * DT;
        position.z += velocity.z * DT;
        nextPositions[first + local] = position;
    }
}

// Potential-energy targets are assigned cyclically across ranks to balance the
// triangular i<j workload. Each pair is evaluated once, as in the original.
__global__ void totalEnergy(const double3* __restrict__ positions,
                            const double3* __restrict__ velocities,
                            double* __restrict__ blockEnergy,
                            int n, int rank, int ranks, int count) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE], partial[TILE];
    const int local = static_cast<int>(blockIdx.x) * TILE + threadIdx.x;
    const bool active = local < count;
    const size_t target = static_cast<size_t>(rank) + static_cast<size_t>(local) * ranks;
    double3 position = make_double3(0.0, 0.0, 0.0);
    double energy = 0.0;
    if (active) {
        position = positions[target];
        // Kinetic energy uses locally owned velocities; its ownership is
        // independent of the cyclic assignment of potential-energy pairs.
        const double3 v = velocities[local];
        energy = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }
    // All targets in this block can skip sources preceding its first target.
    const size_t begin = (static_cast<size_t>(rank) +
                          static_cast<size_t>(blockIdx.x) * TILE * ranks) / TILE * TILE;
    for (size_t base = begin; base < static_cast<size_t>(n); base += TILE) {
        const size_t source = base + threadIdx.x;
        if (source < static_cast<size_t>(n)) {
            const double3 p = positions[source];
            sx[threadIdx.x] = p.x;
            sy[threadIdx.x] = p.y;
            sz[threadIdx.x] = p.z;
        }
        __syncthreads();
        const int length = static_cast<int>(min(static_cast<size_t>(TILE), static_cast<size_t>(n) - base));
        if (active) {
            #pragma unroll 8
            for (int j = 0; j < length; ++j) {
                if (base + j > target) {
                    const double dx = sx[j] - position.x;
                    const double dy = sy[j] - position.y;
                    const double dz = sz[j] - position.z;
                    energy -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                }
            }
        }
        __syncthreads();
    }
    partial[threadIdx.x] = energy;
    __syncthreads();
    for (int stride = TILE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < static_cast<unsigned>(stride)) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) blockEnergy[blockIdx.x] = partial[0];
}

int validateSimulation(const double3* positions, const double3* velocities, int count) {
    int errors = 0;
    #pragma omp parallel for schedule(static) reduction(|:errors)
    for (int i = 0; i < count; ++i) {
        const double3 p = positions[i], v = velocities[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
            !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
            errors |= 1;
        }
        if (std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6) errors |= 2;
        if (std::abs(v.x) > 1e6 || std::abs(v.y) > 1e6 || std::abs(v.z) > 1e6) errors |= 4;
    }
    return errors;
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

// Body counts use an MPI datatype of three doubles, avoiding 3*n count overflow.
void exchangePositions(double3* devicePositions, double3* hostPositions,
                       int first, int count, int n, bool deviceMPI,
                       const std::vector<int>& counts, const std::vector<int>& offsets,
                       MPI_Datatype positionType, cudaStream_t stream) {
    if (deviceMPI) {
        checkCUDA(cudaStreamSynchronize(stream));
        checkMPI(MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, devicePositions,
                               counts.data(), offsets.data(), positionType, MPI_COMM_WORLD));
    } else {
        if (count != 0) {
            checkCUDA(cudaMemcpyAsync(hostPositions + first, devicePositions + first,
                                      static_cast<size_t>(count) * sizeof(double3),
                                      cudaMemcpyDeviceToHost, stream));
        }
        checkCUDA(cudaStreamSynchronize(stream));
        checkMPI(MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, hostPositions,
                               counts.data(), offsets.data(), positionType, MPI_COMM_WORLD));
        if (n != 0) {
            checkCUDA(cudaMemcpyAsync(devicePositions, hostPositions,
                                      static_cast<size_t>(n) * sizeof(double3),
                                      cudaMemcpyHostToDevice, stream));
        }
    }
}

int runSimulation(int argc, char** argv, int rank, int ranks) {
    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
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
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) fprintf(stderr, "Body and step counts must be nonnegative.\n");
        return 1;
    }

    // CUDA_VISIBLE_DEVICES may expose either all node GPUs or just this rank's
    // assigned GPU. A node-local rank works correctly with both arrangements.
    MPI_Comm nodeComm = MPI_COMM_NULL;
    checkMPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                MPI_INFO_NULL, &nodeComm));
    int localRank = 0;
    checkMPI(MPI_Comm_rank(nodeComm, &localRank));
    checkMPI(MPI_Comm_free(&nodeComm));
    int devices = 0;
    checkCUDA(cudaGetDeviceCount(&devices));
    if (devices == 0) throw std::runtime_error("The MPI/OpenMP/CUDA benchmark requires a CUDA GPU.");
    checkCUDA(cudaSetDevice(localRank % devices));
    cudaDeviceProp deviceProperties;
    checkCUDA(cudaGetDeviceProperties(&deviceProperties, localRank % devices));

    // Host work does not need a full socket's worth of threads by default.
    // Respect an explicit OMP_NUM_THREADS setting and MPI CPU affinity.
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::min(8, omp_get_max_threads()));
    }
    int supportsDeviceMPI = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    supportsDeviceMPI = MPIX_Query_cuda_support();
#endif
    int deviceMPI = 0;
    checkMPI(MPI_Allreduce(&supportsDeviceMPI, &deviceMPI, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
    }
    const int localCount = counts[rank], first = offsets[rank];
    const unsigned blocks = static_cast<unsigned>((static_cast<size_t>(localCount) + TILE - 1) / TILE);
    const int multiprocessors = deviceProperties.multiProcessorCount;
    const int lanes = localCount < multiprocessors * 4 ? 32 :
                      localCount < multiprocessors * 16 ? 8 :
                      localCount < multiprocessors * 64 ? 4 : 1;
    const unsigned forceBlocks = static_cast<unsigned>((static_cast<size_t>(localCount) * lanes + TILE - 1) / TILE);
    const size_t localBytes = static_cast<size_t>(localCount) * sizeof(double3);
    const size_t globalBytes = static_cast<size_t>(numBodies) * sizeof(double3);
    PositionType positionType;
    PinnedBuffer<double3> hostPositions(numBodies), hostVelocities(localCount);
    std::vector<double3> globalVelocities(rank == 0 ? numBodies : 0);
    if (rank == 0) {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const Body& body = bodies[i];
            hostPositions.data[i] = make_double3(body.pos.x, body.pos.y, body.pos.z);
            globalVelocities[i] = make_double3(body.vel.x, body.vel.y, body.vel.z);
        }
    }
    checkMPI(MPI_Bcast(hostPositions.data, numBodies, positionType.value, 0, MPI_COMM_WORLD));
    checkMPI(MPI_Scatterv(globalVelocities.data(), counts.data(), offsets.data(), positionType.value,
                          hostVelocities.data, localCount, positionType.value, 0, MPI_COMM_WORLD));
    if (!printResults) std::vector<double3>().swap(globalVelocities);

    DeviceBuffer<double3> positionsA(numBodies), positionsB(numBodies), velocities(localCount);
    Stream stream;
    double3* positions = positionsA.data;
    double3* nextPositions = positionsB.data;
    if (numBodies != 0) {
        checkCUDA(cudaMemcpyAsync(positions, hostPositions.data, globalBytes,
                                  cudaMemcpyHostToDevice, stream.value));
    }
    if (localCount != 0) {
        checkCUDA(cudaMemcpyAsync(velocities.data, hostVelocities.data, localBytes,
                                  cudaMemcpyHostToDevice, stream.value));
    }
    checkCUDA(cudaStreamSynchronize(stream.value));
    checkMPI(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount != 0) {
            switch (lanes) {
            case 32:
                advanceBodies<32><<<forceBlocks, TILE, 0, stream.value>>>(
                    positions, nextPositions, velocities.data, numBodies, first, localCount);
                break;
            case 8:
                advanceBodies<8><<<forceBlocks, TILE, 0, stream.value>>>(
                    positions, nextPositions, velocities.data, numBodies, first, localCount);
                break;
            case 4:
                advanceBodies<4><<<forceBlocks, TILE, 0, stream.value>>>(
                    positions, nextPositions, velocities.data, numBodies, first, localCount);
                break;
            default:
                advanceBodies<1><<<forceBlocks, TILE, 0, stream.value>>>(
                    positions, nextPositions, velocities.data, numBodies, first, localCount);
            }
            checkCUDA(cudaGetLastError());
        }
        // The final step needs no replicated positions unless energy is asked
        // for. Velocities stay local on the device for the entire simulation.
        if (ranks > 1 && numBodies != 0 && step + 1 < numSteps) {
            exchangePositions(nextPositions, hostPositions.data, first, localCount, numBodies,
                              deviceMPI != 0, counts, offsets, positionType.value, stream.value);
        }
        std::swap(positions, nextPositions);
    }
    checkCUDA(cudaStreamSynchronize(stream.value));
    const double elapsed = MPI_Wtime() - start;
    double simulationTime = 0.0;
    checkMPI(MPI_Reduce(&elapsed, &simulationTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) printf("Simulation time: %lld ms\n", static_cast<long long>(simulationTime * 1000.0));

    if (printResults || validate) {
        if (localCount != 0) {
            checkCUDA(cudaMemcpyAsync(hostPositions.data + first, positions + first, localBytes,
                                      cudaMemcpyDeviceToHost, stream.value));
            checkCUDA(cudaMemcpyAsync(hostVelocities.data, velocities.data, localBytes,
                                      cudaMemcpyDeviceToHost, stream.value));
        }
        checkCUDA(cudaStreamSynchronize(stream.value));
    }
    if (printResults) {
        checkMPI(MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : hostPositions.data + first,
                             localCount, positionType.value, hostPositions.data,
                             counts.data(), offsets.data(), positionType.value, 0, MPI_COMM_WORLD));
        checkMPI(MPI_Gatherv(hostVelocities.data, localCount, positionType.value,
                             globalVelocities.data(), counts.data(), offsets.data(),
                             positionType.value, 0, MPI_COMM_WORLD));
        if (rank == 0) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                const double3 p = hostPositions.data[i], v = globalVelocities[i];
                const size_t k = static_cast<size_t>(i) * 6;
                bodyData[k] = p.x;
                bodyData[k + 1] = p.y;
                bodyData[k + 2] = p.z;
                bodyData[k + 3] = v.x;
                bodyData[k + 4] = v.y;
                bodyData[k + 5] = v.z;
            }
            print_results(bodyData, "Bodies");
        }
    }
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        const int localErrors = validateSimulation(hostPositions.data + first,
                                                   hostVelocities.data, localCount);
        int errors = 0;
        checkMPI(MPI_Allreduce(&localErrors, &errors, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD));
        if (errors != 0) {
            if (rank == 0) {
                if (errors & 1) printf("Validation failed: found NaN or Inf value in body state\n");
                else if (errors & 2) printf("Validation failed: body position exceeds reasonable bounds\n");
                else printf("Validation failed: body velocity exceeds reasonable bounds\n");
                printf("Validation: FAILED\n");
            }
            return 1;
        }
        if (ranks > 1 && numBodies != 0 && numSteps != 0) {
            exchangePositions(positions, hostPositions.data, first, localCount, numBodies,
                              deviceMPI != 0, counts, offsets, positionType.value, stream.value);
        }
        DeviceBuffer<double> energy(blocks);
        PinnedBuffer<double> hostEnergy(blocks);
        if (localCount != 0) {
            totalEnergy<<<blocks, TILE, 0, stream.value>>>(positions, velocities.data, energy.data,
                                                        numBodies, rank, ranks, localCount);
            checkCUDA(cudaGetLastError());
            checkCUDA(cudaMemcpyAsync(hostEnergy.data, energy.data, blocks * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream.value));
        }
        checkCUDA(cudaStreamSynchronize(stream.value));
        double localEnergy = 0.0, finalEnergy = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:localEnergy)
        for (unsigned i = 0; i < blocks; ++i) localEnergy += hostEnergy.data[i];
        checkMPI(MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
        if (rank == 0) {
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    try {
        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI must provide MPI_THREAD_FUNNELED support.");
        }
        const int result = runSimulation(argc, argv, rank, ranks);
        checkMPI(MPI_Finalize());
        return result;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
