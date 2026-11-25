#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#ifdef OMPI_MAJOR_VERSION
#include <mpi-ext.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int TILE_SIZE = 128;

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
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(&data, std::max(size_t{1}, count) * sizeof(T)));
    }
    ~DeviceBuffer() { CUDA_CHECK(cudaFree(data)); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data;
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

__device__ double inverseDistance(double dx, double dy, double dz) {
    // Match the fused distance calculation used by the optimized serial build.
    const double squared = __fma_rn(dz, dz, __fma_rn(dx, dx, dy * dy)) + SOFTENING;
    // Keep sqrt and reciprocal as separately rounded FP64 operations.
    return __drcp_rn(__dsqrt_rn(squared));
}

template <int BlockSize>
__global__ void advanceBodies(const Vec3* __restrict__ positions, Vec3* __restrict__ next,
                              Vec3* __restrict__ velocities, size_t n, int first, size_t count) {
    __shared__ double tileX[BlockSize], tileY[BlockSize], tileZ[BlockSize];
    const size_t local = static_cast<size_t>(blockIdx.x) * BlockSize + threadIdx.x;
    const bool active = local < count;
    const size_t i = first + local;
    const Vec3 target = active ? positions[i] : Vec3{};
    Vec3 force{};

    for (size_t base = 0; base < n; base += BlockSize) {
        const size_t source = base + threadIdx.x;
        if (source < n) {
            const Vec3 pos = positions[source];
            tileX[threadIdx.x] = pos.x;
            tileY[threadIdx.x] = pos.y;
            tileZ[threadIdx.x] = pos.z;
        }
        if constexpr (BlockSize == 32) {
            __syncwarp();
        } else {
            __syncthreads();
        }

        if (active) {
            const int tileCount = min(static_cast<size_t>(BlockSize), n - base);
            // Visit sources in global order on every rank, including the self term.
            #pragma unroll 8
            for (int j = 0; j < tileCount; ++j) {
                const double dx = tileX[j] - target.x;
                const double dy = tileY[j] - target.y;
                const double dz = tileZ[j] - target.z;
                const double invDist = inverseDistance(dx, dy, dz);
                const double invDist3 = invDist * invDist * invDist;
                force.x = __fma_rn(dx, invDist3, force.x);
                force.y = __fma_rn(dy, invDist3, force.y);
                force.z = __fma_rn(dz, invDist3, force.z);
            }
        }
        if constexpr (BlockSize == 32) {
            __syncwarp();
        } else {
            __syncthreads();
        }
    }

    if (active) {
        Vec3 vel = velocities[local];
        vel.x += DT * force.x;
        vel.y += DT * force.y;
        vel.z += DT * force.z;
        velocities[local] = vel;
        // All force reads come from the previous positions buffer.
        next[i] = {__fma_rn(DT, vel.x, target.x),
                   __fma_rn(DT, vel.y, target.y),
                   __fma_rn(DT, vel.z, target.z)};
    }
}

__global__ void bodyEnergies(const Vec3* positions, const Vec3* velocities,
                             double* energies, size_t n, int first) {
    const size_t local = blockIdx.x;
    const size_t i = first + local;
    const Vec3 pos = positions[i];
    const Vec3 vel = velocities[local];
    // Visit half the ring from each body. For even n, only the lower index
    // contributes each diametrically opposite pair.
    const size_t partners = (n - 1) / 2 + (n % 2 == 0 && i < n / 2);
    double energy = 0.0;
    for (size_t offset = threadIdx.x + 1; offset <= partners; offset += TILE_SIZE) {
        size_t j = i + offset;
        if (j >= n) {
            j -= n;
        }
        const Vec3 other = positions[j];
        energy -= inverseDistance(other.x - pos.x, other.y - pos.y, other.z - pos.z);
    }

    __shared__ double partial[TILE_SIZE];
    partial[threadIdx.x] = energy;
    __syncthreads();
    for (int stride = TILE_SIZE / 2; stride >= 32; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        energy = partial[threadIdx.x];
        for (int stride = 16; stride > 0; stride /= 2) {
            energy += __shfl_down_sync(0xffffffff, energy, stride);
        }
        if (threadIdx.x == 0) {
            energies[local] = energy + 0.5 * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
        }
    }
}

int hostThreads(int count) {
    return std::min(omp_get_max_threads(), count / 512 + 1);
}

int runSimulation(int numBodies, int numSteps, bool validate, bool printResults, int rank, int ranks) {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);

    int devices;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        checkCuda(cudaErrorNoDevice, "Selecting a CUDA device");
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));

    int directMpi = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    if (ranks > 1) {
        directMpi = MPIX_Query_cuda_support();
        MPI_Allreduce(MPI_IN_PLACE, &directMpi, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }
#endif
    const bool stagedMpi = ranks > 1 && !directMpi;

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
    }
    const int count = counts[rank];
    const int first = offsets[rank];
    const int last = first + count;
    const size_t localBytes = static_cast<size_t>(count) * sizeof(Vec3);

    MPI_Datatype mpiVec3, mpiBody;
    MPI_Type_contiguous(3, MPI_DOUBLE, &mpiVec3);
    MPI_Type_commit(&mpiVec3);
    MPI_Type_contiguous(2, mpiVec3, &mpiBody);
    MPI_Type_commit(&mpiBody);

    DeviceBuffer<Vec3> positions(numBodies), nextPositions(numBodies), velocities(count);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    Vec3* staging;
    const size_t stagingCount = stagedMpi ? numBodies : count;
    CUDA_CHECK(cudaMallocHost(&staging, std::max(size_t{1}, stagingCount) * sizeof(Vec3)));
    Vec3* localPositions = staging + (stagedMpi ? first : 0);
    std::vector<Vec3> localVelocities(count);
    {
        // The seed is shared across bodies in the serial benchmark.
        std::vector<Body> initial(rank == 0 ? numBodies : 0), localBodies(std::max(1, count));
        if (rank == 0) {
            randomizeBodies(initial);
        }
        MPI_Scatterv(initial.data(), counts.data(), offsets.data(), mpiBody,
                     localBodies.data(), count, mpiBody, 0, MPI_COMM_WORLD);
        #pragma omp parallel for schedule(static) num_threads(hostThreads(count))
        for (int i = 0; i < count; ++i) {
            localPositions[i] = localBodies[i].pos;
            localVelocities[i] = localBodies[i].vel;
        }
    }
    MPI_Type_free(&mpiBody);

    auto allgatherPositions = [&](Vec3* buffer) {
        MPI_Allgatherv(MPI_IN_PLACE, 0, mpiVec3, buffer, counts.data(), offsets.data(),
                       mpiVec3, MPI_COMM_WORLD);
    };

    auto exchangePositions = [&](Vec3* gpuPositions, bool localOnHost = false) {
        if (stagedMpi && !localOnHost && count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(localPositions, gpuPositions + first, localBytes,
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (stagedMpi) {
            allgatherPositions(staging);
            if (count == 0) {
                return;
            }
            // Each GPU already has the slice it produced.
            if (first > 0) {
                CUDA_CHECK(cudaMemcpyAsync(gpuPositions, staging, first * sizeof(Vec3),
                                           cudaMemcpyHostToDevice, stream));
            }
            if (last < numBodies) {
                CUDA_CHECK(cudaMemcpyAsync(gpuPositions + last, staging + last,
                                           (numBodies - last) * sizeof(Vec3),
                                           cudaMemcpyHostToDevice, stream));
            }
        } else {
            allgatherPositions(gpuPositions);
        }
    };

    Vec3* current = positions.data;
    Vec3* next = nextPositions.data;
    if (count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(velocities.data, localVelocities.data(), localBytes,
                                   cudaMemcpyHostToDevice, stream));
    }
    if (numBodies > 0 && stagedMpi && numSteps > 0) {
        allgatherPositions(staging);
        if (count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(current, staging, numBodies * sizeof(Vec3),
                                       cudaMemcpyHostToDevice, stream));
        }
    } else if (numBodies > 0) {
        if (count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(current + first, localPositions, localBytes,
                                       cudaMemcpyHostToDevice, stream));
        }
        if (ranks > 1 && numSteps > 0) {
            exchangePositions(current);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps && numBodies > 0; ++step) {
        if (count > 0) {
            // Smaller blocks improve occupancy as the local slice shrinks.
            if (count <= 4096) {
                advanceBodies<32><<<(count - 1) / 32 + 1, 32, 0, stream>>>(
                    current, next, velocities.data, numBodies, first, count);
            } else {
                advanceBodies<TILE_SIZE><<<(count - 1) / TILE_SIZE + 1, TILE_SIZE, 0, stream>>>(
                    current, next, velocities.data, numBodies, first, count);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        std::swap(current, next);
        if (ranks > 1 && step + 1 < numSteps) {
            exchangePositions(current);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(duration * 1000.0));
    }

    if ((printResults || validate) && count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(localPositions, current + first, localBytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(localVelocities.data(), velocities.data, localBytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (printResults) {
        std::vector<Vec3> finalPositions(rank == 0 ? numBodies : 0);
        std::vector<Vec3> finalVelocities(rank == 0 ? numBodies : 0);
        MPI_Gatherv(localPositions, count, mpiVec3, finalPositions.data(), counts.data(),
                    offsets.data(), mpiVec3, 0, MPI_COMM_WORLD);
        MPI_Gatherv(localVelocities.data(), count, mpiVec3, finalVelocities.data(), counts.data(),
                    offsets.data(), mpiVec3, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
            #pragma omp parallel for schedule(static) num_threads(hostThreads(numBodies))
            for (int i = 0; i < numBodies; ++i) {
                const auto& pos = finalPositions[i];
                const auto& vel = finalVelocities[i];
                double* body = bodyData.data() + static_cast<size_t>(i) * 6;
                body[0] = pos.x;
                body[1] = pos.y;
                body[2] = pos.z;
                body[3] = vel.x;
                body[4] = vel.y;
                body[5] = vel.z;
            }
            print_results(bodyData, "Bodies");
        }
    }

    int result = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }
        // Pack the global body index and error together to preserve diagnostic order.
        constexpr long long valid = std::numeric_limits<long long>::max();
        long long failure = valid;
        #pragma omp parallel for reduction(min:failure) schedule(static) num_threads(hostThreads(count))
        for (int i = 0; i < count; ++i) {
            const Vec3 pos = localPositions[i];
            const Vec3 vel = localVelocities[i];
            int error = 0;
            if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
                !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) {
                error = 1;
            } else if (std::abs(pos.x) > 1e6 || std::abs(pos.y) > 1e6 || std::abs(pos.z) > 1e6) {
                error = 2;
            } else if (std::abs(vel.x) > 1e6 || std::abs(vel.y) > 1e6 || std::abs(vel.z) > 1e6) {
                error = 3;
            }
            if (error != 0) {
                failure = std::min(failure, 4LL * (first + i) + error);
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &failure, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
        if (failure == valid) {
            if (ranks > 1 && numBodies > 0) {
                exchangePositions(current, true);
            }
            double energy = 0.0;
            if (count > 0) {
                DeviceBuffer<double> gpuEnergies(count);
                std::vector<double> energies(count);
                bodyEnergies<<<count, TILE_SIZE, 0, stream>>>(current, velocities.data,
                                                             gpuEnergies.data, numBodies, first);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(energies.data(), gpuEnergies.data, count * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                #pragma omp parallel for reduction(+:energy) schedule(static) num_threads(hostThreads(count))
                for (int i = 0; i < count; ++i) {
                    energy += energies[i];
                }
            }
            double totalEnergy;
            MPI_Reduce(&energy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Final energy: %.6f\n", totalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                const char* messages[] = {
                    "",
                    "found NaN or Inf value in body state",
                    "body position exceeds reasonable bounds",
                    "body velocity exceeds reasonable bounds"
                };
                printf("Validation failed: %s\n", messages[failure % 4]);
                printf("Validation: FAILED\n");
            }
            result = 1;
        }
    }

    CUDA_CHECK(cudaFreeHost(staging));
    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Type_free(&mpiVec3);
    return result;
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

int benchmarkMain(int argc, char** argv, int rank, int ranks) {
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
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    if (numBodies < 0) {
        if (rank == 0) {
            fprintf(stderr, "Number of bodies must be nonnegative\n");
        }
        return 1;
    }
    return runSimulation(numBodies, numSteps, validate, printResults, rank, ranks);
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
            fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int status = benchmarkMain(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
