#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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
static_assert(sizeof(Vec3) == 3 * sizeof(double), "MPI position layout");

void checkMPI(int error, const char* operation) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "%s: %.*s\n", operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

void checkCUDA(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    // MPI_THREAD_FUNNELED permits MPI calls only on the main thread. A worker
    // failure terminates this rank; the MPI launcher then terminates the job.
    if (omp_in_parallel() && omp_get_thread_num() != 0) std::abort();
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}
#define MPI_CHECK(call) checkMPI((call), #call)
#define CUDA_CHECK(call) checkCUDA((call), #call)

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Preserve the original rand_r sequence, including its velocity draws.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Each thread owns one target body. Shared source tiles reduce global loads;
// the j loop remains ordered, including self interactions and softening.
// Separate position buffers keep every force dependent on the old positions.
template<int TileSize>
__global__ __launch_bounds__(TileSize)
void advanceBodies(const Vec3* __restrict__ positions,
                   Vec3* __restrict__ nextPositions,
                   Vec3* __restrict__ velocities,
                   int numBodies, int offset, int count) {
    __shared__ Vec3 tile[TileSize];
    const size_t local = size_t(blockIdx.x) * TileSize + threadIdx.x;
    const bool active = local < size_t(count);
    const size_t body = size_t(offset) + local;
    const Vec3 p = active ? positions[body] : Vec3{0.0, 0.0, 0.0};
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (size_t base = 0; base < size_t(numBodies); base += TileSize) {
        const size_t source = base + threadIdx.x;
        if (source < size_t(numBodies)) tile[threadIdx.x] = positions[source];
        __syncthreads();
        const int limit = int(size_t(numBodies) - base < TileSize
                            ? size_t(numBodies) - base : TileSize);
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < limit; ++j) {
                const double dx = tile[j].x - p.x;
                const double dy = tile[j].y - p.y;
                const double dz = tile[j].z - p.z;
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
        Vec3 v = velocities[local];
        v.x += DT * fx;
        v.y += DT * fy;
        v.z += DT * fz;
        velocities[local] = v;
        nextPositions[body] = {p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT};
    }
}

// Diagnostic energy is also distributed. Each i<j pair is counted once.
template<int TileSize>
__global__ __launch_bounds__(TileSize)
void bodyEnergies(const Vec3* __restrict__ positions,
                  const Vec3* __restrict__ velocities,
                  double* __restrict__ energies, int numBodies, int offset, int count) {
    __shared__ Vec3 tile[TileSize];
    const size_t local = size_t(blockIdx.x) * TileSize + threadIdx.x;
    const bool active = local < size_t(count);
    const size_t body = size_t(offset) + local;
    const Vec3 p = active ? positions[body] : Vec3{0.0, 0.0, 0.0};
    const Vec3 v = active ? velocities[local] : Vec3{0.0, 0.0, 0.0};
    double energy = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    const size_t first = (size_t(offset) + size_t(blockIdx.x) * TileSize) / TileSize * TileSize;
    for (size_t base = first; base < size_t(numBodies); base += TileSize) {
        const size_t source = base + threadIdx.x;
        if (source < size_t(numBodies)) tile[threadIdx.x] = positions[source];
        __syncthreads();
        const int limit = int(size_t(numBodies) - base < TileSize
                            ? size_t(numBodies) - base : TileSize);
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < limit; ++j) {
                if (base + j > body) {
                    const double dx = tile[j].x - p.x;
                    const double dy = tile[j].y - p.y;
                    const double dz = tile[j].z - p.z;
                    energy -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                }
            }
        }
        __syncthreads();
    }
    if (active) energies[local] = energy;
}

struct GPU {
    int device = 0, offset = 0, count = 0, current = 0, tileSize = 128;
    Vec3* positions[2] = {nullptr, nullptr};
    Vec3* velocities = nullptr;
    double* energies = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr;
};

template<int TileSize>
void advance(const GPU& gpu, int numBodies) {
    advanceBodies<TileSize><<<(gpu.count - 1) / TileSize + 1, TileSize, 0, gpu.stream>>>(
        gpu.positions[gpu.current], gpu.positions[1 - gpu.current], gpu.velocities,
        numBodies, gpu.offset, gpu.count);
}

void launchAdvance(const GPU& gpu, int numBodies) {
    if (gpu.tileSize == 32) advance<32>(gpu, numBodies);
    else if (gpu.tileSize == 64) advance<64>(gpu, numBodies);
    else advance<128>(gpu, numBodies);
    CUDA_CHECK(cudaGetLastError());
}

template<int TileSize>
void energy(const GPU& gpu, int numBodies) {
    bodyEnergies<TileSize><<<(gpu.count - 1) / TileSize + 1, TileSize, 0, gpu.stream>>>(
        gpu.positions[gpu.current], gpu.velocities, gpu.energies,
        numBodies, gpu.offset, gpu.count);
}

void launchEnergies(const GPU& gpu, int numBodies) {
    if (gpu.tileSize == 32) energy<32>(gpu, numBodies);
    else if (gpu.tileSize == 64) energy<64>(gpu, numBodies);
    else energy<128>(gpu, numBodies);
    CUDA_CHECK(cudaGetLastError());
}

int hostThreads(int count) {
    // Avoid starting a large CPU team for tiny packing/validation jobs.
    return std::max(1, std::min(omp_get_max_threads(), count / 8192));
}

bool parseNumber(const char* text, int& number) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < 0 || value > INT_MAX) return false;
    number = int(value);
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks finite states and reports energy)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int ranks) {
    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-s") == 0) && i + 1 < argc) {
            const bool bodiesOption = strcmp(argv[i], "-n") == 0;
            if (!parseNumber(argv[++i], bodiesOption ? numBodies : numSteps)) {
                if (rank == 0) std::fprintf(stderr, "Invalid nonnegative integer: %s\n", argv[i]);
                return 1;
            }
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
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm node;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank = 0, localRanks = 0;
    MPI_CHECK(MPI_Comm_rank(node, &localRank));
    MPI_CHECK(MPI_Comm_size(node, &localRanks));
    MPI_CHECK(MPI_Comm_free(&node));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::fprintf(stderr, "Rank %d: the hybrid solver requires a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }

    // Respect per-rank CUDA_VISIBLE_DEVICES and use all visible GPUs when fewer
    // ranks than GPUs are launched. Excess ranks share devices round-robin.
    std::vector<GPU> gpus;
    if (localRanks <= deviceCount) {
        for (int d = localRank; d < deviceCount; d += localRanks) {
            GPU gpu;
            gpu.device = d;
            gpus.push_back(gpu);
        }
    } else {
        GPU gpu;
        gpu.device = localRank % deviceCount;
        gpus.push_back(gpu);
    }
    const int workers = int(gpus.size());
    std::vector<int> gpuCounts(ranks), counts(ranks), offsets(ranks);
    MPI_CHECK(MPI_Allgather(&workers, 1, MPI_INT, gpuCounts.data(), 1, MPI_INT, MPI_COMM_WORLD));
    long long totalGPUs = 0, firstGPU = 0;
    for (int r = 0; r < ranks; ++r) {
        if (r == rank) firstGPU = totalGPUs;
        totalGPUs += gpuCounts[r];
    }
    // Weight the decomposition by GPU count, including heterogeneous nodes.
    auto boundary = [&](long long gpu) { return int((static_cast<long long>(numBodies) * gpu) / totalGPUs); };
    long long cursor = 0;
    for (int r = 0; r < ranks; ++r) {
        offsets[r] = boundary(cursor);
        cursor += gpuCounts[r];
        counts[r] = boundary(cursor) - offsets[r];
    }
    for (int w = 0; w < workers; ++w) {
        gpus[w].offset = boundary(firstGPU + w);
        gpus[w].count = boundary(firstGPU + w + 1) - gpus[w].offset;
    }
    const int localCount = counts[rank], localOffset = offsets[rank];
    MPI_Datatype positionType;
    MPI_CHECK(MPI_Type_contiguous(3, MPI_DOUBLE, &positionType));
    MPI_CHECK(MPI_Type_commit(&positionType));

    // Pinned double buffers allow asynchronous copies without overwriting a
    // buffer that another GPU is still reading during the previous upload.
    const size_t positionBytes = size_t(numBodies) * sizeof(Vec3);
    Vec3* hostPositions[2] = {nullptr, nullptr};
    Vec3* hostVelocities = nullptr;
    CUDA_CHECK(cudaSetDevice(gpus.front().device));
    CUDA_CHECK(cudaHostAlloc(&hostPositions[0], std::max(positionBytes, sizeof(Vec3)), cudaHostAllocPortable));
    if (ranks > 1) CUDA_CHECK(cudaHostAlloc(&hostPositions[1], std::max(positionBytes, sizeof(Vec3)), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&hostVelocities, size_t(std::max(1, localCount)) * sizeof(Vec3), cudaHostAllocPortable));
    std::vector<Body> bodies;
    std::vector<Vec3> allVelocities;
    if (rank == 0) {
        bodies.resize(numBodies);
        allVelocities.resize(numBodies);
        randomizeBodies(bodies);
        #pragma omp parallel for schedule(static) num_threads(hostThreads(numBodies))
        for (int i = 0; i < numBodies; ++i) {
            hostPositions[0][i] = bodies[i].pos;
            allVelocities[i] = bodies[i].vel;
        }
        std::vector<Body>().swap(bodies);
    }
    MPI_CHECK(MPI_Bcast(hostPositions[0], numBodies, positionType, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatterv(allVelocities.data(), counts.data(), offsets.data(), positionType,
                         hostVelocities, localCount, positionType, 0, MPI_COMM_WORLD));
    if (!printResults) std::vector<Vec3>().swap(allVelocities);

    // One persistent OpenMP team and one nonblocking CUDA stream per GPU.
    // MPI collectives in the simulation are executed only by the master.
    omp_set_dynamic(0);
    #pragma omp parallel for schedule(static, 1) num_threads(workers)
    for (int w = 0; w < workers; ++w) {
        GPU& gpu = gpus[w];
        CUDA_CHECK(cudaSetDevice(gpu.device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&gpu.ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaMalloc(&gpu.positions[0], std::max(positionBytes, sizeof(Vec3))));
        CUDA_CHECK(cudaMalloc(&gpu.positions[1], std::max(positionBytes, sizeof(Vec3))));
        CUDA_CHECK(cudaMalloc(&gpu.velocities, size_t(std::max(1, gpu.count)) * sizeof(Vec3)));
        if (validate) CUDA_CHECK(cudaMalloc(&gpu.energies, size_t(std::max(1, gpu.count)) * sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(gpu.positions[0], hostPositions[0], positionBytes,
                                   cudaMemcpyHostToDevice, gpu.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu.velocities, hostVelocities + gpu.offset - localOffset,
                                   size_t(gpu.count) * sizeof(Vec3), cudaMemcpyHostToDevice, gpu.stream));
        if (ranks == 1) {
            for (const auto& other : gpus) {
                if (other.device == gpu.device) continue;
                int accessible = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, gpu.device, other.device));
                if (accessible) {
                    const cudaError_t error = cudaDeviceEnablePeerAccess(other.device, 0);
                    if (error == cudaErrorPeerAccessAlreadyEnabled || error == cudaErrorTooManyPeers)
                        (void)cudaGetLastError();
                    else CUDA_CHECK(error);
                }
            }
        }
        cudaDeviceProp properties;
        CUDA_CHECK(cudaGetDeviceProperties(&properties, gpu.device));
        // Small partitions need more blocks to occupy all SMs. Larger tiles
        // improve source reuse once there are enough targets for occupancy.
        gpu.tileSize = gpu.count < properties.multiProcessorCount * 64 * 4 ? 32
                     : gpu.count < properties.multiProcessorCount * 128 * 4 ? 64 : 128;
        cudaFuncAttributes attributes;
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, advanceBodies<32>));
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, advanceBodies<64>));
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, advanceBodies<128>));
        CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    #pragma omp parallel num_threads(workers)
    {
        // Worksharing also handles an OpenMP runtime limiting the team to
        // fewer threads than GPUs; every GPU still receives its work.
        for (int step = 0; step < numSteps; ++step) {
            #pragma omp for schedule(static, 1)
            for (int w = 0; w < workers; ++w) {
                GPU& gpu = gpus[w];
                CUDA_CHECK(cudaSetDevice(gpu.device));
                if (gpu.count > 0) launchAdvance(gpu, numBodies);
                if (step + 1 < numSteps && (ranks > 1 || workers > 1)) {
                    if (ranks == 1) {
                        CUDA_CHECK(cudaEventRecord(gpu.ready, gpu.stream));
                    } else {
                        Vec3* exchange = hostPositions[(step + 1) & 1];
                        CUDA_CHECK(cudaMemcpyAsync(exchange + gpu.offset,
                            gpu.positions[1 - gpu.current] + gpu.offset,
                            size_t(gpu.count) * sizeof(Vec3), cudaMemcpyDeviceToHost, gpu.stream));
                        CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
                    }
                }
            }
            if (step + 1 < numSteps && (ranks > 1 || workers > 1)) {
                if (ranks > 1) {
                    #pragma omp master
                    {
                        MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, positionType,
                            hostPositions[(step + 1) & 1], counts.data(), offsets.data(),
                            positionType, MPI_COMM_WORLD));
                    }
                    #pragma omp barrier
                }
                #pragma omp for schedule(static, 1)
                for (int w = 0; w < workers; ++w) {
                    GPU& gpu = gpus[w];
                    CUDA_CHECK(cudaSetDevice(gpu.device));
                    if (ranks == 1) {
                        // CUDA stages inaccessible peer links itself.
                        for (int sourceIndex = 0; sourceIndex < workers; ++sourceIndex) {
                            const GPU& source = gpus[sourceIndex];
                            if (sourceIndex == w || source.count == 0) continue;
                            CUDA_CHECK(cudaStreamWaitEvent(gpu.stream, source.ready, 0));
                            CUDA_CHECK(cudaMemcpyPeerAsync(
                                gpu.positions[1 - gpu.current] + source.offset, gpu.device,
                                source.positions[1 - source.current] + source.offset, source.device,
                                size_t(source.count) * sizeof(Vec3), gpu.stream));
                        }
                        // All remote reads finish before buffers are reused.
                        CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
                    } else {
                        CUDA_CHECK(cudaMemcpyAsync(gpu.positions[1 - gpu.current],
                            hostPositions[(step + 1) & 1], positionBytes,
                            cudaMemcpyHostToDevice, gpu.stream));
                    }
                }
            }
            #pragma omp for schedule(static, 1)
            for (int w = 0; w < workers; ++w) gpus[w].current = 1 - gpus[w].current;
        }
        #pragma omp for schedule(static, 1)
        for (int w = 0; w < workers; ++w) {
            CUDA_CHECK(cudaSetDevice(gpus[w].device));
            CUDA_CHECK(cudaStreamSynchronize(gpus[w].stream));
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double maximumElapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) printf("Simulation time: %lld ms\n", static_cast<long long>(maximumElapsed * 1000.0));

    // Velocities stay on their owner GPU throughout the simulation; only
    // requested output/validation transfers them and the final positions.
    if (printResults || validate) {
        #pragma omp parallel for schedule(static, 1) num_threads(workers)
        for (int w = 0; w < workers; ++w) {
            const GPU& gpu = gpus[w];
            CUDA_CHECK(cudaSetDevice(gpu.device));
            CUDA_CHECK(cudaMemcpyAsync(hostPositions[0] + gpu.offset, gpu.positions[gpu.current] + gpu.offset,
                size_t(gpu.count) * sizeof(Vec3), cudaMemcpyDeviceToHost, gpu.stream));
            CUDA_CHECK(cudaMemcpyAsync(hostVelocities + gpu.offset - localOffset, gpu.velocities,
                size_t(gpu.count) * sizeof(Vec3), cudaMemcpyDeviceToHost, gpu.stream));
            CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
        }
        MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, hostPositions[0],
            counts.data(), offsets.data(), positionType, MPI_COMM_WORLD));
    }
    if (printResults) {
        MPI_CHECK(MPI_Gatherv(hostVelocities, localCount, positionType, allVelocities.data(),
            counts.data(), offsets.data(), positionType, 0, MPI_COMM_WORLD));
        if (rank == 0) {
            std::vector<double> bodyData(size_t(numBodies) * 6);
            #pragma omp parallel for schedule(static) num_threads(hostThreads(numBodies))
            for (int i = 0; i < numBodies; ++i) {
                const Vec3 p = hostPositions[0][i], v = allVelocities[i];
                const size_t index = size_t(i) * 6;
                bodyData[index] = p.x;
                bodyData[index + 1] = p.y;
                bodyData[index + 2] = p.z;
                bodyData[index + 3] = v.x;
                bodyData[index + 4] = v.y;
                bodyData[index + 5] = v.z;
            }
            print_results(bodyData, "Bodies");
        }
    }

    int result = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        // Preserve the original first failing body's diagnostic.
        unsigned long long failure = std::numeric_limits<unsigned long long>::max();
        #pragma omp parallel for schedule(static) num_threads(hostThreads(localCount)) reduction(min:failure)
        for (int i = 0; i < localCount; ++i) {
            const Vec3 p = hostPositions[0][localOffset + i], v = hostVelocities[i];
            int reason = -1;
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) reason = 0;
            else if (std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6) reason = 1;
            else if (std::abs(v.x) > 1e6 || std::abs(v.y) > 1e6 || std::abs(v.z) > 1e6) reason = 2;
            if (reason >= 0) failure = std::min(failure, 3ULL * (localOffset + i) + reason);
        }
        unsigned long long globalFailure = 0;
        MPI_CHECK(MPI_Allreduce(&failure, &globalFailure, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD));
        if (globalFailure != std::numeric_limits<unsigned long long>::max()) {
            result = 1;
            if (rank == 0) {
                const char* reasons[] = {"found NaN or Inf value in body state",
                    "body position exceeds reasonable bounds", "body velocity exceeds reasonable bounds"};
                printf("Validation failed: %s\n", reasons[globalFailure % 3]);
                printf("Validation: FAILED\n");
            }
        } else {
            std::vector<double> energies(localCount);
            #pragma omp parallel for schedule(static, 1) num_threads(workers)
            for (int w = 0; w < workers; ++w) {
                const GPU& gpu = gpus[w];
                CUDA_CHECK(cudaSetDevice(gpu.device));
                CUDA_CHECK(cudaMemcpyAsync(gpu.positions[gpu.current], hostPositions[0], positionBytes,
                    cudaMemcpyHostToDevice, gpu.stream));
                if (gpu.count > 0) {
                    launchEnergies(gpu, numBodies);
                    CUDA_CHECK(cudaMemcpyAsync(energies.data() + gpu.offset - localOffset, gpu.energies,
                        size_t(gpu.count) * sizeof(double), cudaMemcpyDeviceToHost, gpu.stream));
                }
                CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
            }
            double localEnergy = 0.0, finalEnergy = 0.0;
            #pragma omp parallel for schedule(static) num_threads(hostThreads(localCount)) reduction(+:localEnergy)
            for (int i = 0; i < localCount; ++i) localEnergy += energies[i];
            MPI_CHECK(MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        }
    }

    #pragma omp parallel for schedule(static, 1) num_threads(workers)
    for (int w = 0; w < workers; ++w) {
        const GPU& gpu = gpus[w];
        CUDA_CHECK(cudaSetDevice(gpu.device));
        CUDA_CHECK(cudaFree(gpu.positions[0]));
        CUDA_CHECK(cudaFree(gpu.positions[1]));
        CUDA_CHECK(cudaFree(gpu.velocities));
        CUDA_CHECK(cudaFree(gpu.energies));
        CUDA_CHECK(cudaEventDestroy(gpu.ready));
        CUDA_CHECK(cudaStreamDestroy(gpu.stream));
    }
    CUDA_CHECK(cudaFreeHost(hostPositions[0]));
    if (hostPositions[1]) CUDA_CHECK(cudaFreeHost(hostPositions[1]));
    CUDA_CHECK(cudaFreeHost(hostVelocities));
    MPI_CHECK(MPI_Type_free(&positionType));
    return result;
}

int main(int argc, char** argv) {
    int provided = 0;
    const int error = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (error != MPI_SUCCESS) return 1;
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank = 0, ranks = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, exception.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Finalize());
    return result;
}
