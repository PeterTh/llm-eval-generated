#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#ifdef NBODY_MPI_CUDA_QUERY
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int TILE = 128;

struct Vec3 { double x, y, z; };
static_assert(std::is_standard_layout<Vec3>::value && sizeof(Vec3) == 3 * sizeof(double),
              "MPI positions must contain exactly three doubles");
struct Diagnostic { double energy; unsigned long long failure; };

static void checkMPI(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        std::fprintf(stderr, "%s: %.*s\n", operation, length, message);
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}
static void checkCUDA(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        // Let the MPI launcher terminate the job; no MPI calls from workers.
        std::abort();
    }
}
#define MPI_CHECK(call) checkMPI((call), #call)
#define CUDA_CHECK(call) checkCUDA((call), #call)
static int blocksFor(int count) { return count / TILE + (count % TILE != 0); }

// Each thread owns one target. All sources come from the same snapshot;
// integration writes a different buffer. Shared-memory tiling reuses sources
// across the block without changing the original order of force accumulation.
__global__ void advanceBodies(const Vec3* __restrict__ positions,
                              Vec3* __restrict__ nextPositions,
                              Vec3* __restrict__ velocities,
                              int n, int first, int count) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];
    const unsigned int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < static_cast<unsigned int>(count);
    const unsigned int i = first + local;
    Vec3 p = {0.0, 0.0, 0.0};
    if (active) p = positions[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (long long base = 0; base < n; base += TILE) {
        const long long source = base + threadIdx.x;
        if (source < n) {
            const Vec3 q = positions[source];
            sx[threadIdx.x] = q.x;
            sy[threadIdx.x] = q.y;
            sz[threadIdx.x] = q.z;
        }
        __syncthreads();
        const int limit = static_cast<int>(min(static_cast<long long>(TILE), n - base));
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < limit; ++j) {
                const double dx = sx[j] - p.x;
                const double dy = sy[j] - p.y;
                const double dz = sz[j] - p.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        // Inactive threads must participate in both barriers of a partial tile.
        __syncthreads();
    }
    if (active) {
        Vec3 v = velocities[local];
        v.x += DT * fx;
        v.y += DT * fy;
        v.z += DT * fz;
        velocities[local] = v;
        nextPositions[i] = {p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT};
    }
}

// For small target partitions, one warp per body exposes enough parallel work
// to fill the GPU. Lanes evaluate consecutive interactions in parallel, then
// shuffle their contributions in ascending j order: there is no reordered sum.
__global__ void advanceWarpBodies(const Vec3* __restrict__ positions,
                                  Vec3* __restrict__ nextPositions,
                                  Vec3* __restrict__ velocities,
                                  int n, int first, int count) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const unsigned int local = blockIdx.x * (TILE / 32) + tid / 32;
    const unsigned int i = first + local;
    const bool active = local < static_cast<unsigned int>(count);
    Vec3 p = {0.0, 0.0, 0.0};
    if (active) p = positions[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (long long base = 0; base < n; base += TILE) {
        if (base + tid < n) {
            const Vec3 q = positions[base + tid];
            sx[tid] = q.x;
            sy[tid] = q.y;
            sz[tid] = q.z;
        }
        __syncthreads();
        const int limit = static_cast<int>(min(static_cast<long long>(TILE), n - base));
        if (active) {
            for (int group = 0; group < limit; group += 32) {
                double cx = 0.0, cy = 0.0, cz = 0.0;
                if (group + lane < limit) {
                    const double dx = sx[group + lane] - p.x;
                    const double dy = sy[group + lane] - p.y;
                    const double dz = sz[group + lane] - p.z;
                    const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    const double invDist = 1.0 / sqrt(distSqr);
                    const double invDist3 = invDist * invDist * invDist;
                    cx = dx * invDist3;
                    cy = dy * invDist3;
                    cz = dz * invDist3;
                }
                const int lanes = min(32, limit - group);
                #pragma unroll 4
                for (int j = 0; j < lanes; ++j) {
                    fx += __shfl_sync(0xffffffffu, cx, j);
                    fy += __shfl_sync(0xffffffffu, cy, j);
                    fz += __shfl_sync(0xffffffffu, cz, j);
                }
            }
        }
        __syncthreads();
    }
    if (active && lane == 0) {
        Vec3 v = velocities[local];
        v.x += DT * fx;
        v.y += DT * fy;
        v.z += DT * fz;
        velocities[local] = v;
        nextPositions[i] = {p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT};
    }
}

// Diagnostics remain distributed on the GPUs. Every potential-energy pair is
// counted once. The minimum failure key identifies the first invalid body.
__global__ void diagnoseBodies(const Vec3* __restrict__ positions,
                               const Vec3* __restrict__ velocities,
                               Diagnostic* __restrict__ partials,
                               int n, int first, int count) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE], energies[TILE];
    __shared__ unsigned long long failures[TILE];
    const int tid = threadIdx.x;
    const unsigned int local = blockIdx.x * blockDim.x + tid;
    const unsigned int i = first + local;
    const bool active = local < static_cast<unsigned int>(count);
    Vec3 p = {0.0, 0.0, 0.0};
    double energy = 0.0;
    unsigned long long failure = ~0ULL;
    if (active) {
        p = positions[i];
        const Vec3 v = velocities[local];
        energy = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
        unsigned int reason = 0;
        if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z) ||
            !isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z)) reason = 1;
        else if (fabs(p.x) > 1e6 || fabs(p.y) > 1e6 || fabs(p.z) > 1e6) reason = 2;
        else if (fabs(v.x) > 1e6 || fabs(v.y) > 1e6 || fabs(v.z) > 1e6) reason = 3;
        if (reason) failure = (static_cast<unsigned long long>(i) << 2) | reason;
    }
    // Skip sources preceding the earliest target in this block.
    for (long long base = (static_cast<long long>(first) + blockIdx.x * blockDim.x) / TILE * TILE;
         base < n; base += TILE) {
        if (base + tid < n) {
            const Vec3 q = positions[base + tid];
            sx[tid] = q.x;
            sy[tid] = q.y;
            sz[tid] = q.z;
        }
        __syncthreads();
        if (active) {
            const int begin = static_cast<int>(max(0LL, static_cast<long long>(i) + 1 - base));
            const int limit = static_cast<int>(min(static_cast<long long>(TILE), n - base));
            #pragma unroll 4
            for (int j = begin; j < limit; ++j) {
                const double dx = sx[j] - p.x;
                const double dy = sy[j] - p.y;
                const double dz = sz[j] - p.z;
                energy -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            }
        }
        __syncthreads();
    }
    energies[tid] = energy;
    failures[tid] = failure;
    __syncthreads();
    for (int stride = TILE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            energies[tid] += energies[tid + stride];
            failures[tid] = min(failures[tid], failures[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = {energies[0], failures[0]};
}

struct GPUWorker {
    int device = 0, first = 0, count = 0;
    bool warpTargets = false;
    Vec3 *positions = nullptr, *nextPositions = nullptr, *velocities = nullptr;
    Diagnostic* partials = nullptr;
    cudaStream_t stream = nullptr;
    double energy = 0.0;
    unsigned long long failure = ~0ULL;
};

static void launchAdvance(const GPUWorker& w, int n, int count) {
    if (w.warpTargets) {
        const int targetsPerBlock = TILE / 32;
        const int blocks = count / targetsPerBlock + (count % targetsPerBlock != 0);
        advanceWarpBodies<<<std::max(blocks, 1), TILE, 0, w.stream>>>(
            w.positions, w.nextPositions, w.velocities, n, w.first, count);
    } else {
        advanceBodies<<<std::max(blocksFor(count), 1), TILE, 0, w.stream>>>(
            w.positions, w.nextPositions, w.velocities, n, w.first, count);
    }
    CUDA_CHECK(cudaGetLastError());
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

static int simulate(int argc, char** argv) {
    int rank = 0, ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
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
        if (rank == 0) std::fprintf(stderr, "Number of bodies must be nonnegative\n");
        return 1;
    }
    if (rank == 0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\n",
                    numBodies, numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0, localRanks = 1, deviceCount = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &localRanks));
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) {
        std::fprintf(stderr, "MPI rank %d requires a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Use every GPU assigned to this rank. Supports one rank per node, one rank
    // per GPU, and scheduler-provided CUDA_VISIBLE_DEVICES masks.
    std::vector<GPUWorker> workers;
    if (deviceCount >= localRanks) {
        for (int device = localRank; device < deviceCount; device += localRanks) {
            GPUWorker worker;
            worker.device = device;
            workers.push_back(worker);
        }
    } else {
        GPUWorker worker;
        worker.device = localRank % deviceCount;
        workers.push_back(worker);
    }
    MPI_CHECK(MPI_Comm_free(&nodeComm));
    const int workerCount = static_cast<int>(workers.size());
    omp_set_dynamic(0);

    // Weight the target partition by accelerator count, including uneven nodes.
    std::vector<int> gpuCounts(ranks), counts(ranks), offsets(ranks);
    MPI_CHECK(MPI_Allgather(&workerCount, 1, MPI_INT, gpuCounts.data(), 1, MPI_INT, MPI_COMM_WORLD));
    long long totalGPUs = 0, precedingGPUs = 0, myPrecedingGPUs = 0;
    for (int count : gpuCounts) totalGPUs += count;
    for (int r = 0; r < ranks; ++r) {
        offsets[r] = static_cast<int>(numBodies * precedingGPUs / totalGPUs);
        if (r == rank) myPrecedingGPUs = precedingGPUs;
        precedingGPUs += gpuCounts[r];
        counts[r] = static_cast<int>(numBodies * precedingGPUs / totalGPUs) - offsets[r];
    }
    for (int g = 0; g < workerCount; ++g) {
        workers[g].first = static_cast<int>(numBodies * (myPrecedingGPUs + g) / totalGPUs);
        workers[g].count = static_cast<int>(numBodies * (myPrecedingGPUs + g + 1) / totalGPUs)
                           - workers[g].first;
    }

    MPI_Datatype positionType;
    MPI_CHECK(MPI_Type_contiguous(3, MPI_DOUBLE, &positionType));
    MPI_CHECK(MPI_Type_commit(&positionType));
    const size_t positionBytes = static_cast<size_t>(numBodies) * sizeof(Vec3);
    Vec3 *hostPositions = nullptr, *hostVelocities = nullptr;
    CUDA_CHECK(cudaSetDevice(workers[0].device));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostPositions),
                            std::max(positionBytes, sizeof(Vec3)), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostVelocities),
                            static_cast<size_t>(std::max(counts[rank], 1)) * sizeof(Vec3),
                            cudaHostAllocPortable));
    // Preserve the original rand_r stream and six draws per body. Generate once
    // on root and distribute only the velocities owned by each rank.
    std::vector<Vec3> rootVelocities(rank == 0 ? numBodies : 0);
    if (rank == 0) {
        unsigned int seed = 42;
        for (int i = 0; i < numBodies; ++i) {
            hostPositions[i].x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
            hostPositions[i].y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
            hostPositions[i].z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
            rootVelocities[i].x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
            rootVelocities[i].y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
            rootVelocities[i].z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        }
    }
    MPI_CHECK(MPI_Bcast(hostPositions, numBodies, positionType, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatterv(rootVelocities.data(), counts.data(), offsets.data(), positionType,
                           hostVelocities, counts[rank], positionType, 0, MPI_COMM_WORLD));
    if (!printResults) std::vector<Vec3>().swap(rootVelocities);

    int directCommunication = 1;
    #pragma omp parallel for num_threads(workerCount) schedule(static) reduction(min:directCommunication)
    for (int g = 0; g < workerCount; ++g) {
        GPUWorker& w = workers[g];
        CUDA_CHECK(cudaSetDevice(w.device));
        cudaDeviceProp properties;
        CUDA_CHECK(cudaGetDeviceProperties(&properties, w.device));
        w.warpTargets = w.count <= properties.multiProcessorCount * 32;
        CUDA_CHECK(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&w.positions), std::max(positionBytes, sizeof(Vec3))));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&w.nextPositions), std::max(positionBytes, sizeof(Vec3))));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&w.velocities),
                              static_cast<size_t>(std::max(w.count, 1)) * sizeof(Vec3)));
        if (validate) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&w.partials),
                                  static_cast<size_t>(std::max(blocksFor(w.count), 1)) * sizeof(Diagnostic)));
        }
        for (int other = 0; other < workerCount; ++other) {
            if (other == g) continue;
            int accessible = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, w.device, workers[other].device));
            if (accessible) {
                cudaError_t status = cudaDeviceEnablePeerAccess(workers[other].device, 0);
                if (status == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                else CUDA_CHECK(status);
            } else directCommunication = 0;
        }
        if (numBodies) CUDA_CHECK(cudaMemcpyAsync(w.positions, hostPositions, positionBytes,
                                                  cudaMemcpyHostToDevice, w.stream));
        if (w.count) {
            CUDA_CHECK(cudaMemcpyAsync(w.velocities, hostVelocities + w.first - offsets[rank],
                                       static_cast<size_t>(w.count) * sizeof(Vec3), cudaMemcpyHostToDevice, w.stream));
        }
        // Load the force kernel before timing, including with lazy CUDA module
        // loading. This zero-body launch does not modify the simulation state.
        launchAdvance(w, 0, 0);
        CUDA_CHECK(cudaStreamSynchronize(w.stream));
    }
    if (ranks > 1) {
        int cudaAwareMPI = 0;
        #ifdef NBODY_MPI_CUDA_QUERY
        cudaAwareMPI = MPIX_Query_cuda_support() != 0;
        #elif defined(NBODY_MPI_GPU_QUERY)
        MPI_CHECK(MPIX_GPU_query_support(MPIX_GPU_SUPPORT_CUDA, &cudaAwareMPI));
        #endif
        directCommunication = directCommunication && cudaAwareMPI;
    }
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &directCommunication, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    // Keep the OpenMP team alive across timesteps. MPI calls stay on the master
    // thread, while CUDA work is ordered independently in each GPU's stream.
    #pragma omp parallel num_threads(workerCount)
    {
        for (int step = 0; step < numSteps; ++step) {
            const bool exchange = totalGPUs > 1 && (step + 1 < numSteps || validate);
            #pragma omp for schedule(static)
            for (int g = 0; g < workerCount; ++g) {
                GPUWorker& w = workers[g];
                CUDA_CHECK(cudaSetDevice(w.device));
                if (w.count) {
                    launchAdvance(w, numBodies, w.count);
                    if (exchange && !directCommunication) {
                        CUDA_CHECK(cudaMemcpyAsync(hostPositions + w.first, w.nextPositions + w.first,
                                                   static_cast<size_t>(w.count) * sizeof(Vec3),
                                                   cudaMemcpyDeviceToHost, w.stream));
                    }
                }
                // One accelerator can queue all steps without host round trips.
                if (totalGPUs > 1) CUDA_CHECK(cudaStreamSynchronize(w.stream));
            }
            if (exchange) {
                if (directCommunication && ranks == 1) {
                    #pragma omp for schedule(static)
                    for (int g = 0; g < workerCount; ++g) {
                        GPUWorker& w = workers[g];
                        CUDA_CHECK(cudaSetDevice(w.device));
                        for (int other = 0; other < workerCount; ++other) {
                            const GPUWorker& source = workers[other];
                            if (other != g && source.count) {
                                CUDA_CHECK(cudaMemcpyPeerAsync(w.nextPositions + source.first, w.device,
                                            source.nextPositions + source.first, source.device,
                                            static_cast<size_t>(source.count) * sizeof(Vec3), w.stream));
                            }
                        }
                        CUDA_CHECK(cudaStreamSynchronize(w.stream));
                    }
                } else {
                    #pragma omp master
                    {
                        if (directCommunication) {
                            GPUWorker& primary = workers[0];
                            CUDA_CHECK(cudaSetDevice(primary.device));
                            for (int g = 1; g < workerCount; ++g) {
                                const GPUWorker& source = workers[g];
                                if (source.count) {
                                    CUDA_CHECK(cudaMemcpyPeerAsync(primary.nextPositions + source.first, primary.device,
                                                source.nextPositions + source.first, source.device,
                                                static_cast<size_t>(source.count) * sizeof(Vec3), primary.stream));
                                }
                            }
                            CUDA_CHECK(cudaStreamSynchronize(primary.stream));
                            MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, primary.nextPositions,
                                        counts.data(), offsets.data(), positionType, MPI_COMM_WORLD));
                        } else if (ranks > 1) {
                            MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, hostPositions,
                                        counts.data(), offsets.data(), positionType, MPI_COMM_WORLD));
                        }
                    }
                    #pragma omp barrier
                    #pragma omp for schedule(static)
                    for (int g = 0; g < workerCount; ++g) {
                        GPUWorker& w = workers[g];
                        CUDA_CHECK(cudaSetDevice(w.device));
                        if (numBodies) {
                            if (directCommunication && g != 0) {
                                CUDA_CHECK(cudaMemcpyPeerAsync(w.nextPositions, w.device,
                                            workers[0].nextPositions, workers[0].device, positionBytes, w.stream));
                            } else if (!directCommunication) {
                                CUDA_CHECK(cudaMemcpyAsync(w.nextPositions, hostPositions, positionBytes,
                                                           cudaMemcpyHostToDevice, w.stream));
                            }
                        }
                        // Complete staging-buffer reads before a worker can
                        // overwrite the shared buffer during the next step.
                        CUDA_CHECK(cudaStreamSynchronize(w.stream));
                    }
                }
            }
            #pragma omp for schedule(static)
            for (int g = 0; g < workerCount; ++g) std::swap(workers[g].positions, workers[g].nextPositions);
        }
        #pragma omp for schedule(static)
        for (int g = 0; g < workerCount; ++g) {
            CUDA_CHECK(cudaSetDevice(workers[g].device));
            CUDA_CHECK(cudaStreamSynchronize(workers[g].stream));
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double slowest = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) std::printf("Simulation time: %lld ms\n", static_cast<long long>(slowest * 1000.0));

    if (printResults) {
        #pragma omp parallel for num_threads(workerCount) schedule(static)
        for (int g = 0; g < workerCount; ++g) {
            const GPUWorker& w = workers[g];
            CUDA_CHECK(cudaSetDevice(w.device));
            if (w.count) {
                CUDA_CHECK(cudaMemcpyAsync(hostPositions + w.first, w.positions + w.first,
                                           static_cast<size_t>(w.count) * sizeof(Vec3), cudaMemcpyDeviceToHost, w.stream));
                CUDA_CHECK(cudaMemcpyAsync(hostVelocities + w.first - offsets[rank], w.velocities,
                                           static_cast<size_t>(w.count) * sizeof(Vec3), cudaMemcpyDeviceToHost, w.stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(w.stream));
        }
        MPI_CHECK(MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : hostPositions + offsets[rank],
                              counts[rank], positionType, hostPositions, counts.data(), offsets.data(),
                              positionType, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Gatherv(hostVelocities, counts[rank], positionType, rootVelocities.data(),
                              counts.data(), offsets.data(), positionType, 0, MPI_COMM_WORLD));
        if (rank == 0) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
            const int hostThreads = std::min(omp_get_max_threads(), std::max(2, workerCount));
            #pragma omp parallel for num_threads(hostThreads) schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                const size_t k = static_cast<size_t>(i) * 6;
                bodyData[k] = hostPositions[i].x;
                bodyData[k + 1] = hostPositions[i].y;
                bodyData[k + 2] = hostPositions[i].z;
                bodyData[k + 3] = rootVelocities[i].x;
                bodyData[k + 4] = rootVelocities[i].y;
                bodyData[k + 5] = rootVelocities[i].z;
            }
            print_results(bodyData, "Bodies");
        }
    }

    int result = 0;
    if (validate) {
        #pragma omp parallel for num_threads(workerCount) schedule(static)
        for (int g = 0; g < workerCount; ++g) {
            GPUWorker& w = workers[g];
            CUDA_CHECK(cudaSetDevice(w.device));
            if (!w.count) continue;
            const int blocks = blocksFor(w.count);
            diagnoseBodies<<<blocks, TILE, 0, w.stream>>>(w.positions, w.velocities, w.partials,
                                                         numBodies, w.first, w.count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(w.stream));
            std::vector<Diagnostic> partials(blocks);
            CUDA_CHECK(cudaMemcpy(partials.data(), w.partials, partials.size() * sizeof(Diagnostic),
                                  cudaMemcpyDeviceToHost));
            for (const Diagnostic& partial : partials) {
                w.energy += partial.energy;
                w.failure = std::min(w.failure, partial.failure);
            }
        }
        double localEnergy = 0.0, finalEnergy = 0.0;
        unsigned long long failure = ~0ULL;
        for (const GPUWorker& w : workers) {
            localEnergy += w.energy;
            failure = std::min(failure, w.failure);
        }
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &failure, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
        result = failure != ~0ULL;
        if (rank == 0) {
            std::printf("Validating simulation results...\n");
            if (!result) std::printf("Final energy: %.6f\nValidation: PASSED\n", finalEnergy);
            else {
                const char* messages[] = {"", "found NaN or Inf value in body state",
                                           "body position exceeds reasonable bounds",
                                           "body velocity exceeds reasonable bounds"};
                std::printf("Validation failed: %s\nValidation: FAILED\n", messages[failure & 3]);
            }
        }
    }

    #pragma omp parallel for num_threads(workerCount) schedule(static)
    for (int g = 0; g < workerCount; ++g) {
        GPUWorker& w = workers[g];
        CUDA_CHECK(cudaSetDevice(w.device));
        CUDA_CHECK(cudaFree(w.positions));
        CUDA_CHECK(cudaFree(w.nextPositions));
        CUDA_CHECK(cudaFree(w.velocities));
        if (w.partials) CUDA_CHECK(cudaFree(w.partials));
        CUDA_CHECK(cudaStreamDestroy(w.stream));
    }
    CUDA_CHECK(cudaFreeHost(hostPositions));
    CUDA_CHECK(cudaFreeHost(hostVelocities));
    MPI_CHECK(MPI_Type_free(&positionType));
    return result;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI initialization failed\n");
        return 1;
    }
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI must support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int result = 1;
    try { result = simulate(argc, argv); }
    catch (...) {
        std::fprintf(stderr, "Unable to allocate simulation state\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Finalize());
    return result;
}
