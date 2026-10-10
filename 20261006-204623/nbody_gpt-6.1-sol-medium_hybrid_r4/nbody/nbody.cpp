#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int TILE = 128;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos, vel; };
static_assert(sizeof(Vec3) == 3 * sizeof(double) &&
              std::is_trivially_copyable<Vec3>::value);

// CUDA errors are fatal on every rank, including inside OpenMP workers. Abort
// the process rather than making MPI calls outside the MPI main thread.
void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::abort();
    }
}

// All targets in a block reuse the same source tile. Even inactive threads
// participate in both barriers, including the last partial block and tile.
// Sources are traversed in the original order and all arithmetic is double.
template<int BLOCK>
__global__ void advance(const Vec3* __restrict__ positions,
                        Vec3* __restrict__ velocities,
                        Vec3* __restrict__ next, int n, int begin, int count) {
    __shared__ Vec3 sources[BLOCK];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < count;
    Vec3 p = {0.0, 0.0, 0.0};
    if (active) p = positions[begin + local];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += BLOCK) {
        const int j = base + threadIdx.x;
        if (j < n) sources[threadIdx.x] = positions[j];
        __syncthreads();
        if (active) {
            const int size = min(BLOCK, n - base);
            #pragma unroll 4
            for (int k = 0; k < size; ++k) {
                const double dx = sources[k].x - p.x;
                const double dy = sources[k].y - p.y;
                const double dz = sources[k].z - p.z;
                const double d2 = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inv = 1.0 / sqrt(d2);
                const double inv3 = inv * inv * inv;
                fx += dx * inv3;
                fy += dy * inv3;
                fz += dz * inv3;
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
        // Positions are written separately, preserving the force snapshot.
        next[local] = {p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT};
    }
}

// Validation's quadratic potential-energy work also runs on the GPUs. Each
// pair is counted once; the kinetic term belongs to the target's owner.
__global__ void energyKernel(const Vec3* positions, const Vec3* velocities,
                             double* energies, int n, int begin, int count) {
    __shared__ Vec3 sources[TILE];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = begin + local;
    const bool active = local < count;
    Vec3 p = {0.0, 0.0, 0.0};
    double energy = 0.0;
    if (active) {
        p = positions[i];
        const Vec3 v = velocities[local];
        energy = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }
    // All threads in a block can skip tiles preceding its first target.
    for (int base = ((begin + int(blockIdx.x) * TILE) / TILE) * TILE;
         base < n; base += TILE) {
        const int j = base + threadIdx.x;
        if (j < n) sources[threadIdx.x] = positions[j];
        __syncthreads();
        if (active) {
            for (int k = max(0, i + 1 - base); k < min(TILE, n - base); ++k) {
                const double dx = sources[k].x - p.x;
                const double dy = sources[k].y - p.y;
                const double dz = sources[k].z - p.z;
                energy -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            }
        }
        __syncthreads();
    }
    if (active) energies[local] = energy;
}

struct GpuPartition {
    int device, offset, count, forceThreads;
    cudaStream_t stream = nullptr;
    Vec3 *positions = nullptr, *velocities = nullptr, *next = nullptr;
    double* energies = nullptr;
};

bool validBody(const Vec3& p, const Vec3& v) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
           std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) &&
           std::abs(p.x) <= 1e6 && std::abs(p.y) <= 1e6 && std::abs(p.z) <= 1e6 &&
           std::abs(v.x) <= 1e6 && std::abs(v.y) <= 1e6 && std::abs(v.z) <= 1e6;
}

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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    // Three doubles per position must fit MPI's count/displacement interface.
    if (numBodies < 0 || numBodies > INT_MAX / 3 || numSteps < 0) {
        if (rank == 0) fprintf(stderr, "Invalid body or step count.\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Partition contiguous target ranges evenly. Source positions are replicated;
    // only positions, never velocities, are exchanged during timestepping.
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const int count = numBodies / ranks + (r < numBodies % ranks);
        const int begin = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
        counts[r] = 3 * count;
        displacements[r] = 3 * begin;
    }
    const int count = counts[rank] / 3, begin = displacements[rank] / 3;

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, localRanks = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localRanks);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> devices;
    // Supports both one-rank-per-GPU launches and one rank driving several GPUs.
    // Scheduler-provided CUDA_VISIBLE_DEVICES masks are respected automatically.
    for (int d = localRank; d < deviceCount; d += localRanks) devices.push_back(d);
    if (devices.empty()) devices.push_back(localRank % deviceCount);
    const int workers = std::min(int(devices.size()), std::max(count, 1));
    omp_set_dynamic(0);

    Vec3 *positions = nullptr, *next = nullptr, *velocities = nullptr;
    // Pinned staging works with ordinary MPI; CUDA-aware MPI is not required.
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&positions),
                             size_t(std::max(numBodies, 1)) * sizeof(Vec3)));
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&next),
                             size_t(std::max(count, 1)) * sizeof(Vec3)));
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&velocities),
                             size_t(std::max(count, 1)) * sizeof(Vec3)));
    std::vector<Vec3> initialVelocities;
    if (rank == 0) {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies); // Preserve the original rand_r stream exactly.
        initialVelocities.resize(numBodies);
        #pragma omp parallel for schedule(static) num_threads(workers)
        for (int i = 0; i < numBodies; ++i) {
            positions[i] = bodies[i].pos;
            initialVelocities[i] = bodies[i].vel;
        }
    }
    MPI_Bcast(positions, numBodies * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(initialVelocities.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                 velocities, count * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<Vec3>().swap(initialVelocities);

    std::vector<GpuPartition> gpus(workers);
    #pragma omp parallel for schedule(static, 1) num_threads(workers)
    for (int g = 0; g < workers; ++g) {
        auto& gpu = gpus[g];
        gpu.device = devices[g];
        gpu.count = count / workers + (g < count % workers);
        gpu.offset = g * (count / workers) + std::min(g, count % workers);
        cudaCheck(cudaSetDevice(gpu.device));
        cudaDeviceProp properties;
        cudaCheck(cudaGetDeviceProperties(&properties, gpu.device));
        // Small target partitions need more blocks to occupy all SMs. Larger
        // partitions benefit from reusing each source tile across more targets.
        gpu.forceThreads = gpu.count < properties.multiProcessorCount * TILE ? 32 : TILE;
        cudaCheck(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&gpu.positions),
                             size_t(std::max(numBodies, 1)) * sizeof(Vec3)));
        const size_t localBytes = size_t(std::max(gpu.count, 1)) * sizeof(Vec3);
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&gpu.velocities), localBytes));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&gpu.next), localBytes));
        if (validate) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&gpu.energies),
                                           size_t(std::max(gpu.count, 1)) * sizeof(double)));
        cudaCheck(cudaMemcpyAsync(gpu.velocities, velocities + gpu.offset,
                                  size_t(gpu.count) * sizeof(Vec3),
                                  cudaMemcpyHostToDevice, gpu.stream));
        cudaCheck(cudaStreamSynchronize(gpu.stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    // A persistent OpenMP team avoids per-step thread creation. MPI remains on
    // the main thread (master), as required by MPI_THREAD_FUNNELED.
    #pragma omp parallel num_threads(workers)
    {
        for (int step = 0; step < numSteps; ++step) {
            #pragma omp for schedule(static, 1)
            for (int g = 0; g < workers; ++g) {
                auto& gpu = gpus[g];
                if (gpu.count == 0) continue;
                cudaCheck(cudaSetDevice(gpu.device));
                cudaCheck(cudaMemcpyAsync(gpu.positions, positions,
                                          size_t(numBodies) * sizeof(Vec3),
                                          cudaMemcpyHostToDevice, gpu.stream));
                if (gpu.forceThreads == 32) {
                    advance<32><<<(gpu.count + 31) / 32, 32, 0, gpu.stream>>>(
                        gpu.positions, gpu.velocities, gpu.next,
                        numBodies, begin + gpu.offset, gpu.count);
                } else {
                    advance<TILE><<<(gpu.count + TILE - 1) / TILE, TILE, 0, gpu.stream>>>(
                        gpu.positions, gpu.velocities, gpu.next,
                        numBodies, begin + gpu.offset, gpu.count);
                }
                cudaCheck(cudaGetLastError());
                cudaCheck(cudaMemcpyAsync(next + gpu.offset, gpu.next,
                                          size_t(gpu.count) * sizeof(Vec3),
                                          cudaMemcpyDeviceToHost, gpu.stream));
                cudaCheck(cudaStreamSynchronize(gpu.stream));
            }
            #pragma omp master
            {
                MPI_Allgatherv(next, count * 3, MPI_DOUBLE, positions,
                               counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            }
            #pragma omp barrier
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %lld ms\n", (long long)(maxElapsed * 1000.0));

    std::vector<double> energies(validate ? count : 0);
    if (printResults || validate) {
        #pragma omp parallel for schedule(static, 1) num_threads(workers)
        for (int g = 0; g < workers; ++g) {
            auto& gpu = gpus[g];
            if (gpu.count == 0) continue;
            cudaCheck(cudaSetDevice(gpu.device));
            cudaCheck(cudaMemcpyAsync(velocities + gpu.offset, gpu.velocities,
                                      size_t(gpu.count) * sizeof(Vec3),
                                      cudaMemcpyDeviceToHost, gpu.stream));
            if (validate) {
                cudaCheck(cudaMemcpyAsync(gpu.positions, positions,
                                          size_t(numBodies) * sizeof(Vec3),
                                          cudaMemcpyHostToDevice, gpu.stream));
                energyKernel<<<(gpu.count + TILE - 1) / TILE, TILE, 0, gpu.stream>>>(
                    gpu.positions, gpu.velocities, gpu.energies,
                    numBodies, begin + gpu.offset, gpu.count);
                cudaCheck(cudaGetLastError());
                cudaCheck(cudaMemcpyAsync(energies.data() + gpu.offset, gpu.energies,
                                          size_t(gpu.count) * sizeof(double),
                                          cudaMemcpyDeviceToHost, gpu.stream));
            }
            cudaCheck(cudaStreamSynchronize(gpu.stream));
        }
    }
    if (printResults) {
        std::vector<Vec3> allVelocities(rank == 0 ? numBodies : 0);
        MPI_Gatherv(velocities, count * 3, MPI_DOUBLE, allVelocities.data(),
                    counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
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
    int success = 1;
    if (validate) {
        int localValid = 1;
        double localEnergy = 0.0, finalEnergy = 0.0;
        #pragma omp parallel for reduction(&:localValid) reduction(+:localEnergy) schedule(static) num_threads(workers)
        for (int i = 0; i < count; ++i) {
            localValid &= validBody(positions[begin + i], velocities[i]);
            localEnergy += energies[i];
        }
        MPI_Allreduce(&localValid, &success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating simulation results...\n");
            if (success) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation failed: found nonfinite or out-of-bounds body state\n");
                printf("Validation: FAILED\n");
            }
        }
    }
    #pragma omp parallel for schedule(static, 1) num_threads(workers)
    for (int g = 0; g < workers; ++g) {
        auto& gpu = gpus[g];
        cudaCheck(cudaSetDevice(gpu.device));
        cudaCheck(cudaFree(gpu.positions));
        cudaCheck(cudaFree(gpu.velocities));
        cudaCheck(cudaFree(gpu.next));
        cudaCheck(cudaFree(gpu.energies));
        cudaCheck(cudaStreamDestroy(gpu.stream));
    }
    cudaCheck(cudaFreeHost(positions));
    cudaCheck(cudaFreeHost(next));
    cudaCheck(cudaFreeHost(velocities));
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return success ? 0 : 1;
}
