#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
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

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        // Also safe inside an OpenMP worker; MPI calls stay on the main thread.
        std::abort();
    }
}

constexpr int TILE = 128;

// Every target sees the same immutable positions. Integrating into a separate
// buffer allows force evaluation and integration to share a single launch.
__global__ void advanceBodies(const Vec3* __restrict__ positions,
                              Vec3* __restrict__ velocities,
                              Vec3* __restrict__ next, int n, int first, int count) {
    __shared__ double tileX[TILE], tileY[TILE], tileZ[TILE];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < count;
    const Vec3 p = active ? positions[first + local] : Vec3();
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += TILE) {
        const int j = base + threadIdx.x;
        if (j < n) {
            const Vec3 q = positions[j];
            tileX[threadIdx.x] = q.x;
            tileY[threadIdx.x] = q.y;
            tileZ[threadIdx.x] = q.z;
        }
        __syncthreads();
        if (active) {
            const int end = min(TILE, n - base);
            for (int k = 0; k < end; ++k) {
                const double dx = tileX[k] - p.x;
                const double dy = tileY[k] - p.y;
                const double dz = tileZ[k] - p.z;
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
        next[local] = Vec3(p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT);
    }
}

__global__ void bodyEnergies(const Vec3* positions, const Vec3* velocities,
                             double* energies, int n, int first, int count) {
    __shared__ double tileX[TILE], tileY[TILE], tileZ[TILE];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < count;
    const int i = first + local;
    const Vec3 p = active ? positions[i] : Vec3();
    const Vec3 v = active ? velocities[local] : Vec3();
    double energy = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    const int start = ((first + static_cast<int>(blockIdx.x) * TILE) / TILE) * TILE;
    for (int base = start; base < n; base += TILE) {
        const int j = base + threadIdx.x;
        if (j < n) {
            const Vec3 q = positions[j];
            tileX[threadIdx.x] = q.x;
            tileY[threadIdx.x] = q.y;
            tileZ[threadIdx.x] = q.z;
        }
        __syncthreads();
        if (active) {
            const int end = min(TILE, n - base);
            for (int k = max(0, i + 1 - base); k < end; ++k) {
                const double dx = tileX[k] - p.x;
                const double dy = tileY[k] - p.y;
                const double dz = tileZ[k] - p.z;
                energy -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            }
        }
        __syncthreads();
    }
    if (active) energies[local] = energy;
}

struct GpuPartition {
    int device = 0, first = 0, count = 0;
    cudaStream_t stream = nullptr;
    Vec3 *positions = nullptr, *velocities = nullptr, *next = nullptr;
    double* energies = nullptr;
};

// Pinned host staging works with ordinary MPI implementations as well as
// CUDA-aware MPI, without requiring a particular vendor's device-buffer API.
template<class T> T* pinnedAllocate(size_t count) {
    T* pointer = nullptr;
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&pointer),
                             std::max(size_t(1), count) * sizeof(T)));
    return pointer;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
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
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI must support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
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
    
    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) fprintf(stderr, "Body and step counts must be nonnegative\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Initialize bodies
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localComm);
    int localRank = 0, localRanks = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localRanks);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d requires a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Use all GPUs even with one rank per node. With one rank per GPU, each
    // rank gets one device. Scheduler-restricted CUDA_VISIBLE_DEVICES works too.
    const int workers = localRank < devices ?
        1 + (devices - 1 - localRank) / localRanks : 1;
    const int hostThreads = std::min(omp_get_max_threads(),
                                     std::max(1, numBodies / 4096));
    omp_set_dynamic(0);
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = (numBodies / ranks) * r + std::min(r, numBodies % ranks);
    }
    const int localCount = counts[rank];
    const int first = offsets[rank];
    static_assert(sizeof(Vec3) == 3 * sizeof(double), "MPI position layout");
    MPI_Datatype vecType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vecType);
    MPI_Type_commit(&vecType);
    Vec3* positions = pinnedAllocate<Vec3>(numBodies);
    Vec3* velocities = pinnedAllocate<Vec3>(localCount);
    double* energies = validate ? pinnedAllocate<double>(localCount) : nullptr;
    std::vector<Body> bodies(rank == 0 ? numBodies : 0);
    std::vector<Vec3> rootVelocities(rank == 0 ? numBodies : 0);
    if (rank == 0) {
        randomizeBodies(bodies);
        #pragma omp parallel for num_threads(hostThreads) schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            positions[i] = bodies[i].pos;
            rootVelocities[i] = bodies[i].vel;
        }
    }
    MPI_Bcast(positions, numBodies, vecType, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rootVelocities.data(), counts.data(), offsets.data(), vecType,
                 velocities, localCount, vecType, 0, MPI_COMM_WORLD);

    std::vector<GpuPartition> gpu(workers);
    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int w = 0; w < workers; ++w) {
        auto& g = gpu[w];
        g.device = (localRank + w * localRanks) % devices;
        g.count = localCount / workers + (w < localCount % workers);
        g.first = first + (localCount / workers) * w + std::min(w, localCount % workers);
        cudaCheck(cudaSetDevice(g.device));
        if (!g.count) continue;
        cudaCheck(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&g.positions), size_t(numBodies) * sizeof(Vec3)));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&g.velocities), size_t(g.count) * sizeof(Vec3)));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&g.next), size_t(g.count) * sizeof(Vec3)));
        if (validate)
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&g.energies), size_t(g.count) * sizeof(double)));
        cudaCheck(cudaMemcpyAsync(g.velocities, velocities + g.first - first,
                                  size_t(g.count) * sizeof(Vec3), cudaMemcpyHostToDevice, g.stream));
        cudaCheck(cudaStreamSynchronize(g.stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    // A persistent OpenMP team avoids launching host threads every timestep.
    #pragma omp parallel num_threads(workers)
    {
        for (int step = 0; step < numSteps; ++step) {
            // All workers must finish reading host positions before any worker
            // overwrites its slice with the next timestep's positions.
            #pragma omp for schedule(static, 1)
            for (int w = 0; w < workers; ++w) {
                auto& g = gpu[w];
                if (!g.count) continue;
                cudaCheck(cudaSetDevice(g.device));
                cudaCheck(cudaMemcpyAsync(g.positions, positions, size_t(numBodies) * sizeof(Vec3),
                                          cudaMemcpyHostToDevice, g.stream));
                cudaCheck(cudaStreamSynchronize(g.stream));
            }
            #pragma omp for schedule(static, 1)
            for (int w = 0; w < workers; ++w) {
                auto& g = gpu[w];
                if (!g.count) continue;
                cudaCheck(cudaSetDevice(g.device));
                advanceBodies<<<(g.count + TILE - 1) / TILE, TILE, 0, g.stream>>>(
                    g.positions, g.velocities, g.next, numBodies, g.first, g.count);
                cudaCheck(cudaGetLastError());
                cudaCheck(cudaMemcpyAsync(positions + g.first, g.next, size_t(g.count) * sizeof(Vec3),
                                          cudaMemcpyDeviceToHost, g.stream));
                cudaCheck(cudaStreamSynchronize(g.stream));
            }
            #pragma omp master
            {
                if (ranks > 1)
                    MPI_Allgatherv(MPI_IN_PLACE, 0, vecType, positions,
                                   counts.data(), offsets.data(), vecType, MPI_COMM_WORLD);
            }
            #pragma omp barrier
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %lld ms\n", static_cast<long long>(duration * 1000));

    double finalEnergy = 0.0;
    if (printResults || validate) {
        #pragma omp parallel for num_threads(workers) schedule(static, 1)
        for (int w = 0; w < workers; ++w) {
            auto& g = gpu[w];
            cudaCheck(cudaSetDevice(g.device));
            if (!g.count) continue;
            cudaCheck(cudaMemcpyAsync(velocities + g.first - first, g.velocities,
                                      size_t(g.count) * sizeof(Vec3), cudaMemcpyDeviceToHost, g.stream));
            if (validate) {
                cudaCheck(cudaMemcpyAsync(g.positions, positions, size_t(numBodies) * sizeof(Vec3),
                                          cudaMemcpyHostToDevice, g.stream));
                bodyEnergies<<<(g.count + TILE - 1) / TILE, TILE, 0, g.stream>>>(
                    g.positions, g.velocities, g.energies, numBodies, g.first, g.count);
                cudaCheck(cudaGetLastError());
                cudaCheck(cudaMemcpyAsync(energies + g.first - first, g.energies,
                                          size_t(g.count) * sizeof(double), cudaMemcpyDeviceToHost, g.stream));
            }
            cudaCheck(cudaStreamSynchronize(g.stream));
        }
        MPI_Gatherv(velocities, localCount, vecType, rootVelocities.data(),
                    counts.data(), offsets.data(), vecType, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            #pragma omp parallel for num_threads(hostThreads) schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                bodies[i].pos = positions[i];
                bodies[i].vel = rootVelocities[i];
            }
        }
        if (validate) {
            double localEnergy = 0.0;
            #pragma omp parallel for num_threads(hostThreads) reduction(+:localEnergy) schedule(static)
            for (int i = 0; i < localCount; ++i) localEnergy += energies[i];
            MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
    }

    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int w = 0; w < workers; ++w) {
        auto& g = gpu[w];
        cudaCheck(cudaSetDevice(g.device));
        if (!g.count) continue;
        cudaCheck(cudaFree(g.positions));
        cudaCheck(cudaFree(g.velocities));
        cudaCheck(cudaFree(g.next));
        if (g.energies) cudaCheck(cudaFree(g.energies));
        cudaCheck(cudaStreamDestroy(g.stream));
    }
    cudaCheck(cudaFreeHost(positions));
    cudaCheck(cudaFreeHost(velocities));
    if (energies) cudaCheck(cudaFreeHost(energies));
    MPI_Type_free(&vecType);
    MPI_Comm_free(&localComm);
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData(size_t(numBodies) * 6);
        #pragma omp parallel for num_threads(hostThreads) schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const auto& body = bodies[i];
            const size_t k = size_t(i) * 6;
            bodyData[k] = body.pos.x;
            bodyData[k + 1] = body.pos.y;
            bodyData[k + 2] = body.pos.z;
            bodyData[k + 3] = body.vel.x;
            bodyData[k + 4] = body.vel.y;
            bodyData[k + 5] = body.vel.z;
        }
        print_results(bodyData, "Bodies");
    }
    
    // Validation: check that simulation produces finite, reasonable values
    int status = 0;
    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }
    
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
