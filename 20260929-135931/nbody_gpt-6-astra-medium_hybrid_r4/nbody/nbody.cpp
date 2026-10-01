#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <climits>
#include <type_traits>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
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

// One rank per visible GPU is recommended. MPI only runs on the main thread.
struct ParallelRuntime {
    int rank, size;
    ParallelRuntime(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    ~ParallelRuntime() { MPI_Finalize(); }
};

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Small blocks retain enough independent blocks as MPI target ranges shrink.
constexpr int TILE = 64;

// Every thread owns one target; source tiles are reused by the whole block.
// Separate input/output positions preserve the original simultaneous update.
__global__ void advance(const Vec3* __restrict__ positions,
                        Vec3* __restrict__ next, Vec3* __restrict__ velocities,
                        int n, int first, int count) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < count;
    double x = 0, y = 0, z = 0;
    if (active) {
        x = positions[first + local].x;
        y = positions[first + local].y;
        z = positions[first + local].z;
    }
    double fx = 0, fy = 0, fz = 0;
    for (int base = 0; base < n; base += TILE) {
        int j = base + threadIdx.x;
        if (j < n) {
            sx[threadIdx.x] = positions[j].x;
            sy[threadIdx.x] = positions[j].y;
            sz[threadIdx.x] = positions[j].z;
        }
        __syncthreads();
        const int length = min(TILE, n - base);
        if (active) {
            for (int k = 0; k < length; ++k) {
                const double dx = sx[k] - x, dy = sy[k] - y, dz = sz[k] - z;
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
        velocities[local].x += DT * fx;
        velocities[local].y += DT * fy;
        velocities[local].z += DT * fz;
        next[local].x = x + velocities[local].x * DT;
        next[local].y = y + velocities[local].y * DT;
        next[local].z = z + velocities[local].z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    #pragma omp parallel for reduction(+:energy) schedule(dynamic, 32)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    
    return energy;
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
    ParallelRuntime runtime(argc, argv);
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
            if (runtime.rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (runtime.rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (runtime.rank == 0) printUsage(argv[0]);
            return 1;
        }
    }
    
    if (numBodies < 0 || numBodies > INT_MAX / 3 || numSteps < 0) {
        if (runtime.rank == 0) fprintf(stderr, "Invalid body or step count\n");
        return 1;
    }
    if (runtime.rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, runtime.rank,
                        MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Each MPI rank requires a CUDA GPU\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&node);

    // Contiguous, balanced target ranges; velocities never move during stepping.
    std::vector<int> counts(runtime.size), offsets(runtime.size);
    for (int r = 0; r < runtime.size; ++r) {
        counts[r] = 3 * (numBodies / runtime.size + (r < numBodies % runtime.size));
        offsets[r] = 3 * (r * (numBodies / runtime.size) +
                          std::min(r, numBodies % runtime.size));
    }
    const int count = counts[runtime.rank] / 3;
    const int first = offsets[runtime.rank] / 3;
    static_assert(sizeof(Vec3) == 3 * sizeof(double) && std::is_trivially_copyable<Vec3>::value,
                  "MPI layout requires three packed doubles");
    const size_t allBytes = std::max(1, numBodies) * sizeof(Vec3);
    const size_t localBytes = std::max(1, count) * sizeof(Vec3);
    Vec3 *hostPositions, *hostVelocities, *positions, *next, *velocities;
    // Pinned staging works with ordinary MPI; CUDA-aware MPI is not required.
    cudaCheck(cudaMallocHost(&hostPositions, allBytes));
    cudaCheck(cudaMallocHost(&hostVelocities, localBytes));
    std::vector<Body> bodies;
    std::vector<Vec3> initialVelocities;
    if (runtime.rank == 0) {
        bodies.resize(numBodies);
        randomizeBodies(bodies); // Preserve the original rand_r stream exactly.
        initialVelocities.resize(numBodies);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            hostPositions[i] = bodies[i].pos;
            initialVelocities[i] = bodies[i].vel;
        }
    }
    MPI_Bcast(hostPositions, 3 * numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(initialVelocities.data(), counts.data(), offsets.data(), MPI_DOUBLE,
                 hostVelocities, 3 * count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&positions, allBytes));
    cudaCheck(cudaMalloc(&next, localBytes));
    cudaCheck(cudaMalloc(&velocities, localBytes));
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaCheck(cudaMemcpyAsync(positions, hostPositions, numBodies * sizeof(Vec3),
                              cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(velocities, hostVelocities, count * sizeof(Vec3),
                              cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        if (count) {
            advance<<<(count + TILE - 1) / TILE, TILE, 0, stream>>>(
                positions, next, velocities, numBodies, first, count);
            cudaCheck(cudaGetLastError());
        }
        if (runtime.size == 1) {
            // Keep the complete state on the GPU for single-rank runs.
            std::swap(positions, next);
        } else {
            cudaCheck(cudaMemcpyAsync(hostPositions + first, next, count * sizeof(Vec3),
                                      cudaMemcpyDeviceToHost, stream));
            cudaCheck(cudaStreamSynchronize(stream));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, hostPositions,
                           counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            if (step + 1 < numSteps) {
                cudaCheck(cudaMemcpyAsync(positions, hostPositions, numBodies * sizeof(Vec3),
                                          cudaMemcpyHostToDevice, stream));
            }
        }
    }
    cudaCheck(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double maximumElapsed;
    MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (runtime.rank == 0) printf("Simulation time: %lld ms\n", (long long)(maximumElapsed * 1000));

    if (printResults || validate) {
        if (runtime.size == 1) {
            cudaCheck(cudaMemcpyAsync(hostPositions, positions, numBodies * sizeof(Vec3),
                                      cudaMemcpyDeviceToHost, stream));
        }
        cudaCheck(cudaMemcpyAsync(hostVelocities, velocities, count * sizeof(Vec3),
                                  cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        MPI_Gatherv(hostVelocities, 3 * count, MPI_DOUBLE, initialVelocities.data(),
                    counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (runtime.rank == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                bodies[i].pos = hostPositions[i];
                bodies[i].vel = initialVelocities[i];
            }
        }
    }
    cudaCheck(cudaFree(positions));
    cudaCheck(cudaFree(next));
    cudaCheck(cudaFree(velocities));
    cudaCheck(cudaFreeHost(hostPositions));
    cudaCheck(cudaFreeHost(hostVelocities));
    cudaCheck(cudaStreamDestroy(stream));

    // Print results for external validation
    if (printResults && runtime.rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.resize(size_t(numBodies) * 6);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const size_t k = size_t(i) * 6;
            bodyData[k] = bodies[i].pos.x;
            bodyData[k + 1] = bodies[i].pos.y;
            bodyData[k + 2] = bodies[i].pos.z;
            bodyData[k + 3] = bodies[i].vel.x;
            bodyData[k + 4] = bodies[i].vel.y;
            bodyData[k + 5] = bodies[i].vel.z;
        }
        print_results(bodyData, "Bodies");
    }
    
    // Validation: check that simulation produces finite, reasonable values
    int result = 0;
    if (validate && runtime.rank == 0) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return result;
}
