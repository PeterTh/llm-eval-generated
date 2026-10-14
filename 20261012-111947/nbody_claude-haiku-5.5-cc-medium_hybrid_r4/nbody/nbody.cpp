// Hybrid MPI + OpenMP + CUDA n-body benchmark.
//
// Decomposition:
//   MPI    - bodies are split into contiguous blocks, one block per rank. Positions are
//            replicated on every rank and exchanged with MPI_Allgatherv after each step.
//   OpenMP - each rank runs one host thread per local GPU; each thread owns the GPU's
//            sub-block of the rank's bodies and drives its device.
//   CUDA   - force and integration kernels. One thread per body; the j-loop is tiled
//            through shared memory but still runs in the original index order, so the
//            per-body accumulation matches the serial reference.
//
// This file is compiled as CUDA (see CMakeLists.txt), so it must remain valid CUDA C++.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// Start of part `index` when [0, total) is split into `parts` balanced contiguous ranges
int partBegin(const int total, const int parts, const int index) {
    return static_cast<int>(static_cast<long long>(total) * index / parts);
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

// Velocity update for bodies [first, first + count). vel is indexed locally, pos is global (n bodies).
__global__ void computeForcesKernel(const Vec3* __restrict__ pos, Vec3* __restrict__ vel,
                                    const int first, const int count, const int n) {
    __shared__ double tileX[BLOCK_SIZE];
    __shared__ double tileY[BLOCK_SIZE];
    __shared__ double tileZ[BLOCK_SIZE];

    const int local = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool active = local < count;
    const int self = active ? first + local : first;  // inactive threads reuse a valid body so the tile loop stays in sync
    const double xi = pos[self].x;
    const double yi = pos[self].y;
    const double zi = pos[self].z;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        if (j < n) {
            tileX[threadIdx.x] = pos[j].x;
            tileY[threadIdx.x] = pos[j].y;
            tileZ[threadIdx.x] = pos[j].z;
        }
        __syncthreads();

        const int tileCount = min(BLOCK_SIZE, n - base);
        #pragma unroll 8
        for (int k = 0; k < tileCount; ++k) {
            const double dx = tileX[k] - xi;
            const double dy = tileY[k] - yi;
            const double dz = tileZ[k] - zi;
            // Fused multiply-adds match the rounding of the reference build's FMA-contracted accumulations
            const double distSqr = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx = fma(dx, invDist3, Fx);
            Fy = fma(dy, invDist3, Fy);
            Fz = fma(dz, invDist3, Fz);
        }
        __syncthreads();
    }

    // Velocity update is a separate multiply and add (not fused), as in the reference build
    if (active) {
        vel[local].x = __dadd_rn(vel[local].x, __dmul_rn(DT, Fx));
        vel[local].y = __dadd_rn(vel[local].y, __dmul_rn(DT, Fy));
        vel[local].z = __dadd_rn(vel[local].z, __dmul_rn(DT, Fz));
    }
}

__global__ void integrateKernel(Vec3* __restrict__ pos, const Vec3* __restrict__ vel,
                                const int first, const int count) {
    const int local = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (local < count) {
        pos[first + local].x = fma(vel[local].x, DT, pos[first + local].x);
        pos[first + local].y = fma(vel[local].y, DT, pos[first + local].y);
        pos[first + local].z = fma(vel[local].z, DT, pos[first + local].z);
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

int runBenchmark(int argc, char** argv, const int rank, const int numRanks) {
    const bool root = rank == 0;

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
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (numBodies < 0) {
        if (root) printf("Number of bodies must be non-negative\n");
        return 1;
    }

    if (root) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Map this rank's share of the node's GPUs. Ranks on a node take disjoint devices
    // round-robin; if there are more ranks than GPUs, ranks share devices.
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0, nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_free(&nodeComm);

    std::vector<int> devices;
    if (nodeSize <= deviceCount) {
        for (int d = nodeRank; d < deviceCount; d += nodeSize) {
            devices.push_back(d);
        }
    } else {
        devices.push_back(nodeRank % deviceCount);
    }
    const int numThreads = static_cast<int>(devices.size());

    // Initialize bodies (identical on every rank, so no broadcast is needed)
    std::vector<Body> initial(numBodies);
    randomizeBodies(initial);
    std::vector<Vec3> pos(numBodies), vel(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        pos[i] = initial[i].pos;
        vel[i] = initial[i].vel;
    }
    // Positions are double-buffered: step s reads buffer s%2 while other threads write buffer (s+1)%2,
    // so device uploads never race with the host write-back of another GPU's slice
    std::vector<Vec3> posBuffers[2] = {pos, pos};

    // Global body ranges: rank block, and per-rank counts/displacements in doubles for MPI
    const int rankFirst = partBegin(numBodies, numRanks, rank);
    const int rankCount = partBegin(numBodies, numRanks, rank + 1) - rankFirst;
    std::vector<int> posCounts(numRanks), posDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const int first = partBegin(numBodies, numRanks, r);
        posCounts[r] = 3 * (partBegin(numBodies, numRanks, r + 1) - first);
        posDispls[r] = 3 * first;
    }

    long elapsedMs = 0;
    omp_set_dynamic(0);

    #pragma omp parallel num_threads(numThreads)
    {
        // Every device needs its own thread, otherwise part of the body range would never be computed
        if (omp_get_num_threads() != numThreads) {
            fprintf(stderr, "Rank %d: OpenMP team of %d threads, need %d (check OMP_THREAD_LIMIT)\n",
                    rank, omp_get_num_threads(), numThreads);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        const int t = omp_get_thread_num();
        const int first = rankFirst + partBegin(rankCount, numThreads, t);
        const int count = rankFirst + partBegin(rankCount, numThreads, t + 1) - first;
        const int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

        CUDA_CHECK(cudaSetDevice(devices[t]));
        Vec3* dPos = nullptr;
        Vec3* dVel = nullptr;
        CUDA_CHECK(cudaMalloc(&dPos, numBodies * sizeof(Vec3)));
        CUDA_CHECK(cudaMalloc(&dVel, count * sizeof(Vec3)));
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(dVel, vel.data() + first, count * sizeof(Vec3), cudaMemcpyHostToDevice));
        }

        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point start;
        #pragma omp master
        start = std::chrono::high_resolution_clock::now();
        #pragma omp barrier

        for (int step = 0; step < numSteps; ++step) {
            const std::vector<Vec3>& cur = posBuffers[step % 2];
            std::vector<Vec3>& next = posBuffers[(step + 1) % 2];
            if (count > 0) {
                // Forces read the full position set; integration runs after them on the same stream
                CUDA_CHECK(cudaMemcpy(dPos, cur.data(), numBodies * sizeof(Vec3), cudaMemcpyHostToDevice));
                computeForcesKernel<<<blocks, BLOCK_SIZE>>>(dPos, dVel, first, count, numBodies);
                integrateKernel<<<blocks, BLOCK_SIZE>>>(dPos, dVel, first, count);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(next.data() + first, dPos + first, count * sizeof(Vec3),
                                      cudaMemcpyDeviceToHost));
            }
            #pragma omp barrier
            #pragma omp master
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, next.data(), posCounts.data(),
                           posDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            #pragma omp barrier
        }

        #pragma omp master
        {
            const auto end = std::chrono::high_resolution_clock::now();
            elapsedMs = static_cast<long>(
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        }

        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(vel.data() + first, dVel, count * sizeof(Vec3), cudaMemcpyDeviceToHost));
        }
        #pragma omp barrier
        if (printResults || validate) {
            #pragma omp master
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, vel.data(), posCounts.data(),
                           posDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            #pragma omp barrier
        }

        CUDA_CHECK(cudaFree(dPos));
        CUDA_CHECK(cudaFree(dVel));
    }

    long maxElapsedMs = 0;
    MPI_Reduce(&elapsedMs, &maxElapsedMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (root) {
        printf("Simulation time: %ld ms\n", maxElapsedMs);
    }

    if (!root || !(printResults || validate)) {
        return 0;
    }

    const std::vector<Vec3>& finalPos = posBuffers[std::max(numSteps, 0) % 2];
    std::vector<Body> bodies(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos = finalPos[i];
        bodies[i].vel = vel[i];
    }

    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : bodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int status = runBenchmark(argc, argv, rank, numRanks);

    MPI_Finalize();
    return status;
}
