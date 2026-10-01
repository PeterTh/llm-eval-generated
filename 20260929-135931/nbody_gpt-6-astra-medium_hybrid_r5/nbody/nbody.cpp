#include <algorithm>
#include <climits>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// All MPI calls are made by the main thread (MPI_THREAD_FUNNELED).
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) checkCuda((call), #call)

constexpr int TILE = 128;

// One thread owns one target. Shared source tiles preserve ascending source
// order, including self interactions, without an inter-thread force reduction.
// Separate input/output positions allow force and integration to be fused.
__global__ void advanceBodies(const double3* __restrict__ positions,
                              double3* __restrict__ velocities,
                              double3* __restrict__ next,
                              int n, int first, int count) {
    __shared__ double3 sources[TILE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < count;
    double3 p = make_double3(0.0, 0.0, 0.0);
    if (active) p = positions[first + i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += TILE) {
        const int j = base + threadIdx.x;
        if (j < n) sources[threadIdx.x] = positions[j];
        __syncthreads();
        if (active) {
            const int length = min(TILE, n - base);
            #pragma unroll 8
            for (int k = 0; k < length; ++k) {
                const double dx = sources[k].x - p.x;
                const double dy = sources[k].y - p.y;
                const double dz = sources[k].z - p.z;
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
        double3 v = velocities[i];
        v.x += DT * fx;
        v.y += DT * fy;
        v.z += DT * fz;
        velocities[i] = v;
        next[i] = make_double3(p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT);
    }
}

// Optional diagnostic: distribute the triangular pair sum across MPI ranks
// and use dynamic OpenMP scheduling for its unequal row lengths.
double computeLocalEnergy(const double3* positions, const double3* velocities,
                          int n, int first, int count) {
    double energy = 0.0;
    #pragma omp parallel for schedule(dynamic, 16) reduction(+:energy)
    for (int local = 0; local < count; ++local) {
        const int i = first + local;
        const double3 v = velocities[local];
        double row = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
        for (int j = i + 1; j < n; ++j) {
            const double dx = positions[j].x - positions[i].x;
            const double dy = positions[j].y - positions[i].y;
            const double dz = positions[j].z - positions[i].z;
            row -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
        energy += row;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
    if (numBodies < 0 || numBodies > INT_MAX / 3 || numSteps < 0) {
        if (rank == 0) fprintf(stderr, "Invalid body or step count\n");
        MPI_Finalize();
        return 1;
    }

    // Node-local rank also works with launcher-provided CUDA_VISIBLE_DEVICES
    // masks (including the common one-visible-device-per-rank configuration).
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(node, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d requires a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&node);

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = 3 * (numBodies / ranks + (r < numBodies % ranks));
        offsets[r] = 3 * (r * (numBodies / ranks) + std::min(r, numBodies % ranks));
    }
    const int count = counts[rank] / 3, first = offsets[rank] / 3;
    const size_t allBytes = size_t(numBodies) * sizeof(double3);
    const size_t localBytes = size_t(count) * sizeof(double3);
    static_assert(sizeof(double3) == 3 * sizeof(double), "MPI position layout");
    double3 *hostPositions = nullptr, *hostVelocities = nullptr;
    double3 *positions = nullptr, *velocities = nullptr, *next = nullptr;
    // Pinned staging works with standard MPI; CUDA-aware MPI is not required.
    CUDA_CHECK(cudaMallocHost(&hostPositions, std::max(allBytes, sizeof(double3))));
    CUDA_CHECK(cudaMallocHost(&hostVelocities, std::max(localBytes, sizeof(double3))));
    CUDA_CHECK(cudaMalloc(&positions, std::max(allBytes, sizeof(double3))));
    CUDA_CHECK(cudaMalloc(&velocities, std::max(localBytes, sizeof(double3))));
    CUDA_CHECK(cudaMalloc(&next, std::max(localBytes, sizeof(double3))));
    std::vector<Body> bodies;
    std::vector<double3> initialVelocities;
    if (rank == 0) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
               numBodies, numSteps, validate ? "enabled" : "disabled");
        bodies.resize(numBodies);
        randomizeBodies(bodies);
        initialVelocities.resize(numBodies);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            hostPositions[i] = make_double3(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z);
            initialVelocities[i] = make_double3(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z);
        }
    }
    MPI_Bcast(hostPositions, 3 * numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(initialVelocities.data(), counts.data(), offsets.data(), MPI_DOUBLE,
                 hostVelocities, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(positions, hostPositions, allBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(velocities, hostVelocities, localBytes, cudaMemcpyHostToDevice));
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        if (count) {
            advanceBodies<<<(count + TILE - 1) / TILE, TILE>>>(positions, velocities, next,
                                                             numBodies, first, count);
            CUDA_CHECK(cudaGetLastError());
        }
        if (ranks == 1) {
            // No PCIe traffic or host synchronization within a single-GPU run.
            std::swap(positions, next);
        } else {
            CUDA_CHECK(cudaMemcpy(hostPositions + first, next, localBytes, cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, hostPositions,
                           counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(positions, hostPositions, allBytes, cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double maximumElapsed = 0.0;
    MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(maximumElapsed * 1000));

    double finalEnergy = 0.0;
    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(hostPositions, positions, allBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostVelocities, velocities, localBytes, cudaMemcpyDeviceToHost));
        MPI_Gatherv(hostVelocities, counts[rank], MPI_DOUBLE, initialVelocities.data(),
                    counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (validate) {
            const double energy = computeLocalEnergy(hostPositions, hostVelocities, numBodies, first, count);
            MPI_Reduce(&energy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        if (rank == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                bodies[i].pos = Vec3(hostPositions[i].x, hostPositions[i].y, hostPositions[i].z);
                bodies[i].vel = Vec3(initialVelocities[i].x, initialVelocities[i].y, initialVelocities[i].z);
            }
        }
    }
    CUDA_CHECK(cudaFree(next));
    CUDA_CHECK(cudaFree(velocities));
    CUDA_CHECK(cudaFree(positions));
    CUDA_CHECK(cudaFreeHost(hostVelocities));
    CUDA_CHECK(cudaFreeHost(hostPositions));
    int result = 0;
    if (rank == 0) {
        if (printResults) {
            std::vector<double> bodyData(size_t(numBodies) * 6);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                const size_t b = size_t(i) * 6;
                bodyData[b] = bodies[i].pos.x;
                bodyData[b + 1] = bodies[i].pos.y;
                bodyData[b + 2] = bodies[i].pos.z;
                bodyData[b + 3] = bodies[i].vel.x;
                bodyData[b + 4] = bodies[i].vel.y;
                bodyData[b + 5] = bodies[i].vel.z;
            }
            print_results(bodyData, "Bodies");
        }
        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(bodies)) {
                printf("Final energy: %.6f\nValidation: PASSED\n", finalEnergy);
            } else {
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
