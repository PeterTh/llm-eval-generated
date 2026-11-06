#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 256;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos, vel; };
static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be contiguous");

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}
#define CUDA(call) checkCuda((call), #call)

// Each thread owns one target. All threads participate in the tile barriers,
// including those in the partially filled final block.
__global__ void advance(const Vec3* positions, Vec3* nextPositions, Vec3* velocities,
                        int n, int first, int localCount) {
    __shared__ Vec3 tile[BLOCK_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < localCount;
    Vec3 p = active ? positions[first + i] : Vec3{0.0, 0.0, 0.0};
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        const int count = min(BLOCK_SIZE, n - base);
        if (active) {
            for (int k = 0; k < count; ++k) {
                const double dx = tile[k].x - p.x;
                const double dy = tile[k].y - p.y;
                const double dz = tile[k].z - p.z;
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
        v.x += DT * fx; v.y += DT * fy; v.z += DT * fz;
        p.x += v.x * DT; p.y += v.y * DT; p.z += v.z * DT;
        velocities[i] = v;
        nextPositions[i] = p;
    }
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

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const int n = static_cast<int>(bodies.size());
    #pragma omp parallel for reduction(+:energy) schedule(dynamic, 16)
    for (int i = 0; i < n; ++i) {
        const auto& v = bodies[i].vel;
        energy += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    // Report the same first failure as the serial implementation.
    int first = static_cast<int>(bodies.size());
    #pragma omp parallel for reduction(min:first) schedule(static)
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        const auto& p = bodies[i].pos;
        const auto& v = bodies[i].vel;
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
            !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z) ||
            std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6 ||
            std::abs(v.x) > 1e6 || std::abs(v.y) > 1e6 || std::abs(v.z) > 1e6)
            first = std::min(first, i);
    }
    if (first == static_cast<int>(bodies.size())) return true;
    const auto& p = bodies[first].pos;
    const auto& v = bodies[first].vel;
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
        printf("Validation failed: found NaN or Inf value in body state\n");
    else if (std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6)
        printf("Validation failed: body position exceeds reasonable bounds\n");
    else printf("Validation failed: body velocity exceeds reasonable bounds\n");
    return false;
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
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (numBodies < 0 || numBodies > std::numeric_limits<int>::max() / 3) {
        if (rank == 0) fprintf(stderr, "Invalid number of bodies\n");
        MPI_Finalize(); return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
               numBodies, numSteps, validate ? "enabled" : "disabled");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0, offset = 0; r < ranks; ++r) {
        const int count = numBodies / ranks + (r < numBodies % ranks ? 1 : 0);
        counts[r] = 3 * count;
        offsets[r] = 3 * offset;
        offset += count;
    }
    const int first = offsets[rank] / 3, localCount = counts[rank] / 3;
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    Vec3 *hostPositions = nullptr, *hostVelocities = nullptr;
    const size_t allBytes = std::max<size_t>(1, numBodies) * sizeof(Vec3);
    const size_t localBytes = std::max<size_t>(1, localCount) * sizeof(Vec3);
    CUDA(cudaMallocHost(&hostPositions, allBytes));
    CUDA(cudaMallocHost(&hostVelocities, localBytes));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) hostPositions[i] = bodies[i].pos;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) hostVelocities[i] = bodies[first + i].vel;

    Vec3 *devicePositions = nullptr, *deviceNext = nullptr, *deviceVelocities = nullptr;
    CUDA(cudaMalloc(&devicePositions, allBytes));
    // With one rank, the output is a full snapshot and can be swapped directly.
    CUDA(cudaMalloc(&deviceNext, ranks == 1 ? allBytes : localBytes));
    CUDA(cudaMalloc(&deviceVelocities, localBytes));
    if (numBodies) CUDA(cudaMemcpy(devicePositions, hostPositions, numBodies * sizeof(Vec3), cudaMemcpyHostToDevice));
    if (localCount) CUDA(cudaMemcpy(deviceVelocities, hostVelocities, localCount * sizeof(Vec3), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount) {
            advance<<<(localCount + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                devicePositions, deviceNext, deviceVelocities, numBodies, first, localCount);
            CUDA(cudaGetLastError());
        }
        if (step + 1 < numSteps) {
            if (ranks == 1) {
                std::swap(devicePositions, deviceNext);
            } else {
                if (localCount) CUDA(cudaMemcpy(hostPositions + first, deviceNext, localCount * sizeof(Vec3), cudaMemcpyDeviceToHost));
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, hostPositions, counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
                if (numBodies) CUDA(cudaMemcpy(devicePositions, hostPositions, numBodies * sizeof(Vec3), cudaMemcpyHostToDevice));
            }
        }
    }
    CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Simulation time: %ld ms\n", static_cast<long>(duration.count()));
    }

    int status = 0;
    if (printResults || validate) {
        if (numSteps > 0 && localCount) {
            CUDA(cudaMemcpy(hostPositions + first, deviceNext, localCount * sizeof(Vec3), cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(hostVelocities, deviceVelocities, localCount * sizeof(Vec3), cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, hostPositions, counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        std::vector<Vec3> gatheredVelocities(rank == 0 ? numBodies : 0);
        MPI_Gatherv(hostVelocities, counts[rank], MPI_DOUBLE, gatheredVelocities.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) { bodies[i].pos = hostPositions[i]; bodies[i].vel = gatheredVelocities[i]; }
            if (printResults) {
                std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < numBodies; ++i) {
                    bodyData[6 * size_t(i)] = bodies[i].pos.x; bodyData[6 * size_t(i) + 1] = bodies[i].pos.y;
                    bodyData[6 * size_t(i) + 2] = bodies[i].pos.z; bodyData[6 * size_t(i) + 3] = bodies[i].vel.x;
                    bodyData[6 * size_t(i) + 4] = bodies[i].vel.y; bodyData[6 * size_t(i) + 5] = bodies[i].vel.z;
                }
                print_results(bodyData, "Bodies");
            }
            if (validate) {
                printf("Validating simulation results...\n");
                if (validateSimulation(bodies)) {
                    printf("Final energy: %.6f\n", computeTotalEnergy(bodies));
                    printf("Validation: PASSED\n");
                } else { printf("Validation: FAILED\n"); status = 1; }
            }
        }
    }
    CUDA(cudaFree(devicePositions)); CUDA(cudaFree(deviceNext)); CUDA(cudaFree(deviceVelocities));
    CUDA(cudaFreeHost(hostPositions)); CUDA(cudaFreeHost(hostVelocities));
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
