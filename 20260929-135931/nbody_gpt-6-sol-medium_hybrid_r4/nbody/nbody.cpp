#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK = 128;
struct Vec3 { double x, y, z; };
struct Body { Vec3 pos, vel; };
static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(sizeof(Body) == 6 * sizeof(double));

void cudaCheck(cudaError_t error, const char* context) {
    if (error != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, context, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
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

// Every CUDA block shares a source tile across its destination bodies.
__global__ void advance(Body* bodies, Vec3* updated, const Vec3* positions,
                        int localCount, int totalCount) {
    __shared__ double sx[BLOCK], sy[BLOCK], sz[BLOCK];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < localCount;
    Body body;
    if (active) body = bodies[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < totalCount; base += BLOCK) {
        const int j = base + threadIdx.x;
        if (j < totalCount) {
            const Vec3 source = positions[j];
            sx[threadIdx.x] = source.x;
            sy[threadIdx.x] = source.y;
            sz[threadIdx.x] = source.z;
        }
        __syncthreads();
        if (active) {
            const int count = min(BLOCK, totalCount - base);
            for (int k = 0; k < count; ++k) {
                const double dx = sx[k] - body.pos.x;
                const double dy = sy[k] - body.pos.y;
                const double dz = sz[k] - body.pos.z;
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
        body.vel.x += DT * fx;
        body.vel.y += DT * fy;
        body.vel.z += DT * fz;
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
        bodies[i] = body;
        updated[i] = body.pos;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    for (const auto& body : bodies)
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n"); return false;
        }
        if (std::abs(body.pos.x) > 1e6 || std::abs(body.pos.y) > 1e6 || std::abs(body.pos.z) > 1e6) {
            printf("Validation failed: body position exceeds reasonable bounds\n"); return false;
        }
        if (std::abs(body.vel.x) > 1e6 || std::abs(body.vel.y) > 1e6 || std::abs(body.vel.z) > 1e6) {
            printf("Validation failed: body velocity exceeds reasonable bounds\n"); return false;
        }
    }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
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
    if (numBodies < 0 || numBodies > std::numeric_limits<int>::max() / 6) {
        if (rank == 0) fprintf(stderr, "Invalid number of bodies\n");
        MPI_Finalize(); return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
               numBodies, numSteps, validate ? "enabled" : "disabled");
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), bodyCounts(ranks), bodyOffsets(ranks), posCounts(ranks), posOffsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        const int offset = (numBodies / ranks) * r + (r < numBodies % ranks ? r : numBodies % ranks);
        bodyCounts[r] = counts[r] * 6;
        bodyOffsets[r] = offset * 6;
        posCounts[r] = counts[r] * 3;
        posOffsets[r] = offset * 3;
    }
    const int localCount = counts[rank];
    std::vector<Body> bodies;
    if (rank == 0) { bodies.resize(numBodies); randomizeBodies(bodies); }
    std::vector<Body> localBodies(localCount);
    MPI_Scatterv(rank == 0 ? bodies.data() : nullptr, bodyCounts.data(), bodyOffsets.data(), MPI_DOUBLE,
                 localBodies.data(), localCount * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<Vec3> positions(numBodies), localPositions(localCount);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) localPositions[i] = localBodies[i].pos;
    MPI_Allgatherv(localPositions.data(), localCount * 3, MPI_DOUBLE,
                   positions.data(), posCounts.data(), posOffsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    Body* deviceBodies = nullptr;
    Vec3* devicePositions = nullptr;
    Vec3* deviceUpdated = nullptr;
    cudaCheck(cudaMalloc(&deviceBodies, sizeof(Body) * (localCount > 0 ? localCount : 1)), "cudaMalloc bodies");
    cudaCheck(cudaMalloc(&devicePositions, sizeof(Vec3) * (numBodies > 0 ? numBodies : 1)), "cudaMalloc positions");
    cudaCheck(cudaMalloc(&deviceUpdated, sizeof(Vec3) * (localCount > 0 ? localCount : 1)), "cudaMalloc updated positions");
    if (localCount > 0)
        cudaCheck(cudaMemcpy(deviceBodies, localBodies.data(), sizeof(Body) * localCount, cudaMemcpyHostToDevice), "initial bodies");

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            cudaCheck(cudaMemcpy(devicePositions, positions.data(), sizeof(Vec3) * numBodies, cudaMemcpyHostToDevice), "positions");
            advance<<<(localCount + BLOCK - 1) / BLOCK, BLOCK>>>(deviceBodies, deviceUpdated,
                                                                    devicePositions, localCount, numBodies);
            cudaCheck(cudaGetLastError(), "advance launch");
        }
        if (step + 1 < numSteps) {
            if (localCount > 0)
                cudaCheck(cudaMemcpy(localPositions.data(), deviceUpdated, sizeof(Vec3) * localCount,
                                     cudaMemcpyDeviceToHost), "updated positions");
            MPI_Allgatherv(localPositions.data(), localCount * 3, MPI_DOUBLE,
                           positions.data(), posCounts.data(), posOffsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        }
    }
    if (localCount > 0)
        cudaCheck(cudaMemcpy(localBodies.data(), deviceBodies, sizeof(Body) * localCount,
                             cudaMemcpyDeviceToHost), "final bodies");
    double localElapsed = MPI_Wtime() - start, elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
    if (printResults || validate)
        MPI_Gatherv(localBodies.data(), localCount * 6, MPI_DOUBLE,
                    rank == 0 ? bodies.data() : nullptr, bodyCounts.data(), bodyOffsets.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    int result = 0;
    if (rank == 0) {
        if (printResults) {
            std::vector<double> data(numBodies * 6);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                data[6 * i] = bodies[i].pos.x; data[6 * i + 1] = bodies[i].pos.y;
                data[6 * i + 2] = bodies[i].pos.z; data[6 * i + 3] = bodies[i].vel.x;
                data[6 * i + 4] = bodies[i].vel.y; data[6 * i + 5] = bodies[i].vel.z;
            }
            print_results(data, "Bodies");
        }
        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(bodies)) {
                printf("Final energy: %.6f\n", computeTotalEnergy(bodies));
                printf("Validation: PASSED\n");
            } else { printf("Validation: FAILED\n"); result = 1; }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaCheck(cudaFree(deviceBodies), "cudaFree bodies");
    cudaCheck(cudaFree(devicePositions), "cudaFree positions");
    cudaCheck(cudaFree(deviceUpdated), "cudaFree updated positions");
    MPI_Finalize();
    return result;
}
