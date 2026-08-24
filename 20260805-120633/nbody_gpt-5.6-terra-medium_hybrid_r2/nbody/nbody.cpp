#include <chrono>
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
constexpr int FORCE_TILE = 128;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(double x = 0, double y = 0, double z = 0) noexcept : x(x), y(y), z(z) {}
};
struct Body { Vec3 pos; Vec3 vel; };
static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be tightly packed for MPI");

[[noreturn]] static void cudaFail(cudaError_t error, const char* operation, int rank) {
    std::fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}
#define CUDA_CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) cudaFail(e, #call, rank); } while (0)

// One rank owns a contiguous body range.  Positions are replicated on every GPU,
// while this kernel updates only that rank's velocities and positions.
__global__ void advanceBodies(const double* __restrict__ positions, Body* __restrict__ local,
                              int localCount, int totalCount) {
    __shared__ double tileX[FORCE_TILE], tileY[FORCE_TILE], tileZ[FORCE_TILE];
    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIndex < localCount;
    double px = 0.0, py = 0.0, pz = 0.0, vx = 0.0, vy = 0.0, vz = 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    if (active) {
        const Body b = local[localIndex];
        px = b.pos.x; py = b.pos.y; pz = b.pos.z;
        vx = b.vel.x; vy = b.vel.y; vz = b.vel.z;
    }
    for (int base = 0; base < totalCount; base += FORCE_TILE) {
        const int j = base + threadIdx.x;
        if (j < totalCount) {
            tileX[threadIdx.x] = positions[3 * j];
            tileY[threadIdx.x] = positions[3 * j + 1];
            tileZ[threadIdx.x] = positions[3 * j + 2];
        }
        __syncthreads();
        const int limit = min(FORCE_TILE, totalCount - base);
        if (active) {
            #pragma unroll 4
            for (int k = 0; k < limit; ++k) {
                const double dx = tileX[k] - px, dy = tileY[k] - py, dz = tileZ[k] - pz;
                const double inv = rsqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                const double inv3 = inv * inv * inv;
                fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
            }
        }
        __syncthreads();
    }
    if (active) {
        vx += DT * fx; vy += DT * fy; vz += DT * fz;
        local[localIndex] = {{px + vx * DT, py + vy * DT, pz + vz * DT}, {vx, vy, vz}};
    }
}

static void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (Body& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

static bool validateSimulation(const std::vector<Body>& bodies) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const Body& b = bodies[i];
        valid &= std::isfinite(b.pos.x) && std::isfinite(b.pos.y) && std::isfinite(b.pos.z) &&
                 std::isfinite(b.vel.x) && std::isfinite(b.vel.y) && std::isfinite(b.vel.z) &&
                 std::abs(b.pos.x) <= 1e6 && std::abs(b.pos.y) <= 1e6 && std::abs(b.pos.z) <= 1e6 &&
                 std::abs(b.vel.x) <= 1e6 && std::abs(b.vel.y) <= 1e6 && std::abs(b.vel.z) <= 1e6;
    }
    return valid != 0;
}

static double computeTotalEnergy(const std::vector<Body>& bodies, int first, int count, int rank) {
    double localEnergy = 0.0;
    #pragma omp parallel for reduction(+:localEnergy) schedule(static)
    for (int ii = 0; ii < count; ++ii) {
        const int i = first + ii;
        const Body& bi = bodies[i];
        localEnergy += 0.5 * (bi.vel.x * bi.vel.x + bi.vel.y * bi.vel.y + bi.vel.z * bi.vel.z);
        for (std::size_t j = static_cast<std::size_t>(i) + 1; j < bodies.size(); ++j) {
            const double dx = bodies[j].pos.x - bi.pos.x, dy = bodies[j].pos.y - bi.pos.y, dz = bodies[j].pos.z - bi.pos.z;
            localEnergy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    double totalEnergy = 0.0;
    MPI_Reduce(&localEnergy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    (void)rank;
    return totalEnergy;
}

static void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num>  Number of bodies (default: 1024)\n  -s <num>  Steps (default: 10)\n  -v        Validate\n  -r        Print results\n  -h        Help\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) { if (rank == 0) std::fprintf(stderr, "MPI lacks required thread support\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    int numBodies = 1024, numSteps = 10; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (numBodies < 1 || numSteps < 0) { if (!rank) std::fprintf(stderr, "Bodies must be positive and steps non-negative\n"); MPI_Finalize(); return 1; }

    // Map ranks by their node-local ordinal, rather than their global ordinal,
    // so every node uses all of its accelerators independently.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0; CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);
    const int first = (numBodies * rank) / ranks;
    const int last = (numBodies * (rank + 1)) / ranks;
    const int localCount = last - first;

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) { const int begin = numBodies * r / ranks; counts[r] = 6 * (numBodies * (r + 1) / ranks - begin); offsets[r] = 6 * begin; }
    std::vector<Body> bodies(numBodies), local(localCount);
    if (!rank) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), 6 * numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) local[i] = bodies[first + i];

    std::vector<double> positions(3 * static_cast<std::size_t>(numBodies));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) { positions[3*i] = bodies[i].pos.x; positions[3*i+1] = bodies[i].pos.y; positions[3*i+2] = bodies[i].pos.z; }
    double *dPositions = nullptr; Body* dLocal = nullptr;
    CUDA_CHECK(cudaMalloc(&dPositions, positions.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dLocal, static_cast<std::size_t>(std::max(1, localCount)) * sizeof(Body)));
    CUDA_CHECK(cudaMemcpy(dLocal, local.data(), localCount * sizeof(Body), cudaMemcpyHostToDevice));

    if (!rank) { std::printf("N-Body Simulation (MPI ranks: %d, CUDA devices/rank: 1, OpenMP threads: %d)\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n", ranks, omp_get_max_threads(), numBodies, numSteps, validate ? "enabled" : "disabled"); }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) { positions[3*i] = bodies[i].pos.x; positions[3*i+1] = bodies[i].pos.y; positions[3*i+2] = bodies[i].pos.z; }
        CUDA_CHECK(cudaMemcpy(dPositions, positions.data(), positions.size() * sizeof(double), cudaMemcpyHostToDevice));
        if (localCount) {
            advanceBodies<<<(localCount + FORCE_TILE - 1) / FORCE_TILE, FORCE_TILE>>>(dPositions, dLocal, localCount, numBodies);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(local.data(), dLocal, localCount * sizeof(Body), cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(local.data(), 6 * localCount, MPI_DOUBLE, bodies.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();
    const long localElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long globalElapsedMs = 0;
    MPI_Reduce(&localElapsedMs, &globalElapsedMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) std::printf("Simulation time: %ld ms\n", globalElapsedMs);

    if (printResults && !rank) { std::vector<double> data(6 * static_cast<std::size_t>(numBodies));
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) { data[6*i]=bodies[i].pos.x; data[6*i+1]=bodies[i].pos.y; data[6*i+2]=bodies[i].pos.z; data[6*i+3]=bodies[i].vel.x; data[6*i+4]=bodies[i].vel.y; data[6*i+5]=bodies[i].vel.z; } print_results(data, "Bodies"); }
    int result = 0;
    if (validate) { const bool ok = validateSimulation(bodies); double energy = computeTotalEnergy(bodies, first, localCount, rank); if (!rank) { std::printf("Validating simulation results...\n"); if (ok) std::printf("Final energy: %.6f\nValidation: PASSED\n", energy); else { std::printf("Validation: FAILED\n"); result = 1; } } }
    CUDA_CHECK(cudaFree(dLocal)); CUDA_CHECK(cudaFree(dPositions));
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Finalize(); return result;
}
