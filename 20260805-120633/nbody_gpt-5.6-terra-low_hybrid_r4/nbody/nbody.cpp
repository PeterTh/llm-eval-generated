#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos; Vec3 vel; };
static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be tightly packed");

[[noreturn]] void cudaFail(cudaError_t error, const char* expression, int rank) {
    std::fprintf(stderr, "Rank %d: CUDA error for %s: %s\n", rank, expression,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}
#define CUDA_CHECK(expr) do { const cudaError_t e = (expr); if (e != cudaSuccess) cudaFail(e, #expr, rank); } while (0)

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

// One GPU block computes several target bodies; source bodies are tiled through shared memory.
__global__ void advanceOwnedBodies(Body* bodies, int first, int count, int total) {
    extern __shared__ Vec3 sourceTile[];
    for (int group = blockIdx.x; group * blockDim.x < count; group += gridDim.x) {
        const int local = group * blockDim.x + threadIdx.x;
        const int i = first + local;
        const bool active = local < count;
        Body self{};
        if (active) self = bodies[i];
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (int tile = 0; tile < total; tile += blockDim.x) {
            const int source = tile + threadIdx.x;
            if (source < total) sourceTile[threadIdx.x] = bodies[source].pos;
            __syncthreads();
            const int tileCount = min(static_cast<int>(blockDim.x), total - tile);
            #pragma unroll 4
            for (int j = 0; j < tileCount; ++j) {
                if (active) {
                    const double dx = sourceTile[j].x - self.pos.x;
                    const double dy = sourceTile[j].y - self.pos.y;
                    const double dz = sourceTile[j].z - self.pos.z;
                    const double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                    const double inv3 = inv * inv * inv;
                    fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
                }
            }
            __syncthreads();
        }
        if (active) {
            Body updated = self;
            updated.vel.x += DT * fx; updated.vel.y += DT * fy; updated.vel.z += DT * fz;
            updated.pos.x += updated.vel.x * DT;
            updated.pos.y += updated.vel.y * DT;
            updated.pos.z += updated.vel.z * DT;
            bodies[i] = updated;
        }
    }
}

bool validateLocal(const std::vector<Body>& bodies, int first, int count) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (int i = first; i < first + count; ++i) {
        const Body& b = bodies[i];
        const bool finite = std::isfinite(b.pos.x) && std::isfinite(b.pos.y) && std::isfinite(b.pos.z) &&
                            std::isfinite(b.vel.x) && std::isfinite(b.vel.y) && std::isfinite(b.vel.z);
        const bool bounded = std::abs(b.pos.x) <= 1e6 && std::abs(b.pos.y) <= 1e6 && std::abs(b.pos.z) <= 1e6 &&
                             std::abs(b.vel.x) <= 1e6 && std::abs(b.vel.y) <= 1e6 && std::abs(b.vel.z) <= 1e6;
        valid &= static_cast<int>(finite && bounded);
    }
    return valid != 0;
}

double computeLocalEnergy(const std::vector<Body>& bodies, int first, int count) {
    const int n = static_cast<int>(bodies.size());
    double energy = 0.0;
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = first; i < first + count; ++i) {
        const Body& bi = bodies[i];
        energy += 0.5 * (bi.vel.x * bi.vel.x + bi.vel.y * bi.vel.y + bi.vel.z * bi.vel.z);
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bi.pos.x;
            const double dy = bodies[j].pos.y - bi.pos.y;
            const double dz = bodies[j].pos.z - bi.pos.z;
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return energy;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>  Number of bodies (default: 1024)\n"
                "  -s <num>  Number of simulation steps (default: 10)\n"
                "  -v        Enable validation\n  -r        Print results for external validation\n  -h        Show help\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) { if (!rank) std::fprintf(stderr, "MPI thread support is insufficient\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (numBodies <= 0 || numSteps < 0) { if (!rank) std::fprintf(stderr, "Body count must be positive and steps non-negative\n"); MPI_Finalize(); return 1; }

    MPI_Comm nodeComm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0; MPI_Comm_rank(nodeComm, &localRank); MPI_Comm_free(&nodeComm);
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));

    const int base = numBodies / ranks, remainder = numBodies % ranks;
    const int count = base + (rank < remainder ? 1 : 0);
    const int first = rank * base + (rank < remainder ? rank : remainder);
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) { counts[r] = (base + (r < remainder ? 1 : 0)) * 6; offsets[r] = (r * base + (r < remainder ? r : remainder)) * 6; }

    std::vector<Body> bodies(numBodies);
    if (!rank) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), numBodies * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    Body* deviceBodies = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceBodies, static_cast<size_t>(numBodies) * sizeof(Body)));
    constexpr int threads = 128;
    const int blocks = std::max(1, std::min((count + threads - 1) / threads, 65535));

    if (!rank) std::printf("N-Body Simulation (MPI ranks: %d, OpenMP threads/rank: %d, CUDA GPUs/node: %d)\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n", ranks, omp_get_max_threads(), devices, numBodies, numSteps, validate ? "enabled" : "disabled");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        CUDA_CHECK(cudaMemcpy(deviceBodies, bodies.data(), static_cast<size_t>(numBodies) * sizeof(Body), cudaMemcpyHostToDevice));
        advanceOwnedBodies<<<blocks, threads, threads * sizeof(Vec3)>>>(deviceBodies, first, count, numBodies);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(bodies.data() + first, deviceBodies + first, static_cast<size_t>(count) * sizeof(Body), cudaMemcpyDeviceToHost));
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bodies.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double localSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double seconds = 0.0; MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(deviceBodies));
    if (!rank) std::printf("Simulation time: %.3f ms\n", seconds * 1000.0);

    if (printResults && !rank) {
        std::vector<double> data(static_cast<size_t>(numBodies) * 6);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) std::memcpy(data.data() + 6LL * i, &bodies[i], 6 * sizeof(double));
        print_results(data, "Bodies");
    }
    int localValid = !validate || validateLocal(bodies, first, count), globalValid = 0;
    MPI_Allreduce(&localValid, &globalValid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (validate) {
        const double localEnergy = computeLocalEnergy(bodies, first, count); double energy = 0.0;
        MPI_Reduce(&localEnergy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (!rank) { std::printf("Final energy: %.6f\nValidation: %s\n", energy, globalValid ? "PASSED" : "FAILED"); }
    }
    MPI_Finalize();
    return globalValid ? 0 : 1;
}
