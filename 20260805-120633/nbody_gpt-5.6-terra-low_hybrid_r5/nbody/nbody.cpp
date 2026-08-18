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
constexpr int TILE_SIZE = 256;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos; Vec3 vel; };

static void cudaCheck(cudaError_t status, const char* expression, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA failure in %s: %s\n", rank, expression,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(expr) cudaCheck((expr), #expr, rank)

// The source array is read-only for the whole launch.  This preserves the
// original explicit-Euler semantics while allowing every target body to run independently.
__global__ void advanceBodies(const Body* __restrict__ source, Body* __restrict__ destination,
                              int totalBodies, int firstBody, int localBodies) {
    __shared__ Body tile[TILE_SIZE];
    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIndex >= localBodies) return;
    const int i = firstBody + localIndex;
    const Vec3 p = source[i].pos;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int base = 0; base < totalBodies; base += TILE_SIZE) {
        const int sourceIndex = base + threadIdx.x;
        if (sourceIndex < totalBodies) tile[threadIdx.x] = source[sourceIndex];
        __syncthreads();
        const int tileCount = min(TILE_SIZE, totalBodies - base);
        #pragma unroll 4
        for (int j = 0; j < tileCount; ++j) {
            const double dx = tile[j].pos.x - p.x;
            const double dy = tile[j].pos.y - p.y;
            const double dz = tile[j].pos.z - p.z;
            const double invDist = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            const double invDist3 = invDist * invDist * invDist;
            fx += dx * invDist3;
            fy += dy * invDist3;
            fz += dz * invDist3;
        }
        __syncthreads();
    }
    Body result = source[i];
    result.vel.x += DT * fx; result.vel.y += DT * fy; result.vel.z += DT * fz;
    result.pos.x += result.vel.x * DT;
    result.pos.y += result.vel.y * DT;
    result.pos.z += result.vel.z * DT;
    destination[i] = result;
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // rand_r's sequence is intentionally kept identical to the baseline.
    for (auto& b : bodies) {
        b.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        b.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        b.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        b.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        b.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        b.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const int n = static_cast<int>(bodies.size());
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < n; ++i)
        energy += .5 * (bodies[i].vel.x * bodies[i].vel.x + bodies[i].vel.y * bodies[i].vel.y + bodies[i].vel.z * bodies[i].vel.z);
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x, dy = bodies[j].pos.y - bodies[i].pos.y, dz = bodies[j].pos.z - bodies[i].pos.z;
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        const Body& b = bodies[i];
        valid &= std::isfinite(b.pos.x) && std::isfinite(b.pos.y) && std::isfinite(b.pos.z) &&
                 std::isfinite(b.vel.x) && std::isfinite(b.vel.y) && std::isfinite(b.vel.z) &&
                 std::abs(b.pos.x) <= 1e6 && std::abs(b.pos.y) <= 1e6 && std::abs(b.pos.z) <= 1e6 &&
                 std::abs(b.vel.x) <= 1e6 && std::abs(b.vel.y) <= 1e6 && std::abs(b.vel.z) <= 1e6;
    }
    return valid != 0;
}

void printUsage(const char* p) { std::printf("Usage: %s [-n bodies] [-s steps] [-v] [-r] [-h]\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int numBodies = 1024, numSteps = 10; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (numBodies <= 0 || numSteps < 0) { if (!rank) std::fprintf(stderr, "Bodies must be positive and steps non-negative\n"); MPI_Finalize(); return 1; }
    if (!rank) std::printf("N-Body Simulation (MPI ranks: %d, OpenMP threads/rank: %d, CUDA: enabled)\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n", ranks, omp_get_max_threads(), numBodies, numSteps, validate ? "enabled" : "disabled");

    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(rank % devices));

    const int base = numBodies / ranks, remainder = numBodies % ranks;
    const int localCount = base + (rank < remainder ? 1 : 0);
    const int first = rank * base + (rank < remainder ? rank : remainder);
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0, offset = 0; r < ranks; ++r) { const int n = base + (r < remainder ? 1 : 0); counts[r] = n * sizeof(Body); displacements[r] = offset * sizeof(Body); offset += n; }
    std::vector<Body> bodies(numBodies);
    if (!rank) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), numBodies * static_cast<int>(sizeof(Body)), MPI_BYTE, 0, MPI_COMM_WORLD);
    Body *current = nullptr, *next = nullptr;
    CUDA_CHECK(cudaMalloc(&current, bodies.size() * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&next, bodies.size() * sizeof(Body)));
    CUDA_CHECK(cudaMemcpy(current, bodies.data(), bodies.size() * sizeof(Body), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        // Empty partitions are possible when more ranks than bodies are launched.
        if (localCount != 0) {
            advanceBodies<<<(localCount + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(current, next, numBodies, first, localCount);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(bodies.data() + first, next + first, localCount * sizeof(Body), cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_BYTE, bodies.data(), counts.data(), displacements.data(), MPI_BYTE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(current, bodies.data(), bodies.size() * sizeof(Body), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaFree(next)); CUDA_CHECK(cudaFree(current));
    if (!rank) std::printf("Simulation time: %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    if (!rank && printResults) { std::vector<double> data; data.reserve(numBodies * 6); for (const auto& b : bodies) { data.insert(data.end(), {b.pos.x,b.pos.y,b.pos.z,b.vel.x,b.vel.y,b.vel.z}); } print_results(data, "Bodies"); }
    int result = 0;
    if (!rank && validate) { std::printf("Validating simulation results...\n"); if (validateSimulation(bodies)) std::printf("Final energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(bodies)); else { std::printf("Validation: FAILED\n"); result = 1; } }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Finalize(); return result;
}
