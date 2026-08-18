#include <algorithm>
#include <chrono>
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

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos; Vec3 vel; };

static void cudaCheck(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

// Each rank owns [first, first + localCount); all bodies are replicated on its GPU.
__global__ void advanceOwnedBodies(Body* bodies, int first, int localCount, int total) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= localCount) return;
    const int i = first + local;
    const double ix = bodies[i].pos.x, iy = bodies[i].pos.y, iz = bodies[i].pos.z;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < total; ++j) {
        const double dx = bodies[j].pos.x - ix;
        const double dy = bodies[j].pos.y - iy;
        const double dz = bodies[j].pos.z - iz;
        const double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        const double inv3 = inv * inv * inv;
        fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
    }
    Body body = bodies[i];
    body.vel.x += DT * fx; body.vel.y += DT * fy; body.vel.z += DT * fz;
    body.pos.x += body.vel.x * DT; body.pos.y += body.vel.y * DT; body.pos.z += body.vel.z * DT;
    bodies[i] = body;
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Kept serial deliberately: rand_r's sequence is part of the benchmark's semantics.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

bool validateOwned(const std::vector<Body>& bodies, int first, int count) {
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

double ownedEnergy(const std::vector<Body>& bodies, int first, int count) {
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
    std::printf("Usage: %s [options]\n  -n <num>     Number of bodies (default: 1024)\n  -s <num>     Number of simulation steps (default: 10)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (!deviceCount) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), displs(ranks), byteCounts(ranks), byteDispls(ranks);
    for (int r = 0; r < ranks; ++r) { counts[r] = numBodies / ranks + (r < numBodies % ranks); displs[r] = (r ? displs[r - 1] + counts[r - 1] : 0); }
    const int localCount = counts[rank], first = displs[rank];
    for (int r = 0; r < ranks; ++r) {
        byteCounts[r] = counts[r] * static_cast<int>(sizeof(Body));
        byteDispls[r] = displs[r] * static_cast<int>(sizeof(Body));
    }
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    Body* deviceBodies = nullptr;
    cudaCheck(cudaMalloc(&deviceBodies, sizeof(Body) * static_cast<size_t>(numBodies)), "cudaMalloc");

    if (!rank) { std::printf("N-Body Simulation (MPI ranks: %d, OpenMP threads/rank: %d, CUDA GPUs: %d)\n", ranks, omp_get_max_threads(), deviceCount); std::printf("Number of bodies: %d\nNumber of steps: %d\nValidation: %s\n", numBodies, numSteps, validate ? "enabled" : "disabled"); }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    constexpr int threads = 256;
    for (int step = 0; step < numSteps; ++step) {
        cudaCheck(cudaMemcpy(deviceBodies, bodies.data(), sizeof(Body) * static_cast<size_t>(numBodies), cudaMemcpyHostToDevice), "copy bodies to GPU");
        if (localCount) {
            advanceOwnedBodies<<<(localCount + threads - 1) / threads, threads>>>(deviceBodies, first, localCount, numBodies);
            cudaCheck(cudaGetLastError(), "force kernel launch");
            cudaCheck(cudaMemcpy(bodies.data() + first, deviceBodies + first, sizeof(Body) * static_cast<size_t>(localCount), cudaMemcpyDeviceToHost), "copy owned bodies from GPU");
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bodies.data(), byteCounts.data(), byteDispls.data(), MPI_BYTE, MPI_COMM_WORLD);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count(), maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) std::printf("Simulation time: %.0f ms\n", maxElapsed);

    if (printResults && !rank) { std::vector<double> data; data.reserve(static_cast<size_t>(numBodies) * 6); for (const auto& b : bodies) { data.insert(data.end(), {b.pos.x,b.pos.y,b.pos.z,b.vel.x,b.vel.y,b.vel.z}); } print_results(data, "Bodies"); }
    int localValid = validateOwned(bodies, first, localCount), globalValid = 0;
    if (validate) { MPI_Allreduce(&localValid, &globalValid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD); double localEnergy = ownedEnergy(bodies, first, localCount), energy = 0.0; MPI_Reduce(&localEnergy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); if (!rank) { std::printf("Final energy: %.6f\nValidation: %s\n", energy, globalValid ? "PASSED" : "FAILED"); } }
    cudaFree(deviceBodies);
    MPI_Finalize();
    return validate && !globalValid ? 1 : 0;
}
