#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos, vel; };

__global__ void forceKernel(const Body* bodies, Body* output, int n, int begin, int count) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;
    const int i = begin + local;
    const double xi = bodies[i].pos.x, yi = bodies[i].pos.y, zi = bodies[i].pos.z;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dx = bodies[j].pos.x - xi;
        const double dy = bodies[j].pos.y - yi;
        const double dz = bodies[j].pos.z - zi;
        const double r2 = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double inv = 1.0 / sqrt(r2);
        const double inv3 = inv * inv * inv;
        fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
    }
    output[local] = bodies[i];
    output[local].vel.x += DT * fx;
    output[local].vel.y += DT * fy;
    output[local].vel.z += DT * fz;
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void computeForces(std::vector<Body>& bodies, int rank, int ranks) {
    const int n = static_cast<int>(bodies.size());
    const int begin = static_cast<int>((static_cast<long long>(n) * rank) / ranks);
    const int end = static_cast<int>((static_cast<long long>(n) * (rank + 1)) / ranks);
    const int count = end - begin;
    if (count == 0) return;

    Body *deviceBodies = nullptr, *deviceOutput = nullptr;
    cudaCheck(cudaMalloc(&deviceBodies, static_cast<size_t>(n) * sizeof(Body)), "allocating bodies");
    cudaCheck(cudaMalloc(&deviceOutput, static_cast<size_t>(count) * sizeof(Body)), "allocating force output");
    cudaCheck(cudaMemcpy(deviceBodies, bodies.data(), static_cast<size_t>(n) * sizeof(Body), cudaMemcpyHostToDevice), "copying bodies to device");
    forceKernel<<<(count + 127) / 128, 128>>>(deviceBodies, deviceOutput, n, begin, count);
    cudaCheck(cudaGetLastError(), "launching force kernel");
    std::vector<Body> local(static_cast<size_t>(count));
    cudaCheck(cudaMemcpy(local.data(), deviceOutput, static_cast<size_t>(count) * sizeof(Body), cudaMemcpyDeviceToHost), "copying forces from device");
    for (int i = 0; i < count; ++i) bodies[begin + i] = local[i];
    cudaFree(deviceOutput);
    cudaFree(deviceBodies);
}

static void integrateBodies(std::vector<Body>& bodies) {
    const long long n = static_cast<long long>(bodies.size());
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

static double computeTotalEnergy(const std::vector<Body>& bodies) {
    const long long n = static_cast<long long>(bodies.size());
    double energy = 0.0;
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (long long i = 0; i < n; ++i) {
        const Body& b = bodies[i];
        energy += 0.5 * (b.vel.x*b.vel.x + b.vel.y*b.vel.y + b.vel.z*b.vel.z);
        for (long long j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - b.pos.x;
            const double dy = bodies[j].pos.y - b.pos.y;
            const double dz = bodies[j].pos.z - b.pos.z;
            energy -= 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
        }
    }
    return energy;
}

static bool validateSimulation(const std::vector<Body>& bodies) {
    int valid = 1;
    const long long n = static_cast<long long>(bodies.size());
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (long long i = 0; i < n; ++i) {
        const Body& b = bodies[i];
        if (!std::isfinite(b.pos.x) || !std::isfinite(b.pos.y) || !std::isfinite(b.pos.z) ||
            !std::isfinite(b.vel.x) || !std::isfinite(b.vel.y) || !std::isfinite(b.vel.z) ||
            std::abs(b.pos.x) > 1e6 || std::abs(b.pos.y) > 1e6 || std::abs(b.pos.z) > 1e6 ||
            std::abs(b.vel.x) > 1e6 || std::abs(b.vel.y) > 1e6 || std::abs(b.vel.z) > 1e6) valid = 0;
    }
    if (!valid) printf("Validation failed: body state contains non-finite or extreme values\n");
    return valid != 0;
}

static void randomizeBodies(std::vector<Body>& bodies) {
    unsigned int seed = 42;
    for (auto& b : bodies) {
        b.pos.x=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; b.pos.y=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        b.pos.z=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; b.vel.x=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        b.vel.y=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; b.vel.z=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "This benchmark requires an available CUDA device on every MPI rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(rank % deviceCount), "selecting device");

    int numBodies = 1024, numSteps = 10, validate = 0, printResults = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i+1 < argc) numBodies = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i+1 < argc) numSteps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = 1;
        else if (!strcmp(argv[i], "-r")) printResults = 1;
        else if (!strcmp(argv[i], "-h")) {
            if (rank == 0) printf("Usage: %s [-n num] [-s num] [-v] [-r] [-h]\n", argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) fprintf(stderr, "Unknown option: %s\n", argv[i]);
            MPI_Finalize(); return 1;
        }
    }
    if (numBodies < 0 || numSteps < 0) { if (rank == 0) fprintf(stderr,"Body and step counts must be nonnegative.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    if (rank == 0) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), numBodies * static_cast<int>(sizeof(Body)), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n", numBodies, numSteps, validate ? "enabled" : "disabled");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, rank, ranks);
        std::vector<int> counts(ranks), displacements(ranks);
        for (int r = 0; r < ranks; ++r) {
            const int first = static_cast<int>((static_cast<long long>(numBodies) * r) / ranks);
            const int last = static_cast<int>((static_cast<long long>(numBodies) * (r + 1)) / ranks);
            counts[r] = (last - first) * static_cast<int>(sizeof(Body));
            displacements[r] = first * static_cast<int>(sizeof(Body));
        }
        const int first = static_cast<int>((static_cast<long long>(numBodies) * rank) / ranks);
        const int last = static_cast<int>((static_cast<long long>(numBodies) * (rank + 1)) / ranks);
        std::vector<Body> localBodies(bodies.begin() + first, bodies.begin() + last);
        MPI_Allgatherv(localBodies.data(), (last - first) * static_cast<int>(sizeof(Body)), MPI_BYTE,
                       bodies.data(), counts.data(), displacements.data(), MPI_BYTE, MPI_COMM_WORLD);
        integrateBodies(bodies);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    long max_elapsed_ms = 0;
    MPI_Reduce(&elapsed_ms, &max_elapsed_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", max_elapsed_ms);
    if (rank == 0 && printResults) {
        std::vector<double> data; data.reserve(static_cast<size_t>(numBodies)*6);
        for (const auto& b : bodies) { data.push_back(b.pos.x); data.push_back(b.pos.y); data.push_back(b.pos.z); data.push_back(b.vel.x); data.push_back(b.vel.y); data.push_back(b.vel.z); }
        print_results(data, "Bodies");
    }
    int valid = !validate || validateSimulation(bodies);
    if (validate && rank == 0) {
        if (valid) { printf("Validating simulation results...\nFinal energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(bodies)); }
        else printf("Validation: FAILED\n");
    }
    MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
