#include <mpi.h>
#include <cuda_runtime.h>
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

struct Vec3 {
    double x, y, z;
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Body) == 6 * sizeof(double));

__global__ void computeForcesKernel(Body* bodies, const int first, const int count,
                                    const int n) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;

    const int i = first + local;
    const Body self = bodies[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dx = bodies[j].pos.x - self.pos.x;
        const double dy = bodies[j].pos.y - self.pos.y;
        const double dz = bodies[j].pos.z - self.pos.z;
        const double inv_dist = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        const double inv_dist3 = inv_dist * inv_dist * inv_dist;
        fx += dx * inv_dist3;
        fy += dy * inv_dist3;
        fz += dz * inv_dist3;
    }
    bodies[i].vel.x += DT * fx;
    bodies[i].vel.y += DT * fy;
    bodies[i].vel.z += DT * fz;
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

void integrateBodies(std::vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());
    double energy = 0.0;
#pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < n; ++i) {
        const Body& b = bodies[i];
        energy += 0.5 * (b.vel.x * b.vel.x + b.vel.y * b.vel.y + b.vel.z * b.vel.z);
    }
#pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < n; ++i) {
        const Body& b = bodies[i];
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - b.pos.x;
            const double dy = bodies[j].pos.y - b.pos.y;
            const double dz = bodies[j].pos.z - b.pos.z;
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int invalid = 0;
    const int n = static_cast<int>(bodies.size());
#pragma omp parallel for reduction(|:invalid) schedule(static)
    for (int i = 0; i < n; ++i) {
        const Body& b = bodies[i];
        const bool finite = std::isfinite(b.pos.x) && std::isfinite(b.pos.y) &&
                            std::isfinite(b.pos.z) && std::isfinite(b.vel.x) &&
                            std::isfinite(b.vel.y) && std::isfinite(b.vel.z);
        const bool bounded = std::abs(b.pos.x) <= 1e6 && std::abs(b.pos.y) <= 1e6 &&
                             std::abs(b.pos.z) <= 1e6 && std::abs(b.vel.x) <= 1e6 &&
                             std::abs(b.vel.y) <= 1e6 && std::abs(b.vel.z) <= 1e6;
        invalid |= !(finite && bounded);
    }
    if (invalid) std::printf("Validation failed: body state is non-finite or out of bounds\n");
    return invalid == 0;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -n <num>     Number of bodies (default: 1024)\n"
                "  -s <num>     Number of simulation steps (default: 10)\n"
                "  -v           Enable validation (checks energy conservation)\n"
                "  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (numBodies < 0 || numSteps < 0) { if (rank == 0) std::printf("Invalid negative size or step count\n"); MPI_Finalize(); return 1; }

    int devices = 0;
    cudaGetDeviceCount(&devices);
    if (devices == 0) { if (rank == 0) std::printf("No CUDA device available\n"); MPI_Finalize(); return 1; }
    int localRank = rank;
    if (const char* env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK")) localRank = std::atoi(env);
    else if (const char* env = std::getenv("SLURM_LOCALID")) localRank = std::atoi(env);
    cudaSetDevice(localRank % devices);

    if (rank == 0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
                    numBodies, numSteps, validate ? "enabled" : "disabled");
    }
    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    if (rank == 0) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), numBodies * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const int base = numBodies / world, extra = numBodies % world;
    const int first = rank * base + (rank < extra ? rank : extra);
    const int localCount = base + (rank < extra ? 1 : 0);
    std::vector<int> counts(world), displacements(world);
    for (int r = 0; r < world; ++r) {
        counts[r] = (base + (r < extra ? 1 : 0)) * 6;
        displacements[r] = (r * base + (r < extra ? r : extra)) * 6;
    }

    Body* deviceBodies = nullptr;
    cudaMalloc(&deviceBodies, static_cast<size_t>(std::max(1, numBodies)) * sizeof(Body));
    cudaMemcpy(deviceBodies, bodies.data(), static_cast<size_t>(numBodies) * sizeof(Body), cudaMemcpyHostToDevice);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        cudaMemcpy(deviceBodies, bodies.data(), static_cast<size_t>(numBodies) * sizeof(Body), cudaMemcpyHostToDevice);
        if (localCount > 0) {
            computeForcesKernel<<<(localCount + 255) / 256, 256>>>(deviceBodies, first, localCount, numBodies);
            cudaGetLastError();
            cudaDeviceSynchronize();
        }
        if (localCount > 0) cudaMemcpy(bodies.data() + first, deviceBodies + first,
                                       static_cast<size_t>(localCount) * sizeof(Body), cudaMemcpyDeviceToHost);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bodies.data(), counts.data(),
                       displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        integrateBodies(bodies);
    }
    cudaFree(deviceBodies);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long maxMs = 0;
    MPI_Reduce(&ms, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Simulation time: %lld ms\n", maxMs);
        if (printResults) {
            std::vector<double> data;
            data.reserve(static_cast<size_t>(numBodies) * 6);
            for (const Body& b : bodies) {
                data.insert(data.end(), {b.pos.x, b.pos.y, b.pos.z, b.vel.x, b.vel.y, b.vel.z});
            }
            print_results(data, "Bodies");
        }
        if (validate) {
            std::printf("Validating simulation results...\n");
            const bool ok = validateSimulation(bodies);
            if (ok) std::printf("Final energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(bodies));
            else std::printf("Validation: FAILED\n");
            MPI_Finalize();
            return ok ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
