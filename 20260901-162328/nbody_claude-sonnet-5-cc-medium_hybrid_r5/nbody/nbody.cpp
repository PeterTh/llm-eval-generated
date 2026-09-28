#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

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

// One CUDA thread owns one locally-assigned body. Force accumulation walks the
// full (replicated) position array in index order 0..N-1, matching the inner
// loop order of the original sequential computeForces() exactly.
__global__ void nbodyStepKernel(const double* __restrict__ gx, const double* __restrict__ gy,
                                 const double* __restrict__ gz, double* __restrict__ lx,
                                 double* __restrict__ ly, double* __restrict__ lz,
                                 double* __restrict__ lvx, double* __restrict__ lvy,
                                 double* __restrict__ lvz, int N, int localN, int offset, double dt,
                                 double softening) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= localN) return;

    const int gi = offset + i;
    const double xi = gx[gi];
    const double yi = gy[gi];
    const double zi = gz[gi];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int j = 0; j < N; ++j) {
        const double dx = gx[j] - xi;
        const double dy = gy[j] - yi;
        const double dz = gz[j] - zi;
        const double distSqr = dx * dx + dy * dy + dz * dz + softening;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    const double nvx = lvx[i] + dt * Fx;
    const double nvy = lvy[i] + dt * Fy;
    const double nvz = lvz[i] + dt * Fz;

    lvx[i] = nvx;
    lvy[i] = nvy;
    lvz[i] = nvz;

    lx[i] = xi + nvx * dt;
    ly[i] = yi + nvy * dt;
    lz[i] = zi + nvz * dt;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
#pragma omp parallel for reduction(+ : energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
#pragma omp parallel for reduction(+ : energy) schedule(dynamic, 64)
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
    const size_t n = bodies.size();
    int ok = 1;

#pragma omp parallel for reduction(&& : ok) schedule(static)
    for (size_t idx = 0; idx < n; ++idx) {
        const auto& body = bodies[idx];
        bool localOk = true;

        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            localOk = false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            localOk = false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            localOk = false;
        }

        ok = ok && (localOk ? 1 : 0);
    }

    if (!ok) {
        printf("Validation failed: found NaN, Inf, or out-of-bounds value in body state\n");
    }
    return ok != 0;
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
    MPI_Init(&argc, &argv);

    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Determine node-local rank so each MPI rank binds to a distinct GPU on its node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA-capable devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = nodeRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs/node: %d\n", worldSize,
               omp_get_max_threads(), deviceCount);
    }

    // Every rank builds the identical, fully deterministic initial condition set
    // (cheap O(N) work) so no broadcast is required and semantics match the
    // original sequential implementation exactly.
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Static block decomposition of bodies across MPI ranks.
    const int base = numBodies / worldSize;
    const int rem = numBodies % worldSize;
    std::vector<int> counts(worldSize), displs(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = r * base + std::min(r, rem);
    }
    const int localN = counts[rank];
    const int offset = displs[rank];

    // Replicated global position arrays (SoA), and this rank's local velocity slice.
    std::vector<double> gx(numBodies), gy(numBodies), gz(numBodies);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        gx[i] = bodies[i].pos.x;
        gy[i] = bodies[i].pos.y;
        gz[i] = bodies[i].pos.z;
    }

    std::vector<double> lvx(localN), lvy(localN), lvz(localN);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < localN; ++i) {
        lvx[i] = bodies[offset + i].vel.x;
        lvy[i] = bodies[offset + i].vel.y;
        lvz[i] = bodies[offset + i].vel.z;
    }

    // Device buffers.
    double *d_gx = nullptr, *d_gy = nullptr, *d_gz = nullptr;
    double *d_lx = nullptr, *d_ly = nullptr, *d_lz = nullptr;
    double *d_lvx = nullptr, *d_lvy = nullptr, *d_lvz = nullptr;

    CUDA_CHECK(cudaMalloc(&d_gx, sizeof(double) * numBodies));
    CUDA_CHECK(cudaMalloc(&d_gy, sizeof(double) * numBodies));
    CUDA_CHECK(cudaMalloc(&d_gz, sizeof(double) * numBodies));
    CUDA_CHECK(cudaMalloc(&d_lx, sizeof(double) * std::max(localN, 1)));
    CUDA_CHECK(cudaMalloc(&d_ly, sizeof(double) * std::max(localN, 1)));
    CUDA_CHECK(cudaMalloc(&d_lz, sizeof(double) * std::max(localN, 1)));
    CUDA_CHECK(cudaMalloc(&d_lvx, sizeof(double) * std::max(localN, 1)));
    CUDA_CHECK(cudaMalloc(&d_lvy, sizeof(double) * std::max(localN, 1)));
    CUDA_CHECK(cudaMalloc(&d_lvz, sizeof(double) * std::max(localN, 1)));

    CUDA_CHECK(cudaMemcpy(d_gx, gx.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy, gy.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gz, gz.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));
    if (localN > 0) {
        CUDA_CHECK(cudaMemcpy(d_lvx, lvx.data(), sizeof(double) * localN, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lvy, lvy.data(), sizeof(double) * localN, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lvz, lvz.data(), sizeof(double) * localN, cudaMemcpyHostToDevice));
    }

    constexpr int THREADS_PER_BLOCK = 256;
    const int blocks = (localN + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (localN > 0) {
            nbodyStepKernel<<<blocks, THREADS_PER_BLOCK>>>(d_gx, d_gy, d_gz, d_lx, d_ly, d_lz, d_lvx,
                                                             d_lvy, d_lvz, numBodies, localN, offset, DT,
                                                             SOFTENING);
            CUDA_CHECK(cudaGetLastError());
        }

        if (localN > 0) {
            CUDA_CHECK(cudaMemcpy(gx.data() + offset, d_lx, sizeof(double) * localN, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gy.data() + offset, d_ly, sizeof(double) * localN, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gz.data() + offset, d_lz, sizeof(double) * localN, cudaMemcpyDeviceToHost));
        }

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, gx.data(), counts.data(), displs.data(),
                        MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, gy.data(), counts.data(), displs.data(),
                        MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, gz.data(), counts.data(), displs.data(),
                        MPI_DOUBLE, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(d_gx, gx.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gy, gy.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gz, gz.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    long long maxDuration = 0;
    long long localDuration = duration;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Pull final local velocities back from the GPU (positions are already
    // replicated on the host via the last Allgatherv).
    if (localN > 0) {
        CUDA_CHECK(cudaMemcpy(lvx.data(), d_lvx, sizeof(double) * localN, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lvy.data(), d_lvy, sizeof(double) * localN, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lvz.data(), d_lvz, sizeof(double) * localN, cudaMemcpyDeviceToHost));
    }

    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_gz);
    cudaFree(d_lx);
    cudaFree(d_ly);
    cudaFree(d_lz);
    cudaFree(d_lvx);
    cudaFree(d_lvy);
    cudaFree(d_lvz);

    // Gather velocities to rank 0 to reconstruct the full body state for
    // reporting/validation (positions are already replicated everywhere).
    std::vector<double> vx, vy, vz;
    if (rank == 0) {
        vx.resize(numBodies);
        vy.resize(numBodies);
        vz.resize(numBodies);
    }
    MPI_Gatherv(lvx.data(), localN, MPI_DOUBLE, vx.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    MPI_Gatherv(lvy.data(), localN, MPI_DOUBLE, vy.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    MPI_Gatherv(lvz.data(), localN, MPI_DOUBLE, vz.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    int exitCode = 0;

    if (rank == 0) {
        printf("Simulation time: %lld ms\n", maxDuration);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos.x = gx[i];
            bodies[i].pos.y = gy[i];
            bodies[i].pos.z = gz[i];
            bodies[i].vel.x = vx[i];
            bodies[i].vel.y = vy[i];
            bodies[i].vel.z = vz[i];
        }

        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                bodyData[i * 6 + 0] = bodies[i].pos.x;
                bodyData[i * 6 + 1] = bodies[i].pos.y;
                bodyData[i * 6 + 2] = bodies[i].pos.z;
                bodyData[i * 6 + 3] = bodies[i].vel.x;
                bodyData[i * 6 + 4] = bodies[i].vel.y;
                bodyData[i * 6 + 5] = bodies[i].vel.z;
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
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
