#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
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

// GPU kernel: each thread computes the force/velocity update for a single body
// owned by this rank, iterating sequentially over all N bodies (identical
// summation order to the original serial implementation).
__global__ void computeForcesKernel(const double* __restrict__ posX,
                                     const double* __restrict__ posY,
                                     const double* __restrict__ posZ,
                                     double* __restrict__ velX,
                                     double* __restrict__ velY,
                                     double* __restrict__ velZ,
                                     int localOffset, int localCount, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= localCount) return;
    const int i = localOffset + idx;

    const double xi = posX[i];
    const double yi = posY[i];
    const double zi = posZ[i];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dx = posX[j] - xi;
        const double dy = posY[j] - yi;
        const double dz = posZ[j] - zi;
        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    velX[i] += DT * Fx;
    velY[i] += DT * Fy;
    velZ[i] += DT * Fz;
}

__global__ void integrateKernel(double* __restrict__ posX,
                                 double* __restrict__ posY,
                                 double* __restrict__ posZ,
                                 const double* __restrict__ velX,
                                 const double* __restrict__ velY,
                                 const double* __restrict__ velZ,
                                 int localOffset, int localCount) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= localCount) return;
    const int i = localOffset + idx;

    posX[i] += velX[i] * DT;
    posY[i] += velY[i] * DT;
    posZ[i] += velZ[i] * DT;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    #pragma omp parallel for reduction(+:energy) schedule(dynamic, 64)
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

// Compute a balanced contiguous partition of n elements across size ranks.
void computeSplit(int n, int size, std::vector<int>& counts, std::vector<int>& displs) {
    counts.resize(size);
    displs.resize(size);
    const int base = n / size;
    const int rem = n % size;
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Select a GPU for this rank: round-robin over the GPUs visible on this node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "Rank %d: no CUDA-capable devices found\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int myDevice = nodeRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(myDevice));

    if (worldRank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs per node: %d\n", worldSize, deviceCount);
        #pragma omp parallel
        {
            #pragma omp single
            printf("OpenMP threads per rank: %d\n", omp_get_num_threads());
        }
    }

    // Every rank deterministically generates the same full initial state
    // (identical seed and sequential RNG order == identical result on all ranks).
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Partition bodies across ranks for force computation.
    std::vector<int> counts, displs;
    computeSplit(numBodies, worldSize, counts, displs);
    const int localOffset = displs[worldRank];
    const int localCount = counts[worldRank];

    // Host-side SoA buffers, kept fully synchronized across ranks each step.
    std::vector<double> posX(numBodies), posY(numBodies), posZ(numBodies);
    std::vector<double> velX(numBodies), velY(numBodies), velZ(numBodies);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        posX[i] = bodies[i].pos.x;
        posY[i] = bodies[i].pos.y;
        posZ[i] = bodies[i].pos.z;
        velX[i] = bodies[i].vel.x;
        velY[i] = bodies[i].vel.y;
        velZ[i] = bodies[i].vel.z;
    }

    // Device buffers hold the full position arrays (needed by every rank for
    // the all-pairs force sum) and the full velocity arrays (only this
    // rank's slice is ever written, but full size avoids extra bookkeping).
    double *d_posX, *d_posY, *d_posZ, *d_velX, *d_velY, *d_velZ;
    const size_t bytesN = static_cast<size_t>(numBodies) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_posX, bytesN));
    CUDA_CHECK(cudaMalloc(&d_posY, bytesN));
    CUDA_CHECK(cudaMalloc(&d_posZ, bytesN));
    CUDA_CHECK(cudaMalloc(&d_velX, bytesN));
    CUDA_CHECK(cudaMalloc(&d_velY, bytesN));
    CUDA_CHECK(cudaMalloc(&d_velZ, bytesN));

    CUDA_CHECK(cudaMemcpy(d_posX, posX.data(), bytesN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posY, posY.data(), bytesN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posZ, posZ.data(), bytesN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velX, velX.data(), bytesN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velY, velY.data(), bytesN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velZ, velZ.data(), bytesN, cudaMemcpyHostToDevice));

    std::vector<double> localSendX(localCount), localSendY(localCount), localSendZ(localCount);

    constexpr int kThreads = 256;
    const int kBlocks = localCount > 0 ? (localCount + kThreads - 1) / kThreads : 1;

    auto start = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            computeForcesKernel<<<kBlocks, kThreads>>>(d_posX, d_posY, d_posZ,
                                                        d_velX, d_velY, d_velZ,
                                                        localOffset, localCount, numBodies);
            integrateKernel<<<kBlocks, kThreads>>>(d_posX, d_posY, d_posZ,
                                                    d_velX, d_velY, d_velZ,
                                                    localOffset, localCount);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(localSendX.data(), d_posX + localOffset,
                                   static_cast<size_t>(localCount) * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(localSendY.data(), d_posY + localOffset,
                                   static_cast<size_t>(localCount) * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(localSendZ.data(), d_posZ + localOffset,
                                   static_cast<size_t>(localCount) * sizeof(double), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Synchronize updated positions across all ranks so every rank has the
        // full up-to-date position array for the next step's force computation.
        MPI_Allgatherv(localSendX.data(), localCount, MPI_DOUBLE,
                        posX.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(localSendY.data(), localCount, MPI_DOUBLE,
                        posY.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(localSendZ.data(), localCount, MPI_DOUBLE,
                        posZ.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(d_posX, posX.data(), bytesN, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_posY, posY.data(), bytesN, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_posZ, posZ.data(), bytesN, cudaMemcpyHostToDevice));
    }

    // Gather the final velocities (only ever written by their owning rank)
    // back to rank 0 for reporting/validation.
    std::vector<double> localVelX(localCount), localVelY(localCount), localVelZ(localCount);
    if (localCount > 0) {
        CUDA_CHECK(cudaMemcpy(localVelX.data(), d_velX + localOffset,
                               static_cast<size_t>(localCount) * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localVelY.data(), d_velY + localOffset,
                               static_cast<size_t>(localCount) * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localVelZ.data(), d_velZ + localOffset,
                               static_cast<size_t>(localCount) * sizeof(double), cudaMemcpyDeviceToHost));
    }

    MPI_Gatherv(localVelX.data(), localCount, MPI_DOUBLE,
                velX.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localVelY.data(), localCount, MPI_DOUBLE,
                velY.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localVelZ.data(), localCount, MPI_DOUBLE,
                velZ.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    CUDA_CHECK(cudaFree(d_posX));
    CUDA_CHECK(cudaFree(d_posY));
    CUDA_CHECK(cudaFree(d_posZ));
    CUDA_CHECK(cudaFree(d_velX));
    CUDA_CHECK(cudaFree(d_velY));
    CUDA_CHECK(cudaFree(d_velZ));

    int exitCode = 0;

    if (worldRank == 0) {
        // Positions are already fully synchronized on every rank; only
        // velocities needed the final gather above.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos.x = posX[i];
            bodies[i].pos.y = posY[i];
            bodies[i].pos.z = posZ[i];
            bodies[i].vel.x = velX[i];
            bodies[i].vel.y = velY[i];
            bodies[i].vel.z = velZ[i];
        }

        printf("Simulation time: %ld ms\n", duration.count());

        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (const auto& body : bodies) {
                bodyData.push_back(body.pos.x);
                bodyData.push_back(body.pos.y);
                bodyData.push_back(body.pos.z);
                bodyData.push_back(body.vel.x);
                bodyData.push_back(body.vel.y);
                bodyData.push_back(body.vel.z);
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
