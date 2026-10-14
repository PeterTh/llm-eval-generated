#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int FORCE_BLOCK_SIZE = 128;
constexpr int INTEGRATE_BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

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

// Force phase for the locally owned bodies [firstBody, firstBody + count).
// Each thread owns one body and accumulates over all j in ascending order,
// exactly as the serial reference does, so results are bit-identical.
// Tiles of positions are staged in shared memory so each load is reused by the whole block.
__global__ void computeForcesKernel(const double* __restrict__ pos, double* __restrict__ vel,
                                    const int n, const int firstBody, const int count) {
    __shared__ double tileX[FORCE_BLOCK_SIZE];
    __shared__ double tileY[FORCE_BLOCK_SIZE];
    __shared__ double tileZ[FORCE_BLOCK_SIZE];

    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < count;

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (active) {
        const int i = firstBody + local;
        xi = pos[3 * i + 0];
        yi = pos[3 * i + 1];
        zi = pos[3 * i + 2];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < n; tile += FORCE_BLOCK_SIZE) {
        const int j = tile + threadIdx.x;
        if (j < n) {
            tileX[threadIdx.x] = pos[3 * j + 0];
            tileY[threadIdx.x] = pos[3 * j + 1];
            tileZ[threadIdx.x] = pos[3 * j + 2];
        }
        __syncthreads();

        if (active) {
            const int limit = min(FORCE_BLOCK_SIZE, n - tile);
            for (int k = 0; k < limit; ++k) {
                const double dx = tileX[k] - xi;
                const double dy = tileY[k] - yi;
                const double dz = tileZ[k] - zi;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        vel[3 * local + 0] += DT * Fx;
        vel[3 * local + 1] += DT * Fy;
        vel[3 * local + 2] += DT * Fz;
    }
}

// Advances the owned positions. Runs as a separate launch so that every force
// evaluation in a step reads the positions from before any body moves.
__global__ void integrateKernel(double* __restrict__ pos, const double* __restrict__ vel,
                                const int firstBody, const int count) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) {
        return;
    }
    const int i = firstBody + local;
    pos[3 * i + 0] += vel[3 * local + 0] * DT;
    pos[3 * i + 1] += vel[3 * local + 1] * DT;
    pos[3 * i + 2] += vel[3 * local + 2] * DT;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies), summed across OpenMP threads
    double potential = 0.0;
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : potential)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential += 1.0 / dist;
        }
    }

    return energy - potential;
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

// Runs the simulation on every MPI rank. Bodies are block-distributed across ranks;
// each rank computes forces for its own bodies on its GPU and the updated positions
// are exchanged with MPI_Allgatherv after every step. Rank 0 reports results.
int runSimulation(const int numBodies, const int numSteps, const bool validate, const bool printResults) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = numBodies;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Block distribution of bodies across ranks (the first n % size ranks get one extra)
    std::vector<int> counts3(size), displs3(size);
    for (int r = 0, offset = 0; r < size; ++r) {
        const int count = n / size + (r < n % size ? 1 : 0);
        counts3[r] = 3 * count;
        displs3[r] = 3 * offset;
        offset += count;
    }
    const int first = displs3[rank] / 3;
    const int count = counts3[rank] / 3;

    // Bind each local rank to its own GPU
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Initialize bodies (identical on every rank, since the seed is fixed)
    std::vector<Body> bodies(n);
    randomizeBodies(bodies);

    // Pinned host staging buffers for fast transfers. Positions are the full array
    // (kept in sync across ranks); velocities are only needed for owned bodies until the end.
    const size_t posLen = std::max<size_t>(3 * (size_t)n, 1);
    double* hPos = nullptr;
    CUDA_CHECK(cudaMallocHost(&hPos, posLen * sizeof(double)));
    std::vector<double> hVel(posLen);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        hPos[3 * i + 0] = bodies[i].pos.x;
        hPos[3 * i + 1] = bodies[i].pos.y;
        hPos[3 * i + 2] = bodies[i].pos.z;
        hVel[3 * i + 0] = bodies[i].vel.x;
        hVel[3 * i + 1] = bodies[i].vel.y;
        hVel[3 * i + 2] = bodies[i].vel.z;
    }

    double* dPos = nullptr;
    double* dVel = nullptr;
    CUDA_CHECK(cudaMalloc(&dPos, posLen * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dVel, std::max<size_t>(3 * (size_t)count, 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dPos, hPos, posLen * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, hVel.data() + 3 * (size_t)first,
                          3 * (size_t)count * sizeof(double), cudaMemcpyHostToDevice));

    const int forceBlocks = (count + FORCE_BLOCK_SIZE - 1) / FORCE_BLOCK_SIZE;
    const int integrateBlocks = (count + INTEGRATE_BLOCK_SIZE - 1) / INTEGRATE_BLOCK_SIZE;

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (count > 0) {
            computeForcesKernel<<<forceBlocks, FORCE_BLOCK_SIZE>>>(dPos, dVel, n, first, count);
            integrateKernel<<<integrateBlocks, INTEGRATE_BLOCK_SIZE>>>(dPos, dVel, first, count);
            CUDA_CHECK(cudaGetLastError());
        }

        // Share the updated positions of owned bodies with all ranks
        CUDA_CHECK(cudaMemcpy(hPos + 3 * (size_t)first, dPos + 3 * (size_t)first,
                              3 * (size_t)count * sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hPos, counts3.data(), displs3.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Refresh the device copy of bodies owned by other ranks (owned slice is already current)
        if (first > 0) {
            CUDA_CHECK(cudaMemcpy(dPos, hPos, 3 * (size_t)first * sizeof(double), cudaMemcpyHostToDevice));
        }
        if (first + count < n) {
            const size_t tail = 3 * (size_t)(first + count);
            CUDA_CHECK(cudaMemcpy(dPos + tail, hPos + tail, (3 * (size_t)n - tail) * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Gather final velocities so rank 0 has the complete state for reporting and validation
    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(hVel.data() + 3 * (size_t)first, dVel,
                              3 * (size_t)count * sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hVel.data(), counts3.data(), displs3.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        if (rank == 0) {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                bodies[i].pos = Vec3(hPos[3 * i + 0], hPos[3 * i + 1], hPos[3 * i + 2]);
                bodies[i].vel = Vec3(hVel[3 * i + 0], hVel[3 * i + 1], hVel[3 * i + 2]);
            }
        }
    }

    // Print results for external validation
    if (printResults && rank == 0) {
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
    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
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
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFree(dPos));
    CUDA_CHECK(cudaFree(dVel));
    CUDA_CHECK(cudaFreeHost(hPos));

    return exitCode;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank parses the same arguments)
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    if (numBodies < 0) {
        if (rank == 0) {
            printf("Number of bodies must be non-negative\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int exitCode = runSimulation(numBodies, numSteps, validate, printResults);

    MPI_Finalize();
    return exitCode;
}
