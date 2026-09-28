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
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
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

// GPU kernel: for each locally-owned body i in [localStart, localStart+localCount),
// accumulate gravitational acceleration from every other body (full N-body sum, same
// summation order as the original serial implementation) and update its velocity.
__global__ void computeForcesKernel(const double* __restrict__ posX, const double* __restrict__ posY,
                                     const double* __restrict__ posZ, double* __restrict__ velX,
                                     double* __restrict__ velY, double* __restrict__ velZ, int n,
                                     int localStart, int localCount, double dt, double softening) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= localCount) return;

    const int i = localStart + idx;
    const double xi = posX[i];
    const double yi = posY[i];
    const double zi = posZ[i];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dx = posX[j] - xi;
        const double dy = posY[j] - yi;
        const double dz = posZ[j] - zi;
        const double distSqr = dx * dx + dy * dy + dz * dz + softening;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    velX[idx] += dt * Fx;
    velY[idx] += dt * Fy;
    velZ[idx] += dt * Fz;
}

// GPU kernel: integrate the locally-owned bodies' positions using their (already updated) velocity.
__global__ void integrateBodiesKernel(double* __restrict__ posX, double* __restrict__ posY,
                                       double* __restrict__ posZ, const double* __restrict__ velX,
                                       const double* __restrict__ velY, const double* __restrict__ velZ,
                                       int localStart, int localCount, double dt) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= localCount) return;

    const int i = localStart + idx;
    posX[i] += velX[idx] * dt;
    posY[i] += velY[idx] * dt;
    posZ[i] += velZ[idx] * dt;
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

    // Potential energy (assuming unit mass for all bodies)
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
    bool ok = true;
    const long n = static_cast<long>(bodies.size());

    #pragma omp parallel for reduction(&& : ok) schedule(static)
    for (long idx = 0; idx < n; ++idx) {
        const Body& body = bodies[idx];
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
        ok = ok && localOk;
    }

    if (!ok) {
        printf("Validation failed: found NaN/Inf or out-of-bounds value in body state\n");
    }
    return ok;
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
    int provided = MPI_THREAD_FUNNELED;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank; argv is replicated by mpirun)
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

    // Select a GPU for this rank (round-robin across the visible devices on the node)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "Rank %d: no CUDA devices visible\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs visible per node: %d, OpenMP threads: %d\n", worldSize, deviceCount,
               omp_get_max_threads());
    }

    // Initialize bodies identically to the original serial program (rank 0 generates,
    // then broadcasts so every rank observes exactly the same initial condition).
    std::vector<Body> bodies(numBodies);
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    if (numBodies > 0) {
        MPI_Bcast(bodies.data(), numBodies * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Decompose the N bodies across ranks into contiguous, near-equal chunks.
    std::vector<int> counts(worldSize), displs(worldSize);
    {
        const int base = numBodies / worldSize;
        const int rem = numBodies % worldSize;
        int offset = 0;
        for (int r = 0; r < worldSize; ++r) {
            counts[r] = base + (r < rem ? 1 : 0);
            displs[r] = offset;
            offset += counts[r];
        }
    }
    const int localStart = displs[rank];
    const int localCount = counts[rank];

    // Host-side position array (Vec3 is exactly 3 contiguous doubles, no padding),
    // kept in sync across all ranks at the start of every step.
    std::vector<Vec3> allPos(numBodies);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        allPos[i] = bodies[i].pos;
    }

    std::vector<double> localVelHost(static_cast<size_t>(localCount) * 3, 0.0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) {
        localVelHost[3 * i + 0] = bodies[localStart + i].vel.x;
        localVelHost[3 * i + 1] = bodies[localStart + i].vel.y;
        localVelHost[3 * i + 2] = bodies[localStart + i].vel.z;
    }

    // Device buffers: full position set (SoA), and this rank's local velocity slice (SoA).
    double *d_posX = nullptr, *d_posY = nullptr, *d_posZ = nullptr;
    double *d_velX = nullptr, *d_velY = nullptr, *d_velZ = nullptr;
    CUDA_CHECK(cudaMalloc(&d_posX, sizeof(double) * numBodies));
    CUDA_CHECK(cudaMalloc(&d_posY, sizeof(double) * numBodies));
    CUDA_CHECK(cudaMalloc(&d_posZ, sizeof(double) * numBodies));
    if (localCount > 0) {
        CUDA_CHECK(cudaMalloc(&d_velX, sizeof(double) * localCount));
        CUDA_CHECK(cudaMalloc(&d_velY, sizeof(double) * localCount));
        CUDA_CHECK(cudaMalloc(&d_velZ, sizeof(double) * localCount));
    }

    std::vector<double> hostPosX(numBodies), hostPosY(numBodies), hostPosZ(numBodies);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        hostPosX[i] = allPos[i].x;
        hostPosY[i] = allPos[i].y;
        hostPosZ[i] = allPos[i].z;
    }
    CUDA_CHECK(cudaMemcpy(d_posX, hostPosX.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posY, hostPosY.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posZ, hostPosZ.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));
    if (localCount > 0) {
        std::vector<double> velXInit(localCount), velYInit(localCount), velZInit(localCount);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < localCount; ++i) {
            velXInit[i] = localVelHost[3 * i + 0];
            velYInit[i] = localVelHost[3 * i + 1];
            velZInit[i] = localVelHost[3 * i + 2];
        }
        CUDA_CHECK(cudaMemcpy(d_velX, velXInit.data(), sizeof(double) * localCount, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_velY, velYInit.data(), sizeof(double) * localCount, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_velZ, velZInit.data(), sizeof(double) * localCount, cudaMemcpyHostToDevice));
    }

    // Gather (recv) buffers for the per-step position exchange, sized in units of doubles
    // per body's 3 coordinates.
    std::vector<int> recvCounts3(worldSize), recvDispls3(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        recvCounts3[r] = counts[r] * 3;
        recvDispls3[r] = displs[r] * 3;
    }
    std::vector<double> sendPos(static_cast<size_t>(localCount) * 3);
    std::vector<double> gatheredPos(static_cast<size_t>(numBodies) * 3);

    const int threadsPerBlock = 256;
    const int blocks = localCount > 0 ? (localCount + threadsPerBlock - 1) / threadsPerBlock : 1;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            computeForcesKernel<<<blocks, threadsPerBlock>>>(d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ,
                                                              numBodies, localStart, localCount, DT, SOFTENING);
            CUDA_CHECK(cudaGetLastError());
            integrateBodiesKernel<<<blocks, threadsPerBlock>>>(d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ,
                                                                localStart, localCount, DT);
            CUDA_CHECK(cudaGetLastError());
        }

        if (numBodies > 0) {
            // Pull this rank's freshly-integrated positions back to host and exchange
            // with all other ranks so every rank sees the full, up-to-date position set
            // for the next step's force evaluation (MPI here is not CUDA-aware).
            if (localCount > 0) {
                std::vector<double> localX(localCount), localY(localCount), localZ(localCount);
                CUDA_CHECK(cudaMemcpy(localX.data(), d_posX + localStart, sizeof(double) * localCount,
                                       cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(localY.data(), d_posY + localStart, sizeof(double) * localCount,
                                       cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(localZ.data(), d_posZ + localStart, sizeof(double) * localCount,
                                       cudaMemcpyDeviceToHost));
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < localCount; ++i) {
                    sendPos[3 * i + 0] = localX[i];
                    sendPos[3 * i + 1] = localY[i];
                    sendPos[3 * i + 2] = localZ[i];
                }
            }

            MPI_Allgatherv(sendPos.data(), localCount * 3, MPI_DOUBLE, gatheredPos.data(), recvCounts3.data(),
                            recvDispls3.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                hostPosX[i] = gatheredPos[3 * i + 0];
                hostPosY[i] = gatheredPos[3 * i + 1];
                hostPosZ[i] = gatheredPos[3 * i + 2];
            }
            CUDA_CHECK(cudaMemcpy(d_posX, hostPosX.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_posY, hostPosY.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_posZ, hostPosZ.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long localMs = duration.count();
    long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Reconstruct the full Body array (positions already synced; gather the
    // per-rank-owned velocities) so results/validation match the serial semantics.
    std::vector<double> finalLocalVelX(localCount), finalLocalVelY(localCount), finalLocalVelZ(localCount);
    if (localCount > 0) {
        CUDA_CHECK(cudaMemcpy(finalLocalVelX.data(), d_velX, sizeof(double) * localCount, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(finalLocalVelY.data(), d_velY, sizeof(double) * localCount, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(finalLocalVelZ.data(), d_velZ, sizeof(double) * localCount, cudaMemcpyDeviceToHost));
    }
    std::vector<double> sendVel(static_cast<size_t>(localCount) * 3);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) {
        sendVel[3 * i + 0] = finalLocalVelX[i];
        sendVel[3 * i + 1] = finalLocalVelY[i];
        sendVel[3 * i + 2] = finalLocalVelZ[i];
    }
    std::vector<double> gatheredVel(static_cast<size_t>(numBodies) * 3);
    if (numBodies > 0) {
        MPI_Allgatherv(sendVel.data(), localCount * 3, MPI_DOUBLE, gatheredVel.data(), recvCounts3.data(),
                        recvDispls3.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = hostPosX[i];
        bodies[i].pos.y = hostPosY[i];
        bodies[i].pos.z = hostPosZ[i];
        bodies[i].vel.x = gatheredVel[3 * i + 0];
        bodies[i].vel.y = gatheredVel[3 * i + 1];
        bodies[i].vel.z = gatheredVel[3 * i + 2];
    }

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxMs);
    }

    int exitCode = 0;

    if (rank == 0 && printResults) {
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

    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }
        bool ok = true;
        if (rank == 0) {
            ok = validateSimulation(bodies);
        }
        int okInt = ok ? 1 : 0;
        MPI_Bcast(&okInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        ok = okInt != 0;

        if (ok) {
            if (rank == 0) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            exitCode = 0;
        } else {
            if (rank == 0) {
                printf("Validation: FAILED\n");
            }
            exitCode = 1;
        }
    }

    CUDA_CHECK(cudaFree(d_posX));
    CUDA_CHECK(cudaFree(d_posY));
    CUDA_CHECK(cudaFree(d_posZ));
    if (localCount > 0) {
        CUDA_CHECK(cudaFree(d_velX));
        CUDA_CHECK(cudaFree(d_velY));
        CUDA_CHECK(cudaFree(d_velZ));
    }

    MPI_Finalize();
    return exitCode;
}
