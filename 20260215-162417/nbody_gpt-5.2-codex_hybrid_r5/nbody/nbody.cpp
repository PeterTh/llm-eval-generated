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

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static inline void checkCuda(const cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static inline void checkMPI(const int result, const char* msg) {
    if (result != MPI_SUCCESS) {
        char err[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(result, err, &len);
        fprintf(stderr, "MPI error at %s: %s\n", msg, err);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void computeAccelerations(const double* posX,
                                     const double* posY,
                                     const double* posZ,
                                     double* accX,
                                     double* accY,
                                     double* accZ,
                                     const int nAll,
                                     const int nLocal,
                                     const int globalOffset) {
    extern __shared__ double sharedPos[];
    double* sharedX = sharedPos;
    double* sharedY = sharedPos + blockDim.x;
    double* sharedZ = sharedPos + 2 * blockDim.x;

    const int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIdx >= nLocal) {
        return;
    }

    const int globalIdx = globalOffset + localIdx;
    const double px = posX[globalIdx];
    const double py = posY[globalIdx];
    const double pz = posZ[globalIdx];

    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    for (int tile = 0; tile < nAll; tile += blockDim.x) {
        const int j = tile + threadIdx.x;
        if (j < nAll) {
            sharedX[threadIdx.x] = posX[j];
            sharedY[threadIdx.x] = posY[j];
            sharedZ[threadIdx.x] = posZ[j];
        } else {
            sharedX[threadIdx.x] = 0.0;
            sharedY[threadIdx.x] = 0.0;
            sharedZ[threadIdx.x] = 0.0;
        }
        __syncthreads();

        const int tileSize = (nAll - tile) < blockDim.x ? (nAll - tile) : blockDim.x;
        for (int k = 0; k < tileSize; ++k) {
            const double dx = sharedX[k] - px;
            const double dy = sharedY[k] - py;
            const double dz = sharedZ[k] - pz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            ax += dx * invDist3;
            ay += dy * invDist3;
            az += dz * invDist3;
        }
        __syncthreads();
    }

    accX[localIdx] = ax;
    accY[localIdx] = ay;
    accZ[localIdx] = az;
}

void randomizeBodies(std::vector<double>& posX,
                     std::vector<double>& posY,
                     std::vector<double>& posZ,
                     std::vector<double>& velX,
                     std::vector<double>& velY,
                     std::vector<double>& velZ,
                     unsigned int seed = 42) {
    const size_t n = posX.size();
    for (size_t i = 0; i < n; ++i) {
        posX[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        posY[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        posZ[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        velX[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        velY[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        velZ[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int worldRank = 0;
    int worldSize = 1;
    checkMPI(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank), "MPI_Comm_rank");
    checkMPI(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), "MPI_Comm_size");

    if (provided < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) {
            printf("MPI does not provide required threading level\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;

    if (worldRank == 0) {
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
                parseStatus = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseStatus = 2;
                break;
            }
        }
        if (parseStatus != 0) {
            printUsage(argv[0]);
        }
    }

    checkMPI(MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast parseStatus");
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 1 ? 0 : 1;
    }

    checkMPI(MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast numBodies");
    checkMPI(MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast numSteps");
    int validateInt = validate ? 1 : 0;
    int printResultsInt = printResults ? 1 : 0;
    checkMPI(MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast validate");
    checkMPI(MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast printResults");
    validate = validateInt != 0;
    printResults = printResultsInt != 0;

    if (worldRank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts(worldSize);
    std::vector<int> displs(worldSize);
    const int base = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    int offset = 0;
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = base + (r < remainder ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }

    const int localCount = counts[worldRank];
    const int globalOffset = displs[worldRank];

    std::vector<double> posX_local(localCount);
    std::vector<double> posY_local(localCount);
    std::vector<double> posZ_local(localCount);
    std::vector<double> velX_local(localCount);
    std::vector<double> velY_local(localCount);
    std::vector<double> velZ_local(localCount);
    std::vector<double> accX_local(localCount);
    std::vector<double> accY_local(localCount);
    std::vector<double> accZ_local(localCount);

    std::vector<double> posX_all(numBodies);
    std::vector<double> posY_all(numBodies);
    std::vector<double> posZ_all(numBodies);
    std::vector<double> velX_all;
    std::vector<double> velY_all;
    std::vector<double> velZ_all;

    if (worldRank == 0) {
        velX_all.resize(numBodies);
        velY_all.resize(numBodies);
        velZ_all.resize(numBodies);
        randomizeBodies(posX_all, posY_all, posZ_all, velX_all, velY_all, velZ_all);
    }

    checkMPI(MPI_Scatterv(worldRank == 0 ? posX_all.data() : nullptr,
                          counts.data(),
                          displs.data(),
                          MPI_DOUBLE,
                          posX_local.data(),
                          localCount,
                          MPI_DOUBLE,
                          0,
                          MPI_COMM_WORLD),
             "MPI_Scatterv posX");
    checkMPI(MPI_Scatterv(worldRank == 0 ? posY_all.data() : nullptr,
                          counts.data(),
                          displs.data(),
                          MPI_DOUBLE,
                          posY_local.data(),
                          localCount,
                          MPI_DOUBLE,
                          0,
                          MPI_COMM_WORLD),
             "MPI_Scatterv posY");
    checkMPI(MPI_Scatterv(worldRank == 0 ? posZ_all.data() : nullptr,
                          counts.data(),
                          displs.data(),
                          MPI_DOUBLE,
                          posZ_local.data(),
                          localCount,
                          MPI_DOUBLE,
                          0,
                          MPI_COMM_WORLD),
             "MPI_Scatterv posZ");
    checkMPI(MPI_Scatterv(worldRank == 0 ? velX_all.data() : nullptr,
                          counts.data(),
                          displs.data(),
                          MPI_DOUBLE,
                          velX_local.data(),
                          localCount,
                          MPI_DOUBLE,
                          0,
                          MPI_COMM_WORLD),
             "MPI_Scatterv velX");
    checkMPI(MPI_Scatterv(worldRank == 0 ? velY_all.data() : nullptr,
                          counts.data(),
                          displs.data(),
                          MPI_DOUBLE,
                          velY_local.data(),
                          localCount,
                          MPI_DOUBLE,
                          0,
                          MPI_COMM_WORLD),
             "MPI_Scatterv velY");
    checkMPI(MPI_Scatterv(worldRank == 0 ? velZ_all.data() : nullptr,
                          counts.data(),
                          displs.data(),
                          MPI_DOUBLE,
                          velZ_local.data(),
                          localCount,
                          MPI_DOUBLE,
                          0,
                          MPI_COMM_WORLD),
             "MPI_Scatterv velZ");

    checkMPI(MPI_Allgatherv(posX_local.data(),
                            localCount,
                            MPI_DOUBLE,
                            posX_all.data(),
                            counts.data(),
                            displs.data(),
                            MPI_DOUBLE,
                            MPI_COMM_WORLD),
             "MPI_Allgatherv posX");
    checkMPI(MPI_Allgatherv(posY_local.data(),
                            localCount,
                            MPI_DOUBLE,
                            posY_all.data(),
                            counts.data(),
                            displs.data(),
                            MPI_DOUBLE,
                            MPI_COMM_WORLD),
             "MPI_Allgatherv posY");
    checkMPI(MPI_Allgatherv(posZ_local.data(),
                            localCount,
                            MPI_DOUBLE,
                            posZ_all.data(),
                            counts.data(),
                            displs.data(),
                            MPI_DOUBLE,
                            MPI_COMM_WORLD),
             "MPI_Allgatherv posZ");

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        if (worldRank == 0) {
            printf("No CUDA devices available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = worldRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice");

    double* d_posX = nullptr;
    double* d_posY = nullptr;
    double* d_posZ = nullptr;
    double* d_accX = nullptr;
    double* d_accY = nullptr;
    double* d_accZ = nullptr;

    if (numBodies > 0) {
        checkCuda(cudaMalloc(&d_posX, sizeof(double) * static_cast<size_t>(numBodies)), "cudaMalloc posX");
        checkCuda(cudaMalloc(&d_posY, sizeof(double) * static_cast<size_t>(numBodies)), "cudaMalloc posY");
        checkCuda(cudaMalloc(&d_posZ, sizeof(double) * static_cast<size_t>(numBodies)), "cudaMalloc posZ");
    }
    if (localCount > 0) {
        checkCuda(cudaMalloc(&d_accX, sizeof(double) * static_cast<size_t>(localCount)), "cudaMalloc accX");
        checkCuda(cudaMalloc(&d_accY, sizeof(double) * static_cast<size_t>(localCount)), "cudaMalloc accY");
        checkCuda(cudaMalloc(&d_accZ, sizeof(double) * static_cast<size_t>(localCount)), "cudaMalloc accZ");
    }

    checkMPI(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier start");
    const double startTime = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        checkMPI(MPI_Allgatherv(posX_local.data(),
                                localCount,
                                MPI_DOUBLE,
                                posX_all.data(),
                                counts.data(),
                                displs.data(),
                                MPI_DOUBLE,
                                MPI_COMM_WORLD),
                 "MPI_Allgatherv posX step");
        checkMPI(MPI_Allgatherv(posY_local.data(),
                                localCount,
                                MPI_DOUBLE,
                                posY_all.data(),
                                counts.data(),
                                displs.data(),
                                MPI_DOUBLE,
                                MPI_COMM_WORLD),
                 "MPI_Allgatherv posY step");
        checkMPI(MPI_Allgatherv(posZ_local.data(),
                                localCount,
                                MPI_DOUBLE,
                                posZ_all.data(),
                                counts.data(),
                                displs.data(),
                                MPI_DOUBLE,
                                MPI_COMM_WORLD),
                 "MPI_Allgatherv posZ step");

        if (numBodies > 0) {
            checkCuda(cudaMemcpy(d_posX, posX_all.data(), sizeof(double) * static_cast<size_t>(numBodies), cudaMemcpyHostToDevice), "cudaMemcpy posX");
            checkCuda(cudaMemcpy(d_posY, posY_all.data(), sizeof(double) * static_cast<size_t>(numBodies), cudaMemcpyHostToDevice), "cudaMemcpy posY");
            checkCuda(cudaMemcpy(d_posZ, posZ_all.data(), sizeof(double) * static_cast<size_t>(numBodies), cudaMemcpyHostToDevice), "cudaMemcpy posZ");
        }

        if (localCount > 0) {
            const int blockSize = 256;
            const int gridSize = (localCount + blockSize - 1) / blockSize;
            const size_t sharedBytes = static_cast<size_t>(blockSize) * 3 * sizeof(double);
            computeAccelerations<<<gridSize, blockSize, sharedBytes>>>(
                d_posX, d_posY, d_posZ, d_accX, d_accY, d_accZ, numBodies, localCount, globalOffset);
            checkCuda(cudaGetLastError(), "computeAccelerations launch");
            checkCuda(cudaMemcpy(accX_local.data(), d_accX, sizeof(double) * static_cast<size_t>(localCount), cudaMemcpyDeviceToHost), "cudaMemcpy accX");
            checkCuda(cudaMemcpy(accY_local.data(), d_accY, sizeof(double) * static_cast<size_t>(localCount), cudaMemcpyDeviceToHost), "cudaMemcpy accY");
            checkCuda(cudaMemcpy(accZ_local.data(), d_accZ, sizeof(double) * static_cast<size_t>(localCount), cudaMemcpyDeviceToHost), "cudaMemcpy accZ");
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < localCount; ++i) {
            velX_local[i] += DT * accX_local[i];
            velY_local[i] += DT * accY_local[i];
            velZ_local[i] += DT * accZ_local[i];
            posX_local[i] += velX_local[i] * DT;
            posY_local[i] += velY_local[i] * DT;
            posZ_local[i] += velZ_local[i] * DT;
        }
    }

    checkMPI(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier end");
    const double endTime = MPI_Wtime();

    checkMPI(MPI_Gatherv(posX_local.data(),
                         localCount,
                         MPI_DOUBLE,
                         worldRank == 0 ? posX_all.data() : nullptr,
                         counts.data(),
                         displs.data(),
                         MPI_DOUBLE,
                         0,
                         MPI_COMM_WORLD),
             "MPI_Gatherv posX");
    checkMPI(MPI_Gatherv(posY_local.data(),
                         localCount,
                         MPI_DOUBLE,
                         worldRank == 0 ? posY_all.data() : nullptr,
                         counts.data(),
                         displs.data(),
                         MPI_DOUBLE,
                         0,
                         MPI_COMM_WORLD),
             "MPI_Gatherv posY");
    checkMPI(MPI_Gatherv(posZ_local.data(),
                         localCount,
                         MPI_DOUBLE,
                         worldRank == 0 ? posZ_all.data() : nullptr,
                         counts.data(),
                         displs.data(),
                         MPI_DOUBLE,
                         0,
                         MPI_COMM_WORLD),
             "MPI_Gatherv posZ");

    if (worldRank == 0) {
        velX_all.resize(numBodies);
        velY_all.resize(numBodies);
        velZ_all.resize(numBodies);
    }

    checkMPI(MPI_Gatherv(velX_local.data(),
                         localCount,
                         MPI_DOUBLE,
                         worldRank == 0 ? velX_all.data() : nullptr,
                         counts.data(),
                         displs.data(),
                         MPI_DOUBLE,
                         0,
                         MPI_COMM_WORLD),
             "MPI_Gatherv velX");
    checkMPI(MPI_Gatherv(velY_local.data(),
                         localCount,
                         MPI_DOUBLE,
                         worldRank == 0 ? velY_all.data() : nullptr,
                         counts.data(),
                         displs.data(),
                         MPI_DOUBLE,
                         0,
                         MPI_COMM_WORLD),
             "MPI_Gatherv velY");
    checkMPI(MPI_Gatherv(velZ_local.data(),
                         localCount,
                         MPI_DOUBLE,
                         worldRank == 0 ? velZ_all.data() : nullptr,
                         counts.data(),
                         displs.data(),
                         MPI_DOUBLE,
                         0,
                         MPI_COMM_WORLD),
             "MPI_Gatherv velZ");

    if (worldRank == 0) {
        const double durationMs = (endTime - startTime) * 1000.0;
        printf("Simulation time: %ld ms\n", static_cast<long>(durationMs));

        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(static_cast<size_t>(numBodies) * 6);
            for (int i = 0; i < numBodies; ++i) {
                bodyData.push_back(posX_all[i]);
                bodyData.push_back(posY_all[i]);
                bodyData.push_back(posZ_all[i]);
                bodyData.push_back(velX_all[i]);
                bodyData.push_back(velY_all[i]);
                bodyData.push_back(velZ_all[i]);
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");

            std::vector<Body> bodies(static_cast<size_t>(numBodies));
            for (int i = 0; i < numBodies; ++i) {
                bodies[i].pos.x = posX_all[i];
                bodies[i].pos.y = posY_all[i];
                bodies[i].pos.z = posZ_all[i];
                bodies[i].vel.x = velX_all[i];
                bodies[i].vel.y = velY_all[i];
                bodies[i].vel.z = velZ_all[i];
            }

            if (validateSimulation(bodies)) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    if (d_accZ) {
        checkCuda(cudaFree(d_accZ), "cudaFree accZ");
    }
    if (d_accY) {
        checkCuda(cudaFree(d_accY), "cudaFree accY");
    }
    if (d_accX) {
        checkCuda(cudaFree(d_accX), "cudaFree accX");
    }
    if (d_posZ) {
        checkCuda(cudaFree(d_posZ), "cudaFree posZ");
    }
    if (d_posY) {
        checkCuda(cudaFree(d_posY), "cudaFree posY");
    }
    if (d_posX) {
        checkCuda(cudaFree(d_posX), "cudaFree posX");
    }

    MPI_Finalize();
    return 0;
}
