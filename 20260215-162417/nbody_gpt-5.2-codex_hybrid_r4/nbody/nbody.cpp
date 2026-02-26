#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
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

static_assert(std::is_standard_layout<Body>::value, "Body must be standard layout");
static_assert(sizeof(Body) == sizeof(double) * 6, "Body must be tightly packed");

void checkCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", message, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void computeForcesKernel(const double* posX, const double* posY, const double* posZ,
                                    double* velX, double* velY, double* velZ,
                                    int nGlobal, int localN, int offset) {
    const int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIdx >= localN) {
        return;
    }

    const int i = offset + localIdx;
    const double ix = posX[i];
    const double iy = posY[i];
    const double iz = posZ[i];
    double Fx = 0.0;
    double Fy = 0.0;
    double Fz = 0.0;

    for (int j = 0; j < nGlobal; ++j) {
        const double dx = posX[j] - ix;
        const double dy = posY[j] - iy;
        const double dz = posZ[j] - iz;
        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    velX[localIdx] += DT * Fx;
    velY[localIdx] += DT * Fy;
    velZ[localIdx] += DT * Fz;
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

void computeForces(std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    
    for (size_t i = 0; i < n; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        bodies[i].vel.x += DT * Fx;
        bodies[i].vel.y += DT * Fy;
        bodies[i].vel.z += DT * Fz;
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < bodies.size(); ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
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
    #pragma omp parallel for reduction(+:energy) schedule(static)
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int numBodies = 1024;
    int numSteps = 10;
    int validateFlag = 0;
    int printResultsFlag = 0;
    int showHelp = 0;
    int parseOk = 1;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numBodies = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                numSteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validateFlag = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResultsFlag = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseOk = 0;
                break;
            }
        }

        if (showHelp) {
            printUsage(argv[0]);
        }
    }

    MPI_Bcast(&parseOk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&showHelp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parseOk || showHelp) {
        MPI_Finalize();
        return parseOk ? 0 : 1;
    }

    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validateFlag ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", worldSize);
    }

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount < 1) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices available.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int deviceId = rank % deviceCount;
    checkCuda(cudaSetDevice(deviceId), "cudaSetDevice");

    std::vector<Body> globalBodies(numBodies);
    if (rank == 0) {
        randomizeBodies(globalBodies);
    }

    MPI_Bcast(reinterpret_cast<double*>(globalBodies.data()), numBodies * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> countsBodies(worldSize);
    std::vector<int> displsBodies(worldSize);
    const int base = numBodies / worldSize;
    const int rem = numBodies % worldSize;
    int currentOffset = 0;
    for (int r = 0; r < worldSize; ++r) {
        countsBodies[r] = base + (r < rem ? 1 : 0);
        displsBodies[r] = currentOffset;
        currentOffset += countsBodies[r];
    }

    const int localN = countsBodies[rank];
    const int bodyOffset = displsBodies[rank];

    std::vector<int> countsDoubles(worldSize);
    std::vector<int> displsDoubles(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        countsDoubles[r] = countsBodies[r] * 6;
        displsDoubles[r] = displsBodies[r] * 6;
    }

    std::vector<Body> localBodies(localN);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < localN; ++i) {
        localBodies[i] = globalBodies[bodyOffset + i];
    }

    std::vector<double> allPosX(numBodies);
    std::vector<double> allPosY(numBodies);
    std::vector<double> allPosZ(numBodies);
    std::vector<double> localVelX(localN);
    std::vector<double> localVelY(localN);
    std::vector<double> localVelZ(localN);

    double* d_posX = nullptr;
    double* d_posY = nullptr;
    double* d_posZ = nullptr;
    double* d_velX = nullptr;
    double* d_velY = nullptr;
    double* d_velZ = nullptr;

    checkCuda(cudaMalloc(&d_posX, numBodies * sizeof(double)), "cudaMalloc posX");
    checkCuda(cudaMalloc(&d_posY, numBodies * sizeof(double)), "cudaMalloc posY");
    checkCuda(cudaMalloc(&d_posZ, numBodies * sizeof(double)), "cudaMalloc posZ");

    const size_t localAlloc = static_cast<size_t>(std::max(1, localN));
    checkCuda(cudaMalloc(&d_velX, localAlloc * sizeof(double)), "cudaMalloc velX");
    checkCuda(cudaMalloc(&d_velY, localAlloc * sizeof(double)), "cudaMalloc velY");
    checkCuda(cudaMalloc(&d_velZ, localAlloc * sizeof(double)), "cudaMalloc velZ");

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    const int blockSize = 256;
    for (int step = 0; step < numSteps; ++step) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            allPosX[i] = globalBodies[i].pos.x;
            allPosY[i] = globalBodies[i].pos.y;
            allPosZ[i] = globalBodies[i].pos.z;
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < localN; ++i) {
            localVelX[i] = localBodies[i].vel.x;
            localVelY[i] = localBodies[i].vel.y;
            localVelZ[i] = localBodies[i].vel.z;
        }

        checkCuda(cudaMemcpy(d_posX, allPosX.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice), "copy posX");
        checkCuda(cudaMemcpy(d_posY, allPosY.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice), "copy posY");
        checkCuda(cudaMemcpy(d_posZ, allPosZ.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice), "copy posZ");

        if (localN > 0) {
            checkCuda(cudaMemcpy(d_velX, localVelX.data(), localN * sizeof(double), cudaMemcpyHostToDevice), "copy velX");
            checkCuda(cudaMemcpy(d_velY, localVelY.data(), localN * sizeof(double), cudaMemcpyHostToDevice), "copy velY");
            checkCuda(cudaMemcpy(d_velZ, localVelZ.data(), localN * sizeof(double), cudaMemcpyHostToDevice), "copy velZ");

            const int gridSize = (localN + blockSize - 1) / blockSize;
            computeForcesKernel<<<gridSize, blockSize>>>(d_posX, d_posY, d_posZ,
                                                         d_velX, d_velY, d_velZ,
                                                         numBodies, localN, bodyOffset);
            checkCuda(cudaGetLastError(), "kernel launch");
            checkCuda(cudaDeviceSynchronize(), "kernel sync");

            checkCuda(cudaMemcpy(localVelX.data(), d_velX, localN * sizeof(double), cudaMemcpyDeviceToHost), "copy velX back");
            checkCuda(cudaMemcpy(localVelY.data(), d_velY, localN * sizeof(double), cudaMemcpyDeviceToHost), "copy velY back");
            checkCuda(cudaMemcpy(localVelZ.data(), d_velZ, localN * sizeof(double), cudaMemcpyDeviceToHost), "copy velZ back");
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < localN; ++i) {
            localBodies[i].vel.x = localVelX[i];
            localBodies[i].vel.y = localVelY[i];
            localBodies[i].vel.z = localVelZ[i];
            localBodies[i].pos.x += localBodies[i].vel.x * DT;
            localBodies[i].pos.y += localBodies[i].vel.y * DT;
            localBodies[i].pos.z += localBodies[i].vel.z * DT;
        }

        const double* sendbuf = localN > 0 ? reinterpret_cast<double*>(localBodies.data()) : nullptr;
        MPI_Allgatherv(sendbuf, localN * 6, MPI_DOUBLE,
                       reinterpret_cast<double*>(globalBodies.data()),
                       countsDoubles.data(), displsDoubles.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double localDuration = end - start;
    double maxDuration = 0.0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %.0f ms\n", maxDuration * 1000.0);
    }

    if (printResultsFlag && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : globalBodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    int validationOk = 1;
    if (validateFlag && rank == 0) {
        printf("Validating simulation results...\n");
        if (validateSimulation(globalBodies)) {
            double finalEnergy = computeTotalEnergy(globalBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            validationOk = 1;
        } else {
            printf("Validation: FAILED\n");
            validationOk = 0;
        }
    }

    MPI_Bcast(&validationOk, 1, MPI_INT, 0, MPI_COMM_WORLD);

    checkCuda(cudaFree(d_posX), "cudaFree posX");
    checkCuda(cudaFree(d_posY), "cudaFree posY");
    checkCuda(cudaFree(d_posZ), "cudaFree posZ");
    checkCuda(cudaFree(d_velX), "cudaFree velX");
    checkCuda(cudaFree(d_velY), "cudaFree velY");
    checkCuda(cudaFree(d_velZ), "cudaFree velZ");

    MPI_Finalize();
    return validationOk ? 0 : 1;
}
