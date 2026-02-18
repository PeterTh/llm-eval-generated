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

__global__ void compute_accel_kernel(const double* __restrict__ posX,
                                    const double* __restrict__ posY,
                                    const double* __restrict__ posZ,
                                    double* __restrict__ accX,
                                    double* __restrict__ accY,
                                    double* __restrict__ accZ,
                                    int n,
                                    int localCount,
                                    int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= localCount) {
        return;
    }
    int i = offset + idx;
    double xi = posX[i];
    double yi = posY[i];
    double zi = posZ[i];
    double Fx = 0.0;
    double Fy = 0.0;
    double Fz = 0.0;
    for (int j = 0; j < n; ++j) {
        double dx = posX[j] - xi;
        double dy = posY[j] - yi;
        double dz = posZ[j] - zi;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
    accX[idx] = Fx;
    accY[idx] = Fy;
    accZ[idx] = Fz;
}

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double kinetic = 0.0;
    const size_t n = bodies.size();

    #pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        kinetic += 0.5 * (body.vel.x * body.vel.x +
                          body.vel.y * body.vel.y +
                          body.vel.z * body.vel.z);
    }

    double potential = 0.0;
    #pragma omp parallel for reduction(+:potential) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential -= 1.0 / dist;
        }
    }

    return kinetic + potential;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

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

void computeCounts(int numBodies, int worldSize, std::vector<int>& counts, std::vector<int>& displs) {
    counts.resize(worldSize);
    displs.resize(worldSize);
    const int base = numBodies / worldSize;
    const int rem = numBodies % worldSize;
    int offset = 0;
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int numBodies = 1024;
    int numSteps = 10;
    int validate = 0;
    int printResults = 0;
    int parseStatus = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numBodies = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                numSteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseStatus = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = 1;
                break;
            }
        }
    }

    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 1 ? 1 : 0;
    }

    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", worldSize);
    }

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(rank % deviceCount), "cudaSetDevice");

    std::vector<int> counts;
    std::vector<int> displs;
    computeCounts(numBodies, worldSize, counts, displs);

    const int localCount = counts[rank];
    const int offset = displs[rank];

    std::vector<double> localPosX(localCount);
    std::vector<double> localPosY(localCount);
    std::vector<double> localPosZ(localCount);
    std::vector<double> localVelX(localCount);
    std::vector<double> localVelY(localCount);
    std::vector<double> localVelZ(localCount);

    std::vector<double> globalPosX(numBodies);
    std::vector<double> globalPosY(numBodies);
    std::vector<double> globalPosZ(numBodies);

    std::vector<double> fullPosX;
    std::vector<double> fullPosY;
    std::vector<double> fullPosZ;
    std::vector<double> fullVelX;
    std::vector<double> fullVelY;
    std::vector<double> fullVelZ;

    if (rank == 0) {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);

        fullPosX.resize(numBodies);
        fullPosY.resize(numBodies);
        fullPosZ.resize(numBodies);
        fullVelX.resize(numBodies);
        fullVelY.resize(numBodies);
        fullVelZ.resize(numBodies);

        for (int i = 0; i < numBodies; ++i) {
            fullPosX[i] = bodies[i].pos.x;
            fullPosY[i] = bodies[i].pos.y;
            fullPosZ[i] = bodies[i].pos.z;
            fullVelX[i] = bodies[i].vel.x;
            fullVelY[i] = bodies[i].vel.y;
            fullVelZ[i] = bodies[i].vel.z;
        }
    }

    MPI_Scatterv(fullPosX.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 localPosX.data(), localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(fullPosY.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 localPosY.data(), localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(fullPosZ.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 localPosZ.data(), localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(fullVelX.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 localVelX.data(), localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(fullVelY.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 localVelY.data(), localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(fullVelZ.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 localVelZ.data(), localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* d_posX = nullptr;
    double* d_posY = nullptr;
    double* d_posZ = nullptr;
    double* d_accX = nullptr;
    double* d_accY = nullptr;
    double* d_accZ = nullptr;

    if (numBodies > 0) {
        checkCuda(cudaMalloc(&d_posX, static_cast<size_t>(numBodies) * sizeof(double)), "cudaMalloc posX");
        checkCuda(cudaMalloc(&d_posY, static_cast<size_t>(numBodies) * sizeof(double)), "cudaMalloc posY");
        checkCuda(cudaMalloc(&d_posZ, static_cast<size_t>(numBodies) * sizeof(double)), "cudaMalloc posZ");
    }
    if (localCount > 0) {
        checkCuda(cudaMalloc(&d_accX, static_cast<size_t>(localCount) * sizeof(double)), "cudaMalloc accX");
        checkCuda(cudaMalloc(&d_accY, static_cast<size_t>(localCount) * sizeof(double)), "cudaMalloc accY");
        checkCuda(cudaMalloc(&d_accZ, static_cast<size_t>(localCount) * sizeof(double)), "cudaMalloc accZ");
    }

    std::vector<double> localAccX(localCount);
    std::vector<double> localAccY(localCount);
    std::vector<double> localAccZ(localCount);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        MPI_Allgatherv(localPosX.data(), localCount, MPI_DOUBLE,
                       globalPosX.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(localPosY.data(), localCount, MPI_DOUBLE,
                       globalPosY.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(localPosZ.data(), localCount, MPI_DOUBLE,
                       globalPosZ.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        if (numBodies > 0) {
            checkCuda(cudaMemcpy(d_posX, globalPosX.data(), static_cast<size_t>(numBodies) * sizeof(double), cudaMemcpyHostToDevice), "copy posX");
            checkCuda(cudaMemcpy(d_posY, globalPosY.data(), static_cast<size_t>(numBodies) * sizeof(double), cudaMemcpyHostToDevice), "copy posY");
            checkCuda(cudaMemcpy(d_posZ, globalPosZ.data(), static_cast<size_t>(numBodies) * sizeof(double), cudaMemcpyHostToDevice), "copy posZ");
        }

        if (localCount > 0) {
            const int threads = 256;
            const int blocks = (localCount + threads - 1) / threads;
            compute_accel_kernel<<<blocks, threads>>>(d_posX, d_posY, d_posZ,
                                                      d_accX, d_accY, d_accZ,
                                                      numBodies, localCount, offset);
            checkCuda(cudaGetLastError(), "kernel launch");
            checkCuda(cudaDeviceSynchronize(), "kernel sync");
            checkCuda(cudaMemcpy(localAccX.data(), d_accX, static_cast<size_t>(localCount) * sizeof(double), cudaMemcpyDeviceToHost), "copy accX");
            checkCuda(cudaMemcpy(localAccY.data(), d_accY, static_cast<size_t>(localCount) * sizeof(double), cudaMemcpyDeviceToHost), "copy accY");
            checkCuda(cudaMemcpy(localAccZ.data(), d_accZ, static_cast<size_t>(localCount) * sizeof(double), cudaMemcpyDeviceToHost), "copy accZ");
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < localCount; ++i) {
            localVelX[i] += DT * localAccX[i];
            localVelY[i] += DT * localAccY[i];
            localVelZ[i] += DT * localAccZ[i];
            localPosX[i] += localVelX[i] * DT;
            localPosY[i] += localVelY[i] * DT;
            localPosZ[i] += localVelZ[i] * DT;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    if (rank == 0) {
        long long durationMs = static_cast<long long>((end - start) * 1000.0);
        printf("Simulation time: %lld ms\n", durationMs);
    }

    if (printResults || validate) {
        if (rank == 0) {
            fullPosX.resize(numBodies);
            fullPosY.resize(numBodies);
            fullPosZ.resize(numBodies);
            fullVelX.resize(numBodies);
            fullVelY.resize(numBodies);
            fullVelZ.resize(numBodies);
        }

        MPI_Gatherv(localPosX.data(), localCount, MPI_DOUBLE,
                    rank == 0 ? fullPosX.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(localPosY.data(), localCount, MPI_DOUBLE,
                    rank == 0 ? fullPosY.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(localPosZ.data(), localCount, MPI_DOUBLE,
                    rank == 0 ? fullPosZ.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(localVelX.data(), localCount, MPI_DOUBLE,
                    rank == 0 ? fullVelX.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(localVelY.data(), localCount, MPI_DOUBLE,
                    rank == 0 ? fullVelY.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(localVelZ.data(), localCount, MPI_DOUBLE,
                    rank == 0 ? fullVelZ.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int validationStatus = 0;
    if (rank == 0) {
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(static_cast<size_t>(numBodies) * 6u);
            for (int i = 0; i < numBodies; ++i) {
                bodyData.push_back(fullPosX[i]);
                bodyData.push_back(fullPosY[i]);
                bodyData.push_back(fullPosZ[i]);
                bodyData.push_back(fullVelX[i]);
                bodyData.push_back(fullVelY[i]);
                bodyData.push_back(fullVelZ[i]);
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");
            std::vector<Body> bodies(numBodies);

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                bodies[i].pos.x = fullPosX[i];
                bodies[i].pos.y = fullPosY[i];
                bodies[i].pos.z = fullPosZ[i];
                bodies[i].vel.x = fullVelX[i];
                bodies[i].vel.y = fullVelY[i];
                bodies[i].vel.z = fullVelZ[i];
            }

            if (validateSimulation(bodies)) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
                validationStatus = 0;
            } else {
                printf("Validation: FAILED\n");
                validationStatus = 1;
            }
        }
    }

    if (validate) {
        MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (d_posX) {
        cudaFree(d_posX);
    }
    if (d_posY) {
        cudaFree(d_posY);
    }
    if (d_posZ) {
        cudaFree(d_posZ);
    }
    if (d_accX) {
        cudaFree(d_accX);
    }
    if (d_accY) {
        cudaFree(d_accY);
    }
    if (d_accZ) {
        cudaFree(d_accZ);
    }

    MPI_Finalize();
    if (validate) {
        return validationStatus;
    }
    return 0;
}
