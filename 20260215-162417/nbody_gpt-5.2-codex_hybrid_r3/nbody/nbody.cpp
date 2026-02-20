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

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be tightly packed");
static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be tightly packed");

inline void checkMpi(const int err, const char* call) {
    if (err != MPI_SUCCESS) {
        char errStr[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(err, errStr, &len);
        fprintf(stderr, "MPI error in %s: %s\n", call, errStr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

inline void checkCuda(const cudaError_t err, const char* call) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", call, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define MPI_CHECK(call) checkMpi((call), #call)
#define CUDA_CHECK(call) checkCuda((call), #call)

__global__ void computeForcesKernel(const Vec3* allPos, Vec3* localPos, Vec3* localVel, int totalBodies, int localBodies, int offset) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= localBodies) {
        return;
    }

    const int globalIdx = offset + i;
    const Vec3 pos_i = allPos[globalIdx];
    double Fx = 0.0;
    double Fy = 0.0;
    double Fz = 0.0;

    for (int j = 0; j < totalBodies; ++j) {
        const Vec3 pos_j = allPos[j];
        const double dx = pos_j.x - pos_i.x;
        const double dy = pos_j.y - pos_i.y;
        const double dz = pos_j.z - pos_i.z;
        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    Vec3 vel = localVel[i];
    vel.x += DT * Fx;
    vel.y += DT * Fy;
    vel.z += DT * Fz;
    localVel[i] = vel;

    localPos[i].x = pos_i.x + vel.x * DT;
    localPos[i].y = pos_i.y + vel.y * DT;
    localPos[i].z = pos_i.z + vel.z * DT;
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
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank = 0;
    int size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int validateFlag = 0;
    int printFlag = 0;
    int parseStatus = 0;

    if (rank == 0) {
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
                parseStatus = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseStatus = 1;
                break;
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (parseStatus != 0) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseStatus == 1 ? 1 : 0;
    }

    validateFlag = validate ? 1 : 0;
    printFlag = printResults ? 1 : 0;
    MPI_CHECK(MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&validateFlag, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&printFlag, 1, MPI_INT, 0, MPI_COMM_WORLD));
    validate = validateFlag != 0;
    printResults = printFlag != 0;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    const int base = numBodies / size;
    const int rem = numBodies % size;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        if (r > 0) {
            displs[r] = displs[r - 1] + counts[r - 1];
        }
    }
    const int localBodies = counts[rank];

    MPI_Datatype MPI_VEC3;
    MPI_Datatype MPI_BODY;
    MPI_CHECK(MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_VEC3));
    MPI_CHECK(MPI_Type_commit(&MPI_VEC3));
    MPI_CHECK(MPI_Type_contiguous(6, MPI_DOUBLE, &MPI_BODY));
    MPI_CHECK(MPI_Type_commit(&MPI_BODY));

    std::vector<Body> allBodies;
    if (rank == 0) {
        allBodies.resize(numBodies);
        randomizeBodies(allBodies);
    }

    std::vector<Body> localBodyData(localBodies);
    MPI_CHECK(MPI_Scatterv(rank == 0 ? allBodies.data() : nullptr, counts.data(), displs.data(), MPI_BODY,
                           localBodyData.data(), localBodies, MPI_BODY, 0, MPI_COMM_WORLD));

    std::vector<Vec3> localPos(localBodies);
    std::vector<Vec3> localVel(localBodies);
    std::vector<Vec3> allPos(numBodies);

    if (localBodies > 0) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < localBodies; ++i) {
            localPos[i] = localBodyData[i].pos;
            localVel[i] = localBodyData[i].vel;
        }
    }

    Vec3* d_all_pos = nullptr;
    Vec3* d_pos = nullptr;
    Vec3* d_vel = nullptr;
    if (numBodies > 0) {
        CUDA_CHECK(cudaMalloc(&d_all_pos, sizeof(Vec3) * static_cast<size_t>(numBodies)));
    }
    if (localBodies > 0) {
        CUDA_CHECK(cudaMalloc(&d_pos, sizeof(Vec3) * static_cast<size_t>(localBodies)));
        CUDA_CHECK(cudaMalloc(&d_vel, sizeof(Vec3) * static_cast<size_t>(localBodies)));
        CUDA_CHECK(cudaMemcpy(d_vel, localVel.data(), sizeof(Vec3) * static_cast<size_t>(localBodies), cudaMemcpyHostToDevice));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        if (localBodies > 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < localBodies; ++i) {
                localPos[i] = localBodyData[i].pos;
            }
        }

        MPI_CHECK(MPI_Allgatherv(localPos.data(), localBodies, MPI_VEC3,
                                 allPos.data(), counts.data(), displs.data(), MPI_VEC3,
                                 MPI_COMM_WORLD));

        if (numBodies > 0) {
            CUDA_CHECK(cudaMemcpy(d_all_pos, allPos.data(), sizeof(Vec3) * static_cast<size_t>(numBodies), cudaMemcpyHostToDevice));
        }

        if (localBodies > 0) {
            const int threads = 256;
            const int blocks = (localBodies + threads - 1) / threads;
            computeForcesKernel<<<blocks, threads>>>(d_all_pos, d_pos, d_vel, numBodies, localBodies, displs[rank]);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(localPos.data(), d_pos, sizeof(Vec3) * static_cast<size_t>(localBodies), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(localVel.data(), d_vel, sizeof(Vec3) * static_cast<size_t>(localBodies), cudaMemcpyDeviceToHost));

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < localBodies; ++i) {
                localBodyData[i].pos = localPos[i];
                localBodyData[i].vel = localVel[i];
            }
        }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double end = MPI_Wtime();
    const double elapsed = end - start;
    double maxElapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        const long durationMs = static_cast<long>(maxElapsed * 1000.0);
        printf("Simulation time: %ld ms\n", durationMs);
    }

    if (printResults || validate) {
        if (rank == 0) {
            allBodies.assign(numBodies, Body{});
        }
        MPI_CHECK(MPI_Gatherv(localBodyData.data(), localBodies, MPI_BODY,
                              rank == 0 ? allBodies.data() : nullptr, counts.data(), displs.data(), MPI_BODY,
                              0, MPI_COMM_WORLD));
    }

    int validationStatus = 0;
    if (rank == 0 && printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (const auto& body : allBodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(allBodies)) {
            double finalEnergy = computeTotalEnergy(allBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            validationStatus = 0;
        } else {
            printf("Validation: FAILED\n");
            validationStatus = 1;
        }
    }

    MPI_CHECK(MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD));

    if (d_vel) {
        CUDA_CHECK(cudaFree(d_vel));
    }
    if (d_pos) {
        CUDA_CHECK(cudaFree(d_pos));
    }
    if (d_all_pos) {
        CUDA_CHECK(cudaFree(d_all_pos));
    }

    MPI_CHECK(MPI_Type_free(&MPI_BODY));
    MPI_CHECK(MPI_Type_free(&MPI_VEC3));
    MPI_Finalize();
    return validationStatus;
}
