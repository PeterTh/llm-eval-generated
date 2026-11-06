#include <mpi.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 128;

struct Vec3 {
    double x, y, z;
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Each thread owns one target body. All threads, including those outside the
// local range, participate in loading the source tiles and in the barriers.
__global__ void advanceBodies(const Vec3* positions, Vec3* nextPositions,
                              Vec3* velocities, int n, int first, int count) {
    __shared__ Vec3 tile[BLOCK_SIZE];
    const size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = local < static_cast<size_t>(count);
    Vec3 p = {0.0, 0.0, 0.0};
    if (active) p = positions[first + local];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (size_t base = 0; base < static_cast<size_t>(n); base += BLOCK_SIZE) {
        const size_t j = base + threadIdx.x;
        if (j < static_cast<size_t>(n)) tile[threadIdx.x] = positions[j];
        __syncthreads();
        const int tileCount = static_cast<int>(min(static_cast<size_t>(BLOCK_SIZE), static_cast<size_t>(n) - base));
        if (active) {
            for (int k = 0; k < tileCount; ++k) {
                const double dx = tile[k].x - p.x;
                const double dy = tile[k].y - p.y;
                const double dz = tile[k].z - p.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }
    if (active) {
        Vec3 v = velocities[local];
        v.x += DT * fx;
        v.y += DT * fy;
        v.z += DT * fz;
        velocities[local] = v;
        p.x += v.x * DT;
        p.y += v.y * DT;
        p.z += v.z * DT;
        nextPositions[local] = p;
    }
}

static void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

static double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Vec3 v = bodies[i].vel;
        energy += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }
    #pragma omp parallel for reduction(+:energy) schedule(dynamic, 32)
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

static bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        if (std::abs(body.pos.x) > 1e6 || std::abs(body.pos.y) > 1e6 || std::abs(body.pos.z) > 1e6) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > 1e6 || std::abs(body.vel.y) > 1e6 || std::abs(body.vel.z) > 1e6) {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

static void printUsage(const char* progName) {
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
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
    if (numBodies < 0) {
        if (rank == 0) fprintf(stderr, "Number of bodies must be nonnegative\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0, offset = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks ? 1 : 0);
        offsets[r] = offset;
        offset += counts[r];
    }
    const int first = offsets[rank], count = counts[rank];
    MPI_Datatype vecType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vecType);
    MPI_Type_commit(&vecType);

    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    Vec3* hostPositions = nullptr;
    cudaCheck(cudaMallocHost(&hostPositions, sizeof(Vec3) * (numBodies > 0 ? numBodies : 1)), "allocate host positions");
    std::vector<Vec3> localVelocities(count);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) hostPositions[i] = bodies[i].pos;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) localVelocities[i] = bodies[first + i].vel;

    Vec3 *positions = nullptr, *nextPositions = nullptr, *velocities = nullptr;
    const size_t positionsBytes = sizeof(Vec3) * static_cast<size_t>(numBodies);
    const size_t localBytes = sizeof(Vec3) * static_cast<size_t>(count);
    cudaCheck(cudaMalloc(&positions, positionsBytes ? positionsBytes : sizeof(Vec3)), "allocate positions");
    cudaCheck(cudaMalloc(&nextPositions, localBytes ? localBytes : sizeof(Vec3)), "allocate next positions");
    cudaCheck(cudaMalloc(&velocities, localBytes ? localBytes : sizeof(Vec3)), "allocate velocities");
    if (positionsBytes) cudaCheck(cudaMemcpy(positions, hostPositions, positionsBytes, cudaMemcpyHostToDevice), "copy initial positions");
    if (localBytes) cudaCheck(cudaMemcpy(velocities, localVelocities.data(), localBytes, cudaMemcpyHostToDevice), "copy initial velocities");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (count) {
            advanceBodies<<<1 + (count - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(positions, nextPositions, velocities, numBodies, first, count);
            cudaCheck(cudaGetLastError(), "launch force kernel");
        }
        if (ranks == 1) {
            std::swap(positions, nextPositions);
        } else {
            if (count) cudaCheck(cudaMemcpy(hostPositions + first, nextPositions, localBytes, cudaMemcpyDeviceToHost), "copy updated positions");
            MPI_Allgatherv(MPI_IN_PLACE, 0, vecType, hostPositions, counts.data(), offsets.data(), vecType, MPI_COMM_WORLD);
            if (positionsBytes) cudaCheck(cudaMemcpy(positions, hostPositions, positionsBytes, cudaMemcpyHostToDevice), "copy gathered positions");
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "finish simulation");
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Simulation time: %ld ms\n", static_cast<long>(duration.count()));
    }

    int status = 0;
    if (validate || printResults) {
        if (ranks == 1 && positionsBytes) cudaCheck(cudaMemcpy(hostPositions, positions, positionsBytes, cudaMemcpyDeviceToHost), "copy final positions");
        if (localBytes) cudaCheck(cudaMemcpy(localVelocities.data(), velocities, localBytes, cudaMemcpyDeviceToHost), "copy final velocities");
        std::vector<Vec3> allVelocities(rank == 0 ? numBodies : 0);
        MPI_Gatherv(localVelocities.data(), count, vecType, allVelocities.data(), counts.data(), offsets.data(), vecType, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                bodies[i].pos = hostPositions[i];
                bodies[i].vel = allVelocities[i];
            }
            if (printResults) {
                std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < numBodies; ++i) {
                    const size_t j = static_cast<size_t>(i) * 6;
                    bodyData[j] = bodies[i].pos.x;
                    bodyData[j + 1] = bodies[i].pos.y;
                    bodyData[j + 2] = bodies[i].pos.z;
                    bodyData[j + 3] = bodies[i].vel.x;
                    bodyData[j + 4] = bodies[i].vel.y;
                    bodyData[j + 5] = bodies[i].vel.z;
                }
                print_results(bodyData, "Bodies");
            }
            if (validate) {
                printf("Validating simulation results...\n");
                if (validateSimulation(bodies)) {
                    printf("Final energy: %.6f\n", computeTotalEnergy(bodies));
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    status = 1;
                }
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaCheck(cudaFree(positions), "free positions");
    cudaCheck(cudaFree(nextPositions), "free next positions");
    cudaCheck(cudaFree(velocities), "free velocities");
    cudaCheck(cudaFreeHost(hostPositions), "free host positions");
    MPI_Type_free(&vecType);
    MPI_Finalize();
    return status;
}
