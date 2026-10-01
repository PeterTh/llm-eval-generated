#include <algorithm>
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
constexpr int TILE = 128;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(sizeof(Body) == 6 * sizeof(double));

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
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

// Every thread owns one target body. All targets traverse source bodies in the
// same order as the serial algorithm, with each source tile shared by a block.
__global__ void advanceBodies(Body* local, const Vec3* allPositions,
                              Vec3* newPositions, int count, int n) {
    __shared__ Vec3 sources[TILE];
    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIndex < count;
    Body body;
    if (active) body = local[localIndex];
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int base = 0; base < n; base += TILE) {
        const int source = base + threadIdx.x;
        if (source < n) sources[threadIdx.x] = allPositions[source];
        __syncthreads();
        const int tileCount = min(TILE, n - base);
        if (active) {
            for (int j = 0; j < tileCount; ++j) {
                const double dx = sources[j].x - body.pos.x;
                const double dy = sources[j].y - body.pos.y;
                const double dz = sources[j].z - body.pos.z;
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
        body.vel.x += DT * fx;
        body.vel.y += DT * fy;
        body.vel.z += DT * fz;
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
        local[localIndex] = body;
        newPositions[localIndex] = body.pos;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());
    double kinetic = 0.0;
    double potential = 0.0;
    #pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (int i = 0; i < n; ++i) {
        const Body& body = bodies[i];
        kinetic += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }
    #pragma omp parallel for reduction(+:potential) schedule(static)
    for (int i = 0; i < n; ++i) {
        double partial = 0.0;
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            partial -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
        potential += partial;
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

int main(int argc, char** argv) {
    int threadSupport = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &threadSupport);
    if (threadSupport < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
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
    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) fprintf(stderr, "Body and step counts must be nonnegative\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Map ranks to GPUs by their rank within each shared-memory node.
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices), "device count");
    if (devices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % devices), "device selection");

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = (numBodies / ranks) * r + (r < numBodies % ranks ? r : numBodies % ranks);
    }
    const int count = counts[rank];
    const int first = offsets[rank];
    MPI_Datatype positionType, bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &positionType);
    MPI_Type_commit(&positionType);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    Vec3 *hostPositions = nullptr, *hostNewPositions = nullptr;
    checkCuda(cudaMallocHost(&hostPositions, sizeof(Vec3) * static_cast<size_t>(std::max(1, numBodies))), "host positions allocation");
    checkCuda(cudaMallocHost(&hostNewPositions, sizeof(Vec3) * static_cast<size_t>(std::max(1, count))), "host new positions allocation");
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) hostPositions[i] = bodies[i].pos;

    Body* deviceBodies = nullptr;
    Vec3 *devicePositions = nullptr, *deviceNewPositions = nullptr;
    checkCuda(cudaMalloc(&deviceBodies, sizeof(Body) * static_cast<size_t>(std::max(1, count))), "device bodies allocation");
    checkCuda(cudaMalloc(&devicePositions, sizeof(Vec3) * static_cast<size_t>(std::max(1, numBodies))), "device positions allocation");
    checkCuda(cudaMalloc(&deviceNewPositions, sizeof(Vec3) * static_cast<size_t>(std::max(1, count))), "device new positions allocation");
    checkCuda(cudaMemcpy(deviceBodies, bodies.data() + first, sizeof(Body) * count, cudaMemcpyHostToDevice), "initial bodies copy");
    checkCuda(cudaMemcpy(devicePositions, hostPositions, sizeof(Vec3) * numBodies, cudaMemcpyHostToDevice), "initial positions copy");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (count > 0) {
            advanceBodies<<<(count + TILE - 1) / TILE, TILE>>>(deviceBodies, devicePositions,
                                                                deviceNewPositions, count, numBodies);
            checkCuda(cudaGetLastError(), "advance kernel launch");
        }
        if (step + 1 < numSteps) {
            checkCuda(cudaMemcpy(hostNewPositions, deviceNewPositions, sizeof(Vec3) * count,
                                 cudaMemcpyDeviceToHost), "new positions copy");
            MPI_Allgatherv(hostNewPositions, count, positionType,
                           hostPositions, counts.data(), offsets.data(), positionType, MPI_COMM_WORLD);
            checkCuda(cudaMemcpy(devicePositions, hostPositions, sizeof(Vec3) * numBodies,
                                 cudaMemcpyHostToDevice), "global positions copy");
        }
    }
    checkCuda(cudaDeviceSynchronize(), "final kernel completion");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long localDurationMs = duration.count();
    long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", maxDurationMs);

    if (printResults || validate) {
        checkCuda(cudaMemcpy(bodies.data() + first, deviceBodies, sizeof(Body) * count,
                             cudaMemcpyDeviceToHost), "final bodies copy");
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : bodies.data() + first, count, bodyType,
                    rank == 0 ? bodies.data() : nullptr, counts.data(), offsets.data(), bodyType,
                    0, MPI_COMM_WORLD);
    }
    checkCuda(cudaFree(deviceBodies), "device bodies free");
    checkCuda(cudaFree(devicePositions), "device positions free");
    checkCuda(cudaFree(deviceNewPositions), "device new positions free");
    checkCuda(cudaFreeHost(hostPositions), "host positions free");
    checkCuda(cudaFreeHost(hostNewPositions), "host new positions free");
    MPI_Type_free(&positionType);
    MPI_Type_free(&bodyType);

    int result = 0;
    if (rank == 0) {
        if (printResults) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                bodyData[6 * i] = bodies[i].pos.x;
                bodyData[6 * i + 1] = bodies[i].pos.y;
                bodyData[6 * i + 2] = bodies[i].pos.z;
                bodyData[6 * i + 3] = bodies[i].vel.x;
                bodyData[6 * i + 4] = bodies[i].vel.y;
                bodyData[6 * i + 5] = bodies[i].vel.z;
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
                result = 1;
            }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
