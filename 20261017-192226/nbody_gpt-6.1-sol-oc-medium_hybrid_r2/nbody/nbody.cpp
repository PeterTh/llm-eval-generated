#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>

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

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

constexpr int TILE = 128;

// One thread owns one target. All threads cooperatively load source tiles,
// retaining the original ascending-j summation order (including self terms).
// The separate output array prevents integration from racing with force reads.
__global__ void advanceBodies(const Vec3* __restrict__ positions,
                              Vec3* __restrict__ velocities,
                              Vec3* __restrict__ nextPositions,
                              int n, int offset, int localCount) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < localCount;
    const Vec3 p = active ? positions[offset + local] : Vec3();
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += TILE) {
        const int source = base + threadIdx.x;
        if (source < n) {
            const Vec3 q = positions[source];
            sx[threadIdx.x] = q.x;
            sy[threadIdx.x] = q.y;
            sz[threadIdx.x] = q.z;
        }
        __syncthreads();
        if (active) {
            const int count = min(TILE, n - base);
            for (int j = 0; j < count; ++j) {
                const double dx = sx[j] - p.x;
                const double dy = sy[j] - p.y;
                const double dz = sz[j] - p.z;
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
        nextPositions[local] = Vec3(p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT);
    }
}

double computeTotalEnergy(const Vec3* positions, const Vec3* velocities,
                          int n, int offset, int localCount) {
    double energy = 0.0;
    // Distribute validation work over ranks and dynamically balance the
    // triangular potential-energy loop over each rank's OpenMP team.
    #pragma omp parallel for reduction(+:energy) schedule(dynamic, 16)
    for (int local = 0; local < localCount; ++local) {
        const int i = offset + local;
        const Vec3 v = velocities[local];
        double e = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
        for (int j = i + 1; j < n; ++j) {
            const double dx = positions[j].x - positions[i].x;
            const double dy = positions[j].y - positions[i].y;
            const double dz = positions[j].z - positions[i].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            e -= 1.0 / dist;
        }
        energy += e;
    }
    
    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    int invalid = 0;
    #pragma omp parallel for reduction(|:invalid) schedule(static)
    for (size_t i = 0; i < bodies.size(); ++i) {
        const auto& body = bodies[i];
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            invalid |= 1;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            invalid |= 2;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            invalid |= 4;
        }
    }
    if (invalid & 1) printf("Validation failed: found NaN or Inf value in body state\n");
    if (invalid & 2) printf("Validation failed: body position exceeds reasonable bounds\n");
    if (invalid & 4) printf("Validation failed: body velocity exceeds reasonable bounds\n");
    return invalid == 0;
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
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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

    // Map ranks to GPUs by node-local rank. This also supports launchers that
    // expose only a single GPU to each rank through CUDA_VISIBLE_DEVICES.
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(node, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&node);

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be packed");
    MPI_Datatype vecType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vecType);
    MPI_Type_commit(&vecType);
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
    }
    const int localCount = counts[rank], offset = offsets[rank];
    const size_t globalBytes = std::max(size_t(1), size_t(numBodies)) * sizeof(Vec3);
    const size_t localBytes = std::max(size_t(1), size_t(localCount)) * sizeof(Vec3);
    Vec3 *positions = nullptr, *velocities = nullptr;
    Vec3 *dPositions = nullptr, *dVelocities = nullptr, *dNext = nullptr;
    // Pinned staging works with ordinary MPI; CUDA-aware MPI is not required.
    CUDA_CHECK(cudaMallocHost(&positions, globalBytes));
    CUDA_CHECK(cudaMallocHost(&velocities, localBytes));
    CUDA_CHECK(cudaMalloc(&dPositions, globalBytes));
    CUDA_CHECK(cudaMalloc(&dVelocities, localBytes));
    CUDA_CHECK(cudaMalloc(&dNext, localBytes));
    std::vector<Body> bodies;
    std::vector<Vec3> allVelocities;
    if (rank == 0) {
        bodies.resize(numBodies);
        allVelocities.resize(numBodies);
        randomizeBodies(bodies); // Preserve the exact rand_r sequence.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            positions[i] = bodies[i].pos;
            allVelocities[i] = bodies[i].vel;
        }
    }
    MPI_Bcast(positions, numBodies, vecType, 0, MPI_COMM_WORLD);
    MPI_Scatterv(allVelocities.data(), counts.data(), offsets.data(), vecType,
                 velocities, localCount, vecType, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(dPositions, positions, size_t(numBodies) * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVelocities, velocities, size_t(localCount) * sizeof(Vec3), cudaMemcpyHostToDevice));
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount) {
            advanceBodies<<<(localCount + TILE - 1) / TILE, TILE>>>(
                dPositions, dVelocities, dNext, numBodies, offset, localCount);
            CUDA_CHECK(cudaGetLastError());
        }
        if (step + 1 < numSteps) {
            if (ranks == 1) {
                // Avoid all PCIe traffic during single-rank time stepping.
                std::swap(dPositions, dNext);
            } else {
                CUDA_CHECK(cudaMemcpy(positions + offset, dNext,
                    size_t(localCount) * sizeof(Vec3), cudaMemcpyDeviceToHost));
                MPI_Allgatherv(MPI_IN_PLACE, 0, vecType, positions,
                    counts.data(), offsets.data(), vecType, MPI_COMM_WORLD);
                CUDA_CHECK(cudaMemcpy(dPositions, positions,
                    size_t(numBodies) * sizeof(Vec3), cudaMemcpyHostToDevice));
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(duration * 1000.0));

    double finalEnergy = 0.0;
    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(positions + offset,
            numSteps ? dNext : dPositions + offset,
            size_t(localCount) * sizeof(Vec3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velocities, dVelocities,
            size_t(localCount) * sizeof(Vec3), cudaMemcpyDeviceToHost));
        // Validation needs all final positions on every rank for energy.
        if (validate) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, vecType, positions,
                counts.data(), offsets.data(), vecType, MPI_COMM_WORLD);
            const double energy = computeTotalEnergy(positions, velocities, numBodies, offset, localCount);
            MPI_Reduce(&energy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : positions + offset,
                localCount, vecType, positions, counts.data(), offsets.data(), vecType, 0, MPI_COMM_WORLD);
        }
        MPI_Gatherv(velocities, localCount, vecType, allVelocities.data(),
            counts.data(), offsets.data(), vecType, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                bodies[i].pos = positions[i];
                bodies[i].vel = allVelocities[i];
            }
        }
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData(size_t(numBodies) * 6);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const auto& body = bodies[i];
            bodyData[6 * size_t(i)] = body.pos.x;
            bodyData[6 * size_t(i) + 1] = body.pos.y;
            bodyData[6 * size_t(i) + 2] = body.pos.z;
            bodyData[6 * size_t(i) + 3] = body.vel.x;
            bodyData[6 * size_t(i) + 4] = body.vel.y;
            bodyData[6 * size_t(i) + 5] = body.vel.z;
        }
        print_results(bodyData, "Bodies");
    }
    
    // Validation: check that simulation produces finite, reasonable values
    int status = 0;
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(dPositions));
    CUDA_CHECK(cudaFree(dVelocities));
    CUDA_CHECK(cudaFree(dNext));
    CUDA_CHECK(cudaFreeHost(positions));
    CUDA_CHECK(cudaFreeHost(velocities));
    MPI_Type_free(&vecType);
    MPI_Finalize();
    return status;
}
