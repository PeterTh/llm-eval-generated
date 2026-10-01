#include <algorithm>
#include <cstddef>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// All MPI calls run on the main thread (MPI_THREAD_FUNNELED).
void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

constexpr int TILE = 128;

// Each thread owns one target; shared source tiles are reused by the block.
// Accumulate sources in their original order, without approximate reciprocal sqrt.
__global__ void computeForces(const double3* positions, double3* velocities,
                              int n, int first, int count) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < count;
    const double3 p = active ? positions[first + i] : make_double3(0, 0, 0);
    double fx = 0, fy = 0, fz = 0;
    for (size_t base = 0; base < static_cast<size_t>(n); base += TILE) {
        const size_t j = base + threadIdx.x;
        if (j < static_cast<size_t>(n)) {
            const double3 q = positions[j];
            sx[threadIdx.x] = q.x;
            sy[threadIdx.x] = q.y;
            sz[threadIdx.x] = q.z;
        }
        __syncthreads();
        const int length = static_cast<int>(min(static_cast<size_t>(TILE),
                                                static_cast<size_t>(n) - base));
        if (active) {
            #pragma unroll 8
            for (int k = 0; k < length; ++k) {
                const double dx = sx[k] - p.x;
                const double dy = sy[k] - p.y;
                const double dz = sz[k] - p.z;
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
        double3 v = velocities[i];
        v.x += DT * fx;
        v.y += DT * fy;
        v.z += DT * fz;
        velocities[i] = v;
    }
}

// A separate kernel guarantees every force used the same pre-step positions.
__global__ void integrateBodies(double3* positions, const double3* velocities,
                                int first, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        double3 p = positions[first + i];
        const double3 v = velocities[i];
        p.x += v.x * DT;
        p.y += v.y * DT;
        p.z += v.z * DT;
        positions[first + i] = p;
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
    #pragma omp parallel for reduction(+:energy) schedule(dynamic, 16)
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
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
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
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Node-local ranks select different GPUs; scheduler-masked single GPUs work too.
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

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = (numBodies / ranks) * r + std::min(r, numBodies % ranks);
    }
    const int count = counts[rank], first = offsets[rank];
    // Derived types keep counts in bodies rather than overflowing 3*N or 6*N.
    static_assert(sizeof(Body) == 6 * sizeof(double) && offsetof(Body, vel) == 3 * sizeof(double));
    static_assert(sizeof(double3) == 3 * sizeof(double));
    MPI_Datatype positionType, bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &positionType);
    MPI_Type_commit(&positionType);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);
    const size_t positionBytes = static_cast<size_t>(std::max(numBodies, 1)) * sizeof(double3);
    const size_t velocityBytes = static_cast<size_t>(std::max(count, 1)) * sizeof(double3);
    double3 *positions, *velocities, *devicePositions, *deviceVelocities;
    // Pinned staging supports ordinary MPI implementations without CUDA-aware MPI.
    CUDA_CHECK(cudaMallocHost(&positions, positionBytes));
    CUDA_CHECK(cudaMallocHost(&velocities, velocityBytes));
    CUDA_CHECK(cudaMalloc(&devicePositions, positionBytes));
    CUDA_CHECK(cudaMalloc(&deviceVelocities, velocityBytes));
    std::vector<Body> bodies(rank == 0 ? numBodies : 0), localBodies(count);
    if (rank == 0) {
        randomizeBodies(bodies);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i)
            positions[i] = make_double3(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z);
    }
    MPI_Bcast(positions, numBodies, positionType, 0, MPI_COMM_WORLD);
    MPI_Scatterv(bodies.data(), counts.data(), offsets.data(), bodyType,
                 localBodies.data(), count, bodyType, 0, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i)
        velocities[i] = make_double3(localBodies[i].vel.x, localBodies[i].vel.y, localBodies[i].vel.z);
    CUDA_CHECK(cudaMemcpy(devicePositions, positions, static_cast<size_t>(numBodies) * sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceVelocities, velocities, static_cast<size_t>(count) * sizeof(double3), cudaMemcpyHostToDevice));
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const int blocks = count / TILE + (count % TILE != 0);
    for (int step = 0; step < numSteps; ++step) {
        if (count) {
            computeForces<<<blocks, TILE>>>(devicePositions, deviceVelocities, numBodies, first, count);
            CUDA_CHECK(cudaGetLastError());
            integrateBodies<<<blocks, TILE>>>(devicePositions, deviceVelocities, first, count);
            CUDA_CHECK(cudaGetLastError());
        }
        // No exchange is needed after the last step or for a single rank.
        if (ranks > 1 && step + 1 < numSteps) {
            CUDA_CHECK(cudaMemcpy(positions + first, devicePositions + first,
                                  static_cast<size_t>(count) * sizeof(double3), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions, counts.data(),
                           offsets.data(), positionType, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(devicePositions, positions,
                                  static_cast<size_t>(numBodies) * sizeof(double3), cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(duration * 1000));

    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(positions + first, devicePositions + first,
                              static_cast<size_t>(count) * sizeof(double3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velocities, deviceVelocities,
                              static_cast<size_t>(count) * sizeof(double3), cudaMemcpyDeviceToHost));
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < count; ++i) {
            const double3 p = positions[first + i], v = velocities[i];
            localBodies[i] = {{p.x, p.y, p.z}, {v.x, v.y, v.z}};
        }
        MPI_Gatherv(localBodies.data(), count, bodyType, bodies.data(), counts.data(),
                    offsets.data(), bodyType, 0, MPI_COMM_WORLD);
    }
    CUDA_CHECK(cudaFree(devicePositions));
    CUDA_CHECK(cudaFree(deviceVelocities));
    CUDA_CHECK(cudaFreeHost(positions));
    CUDA_CHECK(cudaFreeHost(velocities));
    MPI_Type_free(&positionType);
    MPI_Type_free(&bodyType);
    int status = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.resize(static_cast<size_t>(numBodies) * 6);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                const auto& body = bodies[i];
                const size_t k = static_cast<size_t>(i) * 6;
                bodyData[k] = body.pos.x;
                bodyData[k + 1] = body.pos.y;
                bodyData[k + 2] = body.pos.z;
                bodyData[k + 3] = body.vel.x;
                bodyData[k + 4] = body.vel.y;
                bodyData[k + 5] = body.vel.z;
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
                status = 1;
            }
        }
    
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
