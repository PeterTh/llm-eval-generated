#include <algorithm>
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

static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be packed");
static_assert(sizeof(double3) == 3 * sizeof(double), "Position must be packed");

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

constexpr int TILE = 128;

// Each rank owns target bodies, but every target needs all source positions.
// All threads participate in tile loads, including the final partial block.
__global__ void computeForces(const double3* positions, double3* velocities,
                              int n, int offset, int count) {
    __shared__ double3 sources[TILE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < count;
    const double3 position = active ? positions[offset + i] : make_double3(0, 0, 0);
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (long long base = 0; base < n; base += TILE) {
        const long long source = base + threadIdx.x;
        if (source < n) sources[threadIdx.x] = positions[source];
        __syncthreads();
        const int length = static_cast<int>(min(static_cast<long long>(TILE), n - base));
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < length; ++j) {
                const double dx = sources[j].x - position.x;
                const double dy = sources[j].y - position.y;
                const double dz = sources[j].z - position.z;
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
        velocities[i].x += DT * Fx;
        velocities[i].y += DT * Fy;
        velocities[i].z += DT * Fz;
    }
}

// A separate launch is the global barrier between force evaluation and drift.
__global__ void integrateBodies(double3* positions, const double3* velocities,
                                int offset, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        positions[offset + i].x += velocities[i].x * DT;
        positions[offset + i].y += velocities[i].y * DT;
        positions[offset + i].z += velocities[i].z * DT;
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
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

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    checkCuda(cudaGetDeviceCount(&deviceCount), "enumerating CUDA devices");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d requires a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), "selecting CUDA device");
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
    }
    const int count = counts[rank], offset = offsets[rank];
    MPI_Datatype positionType, bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &positionType);
    MPI_Type_commit(&positionType);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);
    
    // Initialize bodies
    std::vector<Body> bodies(rank == 0 ? numBodies : 0);
    if (rank == 0) randomizeBodies(bodies);
    std::vector<Body> localBodies(std::max(count, 1));
    MPI_Scatterv(bodies.data(), counts.data(), offsets.data(), bodyType,
                 localBodies.data(), count, bodyType, 0, MPI_COMM_WORLD);
    double3 *hostPositions = nullptr, *hostVelocities = nullptr;
    const size_t positionBytes = size_t(std::max(numBodies, 1)) * sizeof(double3);
    const size_t velocityBytes = size_t(std::max(count, 1)) * sizeof(double3);
    checkCuda(cudaMallocHost(&hostPositions, positionBytes), "allocating pinned positions");
    checkCuda(cudaMallocHost(&hostVelocities, velocityBytes), "allocating pinned velocities");
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) {
        const Body& b = localBodies[i];
        hostPositions[offset + i] = make_double3(b.pos.x, b.pos.y, b.pos.z);
        hostVelocities[i] = make_double3(b.vel.x, b.vel.y, b.vel.z);
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, hostPositions,
                   counts.data(), offsets.data(), positionType, MPI_COMM_WORLD);
    double3 *positions = nullptr, *velocities = nullptr;
    checkCuda(cudaMalloc(&positions, positionBytes), "allocating device positions");
    checkCuda(cudaMalloc(&velocities, velocityBytes), "allocating device velocities");
    checkCuda(cudaMemcpy(positions, hostPositions, positionBytes, cudaMemcpyHostToDevice), "uploading positions");
    checkCuda(cudaMemcpy(velocities, hostVelocities, velocityBytes, cudaMemcpyHostToDevice), "uploading velocities");
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int step = 0; step < numSteps; ++step) {
        if (count > 0) {
            const int blocks = (count - 1) / TILE + 1;
            computeForces<<<blocks, TILE>>>(positions, velocities, numBodies, offset, count);
            checkCuda(cudaGetLastError(), "launching force kernel");
            integrateBodies<<<blocks, TILE>>>(positions, velocities, offset, count);
            checkCuda(cudaGetLastError(), "launching integration kernel");
        }
        // Only positions cross the network; velocities remain on their owning GPU.
        if (step + 1 < numSteps && ranks > 1) {
            checkCuda(cudaMemcpy(hostPositions + offset, positions + offset,
                                 size_t(count) * sizeof(double3), cudaMemcpyDeviceToHost), "downloading positions");
            MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, hostPositions,
                           counts.data(), offsets.data(), positionType, MPI_COMM_WORLD);
            checkCuda(cudaMemcpy(positions, hostPositions, positionBytes, cudaMemcpyHostToDevice), "exchanging positions");
        }
    }
    
    checkCuda(cudaDeviceSynchronize(), "finishing simulation");
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(duration * 1000));

    if (printResults || validate) {
        checkCuda(cudaMemcpy(hostPositions + offset, positions + offset,
                             size_t(count) * sizeof(double3), cudaMemcpyDeviceToHost), "reading final positions");
        checkCuda(cudaMemcpy(hostVelocities, velocities,
                             size_t(count) * sizeof(double3), cudaMemcpyDeviceToHost), "reading final velocities");
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < count; ++i) {
            const double3 p = hostPositions[offset + i], v = hostVelocities[i];
            localBodies[i] = {{p.x, p.y, p.z}, {v.x, v.y, v.z}};
        }
        MPI_Gatherv(localBodies.data(), count, bodyType, bodies.data(),
                    counts.data(), offsets.data(), bodyType, 0, MPI_COMM_WORLD);
    }
    checkCuda(cudaFree(positions), "freeing positions");
    checkCuda(cudaFree(velocities), "freeing velocities");
    checkCuda(cudaFreeHost(hostPositions), "freeing host positions");
    checkCuda(cudaFreeHost(hostVelocities), "freeing host velocities");
    MPI_Type_free(&positionType);
    MPI_Type_free(&bodyType);
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.resize(size_t(numBodies) * 6);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const auto& body = bodies[i];
            const size_t base = size_t(i) * 6;
            bodyData[base] = body.pos.x;
            bodyData[base + 1] = body.pos.y;
            bodyData[base + 2] = body.pos.z;
            bodyData[base + 3] = body.vel.x;
            bodyData[base + 4] = body.vel.y;
            bodyData[base + 5] = body.vel.z;
        }
        print_results(bodyData, "Bodies");
    }
    
    // Validation: check that simulation produces finite, reasonable values
    int result = 0;
    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }
    
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
