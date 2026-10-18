#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>

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

constexpr int TILE = 128;

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

// Every rank owns a contiguous target range. Source positions are read-only
// throughout a step; writing next positions separately preserves Euler semantics.
__global__ void advanceBodies(const double3* __restrict__ positions,
                              double3* __restrict__ velocities,
                              double3* __restrict__ nextPositions,
                              int n, int first, int count) {
    __shared__ double3 tile[TILE];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < count;
    const double3 pos = active ? positions[first + local] : make_double3(0, 0, 0);
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int base = 0; base < n;) {
        const int source = base + threadIdx.x;
        if (source < n) tile[threadIdx.x] = positions[source];
        __syncthreads();
        const int length = min(TILE, n - base);
        if (active) {
            for (int j = 0; j < length; ++j) {
                const double dx = tile[j].x - pos.x;
                const double dy = tile[j].y - pos.y;
                const double dz = tile[j].z - pos.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
        base += length;
    }
    if (active) {
        double3 vel = velocities[local];
        vel.x += DT * Fx;
        vel.y += DT * Fy;
        vel.z += DT * Fz;
        velocities[local] = vel;
        nextPositions[local] = make_double3(pos.x + vel.x * DT,
                                           pos.y + vel.y * DT,
                                           pos.z + vel.z * DT);
    }
}

// Validation uses the same target decomposition, counting each pair once.
__global__ void bodyEnergies(const double3* positions, const double3* velocities,
                             double* energies, int n, int first, int count) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local < count) {
        const int i = first + local;
        const double3 pos = positions[i];
        const double3 vel = velocities[local];
        double energy = 0.5 * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
        for (int j = i + 1; j < n; ++j) {
            const double dx = positions[j].x - pos.x;
            const double dy = positions[j].y - pos.y;
            const double dz = positions[j].z - pos.z;
            energy -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
        energies[local] = energy;
    }
}

// Portable MPI staging: only positions are communicated per step, while local
// velocities stay on the GPU. Pinned memory avoids pageable CUDA staging copies.
double simulate(std::vector<Body>& bodies, int n, int steps, bool validate,
                bool collect, int rank, int ranks) {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank = 0;
    MPI_Comm_rank(node, &localRank);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Every MPI rank requires a CUDA device.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&node);

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = n / ranks + (r < n % ranks);
        offsets[r] = r * (n / ranks) + std::min(r, n % ranks);
    }
    const int count = counts[rank], first = offsets[rank];
    const size_t globalBytes = size_t(std::max(n, 1)) * sizeof(double3);
    const size_t localBytes = size_t(std::max(count, 1)) * sizeof(double3);
    static_assert(sizeof(double3) == 3 * sizeof(double), "MPI vector layout");
    MPI_Datatype vectorType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vectorType);
    MPI_Type_commit(&vectorType);

    double3 *positions, *velocities, *devicePositions, *deviceVelocities, *nextPositions;
    CUDA_CHECK(cudaMallocHost(&positions, globalBytes));
    CUDA_CHECK(cudaMallocHost(&velocities, rank == 0 ? globalBytes : localBytes));
    CUDA_CHECK(cudaMalloc(&devicePositions, globalBytes));
    CUDA_CHECK(cudaMalloc(&deviceVelocities, localBytes));
    CUDA_CHECK(cudaMalloc(&nextPositions, localBytes));
    if (rank == 0) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            positions[i] = make_double3(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z);
            velocities[i] = make_double3(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z);
        }
    }
    MPI_Bcast(positions, n, vectorType, 0, MPI_COMM_WORLD);
    MPI_Scatterv(velocities, counts.data(), offsets.data(), vectorType,
                 rank == 0 ? MPI_IN_PLACE : velocities, count, vectorType, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(devicePositions, positions, size_t(n) * sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceVelocities, velocities, size_t(count) * sizeof(double3), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < steps; ++step) {
        if (count) {
            advanceBodies<<<(count - 1) / TILE + 1, TILE>>>(
                devicePositions, deviceVelocities, nextPositions, n, first, count);
            CUDA_CHECK(cudaGetLastError());
        }
        if (ranks == 1) {
            // With a single rank there is no exchange: keep the whole state on
            // the accelerator and ping-pong positions without PCIe round trips.
            std::swap(devicePositions, nextPositions);
            continue;
        }
        if (count) {
            CUDA_CHECK(cudaMemcpy(positions + first, nextPositions,
                                  size_t(count) * sizeof(double3), cudaMemcpyDeviceToHost));
        }
        // The final state only needs a global exchange for output/validation.
        if (step + 1 < steps || validate || collect) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, vectorType, positions,
                           counts.data(), offsets.data(), vectorType, MPI_COMM_WORLD);
        }
        if (step + 1 < steps || validate) {
            CUDA_CHECK(cudaMemcpy(devicePositions, positions, size_t(n) * sizeof(double3), cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %lld ms\n", static_cast<long long>(maxElapsed * 1000));

    double finalEnergy = 0.0;
    if (validate) {
        double* deviceEnergies;
        CUDA_CHECK(cudaMalloc(&deviceEnergies, size_t(std::max(count, 1)) * sizeof(double)));
        std::vector<double> energies(count);
        if (count) {
            bodyEnergies<<<(count - 1) / TILE + 1, TILE>>>(
                devicePositions, deviceVelocities, deviceEnergies, n, first, count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(energies.data(), deviceEnergies, size_t(count) * sizeof(double), cudaMemcpyDeviceToHost));
        }
        double localEnergy = 0.0;
        #pragma omp parallel for reduction(+:localEnergy) schedule(static)
        for (int i = 0; i < count; ++i) localEnergy += energies[i];
        MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        CUDA_CHECK(cudaFree(deviceEnergies));
    }
    if (collect) {
        if (ranks == 1 && steps > 0) {
            CUDA_CHECK(cudaMemcpy(positions, devicePositions, size_t(n) * sizeof(double3), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaMemcpy(velocities + (rank == 0 ? first : 0), deviceVelocities,
                              size_t(count) * sizeof(double3), cudaMemcpyDeviceToHost));
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : velocities, count, vectorType,
                    velocities, counts.data(), offsets.data(), vectorType, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                bodies[i].pos = Vec3(positions[i].x, positions[i].y, positions[i].z);
                bodies[i].vel = Vec3(velocities[i].x, velocities[i].y, velocities[i].z);
            }
        }
    }
    CUDA_CHECK(cudaFree(nextPositions));
    CUDA_CHECK(cudaFree(deviceVelocities));
    CUDA_CHECK(cudaFree(devicePositions));
    CUDA_CHECK(cudaFreeHost(velocities));
    CUDA_CHECK(cudaFreeHost(positions));
    MPI_Type_free(&vectorType);
    return finalEnergy;
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

int parseCount(const char* text) {
    char* end = nullptr;
    errno = 0;
    const long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < 0 || value > std::numeric_limits<int>::max()) return -1;
    return static_cast<int>(value);
}

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = parseCount(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = parseCount(argv[++i]);
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
        if (rank == 0) fprintf(stderr, "Body and step counts must be nonnegative integers within the int range.\n");
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Initialize bodies
    std::vector<Body> bodies(rank == 0 ? numBodies : 0);
    if (rank == 0) randomizeBodies(bodies);
    
    // Run simulation
    const double finalEnergy = simulate(bodies, numBodies, numSteps, validate,
                                        validate || printResults, rank, ranks);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData(size_t(numBodies) * 6);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const auto& body = bodies[i];
            const size_t offset = size_t(i) * 6;
            bodyData[offset] = body.pos.x;
            bodyData[offset + 1] = body.pos.y;
            bodyData[offset + 2] = body.pos.z;
            bodyData[offset + 3] = body.vel.x;
            bodyData[offset + 4] = body.vel.y;
            bodyData[offset + 5] = body.vel.z;
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
    MPI_Finalize();
    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        return runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
