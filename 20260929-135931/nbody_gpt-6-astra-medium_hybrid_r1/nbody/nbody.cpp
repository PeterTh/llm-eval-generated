#include <climits>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
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

// All MPI calls stay on the main thread (MPI_THREAD_FUNNELED). Every rank
// owns a contiguous range of targets and retains its velocities on its GPU.
void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

constexpr int TILE = 128;

// Each thread accumulates one target in the original j order. Separate output
// positions ensure every force reads the same, pre-integration time slice.
__global__ void advance(const double3* __restrict__ positions,
                        double3* __restrict__ velocities,
                        double3* __restrict__ next, int n, int first, int count) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];
    const int t = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + t;
    const bool active = i < count;
    const double3 p = active ? positions[first + i] : make_double3(0, 0, 0);
    double fx = 0, fy = 0, fz = 0;
    for (int base = 0; base < n; base += TILE) {
        if (base + t < n) {
            const double3 q = positions[base + t];
            sx[t] = q.x; sy[t] = q.y; sz[t] = q.z;
        }
        __syncthreads();
        const int length = min(TILE, n - base);
        if (active) {
            for (int j = 0; j < length; ++j) {
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
        double3 v = velocities[i];
        v.x += DT * fx; v.y += DT * fy; v.z += DT * fz;
        velocities[i] = v;
        next[i] = make_double3(p.x + DT * v.x, p.y + DT * v.y, p.z + DT * v.z);
    }
}

// Count each potential pair twice with half weight. Full rows give balanced
// work across ranks, unlike assigning triangular rows to contiguous owners.
__global__ void energyRows(const double3* positions, const double3* velocities,
                           double* energies, int n, int first, int count) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];
    const int t = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + t;
    const bool active = i < count;
    const double3 p = active ? positions[first + i] : make_double3(0, 0, 0);
    double potential = 0;
    for (int base = 0; base < n; base += TILE) {
        if (base + t < n) {
            const double3 q = positions[base + t];
            sx[t] = q.x; sy[t] = q.y; sz[t] = q.z;
        }
        __syncthreads();
        if (active) {
            for (int j = 0; j < min(TILE, n - base); ++j) {
                if (base + j == first + i) continue;
                const double dx = sx[j] - p.x, dy = sy[j] - p.y, dz = sz[j] - p.z;
                potential -= 0.5 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            }
        }
        __syncthreads();
    }
    if (active) {
        const double3 v = velocities[i];
        energies[i] = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z) + potential;
    }
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
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank, ranks;
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
    
    // Bound index arithmetic (including tile padding) before allocating.
    if (numBodies < 0 || numBodies > INT_MAX / 3 || numSteps < 0) {
        if (rank == 0) fprintf(stderr, "Invalid body or step count\n");
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
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Also supports launchers that expose just one GPU to each process.
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
    }
    const int count = counts[rank], first = offsets[rank];
    const size_t allBytes = size_t(std::max(numBodies, 1)) * sizeof(double3);
    const size_t localBytes = size_t(std::max(count, 1)) * sizeof(double3);
    double3 *hostPositions, *hostVelocities;
    double3 *positions, *velocities, *next;
    CUDA_CHECK(cudaMallocHost(&hostPositions, allBytes));
    CUDA_CHECK(cudaMallocHost(&hostVelocities, localBytes));
    CUDA_CHECK(cudaMalloc(&positions, allBytes));
    CUDA_CHECK(cudaMalloc(&velocities, localBytes));
    // One rank can swap entire device position buffers without any host traffic.
    CUDA_CHECK(cudaMalloc(&next, ranks == 1 ? allBytes : localBytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    MPI_Datatype triple, bodyType;
    static_assert(sizeof(double3) == 3 * sizeof(double));
    static_assert(sizeof(Body) == 6 * sizeof(double));
    MPI_Type_contiguous(3, MPI_DOUBLE, &triple);
    MPI_Type_commit(&triple);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);
    std::vector<Body> bodies(rank == 0 ? numBodies : 0);
    if (rank == 0) randomizeBodies(bodies);
    std::vector<Body> localBodies(count);
    MPI_Scatterv(bodies.data(), counts.data(), offsets.data(), bodyType,
                 localBodies.data(), count, bodyType, 0, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) {
        const auto& b = localBodies[i];
        hostPositions[first + i] = make_double3(b.pos.x, b.pos.y, b.pos.z);
        hostVelocities[i] = make_double3(b.vel.x, b.vel.y, b.vel.z);
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, triple, hostPositions, counts.data(), offsets.data(), triple, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpyAsync(positions, hostPositions, size_t(numBodies) * sizeof(double3), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(velocities, hostVelocities, size_t(count) * sizeof(double3), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const int blocks = (count + TILE - 1) / TILE;
    for (int step = 0; step < numSteps; ++step) {
        if (count) {
            advance<<<blocks, TILE, 0, stream>>>(positions, velocities, next, numBodies, first, count);
            CUDA_CHECK(cudaGetLastError());
        }
        if (ranks == 1) {
            std::swap(positions, next);
        } else if (step + 1 < numSteps || validate) {
            // Pinned staging works with any MPI implementation, including those
            // without CUDA-aware MPI. Only positions are communicated each step.
            CUDA_CHECK(cudaMemcpyAsync(hostPositions + first, next, size_t(count) * sizeof(double3), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Allgatherv(MPI_IN_PLACE, 0, triple, hostPositions, counts.data(), offsets.data(), triple, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpyAsync(positions, hostPositions, size_t(numBodies) * sizeof(double3), cudaMemcpyHostToDevice, stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double elapsed = MPI_Wtime() - start, maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));

    double finalEnergy = 0;
    if (validate) {
        double* deviceEnergy;
        CUDA_CHECK(cudaMalloc(&deviceEnergy, size_t(std::max(count, 1)) * sizeof(double)));
        if (count) {
            energyRows<<<blocks, TILE, 0, stream>>>(positions, velocities, deviceEnergy, numBodies, first, count);
            CUDA_CHECK(cudaGetLastError());
        }
        std::vector<double> energies(count);
        CUDA_CHECK(cudaMemcpyAsync(energies.data(), deviceEnergy, size_t(count) * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double localEnergy = 0;
        #pragma omp parallel for reduction(+:localEnergy) schedule(static)
        for (int i = 0; i < count; ++i) localEnergy += energies[i];
        MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        CUDA_CHECK(cudaFree(deviceEnergy));
    }
    if (printResults || validate) {
        const double3* finalPositions = (ranks == 1 || numSteps == 0) ? positions + first : next;
        CUDA_CHECK(cudaMemcpyAsync(hostPositions + first, finalPositions, size_t(count) * sizeof(double3), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hostVelocities, velocities, size_t(count) * sizeof(double3), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < count; ++i) {
            const double3 p = hostPositions[first + i], v = hostVelocities[i];
            localBodies[i] = {Vec3(p.x, p.y, p.z), Vec3(v.x, v.y, v.z)};
        }
        MPI_Gatherv(localBodies.data(), count, bodyType, bodies.data(), counts.data(), offsets.data(), bodyType, 0, MPI_COMM_WORLD);
    }
    CUDA_CHECK(cudaFree(positions));
    CUDA_CHECK(cudaFree(velocities));
    CUDA_CHECK(cudaFree(next));
    CUDA_CHECK(cudaFreeHost(hostPositions));
    CUDA_CHECK(cudaFreeHost(hostVelocities));
    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Type_free(&triple);
    MPI_Type_free(&bodyType);

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData(size_t(numBodies) * 6);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const auto& body = bodies[i];
            const size_t k = size_t(i) * 6;
            bodyData[k] = body.pos.x;
            bodyData[k + 1] = body.pos.y;
            bodyData[k + 2] = body.pos.z;
            bodyData[k + 3] = body.vel.x;
            bodyData[k + 4] = body.vel.y;
            bodyData[k + 5] = body.vel.z;
        }
        print_results(bodyData, "Bodies");
    }
    
    int result = 0;
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) {
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
